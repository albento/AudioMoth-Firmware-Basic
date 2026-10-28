/****************************************************************************
 * vit_inference.c
 * openacousticdevices.info
 *****************************************************************************/

/* Streaming INT8 inference of test-audiomoth-cnn's TinyViT (student_model.py), one token per
   PCEN frame, matching its eval-mode forward: patch_embed -> VIT_DEPTH post-LN encoder layers
   (causal ALiBi attention over the last VIT_CONTEXT_SIZE tokens, ReLU FFN) -> classifier,
   token logits averaged per buffer. Weights/scales come from vit_model.c (export_vit.py).

   Quantization: symmetric int8 everywhere (zero-point 0), except softmax's output, which uses
   CMSIS-NN/TFLite's zero-point -128 (prob = (q + 128) / 256). Scores are requantized as their
   gap to the row max on a fixed grid (see export_vit.py), not on an absolute scale. LayerNorm
   (with its residual add) runs in float. Builds unchanged on a host for checking against the
   PyTorch model - CMSIS-NN's pure-C kernels are bit-exact with its Cortex-M4 DSP ones. */

#include <stdint.h>
#include <string.h>
#include <math.h>

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

#include "vit_inference.h"

#if defined(__arm__)
#include "em_device.h"
#define VIT_GET_CYCLES()                         (DWT->CYCCNT)
#else
#define VIT_GET_CYCLES()                         0u
#endif

#define VIT_TIME(field, code) { \
    uint32_t vitStartCycles = VIT_GET_CYCLES(); \
    code; \
    timing->field += VIT_GET_CYCLES() - vitStartCycles; \
}

#define VIT_LAYER_NORM_EPS                       1.0e-5f

/* 1 = table softmax (runSoftmaxTable below), 0 = CMSIS-NN's arm_softmax_s8 - measured on the
   device at ~1100 cycles per element, over half of the whole ISR. Both take the same int8
   input and produce the same output convention. */

#ifndef VIT_SOFTMAX_LUT
#define VIT_SOFTMAX_LUT                          1
#endif

/* Per-layer, per-head history of the last VIT_CONTEXT_SIZE tokens' K and V, indexed by ring
   position. V is stored transposed ([head_dim][context]) so attn.V is a plain vec x mat^T.
   Both layers advance together, so they share one ring position. Internal RAM. */

static int8_t kWindow[VIT_DEPTH][VIT_N_HEADS][VIT_CONTEXT_SIZE][VIT_HEAD_DIM];

static int8_t vWindowT[VIT_DEPTH][VIT_N_HEADS][VIT_HEAD_DIM][VIT_CONTEXT_SIZE];

static uint32_t currentPosition;

static uint32_t validCount;

void ViTInference_reset(void) {

    memset(kWindow, 0, sizeof(kWindow));

    memset(vWindowT, 0, sizeof(vWindowT));

    currentPosition = VIT_CONTEXT_SIZE - 1;

    validCount = 0;

}

static inline int8_t clampToInt8(int32_t value) {

    return (int8_t)(value < -128 ? -128 : (value > 127 ? 127 : value));

}

static inline int32_t roundToInt(float value) {

    return (int32_t)(value >= 0.0f ? value + 0.5f : value - 0.5f);

}

static void runFullyConnected(const vit_fc_t *fc, const int8_t *input, int32_t inputSize, int8_t *output, int32_t outputSize, int32_t activationMin) {

    cmsis_nn_context ctx = { NULL, 0 };

    cmsis_nn_fc_params fcParams = { 0, 0, 0, { activationMin, 127 } };

    cmsis_nn_per_tensor_quant_params quantParams = { fc->multiplier, fc->shift };

    /* arm_fully_connected_s8 reads filter_dims->n as the accumulation depth (input size) and
       output_dims->c as the number of outputs. */

    cmsis_nn_dims inputDims = { 1, 1, 1, inputSize };

    cmsis_nn_dims filterDims = { inputSize, 1, 1, outputSize };

    cmsis_nn_dims biasDims = { 1, 1, 1, outputSize };

    cmsis_nn_dims outputDims = { 1, 1, 1, outputSize };

    arm_fully_connected_s8(&ctx, &fcParams, &quantParams, &inputDims, input, &filterDims, fc->weight, &biasDims, fc->bias, &outputDims, output);

}

/* output = LayerNorm(scaleA * a + scaleB * b), requantized to int8 */

static void runLayerNorm(const vit_ln_t *ln, const int8_t *a, const int8_t *b, int8_t *output) {

    float x[VIT_EMBED_DIM];

    float sum = 0.0f;

    for (uint32_t i = 0; i < VIT_EMBED_DIM; i += 1) {

        x[i] = ln->inputScaleA * (float)a[i] + ln->inputScaleB * (float)b[i];

        sum += x[i];

    }

    float mean = sum / (float)VIT_EMBED_DIM;

    float sumSquaredDiff = 0.0f;

    for (uint32_t i = 0; i < VIT_EMBED_DIM; i += 1) {

        float diff = x[i] - mean;

        sumSquaredDiff += diff * diff;

    }

    float invStd = 1.0f / sqrtf(sumSquaredDiff / (float)VIT_EMBED_DIM + VIT_LAYER_NORM_EPS);

    for (uint32_t i = 0; i < VIT_EMBED_DIM; i += 1) {

        float y = (x[i] - mean) * invStd * ln->gamma[i] + ln->beta[i];

        output[i] = clampToInt8(roundToInt(y * ln->outputInvScale));

    }

}

#if VIT_SOFTMAX_LUT

/* Scores arrive as (score - row max) on a fixed grid, so every value is in [-128, 0] and exp()
   is a lookup in vit_softmax_exp_q15 (Q15, exp(0) = 32768). One integer reciprocal per row
   replaces arm_softmax_s8's two fixed-point exp() per element. Output convention unchanged:
   prob = (q + 128) / 256, i.e. q = round(256 * e / sum) - 128. The row max contributes
   exp(0), so sum >= 2^15, reciprocal <= 2^16 and e * reciprocal <= 2^31 - no overflow. */

static void runSoftmaxTable(const int8_t *scores, uint32_t count, int8_t *probs) {

    uint32_t sum = 0;

    for (uint32_t j = 0; j < count; j += 1) sum += vit_softmax_exp_q15[-scores[j]];

    uint32_t reciprocal = 0x80000000u / sum;

    for (uint32_t j = 0; j < count; j += 1) {

        uint32_t prob = ((uint32_t)vit_softmax_exp_q15[-scores[j]] * reciprocal + (1u << 22)) >> 23;

        probs[j] = (int8_t)((int32_t)(prob > 255 ? 255 : prob) - 128);

    }

}

#endif

static void runAttention(uint32_t layer, const int8_t *q, int8_t *attnOut, ViTInferenceTiming_t *timing) {

    const vit_layer_t *params = &vit_layers[layer];

    for (uint32_t h = 0; h < VIT_N_HEADS; h += 1) {

        int8_t scores[VIT_CONTEXT_SIZE];

        int8_t probs[VIT_CONTEXT_SIZE];

        VIT_TIME(qkCycles[layer], {

            /* alibiAcc holds each head's ALiBi bias (already in QK accumulator units) by age,
               reversed and doubled, so that alibi[j] is ring entry j's bias for the current
               ring position without any per-token rotation. Before the ring has wrapped, the
               valid entries are exactly 0..validCount-1. */

            const int32_t *alibi = params->alibiAcc + h * 2 * VIT_CONTEXT_SIZE + (VIT_CONTEXT_SIZE - 1 - currentPosition);

            const int8_t *qHead = q + h * VIT_HEAD_DIM;

            int32_t acc[VIT_CONTEXT_SIZE];

            int32_t accMax = INT32_MIN;

            for (uint32_t j = 0; j < validCount; j += 1) {

                const int8_t *kRow = kWindow[layer][h][j];

                int32_t sum = alibi[j];

                for (uint32_t i = 0; i < VIT_HEAD_DIM; i += 1) sum += (int32_t)qHead[i] * (int32_t)kRow[i];

                acc[j] = sum;

                if (sum > accMax) accMax = sum;

            }

            for (uint32_t j = 0; j < validCount; j += 1) {

                int32_t diff = arm_nn_requantize(acc[j] - accMax, params->qkMultiplier, params->qkShift);

                scores[j] = (int8_t)(diff < -128 ? -128 : diff);

            }

        })

        VIT_TIME(softmaxCycles[layer], {

#if VIT_SOFTMAX_LUT

            runSoftmaxTable(scores, validCount, probs);

#else

            arm_softmax_s8(scores, 1, (int32_t)validCount, params->softmaxMultiplier, params->softmaxShift, params->softmaxDiffMin, probs);

#endif

            for (uint32_t j = validCount; j < VIT_CONTEXT_SIZE; j += 1) probs[j] = -128;

        })

        VIT_TIME(avCycles[layer], {

            arm_nn_vec_mat_mult_t_s8(probs, (const int8_t*)vWindowT[layer][h], NULL, NULL, attnOut + h * VIT_HEAD_DIM,
                                     128, 0, params->avMultiplier, params->avShift,
                                     VIT_CONTEXT_SIZE, VIT_HEAD_DIM, -128, 127, 1, 0);

        })

    }

}

static void runLayer(uint32_t layer, int8_t *embedding, ViTInferenceTiming_t *timing) {

    const vit_layer_t *params = &vit_layers[layer];

    int8_t q[VIT_EMBED_DIM], k[VIT_EMBED_DIM], v[VIT_EMBED_DIM];

    VIT_TIME(qkvCycles[layer], {

        runFullyConnected(&params->q, embedding, VIT_EMBED_DIM, q, VIT_EMBED_DIM, -128);

        runFullyConnected(&params->k, embedding, VIT_EMBED_DIM, k, VIT_EMBED_DIM, -128);

        runFullyConnected(&params->v, embedding, VIT_EMBED_DIM, v, VIT_EMBED_DIM, -128);

        for (uint32_t h = 0; h < VIT_N_HEADS; h += 1) {

            for (uint32_t i = 0; i < VIT_HEAD_DIM; i += 1) {

                kWindow[layer][h][currentPosition][i] = k[h * VIT_HEAD_DIM + i];

                vWindowT[layer][h][i][currentPosition] = v[h * VIT_HEAD_DIM + i];

            }

        }

    })

    int8_t attnOut[VIT_EMBED_DIM];

    runAttention(layer, q, attnOut, timing);

    int8_t projected[VIT_EMBED_DIM];

    VIT_TIME(outProjCycles[layer], runFullyConnected(&params->outProj, attnOut, VIT_EMBED_DIM, projected, VIT_EMBED_DIM, -128))

    int8_t normalised[VIT_EMBED_DIM];

    VIT_TIME(layerNorm1Cycles[layer], runLayerNorm(&params->ln1, embedding, projected, normalised))

    int8_t hidden[VIT_FFN_DIM];

    VIT_TIME(ffn1Cycles[layer], runFullyConnected(&params->ffn1, normalised, VIT_EMBED_DIM, hidden, VIT_FFN_DIM, 0))

    int8_t feedForward[VIT_EMBED_DIM];

    VIT_TIME(ffn2Cycles[layer], runFullyConnected(&params->ffn2, hidden, VIT_FFN_DIM, feedForward, VIT_EMBED_DIM, -128))

    VIT_TIME(layerNorm2Cycles[layer], runLayerNorm(&params->ln2, normalised, feedForward, embedding))

}

float ViTInference_run(const volatile float spectrogram[VIT_INFERENCE_INPUT_FRAMES][VIT_N_BINS], ViTInferenceTiming_t *timing, uint32_t deadlineCycles) {

    memset(timing, 0, sizeof(ViTInferenceTiming_t));

    float logitSum = 0.0f;

    uint32_t lastTokenCycles = 0;

    for (uint32_t frame = 0; frame < VIT_INFERENCE_INPUT_FRAMES; frame += 1) {

        uint32_t tokenStartCycles = VIT_GET_CYCLES();

        if (deadlineCycles != 0 && (int32_t)(deadlineCycles - (tokenStartCycles + lastTokenCycles)) < 0) break;

        currentPosition = (currentPosition + 1) % VIT_CONTEXT_SIZE;

        if (validCount < VIT_CONTEXT_SIZE) validCount += 1;

        int8_t embedding[VIT_EMBED_DIM];

        VIT_TIME(patchEmbedCycles, {

            int8_t input[VIT_N_BINS];

            for (uint32_t bin = 0; bin < VIT_N_BINS; bin += 1) input[bin] = clampToInt8(roundToInt(spectrogram[frame][bin] * VIT_INPUT_INV_SCALE));

            runFullyConnected(&vit_patch_embed, input, VIT_N_BINS, embedding, VIT_EMBED_DIM, -128);

        })

        for (uint32_t layer = 0; layer < VIT_DEPTH; layer += 1) runLayer(layer, embedding, timing);

        VIT_TIME(classifierCycles, {

            int32_t acc = 0;

            for (uint32_t i = 0; i < VIT_EMBED_DIM; i += 1) acc += (int32_t)vit_classifier_weight[i] * (int32_t)embedding[i];

            logitSum += (float)acc * VIT_CLASSIFIER_LOGIT_SCALE + VIT_CLASSIFIER_BIAS;

        })

        timing->tokensDone += 1;

        lastTokenCycles = VIT_GET_CYCLES() - tokenStartCycles;

    }

    return timing->tokensDone == 0 ? 0.0f : logitSum / (float)timing->tokensDone;

}
