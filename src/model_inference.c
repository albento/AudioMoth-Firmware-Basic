/****************************************************************************
 * model_inference.c
 * openacousticdevices.info
 *
 * Runs the TinyLSTMCNN classifier exported by test-audiomoth-cnn/export_int8.py
 * (model/model_int8_lstm_small_16_focal.{h,c}) on one buffer's spectrogram at a time,
 * via real CMSIS-NN int8 kernels - the on-device counterpart of int8_sim.py, which this
 * mirrors stage for stage. Tied to this specific export (its exact layer shapes/symbol
 * names are hard-coded below), not a generic model loader.
 *****************************************************************************/

#include <string.h>

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

#include "audiomoth.h"
#include "model_inference.h"
#include "model_int8_lstm_small_16_focal.h"

/* Spatial (frequency x time) dimensions after each stage, hand-computed from the fixed
   input size (129 x 7) and each stage's own kernel/stride/padding (matches
   export_int8.py's FEATURE_LAYERS exactly - see EXPERIMENTS.md/the model export for the
   derivation). Channel counts come directly from the generated header. */

#define STAGE0_OUT_H     65
#define STAGE0_OUT_W     4
#define STAGE1_DW_OUT_H  33
#define STAGE1_DW_OUT_W  2
#define STAGE1_PW_OUT_H  33
#define STAGE1_PW_OUT_W  2
#define STAGE2_DW_OUT_H  17
#define STAGE2_DW_OUT_W  1
#define STAGE2_PW_OUT_H  17
#define STAGE2_PW_OUT_W  1

/* Largest single-stage activation tensor is stage0's output: 16 * 65 * 4 = 4160 bytes -
   too big for the internal RAM already committed elsewhere (see EXPERIMENTS.md), so both
   ping-pong activation buffers live in external SRAM, which is otherwise idle in
   STFT_OUTPUT_MODE_CSV (the same reasoning as stftRecordQueue in main.c).

   In STFT_OUTPUT_MODE_BINARY, main.c's stftRecordQueue ALSO lives in external SRAM,
   starting at the same AM_EXTERNAL_SRAM_START_ADDRESS - 8 records of (4+4+1024*2+7*129*4+4)
   = 5672 bytes each = 45376 bytes. These buffers are placed at a 64KB offset, safely past
   that, so the two never overlap regardless of which STFT_OUTPUT_MODE is active. Not
   computed from main.c's own definitions (STFT_RECORD_QUEUE_LENGTH/stftRecord_t aren't
   visible here) - if that queue's size ever grows past 64KB this offset must move too. */

#define ACTIVATION_BUFFER_SIZE   4160
#define ACTIVATION_BUFFER_BASE_OFFSET   65536

static int8_t *activationBufferA = (int8_t*)(AM_EXTERNAL_SRAM_START_ADDRESS + ACTIVATION_BUFFER_BASE_OFFSET);
static int8_t *activationBufferB = (int8_t*)(AM_EXTERNAL_SRAM_START_ADDRESS + ACTIVATION_BUFFER_BASE_OFFSET + ACTIVATION_BUFFER_SIZE);

/* Input tensor (quantized, transposed spectrogram) and the LSTM's tiny input vector both
   fit easily in internal RAM. */

static int8_t inputBuffer[MODEL_INFERENCE_INPUT_BINS * MODEL_INFERENCE_INPUT_FRAMES];

static int8_t pooledBuffer[MODEL_INT8_LSTM_SMALL_16_FOCAL_INPUT_SIZE];

/* Scratch buffer shared by runConv()'s arm_convolve_s8 calls (stage1_pw/stage2_pw) and
   runDepthwiseConv()'s arm_depthwise_conv_s8_opt call (stage1_dw/stage2_dw) - stage0 is
   hand-written (runStage0Conv()) and needs no scratch buffer at all. Sized for the larger
   requirement: Cin * kH * kW * sizeof(int16_t) (confirmed from
   arm_depthwise_conv_s8_opt_get_buffer_size_dsp()), 32*3*3*2 = 576 bytes for stage2_dw. */

static int16_t convScratchBuffer[288];

/* arm_nn_lstm_step_s8's temp1/temp2/cell_state are each only hidden_size * sizeof(int16_t)
   (confirmed from Source/NNSupportFunctions/arm_nn_lstm_step_s8.c) - negligible, and
   cell_state/hidden_state persist across calls for real streaming (state carried from one
   64ms buffer to the next, never reset mid-recording). */

static int16_t lstmTemp1[MODEL_INT8_LSTM_SMALL_16_FOCAL_HIDDEN_SIZE];
static int16_t lstmTemp2[MODEL_INT8_LSTM_SMALL_16_FOCAL_HIDDEN_SIZE];
static int16_t lstmCellState[MODEL_INT8_LSTM_SMALL_16_FOCAL_HIDDEN_SIZE];

/* arm_nn_lstm_calculate_gate_s8_s16() (the function arm_nn_lstm_step_s8() actually calls
   per gate) never reads cmsis_nn_lstm_gate's own `bias` field at all - only
   input_effective_bias/hidden_effective_bias, added to the raw accumulator BEFORE each
   requantization step (confirmed directly from its source, and from CMSIS-NN's own
   lstm_settings.py test generator's calc_effective_bias()). Leaving these NULL (as this
   file did until now) makes the kernel dereference a null pointer - which doesn't fault on
   this core, since address 0 is mapped to the start of flash (the vector table), so it
   silently reads whatever bytes live there as "bias" instead - a real, confirmed-on-hardware
   bug, not a hypothetical one. The real bias (same value as bias_q3_12, just in the raw
   accumulator's pre-multiply-shift domain - see export_int8.py's _quantize_gate) belongs in
   input_effective_bias; hidden_effective_bias is always zero (the bias is added exactly
   once, during the input-weights pass only) - this array is shared by all four gates. */

static const int32_t lstmZeroEffectiveBias[MODEL_INT8_LSTM_SMALL_16_FOCAL_HIDDEN_SIZE] = {0};
static int8_t lstmHiddenState[MODEL_INT8_LSTM_SMALL_16_FOCAL_HIDDEN_SIZE];

void ModelInference_reset(void) {

    memset(lstmCellState, 0, sizeof(lstmCellState));
    memset(lstmHiddenState, 0, sizeof(lstmHiddenState));

}

/* Hand-written replacement for stage0 (in_ch=1, 3x3, stride 2, pad 1), which
   arm_convolve_s8 spent most of its time on not because of the arithmetic (37440 MACs is
   tiny) but because of its im2col gathering step: with kernel_ch=1, every one of the 2340
   (65*4 output positions x 9 kernel taps) im2col writes is a 1-byte arm_memcpy_s8/
   arm_memset_s8 call, and neither is MVE-accelerated on this core (Cortex-M4, no MVE) so
   each one is a genuine call into libc's generic memcpy()/memset() to move a single byte -
   see EXPERIMENTS.md. Since convParams.input_offset is 0 for every stage in this model, a
   padded tap contributes input_offset*weight = 0 to the sum either way, so padding taps can
   just be skipped instead of memset - no im2col buffer is needed at all when in_ch=1, the
   9 taps can be read directly from the input tensor. Requantization goes through the exact
   same arm_nn_requantize() CMSIS-NN uses internally, so this is bit-identical to
   arm_convolve_s8's output, just without the im2col overhead - not an approximation. */

static void runStage0Conv(const int8_t *input, int8_t *output) {

    const int8_t *weight = model_int8_lstm_small_16_focal_stage0_weight;
    const int32_t *bias = model_int8_lstm_small_16_focal_stage0_bias;
    const int32_t *multiplier = model_int8_lstm_small_16_focal_stage0_multiplier;
    const int32_t *shift = model_int8_lstm_small_16_focal_stage0_shift;

    const int32_t kernel = MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE0_KERNEL;
    const int32_t stride = MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE0_STRIDE;
    const int32_t padding = MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE0_PADDING;
    const int32_t outCh = MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE0_OUT_CH;

    int32_t oh, ow, oc, ky, kx;

    for (oh = 0; oh < STAGE0_OUT_H; oh += 1) {

        int32_t iyBase = oh * stride - padding;

        for (ow = 0; ow < STAGE0_OUT_W; ow += 1) {

            int32_t ixBase = ow * stride - padding;

            int8_t *outPixel = output + (oh * STAGE0_OUT_W + ow) * outCh;

            for (oc = 0; oc < outCh; oc += 1) {

                int32_t sum = bias[oc];

                const int8_t *w = weight + oc * kernel * kernel;

                for (ky = 0; ky < kernel; ky += 1) {

                    int32_t iy = iyBase + ky;

                    if (iy < 0 || iy >= MODEL_INFERENCE_INPUT_BINS) continue;

                    for (kx = 0; kx < kernel; kx += 1) {

                        int32_t ix = ixBase + kx;

                        if (ix < 0 || ix >= MODEL_INFERENCE_INPUT_FRAMES) continue;

                        sum += (int32_t)input[iy * MODEL_INFERENCE_INPUT_FRAMES + ix] * (int32_t)w[ky * kernel + kx];

                    }

                }

                sum = arm_nn_requantize(sum, multiplier[oc], shift[oc]);
                sum = MAX(sum, 0);
                sum = MIN(sum, 127);

                outPixel[oc] = (int8_t)sum;

            }

        }

    }

}

/* Pointwise (1x1, stride 1, pad 0) convolution via arm_convolve_s8. Tried a hand-written
   replacement mirroring runStage0Conv() above - skip the im2col memcpy, feed
   arm_s8_to_s16_unordered_with_offset()/arm_nn_mat_mult_kernel_s8_s16() directly (same
   approach, verified bit-exact against arm_convolve_s8 with an on-device check, 0
   mismatches). stage1_pw did get faster in isolation (-7%), but stage2_pw got measurably
   slower (+10%) and, surprisingly, so did stage0 (+14%) despite not touching it - the net
   effect across the whole pipeline was ~1.1ms SLOWER, not faster (32928 -> 34027us
   inference). Best explanation: changing which functions exist/how large they are shifts
   .text layout enough to change flash-prefetch behaviour for unrelated code - a real,
   measured effect on this target, not something visible from source alone. Reverted;
   arm_convolve_s8 is the measured-best option for these two stages, same conclusion as the
   arm_convolve_wrapper_s8 attempt above. */

static void runConv(const int8_t *input, int8_t *output, const int8_t *weight, const int32_t *bias,
                     const int32_t *multiplier, const int32_t *shift, int32_t inCh, int32_t outCh,
                     int32_t numPositions) {

    cmsis_nn_context ctx = { convScratchBuffer, sizeof(convScratchBuffer) };

    cmsis_nn_conv_params convParams;
    convParams.input_offset = 0;
    convParams.output_offset = 0;
    convParams.stride.h = 1;
    convParams.stride.w = 1;
    convParams.padding.h = 0;
    convParams.padding.w = 0;
    convParams.dilation.h = 1;
    convParams.dilation.w = 1;
    convParams.activation.min = 0;
    convParams.activation.max = 127;

    cmsis_nn_per_channel_quant_params quantParams;
    quantParams.multiplier = (int32_t*)multiplier;
    quantParams.shift = (int32_t*)shift;

    cmsis_nn_dims inputDims = { 1, numPositions, 1, inCh };
    cmsis_nn_dims filterDims = { outCh, 1, 1, inCh };
    cmsis_nn_dims biasDims = { 1, 1, 1, outCh };
    cmsis_nn_dims outputDims = { 1, numPositions, 1, outCh };

    arm_convolve_s8(&ctx, &convParams, &quantParams, &inputDims, input, &filterDims, weight,
                     &biasDims, bias, NULL, &outputDims, output);

}

/* Depthwise convolution (groups == channels, channel multiplier 1) via
   arm_depthwise_conv_s8_opt - same ReLU fusion (Conv+BatchNorm+ReLU folded at export time)
   as every other stage here. Deliberately NOT
   arm_depthwise_conv_s8: profiling showed that function falls back to a fully scalar,
   non-SIMD path whenever ch_mult != 4 (confirmed from its source, depthwise_conv_s8_generic
   is used unless ch_mult%4==0, which never holds for this model's plain, channel-preserving
   depthwise stages). arm_depthwise_conv_s8_opt has a dedicated DSP/SMLAD-accelerated path
   for exactly this in_ch==out_ch case (needs output_ch%4==0 to avoid its own scalar
   leftover loop - true here, 16 and 32 both divide evenly) and takes the same weight array
   layout, so no re-export needed. */

static void runDepthwiseConv(const int8_t *input, int8_t *output, const int8_t *weight, const int32_t *bias,
                              const int32_t *multiplier, const int32_t *shift, int32_t channels,
                              int32_t inH, int32_t inW, int32_t outH, int32_t outW, int32_t kernel,
                              int32_t stride, int32_t padding) {

    cmsis_nn_context ctx = { convScratchBuffer, sizeof(convScratchBuffer) };

    cmsis_nn_dw_conv_params dwConvParams;
    dwConvParams.input_offset = 0;
    dwConvParams.output_offset = 0;
    dwConvParams.ch_mult = 1;
    dwConvParams.stride.h = stride;
    dwConvParams.stride.w = stride;
    dwConvParams.padding.h = padding;
    dwConvParams.padding.w = padding;
    dwConvParams.dilation.h = 1;
    dwConvParams.dilation.w = 1;
    dwConvParams.activation.min = 0;
    dwConvParams.activation.max = 127;

    cmsis_nn_per_channel_quant_params quantParams;
    quantParams.multiplier = (int32_t*)multiplier;
    quantParams.shift = (int32_t*)shift;

    cmsis_nn_dims inputDims = { 1, inH, inW, channels };
    cmsis_nn_dims filterDims = { 1, kernel, kernel, channels };
    cmsis_nn_dims biasDims = { 1, 1, 1, channels };
    cmsis_nn_dims outputDims = { 1, outH, outW, channels };

    arm_depthwise_conv_s8_opt(&ctx, &dwConvParams, &quantParams, &inputDims, input, &filterDims, weight,
                              &biasDims, bias, &outputDims, output);

}

/* AdaptiveAvgPool2d((1, None)): averages over frequency (H) only, keeping time (W) - no
   CMSIS-NN kernel does this exact shape (arm_avgpool_s8 pools both axes), so this is a
   plain int32 accumulate-and-round-divide, matching int8_sim.py's avgpool_freq_s8()
   exactly. With this model's fixed input size, W is always 1 by this point (see the
   STAGE2_PW_OUT_W define above), so this reduces to one scalar average per channel.

   input is HWC (CMSIS-NN's actual output layout - position-major, channel-minor: index =
   (y*w+x)*channels+c, confirmed directly from arm_convolve_s8.c/arm_nn_mat_mult_kernel_s8_s16.c/
   arm_depthwise_conv_s8_opt.c's own write patterns, all of which write every channel for one
   position before moving to the next - and from cmsis_nn_dims itself, {n,h,w,c}, c last).
   This function used to read it as CHW (channel-planar: (c*h+y)*w+x) instead - silently wrong
   for any channels>1, confirmed by a direct numeric check against int8_sim.py's
   avgpool_freq_s8 (max error 29 on a 0-127 scale) - not something a compile or a shape check
   would ever catch, since both formulas produce valid in-bounds indices. output stays a
   plain per-channel list (w=1 by the time this runs, so channel order is unambiguous either
   way). */

static void runAvgPoolFreq(const int8_t *input, int8_t *output, int32_t channels, int32_t h, int32_t w) {

    int32_t c, y, x;

    for (c = 0; c < channels; c += 1) {

        for (x = 0; x < w; x += 1) {

            int32_t sum = 0;

            for (y = 0; y < h; y += 1) sum += input[(y * w + x) * channels + c];

            output[c * w + x] = (int8_t)((sum >= 0 ? sum + h / 2 : sum - h / 2) / h);

        }

    }

}

static void quantizeInput(const float spectrogram[MODEL_INFERENCE_INPUT_FRAMES][MODEL_INFERENCE_INPUT_BINS]) {

    /* stftSpectrogramDb is [frame][bin] (time-major); the model's conv stack expects
       [bin][frame] (frequency = H, time = W) - transposed here while quantizing. */

    uint32_t frame, bin;

    for (bin = 0; bin < MODEL_INFERENCE_INPUT_BINS; bin += 1) {

        for (frame = 0; frame < MODEL_INFERENCE_INPUT_FRAMES; frame += 1) {

            float value = spectrogram[frame][bin] / MODEL_INT8_LSTM_SMALL_16_FOCAL_INPUT_SCALE;

            if (value > 127.0f) value = 127.0f;
            if (value < -128.0f) value = -128.0f;

            inputBuffer[bin * MODEL_INFERENCE_INPUT_FRAMES + frame] = (int8_t)(value >= 0.0f ? value + 0.5f : value - 0.5f);

        }

    }

}

static void runLSTMStep(void) {

    cmsis_nn_lstm_params params;

    params.time_major = 0;
    params.batch_size = 1;
    params.time_steps = 1;
    params.input_size = MODEL_INT8_LSTM_SMALL_16_FOCAL_INPUT_SIZE;
    params.hidden_size = MODEL_INT8_LSTM_SMALL_16_FOCAL_HIDDEN_SIZE;
    params.input_offset = 0;

    params.forget_to_cell_multiplier = MODEL_INT8_LSTM_SMALL_16_FOCAL_FORGET_TO_CELL_MULTIPLIER;
    params.forget_to_cell_shift = MODEL_INT8_LSTM_SMALL_16_FOCAL_FORGET_TO_CELL_SHIFT;
    params.input_to_cell_multiplier = MODEL_INT8_LSTM_SMALL_16_FOCAL_INPUT_TO_CELL_MULTIPLIER;
    params.input_to_cell_shift = MODEL_INT8_LSTM_SMALL_16_FOCAL_INPUT_TO_CELL_SHIFT;
    params.cell_clip = 32767;
    params.cell_scale_power = MODEL_INT8_LSTM_SMALL_16_FOCAL_CELL_SCALE_POWER;

    params.output_multiplier = MODEL_INT8_LSTM_SMALL_16_FOCAL_OUTPUT_MULTIPLIER;
    params.output_shift = MODEL_INT8_LSTM_SMALL_16_FOCAL_OUTPUT_SHIFT;
    params.output_offset = 0;

    params.forget_gate.input_multiplier = MODEL_INT8_LSTM_SMALL_16_FOCAL_FORGET_GATE_INPUT_MULTIPLIER;
    params.forget_gate.input_shift = MODEL_INT8_LSTM_SMALL_16_FOCAL_FORGET_GATE_INPUT_SHIFT;
    params.forget_gate.input_weights = model_int8_lstm_small_16_focal_forget_gate_input_weight;
    params.forget_gate.input_effective_bias = model_int8_lstm_small_16_focal_forget_gate_input_effective_bias;
    params.forget_gate.hidden_multiplier = MODEL_INT8_LSTM_SMALL_16_FOCAL_FORGET_GATE_HIDDEN_MULTIPLIER;
    params.forget_gate.hidden_shift = MODEL_INT8_LSTM_SMALL_16_FOCAL_FORGET_GATE_HIDDEN_SHIFT;
    params.forget_gate.hidden_weights = model_int8_lstm_small_16_focal_forget_gate_hidden_weight;
    params.forget_gate.hidden_effective_bias = lstmZeroEffectiveBias;
    params.forget_gate.bias = model_int8_lstm_small_16_focal_forget_gate_bias_q3_12;
    params.forget_gate.activation_type = ARM_SIGMOID;

    params.input_gate.input_multiplier = MODEL_INT8_LSTM_SMALL_16_FOCAL_INPUT_GATE_INPUT_MULTIPLIER;
    params.input_gate.input_shift = MODEL_INT8_LSTM_SMALL_16_FOCAL_INPUT_GATE_INPUT_SHIFT;
    params.input_gate.input_weights = model_int8_lstm_small_16_focal_input_gate_input_weight;
    params.input_gate.input_effective_bias = model_int8_lstm_small_16_focal_input_gate_input_effective_bias;
    params.input_gate.hidden_multiplier = MODEL_INT8_LSTM_SMALL_16_FOCAL_INPUT_GATE_HIDDEN_MULTIPLIER;
    params.input_gate.hidden_shift = MODEL_INT8_LSTM_SMALL_16_FOCAL_INPUT_GATE_HIDDEN_SHIFT;
    params.input_gate.hidden_weights = model_int8_lstm_small_16_focal_input_gate_hidden_weight;
    params.input_gate.hidden_effective_bias = lstmZeroEffectiveBias;
    params.input_gate.bias = model_int8_lstm_small_16_focal_input_gate_bias_q3_12;
    params.input_gate.activation_type = ARM_SIGMOID;

    params.cell_gate.input_multiplier = MODEL_INT8_LSTM_SMALL_16_FOCAL_CELL_GATE_INPUT_MULTIPLIER;
    params.cell_gate.input_shift = MODEL_INT8_LSTM_SMALL_16_FOCAL_CELL_GATE_INPUT_SHIFT;
    params.cell_gate.input_weights = model_int8_lstm_small_16_focal_cell_gate_input_weight;
    params.cell_gate.input_effective_bias = model_int8_lstm_small_16_focal_cell_gate_input_effective_bias;
    params.cell_gate.hidden_multiplier = MODEL_INT8_LSTM_SMALL_16_FOCAL_CELL_GATE_HIDDEN_MULTIPLIER;
    params.cell_gate.hidden_shift = MODEL_INT8_LSTM_SMALL_16_FOCAL_CELL_GATE_HIDDEN_SHIFT;
    params.cell_gate.hidden_weights = model_int8_lstm_small_16_focal_cell_gate_hidden_weight;
    params.cell_gate.hidden_effective_bias = lstmZeroEffectiveBias;
    params.cell_gate.bias = model_int8_lstm_small_16_focal_cell_gate_bias_q3_12;
    params.cell_gate.activation_type = ARM_TANH;

    params.output_gate.input_multiplier = MODEL_INT8_LSTM_SMALL_16_FOCAL_OUTPUT_GATE_INPUT_MULTIPLIER;
    params.output_gate.input_shift = MODEL_INT8_LSTM_SMALL_16_FOCAL_OUTPUT_GATE_INPUT_SHIFT;
    params.output_gate.input_weights = model_int8_lstm_small_16_focal_output_gate_input_weight;
    params.output_gate.input_effective_bias = model_int8_lstm_small_16_focal_output_gate_input_effective_bias;
    params.output_gate.hidden_multiplier = MODEL_INT8_LSTM_SMALL_16_FOCAL_OUTPUT_GATE_HIDDEN_MULTIPLIER;
    params.output_gate.hidden_shift = MODEL_INT8_LSTM_SMALL_16_FOCAL_OUTPUT_GATE_HIDDEN_SHIFT;
    params.output_gate.hidden_weights = model_int8_lstm_small_16_focal_output_gate_hidden_weight;
    params.output_gate.hidden_effective_bias = lstmZeroEffectiveBias;
    params.output_gate.bias = model_int8_lstm_small_16_focal_output_gate_bias_q3_12;
    params.output_gate.activation_type = ARM_SIGMOID;

    cmsis_nn_lstm_context buffers;
    buffers.temp1 = lstmTemp1;
    buffers.temp2 = lstmTemp2;
    buffers.cell_state = lstmCellState;

    arm_nn_lstm_step_s8(pooledBuffer, lstmHiddenState, lstmHiddenState, &params, &buffers, 1);

}

static float runClassifier(void) {

    int32_t acc = model_int8_lstm_small_16_focal_classifier_bias[0];

    uint32_t i;

    for (i = 0; i < MODEL_INT8_LSTM_SMALL_16_FOCAL_HIDDEN_SIZE; i += 1) {

        acc += (int32_t)model_int8_lstm_small_16_focal_classifier_weight[i] * (int32_t)lstmHiddenState[i];

    }

    return (float)acc * MODEL_INT8_LSTM_SMALL_16_FOCAL_CLASSIFIER_LOGIT_SCALE;

}

/* Elapsed microseconds between two AudioMoth_getTimeMicroseconds() samples - same formula
   used throughout main.c, duplicated here so this file doesn't depend on main.c's own
   MICROSECONDS_IN_SECOND define. */

static uint32_t elapsedMicroseconds(uint32_t startTime, uint32_t startUs, uint32_t endTime, uint32_t endUs) {

    return (endTime - startTime) * 1000000UL + endUs - startUs;

}

#define TIME_STAGE(field, call) { \
    uint32_t stageStartTime, stageStartUs, stageEndTime, stageEndUs; \
    AudioMoth_getTimeMicroseconds(&stageStartTime, &stageStartUs); \
    call; \
    AudioMoth_getTimeMicroseconds(&stageEndTime, &stageEndUs); \
    if (timing != NULL) timing->field = elapsedMicroseconds(stageStartTime, stageStartUs, stageEndTime, stageEndUs); \
}

float ModelInference_run(const float spectrogram[MODEL_INFERENCE_INPUT_FRAMES][MODEL_INFERENCE_INPUT_BINS],
                          ModelInferenceTiming_t *timing) {

    quantizeInput(spectrogram);

    TIME_STAGE(stage0Microseconds, runStage0Conv(inputBuffer, activationBufferA))

    TIME_STAGE(stage1DepthwiseMicroseconds, runDepthwiseConv(activationBufferA, activationBufferB,
                      model_int8_lstm_small_16_focal_stage1_dw_weight, model_int8_lstm_small_16_focal_stage1_dw_bias,
                      model_int8_lstm_small_16_focal_stage1_dw_multiplier, model_int8_lstm_small_16_focal_stage1_dw_shift,
                      MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE1_DW_IN_CH,
                      STAGE0_OUT_H, STAGE0_OUT_W, STAGE1_DW_OUT_H, STAGE1_DW_OUT_W,
                      MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE1_DW_KERNEL, MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE1_DW_STRIDE,
                      MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE1_DW_PADDING))

    TIME_STAGE(stage1PointwiseMicroseconds, runConv(activationBufferB, activationBufferA,
            model_int8_lstm_small_16_focal_stage1_pw_weight, model_int8_lstm_small_16_focal_stage1_pw_bias,
            model_int8_lstm_small_16_focal_stage1_pw_multiplier, model_int8_lstm_small_16_focal_stage1_pw_shift,
            MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE1_PW_IN_CH, MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE1_PW_OUT_CH,
            STAGE1_PW_OUT_H * STAGE1_PW_OUT_W))

    TIME_STAGE(stage2DepthwiseMicroseconds, runDepthwiseConv(activationBufferA, activationBufferB,
                      model_int8_lstm_small_16_focal_stage2_dw_weight, model_int8_lstm_small_16_focal_stage2_dw_bias,
                      model_int8_lstm_small_16_focal_stage2_dw_multiplier, model_int8_lstm_small_16_focal_stage2_dw_shift,
                      MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE2_DW_IN_CH,
                      STAGE1_PW_OUT_H, STAGE1_PW_OUT_W, STAGE2_DW_OUT_H, STAGE2_DW_OUT_W,
                      MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE2_DW_KERNEL, MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE2_DW_STRIDE,
                      MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE2_DW_PADDING))

    TIME_STAGE(stage2PointwiseMicroseconds, runConv(activationBufferB, activationBufferA,
            model_int8_lstm_small_16_focal_stage2_pw_weight, model_int8_lstm_small_16_focal_stage2_pw_bias,
            model_int8_lstm_small_16_focal_stage2_pw_multiplier, model_int8_lstm_small_16_focal_stage2_pw_shift,
            MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE2_PW_IN_CH, MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE2_PW_OUT_CH,
            STAGE2_PW_OUT_H * STAGE2_PW_OUT_W))

    TIME_STAGE(avgPoolMicroseconds, runAvgPoolFreq(activationBufferA, pooledBuffer, MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE2_PW_OUT_CH,
                   STAGE2_PW_OUT_H, STAGE2_PW_OUT_W))

    TIME_STAGE(lstmMicroseconds, runLSTMStep())

    float logit;

    TIME_STAGE(classifierMicroseconds, logit = runClassifier())

    return logit;

}
