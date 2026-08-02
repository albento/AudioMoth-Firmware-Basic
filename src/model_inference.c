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
   STFT_OUTPUT_MODE_CSV (the same reasoning as stftRecordQueue in main.c). */

#define ACTIVATION_BUFFER_SIZE   4160

static int8_t *activationBufferA = (int8_t*)(AM_EXTERNAL_SRAM_START_ADDRESS);
static int8_t *activationBufferB = (int8_t*)(AM_EXTERNAL_SRAM_START_ADDRESS + ACTIVATION_BUFFER_SIZE);

/* Input tensor (quantized, transposed spectrogram) and the LSTM's tiny input vector both
   fit easily in internal RAM. */

static int8_t inputBuffer[MODEL_INFERENCE_INPUT_BINS * MODEL_INFERENCE_INPUT_FRAMES];

static int8_t pooledBuffer[MODEL_INT8_LSTM_SMALL_16_FOCAL_INPUT_SIZE];

/* Shared scratch buffer, reused sequentially (never concurrently) by both runConv() and
   runDepthwiseConv() - sized for the largest single requirement across all five stages.
   arm_convolve_s8's im2col-style need (2 * roundup4(kH*kW*Cin) * sizeof(int16_t)) tops out
   at 128 bytes (stage2_pw). arm_depthwise_conv_s8_opt's need
   (Cin * kH * kW * sizeof(int16_t), confirmed from
   arm_depthwise_conv_s8_opt_get_buffer_size_dsp()) tops out at stage2_dw: 32*3*3*2 = 576
   bytes - the largest of the two, so it sizes the buffer. */

static int16_t convScratchBuffer[288];

/* arm_nn_lstm_step_s8's temp1/temp2/cell_state are each only hidden_size * sizeof(int16_t)
   (confirmed from Source/NNSupportFunctions/arm_nn_lstm_step_s8.c) - negligible, and
   cell_state/hidden_state persist across calls for real streaming (state carried from one
   64ms buffer to the next, never reset mid-recording). */

static int16_t lstmTemp1[MODEL_INT8_LSTM_SMALL_16_FOCAL_HIDDEN_SIZE];
static int16_t lstmTemp2[MODEL_INT8_LSTM_SMALL_16_FOCAL_HIDDEN_SIZE];
static int16_t lstmCellState[MODEL_INT8_LSTM_SMALL_16_FOCAL_HIDDEN_SIZE];
static int8_t lstmHiddenState[MODEL_INT8_LSTM_SMALL_16_FOCAL_HIDDEN_SIZE];

void ModelInference_reset(void) {

    memset(lstmCellState, 0, sizeof(lstmCellState));
    memset(lstmHiddenState, 0, sizeof(lstmHiddenState));

}

/* Regular (groups=1) or pointwise (1x1) convolution via arm_convolve_s8, ReLU-fused
   (every stage in this model is Conv+BatchNorm+ReLU, BatchNorm already folded into the
   weights/bias at export time - see export_int8.py's fold_conv_bn()). Tried routing
   through arm_convolve_wrapper_s8 to get arm_convolve_1x1_s8_fast (im2col-free,
   arm_nn_mat_mult_nt_t_s8) for the two pointwise stages - measured SLOWER on real hardware
   (stage1_pw +41%, stage2_pw +75%), not faster as the source comments suggested. Reverted;
   arm_convolve_s8 is the measured-best option here. */

static void runConv(const int8_t *input, int8_t *output, const int8_t *weight, const int32_t *bias,
                     const int32_t *multiplier, const int32_t *shift, int32_t inCh, int32_t outCh,
                     int32_t inH, int32_t inW, int32_t outH, int32_t outW, int32_t kernel,
                     int32_t stride, int32_t padding) {

    cmsis_nn_context ctx = { convScratchBuffer, sizeof(convScratchBuffer) };

    cmsis_nn_conv_params convParams;
    convParams.input_offset = 0;
    convParams.output_offset = 0;
    convParams.stride.h = stride;
    convParams.stride.w = stride;
    convParams.padding.h = padding;
    convParams.padding.w = padding;
    convParams.dilation.h = 1;
    convParams.dilation.w = 1;
    convParams.activation.min = 0;
    convParams.activation.max = 127;

    cmsis_nn_per_channel_quant_params quantParams;
    quantParams.multiplier = (int32_t*)multiplier;
    quantParams.shift = (int32_t*)shift;

    cmsis_nn_dims inputDims = { 1, inH, inW, inCh };
    cmsis_nn_dims filterDims = { outCh, kernel, kernel, inCh };
    cmsis_nn_dims biasDims = { 1, 1, 1, outCh };
    cmsis_nn_dims outputDims = { 1, outH, outW, outCh };

    arm_convolve_s8(&ctx, &convParams, &quantParams, &inputDims, input, &filterDims, weight,
                     &biasDims, bias, NULL, &outputDims, output);

}

/* Depthwise convolution (groups == channels, channel multiplier 1) via
   arm_depthwise_conv_s8_opt - same ReLU fusion as runConv() above. Deliberately NOT
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
   STAGE2_PW_OUT_W define above), so this reduces to one scalar average per channel. */

static void runAvgPoolFreq(const int8_t *input, int8_t *output, int32_t channels, int32_t h, int32_t w) {

    int32_t c, y, x;

    for (c = 0; c < channels; c += 1) {

        for (x = 0; x < w; x += 1) {

            int32_t sum = 0;

            for (y = 0; y < h; y += 1) sum += input[(c * h + y) * w + x];

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
    params.forget_gate.input_effective_bias = NULL;
    params.forget_gate.hidden_multiplier = MODEL_INT8_LSTM_SMALL_16_FOCAL_FORGET_GATE_HIDDEN_MULTIPLIER;
    params.forget_gate.hidden_shift = MODEL_INT8_LSTM_SMALL_16_FOCAL_FORGET_GATE_HIDDEN_SHIFT;
    params.forget_gate.hidden_weights = model_int8_lstm_small_16_focal_forget_gate_hidden_weight;
    params.forget_gate.hidden_effective_bias = NULL;
    params.forget_gate.bias = model_int8_lstm_small_16_focal_forget_gate_bias_q3_12;
    params.forget_gate.activation_type = ARM_SIGMOID;

    params.input_gate.input_multiplier = MODEL_INT8_LSTM_SMALL_16_FOCAL_INPUT_GATE_INPUT_MULTIPLIER;
    params.input_gate.input_shift = MODEL_INT8_LSTM_SMALL_16_FOCAL_INPUT_GATE_INPUT_SHIFT;
    params.input_gate.input_weights = model_int8_lstm_small_16_focal_input_gate_input_weight;
    params.input_gate.input_effective_bias = NULL;
    params.input_gate.hidden_multiplier = MODEL_INT8_LSTM_SMALL_16_FOCAL_INPUT_GATE_HIDDEN_MULTIPLIER;
    params.input_gate.hidden_shift = MODEL_INT8_LSTM_SMALL_16_FOCAL_INPUT_GATE_HIDDEN_SHIFT;
    params.input_gate.hidden_weights = model_int8_lstm_small_16_focal_input_gate_hidden_weight;
    params.input_gate.hidden_effective_bias = NULL;
    params.input_gate.bias = model_int8_lstm_small_16_focal_input_gate_bias_q3_12;
    params.input_gate.activation_type = ARM_SIGMOID;

    params.cell_gate.input_multiplier = MODEL_INT8_LSTM_SMALL_16_FOCAL_CELL_GATE_INPUT_MULTIPLIER;
    params.cell_gate.input_shift = MODEL_INT8_LSTM_SMALL_16_FOCAL_CELL_GATE_INPUT_SHIFT;
    params.cell_gate.input_weights = model_int8_lstm_small_16_focal_cell_gate_input_weight;
    params.cell_gate.input_effective_bias = NULL;
    params.cell_gate.hidden_multiplier = MODEL_INT8_LSTM_SMALL_16_FOCAL_CELL_GATE_HIDDEN_MULTIPLIER;
    params.cell_gate.hidden_shift = MODEL_INT8_LSTM_SMALL_16_FOCAL_CELL_GATE_HIDDEN_SHIFT;
    params.cell_gate.hidden_weights = model_int8_lstm_small_16_focal_cell_gate_hidden_weight;
    params.cell_gate.hidden_effective_bias = NULL;
    params.cell_gate.bias = model_int8_lstm_small_16_focal_cell_gate_bias_q3_12;
    params.cell_gate.activation_type = ARM_TANH;

    params.output_gate.input_multiplier = MODEL_INT8_LSTM_SMALL_16_FOCAL_OUTPUT_GATE_INPUT_MULTIPLIER;
    params.output_gate.input_shift = MODEL_INT8_LSTM_SMALL_16_FOCAL_OUTPUT_GATE_INPUT_SHIFT;
    params.output_gate.input_weights = model_int8_lstm_small_16_focal_output_gate_input_weight;
    params.output_gate.input_effective_bias = NULL;
    params.output_gate.hidden_multiplier = MODEL_INT8_LSTM_SMALL_16_FOCAL_OUTPUT_GATE_HIDDEN_MULTIPLIER;
    params.output_gate.hidden_shift = MODEL_INT8_LSTM_SMALL_16_FOCAL_OUTPUT_GATE_HIDDEN_SHIFT;
    params.output_gate.hidden_weights = model_int8_lstm_small_16_focal_output_gate_hidden_weight;
    params.output_gate.hidden_effective_bias = NULL;
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

    TIME_STAGE(stage0Microseconds, runConv(inputBuffer, activationBufferA,
            model_int8_lstm_small_16_focal_stage0_weight, model_int8_lstm_small_16_focal_stage0_bias,
            model_int8_lstm_small_16_focal_stage0_multiplier, model_int8_lstm_small_16_focal_stage0_shift,
            MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE0_IN_CH, MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE0_OUT_CH,
            MODEL_INFERENCE_INPUT_BINS, MODEL_INFERENCE_INPUT_FRAMES, STAGE0_OUT_H, STAGE0_OUT_W,
            MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE0_KERNEL, MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE0_STRIDE,
            MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE0_PADDING))

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
            STAGE1_DW_OUT_H, STAGE1_DW_OUT_W, STAGE1_PW_OUT_H, STAGE1_PW_OUT_W,
            MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE1_PW_KERNEL, MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE1_PW_STRIDE,
            MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE1_PW_PADDING))

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
            STAGE2_DW_OUT_H, STAGE2_DW_OUT_W, STAGE2_PW_OUT_H, STAGE2_PW_OUT_W,
            MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE2_PW_KERNEL, MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE2_PW_STRIDE,
            MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE2_PW_PADDING))

    TIME_STAGE(avgPoolMicroseconds, runAvgPoolFreq(activationBufferA, pooledBuffer, MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE2_PW_OUT_CH,
                   STAGE2_PW_OUT_H, STAGE2_PW_OUT_W))

    TIME_STAGE(lstmMicroseconds, runLSTMStep())

    float logit;

    TIME_STAGE(classifierMicroseconds, logit = runClassifier())

    return logit;

}
