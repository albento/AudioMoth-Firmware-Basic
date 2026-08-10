#ifndef MODEL_H
#define MODEL_H

/* Exported from --tag "lstm_16_32_64_h64_pcen" - see export_int8.py's --tag/--model-name. */
#define MODEL_TAG "lstm_16_32_64_h64_pcen"

#include <stdint.h>

/* stage0: weight [Cout=16, Cin/g=1, kH=3, kW=3] row-major, stride=2 padding=1 groups=1 */
#define MODEL_STAGE0_IN_CH 1
#define MODEL_STAGE0_OUT_CH 16
#define MODEL_STAGE0_KERNEL 3
#define MODEL_STAGE0_STRIDE 2
#define MODEL_STAGE0_PADDING 1
#define MODEL_STAGE0_GROUPS 1
extern const int8_t model_stage0_weight[144];
extern const int32_t model_stage0_bias[16];
extern const int32_t model_stage0_multiplier[16];
extern const int32_t model_stage0_shift[16];

/* stage1_dw (depthwise): weight [kH=3, kW=3, channels=16] row-major (CMSIS-NN depthwise layout, NOT PyTorch's native [Cout, 1, kH, kW]), stride=2 padding=1 groups=16 */
#define MODEL_STAGE1_DW_IN_CH 16
#define MODEL_STAGE1_DW_OUT_CH 16
#define MODEL_STAGE1_DW_KERNEL 3
#define MODEL_STAGE1_DW_STRIDE 2
#define MODEL_STAGE1_DW_PADDING 1
#define MODEL_STAGE1_DW_GROUPS 16
extern const int8_t model_stage1_dw_weight[144];
extern const int32_t model_stage1_dw_bias[16];
extern const int32_t model_stage1_dw_multiplier[16];
extern const int32_t model_stage1_dw_shift[16];

/* stage1_pw: weight [Cout=32, Cin/g=16, kH=1, kW=1] row-major, stride=1 padding=0 groups=1 */
#define MODEL_STAGE1_PW_IN_CH 16
#define MODEL_STAGE1_PW_OUT_CH 32
#define MODEL_STAGE1_PW_KERNEL 1
#define MODEL_STAGE1_PW_STRIDE 1
#define MODEL_STAGE1_PW_PADDING 0
#define MODEL_STAGE1_PW_GROUPS 1
extern const int8_t model_stage1_pw_weight[512];
extern const int32_t model_stage1_pw_bias[32];
extern const int32_t model_stage1_pw_multiplier[32];
extern const int32_t model_stage1_pw_shift[32];

/* stage2_dw (depthwise): weight [kH=3, kW=3, channels=32] row-major (CMSIS-NN depthwise layout, NOT PyTorch's native [Cout, 1, kH, kW]), stride=2 padding=1 groups=32 */
#define MODEL_STAGE2_DW_IN_CH 32
#define MODEL_STAGE2_DW_OUT_CH 32
#define MODEL_STAGE2_DW_KERNEL 3
#define MODEL_STAGE2_DW_STRIDE 2
#define MODEL_STAGE2_DW_PADDING 1
#define MODEL_STAGE2_DW_GROUPS 32
extern const int8_t model_stage2_dw_weight[288];
extern const int32_t model_stage2_dw_bias[32];
extern const int32_t model_stage2_dw_multiplier[32];
extern const int32_t model_stage2_dw_shift[32];

/* stage2_pw: weight [Cout=64, Cin/g=32, kH=1, kW=1] row-major, stride=1 padding=0 groups=1 */
#define MODEL_STAGE2_PW_IN_CH 32
#define MODEL_STAGE2_PW_OUT_CH 64
#define MODEL_STAGE2_PW_KERNEL 1
#define MODEL_STAGE2_PW_STRIDE 1
#define MODEL_STAGE2_PW_PADDING 0
#define MODEL_STAGE2_PW_GROUPS 1
extern const int8_t model_stage2_pw_weight[2048];
extern const int32_t model_stage2_pw_bias[64];
extern const int32_t model_stage2_pw_multiplier[64];
extern const int32_t model_stage2_pw_shift[64];

#define MODEL_INPUT_SIZE 64
#define MODEL_HIDDEN_SIZE 64
#define MODEL_CELL_SCALE_POWER -2
#define MODEL_FORGET_TO_CELL_MULTIPLIER 1073741824
#define MODEL_FORGET_TO_CELL_SHIFT -14
#define MODEL_INPUT_TO_CELL_MULTIPLIER 1073741824
#define MODEL_INPUT_TO_CELL_SHIFT -27
#define MODEL_OUTPUT_MULTIPLIER 1076350753
#define MODEL_OUTPUT_SHIFT -22

/* forget gate: input_weight [hidden=64, input=64], hidden_weight [hidden=64, hidden=64]. bias_q3_12 is for a from-scratch Python simulator only (int8_sim.py) - the real arm_nn_lstm_step_s8() kernel never reads it, only input_effective_bias (same bias, pre-multiply-shift accumulator domain - see GateParams' docstring in int8_sim.py). hidden_effective_bias is always all-zero (not exported - see model_inference.c, which supplies its own static zero array), since the real bias is added exactly once, during the input-weights pass only. */
#define MODEL_FORGET_GATE_INPUT_MULTIPLIER 1154236785
#define MODEL_FORGET_GATE_INPUT_SHIFT 3
#define MODEL_FORGET_GATE_HIDDEN_MULTIPLIER 1577320256
#define MODEL_FORGET_GATE_HIDDEN_SHIFT -1
extern const int8_t model_forget_gate_input_weight[4096];
extern const int8_t model_forget_gate_hidden_weight[4096];
extern const int32_t model_forget_gate_bias_q3_12[64];
extern const int32_t model_forget_gate_input_effective_bias[64];

/* input gate: input_weight [hidden=64, input=64], hidden_weight [hidden=64, hidden=64]. bias_q3_12 is for a from-scratch Python simulator only (int8_sim.py) - the real arm_nn_lstm_step_s8() kernel never reads it, only input_effective_bias (same bias, pre-multiply-shift accumulator domain - see GateParams' docstring in int8_sim.py). hidden_effective_bias is always all-zero (not exported - see model_inference.c, which supplies its own static zero array), since the real bias is added exactly once, during the input-weights pass only. */
#define MODEL_INPUT_GATE_INPUT_MULTIPLIER 1128972420
#define MODEL_INPUT_GATE_INPUT_SHIFT 3
#define MODEL_INPUT_GATE_HIDDEN_MULTIPLIER 2093975149
#define MODEL_INPUT_GATE_HIDDEN_SHIFT -1
extern const int8_t model_input_gate_input_weight[4096];
extern const int8_t model_input_gate_hidden_weight[4096];
extern const int32_t model_input_gate_bias_q3_12[64];
extern const int32_t model_input_gate_input_effective_bias[64];

/* cell gate: input_weight [hidden=64, input=64], hidden_weight [hidden=64, hidden=64]. bias_q3_12 is for a from-scratch Python simulator only (int8_sim.py) - the real arm_nn_lstm_step_s8() kernel never reads it, only input_effective_bias (same bias, pre-multiply-shift accumulator domain - see GateParams' docstring in int8_sim.py). hidden_effective_bias is always all-zero (not exported - see model_inference.c, which supplies its own static zero array), since the real bias is added exactly once, during the input-weights pass only. */
#define MODEL_CELL_GATE_INPUT_MULTIPLIER 1298187574
#define MODEL_CELL_GATE_INPUT_SHIFT 3
#define MODEL_CELL_GATE_HIDDEN_MULTIPLIER 1131058664
#define MODEL_CELL_GATE_HIDDEN_SHIFT -1
extern const int8_t model_cell_gate_input_weight[4096];
extern const int8_t model_cell_gate_hidden_weight[4096];
extern const int32_t model_cell_gate_bias_q3_12[64];
extern const int32_t model_cell_gate_input_effective_bias[64];

/* output gate: input_weight [hidden=64, input=64], hidden_weight [hidden=64, hidden=64]. bias_q3_12 is for a from-scratch Python simulator only (int8_sim.py) - the real arm_nn_lstm_step_s8() kernel never reads it, only input_effective_bias (same bias, pre-multiply-shift accumulator domain - see GateParams' docstring in int8_sim.py). hidden_effective_bias is always all-zero (not exported - see model_inference.c, which supplies its own static zero array), since the real bias is added exactly once, during the input-weights pass only. */
#define MODEL_OUTPUT_GATE_INPUT_MULTIPLIER 1367121256
#define MODEL_OUTPUT_GATE_INPUT_SHIFT 3
#define MODEL_OUTPUT_GATE_HIDDEN_MULTIPLIER 1633196484
#define MODEL_OUTPUT_GATE_HIDDEN_SHIFT -1
extern const int8_t model_output_gate_input_weight[4096];
extern const int8_t model_output_gate_hidden_weight[4096];
extern const int32_t model_output_gate_bias_q3_12[64];
extern const int32_t model_output_gate_input_effective_bias[64];

/* classifier: weight [1, hidden=64] */
extern const int8_t model_classifier_weight[64];
extern const int32_t model_classifier_bias[1];

/* Multiply the classifier's raw int32 accumulator by this to get a float logit (no further int8 requantization on the final output - see int8_forward()'s docstring) */
#define MODEL_CLASSIFIER_LOGIT_SCALE 5.7823817770e-05f

/* Quantize the incoming spectrogram with this scale before feeding it to stage0 */
#define MODEL_INPUT_SCALE 6.8975652259e+00f

#endif
