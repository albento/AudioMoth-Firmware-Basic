#ifndef MODEL_INT8_LSTM_SMALL_16_FOCAL_H
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_H

#include <stdint.h>

/* stage0: weight [Cout=16, Cin/g=1, kH=3, kW=3] row-major, stride=2 padding=1 groups=1 */
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE0_IN_CH 1
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE0_OUT_CH 16
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE0_KERNEL 3
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE0_STRIDE 2
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE0_PADDING 1
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE0_GROUPS 1
extern const int8_t model_int8_lstm_small_16_focal_stage0_weight[144];
extern const int32_t model_int8_lstm_small_16_focal_stage0_bias[16];
extern const int32_t model_int8_lstm_small_16_focal_stage0_multiplier[16];
extern const int32_t model_int8_lstm_small_16_focal_stage0_shift[16];

/* stage1_dw: weight [Cout=16, Cin/g=1, kH=3, kW=3] row-major, stride=2 padding=1 groups=16 */
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE1_DW_IN_CH 16
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE1_DW_OUT_CH 16
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE1_DW_KERNEL 3
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE1_DW_STRIDE 2
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE1_DW_PADDING 1
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE1_DW_GROUPS 16
extern const int8_t model_int8_lstm_small_16_focal_stage1_dw_weight[144];
extern const int32_t model_int8_lstm_small_16_focal_stage1_dw_bias[16];
extern const int32_t model_int8_lstm_small_16_focal_stage1_dw_multiplier[16];
extern const int32_t model_int8_lstm_small_16_focal_stage1_dw_shift[16];

/* stage1_pw: weight [Cout=32, Cin/g=16, kH=1, kW=1] row-major, stride=1 padding=0 groups=1 */
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE1_PW_IN_CH 16
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE1_PW_OUT_CH 32
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE1_PW_KERNEL 1
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE1_PW_STRIDE 1
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE1_PW_PADDING 0
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE1_PW_GROUPS 1
extern const int8_t model_int8_lstm_small_16_focal_stage1_pw_weight[512];
extern const int32_t model_int8_lstm_small_16_focal_stage1_pw_bias[32];
extern const int32_t model_int8_lstm_small_16_focal_stage1_pw_multiplier[32];
extern const int32_t model_int8_lstm_small_16_focal_stage1_pw_shift[32];

/* stage2_dw: weight [Cout=32, Cin/g=1, kH=3, kW=3] row-major, stride=2 padding=1 groups=32 */
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE2_DW_IN_CH 32
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE2_DW_OUT_CH 32
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE2_DW_KERNEL 3
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE2_DW_STRIDE 2
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE2_DW_PADDING 1
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE2_DW_GROUPS 32
extern const int8_t model_int8_lstm_small_16_focal_stage2_dw_weight[288];
extern const int32_t model_int8_lstm_small_16_focal_stage2_dw_bias[32];
extern const int32_t model_int8_lstm_small_16_focal_stage2_dw_multiplier[32];
extern const int32_t model_int8_lstm_small_16_focal_stage2_dw_shift[32];

/* stage2_pw: weight [Cout=64, Cin/g=32, kH=1, kW=1] row-major, stride=1 padding=0 groups=1 */
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE2_PW_IN_CH 32
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE2_PW_OUT_CH 64
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE2_PW_KERNEL 1
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE2_PW_STRIDE 1
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE2_PW_PADDING 0
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_STAGE2_PW_GROUPS 1
extern const int8_t model_int8_lstm_small_16_focal_stage2_pw_weight[2048];
extern const int32_t model_int8_lstm_small_16_focal_stage2_pw_bias[64];
extern const int32_t model_int8_lstm_small_16_focal_stage2_pw_multiplier[64];
extern const int32_t model_int8_lstm_small_16_focal_stage2_pw_shift[64];

#define MODEL_INT8_LSTM_SMALL_16_FOCAL_INPUT_SIZE 64
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_HIDDEN_SIZE 16
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_CELL_SCALE_POWER -12
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_FORGET_TO_CELL_MULTIPLIER 1073741824
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_FORGET_TO_CELL_SHIFT -14
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_INPUT_TO_CELL_MULTIPLIER 1073741824
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_INPUT_TO_CELL_SHIFT -17
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_OUTPUT_MULTIPLIER 2132012665
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_OUTPUT_SHIFT -23

/* forget gate: input_weight [hidden=16, input=64], hidden_weight [hidden=16, hidden=16], bias already in fixed Q3.12 */
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_FORGET_GATE_INPUT_MULTIPLIER 1607176982
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_FORGET_GATE_INPUT_SHIFT 4
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_FORGET_GATE_HIDDEN_MULTIPLIER 1148300376
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_FORGET_GATE_HIDDEN_SHIFT -1
extern const int8_t model_int8_lstm_small_16_focal_forget_gate_input_weight[1024];
extern const int8_t model_int8_lstm_small_16_focal_forget_gate_hidden_weight[256];
extern const int32_t model_int8_lstm_small_16_focal_forget_gate_bias_q3_12[16];

/* input gate: input_weight [hidden=16, input=64], hidden_weight [hidden=16, hidden=16], bias already in fixed Q3.12 */
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_INPUT_GATE_INPUT_MULTIPLIER 1646918627
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_INPUT_GATE_INPUT_SHIFT 4
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_INPUT_GATE_HIDDEN_MULTIPLIER 1768098014
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_INPUT_GATE_HIDDEN_SHIFT -1
extern const int8_t model_int8_lstm_small_16_focal_input_gate_input_weight[1024];
extern const int8_t model_int8_lstm_small_16_focal_input_gate_hidden_weight[256];
extern const int32_t model_int8_lstm_small_16_focal_input_gate_bias_q3_12[16];

/* cell gate: input_weight [hidden=16, input=64], hidden_weight [hidden=16, hidden=16], bias already in fixed Q3.12 */
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_CELL_GATE_INPUT_MULTIPLIER 2036284874
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_CELL_GATE_INPUT_SHIFT 3
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_CELL_GATE_HIDDEN_MULTIPLIER 1792496823
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_CELL_GATE_HIDDEN_SHIFT -2
extern const int8_t model_int8_lstm_small_16_focal_cell_gate_input_weight[1024];
extern const int8_t model_int8_lstm_small_16_focal_cell_gate_hidden_weight[256];
extern const int32_t model_int8_lstm_small_16_focal_cell_gate_bias_q3_12[16];

/* output gate: input_weight [hidden=16, input=64], hidden_weight [hidden=16, hidden=16], bias already in fixed Q3.12 */
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_OUTPUT_GATE_INPUT_MULTIPLIER 1553844503
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_OUTPUT_GATE_INPUT_SHIFT 4
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_OUTPUT_GATE_HIDDEN_MULTIPLIER 1181051987
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_OUTPUT_GATE_HIDDEN_SHIFT 0
extern const int8_t model_int8_lstm_small_16_focal_output_gate_input_weight[1024];
extern const int8_t model_int8_lstm_small_16_focal_output_gate_hidden_weight[256];
extern const int32_t model_int8_lstm_small_16_focal_output_gate_bias_q3_12[16];

/* classifier: weight [1, hidden=16] */
extern const int8_t model_int8_lstm_small_16_focal_classifier_weight[16];
extern const int32_t model_int8_lstm_small_16_focal_classifier_bias[1];

/* Multiply the classifier's raw int32 accumulator by this to get a float logit (no further int8 requantization on the final output - see int8_forward()'s docstring) */
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_CLASSIFIER_LOGIT_SCALE 5.3132862377e-05f

/* Quantize the incoming spectrogram with this scale before feeding it to stage0 */
#define MODEL_INT8_LSTM_SMALL_16_FOCAL_INPUT_SCALE 9.4376931979e-01f

#endif
