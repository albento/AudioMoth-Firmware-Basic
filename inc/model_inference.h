/****************************************************************************
 * model_inference.h
 * openacousticdevices.info
 *****************************************************************************/

#ifndef __MODEL_INFERENCE_H
#define __MODEL_INFERENCE_H

#include <stdint.h>

#define MODEL_INFERENCE_INPUT_FRAMES     7
#define MODEL_INFERENCE_INPUT_BINS       129

/* Per-stage microsecond breakdown of the last ModelInference_run() call, for profiling -
   filled in whenever a non-NULL pointer is passed to ModelInference_run(). */

typedef struct {
    uint32_t stage0Microseconds;
    uint32_t stage1DepthwiseMicroseconds;
    uint32_t stage1PointwiseMicroseconds;
    uint32_t stage2DepthwiseMicroseconds;
    uint32_t stage2PointwiseMicroseconds;
    uint32_t avgPoolMicroseconds;
    uint32_t lstmMicroseconds;
    uint32_t classifierMicroseconds;
} ModelInferenceTiming_t;

/* Resets the LSTM's persistent hidden/cell state - call once before the first buffer of a
   new recording, so state doesn't leak in from a previous run. */

void ModelInference_reset(void);

/* Runs the classifier on one buffer's spectrogram and returns its float logit (apply
   sigmoid for a probability). spectrogram must be [MODEL_INFERENCE_INPUT_FRAMES]
   [MODEL_INFERENCE_INPUT_BINS], time-major/frequency-minor - the same layout as main.c's
   stftSpectrogramDb (this function transposes internally to the model's expected
   [frequency][time] layout). Carries LSTM state across calls - call once per buffer, in
   order, for real streaming inference, not a fresh reset each time. timing may be NULL if
   the per-stage breakdown isn't needed. */

float ModelInference_run(const float spectrogram[MODEL_INFERENCE_INPUT_FRAMES][MODEL_INFERENCE_INPUT_BINS],
                          ModelInferenceTiming_t *timing);

#endif
