/****************************************************************************
 * vit_inference.h
 * openacousticdevices.info
 *****************************************************************************/

#ifndef __VIT_INFERENCE_H
#define __VIT_INFERENCE_H

#include <stdint.h>

#include "vit_model.h"

#define VIT_INFERENCE_INPUT_FRAMES       7

/* Per-stage CPU-cycle breakdown (DWT->CYCCNT on the device, 0 on a host build) of the last
   ViTInference_run() call, summed over the buffer's VIT_INFERENCE_INPUT_FRAMES tokens. */

typedef struct {
    uint32_t patchEmbedCycles;                  /* input int8 quantization + patch_embed */
    uint32_t qkvCycles[VIT_DEPTH];              /* q/k/v projections + K/V ring buffer write */
    uint32_t qkCycles[VIT_DEPTH];               /* Q.K^T + ALiBi + row max + diff requantization */
    uint32_t softmaxCycles[VIT_DEPTH];          /* arm_softmax_s8 */
    uint32_t avCycles[VIT_DEPTH];               /* attn.V */
    uint32_t outProjCycles[VIT_DEPTH];
    uint32_t layerNorm1Cycles[VIT_DEPTH];       /* residual add + LayerNorm, float */
    uint32_t ffn1Cycles[VIT_DEPTH];             /* linear1 + fused ReLU */
    uint32_t ffn2Cycles[VIT_DEPTH];
    uint32_t layerNorm2Cycles[VIT_DEPTH];
    uint32_t classifierCycles;
    uint32_t tokensDone;                        /* < VIT_INFERENCE_INPUT_FRAMES if the deadline cut the buffer short */
} ViTInferenceTiming_t;

/* Clears the K/V history - call once before the first buffer of a recording. */

void ViTInference_reset(void);

/* Streams one buffer's VIT_INFERENCE_INPUT_FRAMES PCEN frames through the model, one token per
   frame, and returns the buffer's logit: the mean of its tokens' logits, as TinyViT's
   frame_logits() average-pools them. spectrogram has main.c's stftSpectrogramDb layout
   [frame][bin]. Carries K/V history across calls. timing must not be NULL.

   deadlineCycles: absolute DWT->CYCCNT value; if non-zero, no token is started that would,
   at the previous token's cost, end after it - the remaining tokens are skipped (see
   timing->tokensDone) and the logit is the mean over the tokens actually run. 0 = no limit. */

float ViTInference_run(const volatile float spectrogram[VIT_INFERENCE_INPUT_FRAMES][VIT_N_BINS],
                       ViTInferenceTiming_t *timing, uint32_t deadlineCycles);

#endif
