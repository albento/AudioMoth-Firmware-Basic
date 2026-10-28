/* Host (x86) run of the exact firmware ViT code (fw/vit_inference.c + generated vit_model.c +
   CMSIS-NN's portable C kernels) on a .spec test vector written by export_vit.py:
   int32 T, int32 F, then float32 [T][F] PCEN frames. Writes one float32 logit per buffer. */

#include <stdio.h>
#include <stdlib.h>

#include "vit_inference.h"

int main(int argc, char **argv) {

    if (argc != 3) {
        fprintf(stderr, "usage: %s input.spec output.logits\n", argv[0]);
        return 1;
    }

    FILE *in = fopen(argv[1], "rb");
    int32_t shape[2];
    if (!in || fread(shape, sizeof(int32_t), 2, in) != 2 || shape[1] != VIT_N_BINS) {
        fprintf(stderr, "bad input %s\n", argv[1]);
        return 1;
    }

    int32_t numberOfBuffers = shape[0] / VIT_INFERENCE_INPUT_FRAMES;
    float buffer[VIT_INFERENCE_INPUT_FRAMES][VIT_N_BINS];
    float *logits = malloc(sizeof(float) * numberOfBuffers);
    ViTInferenceTiming_t timing;

    ViTInference_reset();

    for (int32_t b = 0; b < numberOfBuffers; b += 1) {
        if (fread(buffer, sizeof(buffer), 1, in) != 1) {
            fprintf(stderr, "short read at buffer %d\n", b);
            return 1;
        }
        logits[b] = ViTInference_run(buffer, &timing, 0);
    }

    fclose(in);

    FILE *out = fopen(argv[2], "wb");
    fwrite(logits, sizeof(float), numberOfBuffers, out);
    fclose(out);
    free(logits);

    return 0;
}
