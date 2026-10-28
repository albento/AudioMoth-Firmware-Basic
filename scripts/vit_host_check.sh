#!/bin/sh
# Builds scripts/vit_host_test.c against src/vit_inference.c + EXPORT_DIR/vit_model.c + CMSIS-NN's
# portable C kernels (bit-exact with its Cortex-M4 DSP ones) for the host, runs it on every
# EXPORT_DIR/testvec/*.spec written by export_vit.py, then compares with the float PyTorch logits.
#
# Usage: scripts/vit_host_check.sh EXPORT_DIR [CMSIS_NN_DIR]
#   CMSIS_NN_DIR defaults to ../AudioMoth-Project/CMSIS-NN; PYTHON to test-audiomoth-cnn's venv.
#   CFLAGS_EXTRA=-DVIT_SOFTMAX_LUT=0 checks the arm_softmax_s8 variant instead.
set -e
REPO=$(cd "$(dirname "$0")/.." && pwd)
GEN=$(cd "$1" && pwd)
NN=${2:-$REPO/../AudioMoth-Project/CMSIS-NN}
PYTHON=${PYTHON:-$REPO/../test-audiomoth-cnn/.venv/bin/python}
BIN=$(mktemp -d)/vit_host_test
gcc -O2 -std=c99 $CFLAGS_EXTRA -Wall -Wno-unused-function -I"$REPO/inc" -I"$GEN" -I"$NN/Include" -o "$BIN" \
    "$REPO/scripts/vit_host_test.c" "$REPO/src/vit_inference.c" "$GEN/vit_model.c" \
    "$NN/Source/FullyConnectedFunctions/arm_fully_connected_s8.c" \
    "$NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s8.c" \
    "$NN/Source/SoftmaxFunctions/arm_softmax_s8.c" \
    "$NN/Source/SoftmaxFunctions/arm_nn_softmax_common_s8.c" -lm
for spec in "$GEN"/testvec/*.spec; do
    "$BIN" "$spec" "${spec%.spec}.c_logits"
done
"$PYTHON" -I "$REPO/scripts/vit_host_compare.py" "$GEN/testvec"
