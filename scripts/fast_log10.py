#!/usr/bin/env python3
"""
Bit-for-bit Python port of the fixed-point log10 approximation used on-device
(see main.c: STFT_LOG2_BIAS_Q16_16, STFT_LOG2_TO_LOG10_Q16_16,
stftFastLog2Q16_16(), stftFastLog10()), for use as the feature extraction step
of a training pipeline - so the model is trained on exactly the same values
the firmware will produce at inference time, rather than on "clean" log10.

Assumes a little-endian host (true for essentially all modern desktop/laptop
CPUs, and matches the AudioMoth's Cortex-M4 in its default configuration).

Usage:
    from fast_log10 import fast_log10_db
    spectrogram_db = fast_log10_db(magnitude)   # magnitude: any shape, any float dtype
"""

import numpy as np

STFT_LOG2_BIAS_Q16_16 = np.int32(8320252)      # round(126.95696433397204 * 65536)
STFT_LOG2_TO_LOG10_Q16_16 = np.int32(19728)    # round(log10(2) * 65536)
STFT_LOG_EPSILON = np.float32(1.0e-6)


def fast_log2_q16_16(x):
    """Equivalent of stftFastLog2Q16_16(): log2(x) as a Q16.16 fixed-point
    value (real value = result / 65536), packed in an int32."""

    x = np.asarray(x, dtype=np.float32)
    i = x.view(np.int32)                                    # reinterpret IEEE-754 bits, like the C memcpy
    shifted = (i >> np.int32(7)).astype(np.int32)
    return (shifted - STFT_LOG2_BIAS_Q16_16).astype(np.int32)


def fast_log10(x):
    """Equivalent of stftFastLog10(): fixed-point approximation of log10(x)."""

    log2_fixed = fast_log2_q16_16(x)
    product = log2_fixed.astype(np.int64) * np.int64(STFT_LOG2_TO_LOG10_Q16_16)
    log10_fixed = (product >> np.int64(16)).astype(np.int32)
    return log10_fixed.astype(np.float32) / np.float32(65536.0)


def fast_log10_db(magnitude):
    """Equivalent of the firmware's 20.0f * stftFastLog10(magnitude + STFT_LOG_EPSILON) -
    exactly what ends up stored in the on-device spectrogram."""

    magnitude = np.asarray(magnitude, dtype=np.float32)
    return np.float32(20.0) * fast_log10(magnitude + STFT_LOG_EPSILON)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.uniform(1e-6, 1e6, size=200000).astype(np.float32)

    approx = fast_log10(x)
    real = np.log10(x)
    max_abs_err = np.max(np.abs(approx - real))

    print(f"max abs error on log10(x): {max_abs_err:.6f}")
    print(f"max abs error on 20*log10(x) (dB): {max_abs_err * 20:.4f}")

    for value in (1.0, 10.0, 100.0, 0.5, 2.0, 1e-6, 123456.0):
        print(f"x={value:<12g} real={np.log10(value):.5f} approx={fast_log10(np.float32(value)):.5f}")
