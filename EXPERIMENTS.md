# STFT feasibility experiments (`debug` branch)

Summary of the experiments run to check whether computing a spectrogram on-device, in real
time, is feasible on the AudioMoth Dev as a first step toward on-device animal-call detection
(STFT feature extraction now, CMSIS-NN classifier later).

## Goal and method

Can the AudioMoth compute a usable spectrogram of each incoming audio buffer fast enough to
leave real time budget for a classifier afterwards? `STFT_DEBUG_MODE` in `src/main.c` replaces
the normal recording ISR with STFT computation, timed with `AudioMoth_getTimeMicroseconds()`
and logged to the SD card, so the numbers below come from the real AudioMoth Dev hardware, not
estimates.

## Hardware and parameters

- **Board**: AudioMoth Dev (EFM32WG380F256, Cortex-M4F @ 48MHz, 256KB flash, 32KB internal RAM +
  256KB external SRAM)
- **Sample rate**: 16kHz, raw/undecimated (`sampleRateDivider = 1`)
- **Buffer**: 1024 samples = 64ms (`MAXIMUM_SAMPLES_IN_DMA_TRANSFER`) - this is the real-time
  budget every result below is measured against
- **STFT**: 256-point FFT (CMSIS-DSP `arm_rfft_fast_f32`), Hann window, 128-sample hop (50%
  overlap) → 7 frames x 129 frequency bins per buffer (62.5 Hz/bin, up to 8kHz)

## Results: magnitude-scaling comparison

One buffer's STFT (7 frames) computed and timed per DMA interrupt; CSV logged to the SD card,
one line per buffer (`buffer_index,stft_time_us`).

| Method | Mean time/buffer | % of 64ms budget | Cost of the log step alone | Cost per log call (903/buffer) |
|---|---|---|---|---|
| Linear (magnitude only, no log) | 4162 µs | 6.5% | - | - |
| **Fixed-point log10 (bit-hack)** | **4482 µs** | **7.0%** | **320 µs** | **~0.35 µs** |
| `arm_vlog_f32` (natural log + multiply) | 8412 µs | 13.1% | 4249 µs | ~4.71 µs |
| `log10f` (libm) | 10887 µs | 17.0% | 6724 µs | ~7.45 µs |

**Takeaways**:
- The FFT + windowing + magnitude themselves cost only ~6.5% of the budget - never the
  bottleneck.
- `log10f` dominates STFT cost (~62% of total time) and is the first thing worth optimising.
- `arm_vlog_f32` on Cortex-M4 falls back to a scalar `logf()` loop (no MVE/Helium on this core),
  but is still ~37% cheaper than `log10f` for the log step - `log10f` does its own, more
  expensive base-10 range reduction internally.
- A fixed-point log10 approximation (IEEE-754 bit-hack, see below) is ~21x cheaper than
  `log10f` and ~13x cheaper than `arm_vlog_f32`, at a cost barely above the no-log baseline
  (+320 µs, +7.7% relative to linear). This is the clear winner if a dB-like scale is wanted.

*(A first fixed-point measurement of 3669 µs was invalid: `stftSpectrogramDb` had lost its
`volatile` qualifier when the audio/spectrogram-saving feature was added, and in CSV-only
output mode nothing reads it back - the compiler proved the whole magnitude/log computation
was dead code and removed it, so that number only measured the FFT+windowing. Fixed by keeping
`stftSpectrogramDb` `volatile` regardless of output mode; confirmed via disassembly that the
`vsqrt.f32` and log10 bit-hack instructions are present before re-measuring.)*

## Fixed-point log10 approximation

Classic bit-hack: for a positive `float32` `x`, its raw IEEE-754 bit pattern `i` satisfies
`log2(x) ~= i / 2^23 - 127`. Since `(i / 2^23) * 2^16 == i >> 7`, this gives `log2(x)` in
Q16.16 fixed point using only a shift and a subtract - no float division/multiplication.
`log10(x)` follows from a single Q16.16 fixed-point multiply by `log10(2)`.

The naive bias of 127 is exact only at powers of two and otherwise always undershoots (the
true `log2(1+m)` curve for mantissa `m` in `[0,1)` is concave, so the linear approximation
stays below it), with a worst case of 0.0861 (log2 units) at `m = 1/ln(2) - 1`. Using a
corrected bias of **126.95696433397204** instead (derived analytically and verified by
numerical scan) re-centres that error around zero, halving the worst case to **0.0430**, i.e.
**~0.26dB** max error on the final `20*log10(x)` result - for the same cost, since it's still
one constant.

Implementation: `main.c` (`stftFastLog2Q16_16`, `stftFastLog10`,
`STFT_LOG_MODE_LOG10_FIXED`). Bit-for-bit equivalent Python port for training/inference
feature-parity: `scripts/fast_log10.py`.

## Memory footprint

- All STFT working buffers (window, frame, FFT output, spectrogram scratch, timing queue) live
  in **internal RAM** - verified via `nm` symbol addresses (`0x2000xxxx` range), ~7.1KB total.
- **External SRAM (256KB)** is enabled but otherwise idle in the CSV-timing configuration; it is
  used as a per-buffer record queue (audio + full spectrogram, ~5.5KB/record, 8-deep ring) when
  `STFT_OUTPUT_MODE_BINARY` is selected instead of `STFT_OUTPUT_MODE_CSV`.
- Doubling the buffer size (1024 -> 2048 samples, 128ms) would not currently fit in the ~6KB of
  free internal RAM left after the normal firmware's own usage, unless the per-buffer
  spectrogram result were moved into external SRAM instead of internal RAM (not implemented,
  discussed as a future option, no information loss unlike the "single reused row" alternative
  which was ruled out for real classifier use since it would discard all past frames).
- CMSIS-DSP's generic `arm_rfft_fast_init_f32()` statically references twiddle tables for every
  supported FFT size (16 to 4096), costing ~64KB of otherwise-unreachable flash even though only
  256 is used; the fix (call the size-specific `arm_rfft_fast_init_256_f32()` instead) was
  identified but not yet applied.
- Architectural note for the eventual classifier: a streaming/recurrent model (GRU/LSTM, backed
  by CMSIS-NN's `LSTMFunctions`) could consume one frame (129 values) at a time and only needs a
  small persistent hidden state - no need to buffer multiple frames. A 2D-CNN-over-spectrogram
  model would need an actual multi-frame buffer (not necessarily every frame, but more than one).

## Data capture (audio + spectrogram) for offline inspection

`STFT_OUTPUT_MODE_BINARY` writes a self-describing binary file (`STFT_DATA_*.BIN`): a header
(sample rate, FFT/hop length, frame/bin counts) followed by one record per buffer (raw 1024-
sample audio, full 7x129 spectrogram, timing). Read and plotted with `scripts/plot_stft_data.py`
(single record, or `--all` to concatenate every record and compare against a reference STFT
computed continuously over the whole audio).

That comparison surfaced a real edge effect: per-buffer STFT restarts its frames at every 1024-
sample boundary, so it is missing the overlap frame (at hop spacing) that a continuous STFT
would compute across each boundary - one "gap" every 64ms, visible as the buffer-boundary frames
where the per-buffer and continuous versions diverge in `--all`'s time-aligned comparison plot.

## On-device INT8 model inference (TinyLSTMCNN via CMSIS-NN)

`test-audiomoth-cnn/export_int8.py` exports a trained `TinyLSTMCNN` (conv channels
16/32/64, LSTM hidden size 16) to INT8 C arrays (`model/model_int8_lstm_small_16_focal.{h,c}`).
`src/model_inference.c` runs it on-device with CMSIS-NN kernels, one buffer at a time,
carrying LSTM state across buffers (true streaming, not reset per-buffer) - the on-device
counterpart of `test-audiomoth-cnn/int8_sim.py`, mirrored stage for stage. Wired into
`STFT_DEBUG_MODE`'s CSV output right after the STFT: `ModelInference_run()` is called on
`stftSpectrogramDb`, timed the same way as the STFT itself.

Five conv stages (stage0 regular conv, stage1_dw/stage2_dw depthwise, stage1_pw/stage2_pw
pointwise 1x1) collapse the input `[1,129,7]` (freq x time) spectrogram to `T'=1` via three
stride-2 convs - exactly one model output per 64ms buffer, matching the real streaming
architecture (one LSTM step per buffer) rather than something needing reconciliation with a
longer clip.

### Build integration notes

- Unlike CMSIS-DSP, CMSIS-NN's `Source/*Functions` directories have no aggregator `.c` file,
  so their source directories were added directly to the Makefile's `SRC` (no glue-file trick
  needed, unlike `cmsis_dsp_glue/`).
- `audiomoth.ld` had a genuine off-by-`ORIGIN(FLASH)` (16384 bytes) bug: its flash-overflow
  `ASSERT` compared the absolute address `__etext` against a bare `LENGTH(FLASH)` instead of
  `ORIGIN(FLASH) + LENGTH(FLASH)`, making the build's safety check ~16KB too strict. Fixed to
  `ASSERT( (ORIGIN(FLASH) + LENGTH(FLASH)) >= (__etext + SIZEOF(.data)), ... )`.
- Explicitly adding `-DARM_MATH_DSP=1` to `CFLAGS` was tried as a hypothesis for the high
  inference cost and turned out to be a no-op: `CMSIS-NN/Include/arm_nn_math_types.h` already
  auto-defines `ARM_MATH_DSP` whenever the compiler's `__ARM_FEATURE_DSP == 1`, which GCC
  always sets for `-mcpu=cortex-m4`. Confirmed via `objdump` disassembly - identical SMLAD
  instruction counts and identical `.text` size with/without the define. The SIMD-accelerated
  kernels were already active from the start; this was a dead end, not a fix.

### Flash/RAM footprint

| Stage | Flash used (`.text`+`.data` / 245760) | Margin | Internal RAM (`.data`+`.bss`+`.heap`+`.stack` / 32768) | Margin |
|---|---|---|---|---|
| Full model integrated (first working build) | 234816 | ~10.7KB | 27400 | ~5.2KB |
| + per-stage timing instrumentation | 235456 | ~10.1KB | 27800 | ~4.9KB |
| + `arm_depthwise_conv_s8_opt` (kept) | 235296 | ~10.5KB | 28248 | ~4.4KB |
| + hand-written `runStage0Conv` (kept) | 236544 | ~9.0KB | 28280 | ~4.4KB |

Tight but fits at every step, verified by building and reading `arm-none-eabi-size` output
after each change rather than assumed - the model's own weight/param data is only ~10.5KB,
but CMSIS-NN kernel code (im2col, matmul, requantize) adds meaningfully more.

### Results: end-to-end timing (real hardware, 94-124 buffers per run)

| Model code path | STFT | Inference | Total | % of 64ms budget |
|---|---|---|---|---|
| Before depthwise optimization | 4486 µs | 43579 µs | 48065 µs | 75.1% |
| After `arm_depthwise_conv_s8_opt` | 4486 µs | 40466 µs | 44952 µs | 70.2% |
| **After hand-written `runStage0Conv` (kept)** | **4486 µs** | **32868 µs** | **37354 µs** | **58.4%** |

Model inference dominates the buffer's compute cost, but the combined pipeline fits with
margin to spare - ~41.6% of the 64ms budget free after both kept optimizations, up from
~24.9% at the first working build. Both wins came from *how* the existing CMSIS-NN kernels
are invoked, not from changing the model or its weights.

### Stage-by-stage profiling breakdown

Added a `ModelInferenceTiming_t` struct (`inc/model_inference.h`) filled in by
`ModelInference_run()`, wrapping each of the 5 conv stages + avgpool + LSTM step +
classifier in a `TIME_STAGE` macro, and extended the CSV with 8 columns
(`stage0_us,stage1_dw_us,stage1_pw_us,stage2_dw_us,stage2_pw_us,avgpool_us,lstm_us,classifier_us`).

| Stage | Before (µs) | % | After `arm_depthwise_conv_s8_opt` (µs) | % | After `runStage0Conv` (µs) | % |
|---|---|---|---|---|---|---|
| stage0 (regular conv, in_ch=1) | 16601 | 38.1% | 16632 | 41.1% | **9063** | 27.6% |
| stage1_dw (depthwise) | 9583 | 22.0% | **7141** | 17.6% | 7141 | 21.7% |
| stage1_pw (pointwise 1x1) | 6744 | 15.5% | 6745 | 16.7% | 6805 | 20.7% |
| stage2_dw (depthwise) | 4395 | 10.1% | **3449** | 8.5% | 3449 | 10.5% |
| stage2_pw (pointwise 1x1) | 4211 | 9.7% | 4211 | 10.4% | 4242 | 12.9% |
| avgpool | 244 | 0.6% | 244 | 0.6% | 244 | 0.7% |
| LSTM step | 702 | 1.6% | 702 | 1.7% | 672 | 2.0% |
| classifier | 0 | 0.0% | 0 | 0.0% | 0 | 0.0% |
| (unaccounted: quantize/transpose, not instrumented) | ~1099 | 2.5% | ~1342 | 3.3% | ~1252 | 3.8% |

The 5 conv/depthwise stages account for ~95% of inference time; LSTM and the classifier are
negligible (a 16-element dot product and a single LSTM step over a 16-wide hidden state are
just cheap).

### Depthwise convolution: `arm_depthwise_conv_s8` -> `arm_depthwise_conv_s8_opt`

Root cause (confirmed by reading the CMSIS-NN source, not assumed): `arm_depthwise_conv_s8`
only takes its SIMD-accelerated path (`depthwise_conv_s8_mult_4`) when the channel multiplier
is a multiple of 4. This model's depthwise stages are channel-preserving (`ch_mult=1`,
in_ch==out_ch), so every call fell through to `depthwise_conv_s8_generic` - a fully scalar,
non-SIMD fallback, confirmed as the least MAC-efficient stage measured (~1008 ns/MAC, worse
than stage0).

`arm_depthwise_conv_s8_opt` has a dedicated, DSP/SMLAD-accelerated path for exactly this
in_ch==out_ch case (its own doc comment: *"channel multiplier of 1"*), needs `output_ch % 4
== 0` to avoid its own scalar leftover loop (true here: 16 and 32 both divide evenly), and
uses the same weight array layout as the plain function - a drop-in swap, no re-export
needed. Only requirement: a small scratch buffer,
`in_ch * kernel_w * kernel_h * sizeof(int16_t)` (576 bytes for the larger of the two
stages), reused from the same buffer already shared with `runConv()`'s im2col scratch.

Result: stage1_dw -25.5% (9583 -> 7141 µs), stage2_dw -21.5% (4395 -> 3449 µs), total
inference -7.1% (43579 -> 40466 µs). Confirmed on real hardware, not simulated.

### Hand-written `runStage0Conv`: bypassing im2col for in_ch=1

With `arm_convolve_wrapper_s8` ruled out (below), stage0 (in_ch=1, kernel 3x3, stride 2,
pad 1) remained the largest single cost at ~41% of inference. Root cause, read directly from
`arm_convolve_s8.c`: with `kernel_ch=1`, every one of the 2340 (65*4 output positions x 9
kernel taps) im2col writes is a 1-byte `arm_memcpy_s8`/`arm_memset_s8` call - and on Cortex-M4
(no MVE), both are just wrappers around libc's generic `memcpy()`/`memset()`, real function
calls to move a single byte each. The arithmetic itself is tiny (37440 MACs) and was never
the bottleneck.

Fix: a hand-written `runStage0Conv()` that reads the 3x3 window directly from the input
tensor with no im2col buffer at all. Two things make this safe rather than an approximation:
`convParams.input_offset` is 0 for every stage in this model, so a padded tap contributes
`input_offset*weight = 0` either way - padding taps can just be skipped instead of memset;
and requantization goes through the exact same `arm_nn_requantize()` CMSIS-NN uses
internally. **Verified bit-exact** against `arm_convolve_s8` with a native x86 test (200
random trials, byte-identical output) before ever touching hardware - possible here because
this replacement is plain portable C, unlike the pointwise attempt below which depends on
ARM-only DSP intrinsics that don't compile on x86.

Result: stage0 -45% (16632 -> 9063 µs), total inference -18.6% (40466 -> 32868 µs), total
buffer -16.9% (44952 -> 37354 µs). The single biggest win of this whole investigation, and
it needed no CMSIS-NN "opt" variant to exist for this shape - none does.

### Rejected optimization: `arm_convolve_wrapper_s8` for the pointwise stages

Hypothesis: since `stage1_pw`/`stage2_pw` are 1x1/stride-1/pad-0 convolutions,
`arm_convolve_wrapper_s8` should route them to `arm_convolve_1x1_s8_fast`, which (under GCC)
skips im2col entirely and calls `arm_nn_mat_mult_nt_t_s8` - a 2-output-rows x
2-output-channels SMLAD-tiled kernel. Looked correct from reading the source (same weight
layout, drop-in signature change).

**Measured worse on real hardware**: stage1_pw +41% (6745 -> 9491 µs), stage2_pw +75%
(4211 -> 7355 µs), total inference +15% (worse than before the depthwise optimization).
Reverted back to the direct `arm_convolve_s8` call. Lesson: a kernel that looks faster from
its source (skips a whole processing step) is not automatically faster on this specific
core/compiler/shape combination - measure on hardware before keeping a change, the same
discipline that caught the `ARM_MATH_DSP` no-op above.

### Rejected optimization: hand-written pointwise conv (correct, but net negative)

After `runStage0Conv()` worked, tried the same im2col-skipping idea on `stage1_pw`/
`stage2_pw`, but keeping CMSIS-NN's own SIMD dot-product kernel this time instead of a naive
scalar loop (unlike stage0, these have up to 34K MACs, not trivial): call
`arm_s8_to_s16_unordered_with_offset()` directly on the input (a 1x1/stride-1/pad-0 im2col
"row" is already exactly the input's contiguous per-position channel vector - no gathering
needed) and feed it straight into `arm_nn_mat_mult_kernel_s8_s16()`, the same function
`arm_convolve_s8` already calls internally - just without the redundant memcpy in between.

Correctness was confirmed **on real hardware** first this time (not just x86), since these
two support functions depend on ARM DSP inline assembly (`SXTAB16`, `SMLAD`, `PKHBT`) that
won't compile or run on x86: a temporary on-device check ran both the new and old
(`arm_convolve_s8`) paths on every buffer and counted mismatched bytes into two extra CSV
columns (`pw1_mismatch`, `pw2_mismatch`) - confirmed 0 mismatches across ~149 buffers before
trusting the timing numbers.

Timing was mixed and, net, worse: stage1_pw did improve in isolation (6805 -> 6317 µs,
-7.2%), but stage2_pw got slower (4242 -> 4639 µs, +9.4%) and - unexpectedly, since it was
never touched in this change - so did **stage0** (9063 -> 10437 µs, +15.2%). Total inference
went from 32868 to 34027 µs, ~1.2ms *slower* overall, despite the stage1_pw win. Best
explanation: adding/removing functions shifts `.text` layout enough to change flash-prefetch
behaviour for unrelated code on this target - a real, measured side effect, not visible from
reading the source of any individual function. Reverted both stages back to `arm_convolve_s8`
(the same function used in the `runConv()` helper before this attempt), which restored
stage0 to 9063 µs and inference to 32868 µs, confirming the layout-sensitivity theory rather
than a fluke.

**Takeaway for this codebase**: a correctness-verified, individually-faster kernel swap can
still make the *overall* pipeline slower once linked into the real binary. The only way to
know is to measure the whole pipeline on real hardware after each change, not just the stage
being changed - this is the second time in this investigation a "should be faster from
reading the source" change measured worse (see `arm_convolve_wrapper_s8` above), and the
first time changing one stage measurably moved an unrelated stage's timing.

### Timer resolution caveat for reading these CSVs

`AudioMoth_getTimeMicroseconds()` does not read a true microsecond-resolution timer: it
rescales the Backup RTC (BURTC) tick count, clocked at 32768 Hz, into microseconds
(`microseconds = 1000000 * remainderTicks / 32768`). One tick is therefore ~30.5 µs, and the
function's output only ever advances in jumps of that size - it cannot report values in
between. This is why `classifier_us` (a 16-element dot product, genuinely a few microseconds
of real work) reads as 0 most of the time and ~30-31 µs occasionally (whichever side of a
tick boundary the start/end samples land on), and why every other per-stage column jitters
in ~30 µs steps between runs of the same buffer - real execution time is stable, the
apparent jitter is clock quantization, not measurement noise.

## Operational notes from this debug branch

- A custom `main()` that unconditionally bypasses the switch-position check also bypasses
  `AudioMoth_handleUSB()` - the device then never enumerates over USB and can't be reflashed via
  the AudioMoth Flash App, regardless of switch position. Fixed by checking
  `AudioMoth_getSwitchPosition() == AM_SWITCH_USB` once at start-up *and* on every loop iteration
  (~64ms), since this firmware can't rely on the normal firmware's sleep/wake/reset cycle to
  re-check the switch periodically.
- Debug firmware must not assume the SD card is present: `AudioMoth_enableFileSystem()` /
  `AudioMoth_openFile()` failures are retried in a loop with a visible LED signal, recoverable by
  inserting the card without re-flashing.
