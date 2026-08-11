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
  supported FFT size (16 to 4096), costing otherwise-unreachable flash even though only 256 is
  used - `arm_rfft_fast_init_f32()` is a switch-case dispatcher over every size, and since it
  takes the length as a runtime argument (even though this call site always passes the same
  compile-time constant), the compiler can't prove which case executes, so `--gc-sections` can't
  strip the other sizes' tables. Fixed by calling the size-specific `arm_rfft_fast_init_256_f32()`
  directly instead (already existed in CMSIS-DSP, no dispatch) - confirmed via the linker's
  symbol table (`arm-none-eabi-nm --print-size --size-sort`) that every `twiddleCoef_*`/
  `armBitRevIndexTable*` table above 256 points (512/1024/2048/4096, each pair costing 2-16KB)
  disappeared from the link. Real measured saving: `.text` 235280 -> 158472 bytes, **-76808
  bytes (~75KB)** - bigger than the ~64KB first estimated, since some of the dispatcher's own
  supporting code also became unreachable. Flash margin after the model integration (see below)
  went from ~9.4KB to ~84.2KB. A compile-time `#error` guards the 256-only assumption in case
  `STFT_FFT_LENGTH` ever changes without updating this call to match.
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
| + `arm_rfft_fast_init_256_f32()` fix (kept) | 159568 | **~84.2KB** | 27712 | ~4.9KB |

Tight through every step above, verified by building and reading `arm-none-eabi-size` output
after each change rather than assumed - the model's own weight/param data is only ~10.5KB,
but CMSIS-NN kernel code (im2col, matmul, requantize) adds meaningfully more. The last row
(see the FFT init fix above) changes that picture entirely: ~75KB of flash that was never
actually reachable code came free just from calling the right init function, most of a whole
extra model's worth of budget freed up without touching the model itself.

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

### `lstm-hidden-size=64` comparison (same architecture otherwise)

Exploring whether the frame-level precision plateau (see "Model training" section below) is a
capacity problem rather than a data/normalization one: re-exported with `--lstm-hidden-size 64`
(channels unchanged, `16,32,64`) and re-measured with the same `STFT_DEBUG_MODE`/CSV setup, 123
buffers, real hardware.

| Stage | hidden=16 (µs) | hidden=64 (µs) | Δ |
|---|---|---|---|
| stage0 | 9063 | 9920 | +9.5% |
| stage1_dw | 7141 | 7142 | ~0% |
| stage1_pw | 6805 | 6793 | ~0% (clock quantization) |
| stage2_dw | 3449 | 3451 | ~0% |
| stage2_pw | 4242 | 4234 | ~0% (clock quantization) |
| avgpool | 244 | 239 | ~0% (clock quantization) |
| LSTM step | 672 | **3299** | **×4.9** |
| classifier | 0 | ~16 (0 or 30-31, see timer caveat above) | clock quantization |
| **Inference total** | 32868 | **36092** | **+9.8%** |
| Buffer total (STFT+inference) | 37354 (58.4% of 64ms) | **40877 (63.9% of 64ms)** | +5.5 points |

LSTM step scales ×4.9, somewhat better than the ×6.4 naive MAC-count prediction
(`input_size*hidden_size + hidden_size*hidden_size` per gate, ×4 gates: 5120 -> 32768 MACs) -
consistent with a fixed per-call overhead that doesn't grow with `hidden_size`. Every conv
stage's cost is unchanged (they depend on `channels`, not `lstm-hidden-size`), except stage0,
which moved +9.5% (9063 -> 9920 µs) despite having zero code-level dependency on hidden size -
the same `.text`-layout-shift-from-an-unrelated-binary-size-change effect already measured
above (see the rejected hand-written pointwise conv, where stage0 moved +15.2% from a change
that never touched it). The LSTM weight arrays growing (~29KB more flash - confirmed by a
build with dummy hidden=64 weights before the real training run even finished, `.text`
159928 -> 188600 bytes) is the most likely trigger here too.

Still comfortably inside the 64ms budget (63.9%, ~36% margin left) if `hidden=64` turns out to
help the frame-level metric enough to justify it.

## Device-vs-PC numerical parity (STFT, RMS normalization, and INT8 model bugs)

After the model ran end-to-end on-device (previous section), the next question was whether it
computed the *right* thing, not just something fast: `test-audiomoth-cnn/compare_device_output.py`
rebuilds the exact `QuantizedModel` `export_int8.py` exported (same checkpoint/calibration) and
replays the firmware's own captured spectrograms (`STFT_OUTPUT_MODE_BINARY`) through
`int8_sim.py`'s kernels, carrying LSTM state across buffers the same way `ModelInference_run()`
does, then compares its logit to the device's own logit buffer-by-buffer. A first run showed a
large, weakly-correlated mismatch (mean `|diff|` 0.474, correlation 0.317) - flat across the
recording (trend correlation 0.02, thirds ~0.45/0.51/0.46 each), ruling out an accumulating
LSTM-state bug and pointing at a systematic, per-buffer computation difference instead. Six real,
independent bugs were found and fixed chasing this down, in the order below; the last (LSTM
`effective_bias`) accounted for most of the remaining gap.

### Buffer-restarting STFT and raw int16 scale

`student_model.py`'s `LogSpectrogram` computed one continuous STFT over a whole training clip;
the firmware restarts frame/hop counting every 1024-sample buffer and never computes a frame
spanning two buffers (confirmed earlier, see "Data capture" above). Restructured
`LogSpectrogram.forward()` to do the same: reshape the waveform into buffers, run
`frames_per_buffer` short-time FFTs per buffer independently, concatenate. Also found training
loaded audio via `soundfile.read(dtype="float32")`, which normalizes samples to `[-1, 1]`, while
the firmware's STFT runs directly on raw ADC `int16` values - a constant ~90.3dB
(`20*log10(32768)`) scale mismatch on every log-magnitude value. Fixed by switching the deployed
training/calibration/inference pipeline to `sf.read(path, dtype="int16")` throughout.

Validated the restructured `LogSpectrogram` against a real device capture (audio + firmware-
computed spectrogram from the same recording): max diff 0.19dB, concentrated almost entirely in
the Nyquist bin (bin 128: mean 0.000726dB vs 0.000025dB for every other bin) - consistent with
CMSIS-DSP's `arm_rfft_fast_f32` and `torch.fft.rfft` computing that one bin very slightly
differently, not a bug in either implementation (the fixed-point log10 approximation itself is
deterministic bit manipulation and contributes exactly 0 given identical input).

### Streaming (causal) RMS normalization

The firmware had no RMS normalization at all; training normalized each whole clip by its own
RMS (a non-causal, whole-clip statistic the firmware cannot compute in real time from a stream of
64ms buffers). Added `dataset.py`'s `streaming_rms_normalize()`: a per-buffer EMA of mean-square
power (`RMS_EMA_ALPHA = 2/(RMS_EMA_BUFFERS+1)`, `RMS_EMA_BUFFERS ≈ 46.9` buffers ≈ 3s time
constant), gain `= target_rms_raw / sqrt(running_mean_sq + eps)` applied to the waveform before
`LogSpectrogram` - matched on-device by `main.c`'s `computeBufferGain()`/
`resetRMSNormalization()`, gain folded directly into the existing windowing multiply
(`stftFrame[i] = source[offset+i] * stftWindow[i] * bufferGain`).

Training seeds each clip's EMA with a randomized `initial_rms = clip_rms * jitter` so the model
sees plausible mid-stream RMS states, not just clips starting from silence. This `jitter` was
first drawn from `torch.randn(1)` on the *global* RNG state - meaning `export_int8.py`'s
`calibrate()` (which reuses the same dataset class to pick calibration clips) produced different
per-channel quantization scales on every separate process invocation, silently breaking
reproducibility between the export that generated the on-device C arrays and any later
re-verification. Fixed by seeding an idx-local `torch.Generator().manual_seed(idx)` for the
jitter draw; confirmed fixed via a double-call test showing identical `input`/`stage1_dw`/`cell`
scales across repeated invocations.

### Missing 150Hz high-pass filter on-device

`LogSpectrogram` masks the lowest bins (below ~150Hz) to zero before feeding the model; the
firmware's STFT never did this. Added `STFT_HIGHPASS_BIN = 3` masking to all three
`STFT_LOG_MODE` branches (`LOG10F`/`VLOG`/`LOG10_FIXED`) in `main.c`, matching
`LogSpectrogram`'s mask exactly.

### `runAvgPoolFreq()`: NCHW vs NHWC indexing bug

CMSIS-NN's conv/depthwise kernels produce and consume NHWC (channel-innermost) tensors,
confirmed by reading `cmsis_nn_dims={n,h,w,c}` and the kernel source's own indexing -
`model_inference.c`'s hand-written frequency-only average pool (`runAvgPoolFreq()`, needed
because no CMSIS-NN kernel pools one spatial axis only) indexed it as NCHW
(`input[(c*h+y)*w+x]`) instead. Fixed to `input[(y*w+x)*channels+c]`. Confirmed via a native
numeric test against `int8_sim.py`'s reference: max error 29/127 before the fix, 0 after.

### Depthwise conv weight export layout bug

CMSIS-NN's depthwise kernels expect weights as `[1, kH, kW, channels]` (kernel-position-major,
channel-minor); PyTorch's native depthwise weight layout is `[Cout, 1, kH, kW]`
(channel-major). `export_int8.py`'s `export_c()` wrote the raw PyTorch layout straight to the C
arrays. Fixed by detecting depthwise layers (`is_depthwise = groups == in_ch == out_ch`) and
permuting via `np.transpose(layer.weight[:, 0, :, :], (1, 2, 0))` before export. Confirmed via a
native test: 367/648 weight-affected outputs wrong before the fix, 0/648 after.

### The major bug: LSTM `effective_bias` NULL pointer

Even after all five fixes above, a temporary on-device verification check (comparing the real
`arm_nn_lstm_step_s8()` kernel's output against an `int8_sim.py`-equivalent hand reference,
exposed via extra CSV/binary columns, same pattern used earlier for the pointwise-conv
optimization check) showed a massive 22.66/32 mean mismatch per buffer, out of 305 buffers - the
LSTM step itself was wrong on real hardware, not just a downstream scale/layout issue.

Root cause, confirmed by reading `arm_nn_lstm_calculate_gate_s8_s16()`'s source directly (called
internally by `arm_nn_lstm_step_s8()`): it never reads `cmsis_nn_lstm_gate.bias` at all - only
`input_effective_bias` and `hidden_effective_bias`. `model_inference.c`'s `runLSTMStep()` set
both of those to `NULL` for all four gates (forget/input/cell/output), wiring only the unused
`bias` field. Because address 0 happens to be mapped to the Cortex-M vector table in flash (not
an unmapped, faulting region on this core), every gate's "bias" silently read vector-table bytes
instead of crashing - a quiet wrong-answer bug, not a hard fault, which is why it survived
undetected through every earlier build/flash/run cycle.

Fix, cross-checked against CMSIS-NN's own test generator (`Tests/UnitTest/lstm_settings.py`'s
`calc_effective_bias()`) to get the convention exactly right:
- `input_effective_bias = round(bias_float / (input_scale * weight_scale))` - the bias in the
  same pre-requantization accumulator domain as the input-side matmul (zero_point is 0 here, so
  no `row_sum(weights)*zero_point` term), the standard TFLite/CMSIS-NN bias quantization
  convention.
- `hidden_effective_bias = 0` for all gates (a zero array, `lstmZeroEffectiveBias[16]`) - the
  bias must be added exactly once, and it's already folded into the input-side pass.

Implemented in `export_int8.py`'s `_quantize_gate()` (computes and exports
`input_effective_bias` per gate as a new C array) and `model_inference.c`'s `runLSTMStep()`
(wires `input_effective_bias`/`hidden_effective_bias` instead of `NULL`; the old `bias_q3_12`
assignments were left in place, confirmed harmless/unused by the real kernel).

Validated in two stages before trusting it: first a native x86 test against the real
`arm_nn_lstm_step_s8()` kernel (not just a hand reference mirroring `int8_sim.py`'s math, which
had given an ambiguous first result) confirmed hidden state matched exactly and cell state was
off by only 1-2 units per element - consistent with the known sigmoid/tanh float-vs-LUT
approximation gap, not a remaining bug. Then on real hardware:

| | Before fix | After fix |
|---|---|---|
| LSTM step mismatches (out of 32 = 16 hidden + 16 cell) | 22.66/32 mean, 0/305 buffers exact | 7.38/32 mean, 0/294 buffers exact (residual = known LUT-vs-float noise) |
| Device vs PC logit mean `\|diff\|` | 0.474 | 0.0012 |
| Device vs PC logit correlation | 0.317 | 0.9999 |

The temporary verification scaffolding (`verifyLSTMStep()` and friends in `model_inference.c`,
the `lstmMismatches` timing field, the corresponding CSV/binary column) was removed once the fix
was confirmed; the permanent fix (`input_effective_bias`/`hidden_effective_bias` wiring,
`export_int8.py`'s `_quantize_gate()` change) stayed.

**Takeaway**: five independent, real bugs (STFT framing, scale, RMS normalization, high-pass
filter, avgpool indexing, weight export layout) each moved the numbers, but none of them alone
explained a mismatch this large and this flat across the recording - only reading the actual
CMSIS-NN kernel source (rather than trusting the struct field names `bias` vs `*_effective_bias`
to mean what they suggest) found the dominant cause. As elsewhere in this investigation, the
fix that mattered most came from checking against the library's own source and test generator,
not from guessing based on plausible-looking field names.

### Performance re-check after all six correctness fixes

All six fixes above change what gets computed (or which pointers a kernel reads), not how much
work is done, so a re-run of the same `STFT_OUTPUT_MODE_CSV` per-stage timing test (186 buffers,
real hardware) was expected to land close to the pre-fix numbers:

| | Before fixes | After fixes | Δ |
|---|---|---|---|
| STFT | 4486 µs | 4718 µs | +232 µs (+5.2%) |
| Inference | 32868 µs | 32688 µs | -180 µs (-0.6%) |
| **Total** | **37354 µs** | **37406 µs** | **+52 µs, still 58.4%** |

The small STFT increase is exactly accounted for by the two real additions to that stage: the
per-buffer RMS gain computation and the high-pass bin masking, neither of which existed at the
earlier measurement. Inference time is unchanged within timer-quantization noise (the LSTM
`effective_bias` fix only changes which pointers `arm_nn_lstm_step_s8()` reads, not how much
arithmetic it does), confirming the correctness fixes cost nothing in the real-time budget -
margin stays at ~41.6% of the 64ms buffer.

## Model training: two more real bugs, and results from the fixed pipeline

Real-world testing of `TRIGGERED_RECORDING_MODE` (previous checkpoint) surfaced two further
train/inference mismatches, both fixed in `test-audiomoth-cnn` (not this repo) and both now
folded into the retrained checkpoint below:

- **Buffer-restarted CNN backbone.** `TinyLSTMCNN.frame_logits()` (and `export_int8.py`'s
  `float_forward_with_traces()`/`int8_forward()`, independent reimplementations) ran the 5-stage
  conv backbone *continuously* over a whole training clip's spectrogram, giving every conv
  layer several buffers' worth of real cross-frame receptive field. `ModelInference_run()` on
  the device restarts the same backbone fresh every 64ms buffer, with zero cross-buffer
  receptive field - only the LSTM carries state across buffers. Confirmed structurally different
  (not quantization noise) via a known-positive eval clip: 0.6-0.95 confidence under the
  continuous evaluation, ~0.13-0.18 under a buffer-restarted replica of the real firmware
  computation. Fixed by making `frame_logits()` itself buffer-restart the conv stack
  (`student_model._per_buffer_pooled_features()`), and `export_int8.py`'s two independent
  reimplementations to match, backed by a new direct consistency check between them (previously
  nothing enforced they agreed).
- **RMS-normalization convergence gap.** The streaming RMS normalizer (`streaming_rms_normalize()`,
  same EMA both in training and in `computeBufferGain()`/`resetRMSNormalization()` on-device)
  seeds its running estimate from a fixed "assume nominal level" constant and adapts from
  there. Training clips (`CLIP_SECONDS=3.0` at the time, ~1 EMA time constant) seeded
  `initial_rms` from each clip's own true RMS - so the model only ever saw an estimate that was
  *already close to right* for a clip's whole (short) duration, never a real *mid-clip
  convergence* (estimate starts wrong, drifts toward correct partway through), which is exactly
  what happens on a real, longer recording session on-device. Confirmed on a real 82s field
  recording of speech: short 3s windows re-seeded with their own true RMS scored 0.70-0.77
  peak probability; the same moments scored via one continuous whole-file pass (fixed-seed EMA
  drifting for 82s) scored only 0.10-0.27 at the identical positions. Fixed by retraining on much
  longer clips (`CLIP_SECONDS=12.0`, ~4 EMA time constants - long enough for real mid-clip
  convergence) with multiple independently-leveled speech segments per positive clip (SNR
  widened from a fixed 5-15dB to -5-20dB, 1-4 segments instead of 1) for genuine relative-volume
  diversity beyond what RMS normalization + per-clip jitter alone provide - **and** a
  since-caught-and-fixed coupling bug where the EMA time constant itself was accidentally
  derived from `CLIP_SECONDS` (`dataset.py`), which would have silently invalidated the whole
  fix by stretching the training-time EMA to match the longer clips instead of staying pinned
  to the firmware's real, fixed ~3s constant.

### Evaluation results: `lstm_small_16` (no focal loss), buffer-restarted + longer/volume-diverse retrain

| | AP | Precision @ recall≥95% (threshold) | n |
|---|---|---|---|
| Clip-level | 99.6% | **99.3%** (threshold=0.9546) | 2000 |
| Frame-level | 91.4% | **47.5%** (threshold=0.0561) | 374000 |

**Going forward, judge model quality primarily by precision @ recall≥95%, not raw AP** (user
preference, this entry on) - AP averages precision across every recall level including ones far
below what a usable detector would ever operate at, while precision-at-high-recall reflects the
actual operating point ("catch ~everything, how many false positives does that cost") a real
deployment cares about. Note the frame-level and clip-level thresholds for the same recall
target are very different (0.056 vs 0.95, both as raw probabilities) - the firmware makes a
per-buffer (frame-level) decision in real time, never a per-clip aggregate one, so
`TRIGGER_LOGIT_THRESHOLD` (`src/main.c`) should be calibrated against the **frame-level** PR
curve, not the clip-level one, or it will end up far stricter than intended.

## Triggered recording: a new WAV file per detection event

A third top-level firmware mode, `TRIGGERED_RECORDING_MODE` (mutually exclusive with
`STFT_DEBUG_MODE`, `src/main.c`), continuously listens and runs the classifier every 64ms
buffer via the same shared STFT/RMS/high-pass pipeline (factored into `computeSpectrogram()`
so both debug modes reuse it unchanged). Whenever the logit crosses `TRIGGER_LOGIT_THRESHOLD`,
it opens a new WAV file (with `TRIGGER_PREROLL_SECONDS` ≈ 0.75s of pre-roll audio from a
small always-on ring buffer, so the true onset of the sound isn't clipped), keeps recording
while re-triggered (`TRIGGER_TAIL_SECONDS` = 5s countdown, reset on every re-trigger - a
sliding window, not a hard cap), and closes the file once the countdown lapses - one distinct
file per detection event, unlike the existing amplitude/frequency trigger feature in
`makeRecording()`, which only ever keeps one continuous file with compression-encoded gaps.

Reuses the existing WAV header/filename machinery (`wavHeader_t`, `setHeaderDetails()`,
`generateFolderAndFilename()`, already supporting a `T.WAV`-suffixed name via a bool arg) and
the established ISR-producer/main-loop-consumer ring-buffer pattern (`stftRecordQueue`'s
precedent), with its own dedicated external-SRAM regions (pre-roll ring, event-tagged audio
queue, sized/offset to not overlap the model's activation buffers - see
`model_inference.c`'s `ACTIVATION_BUFFER_BASE_OFFSET` comment for the full layout). A solid
(non-blinking) green LED stays on for the whole time a file is open, distinct from the
default firmware's brief red activity flashes.

A design-review pass (before writing any code) caught several real bugs that would otherwise
have shipped: an external-SRAM offset collision between the new trigger queue and the model's
activation buffers (both would have silently corrupted each other), a burst-enqueue race that
could leave the consumer with a `CONTINUE`/`END` record and no preceding `START` under SD
backlog (fixed by making a trigger onset's whole pre-roll flush all-or-nothing), and an
uninitialized-backup-RAM bug where a cold boot straight into this mode would skip the normal
firmware's `deploymentID` seeding, embedding electrically-arbitrary bytes in every filename
if `enableFilenameWithDeviceID` was set (fixed by factoring that seeding into
`initialiseDeploymentID()`, called from both places).

### ISR timing verification (real hardware)

The existing ~37ms/64ms STFT+inference figure never included any external-SRAM write cost -
the STFT-debug measurement explicitly excludes its own record-queue memcpy from the timed
region. This mode adds two new SRAM-write costs with no prior measurement: an always-on 2KB
pre-roll copy (every buffer) and a burst of up to 12 extra 2KB copies (only on trigger
onset, flushing the whole pre-roll ring into the queue in one ISR call) - measured with a
temporary instrumented build (`TRIGGER_TIMING_TEST`, since removed) logging per-buffer ISR
elapsed time to `TRIGGER_TIMING.TXT`:

| | ISR time | % of 64ms budget | Margin |
|---|---|---|---|
| Idle (no detection, 206 buffers) | 37812 µs | 59.1% | ~26.2ms |
| **Worst case (trigger-onset burst, `TRIGGER_LOGIT_THRESHOLD` temporarily forced to -10.0f to guarantee a trigger)** | **45624 µs** | **71.3%** | **~18.4ms (28.7%)** |

The burst itself costs ~7.8ms (45624 - 37812 µs) - consistent with external SRAM via EBI
being slower than flash (see "Memory footprint" above), for 12 extra 2KB copies (~650µs
each). Even the worst case leaves ~28.7% of the buffer's real-time budget free - the design
fits with comfortable margin, no optimization needed. Confirmed on real hardware, not
estimated - same discipline as every other timing claim in this document.

### Two real bugs found via first real recordings: wrong gain, and a missing ~16x sample scale

The first triggered recordings, tested on real hardware, came out noticeably quieter than
normal deployments made in similar conditions - two independent, compounding bugs, both in
how `runTriggeredRecording()` set up the microphone (copied from `runSTFTFeasibilityTest()`,
a test harness never meant to produce real listenable recordings):

1. **Hardcoded gain.** `AudioMoth_enableMicrophone(AM_NORMAL_GAIN_RANGE, AM_GAIN_MEDIUM, 4,
   16, 1)` ignored the device's actual configured gain entirely - `makeRecording()` uses
   `configSettings->gain` and `configSettings->enableLowGainRange` (set via the AudioMoth
   Config App), so any deployment configured above `medium` gain would always sound quieter
   from this mode regardless. Fixed by reading `configSettings->gain`/
   `enableLowGainRange` the same way `makeRecording()` does - `clockDivider`/
   `acquisitionCycles`/`oversampleRate` stay hardcoded at `4`/`16`/`1`, since those (unlike
   gain) do affect what the model expects as input and must stay tied to the raw 16kHz
   configuration it was trained on, not the user's configured recording quality.

2. **Missing ~16x (~24dB) sample scale - the bigger effect.** `makeRecording()` never writes
   raw ADC samples to a WAV file directly - every sample passes through
   `DigitalFilter_applyFilter()`, which folds in an additional gain
   (`DigitalFilter_setAdditionalGain()`, `sampleMultiplier = 16.0f / (oversampleRate *
   sampleRateDivider)`) as part of its own filter coefficient (confirmed by reading
   `digitalfilter.c`'s `applyHighPassFilter()`/`applyBandPassFilter()`, both of which
   multiply the incoming sample by this folded-in `gain` before any filtering happens, on
   every sample, filter type notwithstanding). With `oversampleRate=1` (this mode's raw,
   non-oversampled ADC reading - the same convention `STFT_DEBUG_MODE`'s BINARY capture and
   the model's whole training pipeline use), that multiplier is a constant `16.0f`: the
   AudioMoth ADC's raw one-shot reading at `oversampleRate=1` sits at a fixed fraction of
   the standard int16 full-scale convention that `oversampleRate=16` (the usual default)
   reaches without correction. `runTriggeredRecording()` samples directly via DMA and never
   calls `DigitalFilter_applyFilter()` at all, so this scaling was never applied - every
   triggered recording was ~16x (~24dB) quieter than it should have been, independent of
   the gain-setting bug above and much larger in effect.

   Fixed by computing the same `triggerSampleMultiplier` (`16.0f` here, plus the same
   `AudioMoth_hasInvertedOutput()`/external-microphone sign handling `makeRecording()` uses)
   once at startup, and applying it (with clamping to `±INT16_MAX`) only to the two copies
   of `source` that feed the pre-roll ring / trigger queue (`copyScaledAudioForWav()`) -
   **deliberately not** to what `computeSpectrogram()`/`ModelInference_run()` consume: the
   model's calibration (streaming RMS normalization target, INT8 quantization scales) is
   tied to the same raw, unscaled ADC convention it was trained and validated against (see
   the device-vs-PC parity section above) - scaling that input too would silently reintroduce
   a train/inference mismatch, not fix one.

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

## On-device PCEN (`STFT_NORMALIZATION_PCEN`)

Ported while the corresponding `NORMALIZATION_MODE=pcen` training run (learnable per-band
alpha/delta/r/ema_alpha - see `test-audiomoth-cnn/student_model.py`'s `LearnablePCEN`) was
still in progress, so the firmware mechanism could be built/size/build-tested ahead of the
real checkpoint rather than after it - `model/pcen.h`/`pcen.c` briefly shipped as a
placeholder (untrained init values replicated across every bin) for that reason, now replaced
(see below) by a real export.

**`model/pcen.h`/`pcen.c` now hold real learned values**, exported via `export_pcen.py` from
`distill/student_cnn_lstm_small_64_learned_pcen.pt` (50000x2 training clips, hidden=64,
pos-weight) - test-set results: clip AP 99.2%/P@R95 97.7%, frame AP 92.4%/P@R95 69.1%, the
best of every normalization scheme compared this session (EMA 58.9%, none 59.3%, fixed PCEN
65.1%, learned PCEN 69.1% - all frame-level P@R95, the primary metric). Learned per-band
values sanity-checked before export: no band saturated at alpha/ema_alpha's sigmoid bounds (0
or 1) or otherwise implausible (alpha in [0.75, 0.99], delta in [0.24, 3.13], r in [0.21,
0.82], ema_alpha in [0.0086, 0.077] - i.e. bands learned to react noticeably *faster* than the
~3s fixed-PCEN init in places, never degenerately fast/slow) - no sign of the per-band
overfitting flagged as a risk when learnable PCEN was first implemented. Regenerate again
(`export_pcen.py`'s own docstring has the exact command) whenever this checkpoint is retrained
- do not hand-edit, and do not mix pcen.h/pcen.c from one checkpoint with model.h/model.c from
another.

### Why PCEN needs its own numeric approximation, not just `stftFastLog10`

PCEN's compression curve, `(E/(M+eps)^alpha + delta)^r - delta^r`, uses two fractional
exponents (`alpha`, `r`) that are *learned per frequency bin*, not a single fixed exponent
the way `stftFastLog10` always computes base-10 log. `stftFastLog10`'s bit-trick (reinterpret
a float's IEEE-754 bits as a linear-in-mantissa approximation of `log2(x)`, then one Q16.16
fixed-point multiply to convert to `log10`) can't be reused as-is for an arbitrary exponent -
so `stftFastPow(x, a) = exp2(a * log2(x))` was added, reusing `stftFastLog2Q16_16()` exactly
(same bias constant) and adding its algebraic inverse, `stftFastExp2Q16_16()` (the same bit
trick run backwards: reconstruct a float's bits from a Q16.16 log2 value). This is the same
approximation quality already validated and deployed for `stftFastLog10` (~0.26dB / ~3% worst
case on one log10 call), not a new/unproven numerical scheme.

Validated end-to-end (not just the isolated `pow()` call) against exact float64 PCEN on a
real recording (`long_speech_test.wav`, 3749 frames): mean absolute error 0.052, p99 0.135,
against PCEN's own ~0-14 output range - both compounded `pow()` calls included, computed with
the same fixed-point bias/rounding the C code uses. Isolated `pow()` error alone (synthetic,
covering PCEN's realistic input ranges) is ~1-6% relative, worse in the worst case than the
end-to-end PCEN error above - the two `pow()` calls partially cancel rather than compound
worst-case-to-worst-case in practice.

### Integration

- `STFT_NORMALIZATION_MODE` gains a third value, `STFT_NORMALIZATION_PCEN` (`EMA`=0, `NONE`=1,
  `PCEN`=2) - still compile-time only, same pattern as `STFT_DEBUG_MODE`/
  `TRIGGERED_RECORDING_MODE`.
- `computeBufferGain()` returns `1.0f` in PCEN mode, same as `NONE` - PCEN's own adaptation
  happens *after* the FFT, on the per-bin magnitude, not as a waveform-domain gain before it
  (matching `student_model.py`: `NORMALIZATION_MODE=pcen` skips `streaming_rms_normalize()`
  entirely, same as `none`).
- `computeSpectrogram()` gains a new branch (ahead of, and independent from, the existing
  `STFT_LOG_MODE` dB-variant switch, which it bypasses entirely - PCEN doesn't take a log at
  all) that calls a new per-bin helper, `computePcenBin()`, which updates a **persistent
  per-bin** running state `pcenRunningM[STFT_NUMBER_OF_BINS]` (the PCEN equivalent of
  `stftRunningMeanSq`, but one value per frequency bin instead of one global scalar - reset
  once per recording from `model_pcen_initial_m`, never mid-recording, same convention as the
  EMA scheme's own reset).
- Masked (sub-150Hz) bins need no special-casing beyond the existing magnitude-zeroing: PCEN
  of an always-zero input is always exactly `delta^r - delta^r == 0`, regardless of the
  learned per-bin constants or `M`'s state - the same "fixed, magnitude-independent output"
  property every other `STFT_LOG_MODE` variant already relies on for those bins.

### Flash/RAM footprint (build-tested via `AudioMoth-Project`, both placeholder AND real `pcen.h`/`pcen.c` - identical, since array shapes don't depend on the values)

| | `.text` | `.bss` |
|---|---|---|
| `STFT_NORMALIZATION_EMA` (baseline, unchanged) | 188600 | 23008 |
| `STFT_NORMALIZATION_PCEN` | 192208 (+3608) | 23520 (+512) |

`.bss`'s `+512B` is exactly accounted for: `pcenRunningM[129]` (+516B) minus the now-compiled-out
`stftRunningMeanSq` (-4B). `.text`'s `+3608B` is dominated by `pcen.c`'s own baked-in constants
(6 arrays x 129 bins x 4B = 3096B) plus the new `stftFastPow`/`computePcenBin`/`computeSpectrogram`
branch code, minus the EMA-only code (`computeBufferGain`'s accumulate loop, `stftFastLog10`
calls) it replaces. `STFT_NORMALIZATION_EMA` (still the default, unchanged) built to the
byte-identical baseline both before and after this change - the new code is fully compiled out
of the default build, not merely dead-code-eliminated at link time. Confirmed identical between
the placeholder and real-weights builds too, as expected (same array sizes, only values differ).

No real hardware timing measurement yet (no SWD probe available - see project notes); a rough
op-count estimate (`computePcenBin` does ~2 `stftFastPow` calls plus a handful of multiply/add/
divide per bin, x129 bins x7 frames/buffer) suggests a modest increase over the existing
~10.9ms/buffer STFT measurement, still comfortably inside the 64ms budget - but this is an
estimate, not a measurement. Use the existing `STFT_DEBUG_MODE` CSV timing infrastructure
(`ModelInferenceTiming_t`, per-stage `elapsedMicroseconds`) to get a real number now that this
is flashable with real (non-placeholder) `pcen.h`/`pcen.c`.

### Still needed before this is actually deployable

1. ~~Regenerate `model/pcen.h`/`pcen.c` for real~~ - done (see above).
2. ~~Real hardware timing~~ - done, indirectly: "Triggered recording" below's real-hardware
   budget check (~59%/71% of 64ms) runs the full STFT+PCEN+model pipeline on real hardware,
   PCEN included (it's the compiled-in normalization for that build) - not a PCEN-isolated
   number, but confirms the whole pipeline (PCEN's own cost included) has real headroom.
3. ~~Device-vs-PC numerical parity check~~ - done: `compare_device_output.py --long-clips`
   (see "Post-PCEN-deployment findings" below) now rebuilds the exact `QuantizedModel`
   (channels/hidden_size/calibration-percentile all threaded through) and replays a real
   on-device BINARY capture against `int8_sim.py` for a PCEN/long-clip checkpoint. Result:
   device tracks the simulator closely (small, if slightly growing, residual - see below), but
   both drift substantially from the float32 reference over a long continuous capture - a
   separate, not-yet-fully-understood finding (see below), not the parity gap this item was
   originally about.

## ViT attention-layer compute-cost probe (`STFT_DEBUG_ATTENTION_TEST`)

A dummy-weight, single-layer ALiBi windowed-attention encoder layer (`attnRunLayer()`),
compiled into `STFT_DEBUG_MODE` alongside (not instead of) the real STFT/LSTM/PCEN pipeline,
purely to get a real hardware timing number for whether the trained ViT
(`student_cnn_vit_alibi_pcen.pt`, embed_dim=64/n_heads=4/context_size=32/depth=2) could
plausibly run in real time - see this session's own design discussion for the back-of-envelope
MAC-count estimate (~800k MAC/buffer, ~15-20ms) this is meant to replace with a measurement.
**Never touches `model_inference.c`/`model.h`/the real LSTM classifier** - the deployed
LSTM path keeps working exactly as before, in this mode and every other; this is purely an
additional, optional measurement, gated behind its own flag.

### Enabling it

1. Set both `STFT_DEBUG_MODE` and the new `STFT_DEBUG_ATTENTION_TEST` to `1` in `main.c`
   (`TRIGGERED_RECORDING_MODE` must be `0` - they're mutually exclusive, same as always).
2. Add `SoftmaxFunctions`/`arm_softmax_s8` to the build - **not present in
   `AudioMoth-Project/build/Makefile`'s default `SRC` list**, confirmed directly (a clean
   `STFT_DEBUG_ATTENTION_TEST=1` build fails at link time otherwise). Append
   `$(CMSIS_NN_PATH)/Source/SoftmaxFunctions` to the `SRC =` line (line 23 at the time of
   writing) - **anchor the edit to that exact line** (`sed -i '23s#$# ...#'` or equivalent),
   not a substring match on `SRC = `: `_CSRC`/`_SSRC` (the derived object-list lines just below
   `SRC`) both contain that same substring (`_CSRC = `, `_SSRC = `) and get silently corrupted
   by an unanchored replace, confirmed the hard way. Do **not** also add
   `CMSIS-DSP/Source/StatisticsFunctions` - see the "LayerNorm" note below.
3. Rebuild. Footprint is config-dependent (`ATTN_EMBED_DIM`, `ATTN_N_LAYERS`) and changed
   several times across this probe's own evolution (see Results below) - at the final
   configuration measured (embed_dim=32, depth=2, dummy weights in flash, layer 0's K/V in
   internal RAM/layer 1's in external SRAM): flash usage ~220.7KB/256KB (~41.5KB headroom),
   `.bss` +2KB over the non-attention-test baseline (just layer 0's K/V window - everything
   else in this probe now lives in flash or external SRAM). Confirmed clean at every step
   (real build, real link) before being written up.

### Design notes worth keeping in mind before reading real numbers

- **Genuinely windowed, not dense**: unfolds K/V into `context_size`-wide windows
  (`arm_nn_vec_mat_mult_t_s8`, not the dense `q @ k.T` a naive port would do) - matches the
  training-side fix (see `_AlibiSelfAttention`'s own docstring in `student_model.py`) that cut
  a >4x wall-clock training slowdown down to something tractable; a dense on-device version
  would be a dramatically less realistic (slower) measurement.
- **Runs `STFT_NUMBER_OF_FRAMES` (7) times per 64ms buffer**, not once - TinyViT's own
  attention operates at raw-STFT-frame resolution, only pooling down to buffer resolution at
  the classifier's output (confirmed empirically on the training side - see this session's own
  "context_size in frames, not buffers" clarification). Reading only a single call's cost
  here would understate the real per-buffer budget by 7x.
- **Memory placement evolved across this probe's own history, and turned out to matter more
  than any compute-side change tried** (see Results below for the numbers). Final state:
  weights are `const` arrays generated into `src/attn_dummy_weights.h`
  (`scripts/gen_attn_dummy_weights.py`), placed in flash by the linker, exactly like a real
  exported ViT's weights would be (like `model.h`'s own). K/V ring buffers are genuine runtime
  state (never `const`) - layer 0's fits internal RAM (2KB); layer 1's doesn't (internal RAM
  had no headroom left with layer 0's buffer already in it), so it falls back to external SRAM,
  same as every array here originally did before this was investigated. This history is kept
  in the Results section below because the *size* of each move (not just its direction) was
  itself the finding.
- **LayerNorm's mean/variance are a hand-written float loop, not CMSIS-DSP's own
  `arm_mean_f32`/`arm_var_f32`** - not a performance choice: `CMSIS-DSP/Source/StatisticsFunctions`
  ships both the individual per-function files and a `StatisticsFunctions.c` aggregator that
  `#include`s all of them, and this Makefile's wildcard-based `SRC` mechanism pulls in every
  `.c` file in a given directory - added as a whole directory, this produced "multiple
  definition" link errors for every statistics function, confirmed directly. `arm_sqrt_f32` is
  still used (inline in `arm_math.h`, no separate source file to conflict).
- **Dummy weights/ALiBi table/input are placeholder values** (a small deterministic pattern,
  not exported from the real checkpoint) - this measures compute cost, not accuracy; nothing
  here should be used to draw any conclusion about the model's real output.

### Results

All measured on real hardware (the user flashed and reported CSV output at each step - no SWD
probe available on this dev machine, see project notes). Budget throughout: 64ms (one buffer).

**1. Baseline, embed_dim=64 (matches `student_cnn_vit_alibi_pcen.pt`), single layer, no
optimization**: `attention_us` ~195-209ms - **>3x over budget**. Per-substage breakdown (added
right after this first number came back so catastrophically over): qkv 56243us, attn_core
24628us, out_proj 7202us, LayerNorm 580us x2, FFN 105470us (**~50% of the total, the single
largest stage**).

**2. embed_dim=32 (half), single layer, otherwise identical**: `attention_us` ~66-71ms -
**~1.05-1.11x over budget**. qkv 15075us (3.7x smaller, close to the ideal 4x from halving both
matmul dimensions), out_proj 2165us (3.3x smaller), FFN 28075us (3.76x smaller, still ~40% of
the total), LayerNorm ~275us x2 (~2.1x smaller, expected for an O(n) hand loop). **attn_core
only dropped to 20568us (1.2x, not the ~2x its own MAC count would predict)** - the one stage
that didn't scale down with embed_dim, and the reason for everything that follows.

**3. Idea #1 - batch all 4 heads' softmax into one `arm_softmax_s8` call** (`num_rows=4`
instead of 4x `num_rows=1`) instead of attn_core's fixed cost being softmax's own per-call
setup: attn_core 20568 → 20445us, **~0.5% - negligible**. Reverted per direct instruction after
this result came back.

**4. Idea #2 - hand-written Q.K^T/attn.V dot-product loop** (`attnVecMatMultS8()`), replacing
`arm_nn_vec_mat_mult_t_s8()` to eliminate its 15-argument stack-marshaled call overhead
(Cortex-M4/AAPCS only passes 4 args in registers): attn_core 20447 → 20126us, **~1.6% -
also small**. Read the full source of `arm_nn_vec_mat_mult_t_s8.c` before writing this and
found why: on this target (`ARM_MATH_DSP` defined, no MVE), the path actually compiled already
uses SMLAD-paired 16-bit MACs and processes 2 output rows/iteration - not a naive scalar loop.
A hand-written scalar replacement does *more* arithmetic instructions per element, not fewer;
what it saves (call overhead, an unconditional-but-zero `lhs_offset` adjustment) turned out to
be small next to that trade. Kept anyway (not reverted) - it's a real, if modest, net win and
does no harm.

**5. Move K/V ring buffer from external SRAM to internal RAM** (single layer; at
embed_dim=32/head_dim=8 the window is only 2KB, small enough to fit): attn_core 20126 → 17548us,
**~12.8% - by far the largest compute-cost-probe-only win up to this point**, and a real
finding: attn_core's row count (`ATTN_CONTEXT_SIZE=32`, the K/V window size) never changes with
embed_dim, unlike every other stage - external SRAM's per-access latency, not CPU instruction
count, was the actual fixed cost neither idea #1 nor #2 could touch. Total `attention_us`
66284 → 66033 (attn_core is only ~27% of the total, so the effect on the grand total is
smaller than on attn_core alone).

**6. Move dummy weights from runtime-filled external SRAM to `const` arrays in flash**
(`attn_dummy_weights.h`, generated by `scripts/gen_attn_dummy_weights.py` with the exact same
deterministic values, bit-for-bit - isolating storage location as the only variable): **by far
the single biggest lever in this entire series**. qkv 15076 → 5844us (2.58x), out_proj
2136 → 1114us (1.92x), FFN 28106 → 10681us (2.63x), attn_core unchanged (17548 → 17716us,
correctly - it doesn't read these weights, confirming the isolation was clean). **Total
`attention_us` 66284 → 36362us, a 45% reduction - one layer at embed_dim=32 is now ~43% *under*
budget (0.57x)**, down from >3x *over* at the very first (embed_dim=64) measurement.

**7. Real depth=2 measurement** (the actual checkpoint's depth - every result above was a
single layer, understating the real stacked cost). Internal RAM had no room for a second
2KB K/V window alongside layer 0's (confirmed directly: overflowed), so layer 0 keeps its K/V
in internal RAM and layer 1 falls back to external SRAM (see "Design notes" above) - making
layer 1 somewhat pessimistic relative to layer 0. Result: **`attention_us` ~75165-75805us
(avg 75485), ~1.18x over the 64ms budget (~18% over)**. Every stage but attn_core scaled
almost exactly 2x from the single-layer numbers (qkv 11703 vs 11688 expected, FFN 21363 vs
21362, out_proj 2212 vs 2228) - a good internal consistency check that layer-to-layer weight
sharing doesn't distort the timing. attn_core came in at 38147us vs 35432 expected from a
clean 2x, implying layer 1 alone cost ~20431us - closely matching step 5's own external-SRAM
K/V number (20126us), exactly as expected given layer 1's K/V wasn't moved to internal RAM.

**End-to-end summary**: from >3.15x over budget (embed_dim=64, 1 layer, no optimization) to
**1.18x over budget (embed_dim=32, real depth=2, every implementation-level optimization
applied)** - driven almost entirely by memory placement (flash for weights, internal RAM for
K/V where it fits), not by the compute-side rewrites (ideas #1/#2 together saved <2%).

**What's left is no longer a pure-implementation lever**: FFN is now the largest single stage
(21363us, ~28% of the depth=2 total) at its standard 4x expansion ratio
(`ATTN_FFN_DIM = 4 * ATTN_EMBED_DIM`). Halving that ratio (2x instead of 4x) would save
roughly half of FFN's cost (~10.7ms), which would bring the depth=2 total to **~64.8ms - just
~1.3% over budget**, i.e. very close to feasible. This is a genuine architecture change (a
narrower FFN expansion), not a firmware rewrite - it needs to be validated on the training side
first (does `student_cnn_vit_alibi_pcen`-style TinyViT hold onto its accuracy with
`ffn_ratio=2`?) before any more firmware time goes into it. Recommended next step for
`test-audiomoth-cnn`, not `AudioMoth-Firmware-Basic`.

## Post-PCEN-deployment findings (long-clip/TBPTT training, INT8 calibration, real-world mismatch)

Everything below happened after the PCEN deployment/ViT probe work above, mostly on the
`test-audiomoth-cnn` side, but with real consequences for this repo's own `model.h`/`model.c`/
`pcen.h`/`pcen.c`/`main.c`. See `GUIDE_REIMPLEMENTATION.md`'s §6/§14.5/§16 for the reference-doc
version of the two most important findings here.

### Highpass mask removed, then reinstated - a real cause found, not just reverted on a hunch

The `<150Hz` highpass mask (`STFT_HIGHPASS_BIN`, both `student_model.py`'s `LogSpectrogram` and
`main.c`'s `computeSpectrogram()`) was removed entirely on the reasoning that the 150Hz cutoff
was somewhat arbitrary. Real cost, found afterward on a trained checkpoint: threshold >0.95
needed to separate speech from background on a real recording, where <0.5 sufficed before.
Ablation (masking the same 3 bins post-hoc, same trained weights, no retraining) roughly
doubled the fraction of frames scoring confidently near-zero (32.8% -> 58.6%) - these bins
carry large, mostly non-informative energy (DC offset, wind/handling rumble) a small model
struggles to learn to discount. Reinstated in both `student_model.py` and `main.c` (a clean
`git revert` on the firmware side, since it was an isolated commit).

Real bug found and fixed while reinstating it on the firmware side: bin 0 (DC, packed
separately into `stftFFTOutput[0]` by `arm_rfft_fast_f32`) was hardcoded to the masked/epsilon
floor in all four `STFT_LOG_MODE`/PCEN branches instead of computing its real value - fixed to
use `fabsf(stftFFTOutput[0])` like the Nyquist bin already did.

**Verified after the fact - the checkpoint blamed for needing threshold >0.95 was not actually
missing this mask at training time.** PCEN's own learned per-band `alpha` for bins 0-2 turned
out to be bit-for-bit IDENTICAL across all three bins (`0.613351941108704`, to full float
precision) while every other bin's value is unique - the exact signature of a bin that received
zero real gradient the whole time (PCEN's own math makes a masked bin's output exactly
`delta^r-delta^r=0` for any params, so only AdamW's weight decay - identical for identically-
initialized, identically-treated params - can move it, deterministically the same for every
masked bin). This checkpoint was trained WITH the mask present (its cluster's cluster training
run on a separate machine, code state not tracked by this repo's own commit history) - the
>0.95 threshold symptom was actually a testing artifact (evaluating with a code version that
had briefly removed the mask, mismatched against what the checkpoint was actually trained on),
not evidence the checkpoint itself needed the mask added. The mask was still worth reinstating
as a forward-looking design choice - just not for the reason first assumed.

### LSTM h64 embedded in `TRIGGERED_RECORDING_MODE` - real INT8 export results

`model.h`/`model.c`/`pcen.h`/`pcen.c` now hold a real export of
`student_windowed_lstm_lstm_16_32_64_h64_pcen.pt` (`train_windowed.py`'s windowed/TBPTT
training, channels 16/32/64, LSTM hidden=64, PCEN) via `export_int8.py --long-clips`/
`export_pcen.py --long-clips` (new flags - `test-audiomoth-cnn` didn't have a way to export a
`train_windowed.py` checkpoint before this; `export_int8.py`/`export_pcen.py` only knew
`train_student.py`'s short-clip convention). `main.c` switched `STFT_DEBUG_MODE` 1->0,
`TRIGGERED_RECORDING_MODE` 0->1 - this is the first time this repo has shipped a real trained
model in the actual deployment mode, not just a debug/timing build.

First export attempt (default max-based calibration): real INT8 87.1% frame AP vs float32's
96.7% - see "Calibrating on a percentile, not the max" below for the real cause and fix (96.0%
after the fix). `TRIGGER_PROBABILITY_THRESHOLD` set to the precision-at-recall>=95% value from
that export's own PR curve (0.6747 initially, 0.6756 after a later re-export) - not a fixed
constant, re-measure whenever `model.h`/`pcen.h` are re-exported from a different checkpoint.

### Calibrating INT8 on a percentile, not the literal max - PCEN's heavy-tailed output

Root cause of the 87.1%-vs-96.7% gap above: PCEN's rare "attack overshoot" (output spikes when
its slow running-average `M` lags a sudden onset) sets the true `abs_max` ~100-300x the typical
activation range, in well under 0.1% of frames on real calibration data (median 22, p90 111,
p99 318, max 6704). Calibrating INT8's scale on that literal max (the only option before this)
wastes nearly all of int8's resolution on values that occur almost never, crushing the
genuinely informative range into a handful of int8 levels.

Fixed with a new `--calibration-percentile` flag on `export_int8.py` (default 100 = unchanged
literal-max behavior; PCEN checkpoints want e.g. 99.9). Real result on the h64 checkpoint: INT8
frame AP 87.1% -> 96.0% (float32 is 96.7%), precision-at-recall>=95% 62.1% -> 83.0%, for one
flag. Percentile calibration can't just track a running scalar like `max()` does - it needs a
real (random, seeded - `CALIBRATION_SAMPLE_SEED` - not the default global RNG, or two
calibrations of the same checkpoint silently produce different, non-reproducible scales, a real
instance of which briefly looked like a firmware bug during device-vs-PC comparison below)
sample of the whole calibration set's distribution, bounded in memory via a per-clip subsample
cap (`PER_CLIP_SAMPLE_CAP`) rather than accumulating every value verbatim.

### Device-vs-simulator comparison, extended for PCEN/long clips - real hardware capture

`compare_device_output.py` gained `--long-clips`/`--dataset-dir`/`--checkpoint`/
`--calibration-percentile` (mirroring `export_int8.py`'s own) so it can rebuild the exact
`QuantizedModel` a `--long-clips` export produced. Also fixed a real, pre-existing bug this
needed anyway: it never called `load_learned_pcen()` on its own freshly-built `LogSpectrogram`,
so every PCEN checkpoint was silently being rebuilt/calibrated against PCEN's untrained
literature-default init, not the checkpoint's own learned values - `export_int8.py` already had
this right, `compare_device_output.py` just hadn't been touched since before PCEN existed.

Real capture (`main.c` temporarily switched to `STFT_DEBUG_MODE=1`/
`STFT_OUTPUT_MODE=STFT_OUTPUT_MODE_BINARY`/`STFT_DEBUG_ATTENTION_TEST=0`,
`TRIGGERED_RECORDING_MODE=0` - mutually exclusive, and the attention probe's own extra
per-buffer cost risked dropped buffers/broken LSTM continuity for this specific test), 414
buffers, no gaps: device vs `int8_sim.py` mean |diff| 0.43, correlation 0.965, growing over the
recording (thirds: 0.13/0.44/0.71) - by itself, flagged by the tool's own heuristic as a
possible real accumulating bug. Cross-checked against the float32 reference on the same
buffers: **both device and `int8_sim.py` diverge from float32 by a similarly large, similarly
growing amount** (int8_sim vs float32: 2.2/3.6/6.5 by thirds; device vs float32: 2.2/3.8/6.0) -
much larger than the device-vs-simulator gap itself. Conclusion: the growth is dominated by an
apparently inherent property of int8-quantized recurrent LSTM state drifting from its float
reference over a long (~26s), never-reset sequence, shared by the device and the simulator
alike - not a device-specific firmware bug. The smaller residual device-vs-simulator growth is
real and not fully explained (plausibly the sigmoid/tanh float-vs-LUT approximation compounding
a little through the recurrence) but is proportionally minor next to the int8-vs-float
component.

### The most serious bug of this whole project: EMA gain applied before PCEN at training time, never at real inference

Found from a real-world observation, not a code review: the user reported the model performing
much better on synthetic LibriSpeech-based test clips than on real test files or a real
AudioMoth, and - a stronger signal - that the very first LSTM checkpoints (no PCEN, no windowed
training, none of this session's work) generalized to real AudioMoth recordings *better* than
the latest one despite lacking every optimization added since.

Root cause: `train_student.py::DistillationDataset` has always correctly gated
`streaming_rms_normalize()` (the EMA/AGC waveform-domain gain) behind `if NORMALIZATION_MODE ==
"ema":`, with its own documented reasoning (applying it before PCEN double-normalizes, and PCEN
needs the waveform's real dynamic range) - matching the real firmware, where
`computeBufferGain()` returns a fixed `1.0f` in PCEN mode, no waveform-domain gain at all.
`train_windowed.py::_prepare_clip_waveform` (used by `LongClipDataset`/`ParquetLongClipDataset`,
and therefore by every checkpoint in the architecture sweep - h16/h32/h64/vit_e32/e48/e64) had
no such gate at all - applied the gain unconditionally, an oversight from when PCEN support was
added to that script, not a deliberate choice.

Verified directly on a real recording (`test3.wav`) with the h64/PCEN checkpoint: real-inference
path (raw waveform -> PCEN, what `predict_timeline.py`/the firmware actually do) gives median
P(positive) 0.055, 9.5% of frames >0.5; the training-matched path (AGC-leveled -> PCEN, the bug)
gives median 0.292, 28.9% of frames >0.5 - correlation only 0.75 between the two. The model's
learned calibration is fit to a waveform distribution real inference never actually presents it
with.

Fixed by adding the same gate `_prepare_clip_waveform` should have had from the start. No
re-export can fix an already-trained checkpoint - **every existing PCEN checkpoint from
`train_windowed.py` needs retraining under the fix before being trusted on real audio**; this
was a training-time bug, invalidating this session's own earlier real-world-readiness claims
for the h64 checkpoint (the INT8/calibration/device-comparison work above remains valid as
*methodology*, just needs re-running against a correctly-retrained checkpoint).

### Loss calibration: `--pos-weight` and general output compression toward high values

Separate, not yet root-caused: models trained with `--pos-weight` (computed from the balanced-
regime synthetic dataset's own ~22.6% average occupancy, itself an artifact of deliberately
balancing regime representation per split rather than matching real-world prevalence) show
compressed output - speech confidently >0.95, but background sitting around ~0.5 instead of
near 0, leaving little room to widen the gap further. Leading hypothesis: `--pos-weight` biases
the loss enough, on a capacity-limited model, to shift the whole output distribution upward
rather than sharpening separation - to be tested by retraining without `--pos-weight`, and/or
with `--focal-gamma` instead, once the bug above is fixed (no point tuning this on top of a
known-mismatched training pipeline). Not yet resolved as of this entry.
