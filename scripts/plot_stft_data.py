#!/usr/bin/env python3
"""
Reads the binary STFT_DATA_*.BIN files produced by the STFT_DEBUG_MODE
firmware (see main.c, stftFileHeader_t / stftRecord_t) and plots the
audio waveform and spectrogram of a chosen record.

Usage:
    python plot_stft_data.py STFT_DATA_VLOG.BIN
    python plot_stft_data.py STFT_DATA_VLOG.BIN --record 12
    python plot_stft_data.py STFT_DATA_VLOG.BIN --export-wav out.wav
"""

import argparse
import wave

import numpy as np
import matplotlib.pyplot as plt

HEADER_DTYPE = np.dtype([
    ("sample_rate", "<u4"),
    ("audio_length", "<u4"),
    ("fft_length", "<u4"),
    ("hop_length", "<u4"),
    ("number_of_frames", "<u4"),
    ("number_of_bins", "<u4"),
])


def load(path):
    with open(path, "rb") as f:
        header = np.fromfile(f, dtype=HEADER_DTYPE, count=1)[0]

        record_dtype = np.dtype([
            ("buffer_index", "<u4"),
            ("elapsed_us", "<u4"),
            ("audio", "<i2", (int(header["audio_length"]),)),
            ("spectrogram", "<f4", (int(header["number_of_frames"]), int(header["number_of_bins"]))),
        ])

        records = np.fromfile(f, dtype=record_dtype)

    return header, records


def compute_reference_spectrogram(header, records):
    """STFT computed over the continuous, concatenated audio (same window/hop/FFT length and
    dB scaling as the firmware), i.e. without the buffer boundaries the on-device version has -
    used as a reference to spot edge effects at those boundaries."""

    fft_length = int(header["fft_length"])
    hop_length = int(header["hop_length"])
    epsilon = 1.0e-6

    audio = records["audio"].reshape(-1).astype(np.float32)

    window = np.hanning(fft_length).astype(np.float32)  # matches the firmware's 0.5 - 0.5*cos(2*pi*i/(N-1))

    number_of_frames = (audio.shape[0] - fft_length) // hop_length + 1
    frames = np.stack([audio[i * hop_length: i * hop_length + fft_length] * window for i in range(number_of_frames)])

    magnitude = np.abs(np.fft.rfft(frames, axis=1))

    return (20.0 * np.log10(magnitude + epsilon)).T  # shape (bins, frames)


def build_aligned_per_buffer_spectrogram(header, records, reference_frame_count):
    """Places each on-device frame at its true global time slot (using each record's own
    buffer_index, so gaps from dropped buffers show up too), leaving NaN wherever the
    continuous reference has an overlap frame that the per-buffer version never computed."""

    audio_length = int(header["audio_length"])
    hop_length = int(header["hop_length"])
    number_of_frames = int(header["number_of_frames"])
    number_of_bins = int(header["number_of_bins"])

    # audio_length must be a whole number of hops for the slots below to line up exactly
    assert audio_length % hop_length == 0
    slots_per_buffer = audio_length // hop_length

    aligned = np.full((number_of_bins, reference_frame_count), np.nan, dtype=np.float32)

    for record in records:
        base_slot = int(record["buffer_index"]) * slots_per_buffer

        for frame in range(number_of_frames):
            slot = base_slot + frame

            if slot < reference_frame_count:
                aligned[:, slot] = record["spectrogram"][frame]

    return aligned


def plot_all(header, records):
    sample_rate = int(header["sample_rate"])
    hop_length = int(header["hop_length"])
    audio_length = int(header["audio_length"])

    gaps = np.where(np.diff(records["buffer_index"]) != 1)[0]
    if len(gaps) > 0:
        print(f"Warning: {len(gaps)} gap(s) in buffer_index - some buffers were dropped (SD write fell behind the ISR)")

    audio = records["audio"].reshape(-1)
    reference_spectrogram = compute_reference_spectrogram(header, records)
    reference_frame_count = reference_spectrogram.shape[1]

    aligned_per_buffer = build_aligned_per_buffer_spectrogram(header, records, reference_frame_count)

    finite_values = np.concatenate((aligned_per_buffer[np.isfinite(aligned_per_buffer)], reference_spectrogram.ravel()))
    vmin, vmax = finite_values.min(), finite_values.max()

    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="0.7")  # grey = no on-device frame at this time slot (missing overlap frame)

    fig, (ax_wave, ax_per_buffer, ax_reference) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    time_axis = np.arange(audio.shape[0]) / sample_rate
    ax_wave.plot(time_axis, audio, linewidth=0.5)
    ax_wave.set_ylabel("Raw sample")
    ax_wave.set_title(f"{len(records)} buffers, {audio.shape[0] / sample_rate:.2f} s total")

    # Both images now share the exact same time grid, so they use the same extent and stay aligned
    extent = [0, reference_frame_count * hop_length / sample_rate, 0, sample_rate / 2]

    ax_per_buffer.imshow(aligned_per_buffer, origin="lower", aspect="auto", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    ax_per_buffer.set_ylabel("Frequency (Hz)")
    ax_per_buffer.set_title("Per-buffer STFT, time-aligned (grey = missing overlap frame at buffer boundary)")

    ax_reference.imshow(reference_spectrogram, origin="lower", aspect="auto", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    ax_reference.set_xlabel("Time (s)")
    ax_reference.set_ylabel("Frequency (Hz)")
    ax_reference.set_title("Reference STFT over the continuous audio (no buffer boundaries)")

    buffer_duration = audio_length / sample_rate
    for ax in (ax_per_buffer, ax_reference):
        for boundary in np.arange(buffer_duration, audio.shape[0] / sample_rate, buffer_duration):
            ax.axvline(boundary, color="cyan", linewidth=0.5, alpha=0.4)

    fig.tight_layout()
    plt.show()


def plot_record(header, record):
    sample_rate = int(header["sample_rate"])
    hop_length = int(header["hop_length"])

    fig, (ax_wave, ax_spec) = plt.subplots(2, 1, figsize=(9, 6), sharex=False)

    time_axis = np.arange(record["audio"].shape[0]) / sample_rate
    ax_wave.plot(time_axis, record["audio"])
    ax_wave.set_xlabel("Time (s)")
    ax_wave.set_ylabel("Raw sample")
    ax_wave.set_title(f"Buffer {record['buffer_index']} - STFT time {record['elapsed_us']} us")

    spectrogram = record["spectrogram"].T
    extent = [0, spectrogram.shape[1] * hop_length / sample_rate, 0, sample_rate / 2]
    ax_spec.imshow(spectrogram, origin="lower", aspect="auto", extent=extent, cmap="magma")
    ax_spec.set_xlabel("Time (s)")
    ax_spec.set_ylabel("Frequency (Hz)")

    fig.tight_layout()
    plt.show()


def export_wav(path, header, records):
    sample_rate = int(header["sample_rate"])
    audio = records["audio"].reshape(-1)

    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio.tobytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", help="Path to a STFT_DATA_*.BIN file")
    parser.add_argument("--record", type=int, default=0, help="Index of the record to plot (default: 0)")
    parser.add_argument("--all", action="store_true", help="Plot every record's spectrogram concatenated end-to-end, alongside a reference STFT computed over the continuous audio, to compare and spot buffer-boundary edge effects")
    parser.add_argument("--export-wav", metavar="OUT.WAV", help="Concatenate every record's audio into a WAV file instead of plotting")
    args = parser.parse_args()

    header, records = load(args.file)

    print(f"{len(records)} records, {header['number_of_frames']} frames x {header['number_of_bins']} bins, "
          f"{header['sample_rate']} Hz, {header['audio_length']} samples/buffer")

    if args.export_wav:
        export_wav(args.export_wav, header, records)
        print(f"Wrote {args.export_wav}")
    elif args.all:
        plot_all(header, records)
    else:
        plot_record(header, records[args.record])
