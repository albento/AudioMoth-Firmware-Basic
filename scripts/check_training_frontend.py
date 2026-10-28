"""Was a train_windowed.py PCEN checkpoint trained with or without the AGC-before-PCEN bug
(fixed in test-audiomoth-cnn 82b74a00)? Behavioural test: score the float model on the same
clips through both front ends and see which one it is calibrated for.

  raw -> PCEN        : real inference (firmware, predict_timeline.py), post-fix training
  AGC -> PCEN        : pre-fix train_windowed.py (_prepare_clip_waveform's unconditional
                       streaming_rms_normalize with its idx-seeded jitter, reproduced exactly)

Also prints how far the learned PCEN parameters moved from LearnablePCEN's init (proves PCEN
itself was trained). Same environment conventions as export_vit.py.
Usage: check_training_frontend.py CHECKPOINT [CHECKPOINT ...]"""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("RUN_CONFIG", str(Path(__file__).resolve().parent / "run_vit_pcen.yaml"))
os.environ.setdefault("OUTPUT_ROOT", tempfile.mkdtemp(prefix="check_frontend_"))
os.environ["NORMALIZATION_MODE"] = "pcen"
os.environ["WANDB_ENABLED"] = "false"
sys.path.insert(0, os.environ.get("TEST_AUDIOMOTH_CNN", "/home/abento/Documents/test-audiomoth-cnn"))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_curve

from settings import BIRD_SED_DIR, DEVICE
from dataset import streaming_rms_normalize
from student_model import (PCEN_ALPHA, PCEN_DELTA, PCEN_EMA_ALPHA, LogSpectrogram, build_model,
                           load_learned_pcen)
from train_windowed import LongClipDataset, list_long_clip_paths, load_long_frame_labels

SECONDS = 180.0


def p_at_r95(y, s):
    precision, recall, _ = precision_recall_curve(y, s)
    return float(precision[recall >= 0.95].max())


def pre_fix_agc(waveform: torch.Tensor, idx: int) -> torch.Tensor:
    """_prepare_clip_waveform's body before 82b74a00 (applied in every normalization mode)."""
    clip_rms = waveform.pow(2).mean().sqrt().item()
    jitter = float(torch.exp(torch.randn(1, generator=torch.Generator().manual_seed(idx)) * 0.75))
    return streaming_rms_normalize(waveform, initial_rms=max(clip_rms * jitter, 1.0))


def load_model(checkpoint: Path):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "patch_embed.weight" in state:
        model = build_model("vit", (16, 32, 64), vit_context_size=32,
                            vit_embed_dim=state["patch_embed.weight"].shape[0], vit_depth=2, vit_heads=4)
    else:
        hidden = state["temporal_context.weight_hh_l0"].shape[1]
        model = build_model("lstm", (16, 32, 64), lstm_hidden_size=hidden)
    model.load_state_dict(state, strict=True)
    return model.to(DEVICE).eval()


def main():
    for checkpoint in map(Path, sys.argv[1:]):
        print(f"\n=== {checkpoint.name}")
        spectrogram = LogSpectrogram().to(DEVICE).eval()
        load_learned_pcen(spectrogram, checkpoint)
        pcen = spectrogram.pcen
        learned = {"alpha": torch.sigmoid(pcen._raw_alpha), "ema_alpha": torch.sigmoid(pcen._raw_ema_alpha),
                   "delta": F.softplus(pcen._raw_delta), "r": F.softplus(pcen._raw_r)}
        init = {"alpha": PCEN_ALPHA, "ema_alpha": PCEN_EMA_ALPHA, "delta": PCEN_DELTA}
        for name, values in learned.items():
            v = values.detach().cpu().numpy()[3:]  # bins 0-2 are masked (no gradient)
            ref = f" (init {init[name]:.4g})" if name in init else ""
            print(f"  PCEN {name:9s} learned: median {np.median(v):.4g}, range [{v.min():.4g}, {v.max():.4g}]{ref}")
        model = load_model(checkpoint)
        for split, n_clips in (("validation", 4), ("test", 5)):
            paths = list_long_clip_paths(str(BIRD_SED_DIR), split)[:n_clips]
            dataset = LongClipDataset(paths, load_long_frame_labels(str(BIRD_SED_DIR), split))
            scores = {"raw -> PCEN": [], "AGC -> PCEN": []}
            labels = []
            for i in range(len(dataset)):
                waveform, frame_label, _ = dataset[i]
                n = min(int(SECONDS * 16000) // 1024, frame_label.shape[0])
                waveform = waveform[: n * 1024]
                for name, wave in (("raw -> PCEN", waveform), ("AGC -> PCEN", pre_fix_agc(waveform, i))):
                    with torch.no_grad():
                        spec = spectrogram(wave.unsqueeze(0).to(DEVICE))
                        if hasattr(model, "n_frames_out"):
                            model.n_frames_out = n
                        scores[name].append(model.frame_logits(spec)[0].cpu().numpy())
                labels.append(frame_label[:n].numpy() > 0.5)
            y = np.concatenate(labels)
            for name, s in scores.items():
                s = np.concatenate(s)
                p = 1 / (1 + np.exp(-s))
                print(f"  {split:10s} {name}: frame AP {average_precision_score(y, s):.4f}  "
                      f"P@R95 {p_at_r95(y, s):.4f}  mean P(neg) {p[~y].mean():.3f}  mean P(pos) {p[y].mean():.3f}")


if __name__ == "__main__":
    main()
