"""Compares the host-built firmware ViT (INT8, CMSIS-NN) logits against the float PyTorch model's
on the same PCEN spectrograms: per-clip agreement, then pooled frame AP / precision@recall>=95%."""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def precision_at_recall(labels, scores, target=0.95):
    precision, recall, _ = precision_recall_curve(labels, scores)
    return float(precision[recall >= target].max())


tv = Path(sys.argv[1])
all_f, all_c, all_y = [], [], []
for spec in sorted(tv.glob("*.spec")):
    stem = spec.stem
    f = np.load(tv / f"{stem}.float_logits.npy")
    c = np.fromfile(tv / f"{stem}.c_logits", dtype=np.float32)
    y = np.load(tv / f"{stem}.labels.npy") > 0.5
    assert f.shape == c.shape == y.shape, (f.shape, c.shape, y.shape)
    corr = np.corrcoef(f, c)[0, 1] if f.std() > 0 and c.std() > 0 else float("nan")
    pf, pc = 1 / (1 + np.exp(-f)), 1 / (1 + np.exp(-c))
    print(f"{stem:32s} buffers={f.size:5d} logit |diff| mean={np.abs(f - c).mean():.3f} "
          f"p99={np.percentile(np.abs(f - c), 99):.3f} corr={corr:.4f} prob |diff| mean={np.abs(pf - pc).mean():.4f} "
          f"pos={y.mean():.2f}")
    all_f.append(f); all_c.append(c); all_y.append(y)
f, c, y = map(np.concatenate, (all_f, all_c, all_y))
print(f"ALL: logit corr={np.corrcoef(f, c)[0, 1]:.4f}  mean|diff|={np.abs(f - c).mean():.3f}")
print(f"float32 PyTorch : frame AP={average_precision_score(y, f):.4f}  P@R95={precision_at_recall(y, f):.4f}")
print(f"INT8 C (device) : frame AP={average_precision_score(y, c):.4f}  P@R95={precision_at_recall(y, c):.4f}")
