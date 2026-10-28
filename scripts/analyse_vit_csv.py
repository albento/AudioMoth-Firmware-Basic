"""Summarises a VIT_TIMING.CSV written by the VIT_BENCHMARK firmware (48 MHz DWT cycles)."""
import sys

import numpy as np

MHZ = 48.0
BUDGET_MS = 64.0

with open(sys.argv[1]) as fh:
    lines = [l.strip() for l in fh if l.strip() and not l.startswith("#")]
names = lines[0].split(",")
data = np.array([[float(v) for v in l.split(",")] for l in lines[1:]])
col = {n: data[:, i] for i, n in enumerate(names)}

print(f"rows: {len(data)}  ({len(data) * 0.064:.1f} s of audio)")
gaps = np.diff(col["buffer_index"])
print(f"buffer_index gaps (missing rows): {int((gaps != 1).sum())}, vit_skipped: {int(col['vit_skipped'].sum())}, "
      f"dropped_records (last): {int(col['dropped_records'][-1])}, rows with tokens_done < 7: {int((col['tokens_done'] < 7).sum())}")
period = col["period_cycles"][1:] / MHZ / 1000
print(f"ISR period: median {np.median(period):.3f} ms, min {period.min():.3f}, max {period.max():.3f} (64.000 expected)")
print(f"clock check isr_cycles / isr_us: median {np.median(col['isr_cycles'] / col['isr_us']):.2f} MHz")

steady = (col["buffer_index"] >= 5) & (col["vit_skipped"] == 0) & (col["tokens_done"] == 7)
cyc = lambda names_: sum(col[n][steady] for n in names_)
isr = cyc(["isr_cycles"]) / MHZ / 1000
print(f"\nsteady state ({int(steady.sum())} rows, window full):")
print(f"  ISR median {np.median(isr):.2f} ms, p99 {np.percentile(isr, 99):.2f}, max {isr.max():.2f} "
      f"-> {100 * np.median(isr) / BUDGET_MS:.1f}% of 64 ms (max {100 * isr.max() / BUDGET_MS:.1f}%)")

groups = {
    "STFT + PCEN": ["stft_pcen_cycles"],
    "patch_embed (+ input quantization)": ["patch_embed_cycles"],
    "q/k/v + K/V write": ["l0_qkv", "l1_qkv"],
    "QK + ALiBi + max/diff": ["l0_qk", "l1_qk"],
    "softmax (table)": ["l0_softmax", "l1_softmax"],
    "attn.V": ["l0_av", "l1_av"],
    "out_proj": ["l0_outproj", "l1_outproj"],
    "FFN linear1 + ReLU": ["l0_ffn1", "l1_ffn1"],
    "FFN linear2": ["l0_ffn2", "l1_ffn2"],
    "LayerNorm x4 (+ residual)": ["l0_ln1", "l0_ln2", "l1_ln1", "l1_ln2"],
    "classifier": ["classifier_cycles"],
}
total = np.median(cyc(["isr_cycles"]))
accounted = 0.0
print(f"\n  {'stage':38s} {'ms':>7s} {'% ISR':>7s}  spread (cycles)")
for name, cols in groups.items():
    s = cyc(cols)
    accounted += np.median(s)
    print(f"  {name:38s} {np.median(s) / MHZ / 1000:7.2f} {100 * np.median(s) / total:6.1f}%  {int(s.max() - s.min())}")
print(f"  {'other (ISR glue, timer reads, queue)':38s} {(total - accounted) / MHZ / 1000:7.2f} {100 * (total - accounted) / total:6.1f}%")

logit = col["logit_milli"] / 1000.0
prob = 1 / (1 + np.exp(-logit))
print(f"\nlogits: median {np.median(logit):.2f}, p95 {np.percentile(logit, 95):.2f}, max {logit.max():.2f}; "
      f"P > 0.5: {int((prob > 0.5).sum())} buffers ({100 * (prob > 0.5).mean():.1f}%), P > 0.9: {int((prob > 0.9).sum())}")
above = np.concatenate([[False], prob > 0.5, [False]]).astype(int)
starts, ends = np.where(np.diff(above) == 1)[0], np.where(np.diff(above) == -1)[0]
t = col["buffer_index"] * 0.064
print("segments with P > 0.5 lasting >= 3 buffers: " +
      ", ".join(f"{t[a]:.1f}-{t[b - 1] + 0.064:.1f}s" for a, b in zip(starts, ends) if b - a >= 3))
