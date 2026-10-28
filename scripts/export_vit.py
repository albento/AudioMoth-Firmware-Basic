"""Exports a trained TinyViT (test-audiomoth-cnn, train_windowed.py --architecture vit, PCEN) to
INT8 C arrays for the AudioMoth ViT timing benchmark, plus the checkpoint's own learned PCEN.

Run with test-audiomoth-cnn's venv (its modules are imported from TEST_AUDIOMOTH_CNN). Never
writes into test-audiomoth-cnn or the dataset: OUTPUT_ROOT (which those modules mkdir into at
import time) defaults to a fresh temporary directory; run with PYTHONDONTWRITEBYTECODE=1 to also
keep __pycache__ out of that repo. Writes vit_model.h/.c, pcen.h/.c, export_report.json and
testvec/ under --out-dir - copy the four C files into model/ (and src/vit_inference.c is the
consumer):
    PYTHONDONTWRITEBYTECODE=1 ../test-audiomoth-cnn/.venv/bin/python scripts/export_vit.py \\
        --checkpoint .../student_windowed/student_windowed_vit_vit_e32_pcen.pt --out-dir /tmp/vit_e32

Steps:
  1. Load the checkpoint strictly into student_model.TinyViT (shapes must match exactly).
  2. n_heads/context_size are not stored in the state dict: check behaviourally (frame AP on
     validation clips) that the training defaults (heads=4, context=32) are the right ones.
  3. Re-implement TinyViT's forward explicitly (the exact computation the firmware streams,
     token by token) and assert it matches model.frame_logits() to float precision.
  4. Calibrate per-tensor symmetric INT8 scales on real training clips, quantize, write C.
  5. Write PCEN params (export_pcen.export_pcen_c, initial_m as compute_initial_m_long does).
  6. Write test vectors (real PCEN spectrograms + float reference logits) for the host check.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import tempfile

REPO = Path(os.environ.get("TEST_AUDIOMOTH_CNN", "/home/abento/Documents/test-audiomoth-cnn"))

os.environ.setdefault("RUN_CONFIG", str(Path(__file__).resolve().parent / "run_vit_pcen.yaml"))
os.environ.setdefault("OUTPUT_ROOT", tempfile.mkdtemp(prefix="export_vit_"))
os.environ["NORMALIZATION_MODE"] = "pcen"
os.environ["WANDB_ENABLED"] = "false"
sys.path.insert(0, str(REPO))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_curve

from settings import BIRD_SED_DIR, DEVICE, NORMALIZATION_MODE
from student_model import LogSpectrogram, build_model, load_learned_pcen
from train_windowed import LongClipDataset, list_long_clip_paths, load_long_frame_labels
from export_pcen import export_pcen_c

FRAMES_PER_BUFFER = 7
SCORE_DIFF_SCALE = 0.1  # nats per int8 step of (score - row max), see the qk_mult comment below


# ---------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------

def load_clips(split: str, n_clips: int, max_seconds: float, spectrogram: LogSpectrogram):
    """[(name, spec [F, T] float32 cpu, labels [N] float32)] - spec is LogSpectrogram's PCEN
    output on the raw int16-scale waveform, exactly what train_windowed.py feeds the model
    (NORMALIZATION_MODE=pcen: no waveform-domain gain, see _prepare_clip_waveform)."""
    dataset_dir = str(BIRD_SED_DIR)
    paths = list_long_clip_paths(dataset_dir, split)[:n_clips]
    labels = load_long_frame_labels(dataset_dir, split)
    dataset = LongClipDataset(paths, labels)
    max_buffers = int(max_seconds * 16000) // 1024
    clips = []
    for i in range(len(dataset)):
        waveform, frame_label, _ = dataset[i]
        n_buffers = min(waveform.shape[0] // 1024, max_buffers, frame_label.shape[0])
        waveform = waveform[: n_buffers * 1024]
        with torch.no_grad():
            spec = spectrogram(waveform.unsqueeze(0).to(DEVICE))[0].float().cpu()  # [F, T]
        assert spec.shape[1] == n_buffers * FRAMES_PER_BUFFER, spec.shape
        clips.append((paths[i].name, spec, frame_label[:n_buffers].float()))
    return clips


def precision_at_recall(labels: np.ndarray, scores: np.ndarray, target: float = 0.95) -> float:
    precision, recall, _ = precision_recall_curve(labels, scores)
    ok = recall >= target
    return float(precision[ok].max()) if ok.any() else 0.0


# ---------------------------------------------------------------------------------------------
# Explicit float forward (what the firmware streams), with optional activation recording
# ---------------------------------------------------------------------------------------------

@torch.no_grad()
def explicit_forward(model, spec: torch.Tensor, record=None) -> torch.Tensor:
    """spec [F, T] -> per-token logits [T]. Same computation as TinyViT._encode +
    classifier, written out step by step (eval mode: dropout is identity)."""
    x = spec.T.contiguous()  # [T, F]
    T = x.shape[0]
    D = model.patch_embed.out_features
    H = model.alibi_slopes.shape[0]
    hd = D // H
    C = model.context_size
    e = model.patch_embed(x)
    if record is not None:
        record("x", x)
        record("e0", e)
    w = torch.arange(C, device=x.device)
    diff = (C - 1 - w).float()
    alibi = -model.alibi_slopes.view(-1, 1).to(x.device) * diff.view(1, -1)  # [H, C]
    t = torch.arange(T, device=x.device).view(-1, 1)
    valid = (t - diff.view(1, -1)) >= 0  # [T, C]
    for l, layer in enumerate(model.layers):
        sa = layer.self_attn
        qkv = sa.qkv(e)
        q, k, v = qkv[:, :D], qkv[:, D:2 * D], qkv[:, 2 * D:]
        qh = q.view(T, H, hd).transpose(0, 1)
        kh = k.view(T, H, hd).transpose(0, 1)
        vh = v.view(T, H, hd).transpose(0, 1)
        kw = F.pad(kh, (0, 0, C - 1, 0)).unfold(1, C, 1).permute(0, 1, 3, 2)  # [H, T, C, hd]
        vw = F.pad(vh, (0, 0, C - 1, 0)).unfold(1, C, 1).permute(0, 1, 3, 2)
        scores = torch.einsum("htd,htwd->htw", qh, kw) / math.sqrt(hd) + alibi.view(H, 1, C)
        scores = scores.masked_fill(~valid.view(1, T, C), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        ao = torch.einsum("htw,htwd->htd", attn, vw).transpose(0, 1).reshape(T, D)
        o = sa.out_proj(ao)
        x1 = layer.norm1(e + o)
        h = torch.relu(layer.linear1(x1))
        f = layer.linear2(h)
        e = layer.norm2(x1 + f)
        if record is not None:
            record(f"l{l}.q", q)
            record(f"l{l}.k", k)
            record(f"l{l}.v", v)
            record(f"l{l}.scores", scores[:, valid])
            record(f"l{l}.ao", ao)
            record(f"l{l}.o", o)
            record(f"l{l}.x1", x1)
            record(f"l{l}.h", h)
            record(f"l{l}.f", f)
            record(f"l{l}.e", e)
    return model.classifier(e).squeeze(-1)


def buffer_logits(token_logits: torch.Tensor) -> torch.Tensor:
    return token_logits.view(-1, FRAMES_PER_BUFFER).mean(dim=1)


# ---------------------------------------------------------------------------------------------
# Quantization helpers (TFLite/CMSIS-NN conventions, symmetric, zero-point 0)
# ---------------------------------------------------------------------------------------------

def quantize_multiplier(real: float) -> tuple[int, int]:
    """real = multiplier * 2^(shift - 31), multiplier in [2^30, 2^31) - TFLite QuantizeMultiplier,
    the convention arm_nn_requantize() consumes (positive shift = left shift)."""
    if real <= 0.0:
        return 0, 0
    mantissa, exponent = math.frexp(real)
    q = int(round(mantissa * (1 << 31)))
    if q == (1 << 31):
        q //= 2
        exponent += 1
    if exponent < -31:
        return 0, 0
    return q, exponent


def softmax_params(input_scale: float, integer_bits: int = 5) -> tuple[int, int, int]:
    """TFLite PreprocessSoftmaxScaling(beta=1) + CalculateInputRadius - the (mult, shift,
    diff_min) arm_softmax_s8 expects."""
    real = min(input_scale * (1 << (31 - integer_bits)), float((1 << 31) - 1))
    mult, shift = quantize_multiplier(real)
    assert shift >= 0, "softmax input scale too small for arm_softmax_s8's left-shift convention"
    max_input_rescaled = ((1 << integer_bits) - 1) * (1 << (31 - integer_bits)) / (1 << shift)
    return mult, shift, -int(math.floor(max_input_rescaled))


def weight_quant(w: torch.Tensor) -> tuple[np.ndarray, float]:
    w = w.detach().cpu().double().numpy()
    scale = float(np.abs(w).max()) / 127.0
    return np.clip(np.round(w / scale), -127, 127).astype(np.int8), scale


def bias_quant(b: torch.Tensor, input_scale: float, weight_scale: float) -> np.ndarray:
    b = b.detach().cpu().double().numpy()
    return np.round(b / (input_scale * weight_scale)).astype(np.int64).clip(-2**31, 2**31 - 1).astype(np.int32)


# ---------------------------------------------------------------------------------------------
# C emission
# ---------------------------------------------------------------------------------------------

def c_array(ctype: str, name: str, values, fmt=str) -> tuple[str, str]:
    flat = np.asarray(values).flatten()
    body = ", ".join(fmt(v) for v in flat)
    return f"extern const {ctype} {name}[{flat.size}];", f"const {ctype} {name}[{flat.size}] = {{{body}}};"


def ffmt(v) -> str:
    return f"{float(v):.9e}f"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--context-size", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--calib-clips", type=int, default=8)
    parser.add_argument("--calib-seconds", type=float, default=180.0)
    parser.add_argument("--check-clips", type=int, default=4)
    parser.add_argument("--test-clips", type=int, default=5)
    parser.add_argument("--clip-seconds", type=float, default=180.0)
    parser.add_argument("--input-percentile", type=float, default=99.9)
    parser.add_argument("--act-percentile", type=float, default=99.99)
    parser.add_argument("--pcen-calibration-clips", type=int, default=20)
    args = parser.parse_args()

    assert NORMALIZATION_MODE == "pcen", NORMALIZATION_MODE
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint)
    report = {"checkpoint": str(checkpoint), "dataset_dir": str(BIRD_SED_DIR)}

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    embed_dim = state["patch_embed.weight"].shape[0]
    n_bins = state["patch_embed.weight"].shape[1]
    depth = 1 + max(int(k.split(".")[1]) for k in state if k.startswith("layers."))
    assert depth == args.depth, (depth, args.depth)

    spectrogram = LogSpectrogram().to(DEVICE).eval()
    load_learned_pcen(spectrogram, checkpoint)

    def make_model(n_heads: int):
        model = build_model("vit", (16, 32, 64), vit_context_size=args.context_size,
                            vit_embed_dim=embed_dim, vit_depth=depth, vit_heads=n_heads)
        model.load_state_dict(state, strict=True)
        return model.to(DEVICE).eval()

    # --- 2. heads check on validation clips ---------------------------------------------------
    check = load_clips("validation", args.check_clips, args.clip_seconds, spectrogram)
    heads_ap = {}
    for n_heads in (2, 4, 8):
        if embed_dim % n_heads:
            continue
        model = make_model(n_heads)
        scores, labels = [], []
        for _, spec, lab in check:
            model.n_frames_out = lab.shape[0]
            with torch.no_grad():
                scores.append(model.frame_logits(spec.unsqueeze(0).to(DEVICE))[0].cpu().numpy())
            labels.append(lab.numpy())
        heads_ap[n_heads] = float(average_precision_score(np.concatenate(labels) > 0.5, np.concatenate(scores)))
    report["validation_frame_ap_by_n_heads"] = heads_ap
    n_heads = max(heads_ap, key=heads_ap.get)
    print("validation frame AP by n_heads:", heads_ap, "-> using", n_heads)
    assert n_heads == 4, f"expected the training default n_heads=4, best was {n_heads}: {heads_ap}"
    model = make_model(n_heads)
    head_dim = embed_dim // n_heads
    C = args.context_size

    # --- 3. explicit forward == model.frame_logits -------------------------------------------
    max_err = 0.0
    for _, spec, lab in check:
        model.n_frames_out = lab.shape[0]
        with torch.no_grad():
            ref = model.frame_logits(spec.unsqueeze(0).to(DEVICE))[0]
        mine = buffer_logits(explicit_forward(model, spec.to(DEVICE)))
        max_err = max(max_err, float((ref - mine).abs().max()))
    report["explicit_forward_max_abs_err_vs_frame_logits"] = max_err
    print("explicit forward vs frame_logits, max |diff|:", max_err)
    assert max_err < 1e-3, max_err

    # --- 4. calibration ------------------------------------------------------------------------
    calib = load_clips("training", args.calib_clips, args.calib_seconds, spectrogram)
    samples: dict[str, list[np.ndarray]] = {}
    rng = np.random.default_rng(0)

    def record(name, tensor):
        a = tensor.detach().abs().flatten().float().cpu().numpy()
        if a.size > 200_000:
            a = rng.choice(a, 200_000, replace=False)
        samples.setdefault(name, []).append(a)

    for _, spec, _ in calib:
        explicit_forward(model, spec.to(DEVICE), record)

    def act_scale(name: str) -> float:
        p = args.input_percentile if name == "x" else args.act_percentile
        return float(np.percentile(np.concatenate(samples[name]), p)) / 127.0

    s = {name: act_scale(name) for name in samples}
    report["activation_scales"] = s

    header = ["#ifndef VIT_MODEL_H", "#define VIT_MODEL_H", "", "#include <stdint.h>", "",
              f"/* Generated by export_vit.py from {checkpoint.name} - do not hand-edit. */", "",
              f"#define VIT_N_BINS {n_bins}", f"#define VIT_EMBED_DIM {embed_dim}",
              f"#define VIT_N_HEADS {n_heads}", f"#define VIT_HEAD_DIM {head_dim}",
              f"#define VIT_CONTEXT_SIZE {C}", f"#define VIT_FFN_DIM {model.layers[0].linear1.out_features}",
              f"#define VIT_DEPTH {depth}", f'#define VIT_MODEL_TAG "{checkpoint.stem}"', "",
              "typedef struct {",
              "    const int8_t *weight; const int32_t *bias; int32_t multiplier; int32_t shift;",
              "} vit_fc_t;", "",
              "typedef struct {",
              "    float inputScaleA; float inputScaleB; const float *gamma; const float *beta; float outputInvScale;",
              "} vit_ln_t;", "",
              "typedef struct {",
              "    vit_fc_t q, k, v, outProj, ffn1, ffn2;",
              "    int32_t qkMultiplier, qkShift;",
              "    const int32_t *alibiAcc; /* [VIT_N_HEADS][2*VIT_CONTEXT_SIZE], see vit_inference.c */",
              "    int32_t softmaxMultiplier, softmaxShift, softmaxDiffMin;",
              "    int32_t avMultiplier, avShift;",
              "    vit_ln_t ln1, ln2;",
              "} vit_layer_t;", "",
              "extern const vit_fc_t vit_patch_embed;", "extern const vit_layer_t vit_layers[VIT_DEPTH];",
              "extern const int8_t vit_classifier_weight[VIT_EMBED_DIM];",
              f"#define VIT_INPUT_INV_SCALE {ffmt(1.0 / s['x'])}",
              f"#define VIT_CLASSIFIER_LOGIT_SCALE {ffmt(s[f'l{depth-1}.e'] * 0.0)}", ""]
    source = ['#include "vit_model.h"', ""]
    quant_report = {}

    def emit_fc(cname: str, weight: torch.Tensor, bias: torch.Tensor, in_scale: float, out_scale: float):
        wq, ws = weight_quant(weight)
        bq = bias_quant(bias, in_scale, ws)
        mult, shift = quantize_multiplier(in_scale * ws / out_scale)
        for ctype, name, vals in (("int8_t", f"{cname}_w", wq), ("int32_t", f"{cname}_b", bq)):
            _, c = c_array(ctype, name, vals)
            source.append("static " + c)
        quant_report[cname] = {"weight_scale": ws, "multiplier": mult, "shift": shift}
        return f"{{ {cname}_w, {cname}_b, {mult}, {shift} }}"

    def emit_ln(cname: str, norm, scale_a: float, scale_b: float, out_scale: float):
        for suffix, vals in (("g", norm.weight), ("b", norm.bias)):
            _, c = c_array("float", f"{cname}_{suffix}", vals.detach().cpu().numpy(), ffmt)
            source.append("static " + c)
        assert abs(norm.eps - 1e-5) < 1e-12, norm.eps
        return f"{{ {ffmt(scale_a)}, {ffmt(scale_b)}, {cname}_g, {cname}_b, {ffmt(1.0 / out_scale)} }}"

    patch = emit_fc("vit_pe", model.patch_embed.weight, model.patch_embed.bias, s["x"], s["e0"])
    layer_inits = []
    slopes = model.alibi_slopes.detach().cpu().double().numpy()
    for l, layer in enumerate(model.layers):
        sa = layer.self_attn
        D = embed_dim
        s_in = s["e0"] if l == 0 else s[f"l{l-1}.e"]
        fc_q = emit_fc(f"vit_l{l}_q", sa.qkv.weight[:D], sa.qkv.bias[:D], s_in, s[f"l{l}.q"])
        fc_k = emit_fc(f"vit_l{l}_k", sa.qkv.weight[D:2 * D], sa.qkv.bias[D:2 * D], s_in, s[f"l{l}.k"])
        fc_v = emit_fc(f"vit_l{l}_v", sa.qkv.weight[2 * D:], sa.qkv.bias[2 * D:], s_in, s[f"l{l}.v"])
        # Scores are NOT requantized on an absolute, calibrated scale: layer 0's real scores reach
        # ~1e4 (near-hard attention), far beyond what one int8 scale can resolve. Only the gap to
        # the row max matters to softmax, so the firmware requantizes (acc - acc_max) on a fixed
        # SCORE_DIFF_SCALE grid instead: int8 [-128, 0] <-> [-12.8, 0] nats below the max.
        qk_mult, qk_shift = quantize_multiplier(s[f"l{l}.q"] * s[f"l{l}.k"] / (math.sqrt(head_dim) * SCORE_DIFF_SCALE))
        # ALiBi in the QK accumulator domain: score = acc*s_q*s_k/sqrt(hd) + alibi
        acc_per_unit = math.sqrt(head_dim) / (s[f"l{l}.q"] * s[f"l{l}.k"])
        alibi_by_age = np.round(-slopes.reshape(-1, 1) * np.arange(C).reshape(1, -1) * acc_per_unit)
        alibi_by_age = alibi_by_age.clip(-2**31, 2**31 - 1).astype(np.int64)
        reversed_age = alibi_by_age[:, ::-1]  # A'[m] = alibi_by_age[C-1-m]
        doubled = np.concatenate([reversed_age, reversed_age], axis=1).astype(np.int32)  # [H, 2C]
        _, c = c_array("int32_t", f"vit_l{l}_alibi", doubled)
        source.append("static " + c)
        sm_mult, sm_shift, sm_diff_min = softmax_params(SCORE_DIFF_SCALE)
        av_mult, av_shift = quantize_multiplier(s[f"l{l}.v"] / 256.0 / s[f"l{l}.ao"])
        fc_o = emit_fc(f"vit_l{l}_o", sa.out_proj.weight, sa.out_proj.bias, s[f"l{l}.ao"], s[f"l{l}.o"])
        ln1 = emit_ln(f"vit_l{l}_ln1", layer.norm1, s_in, s[f"l{l}.o"], s[f"l{l}.x1"])
        fc_1 = emit_fc(f"vit_l{l}_f1", layer.linear1.weight, layer.linear1.bias, s[f"l{l}.x1"], s[f"l{l}.h"])
        fc_2 = emit_fc(f"vit_l{l}_f2", layer.linear2.weight, layer.linear2.bias, s[f"l{l}.h"], s[f"l{l}.f"])
        ln2 = emit_ln(f"vit_l{l}_ln2", layer.norm2, s[f"l{l}.x1"], s[f"l{l}.f"], s[f"l{l}.e"])
        quant_report[f"l{l}"] = {"qk": [qk_mult, qk_shift], "softmax": [sm_mult, sm_shift, sm_diff_min],
                                 "av": [av_mult, av_shift]}
        layer_inits.append(
            f"    {{ {fc_q}, {fc_k}, {fc_v}, {fc_o}, {fc_1}, {fc_2},\n"
            f"      {qk_mult}, {qk_shift}, vit_l{l}_alibi, {sm_mult}, {sm_shift}, {sm_diff_min},\n"
            f"      {av_mult}, {av_shift}, {ln1}, {ln2} }}")

    wc, wcs = weight_quant(model.classifier.weight[0])
    logit_scale = s[f"l{depth-1}.e"] * wcs
    header[header.index(next(h for h in header if h.startswith("#define VIT_CLASSIFIER_LOGIT_SCALE")))] = \
        f"#define VIT_CLASSIFIER_LOGIT_SCALE {ffmt(logit_scale)}"
    header.append(f"#define VIT_CLASSIFIER_BIAS {ffmt(float(model.classifier.bias[0]))}")
    # exp() of the requantized score gap (-d * SCORE_DIFF_SCALE nats, d = 0..128) in Q15, for
    # vit_inference.c's table softmax
    exp_table = np.round(np.exp(-SCORE_DIFF_SCALE * np.arange(129)) * 32768).astype(np.int64)
    header += ["#define VIT_SOFTMAX_TABLE_SIZE 129",
               "extern const uint16_t vit_softmax_exp_q15[VIT_SOFTMAX_TABLE_SIZE];"]
    _, c = c_array("uint16_t", "vit_softmax_exp_q15", exp_table)
    source.append(c)
    header += ["", "#endif"]
    _, c = c_array("int8_t", "vit_classifier_weight", wc)
    source.append(c)
    source.append(f"const vit_fc_t vit_patch_embed = {patch};")
    source.append("const vit_layer_t vit_layers[VIT_DEPTH] = {\n" + ",\n".join(layer_inits) + "\n};")
    (out_dir / "vit_model.h").write_text("\n".join(header) + "\n")
    (out_dir / "vit_model.c").write_text("\n".join(source) + "\n")
    report["quantization"] = quant_report
    print("wrote", out_dir / "vit_model.h", out_dir / "vit_model.c")

    # --- 5. PCEN (same as export_pcen.py --long-clips, which crashes on LongClipDataset's
    #        current 3-tuple __getitem__ - compute_initial_m_long's logic reproduced here) ------
    paths = list_long_clip_paths(str(BIRD_SED_DIR), "training")[: args.pcen_calibration_clips]
    dataset = LongClipDataset(paths, load_long_frame_labels(str(BIRD_SED_DIR), "training"))
    power_sum, power_count = None, 0
    with torch.no_grad():
        for i in range(len(dataset)):
            waveform, _, _ = dataset[i]
            power = spectrogram.compute_magnitude(waveform.unsqueeze(0).to(DEVICE)).pow(2)
            power_sum = power.sum(dim=(0, 2)) if power_sum is None else power_sum + power.sum(dim=(0, 2))
            power_count += power.shape[0] * power.shape[2]
    export_pcen_c(spectrogram, (power_sum / power_count).cpu().numpy(), out_dir)

    # --- 6. test vectors -----------------------------------------------------------------------
    test = load_clips("test", args.test_clips, args.clip_seconds, spectrogram)
    tv_dir = out_dir / "testvec"
    tv_dir.mkdir(exist_ok=True)
    for name, spec, lab in test:
        model.n_frames_out = lab.shape[0]
        with torch.no_grad():
            ref = model.frame_logits(spec.unsqueeze(0).to(DEVICE))[0].cpu().numpy()
        frames = spec.T.contiguous().numpy().astype(np.float32)  # [T, F], frame-major like stftSpectrogramDb
        with open(tv_dir / f"{Path(name).stem}.spec", "wb") as fh:
            np.array([frames.shape[0], frames.shape[1]], dtype=np.int32).tofile(fh)
            frames.tofile(fh)
        np.save(tv_dir / f"{Path(name).stem}.float_logits.npy", ref.astype(np.float32))
        np.save(tv_dir / f"{Path(name).stem}.labels.npy", lab.numpy())
    (out_dir / "export_report.json").write_text(json.dumps(report, indent=2))
    print("done:", out_dir)


if __name__ == "__main__":
    main()
