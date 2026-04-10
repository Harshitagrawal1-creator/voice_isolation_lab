"""
infer.py — run from ~/Desktop/voice_isolation_desktop/

    # Run on a specific mixture from your dataset
    .venv/bin/python3 infer.py --input data/processed/test/mixtures/test_0000.wav

    # Run and also compare against ground truth sources
    .venv/bin/python3 infer.py --input data/processed/test/mixtures/test_0000.wav --ground_truth

    # Specify a different checkpoint
    .venv/bin/python3 infer.py --input my_audio.wav --checkpoint models/eend_ss_best.pth

Outputs (saved to outputs/demo/):
    speaker0.wav          — separated voice 0
    speaker1.wav          — separated voice 1
    speaker2.wav          — separated voice 2 (if 3 speakers detected)
    report.txt            — speaker count, SI-SDR score, DER, timestamps
    diarization.png       — timeline plot showing who spoke when
        spk 1, spk 2, spk 3 are output and ref 1, ref 2, ref 3 are ground truth
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torchaudio
import torchaudio.functional as AF
import matplotlib
matplotlib.use("Agg")   # no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.models.eend_ss import EENDSS
from src.training.config import get_config


# ── helpers ───────────────────────────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        print("using MPS")
        return torch.device("mps")
    if torch.cuda.is_available():
        print("using CUDA")
        return torch.device("cuda")
    print("using CPU")
    return torch.device("cpu")


def load_audio(path: Path, target_sr: int = 16000) -> torch.Tensor:
    """Load audio, resample if needed, convert to mono. Returns [1, T]."""
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = AF.resample(wav, sr, target_sr)
    return wav


def si_sdr(estimate: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    """Compute SI-SDR in dB between two numpy arrays."""
    estimate = estimate - estimate.mean()
    target   = target   - target.mean()
    dot      = np.dot(estimate, target)
    tgt_energy = np.dot(target, target) + eps
    s_tgt    = (dot / tgt_energy) * target
    e_noise  = estimate - s_tgt
    return float(10 * np.log10(
        (np.dot(s_tgt, s_tgt) + eps) / (np.dot(e_noise, e_noise) + eps)
    ))


def diarization_error_rate(pred: np.ndarray, ref: np.ndarray,
                            collar: int = 25) -> dict:
    """
    Compute DER components from binary frame-level arrays.
    pred, ref: [T_frames, C] binary arrays
    collar: number of frames to ignore around speech boundaries
    Returns dict with FA, MISS, CONF, DER (all as fractions 0-1)
    """
    T, C = ref.shape
    total_ref = ref.sum()
    if total_ref == 0:
        return {"FA": 0.0, "MISS": 0.0, "CONF": 0.0, "DER": 0.0}

    # match predicted speaker columns to reference columns (greedy)
    from itertools import permutations
    best_perm  = list(range(C))
    best_score = -1
    for perm in permutations(range(min(pred.shape[1], C))):
        score = sum((pred[:, perm[i]] * ref[:, i]).sum() for i in range(len(perm)))
        if score > best_score:
            best_score = score
            best_perm  = list(perm)

    pred_aligned = pred[:, best_perm[:C]] if pred.shape[1] >= C else pred

    # compute components
    fa   = (pred_aligned * (1 - ref)).sum() / total_ref
    miss = ((1 - pred_aligned) * ref).sum()  / total_ref
    # confusion: both active but different speaker
    conf = 0.0   # simplified (full CONF needs speaker tracking)
    der  = float(fa + miss + conf)

    return {"FA": float(fa), "MISS": float(miss), "CONF": conf,
            "DER": min(der, 1.0)}


# ── diarization plot ──────────────────────────────────────────────────────────

def plot_diarization(diar_binary: np.ndarray, duration: float,
                     num_speakers: int, out_path: Path,
                     ref_labels: np.ndarray | None = None):
    """
    Draw speaker activity timeline.
    diar_binary: [T_frames, C] binary array
    duration:    audio duration in seconds
    ref_labels:  ground truth [T_frames, C] if available (drawn below prediction)
    """
    COLORS = ["#534AB7", "#D85A30", "#1D9E75"]
    T_frames = diar_binary.shape[0]
    times = np.linspace(0, duration, T_frames)

    n_rows = num_speakers * (2 if ref_labels is not None else 1)
    fig_h  = max(3, n_rows * 0.9 + 1.5)
    fig, ax = plt.subplots(figsize=(14, fig_h))

    row = 0
    legend_patches = []

    for spk in range(num_speakers):
        color  = COLORS[spk % len(COLORS)]
        y_pred = row

        # predicted activity
        activity = diar_binary[:, spk]
        starts, ends = [], []
        in_seg = False
        for t_idx, val in enumerate(activity):
            if val and not in_seg:
                starts.append(times[t_idx])
                in_seg = True
            elif not val and in_seg:
                ends.append(times[t_idx])
                in_seg = False
        if in_seg:
            ends.append(duration)

        for s, e in zip(starts, ends):
            ax.barh(y_pred, e - s, left=s, height=0.7,
                    color=color, alpha=0.85, edgecolor="none")

        ax.text(-0.5, y_pred, f"Spk {spk}", va="center", ha="right",
                fontsize=10, color=color, fontweight="bold")
        legend_patches.append(mpatches.Patch(color=color, label=f"Speaker {spk}"))
        row += 1

        # ground truth (if provided)
        if ref_labels is not None:
            ref_act = ref_labels[:, spk] if spk < ref_labels.shape[1] else np.zeros(T_frames)
            for t_idx, val in enumerate(ref_act):
                pass  # recompute segments
            ref_starts, ref_ends = [], []
            in_seg = False
            for t_idx, val in enumerate(ref_act):
                if val and not in_seg:
                    ref_starts.append(times[t_idx])
                    in_seg = True
                elif not val and in_seg:
                    ref_ends.append(times[t_idx])
                    in_seg = False
            if in_seg:
                ref_ends.append(duration)

            for s, e in zip(ref_starts, ref_ends):
                ax.barh(row, e - s, left=s, height=0.7,
                        color=color, alpha=0.35, edgecolor=color, linewidth=0.5)
            ax.text(-0.5, row, f"Ref {spk}", va="center", ha="right",
                    fontsize=9, color=color)
            row += 1

    ax.set_xlim(0, duration)
    ax.set_ylim(-0.5, row - 0.3)
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_yticks([])
    ax.set_title(f"Speaker Diarization — {num_speakers} speakers detected", fontsize=13)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    if ref_labels is not None:
        solid = mpatches.Patch(color="gray", alpha=0.85, label="Predicted")
        faded = mpatches.Patch(color="gray", alpha=0.35, label="Ground truth")
        ax.legend(handles=[solid, faded], loc="upper right", fontsize=9)
    else:
        ax.legend(handles=legend_patches, loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ── main inference ────────────────────────────────────────────────────────────

def run_inference(args):
    device = get_device()
    cfg    = get_config()
    print(f"\nDevice: {device}")

    # ── load model ─────────────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    model = EENDSS(
        N=cfg.N, L=cfg.L, B=cfg.B, H=cfg.H, P=cfg.P,
        X=cfg.X, R=cfg.R, C=cfg.C,
        d_model=cfg.d_model, n_heads=cfg.n_heads,
        n_layers=cfg.n_layers, d_ff=cfg.d_ff,
        dropout=cfg.dropout, subsample=cfg.subsample,
    )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # remove pos_enc.pe from checkpoint — it is a fixed sine buffer (not learned)
    # and its size changes with max_len. We let the model recompute it fresh.
    state = ckpt["model_state"]
    state = {k: v for k, v in state.items() if 'pos_enc.pe' not in k}
    missing, unexpected = model.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if 'pos_enc.pe' not in k]
    if real_missing:
        print(f'WARNING: missing keys: {real_missing}')
    model = model.to(device)
    model.eval()

    trained_epoch = ckpt.get("epoch", "?")
    trained_loss  = ckpt.get("val_loss", float("nan"))
    print(f"Loaded checkpoint: epoch {trained_epoch}, val_loss {trained_loss:.4f}")

    # ── load input audio ───────────────────────────────────────────────────
    input_path = Path(args.input)
    assert input_path.exists(), f"Input file not found: {input_path}"

    mixture = load_audio(input_path)
    duration = mixture.shape[-1] / 16000
    print(f"\nInput: {input_path.name}")
    print(f"  Duration : {duration:.1f}s")
    print(f"  Samples  : {mixture.shape[-1]:,}")

    # ── run model ──────────────────────────────────────────────────────────
    print("\nRunning EEND-SS inference...")
    mixture_in = mixture.unsqueeze(0).to(device)   # [1, 1, T]

    result = model.inference(mixture_in, threshold=0.5)

    separated   = result["separated"]          # [1, C_hat, T]
    diar_probs  = result["diar_probs"]         # [1, T_sub, C]
    diar_binary = result["diar_binary"]        # [1, T_sub, C]
    num_speakers = result["num_speakers"]
    exist_probs  = result["exist_probs"]

    print(f"  Detected speakers : {num_speakers}")
    print(f"  Exist probs       : {[f'{v:.3f}' for v in exist_probs.squeeze().tolist()]}")

    # ── output directory ───────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving outputs to: {out_dir}/")

    # ── save separated audio ───────────────────────────────────────────────
    SR = 16000
    separated_np = separated.squeeze(0).cpu().numpy()   # [C_hat, T]

    saved_wavs = []
    for i in range(num_speakers):
        wav_tensor = torch.from_numpy(separated_np[i]).unsqueeze(0)
        # normalise to prevent clipping
        peak = wav_tensor.abs().max()
        if peak > 0:
            wav_tensor = wav_tensor / peak * 0.9
        out_path = out_dir / f"speaker{i}.wav"
        torchaudio.save(str(out_path), wav_tensor, SR)
        saved_wavs.append(out_path)
        print(f"  Saved: speaker{i}.wav")

    # ── load ground truth if requested ────────────────────────────────────
    ref_labels   = None
    gt_sources   = []
    manifest_dir = input_path.parent.parent   # .../test/
    mixture_id   = input_path.stem            # e.g. "test_0000"
    manifest_path = manifest_dir / f"{manifest_dir.name}_manifest.json"

    if args.ground_truth and manifest_path.exists():
        print("\nLoading ground truth from manifest...")
        with open(manifest_path) as f:
            manifest = json.load(f)

        sample = next((s for s in manifest if s["mixture_id"] == mixture_id), None)
        if sample:
            # load label
            label_path = input_path.parent / Path(sample["label_path"]).name
            if label_path.exists():
                ref_labels = np.load(str(label_path))
                print(f"  Ground truth labels: {ref_labels.shape}")

            # load clean sources for SI-SDR
            for sp in sample["source_paths"]:
                sp_path = input_path.parent / Path(sp).name
                if sp_path.exists():
                    src_wav, _ = torchaudio.load(str(sp_path))
                    if src_wav.shape[0] > 1:
                        src_wav = src_wav.mean(0, keepdim=True)
                    gt_sources.append(src_wav.squeeze(0).numpy())
        else:
            print(f"  mixture_id {mixture_id} not found in manifest")

    # ── diarization plot ───────────────────────────────────────────────────
    diar_np = diar_binary.squeeze(0).cpu().numpy()   # [T_sub, C]
    # resize ref_labels to match T_sub if available
    ref_labels_resized = None
    if ref_labels is not None:
        T_sub = diar_np.shape[0]
        from scipy.ndimage import zoom
        scale = T_sub / ref_labels.shape[0]
        ref_labels_resized = (zoom(ref_labels.astype(float),
                                   (scale, 1)) > 0.5).astype(float)

    plot_path = out_dir / "diarization.png"
    plot_diarization(diar_np, duration, num_speakers, plot_path,
                     ref_labels=ref_labels_resized)

    # ── compute metrics ────────────────────────────────────────────────────
    metrics = {}

    # SI-SDR (if ground truth available)
    if gt_sources:
        from itertools import permutations
        best_sisnr = -999
        T_min = min(separated_np.shape[1],
                    min(s.shape[0] for s in gt_sources))
        n = min(num_speakers, len(gt_sources))

        for perm in permutations(range(n)):
            scores = [si_sdr(separated_np[perm[i], :T_min], gt_sources[i][:T_min])
                      for i in range(n)]
            avg = sum(scores) / len(scores)
            if avg > best_sisnr:
                best_sisnr  = avg
                best_scores = scores

        metrics["si_sdr_db"] = best_sisnr
        metrics["si_sdr_per_speaker"] = best_scores
        print(f"\n  SI-SDR: {best_sisnr:.2f} dB")
        for i, s in enumerate(best_scores):
            print(f"    Speaker {i}: {s:.2f} dB")

    # DER (if ground truth available)
    if ref_labels is not None:
        T_sub = diar_np.shape[0]
        ref_r = ref_labels_resized if ref_labels_resized is not None else diar_np
        der_results = diarization_error_rate(diar_np, ref_r)
        metrics["DER"]  = der_results["DER"]
        metrics["FA"]   = der_results["FA"]
        metrics["MISS"] = der_results["MISS"]
        print(f"  DER : {der_results['DER']*100:.1f}%  "
              f"(FA={der_results['FA']*100:.1f}%  "
              f"MISS={der_results['MISS']*100:.1f}%)")

    # ── diarization timestamps ─────────────────────────────────────────────
    frame_dur = duration / diar_np.shape[0]
    timestamps = {}
    for spk in range(num_speakers):
        segs = []
        in_seg = False
        t_start = 0.0
        for t_idx, val in enumerate(diar_np[:, spk]):
            t = t_idx * frame_dur
            if val and not in_seg:
                t_start = t
                in_seg  = True
            elif not val and in_seg:
                segs.append((round(t_start, 2), round(t, 2)))
                in_seg = False
        if in_seg:
            segs.append((round(t_start, 2), round(duration, 2)))
        timestamps[f"speaker{spk}"] = segs

    # ── write report ───────────────────────────────────────────────────────
    report_lines = [
        "=" * 60,
        "EEND-SS INFERENCE REPORT",
        "=" * 60,
        f"Input file      : {input_path.name}",
        f"Duration        : {duration:.1f}s",
        f"Checkpoint      : {ckpt_path.name} (epoch {trained_epoch})",
        f"Device          : {device}",
        "",
        "── RESULTS ──────────────────────────────────────────",
        f"Detected speakers : {num_speakers}",
    ]

    if "si_sdr_db" in metrics:
        report_lines.append(f"SI-SDR (mean)     : {metrics['si_sdr_db']:.2f} dB")
        for i, s in enumerate(metrics.get("si_sdr_per_speaker", [])):
            report_lines.append(f"  Speaker {i}       : {s:.2f} dB")

    if "DER" in metrics:
        report_lines.append(f"DER               : {metrics['DER']*100:.1f}%")
        report_lines.append(f"  False Alarm     : {metrics['FA']*100:.1f}%")
        report_lines.append(f"  Missed Speech   : {metrics['MISS']*100:.1f}%")

    report_lines += [
        "",
        "── SPEAKER TIMESTAMPS ───────────────────────────────",
    ]
    for spk_key, segs in timestamps.items():
        seg_str = "  ".join([f"{s:.1f}–{e:.1f}s" for s, e in segs])
        report_lines.append(f"{spk_key}: {seg_str}")

    report_lines += [
        "",
        "── OUTPUT FILES ─────────────────────────────────────",
    ]
    for w in saved_wavs:
        report_lines.append(f"  {w.name}")
    report_lines.append(f"  diarization.png")
    report_lines.append(f"  report.txt")
    report_lines.append("=" * 60)

    report_text = "\n".join(report_lines)
    report_path = out_dir / "report.txt"
    report_path.write_text(report_text, encoding="utf-8")

    print("\n" + report_text)
    print(f"\nAll outputs saved to: {out_dir}/")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(get_device())
    parser = argparse.ArgumentParser(
        description="EEND-SS inference: separate speakers + diarization"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to input mixture .wav file"
    )
    parser.add_argument(
        "--checkpoint", default="models/eend_ss_best.pth",
        help="Path to model checkpoint (default: models/eend_ss_best.pth)"
    )
    parser.add_argument(
        "--output_dir", default="outputs/demo",
        help="Directory to save outputs (default: outputs/demo)"
    )
    parser.add_argument(
        "--ground_truth", action="store_true",
        help="Load ground truth sources from dataset manifest for SI-SDR + DER"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Diarization activity threshold (default: 0.5)"
    )
    args = parser.parse_args()
    run_inference(args)