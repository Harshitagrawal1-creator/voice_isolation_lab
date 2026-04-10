"""
train.py — run from your project root

    # Quick smoke test (200 samples, 3 epochs — verifies everything works)
    python train.py --smoke

    # Full training from scratch (always do this after fixing bugs)
    python train.py

    # Resume from a checkpoint (only use checkpoints trained with fixed code)
    python train.py --resume models/eend_ss_best.pth
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch

from src.training.config import Config, get_config
from src.training.trainer import Trainer, get_device
from src.models.eend_ss import EENDSS
from src.data.loaders import get_loaders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke",  action="store_true",
                        help="Quick smoke test: 200 train samples, 3 epochs")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (only use post-fix checkpoints)")
    args = parser.parse_args()

    cfg = get_config()

    if args.smoke:
        print("[train.py] SMOKE TEST MODE — 200 samples, 3 epochs")
        cfg.max_train   = 200
        cfg.max_val     = 50
        cfg.batch_size  = 2
        cfg.epochs      = 3
        cfg.log_every   = 5

    # ── build model ───────────────────────────────────────────────────────────
    model = EENDSS(
        N=cfg.N, L=cfg.L, B=cfg.B, H=cfg.H, P=cfg.P,
        X=cfg.X, R=cfg.R, C=cfg.C,
        d_model=cfg.d_model, n_heads=cfg.n_heads,
        n_layers=cfg.n_layers, d_ff=cfg.d_ff,
        dropout=cfg.dropout, subsample=cfg.subsample,
    )

    # ── optional: resume from checkpoint ─────────────────────────────────────
    # WARNING: do NOT resume from checkpoints trained before the num_speakers
    # bug fix — those weights have learned incorrect behaviour and will not
    # recover. Always retrain from scratch after applying the fixes.
    if args.resume:
        device = get_device()
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        print(f"[train.py] Resuming from epoch {ckpt['epoch'] + 1}")
        print(f"           checkpoint: {args.resume}")
        print(f"           val loss was: {ckpt.get('val_loss', 'N/A'):.4f}")
    else:
        print("[train.py] Training from scratch with fixed code")

    # ── data loaders ──────────────────────────────────────────────────────────
    train_loader, val_loader, _ = get_loaders({
        "data_dir":       cfg.data_dir,
        "chunk_duration": cfg.chunk_duration,
        "batch_size":     cfg.batch_size,
        "num_workers":    cfg.num_workers,
        "max_train":      cfg.max_train,
        "max_val":        cfg.max_val,
    })

    # ── train ─────────────────────────────────────────────────────────────────
    trainer = Trainer(model, cfg)
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
