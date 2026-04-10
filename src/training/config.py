"""
src/training/config.py

All hyperparameters in one place.
Change values here — do not hardcode them in trainer.py or other files.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:

    # ── paths ──────────────────────────────────────────────────────────────────
    data_dir:     str = "data/processed"
    output_dir:   str = "models"
    log_dir:      str = "outputs"

    # ── model architecture ─────────────────────────────────────────────────────
    # Conv-TasNet
    N: int = 256          # encoder filters      (paper: 512 — reduce if OOM)
    L: int = 16           # encoder kernel size
    B: int = 128          # TCN bottleneck channels
    H: int = 256          # TCN hidden channels  (paper: 512 — reduce if OOM)
    P: int = 3            # TCN kernel size
    X: int = 8            # TCN layers per repeat
    R: int = 3            # TCN repeats
    C: int = 3            # max speakers

    # EEND
    d_model:   int = 256  # transformer hidden dim
    n_heads:   int = 4    # attention heads
    n_layers:  int = 4    # transformer layers
    d_ff:      int = 1024 # feed-forward dim
    dropout:   float = 0.2   # increased from 0.1 to reduce overfitting
    subsample: int = 8    # temporal subsampling factor

    # ── data loading ───────────────────────────────────────────────────────────
    chunk_duration: float = 10.0   # increased from 4.0 — gives diarization branch more context
    batch_size:     int   = 8     # reduced from 16 because chunks are now longer
    num_workers:    int   = 4

    # ── training ───────────────────────────────────────────────────────────────
    epochs:       int   = 200
    lr:           float = 1e-3
    weight_decay: float = 1e-4   # increased from 1e-5 to reduce overfitting
    grad_clip:    float = 5.0

    # early stopping — increased patience so training isn't cut short
    patience:     int   = 15     # was 5 — val loss oscillates so needs more patience
    lr_patience:  int   = 5      # was 3

    # ── loss weights ───────────────────────────────────────────────────────────
    # Boosted diarization weight so the branch actually learns.
    # Previous 0.2 was too small — SI-SNR gradient dominated and
    # diarization BCE sat at ~0.485 (random) for all 35 epochs.
    # Once diarization val loss drops below 0.40, you can rebalance
    # back toward lambda_sisnr=1.0, lambda_diar=0.2, lambda_exist=0.2
    lambda_sisnr: float = 0.5
    lambda_diar:  float = 1.0
    lambda_exist: float = 0.5

    # ── logging ────────────────────────────────────────────────────────────────
    log_every:      int = 10
    val_every:      int = 1
    save_every:     int = 5

    # ── debug / fast runs ──────────────────────────────────────────────────────
    max_train:      int | None = None
    max_val:        int | None = None   # was 100 — do full val to get accurate metrics


def get_config() -> Config:
    return Config()
