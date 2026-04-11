"""
src/training/trainer.py

Production training loop for EEND-SS.

Features:
    - Full epoch training with gradient clipping
    - Validation loop after every epoch
    - Best model checkpointing (saved when val loss improves)
    - ReduceLROnPlateau scheduler (halves LR after N epochs no improvement)
    - Early stopping (stops after patience epochs no improvement)
    - Training history saved to JSON for plotting
    - Handles MPS / CUDA / CPU automatically
"""

import json
import time
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .losses import EENDSSLoss
from .config import Config


# ── Device helper ─────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Trainer ───────────────────────────────────────────────────────────────────

class Trainer:
    """
    Manages the full training lifecycle for EEND-SS.

    Usage:
        trainer = Trainer(model, cfg)
        trainer.train(train_loader, val_loader)
    """

    def __init__(self, model: nn.Module, cfg: Config):
        self.cfg    = cfg
        self.device = get_device()
        self.model  = model.to(self.device)

        print(f"[Trainer] Device: {self.device}")

        # ── output directories ─────────────────────────────────────────────────
        self.output_dir = Path(cfg.output_dir)
        self.log_dir    = Path(cfg.log_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # ── loss ───────────────────────────────────────────────────────────────
        self.criterion = EENDSSLoss(
            lambda_sisnr=cfg.lambda_sisnr,
            lambda_diar=cfg.lambda_diar,
            lambda_exist=cfg.lambda_exist,
        )

        # ── optimiser ─────────────────────────────────────────────────────────
        self.optimiser = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        
        # ── resume training from checkpoint ────────────────────────────────────
        ckpt_path = Path(cfg.output_dir) / "eend_ss_best.pth"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=self.device)
            
            self.model.load_state_dict(ckpt["model_state"])
            self.optimiser.load_state_dict(ckpt["optim_state"])
            # self.scheduler.load_state_dict(ckpt["scheduler_state"])
            
            self.best_val_loss = ckpt.get("val_loss", float("inf"))
            self.start_epoch = ckpt.get("epoch", 0) + 1
            
            print(f"[Trainer] Resumed from checkpoint: epoch {ckpt['epoch']}")
            print(f"[Trainer] Best val loss: {self.best_val_loss:.4f}")
        
        # ── scheduler: halve LR when val loss plateaus ─────────────────────────
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiser,
            mode="min",
            factor=0.5,
            patience=cfg.lr_patience,
        )

        # ── state ──────────────────────────────────────────────────────────────
        self.best_val_loss     = float("inf")
        self.epochs_no_improve = 0
        self.start_epoch       = 1

        # load history from disk if it exists (supports resume)
        history_path = Path(cfg.log_dir) / "training_history.json"
        if history_path.exists():
            with open(history_path) as f:
                self.history = json.load(f)
                self.start_epoch = len(self.history["train_loss"]) + 1
            print(f"[Trainer] Loaded existing history ({len(self.history['train_loss'])} epochs)")
        else:
            self.history = {
                "train_loss": [], "val_loss": [],
                "train_sisnr": [], "val_sisnr": [],
                "train_diar": [], "val_diar": [],
                "lr": [],
            }

    # ── public API ─────────────────────────────────────────────────────────────

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Run full training loop."""
        print(f"\n[Trainer] Starting training for up to {self.cfg.epochs} epochs")
        print(f"  Train batches/epoch : {len(train_loader)}")
        print(f"  Val batches/epoch   : {len(val_loader)}")
        print(f"  Checkpoint dir      : {self.output_dir}\n")

        for epoch in range(self.start_epoch, self.cfg.epochs + 1):
            t0 = time.time()

            # ── train ──────────────────────────────────────────────────────────
            train_metrics = self._train_epoch(train_loader, epoch)

            # ── validate ───────────────────────────────────────────────────────
            val_metrics = self._val_epoch(val_loader, epoch)

            elapsed = time.time() - t0
            current_lr = self.optimiser.param_groups[0]["lr"]

            # ── log ────────────────────────────────────────────────────────────
            self._log_epoch(epoch, train_metrics, val_metrics, elapsed, current_lr)

            # ── scheduler step ─────────────────────────────────────────────────
            self.scheduler.step(val_metrics["loss"])

            # ── checkpoint ────────────────────────────────────────────────────
            is_best = val_metrics["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss     = val_metrics["loss"]
                self.epochs_no_improve = 0
                self._save_checkpoint(epoch, val_metrics, is_best=True)
                print(f"  ✓ New best val loss: {self.best_val_loss:.4f} — saved")
            else:
                self.epochs_no_improve += 1

            if epoch % self.cfg.save_every == 0:
                self._save_checkpoint(epoch, val_metrics, is_best=False)

            # ── early stopping ─────────────────────────────────────────────────
            if self.epochs_no_improve >= self.cfg.patience:
                print(f"\n[Trainer] Early stopping at epoch {epoch} "
                      f"(no improvement for {self.cfg.patience} epochs)")
                break

        self._save_history()
        print(f"\n[Trainer] Training complete. Best val loss: {self.best_val_loss:.4f}")

    # ── private: one training epoch ───────────────────────────────────────────

    def _train_epoch(self, loader: DataLoader, epoch: int) -> dict:
        self.model.train()

        running = {"loss": 0.0, "sisnr": 0.0, "diar": 0.0, "exist": 0.0}
        n_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for batch_idx, batch in enumerate(pbar):
            # move to device
            mixture      = batch["mixture"].to(self.device)       # [B, 1, T]
            sources      = batch["sources"].to(self.device)       # [B, C, T]
            labels       = batch["labels"].to(self.device)        # [B, T_f, C]
            num_speakers = batch["num_speakers"].to(self.device)  # [B]

            # FIX: pass the full per-sample num_speakers tensor — NOT the batch max.
            # The model/EEND uses tensor.max() internally to decide attractor count,
            # but the loss uses per-sample counts to compute per-sample PIT correctly.
            separated, diar_probs, exist_probs, _ = self.model(
                mixture, num_speakers=num_speakers
            )

            # compute loss
            loss_dict = self.criterion(
                separated=separated,
                diar_probs=diar_probs,
                exist_probs=exist_probs,
                sources=sources,
                labels=labels,
                num_speakers=num_speakers,
            )
            loss = loss_dict["loss"]

            # backward
            self.optimiser.zero_grad()
            loss.backward()

            # gradient clipping (prevents exploding gradients in early training)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

            self.optimiser.step()

            # accumulate
            running["loss"]  += loss.item()
            running["sisnr"] += loss_dict["si_snr_db"].item()
            running["diar"]  += loss_dict["loss_diar"].item()
            running["exist"] += loss_dict["loss_exist"].item()
            n_batches += 1

            avg_loss  = running["loss"]  / n_batches
            avg_sisnr = running["sisnr"] / n_batches

            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "SI-SNR": f"{avg_sisnr:.2f}dB",
                "diar": f"{running['diar']/n_batches:.4f}",
            })

        n = max(n_batches, 1)
        return {
            "loss":  running["loss"]  / n,
            "sisnr": running["sisnr"] / n,
            "diar":  running["diar"]  / n,
            "exist": running["exist"] / n,
        }

    # ── private: one validation epoch ────────────────────────────────────────

    def _val_epoch(self, loader: DataLoader, epoch: int) -> dict:
        self.model.eval()

        running = {"loss": 0.0, "sisnr": 0.0, "diar": 0.0, "exist": 0.0}
        n_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]", leave=False)
        with torch.no_grad():
            for batch in pbar:
                mixture      = batch["mixture"].to(self.device)
                sources      = batch["sources"].to(self.device)
                labels       = batch["labels"].to(self.device)
                num_speakers = batch["num_speakers"].to(self.device)
                lengths_f    = batch.get("lengths_f")
                if lengths_f is not None:
                    lengths_f = lengths_f.to(self.device)

                # cap val audio to 8s to avoid O(T^2) attention OOM
                MAX_VAL_SAMPLES = 8 * 16000
                if mixture.shape[-1] > MAX_VAL_SAMPLES:
                    mixture = mixture[:, :, :MAX_VAL_SAMPLES]
                    sources = sources[:, :, :MAX_VAL_SAMPLES]

                # FIX: pass the full per-sample num_speakers tensor
                separated, diar_probs, exist_probs, _ = self.model(
                    mixture, num_speakers=num_speakers
                )

                # trim separated to match sources length
                T_src = sources.shape[-1]
                T_sep = separated.shape[-1]
                if T_sep > T_src:
                    separated = separated[:, :, :T_src]
                elif T_sep < T_src:
                    sources = sources[:, :, :T_sep]

                loss_dict = self.criterion(
                    separated=separated,
                    diar_probs=diar_probs,
                    exist_probs=exist_probs,
                    sources=sources,
                    labels=labels,
                    num_speakers=num_speakers,
                    lengths_f=lengths_f,
                )

                running["loss"]  += loss_dict["loss"].item()
                running["sisnr"] += loss_dict["si_snr_db"].item()
                running["diar"]  += loss_dict["loss_diar"].item()
                running["exist"] += loss_dict["loss_exist"].item()
                n_batches += 1

                avg_loss  = running["loss"]  / n_batches
                avg_sisnr = running["sisnr"] / n_batches

                pbar.set_postfix({
                    "val_loss": f"{avg_loss:.4f}",
                    "val_SI-SNR": f"{avg_sisnr:.2f}dB",
                    "val_diar": f"{running['diar']/n_batches:.4f}",
                })

        n = max(n_batches, 1)
        return {
            "loss":  running["loss"]  / n,
            "sisnr": running["sisnr"] / n,
            "diar":  running["diar"]  / n,
            "exist": running["exist"] / n,
        }

    # ── private: checkpoint ───────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, val_metrics: dict, is_best: bool):
        checkpoint = {
            "epoch":       epoch,
            "model_state": self.model.state_dict(),
            "optim_state": self.optimiser.state_dict(),
            "val_loss":    val_metrics["loss"],
            "val_sisnr":   val_metrics["sisnr"],
            "config":      self.cfg.__dict__,
        }

        if is_best:
            path = self.output_dir / "eend_ss_best.pth"
        else:
            path = self.output_dir / f"eend_ss_epoch{epoch:03d}.pth"

        torch.save(checkpoint, path)

    # ── private: logging ──────────────────────────────────────────────────────

    def _log_epoch(self, epoch, train_m, val_m, elapsed, lr):
        self.history["train_loss"].append(train_m["loss"])
        self.history["val_loss"].append(val_m["loss"])
        self.history["train_sisnr"].append(train_m["sisnr"])
        self.history["val_sisnr"].append(val_m["sisnr"])
        self.history["train_diar"].append(train_m["diar"])
        self.history["val_diar"].append(val_m["diar"])
        self.history["lr"].append(lr)

        print(
            f"\nEpoch {epoch:3d} ({elapsed:.0f}s) | LR {lr:.2e}\n"
            f"  Train  — loss: {train_m['loss']:.4f} | "
            f"SI-SNR: {train_m['sisnr']:+.2f} dB | "
            f"diar: {train_m['diar']:.4f}\n"
            f"  Val    — loss: {val_m['loss']:.4f}  | "
            f"SI-SNR: {val_m['sisnr']:+.2f} dB | "
            f"diar: {val_m['diar']:.4f}"
        )

    def _save_history(self):
        path = self.log_dir / "training_history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"[Trainer] History saved to {path}")
