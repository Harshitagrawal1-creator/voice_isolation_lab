"""
src/data/loaders.py

Single entry point to get train / val / test DataLoaders.

Usage:
    from src.data.loaders import get_loaders
    train_loader, val_loader, test_loader = get_loaders(cfg)

cfg is a dict (or SimpleNamespace) with keys:
    data_dir        str   path to data/processed/
    chunk_duration  float seconds per training chunk (default 4.0)
    batch_size      int   training batch size (default 8)
    num_workers     int   DataLoader workers (default 0 on MPS to avoid issues)
    max_train       int|None  cap training samples (None = use all)
"""

from pathlib import Path
from torch.utils.data import DataLoader

from .dataset import EENDSSDataset
from .collate import collate_train, collate_eval


def get_loaders(cfg: dict):
    """
    Returns (train_loader, val_loader, test_loader).
    """
    data_dir = Path(cfg["data_dir"])

    train_ds = EENDSSDataset(
        manifest_path=data_dir / "train" / "train_manifest.json",
        chunk_duration=cfg.get("chunk_duration", 4.0),
        mode="train",
        max_samples=cfg.get("max_train", None),
    )

    val_ds = EENDSSDataset(
        manifest_path=data_dir / "val" / "val_manifest.json",
        chunk_duration=cfg.get("chunk_duration", 4.0),
        mode="val",
    )

    test_ds = EENDSSDataset(
        manifest_path=data_dir / "test" / "test_manifest.json",
        chunk_duration=cfg.get("chunk_duration", 4.0),
        mode="test",
    )

    # num_workers=0 is safest on MPS (avoids multiprocessing issues with Metal)
    # increase to 2-4 if you have no issues
    nw = cfg.get("num_workers", 0)
    bs = cfg.get("batch_size", 8)

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        collate_fn=collate_train,
        pin_memory=False,   # pin_memory=True only helps CUDA, not MPS
        drop_last=True,     # keeps batch size consistent during training
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=4,       # smaller batch for eval (full-length audio)
        shuffle=False,
        num_workers=nw,
        collate_fn=collate_eval,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=4,
        shuffle=False,
        num_workers=nw,
        collate_fn=collate_eval,
        pin_memory=False,
    )

    print(f"[loaders] train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}")
    print(f"[loaders] train batches/epoch: {len(train_loader)}")

    return train_loader, val_loader, test_loader
