"""
fix_manifests.py — run from your project root on ANY machine before training

    python fix_manifests.py

What it does:
    Rewrites all absolute paths in the manifest JSON files to paths
    relative to the data/processed/ directory.

    Before:  /Users/apple/Desktop/voice_isolation_desktop/data/processed/train/mixtures/train_0000.wav
    After:   train/mixtures/train_0000.wav

This makes the manifests portable — they work on your Mac, on Colab,
on any machine regardless of where the project folder lives.

On Colab you then just set:
    DATA_ROOT = "/content/drive/MyDrive/voice_isolation/data/processed"
And dataset.py resolves all paths as DATA_ROOT / relative_path.
"""

import json
from pathlib import Path


def fix_manifest(manifest_path: Path, data_root: Path):
    """Rewrite all paths in a manifest to be relative to data_root."""
    with open(manifest_path) as f:
        samples = json.load(f)

    fixed = 0
    for sample in samples:
        # mixture_path
        p = Path(sample["mixture_path"])
        try:
            sample["mixture_path"] = str(p.relative_to(data_root))
            fixed += 1
        except ValueError:
            pass  # already relative or different root

        # source_paths
        new_sources = []
        for sp in sample["source_paths"]:
            try:
                new_sources.append(str(Path(sp).relative_to(data_root)))
            except ValueError:
                new_sources.append(sp)
        sample["source_paths"] = new_sources

        # label_path
        try:
            sample["label_path"] = str(
                Path(sample["label_path"]).relative_to(data_root)
            )
        except ValueError:
            pass

    with open(manifest_path, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"  {manifest_path.name}: fixed {fixed}/{len(samples)} samples")


def main():
    # find project root (wherever this script lives)
    project_root = Path(__file__).parent
    data_root    = project_root / "data" / "processed"

    if not data_root.exists():
        print(f"ERROR: data root not found at {data_root}")
        print("Run this script from the voice_isolation_desktop/ project root.")
        return

    print(f"Data root: {data_root}")
    print("Rewriting manifests to relative paths...\n")

    manifests = [
        data_root / "train" / "train_manifest.json",
        data_root / "val"   / "val_manifest.json",
        data_root / "test"  / "test_manifest.json",
    ]

    for m in manifests:
        if m.exists():
            fix_manifest(m, data_root)
        else:
            print(f"  WARNING: {m} not found — skipping")

    print("\nDone. Manifests now use relative paths.")
    print("Verify with: python test_dataset.py")


if __name__ == "__main__":
    main()
