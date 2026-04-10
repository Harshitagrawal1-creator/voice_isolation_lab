"""
src/data/dataset.py

EENDSSDataset: loads mixed audio + sources + diarization labels
for training, validation, and testing the EEND-SS model.

Manifest JSON structure expected:
{
    "mixture_id": "train_0000",
    "mixture_path": ".../train_0000.wav",
    "source_paths": [".../train_0000_speaker0.wav", ...],  # 2 or 3 entries
    "label_path": ".../train_0000_labels.npy",
    "duration": 27.30,
    "num_speakers": 3,
    "num_utterances": 10,
    "actual_overlap_ratio": 0.817
}
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

# ── constants ──────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16_000      # Hz — must match your dataset
MAX_SPEAKERS = 3          # model always works with 3 speaker slots
LABEL_FRAME_RATE = 100    # frames per second (10ms per frame)


class EENDSSDataset(Dataset):
    """
    Dataset for EEND-SS joint diarization + separation training.

    In training mode:  returns a random chunk_duration chunk from each sample.
    In eval mode:      returns the full audio (used for validation / test).

    Args:
        manifest_path (str | Path): path to train/val/test_manifest.json
        chunk_duration (float):     seconds per training chunk (default 4.0)
        mode (str):                 'train' | 'val' | 'test'
        max_samples (int | None):   cap dataset size (useful for fast debug runs)
    """

    def __init__(
        self,
        manifest_path: str | Path,
        chunk_duration: float = 4.0,
        mode: str = "train",
        max_samples: int | None = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(chunk_duration * SAMPLE_RATE)
        self.chunk_frames = int(chunk_duration * LABEL_FRAME_RATE)
        self.mode = mode
        self.training = mode == "train"

        # ── load manifest ──────────────────────────────────────────────────────
        with open(self.manifest_path) as f:
            self.samples = json.load(f)

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        print(
            f"[EENDSSDataset] {mode}: {len(self.samples)} samples loaded "
            f"from {self.manifest_path.name}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns a dict with:
            mixture   (torch.Tensor): [1, T]            mixed waveform
            sources   (torch.Tensor): [MAX_SPEAKERS, T] clean sources (padded with zeros)
            labels    (torch.Tensor): [T_frames, MAX_SPEAKERS] binary diarization
            num_speakers (int):       true number of speakers (2 or 3)
            mixture_id (str):         e.g. "train_0042"
        """
        meta = self.samples[idx]

        # ── resolve paths: relative paths are joined with data_root ───────────
        # Absolute paths work as-is (original machine).
        # Relative paths (after running fix_manifests.py) are joined with
        # the data root = manifest_path.parent.parent (data/processed/).
        def resolve(path_str):
            p = Path(path_str)
            if p.is_absolute():
                return p   # original machine absolute path
            # relative path: join with data/processed/
            return self.manifest_path.parent.parent / path_str

        # ── load mixture waveform ──────────────────────────────────────────────
        mixture, sr = torchaudio.load(resolve(meta["mixture_path"]))
        # torchaudio returns [channels, samples] — squeeze to mono [1, T]
        if mixture.shape[0] > 1:
            mixture = mixture.mean(dim=0, keepdim=True)
        assert sr == SAMPLE_RATE, f"Expected {SAMPLE_RATE}Hz, got {sr}Hz"

        # ── load clean sources ─────────────────────────────────────────────────
        source_list = []
        for path in meta["source_paths"]:
            src, _ = torchaudio.load(resolve(path))
            if src.shape[0] > 1:
                src = src.mean(dim=0, keepdim=True)
            source_list.append(src.squeeze(0))   # [T]

        # ── load diarization labels ────────────────────────────────────────────
        labels_np = np.load(resolve(meta["label_path"]))          # [T_frames, num_speakers]
        labels = torch.from_numpy(labels_np).float()     # float for BCE loss

        # ── chunk or full ──────────────────────────────────────────────────────
        total_samples = mixture.shape[-1]
        total_frames = labels.shape[0]
        num_speakers = meta["num_speakers"]

        # ✅ safety check
        assert num_speakers <= MAX_SPEAKERS, (
            f"num_speakers ({num_speakers}) exceeds MAX_SPEAKERS ({MAX_SPEAKERS})"
        )
        
        if self.training:
            mixture, source_list, labels = self._random_chunk(
                mixture, source_list, labels, total_samples, total_frames
            )

        # ── pad sources to MAX_SPEAKERS ────────────────────────────────────────
        # Model always expects MAX_SPEAKERS source slots.
        # If only 2 speakers: 3rd slot is zeros (silence).
        T = mixture.shape[-1]
        T_frames = labels.shape[0]

        padded_sources = torch.zeros(
        MAX_SPEAKERS, T, dtype=mixture.dtype
        )
        for i, src in enumerate(source_list):
            # src may differ slightly in length due to file rounding — trim/pad
            src_len = min(src.shape[0], T)
            padded_sources[i, :src_len] = src[:src_len]

        # ✅ explicitly ensure unused speakers are silent
        if num_speakers < MAX_SPEAKERS:
            padded_sources[num_speakers:] = 0.0

        # ── pad labels to MAX_SPEAKERS columns ────────────────────────────────
        padded_labels = torch.zeros(T_frames, MAX_SPEAKERS)
        padded_labels[:, :num_speakers] = labels

        return {
            "mixture":      mixture,           # [1, T]
            "sources":      padded_sources,    # [MAX_SPEAKERS, T]
            "labels":       padded_labels,     # [T_frames, MAX_SPEAKERS]
            "num_speakers": num_speakers,      # int 2 or 3
            "mixture_id":   meta["mixture_id"],
        }

    # ── private helpers ────────────────────────────────────────────────────────

    def _random_chunk(
        self,
        mixture: torch.Tensor,
        source_list: list,
        labels: torch.Tensor,
        total_samples: int,
        total_frames: int,
    ):
        """
        Cut a random chunk_duration window from all signals simultaneously.
        The audio chunk and label chunk are aligned by start time.
        """
        max_start = total_samples - self.chunk_samples
        if max_start <= 0:
            # audio shorter than chunk — pad to chunk length instead
            pad = self.chunk_samples - total_samples
            mixture = torch.nn.functional.pad(mixture, (0, pad))
            source_list = [
                torch.nn.functional.pad(s, (0, pad)) for s in source_list
            ]
            frame_pad = self.chunk_frames - total_frames
            if frame_pad > 0:
                labels = torch.nn.functional.pad(
                    labels.T, (0, frame_pad)
                ).T
            return mixture, source_list, labels[:self.chunk_frames]

        # pick random start sample, then compute matching frame start
        start_sample = random.randint(0, max_start)
        end_sample = start_sample + self.chunk_samples

        # convert sample position to frame position
        start_frame = int(start_sample / SAMPLE_RATE * LABEL_FRAME_RATE)
        end_frame = start_frame + self.chunk_frames
        end_frame = min(end_frame, total_frames)

        mixture = mixture[:, start_sample:end_sample]
        source_list = [s[start_sample:end_sample] for s in source_list]
        labels = labels[start_frame:end_frame]

        # if label chunk came up short, pad
        if labels.shape[0] < self.chunk_frames:
            pad = self.chunk_frames - labels.shape[0]
            labels = torch.nn.functional.pad(labels.T, (0, pad)).T

        return mixture, source_list, labels