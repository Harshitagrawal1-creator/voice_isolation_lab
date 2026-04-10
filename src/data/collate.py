"""
src/data/collate.py

Custom collate functions for DataLoader.

During training:  all chunks are the same length (chunk_duration),
                  so the default collate works fine.

During val/test:  audio is full-length and varies per sample.
                  We pad all items in the batch to the longest item.
"""

import torch
from torch.nn.utils.rnn import pad_sequence


def collate_train(batch: list[dict]) -> dict:
    """
    Standard collate for fixed-length training chunks.
    All tensors in the batch have identical shapes — just stack them.
    """
    return {
        "mixture":      torch.stack([b["mixture"] for b in batch]),       # [B, 1, T]
        "sources":      torch.stack([b["sources"] for b in batch]),       # [B, 3, T]
        "labels":       torch.stack([b["labels"] for b in batch]),        # [B, T_f, 3]
        "num_speakers": torch.tensor([b["num_speakers"] for b in batch]), # [B]
        "mixture_id":   [b["mixture_id"] for b in batch],                 # list[str]
    }


def collate_eval(batch: list[dict]) -> dict:
    """
    Padding collate for variable-length val/test batches.

    Pads all waveforms and label sequences to the longest item in the batch.
    Returns an extra 'lengths' tensor so the model can ignore padding.
    """
    # sort by length descending (required by some RNN ops — good habit)
    batch = sorted(batch, key=lambda b: b["mixture"].shape[-1], reverse=True)

    mixtures = [b["mixture"].squeeze(0) for b in batch]   # list of [T_i]
    sources  = [b["sources"] for b in batch]              # list of [3, T_i]
    labels   = [b["labels"] for b in batch]               # list of [T_f_i, 3]

    lengths     = torch.tensor([m.shape[-1] for m in mixtures])   # [B]
    lengths_f   = torch.tensor([l.shape[0] for l in labels])      # [B]
    num_spk     = torch.tensor([b["num_speakers"] for b in batch]) # [B]

    # pad waveforms: pad_sequence expects [T, ...] so we transpose
    mix_padded = pad_sequence(mixtures, batch_first=True)          # [B, T_max]
    mix_padded = mix_padded.unsqueeze(1)                           # [B, 1, T_max]

    # pad sources: [3, T_i] -> stack with zero padding
    T_max = mix_padded.shape[-1]
    src_padded = torch.zeros(len(batch), 3, T_max)
    for i, src in enumerate(sources):
        t = src.shape[-1]
        src_padded[i, :, :t] = src

    # pad labels along time dimension
    T_f_max = max(l.shape[0] for l in labels)
    lbl_padded = torch.zeros(len(batch), T_f_max, 3)
    for i, lbl in enumerate(labels):
        t = lbl.shape[0]
        lbl_padded[i, :t, :] = lbl

    return {
        "mixture":      mix_padded,    # [B, 1, T_max]
        "sources":      src_padded,    # [B, 3, T_max]
        "labels":       lbl_padded,    # [B, T_f_max, 3]
        "num_speakers": num_spk,       # [B]
        "lengths":      lengths,       # [B]  — audio sample lengths
        "lengths_f":    lengths_f,     # [B]  — frame lengths
        "mixture_id":   [b["mixture_id"] for b in batch],
    }
