"""
src/models/eend_ss.py

EEND-SS: Joint End-to-End Neural Speaker Diarization and Separation.

Reference:
    Maiti et al., "EEND-SS: Joint end-to-end neural speaker diarization
    and speech separation for flexible number of speakers", ICASSP 2023.

Architecture (Figure 1 from paper):
    ┌─────────────────────────────────────────────────────┐
    │  Input mixture x  [B, 1, T]                        │
    │         │                                           │
    │    ┌────▼─────┐                                     │
    │    │ Encoder  │   [B, N, T_enc]                     │
    │    └────┬─────┘                                     │
    │         │           SHARED NETWORK                  │
    │    ┌────▼──────┐                                    │
    │    │LayerNorm  │                                    │
    │    │1x1 Conv   │   [B, B_ch, T_enc]                 │
    │    │  TCNs     │  ← TCN bottleneck features         │
    │    └────┬──────┘                                    │
    │         │                                           │
    │    ┌────┴──────────────────────┐                    │
    │    │                           │                    │
    │ ┌──▼──────────────┐    ┌───────▼──────────────┐    │
    │ │ SEPARATION BRANCH│    │ DIARIZATION BRANCH   │    │
    │ │  masks -> decoder│    │ Transformer -> EDA   │    │
    │ │  [B, C, T]      │    │ [B, T_sub, C]        │    │
    │ └────────────────┘     └──────────────────────┘    │
    └─────────────────────────────────────────────────────┘

The key insight: both branches share the TCN bottleneck features.
The separation branch uses those features to estimate masks.
The diarization branch uses those same features to estimate speaker activity.
This forces the shared network to learn representations useful for BOTH tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_tasnet import ConvTasNet
from .eend import EEND


class EENDSS(nn.Module):
    """
    Joint EEND-SS model.

    Args:
        # Conv-TasNet params
        N (int): encoder filters (paper: 512)
        L (int): encoder kernel size (paper: 16)
        B (int): TCN bottleneck channels (paper: 128)
        H (int): TCN hidden channels (paper: 512)
        P (int): TCN kernel size (paper: 3)
        X (int): TCN layers per repeat (paper: 8)
        R (int): TCN repeats (paper: 3)
        C (int): max speakers (3 for our task)

        # EEND params
        d_model (int):  transformer hidden dim (paper: 256)
        n_heads (int):  attention heads (paper: 4)
        n_layers (int): transformer layers (paper: 4)
        subsample (int): temporal subsampling factor (paper: 8)
    """

    def __init__(
        self,
        # Conv-TasNet
        N: int = 512,
        L: int = 16,
        B: int = 128,
        H: int = 512,
        P: int = 3,
        X: int = 8,
        R: int = 3,
        C: int = 3,
        # EEND
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        subsample: int = 8,
    ):
        super().__init__()
        self.C = C

        # ── shared network (Conv-TasNet encoder + TCN) ─────────────────────────
        self.conv_tasnet = ConvTasNet(
            N=N, L=L, B=B, H=H, P=P, X=X, R=R, C=C
        )

        # ── diarization branch (EEND-EDA) ──────────────────────────────────────
        # input_dim = B (TCN bottleneck channels)
        self.eend = EEND(
            input_dim=B,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            C_max=C + 1,       # C speakers + 1 "no more speakers" slot
            subsample=subsample,
            max_speakers=C,
        )

        # log model size
        n_params = sum(p.numel() for p in self.parameters())
        print(f"[EEND-SS] Total parameters: {n_params:,}")
        sep_params  = sum(p.numel() for p in self.conv_tasnet.parameters())
        diar_params = sum(p.numel() for p in self.eend.parameters())
        print(f"  Conv-TasNet: {sep_params:,}")
        print(f"  EEND:        {diar_params:,}")

    def forward(
        self,
        mixture: torch.Tensor,
        num_speakers: torch.Tensor | int | None = None,
    ):
        """
        Full forward pass.

        Args:
            mixture (torch.Tensor): [B, 1, T] — mixed audio waveform
            num_speakers (torch.Tensor | int | None):
                Per-sample oracle speaker counts [B] for training.
                Pass the full tensor — NOT just the batch maximum.
                None during inference (EEND estimates it).

        Returns:
            separated   (torch.Tensor): [B, C, T]         — separated waveforms
            diar_probs  (torch.Tensor): [B, T_sub, C]     — speaker activity probs
            exist_probs (torch.Tensor): [B, C+1]          — attractor existence probs
            masks       (torch.Tensor): [B, C, N, T_enc]  — separation masks
        """
        # ── shared: encode + TCN ───────────────────────────────────────────────
        separated, tcn_features, masks = self.conv_tasnet(mixture)
        # separated:    [B, C, T]
        # tcn_features: [B, B_ch, T_enc]
        # masks:        [B, C, N, T_enc]

        # ── diarization branch ─────────────────────────────────────────────────
        # FIX: pass the full per-sample num_speakers tensor to EEND.
        # The EEND forward uses the batch maximum for attractor generation
        # (so all samples in the batch get the same number of attractor slots),
        # but the loss function uses the per-sample counts correctly.
        diar_probs, exist_probs, _ = self.eend(tcn_features, num_speakers=num_speakers)
        # diar_probs:  [B, T_sub, C]
        # exist_probs: [B, C+1]

        return separated, diar_probs, exist_probs, masks

    def inference(self, mixture: torch.Tensor, threshold: float = 0.5):
        """
        Two-pass inference as described in paper Section 3.5:
            Pass 1: Get diarization probs + estimated speaker count
            Pass 2: Use estimated count to select appropriate masks,
                    then apply fusion (multiply separated audio by activity)

        Args:
            mixture (torch.Tensor): [1, 1, T] — single sample, batch size 1
            threshold (float): speaker activity threshold for binarisation

        Returns:
            dict with keys:
                'separated'     [1, C_hat, T]    — separated waveforms
                'diar_probs'    [1, T_sub, C]    — raw diarization probabilities
                'diar_binary'   [1, T_sub, C]    — thresholded binary labels
                'num_speakers'  int              — estimated speaker count
                'exist_probs'   [1, C+1]         — attractor existence probs
        """
        self.eval()
        with torch.no_grad():
            # inference: pass num_speakers=None so EEND estimates speaker count
            separated, diar_probs, exist_probs, masks = self.forward(
                mixture, num_speakers=None
            )

        # estimate speaker count from existence probabilities
        num_spk = self.eend.predict_num_speakers(exist_probs).item()
        num_spk = max(1, min(num_spk, self.C))   # clamp to [1, C]

        # binarise diarization
        diar_binary = (diar_probs > threshold).float()

        # ── fusion: multiply separated signal by speaker activity ──────────────
        # This reduces background bleed when a speaker is not active.
        # diar_probs: [1, T_sub, C]  needs upsampling to match separated [1, C, T]
        T_audio = separated.shape[-1]
        T_sub   = diar_probs.shape[1]

        # upsample diarization probs from T_sub to T_audio
        # diar_probs: [1, T_sub, C] -> [1, C, T_sub] -> upsample -> [1, C, T_audio]
        diar_up = diar_probs.permute(0, 2, 1)                        # [1, C, T_sub]
        diar_up = F.interpolate(diar_up.cpu(), size=T_audio, mode="linear",
                                align_corners=False).to(separated.device)  # [1, C, T_audio]

        # apply fusion only to the estimated number of active speakers
        separated_fused = separated[:, :num_spk] * diar_up[:, :num_spk]

        return {
            "separated":    separated_fused,                          # [1, C_hat, T]
            "diar_probs":   diar_probs,                               # [1, T_sub, C]
            "diar_binary":  diar_binary,                              # [1, T_sub, C]
            "num_speakers": num_spk,
            "exist_probs":  exist_probs,                              # [1, C+1]
        }

    @property
    def device(self):
        return next(self.parameters()).device
