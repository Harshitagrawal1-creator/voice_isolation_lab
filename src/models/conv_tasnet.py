"""
src/models/conv_tasnet.py

Conv-TasNet implementation following:
    Luo & Mesgarani, "Conv-TasNet: Surpassing ideal time-frequency
    magnitude masking for speech separation", TASLP 2019.

Architecture:
    Encoder   : 1D conv, waveform -> N-dim representation
    Separator : LayerNorm -> 1x1Conv -> stacked TCN blocks -> masks
    Decoder   : 1D transposed conv, masked representation -> waveforms

Paper hyperparameters (Table 1):
    N=512  (encoder filters)
    L=16   (encoder kernel size)
    B=128  (TCN bottleneck channels)
    H=512  (TCN hidden channels)
    P=3    (TCN kernel size)
    X=8    (TCN layers per repeat)
    R=3    (TCN repeats)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Encoder ───────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    Transforms raw waveform into N-dimensional encoded representation.

    Input:  [B, 1, T]         — mixture waveform
    Output: [B, N, T_enc]     — encoded features

    The 1D conv acts as a learned filterbank — the model learns
    what frequency decomposition is best for separation, rather
    than using a fixed STFT.

    Args:
        N (int): number of filters (encoder output channels)
        L (int): filter length (kernel size)
        stride (int): hop size between frames (L//2 for 50% overlap)
    """

    def __init__(self, N: int = 512, L: int = 16, stride: int = 8):
        super().__init__()
        self.N = N
        self.L = L
        self.stride = stride

        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=N,
            kernel_size=L,
            stride=stride,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, T]
        return F.relu(self.conv(x))  # [B, N, T_enc]


# ── TCN Building Blocks ───────────────────────────────────────────────────────

class DepthwiseSeparableConv(nn.Module):
    """
    One dilated depthwise-separable conv block inside the TCN.

    Structure:
        1x1 Conv (expand to H) -> PReLU -> norm
        -> depthwise dilated Conv1d (kernel P, dilation d)
        -> PReLU -> norm
        -> 1x1 Conv (compress back to B)
        + residual skip connection

    Args:
        B (int): bottleneck channels (input/output)
        H (int): hidden channels (expanded inside block)
        P (int): depthwise conv kernel size
        dilation (int): dilation factor (2^layer_index)
    """

    def __init__(self, B: int, H: int, P: int, dilation: int):
        super().__init__()

        # padding to keep time dimension the same (causal = False for offline)
        padding = (P - 1) * dilation // 2

        self.net = nn.Sequential(
            # pointwise expand
            nn.Conv1d(B, H, kernel_size=1),
            nn.PReLU(),
            nn.GroupNorm(1, H, eps=1e-8),
            # depthwise dilated
            nn.Conv1d(
                H, H,
                kernel_size=P,
                dilation=dilation,
                padding=padding,
                groups=H,          # depthwise: one filter per channel
            ),
            nn.PReLU(),
            nn.GroupNorm(1, H, eps=1e-8),
            # pointwise compress
            nn.Conv1d(H, B, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B_batch, B_channels, T]
        return x + self.net(x)   # residual connection


class TCN(nn.Module):
    """
    Temporal Convolutional Network: stacked dilated conv blocks.

    R repeats of X blocks each. Dilation doubles every block:
        1, 2, 4, 8, 16, 32, 64, 128  (X=8 layers)
    then resets for the next repeat.

    This gives a receptive field of:
        R × (2^X - 1) × P × stride frames

    Args:
        B (int): bottleneck input/output channels
        H (int): hidden channels per block
        P (int): depthwise kernel size
        X (int): number of blocks per repeat
        R (int): number of repeats
    """

    def __init__(self, B: int = 128, H: int = 512, P: int = 3,
                 X: int = 8, R: int = 3):
        super().__init__()

        blocks = []
        for r in range(R):
            for x in range(X):
                dilation = 2 ** x
                blocks.append(DepthwiseSeparableConv(B, H, P, dilation))

        self.network = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B_batch, B_channels, T_enc]
        return self.network(x)   # [B_batch, B_channels, T_enc]


# ── Separator ─────────────────────────────────────────────────────────────────

class Separator(nn.Module):
    """
    Full separator module: takes encoder output H, produces C masks.

    Structure:
        LayerNorm -> 1x1Conv (N->B) -> TCN -> 1x1Conv (B->N*C) -> sigmoid masks

    The masks m_c ∈ [0,1]^N are element-wise multiplied with the
    encoder output to produce per-speaker representations.

    Args:
        N (int): encoder output channels
        B (int): TCN bottleneck channels
        H (int): TCN hidden channels
        P (int): TCN kernel size
        X (int): TCN layers per repeat
        R (int): TCN repeats
        C (int): number of speakers (mask outputs)
    """

    def __init__(self, N: int = 512, B: int = 128, H: int = 512,
                 P: int = 3, X: int = 8, R: int = 3, C: int = 3):
        super().__init__()
        self.C = C
        self.N = N

        self.layer_norm = nn.GroupNorm(1, N, eps=1e-8)
        self.bottleneck = nn.Conv1d(N, B, kernel_size=1)
        self.tcn = TCN(B=B, H=H, P=P, X=X, R=R)
        self.mask_conv = nn.Conv1d(B, N * C, kernel_size=1)

    def forward(self, encoded: torch.Tensor):
        """
        Args:
            encoded: [B_batch, N, T_enc]

        Returns:
            masks:         [B_batch, C, N, T_enc]  — per-speaker masks
            tcn_features:  [B_batch, B, T_enc]     — bottleneck features
                                                     (shared with EEND branch)
        """
        # normalise and compress to bottleneck
        x = self.layer_norm(encoded)          # [B, N, T_enc]
        x = self.bottleneck(x)                # [B, B_channels, T_enc]

        # temporal modelling
        tcn_features = self.tcn(x)            # [B, B_channels, T_enc]

        # expand to C masks
        masks_flat = torch.sigmoid(
            self.mask_conv(tcn_features)      # [B, N*C, T_enc]
        )

        B_batch, _, T_enc = masks_flat.shape
        masks = masks_flat.view(B_batch, self.C, self.N, T_enc)

        return masks, tcn_features


# ── Decoder ───────────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    """
    Reconstructs waveforms from masked encoder representations.

    Input:  [B, C, N, T_enc]   — masked per-speaker representations
    Output: [B, C, T]          — separated waveforms (same length as input)

    Uses transposed conv (learnable overlap-add) as the inverse of the encoder.

    Args:
        N (int): encoder output channels (decoder input)
        L (int): filter length (must match encoder kernel size)
        stride (int): hop size (must match encoder stride)
    """

    def __init__(self, N: int = 512, L: int = 16, stride: int = 8):
        super().__init__()
        self.N = N
        self.stride = stride

        self.deconv = nn.ConvTranspose1d(
            in_channels=N,
            out_channels=1,
            kernel_size=L,
            stride=stride,
            bias=False,
        )

    def forward(self, masked: torch.Tensor) -> torch.Tensor:
        """
        Args:
            masked: [B, C, N, T_enc]

        Returns:
            waveforms: [B, C, T_out]
        """
        B, C, N, T_enc = masked.shape

        # reshape to process all speakers in one batch pass
        masked_2d = masked.view(B * C, N, T_enc)  # [B*C, N, T_enc]
        waveforms = self.deconv(masked_2d)          # [B*C, 1, T_out]
        waveforms = waveforms.squeeze(1)            # [B*C, T_out]

        T_out = waveforms.shape[-1]
        return waveforms.view(B, C, T_out)          # [B, C, T_out]


# ── Full Conv-TasNet ──────────────────────────────────────────────────────────

class ConvTasNet(nn.Module):
    """
    Complete Conv-TasNet for speech separation.

    Takes a mixed waveform, returns C separated waveforms + TCN bottleneck
    features (used by the EEND diarization branch in EEND-SS).

    Args:
        N (int): encoder filters          (paper: 512)
        L (int): encoder kernel size      (paper: 16)
        B (int): TCN bottleneck channels  (paper: 128)
        H (int): TCN hidden channels      (paper: 512)
        P (int): TCN kernel size          (paper: 3)
        X (int): TCN layers per repeat    (paper: 8)
        R (int): TCN repeats              (paper: 3)
        C (int): number of speakers       (3 for our task)
    """

    def __init__(
        self,
        N: int = 512,
        L: int = 16,
        B: int = 128,
        H: int = 512,
        P: int = 3,
        X: int = 8,
        R: int = 3,
        C: int = 3,
    ):
        super().__init__()
        self.N = N
        self.L = L
        self.C = C
        stride = L // 2   # 50% overlap between frames

        self.encoder   = Encoder(N=N, L=L, stride=stride)
        self.separator = Separator(N=N, B=B, H=H, P=P, X=X, R=R, C=C)
        self.decoder   = Decoder(N=N, L=L, stride=stride)

    def forward(self, mixture: torch.Tensor):
        """
        Args:
            mixture: [B, 1, T]

        Returns:
            separated:     [B, C, T_out]   — separated waveforms per speaker
            tcn_features:  [B, B_ch, T_enc] — bottleneck for EEND branch
            masks:         [B, C, N, T_enc] — separation masks (for analysis)
        """
        # store original length to trim output later
        T_original = mixture.shape[-1]

        # encode
        encoded = self.encoder(mixture)             # [B, N, T_enc]

        # separate: get masks and TCN features
        masks, tcn_features = self.separator(encoded)  # masks: [B, C, N, T_enc]

        # apply masks: element-wise multiply encoder output with each mask
        # encoded: [B, N, T_enc]  ->  unsqueeze to [B, 1, N, T_enc]
        masked = encoded.unsqueeze(1) * masks       # [B, C, N, T_enc]

        # decode to waveforms
        separated = self.decoder(masked)            # [B, C, T_out]

        # trim to original length (transposed conv may add a few samples)
        separated = separated[:, :, :T_original]

        return separated, tcn_features, masks

    def encode_only(self, mixture: torch.Tensor):
        """
        Returns just TCN bottleneck features.
        Used by EEND-SS when only diarization is needed (e.g. during
        the diarization-only warm-up training stage).

        Args:
            mixture: [B, 1, T]

        Returns:
            tcn_features: [B, B_ch, T_enc]
        """
        encoded = self.encoder(mixture)
        _, tcn_features = self.separator(encoded)
        return tcn_features
