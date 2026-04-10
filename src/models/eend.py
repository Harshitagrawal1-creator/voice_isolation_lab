"""
src/models/eend.py

EEND-EDA: End-to-End Neural Diarization with Encoder-Decoder Attractors.

Reference:
    Horiguchi et al., "End-to-end speaker diarization for an unknown
    number of speakers with encoder-decoder based attractors",
    Interspeech 2020.

This module takes TCN bottleneck features from Conv-TasNet (or raw
log-mel features) and produces:
    1. Speaker activity probabilities  [B, T_sub, C]   — diarization
    2. Attractor existence probs       [B, C+1]        — speaker counting

Architecture:
    subsampling conv -> Transformer encoder stack -> EDA (LSTM)
    -> dot-product with attractors -> sigmoid -> diarization probs
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Positional Encoding ───────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding (Vaswani et al. 2017).

    Adds position information to each frame in the sequence.
    The transformer's self-attention is permutation-equivariant —
    without this, it cannot distinguish frame order at all.

    Encoding formula:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        d_model (int): feature dimension
        max_len (int): maximum sequence length to pre-compute
        dropout (float): dropout applied after adding encoding
    """

    def __init__(self, d_model: int, max_len: int = 20000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # pre-compute position encoding matrix: [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)   # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ── Transformer Encoder ───────────────────────────────────────────────────────

class TransformerEncoderBlock(nn.Module):
    """
    Single transformer encoder block:
        MultiHeadAttention -> residual + LayerNorm
        FeedForward        -> residual + LayerNorm

    Args:
        d_model (int):  feature dimension
        n_heads (int):  number of attention heads
        d_ff (int):     feed-forward hidden dimension (usually 4×d_model)
        dropout (float): dropout rate
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.self_attn  = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, T, d_model]
        # self-attention with pre-norm (more stable than post-norm)
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.drop(attn_out))

        # feed-forward
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


# ── EDA: Encoder-Decoder Attractor ────────────────────────────────────────────

class EncoderDecoderAttractor(nn.Module):
    """
    LSTM-based encoder-decoder that produces attractor vectors.

    The EDA takes diarization embeddings {e_t} as input and produces
    C+1 attractor vectors. Each attractor represents one speaker slot.
    An existence probability for each attractor tells us how many
    speakers are actually present.

    During training: we use the oracle C (number of true speakers)
    During inference: we count attractors with existence prob > threshold

    Args:
        d_model (int): embedding dimension (matches transformer output)
        C_max (int):   maximum number of speakers + 1 (for existence loss)
    """

    def __init__(self, d_model: int, C_max: int = 4):
        super().__init__()
        self.C_max = C_max

        # LSTM encoder: summarises the sequence
        self.encoder_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
        )

        # LSTM decoder: generates attractor vectors one by one
        self.decoder_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
        )

        # existence probability: is this attractor a real speaker?
        self.exist_linear = nn.Linear(d_model, 1)

    def forward(self, embeddings: torch.Tensor, num_attractors: int):
        """
        Args:
            embeddings (torch.Tensor): [B, T, d_model] — transformer output
            num_attractors (int): how many attractors to generate
                                  (oracle C+1 during training, C_max during inference)

        Returns:
            attractors  (torch.Tensor): [B, num_attractors, d_model]
            exist_probs (torch.Tensor): [B, num_attractors]
        """
        B = embeddings.shape[0]

        # encode the full sequence into a context vector
        _, (h_n, c_n) = self.encoder_lstm(embeddings)
        # h_n: [1, B, d_model], c_n: [1, B, d_model]

        # decode: generate one attractor per step
        # start token: zeros
        decoder_input = torch.zeros(B, 1, embeddings.shape[-1],
                                    device=embeddings.device)
        h, c = h_n, c_n

        attractors = []
        exist_probs = []

        for _ in range(num_attractors):
            out, (h, c) = self.decoder_lstm(decoder_input, (h, c))
            # out: [B, 1, d_model]
            attractors.append(out)
            exist_probs.append(torch.sigmoid(self.exist_linear(out)))
            decoder_input = out   # next input is current output (autoregressive)

        attractors  = torch.cat(attractors, dim=1)   # [B, num_attractors, d_model]
        exist_probs = torch.cat(exist_probs, dim=1).squeeze(-1)  # [B, num_attractors]

        return attractors, exist_probs


# ── Full EEND Module ──────────────────────────────────────────────────────────

class EEND(nn.Module):
    """
    EEND-EDA diarization module.

    Takes TCN bottleneck features from Conv-TasNet and produces
    speaker activity probabilities and speaker count estimate.

    Args:
        input_dim (int):   input feature dimension = TCN bottleneck B (default 128)
        d_model (int):     transformer hidden dimension (paper: 256)
        n_heads (int):     attention heads (paper: 4)
        n_layers (int):    transformer encoder layers (paper: 4)
        d_ff (int):        feed-forward dim (4 × d_model)
        dropout (float):   dropout
        C_max (int):       max speakers + 1 (for EDA)
        subsample (int):   temporal subsampling factor (paper: 8)
        max_speakers (int): C — number of output speaker slots
    """

    def __init__(
        self,
        input_dim: int = 128,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        C_max: int = 4,
        subsample: int = 8,
        max_speakers: int = 3,
    ):
        super().__init__()
        self.C_max = C_max
        self.max_speakers = max_speakers
        self.subsample = subsample

        # project input features to d_model
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=subsample, stride=subsample),
            nn.ReLU(),
        )

        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout)

        self.transformer = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.eda = EncoderDecoderAttractor(d_model, C_max=C_max)

        self.exist_threshold = 0.5

    def forward(
        self,
        tcn_features: torch.Tensor,
        num_speakers: torch.Tensor | int | None = None,
    ):
        """
        Args:
            tcn_features (torch.Tensor): [B, B_ch, T_enc]
                TCN bottleneck features from Conv-TasNet separator.
            num_speakers (torch.Tensor | int | None):
                Per-sample oracle speaker counts [B] for training, or a single
                int, or None at inference time.

                FIX: When a tensor is passed, we use its maximum to determine
                how many attractor slots to generate for the whole batch
                (all samples need the same number of slots for batching),
                but the per-sample counts are used correctly in the loss.

        Returns:
            diar_probs  (torch.Tensor): [B, T_sub, max_speakers]
            exist_probs (torch.Tensor): [B, num_attractors]
            embeddings  (torch.Tensor): [B, T_sub, d_model]
        """
        # subsample: [B, B_ch, T_enc] -> [B, d_model, T_sub]
        x = self.input_proj(tcn_features)    # [B, d_model, T_sub]
        x = x.transpose(1, 2)               # [B, T_sub, d_model]

        # add positional encoding
        x = self.pos_enc(x)

        # transformer encoder
        for block in self.transformer:
            x = block(x)
        embeddings = x                       # [B, T_sub, d_model]

        # determine how many attractors to produce
        if num_speakers is not None:
            if isinstance(num_speakers, torch.Tensor):
                # use batch maximum + 1 for attractor slot count
                # (all samples in a batch must have same number of slots)
                n_attract = int(num_speakers.max().item()) + 1
            else:
                n_attract = int(num_speakers) + 1
        else:
            # inference: generate all possible slots
            n_attract = self.C_max

        attractors, exist_probs = self.eda(embeddings, n_attract)
        # attractors:  [B, n_attract, d_model]
        # exist_probs: [B, n_attract]

        # speaker activity: dot product of embeddings and attractors
        scores = torch.bmm(embeddings, attractors.transpose(1, 2))
        # scores: [B, T_sub, n_attract]

        # sigmoid to get per-frame probabilities, take only first max_speakers
        diar_probs = torch.sigmoid(scores[:, :, :self.max_speakers])
        # [B, T_sub, max_speakers]

        return diar_probs, exist_probs, embeddings

    def predict_num_speakers(self, exist_probs: torch.Tensor) -> torch.Tensor:
        """
        Count number of speakers from attractor existence probabilities.

        At inference: count how many existence probs exceed threshold
        before the first one that drops below it.

        Args:
            exist_probs: [B, n_attract]

        Returns:
            counts: [B] — estimated number of speakers per sample
        """
        above = (exist_probs > self.exist_threshold).float()
        counts = above.sum(dim=-1).long()
        counts = counts.clamp(min=1, max=self.max_speakers)
        return counts
