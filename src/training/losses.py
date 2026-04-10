"""
src/training/losses.py

Loss functions for EEND-SS joint training.

Three losses combined (Equation 19 from paper):
    L = λ1 * L_SI-SDR  +  λ2 * L_diar  +  λ3 * L_exist

1. SI-SDR loss with PIT (separation)
   - SI-SDR: Scale-Invariant Signal-to-Distortion Ratio
   - PIT: Permutation Invariant Training — tries all speaker orderings,
     picks the one with lowest loss. Solves the "which output is which speaker" problem.

2. Diarization BCE loss
   - Binary cross-entropy between predicted activity probs and frame-level labels
   - Also uses PIT to match predicted speaker order to label order

3. Attractor existence loss
   - BCE on whether each attractor slot contains a real speaker
   - Enables speaker counting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations


# ── SI-SDR ────────────────────────────────────────────────────────────────────

def si_sdr(estimate: torch.Tensor, target: torch.Tensor,
           eps: float = 1e-8) -> torch.Tensor:
    """
    Scale-Invariant Signal-to-Distortion Ratio for a single speaker pair.

    SI-SDR = 10 * log10( ||s_target||^2 / ||e_noise||^2 )

    where:
        s_target = (<estimate, target> / ||target||^2) * target  (projection)
        e_noise  = estimate - s_target

    Higher is better. Typical range: -10 to +20 dB.
    Untrained model: ~0 dB. Good separation: >10 dB.

    Args:
        estimate: [B, T] or [T]  — model's separated signal
        target:   [B, T] or [T]  — clean reference signal

    Returns:
        si_sdr_val: [B] or scalar
    """
    # zero-mean both signals (SI-SDR is scale invariant, not mean invariant)
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target   = target   - target.mean(dim=-1, keepdim=True)

    # projection of estimate onto target
    dot = (estimate * target).sum(dim=-1, keepdim=True)            # [B, 1]
    target_energy = (target * target).sum(dim=-1, keepdim=True) + eps
    s_target = (dot / target_energy) * target                      # [B, T]

    # noise component
    e_noise = estimate - s_target                                   # [B, T]

    # SI-SDR in dB
    signal_power = (s_target * s_target).sum(dim=-1) + eps
    noise_power  = (e_noise  * e_noise ).sum(dim=-1) + eps

    return 10.0 * torch.log10(signal_power / noise_power)          # [B]


# ── PIT: Permutation Invariant Training ───────────────────────────────────────

def pit_si_sdr_loss(
    estimates: torch.Tensor,
    targets: torch.Tensor,
    num_speakers: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    SI-SDR loss with Permutation Invariant Training.

    For each sample in the batch, tries all C! permutations of speaker outputs
    and selects the permutation that maximises SI-SDR (minimises loss).

    This is necessary because Conv-TasNet outputs speakers in arbitrary order —
    output slot 0 might be speaker A in one sample and speaker B in another.
    PIT ensures we always compare the right estimate to the right reference.

    Args:
        estimates:    [B, C, T]  — model's separated waveforms
        targets:      [B, C, T]  — clean reference waveforms (zero-padded for missing speakers)
        num_speakers: [B]        — actual number of speakers per sample

    Returns:
        loss:      scalar — mean negative SI-SDR across batch (lower = better)
        mean_sisnr: scalar — mean SI-SDR in dB (higher = better, for logging)
    """
    B, C, T = estimates.shape
    device = estimates.device

    total_loss = torch.tensor(0.0, device=device)
    total_sisnr = torch.tensor(0.0, device=device)

    for b in range(B):
        n_spk = int(num_speakers[b])
        n_spk = max(1, min(n_spk, C))
        # only compute loss over active speaker slots
        est = estimates[b, :n_spk]    # [n_spk, T]
        tgt = targets[b,   :n_spk]    # [n_spk, T]

        best_loss = None

        for perm in permutations(range(n_spk)):
            perm_est = est[list(perm)]              # reorder estimates
            # SI-SDR for each speaker in this permutation
            sisnr_vals = si_sdr(perm_est, tgt)      # [n_spk]
            perm_loss  = -sisnr_vals.mean()         # negative because we minimise

            if best_loss is None or perm_loss < best_loss:
                best_loss    = perm_loss
                best_sisnr   = sisnr_vals.mean()

        total_loss  = total_loss  + best_loss
        total_sisnr = total_sisnr + best_sisnr

    loss      = total_loss  / B
    mean_sisnr = total_sisnr / B

    return loss, mean_sisnr


# ── Diarization BCE Loss with PIT ─────────────────────────────────────────────

def pit_diarization_loss(
    diar_probs: torch.Tensor,
    labels: torch.Tensor,
    num_speakers: torch.Tensor,
    lengths_f: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Binary cross-entropy diarization loss with permutation invariance.

    The diarization output is also permutation-ambiguous — the model might
    assign speaker A's activity to output slot 1 and speaker B to slot 0.
    We again try all permutations and pick the best match.

    Args:
        diar_probs:   [B, T_sub, C]  — predicted speaker activity (0-1)
        labels:       [B, T_f, C]    — ground truth binary labels
        num_speakers: [B]            — actual speaker count per sample
        lengths_f:    [B] | None     — valid frame lengths (for padding mask)

    Returns:
        loss: scalar BCE loss
    """
    B, T_sub, C = diar_probs.shape
    device = diar_probs.device

    # resample labels from T_f to T_sub to match diarization output resolution
    # labels: [B, T_f, C] -> [B, C, T_f] -> interpolate -> [B, C, T_sub] -> [B, T_sub, C]
    labels_t = labels.permute(0, 2, 1).float()                      # [B, C, T_f]
    labels_resampled = F.interpolate(
        labels_t.cpu(), size=T_sub, mode="nearest"
    ).to(device)                                                     # [B, C, T_sub]
    labels_resampled = labels_resampled.permute(0, 2, 1)            # [B, T_sub, C]

    total_loss = torch.tensor(0.0, device=device)

    for b in range(B):
        n_spk = max(1, min(int(num_speakers[b].item()), C))

        pred = diar_probs[b, :, :n_spk]           # [T_sub, n_spk]
        tgt  = labels_resampled[b, :, :n_spk]     # [T_sub, n_spk]

        best_loss = None
        for perm in permutations(range(n_spk)):
            perm_pred = pred[:, list(perm)]        # [T_sub, n_spk]
            perm_loss = F.binary_cross_entropy(
                perm_pred, tgt, reduction="mean"
            )
            if best_loss is None or perm_loss < best_loss:
                best_loss = perm_loss

        total_loss = total_loss + best_loss

    return total_loss / B


# ── Attractor Existence Loss ──────────────────────────────────────────────────

def existence_loss(
    exist_probs: torch.Tensor,
    num_speakers: torch.Tensor,
) -> torch.Tensor:
    """
    BCE loss for attractor existence (speaker counting).

    Ground truth: first num_speakers slots = 1 (speaker exists),
                  last slot = 0 (no more speakers).

    This matches Equation 7 from the paper:
        l_c = 1 if c <= C else 0

    Args:
        exist_probs:  [B, C+1]  — predicted existence probabilities
        num_speakers: [B]       — true speaker count per sample

    Returns:
        loss: scalar
    """
    B, n_slots = exist_probs.shape
    device = exist_probs.device

    # build ground truth existence labels
    # for a 3-speaker sample with C_max=4: [1, 1, 1, 0]
    # for a 2-speaker sample with C_max=4: [1, 1, 0, 0]  (wait — paper uses [1,1,0] only up to C+1)
    targets = torch.zeros(B, n_slots, device=device)
    for b in range(B):
        n = min(int(num_speakers[b].item()), n_slots - 1)
        targets[b, :n] = 1.0
        # last slot (index n) stays 0 = "no more speakers"

    return F.binary_cross_entropy(exist_probs, targets)


# ── Joint EEND-SS Loss ────────────────────────────────────────────────────────

class EENDSSLoss(nn.Module):
    """
    Combined loss for EEND-SS joint training (Equation 19):

        L = λ1 * L_SI-SDR  +  λ2 * L_diar  +  λ3 * L_exist

    Paper uses: λ1=1.0, λ2=0.2, λ3=0.2

    Args:
        lambda_sisnr (float): weight for separation loss (default 1.0)
        lambda_diar  (float): weight for diarization loss (default 0.2)
        lambda_exist (float): weight for existence loss (default 0.2)
    """

    def __init__(
        self,
        lambda_sisnr: float = 1.0,
        lambda_diar:  float = 0.2,
        lambda_exist: float = 0.2,
    ):
        super().__init__()
        self.lambda_sisnr = lambda_sisnr
        self.lambda_diar  = lambda_diar
        self.lambda_exist = lambda_exist

    def forward(
        self,
        separated:    torch.Tensor,
        diar_probs:   torch.Tensor,
        exist_probs:  torch.Tensor,
        sources:      torch.Tensor,
        labels:       torch.Tensor,
        num_speakers: torch.Tensor,
        lengths_f:    torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            separated:    [B, C, T]       — model's separated waveforms
            diar_probs:   [B, T_sub, C]   — diarization activity probs
            exist_probs:  [B, C+1]        — attractor existence probs
            sources:      [B, C, T]       — clean reference waveforms
            labels:       [B, T_f, C]     — ground truth diarization labels
            num_speakers: [B]             — true speaker counts
            lengths_f:    [B] | None      — valid frame lengths

        Returns:
            dict with keys:
                'loss'        — total weighted loss (backprop on this)
                'loss_sisnr'  — separation component
                'loss_diar'   — diarization component
                'loss_exist'  — existence component
                'si_snr_db'   — mean SI-SNR in dB (for logging)
        """
        # ── separation loss ────────────────────────────────────────────────────
        l_sisnr, mean_sisnr = pit_si_sdr_loss(separated, sources, num_speakers)

        # ── diarization loss ───────────────────────────────────────────────────
        l_diar = pit_diarization_loss(diar_probs, labels, num_speakers, lengths_f)

        # ── existence loss ─────────────────────────────────────────────────────
        l_exist = existence_loss(exist_probs, num_speakers)

        # ── weighted sum ───────────────────────────────────────────────────────
        total = (
            self.lambda_sisnr * l_sisnr
            + self.lambda_diar  * l_diar
            + self.lambda_exist * l_exist
        )

        return {
            "loss":       total,
            "loss_sisnr": l_sisnr.detach(),
            "loss_diar":  l_diar.detach(),
            "loss_exist": l_exist.detach(),
            "si_snr_db":  mean_sisnr.detach(),
        }
