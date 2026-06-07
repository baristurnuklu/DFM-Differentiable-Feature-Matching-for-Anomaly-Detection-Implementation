"""
Loss functions for DFM two-stage training.

Stage 1 (adapters, no real anomalies): L_unsup
Stage 2 (FMN, CutPaste anomalies):     L_sup

Both use a dynamic reference score s* = β · max(scores_in_batch),
computed fresh from model outputs each forward pass so gradients flow
back through s* to the model (no .detach()).
"""

import torch
import torch.nn.functional as F


def unsupervised_loss(
    s_normal: torch.Tensor,  # (Nn,)    image-level scores for normal images
    S_normal: torch.Tensor,  # (Nn, N)  patch-level scores for normal images
    beta: float = 0.9,
) -> torch.Tensor:
    """
    Stage 1 loss — only normal images available.

    Penalises normal patches/images that score above β·max(batch score),
    compressing the normal score range so anomalies stand out at test time.

    L_unsup = mean(relu(s_normal - s*)) + mean(relu(S_normal - s*_seg))
    """
    s_star     = beta * s_normal.max()   # image-level reference  (scalar)
    s_star_seg = beta * S_normal.max()   # patch-level reference   (scalar)

    loss_img = F.relu(s_normal - s_star).mean()
    loss_seg = F.relu(S_normal - s_star_seg).mean()

    return loss_img + loss_seg


def supervised_loss(
    s_normal:  torch.Tensor,  # (Nn,)    image-level scores for normal images
    s_anomaly: torch.Tensor,  # (Na,)    image-level scores for CutPaste images
    S_anomaly: torch.Tensor,  # (Na, N)  patch-level scores for CutPaste images
    masks:     torch.Tensor,  # (Na, N)  binary patch masks from downsample_mask()
    beta:       float = 0.9,
    lambda_det: float = 1.0,
    lambda_seg: float = 1.0,
) -> torch.Tensor:
    """
    Stage 2 loss — normal + CutPaste anomaly images.

    L_detec = mean(relu(s* - s_anomaly))        ← hinge: pull anomaly scores up
            + mean(|s_normal - μ_norm|)          ← L1:    cluster normal scores
    L_seg   = mean(mask · relu(s*_seg - S_anom)) ← hinge: push masked patch scores up
    L_sup   = λ_det · L_detec + λ_seg · L_seg
    """
    # Reference scores — kept in graph so gradients flow through them
    s_star     = beta * s_anomaly.max()   # image-level reference  (scalar)
    s_star_seg = beta * S_anomaly.max()   # patch-level reference   (scalar)

    # Detection branch
    l_anomaly = F.relu(s_star - s_anomaly).mean()
    l_normal  = (s_normal - s_normal.mean()).abs().mean()
    l_detec   = l_anomaly + l_normal

    # Segmentation branch — only masked (anomalous) patches penalised
    l_seg = (masks * F.relu(s_star_seg - S_anomaly)).mean()

    return lambda_det * l_detec + lambda_seg * l_seg
