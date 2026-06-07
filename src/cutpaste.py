"""
CutPaste augmentation for synthetic anomaly generation.

Used in the few-shot setting where no real anomaly samples exist.
Takes a normal image, cuts a patch from it, applies mild colour jitter,
and pastes it at a random different location.  The paste region becomes
the synthetic "anomaly" and its bounding box is the pixel-level GT mask.

Reference: Li et al., "CutPaste: Self-supervised Learning for Anomaly
Detection and Localization", CVPR 2021.

Interface
---------
cutpaste(img)       : single (3, H, W) CLIP-normalised tensor
cutpaste_batch(imgs): batch  (B, 3, H, W)
downsample_mask(mask): pixel mask (H, W) → patch mask (N,) for L_seg
"""

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from typing import Dict, Tuple


# CLIP normalisation constants (ViT-B/16, OpenAI)
_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275,  0.40821073])
_CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711])


# ── Patch-level normalisation helpers ─────────────────────────────────────────
# Only applied to the extracted patch, not the full image, to avoid clamping
# artefacts on the regions we don't touch.

def _patch_to_unit(patch: torch.Tensor) -> torch.Tensor:
    """CLIP-normalised patch (3,h,w) → [0,1], clamped for jitter ops."""
    mean = _CLIP_MEAN.to(patch.device).view(3, 1, 1)
    std  = _CLIP_STD.to(patch.device).view(3, 1, 1)
    return (patch * std + mean).clamp(0.0, 1.0)


def _patch_to_clip(patch: torch.Tensor) -> torch.Tensor:
    """[0,1] patch (3,h,w) → CLIP-normalised."""
    mean = _CLIP_MEAN.to(patch.device).view(3, 1, 1)
    std  = _CLIP_STD.to(patch.device).view(3, 1, 1)
    return (patch - mean) / std


# ── Colour jitter ──────────────────────────────────────────────────────────────

def _jitter_patch(patch: torch.Tensor,
                  brightness: float = 0.1,
                  contrast:   float = 0.1,
                  saturation: float = 0.1,
                  hue:        float = 0.05) -> torch.Tensor:
    """
    Apply random colour jitter to a (3, h, w) patch in [0, 1].
    Runs on CPU for portability (MPS does not support HSV ops).
    """
    p = patch.cpu()

    b = float(torch.empty(1).uniform_(max(0.0, 1 - brightness), 1 + brightness))
    c = float(torch.empty(1).uniform_(max(0.0, 1 - contrast),   1 + contrast))
    s = float(torch.empty(1).uniform_(max(0.0, 1 - saturation), 1 + saturation))
    h = float(torch.empty(1).uniform_(-hue, hue))

    p = TF.adjust_brightness(p, b)
    p = TF.adjust_contrast(p, c)
    p = TF.adjust_saturation(p, s)
    p = TF.adjust_hue(p, h)

    return p.clamp(0.0, 1.0).to(patch.device)


# ── Patch-size sampling ────────────────────────────────────────────────────────

def _sample_patch_hw(H: int, W: int,
                     area_range:  Tuple[float, float],
                     ratio_range: Tuple[float, float]) -> Tuple[int, int]:
    """
    Sample patch (height, width) from area and aspect-ratio distributions.

    area  ~ Uniform(area_range)   as fraction of H*W
    ratio ~ Uniform(ratio_range)  = w / h

    Returned h, w are clamped to [1, H] and [1, W] respectively.
    """
    area  = H * W * float(torch.empty(1).uniform_(*area_range))
    ratio = float(torch.empty(1).uniform_(*ratio_range))   # w / h

    h = max(1, min(int(round((area / ratio) ** 0.5)), H))
    w = max(1, min(int(round((area * ratio) ** 0.5)), W))
    return h, w


# ── Core function ──────────────────────────────────────────────────────────────

def cutpaste(
    img: torch.Tensor,
    area_range:  Tuple[float, float] = (0.02, 0.15),
    ratio_range: Tuple[float, float] = (0.3,  1.0),
    jitter_kwargs: Dict[str, float]  = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply CutPaste to a single normalised image.

    Args:
        img          : (3, H, W) CLIP-normalised tensor
        area_range   : min/max patch area as fraction of image area
        ratio_range  : min/max patch aspect ratio (w/h)
        jitter_kwargs: kwargs passed to _jitter_patch

    Returns:
        aug_img : (3, H, W) CLIP-normalised
        mask    : (H, W)  binary float — 1 inside paste region, 0 elsewhere
    """
    if jitter_kwargs is None:
        jitter_kwargs = dict(brightness=0.1, contrast=0.1,
                             saturation=0.1, hue=0.05)

    _, H, W = img.shape
    device  = img.device

    # Sample patch dimensions
    ph, pw = _sample_patch_hw(H, W, area_range, ratio_range)

    # Cut: extract patch in CLIP space, convert to [0,1] only for jitter
    cy = int(torch.randint(0, H - ph + 1, (1,)))
    cx = int(torch.randint(0, W - pw + 1, (1,)))
    patch = _patch_to_unit(img[:, cy:cy + ph, cx:cx + pw].clone())

    # Colour jitter in [0,1], then back to CLIP space
    patch = _patch_to_clip(_jitter_patch(patch, **jitter_kwargs))

    # Paste: start from the original image — outside paste region stays identical
    py = int(torch.randint(0, H - ph + 1, (1,)))
    px = int(torch.randint(0, W - pw + 1, (1,)))
    aug = img.clone()
    aug[:, py:py + ph, px:px + pw] = patch

    # Binary pixel mask
    mask = torch.zeros(H, W, device=device)
    mask[py:py + ph, px:px + pw] = 1.0

    return aug, mask


# ── Batch wrapper ──────────────────────────────────────────────────────────────

def cutpaste_batch(
    imgs: torch.Tensor,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply CutPaste independently to every image in a batch.

    Args:
        imgs : (B, 3, H, W) CLIP-normalised

    Returns:
        aug_imgs : (B, 3, H, W)
        masks    : (B, H, W)
    """
    aug_imgs, masks = [], []
    for img in imgs:
        aug_img, mask = cutpaste(img, **kwargs)
        aug_imgs.append(aug_img)
        masks.append(mask)
    return torch.stack(aug_imgs), torch.stack(masks)


# ── Mask downsampling ──────────────────────────────────────────────────────────

def downsample_mask(mask: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    """
    Downsample a pixel-level mask to patch-level using max-pooling.
    A patch is marked anomalous if ANY pixel inside it is masked.

    Args:
        mask       : (H, W) binary float
        patch_size : spatial size of one ViT patch in pixels (16 for ViT-B/16)

    Returns:
        patch_mask : (N,)  where N = (H // patch_size) * (W // patch_size)
    """
    out = F.max_pool2d(
        mask.unsqueeze(0).unsqueeze(0),        # (1, 1, H, W)
        kernel_size=patch_size,
        stride=patch_size,
    )                                           # (1, 1, H/p, W/p)
    return out.squeeze().reshape(-1)            # (N,)
