"""
Evaluation utilities for DFM on MVTec AD.

Metrics:
  I-AUROC : image-level anomaly detection  (roc_auc_score over image scores)
  P-AUROC : pixel-level anomaly localisation (roc_auc_score over patch scores)
  P-AP    : pixel-level average precision   (average_precision_score)

Both AUROC metrics use only ranking, so raw FMN output scores are passed
directly — no normalisation needed.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from src.dataset import MVTEC_CATEGORIES, MVTecTestDataset


def evaluate_category(
    backbone,      # CLIPViTBackbone — already trained / eval mode set inside
    fmn,           # FMN             — already trained / eval mode set inside
    test_loader,   # DataLoader over MVTecTestDataset, batch_size=1
    device,
) -> dict:
    """
    Run inference over all test images of one category and return metrics.

    Returns:
        {'i_auroc': float, 'p_auroc': float, 'p_ap': float}
        Any metric that cannot be computed (single class) is returned as NaN.
    """
    backbone.eval()
    fmn.eval()

    image_scores = []
    image_labels = []
    patch_scores = []
    patch_labels = []

    with torch.no_grad():
        for images, labels, masks, _, _ in test_loader:
            images = images.to(device)          # (B, 3, 224, 224)

            P           = backbone(images)      # (B, 196, 768)
            s, S_prime  = fmn(P)               # (B,), (B, 196)

            for i in range(images.shape[0]):
                image_scores.append(s[i].item())
                image_labels.append(int(labels[i]))

                # Upsample 14×14 score map → 224×224 (paper reports pixel-wise AUROC)
                H, W = images.shape[2], images.shape[3]
                score_map = S_prime[i].reshape(1, 1, 14, 14)
                score_map = F.interpolate(score_map, size=(H, W),
                                          mode='bilinear', align_corners=False)
                score_map = score_map.squeeze().cpu()            # (H, W)
                patch_scores.extend(score_map.reshape(-1).tolist())
                patch_labels.extend((masks[i] > 0.5).int().reshape(-1).tolist())

    image_scores = np.array(image_scores, dtype=np.float32)
    image_labels = np.array(image_labels, dtype=np.int32)
    patch_scores = np.array(patch_scores, dtype=np.float32)
    patch_labels = np.array(patch_labels, dtype=np.int32)

    def _safe_auroc(y_true, y_score):
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score))

    def _safe_ap(y_true, y_score):
        if y_true.sum() == 0:
            return float("nan")
        return float(average_precision_score(y_true, y_score))

    return {
        "i_auroc": _safe_auroc(image_labels, image_scores),
        "p_auroc": _safe_auroc(patch_labels, patch_scores),
        "p_ap":    _safe_ap(patch_labels, patch_scores),
    }


def evaluate_all_categories(
    backbone,
    fmn,
    data_root: str,
    device,
    num_workers: int = 0,
    verbose: bool = False,
) -> dict:
    """
    Evaluate all 15 MVTec categories and return per-category + mean metrics.

    Returns:
        {
          'bottle':     {'i_auroc': ..., 'p_auroc': ..., 'p_ap': ...},
          'cable':      {...},
          ...
          'mean':       {'i_auroc': ..., 'p_auroc': ..., 'p_ap': ...},
        }
        Mean ignores any NaN category (shouldn't happen on MVTec).
    """
    results = {}

    for category in MVTEC_CATEGORIES:
        test_ds     = MVTecTestDataset(data_root, category)
        test_loader = DataLoader(
            test_ds, batch_size=1, shuffle=False,
            num_workers=num_workers, pin_memory=False,
        )
        metrics = evaluate_category(backbone, fmn, test_loader, device)
        results[category] = metrics

        if verbose:
            print(
                f"  {category:<12} "
                f"I-AUROC={metrics['i_auroc']:.3f}  "
                f"P-AUROC={metrics['p_auroc']:.3f}  "
                f"P-AP={metrics['p_ap']:.3f}"
            )

    # Mean across categories (skip NaN entries)
    valid = [v for v in results.values() if not np.isnan(v["i_auroc"])]
    if valid:
        results["mean"] = {
            "i_auroc": float(np.mean([v["i_auroc"] for v in valid])),
            "p_auroc": float(np.mean([v["p_auroc"] for v in valid])),
            "p_ap":    float(np.mean([v["p_ap"]    for v in valid])),
        }

    return results
