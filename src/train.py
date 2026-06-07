"""
Two-stage iterative training for DFM (Algorithm 1 from the paper).

For one MVTec category with k training images:

  SETUP
    Build initial coreset from plain (no-adapter) backbone features.

  FOR round in range(T):                        [T=5]
    STAGE 1  train adapters with L_unsup        [E1 steps, FMN frozen]
    REBUILD  re-derive coreset from adapted features, update FMN.M
    STAGE 2  train FMN with L_sup + CutPaste    [E2 steps, adapters frozen]

Returns (backbone, fmn) in eval mode, ready for evaluate_category().
"""

import torch
from torch.optim import Adam

from src.backbone import CLIPViTBackbone
from src.coreset import build_memory_bank
from src.cutpaste import cutpaste_batch, downsample_mask
from src.dataset import MVTecTrainDataset
from src.fmn import FMN
from src.loss import supervised_loss, unsupervised_loss


def get_device() -> str:
    """Return the best available device: MPS → CUDA → CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ── Internal helpers ───────────────────────────────────────────────────────────

def _load_train_images(
    data_root: str,
    category: str,
    few_shot_k: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Return all k training images as a single (k, 3, 224, 224) tensor."""
    ds = MVTecTrainDataset(data_root, category, few_shot_k=few_shot_k, seed=seed)
    imgs = torch.stack([ds[i][0] for i in range(len(ds))])
    return imgs.to(device)


def _extract_flat_features(
    backbone: CLIPViTBackbone,
    imgs: torch.Tensor,
) -> torch.Tensor:
    """Run backbone with no grad, flatten patch dim → (k*196, 768)."""
    with torch.no_grad():
        P = backbone(imgs)                       # (k, 196, 768)
    return P.reshape(-1, P.shape[-1])            # (k*196, 768)


# ── Main training function ─────────────────────────────────────────────────────

def train_one_category(
    data_root: str,
    category: str,
    few_shot_k: int,
    adapter_layers: list,
    num_rounds: int = 5,
    stage1_epochs: int = 10,
    stage2_epochs: int = 10,
    lr: float = 1e-4,
    device: str = "auto",
    seed: int = 42,
    verbose: bool = False,
) -> tuple:
    """
    Train DFM on one MVTec category and return trained (backbone, fmn).

    Args:
        data_root      : path to the mvtec directory (contains 15 category folders)
        category       : one of MVTEC_CATEGORIES, e.g. 'bottle'
        few_shot_k     : number of support images (1, 2, 4, or 8)
        adapter_layers : 0-indexed block indices where adapters are inserted,
                         e.g. [0, 1] for paper notation {1, 2}.
                         Pass [] for the no-adapter ablation baseline.
        num_rounds     : T — outer iterative loop count (paper default 5)
        stage1_epochs  : gradient steps per Stage 1 round (paper default 10)
        stage2_epochs  : gradient steps per Stage 2 round (paper default 10)
        lr             : learning rate for Adam in both stages
        device         : 'mps', 'cuda', or 'cpu'
        seed           : selects which k images are sampled for training
        verbose        : print per-round loss values

    Returns:
        (backbone, fmn) — both in eval mode
    """
    if device == "auto":
        device = get_device()
    dev = torch.device(device)
    has_adapters = len(adapter_layers) > 0

    # ── Setup ──────────────────────────────────────────────────────────────────

    # Load all k support images onto device once
    train_imgs     = _load_train_images(data_root, category, few_shot_k, seed, dev)
    # CPU copy cached here — CutPaste needs CPU (HSV jitter), same source every epoch
    train_imgs_cpu = train_imgs.cpu()

    # Build initial coreset from a clean backbone (adapters start as identity,
    # so this is equivalent — but keeps the baseline identical to PatchCore)
    plain_bb    = CLIPViTBackbone(adapter_layers=None, device=device)
    init_feats  = _extract_flat_features(plain_bb, train_imgs)   # (k*196, 768)
    init_bank   = build_memory_bank(init_feats)                   # (196, 768)
    del plain_bb

    # Create the trainable backbone and FMN
    backbone = CLIPViTBackbone(adapter_layers=adapter_layers, device=device)
    fmn      = FMN().to(dev)
    fmn.set_memory_bank(init_bank.to(dev))

    # ── Iterative training loop ────────────────────────────────────────────────

    for rnd in range(num_rounds):

        # ── Stage 1: update adapters with L_unsup ────────────────────────────
        if has_adapters:
            backbone.unfreeze_adapters()
            fmn.freeze_memory_bank()

            opt_s1 = Adam(list(backbone.adapter_parameters()), lr=lr)

            for _ in range(stage1_epochs):
                opt_s1.zero_grad()

                P           = backbone(train_imgs)      # (k, 196, 768)
                s, S_prime  = fmn(P)                    # (k,), (k, 196)
                loss_s1     = unsupervised_loss(s, S_prime)

                loss_s1.backward()
                opt_s1.step()

            # Rebuild memory bank from the now-adapted features
            new_feats = _extract_flat_features(backbone, train_imgs)  # (k*196, 768)
            new_bank  = build_memory_bank(new_feats)                   # (196, 768)
            fmn.set_memory_bank(new_bank.to(dev))

            if verbose:
                print(
                    f"  [{category}] round {rnd+1}/{num_rounds}  "
                    f"stage1 loss={loss_s1.item():.4f}",
                    end="",
                )
        else:
            if verbose:
                print(f"  [{category}] round {rnd+1}/{num_rounds}  [no adapters]", end="")

        # ── Stage 2: update FMN with L_sup + CutPaste ────────────────────────
        if stage2_epochs == 0:
            if verbose:
                print(f"  stage2 skipped")
            continue

        backbone.freeze_adapters()
        fmn.unfreeze_memory_bank()

        opt_s2 = Adam(list(fmn.fmn_parameters()), lr=lr)

        for _ in range(stage2_epochs):
            opt_s2.zero_grad()

            # Normal branch
            P_normal           = backbone(train_imgs)          # (k, 196, 768)
            s_normal, _        = fmn(P_normal)                 # (k,)

            # Anomalous branch — CutPaste on cached CPU copy (HSV jitter requires CPU)
            aug_imgs, pix_masks = cutpaste_batch(train_imgs_cpu)
            aug_imgs            = aug_imgs.to(dev)

            P_anomaly           = backbone(aug_imgs)           # (k, 196, 768)
            s_anomaly, S_anomaly = fmn(P_anomaly)              # (k,), (k, 196)

            # Pixel masks → patch masks: max-pool 224×224 → 196
            patch_masks = torch.stack(
                [downsample_mask(pix_masks[i]) for i in range(few_shot_k)]
            ).to(dev)                                          # (k, 196)

            loss_s2 = supervised_loss(s_normal, s_anomaly, S_anomaly, patch_masks)
            loss_s2.backward()
            opt_s2.step()

        if verbose:
            print(f"  stage2 loss={loss_s2.item():.4f}")

    backbone.eval()
    fmn.eval()
    return backbone, fmn
