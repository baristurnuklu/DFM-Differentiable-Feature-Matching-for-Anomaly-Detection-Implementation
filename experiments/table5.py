"""
Table 5 — Component ablation on 2-shot MVTec AD.

4 variants × 15 categories, averaged over N_SEEDS seeds.
Reports I-AUROC and P-AUROC, mean across categories.

Variants:
  (A) PatchCore baseline  — no adapters, no FMN training
  (B) + Adapter           — adapters trained (Stage 1 only)
  (C) + FMN               — no adapters, FMN trained (Stage 2 only)
  (D) Full DFM            — adapters + FMN + iterative

Run from project root:
    python3.11 experiments/table5.py
"""

import sys, json
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.backbone import CLIPViTBackbone
from src.coreset import build_memory_bank
from src.dataset import MVTEC_CATEGORIES, MVTecTestDataset
from src.evaluate import evaluate_category
from src.fmn import FMN
from src.train import get_device, train_one_category, _load_train_images, _extract_flat_features

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_ROOT      = "data/mvtec"
FEW_SHOT_K     = 2
NUM_ROUNDS     = 5
STAGE1_EPOCHS  = 10
STAGE2_EPOCHS  = 10
LR             = 1e-4
SEEDS          = [42, 0, 1]
ADAPTER_LAYERS = [0, 1]
# ──────────────────────────────────────────────────────────────────────────────

def _patchcore(category, seed, device):
    dev = torch.device(device)
    train_imgs = _load_train_images(DATA_ROOT, category, FEW_SHOT_K, seed, dev)
    bb = CLIPViTBackbone(adapter_layers=None, device=device)
    feats = _extract_flat_features(bb, train_imgs)
    bank  = build_memory_bank(feats)
    fmn = FMN().to(dev)
    fmn.set_memory_bank(bank.to(dev))
    loader = DataLoader(MVTecTestDataset(DATA_ROOT, category),
                        batch_size=1, shuffle=False, num_workers=0)
    return evaluate_category(bb, fmn, loader, dev)

def _run_seed(category, adapter_layers, s1_epochs, s2_epochs, seed, device):
    bb, fmn = train_one_category(
        DATA_ROOT, category, FEW_SHOT_K, adapter_layers,
        num_rounds=NUM_ROUNDS, stage1_epochs=s1_epochs, stage2_epochs=s2_epochs,
        lr=LR, device=device, seed=seed,
    )
    loader = DataLoader(MVTecTestDataset(DATA_ROOT, category),
                        batch_size=1, shuffle=False, num_workers=0)
    m = evaluate_category(bb, fmn, loader, torch.device(device))
    del bb, fmn
    return m

def _avg(seed_metrics):
    return {
        "i_auroc": float(np.mean([s["i_auroc"] for s in seed_metrics])),
        "p_auroc": float(np.mean([s["p_auroc"] for s in seed_metrics])),
        "p_ap":    float(np.mean([s["p_ap"]    for s in seed_metrics])),
    }

def main():
    device = get_device()
    print(f"Device: {device}  |  k={FEW_SHOT_K}  |  seeds={SEEDS}\n")

    variants = ["PatchCore", "+ Adapter", "+ FMN", "Full DFM"]
    print(f"{'Category':<14}", end="")
    for name in variants:
        print(f"  {name:<14} I / P", end="")
    print()
    print("-" * (14 + len(variants) * 22))

    results = {v: {} for v in variants}

    for category in MVTEC_CATEGORIES:
        print(f"{category:<14}", end="", flush=True)

        m = _avg([_patchcore(category, s, device) for s in SEEDS])
        results["PatchCore"][category] = m
        print(f"  {m['i_auroc']:.3f} / {m['p_auroc']:.3f}      ", end="", flush=True)

        m = _avg([_run_seed(category, ADAPTER_LAYERS, STAGE1_EPOCHS, 0, s, device) for s in SEEDS])
        results["+ Adapter"][category] = m
        print(f"  {m['i_auroc']:.3f} / {m['p_auroc']:.3f}      ", end="", flush=True)

        m = _avg([_run_seed(category, [], 0, STAGE2_EPOCHS, s, device) for s in SEEDS])
        results["+ FMN"][category] = m
        print(f"  {m['i_auroc']:.3f} / {m['p_auroc']:.3f}      ", end="", flush=True)

        m = _avg([_run_seed(category, ADAPTER_LAYERS, STAGE1_EPOCHS, STAGE2_EPOCHS, s, device) for s in SEEDS])
        results["Full DFM"][category] = m
        print(f"  {m['i_auroc']:.3f} / {m['p_auroc']:.3f}      ", end="", flush=True)

        print()

    print("-" * (14 + len(variants) * 22))
    print(f"{'mean':<14}", end="")
    for name in variants:
        vals = results[name].values()
        mi = np.mean([v["i_auroc"] for v in vals])
        mp = np.mean([v["p_auroc"] for v in vals])
        print(f"  {mi:.3f} / {mp:.3f}      ", end="")
    print()

    Path("results").mkdir(exist_ok=True)
    with open("results/table5.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/table5.json")

if __name__ == "__main__":
    main()
