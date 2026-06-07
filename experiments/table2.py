"""
Table 2 — DFM-PatchCore column.

Few-shot k ∈ {1, 2, 4, 8} on all 15 MVTec AD categories.
Adapter layers: {1,2} → [0,1] (paper default).
Reports I-AUROC and P-AUROC averaged over N_SEEDS seeds, mean across categories.

Run from project root:
    python3.11 experiments/table2.py
"""

import sys, time, json
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import MVTEC_CATEGORIES, MVTecTestDataset
from src.evaluate import evaluate_category
from src.train import get_device, train_one_category

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_ROOT      = "data/mvtec"
ADAPTER_LAYERS = [4, 5]          # paper {5,6} — confirmed from Table 3 ablation
K_VALUES       = [1, 2, 4, 8]
NUM_ROUNDS     = 5
STAGE1_EPOCHS  = 10
STAGE2_EPOCHS  = 10
LR             = 1e-4
SEEDS          = [42, 0, 1]     # average over 3 seeds (standard in few-shot papers)
# ──────────────────────────────────────────────────────────────────────────────

def _run_seed(category, k, seed, device):
    bb, fmn = train_one_category(
        data_root=DATA_ROOT, category=category, few_shot_k=k,
        adapter_layers=ADAPTER_LAYERS, num_rounds=NUM_ROUNDS,
        stage1_epochs=STAGE1_EPOCHS, stage2_epochs=STAGE2_EPOCHS,
        lr=LR, device=device, seed=seed,
    )
    loader = DataLoader(MVTecTestDataset(DATA_ROOT, category),
                        batch_size=1, shuffle=False, num_workers=0)
    m = evaluate_category(bb, fmn, loader, torch.device(device))
    del bb, fmn
    return m

def main():
    device = get_device()
    print(f"Device: {device}  |  seeds={SEEDS}\n")
    print(f"{'Category':<14}", end="")
    for k in K_VALUES:
        print(f"  k={k} I-AUC  k={k} P-AUC", end="")
    print()
    print("-" * (14 + len(K_VALUES) * 22))

    results = {k: {} for k in K_VALUES}

    for category in MVTEC_CATEGORIES:
        print(f"{category:<14}", end="", flush=True)

        for k in K_VALUES:
            seed_metrics = [_run_seed(category, k, s, device) for s in SEEDS]
            m = {
                "i_auroc": float(np.mean([s["i_auroc"] for s in seed_metrics])),
                "p_auroc": float(np.mean([s["p_auroc"] for s in seed_metrics])),
                "p_ap":    float(np.mean([s["p_ap"]    for s in seed_metrics])),
            }
            results[k][category] = m
            print(f"  {m['i_auroc']:.3f}       {m['p_auroc']:.3f}  ", end="", flush=True)
        print()

    print("-" * (14 + len(K_VALUES) * 22))
    print(f"{'mean':<14}", end="")
    for k in K_VALUES:
        vals = results[k].values()
        mi = np.mean([v["i_auroc"] for v in vals])
        mp = np.mean([v["p_auroc"] for v in vals])
        print(f"  {mi:.3f}       {mp:.3f}  ", end="")
    print()

    Path("results").mkdir(exist_ok=True)
    with open("results/table2.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/table2.json")

if __name__ == "__main__":
    main()
