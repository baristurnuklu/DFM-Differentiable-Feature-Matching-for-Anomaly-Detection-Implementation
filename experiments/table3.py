"""
Table 3 — Adapter insertion layer ablation.

Adapter configs × k ∈ {1, 2} on all 15 MVTec AD categories.
Reports I-AUROC and P-AUROC averaged over N_SEEDS seeds, mean across categories.

Run from project root:
    python3.11 experiments/table3.py
"""

import sys, json
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import MVTEC_CATEGORIES, MVTecTestDataset
from src.evaluate import evaluate_category
from src.train import get_device, train_one_category

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_ROOT     = "data/mvtec"
K_VALUES      = [1, 2]
NUM_ROUNDS    = 5
STAGE1_EPOCHS = 10
STAGE2_EPOCHS = 10
LR            = 1e-4
SEEDS         = [42, 0, 1]

ADAPTER_CONFIGS = {
    "{1,2}": [0, 1],
    "{3,4}": [2, 3],
    "{5,6}": [4, 5],
}
# ──────────────────────────────────────────────────────────────────────────────

def _run_seed(category, k, layers, seed, device):
    bb, fmn = train_one_category(
        data_root=DATA_ROOT, category=category, few_shot_k=k,
        adapter_layers=layers, num_rounds=NUM_ROUNDS,
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

    col_labels = [f"{name} k={k}" for name in ADAPTER_CONFIGS for k in K_VALUES]
    print(f"{'Category':<14}", end="")
    for label in col_labels:
        print(f"  {label:<18}", end="")
    print()
    print("-" * (14 + len(col_labels) * 20))

    results = {name: {k: {} for k in K_VALUES} for name in ADAPTER_CONFIGS}

    for category in MVTEC_CATEGORIES:
        print(f"{category:<14}", end="", flush=True)

        for name, layers in ADAPTER_CONFIGS.items():
            for k in K_VALUES:
                seed_metrics = [_run_seed(category, k, layers, s, device) for s in SEEDS]
                m = {
                    "i_auroc": float(np.mean([s["i_auroc"] for s in seed_metrics])),
                    "p_auroc": float(np.mean([s["p_auroc"] for s in seed_metrics])),
                    "p_ap":    float(np.mean([s["p_ap"]    for s in seed_metrics])),
                }
                results[name][k][category] = m
                print(f"  {m['i_auroc']:.3f}/{m['p_auroc']:.3f}      ", end="", flush=True)
        print()

    print("-" * (14 + len(col_labels) * 20))
    print(f"{'mean':<14}", end="")
    for name in ADAPTER_CONFIGS:
        for k in K_VALUES:
            vals = results[name][k].values()
            mi = np.mean([v["i_auroc"] for v in vals])
            mp = np.mean([v["p_auroc"] for v in vals])
            print(f"  {mi:.3f}/{mp:.3f}      ", end="")
    print()

    Path("results").mkdir(exist_ok=True)
    with open("results/table3.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/table3.json")

if __name__ == "__main__":
    main()
