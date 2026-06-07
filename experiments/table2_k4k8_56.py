"""
Table 2 — k=4 and k=8 with adapter layers {5,6} = [4,5] (paper default).

Runs only the two missing k values and merges them into results/table2.json.

Run from project root:
    python3.11 experiments/table2_k4k8_56.py
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
ADAPTER_LAYERS = [4, 5]          # paper {5,6}, 0-indexed
K_VALUES       = [4, 8]          # only the two missing values
NUM_ROUNDS     = 5
STAGE1_EPOCHS  = 10
STAGE2_EPOCHS  = 10
LR             = 1e-4
SEEDS          = [42, 0, 1]
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
    print(f"Device: {device}  |  adapter_layers={ADAPTER_LAYERS}  |  k={K_VALUES}  |  seeds={SEEDS}\n")

    # Load existing table2.json (already has k=1, k=2 from {5,6})
    result_path = Path("results/table2.json")
    with open(result_path) as f:
        results = json.load(f)

    for k in K_VALUES:
        print(f"\n{'='*60}")
        print(f"  k = {k}")
        print(f"{'='*60}")
        results[k] = {}

        for category in MVTEC_CATEGORIES:
            print(f"  {category:<14}", end="", flush=True)
            t0 = time.time()

            seed_metrics = [_run_seed(category, k, s, device) for s in SEEDS]
            m = {
                "i_auroc": float(np.mean([s["i_auroc"] for s in seed_metrics])),
                "p_auroc": float(np.mean([s["p_auroc"] for s in seed_metrics])),
                "p_ap":    float(np.mean([s["p_ap"]    for s in seed_metrics])),
            }
            results[k][category] = m
            print(f"  I={m['i_auroc']:.3f}  P={m['p_auroc']:.3f}  ({time.time()-t0:.0f}s)",
                  flush=True)

        vals = results[k].values()
        mi = np.mean([v["i_auroc"] for v in vals])
        mp = np.mean([v["p_auroc"] for v in vals])
        print(f"  {'mean':<14}  I={mi:.3f}  P={mp:.3f}")

        # Save after each k so progress is preserved
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved to {result_path}")

    print("\nFinal means (all k, adapter {5,6}):")
    for k in ["1", "2", "4", "8"]:
        vals = results[str(k)].values()
        mi = np.mean([v["i_auroc"] for v in vals])
        mp = np.mean([v["p_auroc"] for v in vals])
        print(f"  k={k}: I={mi:.3f}  P={mp:.3f}")

if __name__ == "__main__":
    main()
