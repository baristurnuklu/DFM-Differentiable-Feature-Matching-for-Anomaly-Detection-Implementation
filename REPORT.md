# DFM: Differentiable Feature Matching for Anomaly Detection
## Implementation Report

**Paper:** Wu et al., "DFM: Differentiable Feature Matching for Anomaly Detection," CVPR 2025

---

## 1. Paper Overview

### Problem
Standard anomaly detection methods (e.g., PatchCore) extract features with a frozen backbone and compare them to a memory bank using nearest-neighbour search. This pipeline is **non-differentiable** — the argmin that finds the nearest neighbour cannot be backpropagated through. The feature extractor and the matching algorithm are therefore trained separately, causing a mismatch between what the features encode and what the matching needs.

### Key Idea
DFM replaces the discrete nearest-neighbour search with a **differentiable approximation** using a Feature Matching Network (FMN). Instead of:

```
S_{i,j} = min_{m∈M} ||P_{i,j} - m||₂     (discrete argmin, non-differentiable)
s       = max_{i,j} S_{i,j}
```

DFM computes:

```
Sim_{i,j,k} = ||P_{i,j} - M_k||₂          (full distance matrix — paper Eq. 3)
S           = MinPooling(Sim)              (paper Eq. 4)
s           = MaxPooling(S)               (paper Eq. 5)
```

Two learnable modulate layers (identity-initialised) are inserted before each pooling operation, making the full pipeline end-to-end differentiable.

### Two-Stage Iterative Training
Jointly training everything at once causes instability (paper Table 5: joint training = 84.27% I-AUROC, *worse* than the 85.43% frozen baseline). The paper uses alternating stages:

- **Stage 1 (L_unsup):** Train adapters inserted into the backbone. Only normal images. Compresses normal score distribution.
- **Stage 2 (L_sup):** Train FMN (modulate layers + memory bank M). CutPaste synthetic anomalies. Pulls anomaly scores up, clusters normal scores.
- Repeat for T rounds.

---

## 2. Architecture

### Backbone
- **Model:** CLIP ViT-B/16 (`ViT-B-16-quickgelu`, OpenAI pretrained weights) — paper Section 4.1
- **Feature extraction:** Output of the **6th transformer block** (0-indexed: block 5) → patch tokens only, CLS dropped
- **Output shape:** (B, 196, 768) — 14×14 spatial grid, 768-dimensional features
- **Backbone weights:** Always frozen (only adapters are trained)
- **Feature normalisation:** L2-normalised along the feature dimension (following PatchCore convention, cited as [1] in the paper)

### Adapters
- **Architecture:** `nn.Linear(768, 768)`, bias included
- **Initialisation:** Weight = Identity matrix, bias = zero → adapter(x) = x at init (paper: "initialized as identity transformation")
- **Placement:** Sequential after chosen transformer blocks: `output = adapter(block(x))`
- **Trained in:** Stage 1 only
- **Default for Table 2:** Blocks **{5,6}** = [4,5] (0-indexed) — confirmed from Table 3: row {5,6} gives 93.0/96.3 at k=1, matching Table 2's DFM-PatchCore column exactly
- **Table 5 ablation:** Blocks **{1,2}** = [0,1] — explicitly stated in paper Section 4.5

### Feature Matching Network (FMN)

```
Input P: (B, 196, 768)
          ↓
Matching layer — distance matrix: Sim[b,i,k] = ||P[b,i] - M[k]||₂     (B, 196, 196)
          ↓
modulate1: Linear(196→196), identity init                               (B, 196, 196)
          ↓
MinPool over memory axis (dim=-1)                                       (B, 196)
          ↓
modulate2: Linear(196→196), identity init                               (B, 196)
          ↓
MaxPool over spatial axis (dim=-1)                                      (B,)
```

- Memory bank **M**: (196, 768) — treated as `nn.Parameter`, trainable in Stage 2
- At initialisation: FMN output = PatchCore output exactly (modulate layers are identity, M is the coreset)

### Memory Bank Construction
- **Method:** Greedy farthest-point coreset sampling (following PatchCore [37] and UCAD [1])
- **Size:** 196 × 768 (fixed — paper Section 4.1: "default size of the memory bank is set to 196 × 768")
- **Rebuilt** after each Stage 1 round from the updated (adapted) backbone features

---

## 3. Training Procedure

### Loss Functions

**Stage 1 — Unsupervised loss (normal images only, paper Eq. 9):**

```
s*      = β · max(s_normal)          β = 0.9
s*_seg  = β · max(S_normal)

L_unsup = (1/Nₙ) · Σ max(0, sⁱₙ - s*) + (1/(Nₙ·HW)) · Σ max(0, Sⁱₙ - s*_seg)
```

Penalises any normal image/patch that scores above 90% of the batch maximum — compresses the normal score range.

**Stage 2 — Supervised loss (normal + CutPaste anomalies, paper Eqs. 6–8):**

```
s*      = β · max(s_anomaly)
s*_seg  = β · max(S_anomaly)

L_detec = (1/Nₐ) · Σ max(0, s* - sⁱₐ)          ← hinge: pull anomaly scores up
        + (1/Nₙ) · Σ |sⁱₙ - μ_norm|              ← L1: cluster normal scores tightly

L_seg   = (1/(Nₐ·HW)) · Σ Mⁱₐ · max(0, s*_seg - Sⁱₐ)   ← hinge on masked patches only

L_sup   = λ_det · L_detec + λ_seg · L_seg        (λ_det = λ_seg = 1.0)
```

### CutPaste Augmentation (paper Section 3.3, following Li et al. CVPR 2021 [23])
- Cut a patch from a normal training image (area: 2–15% of image, aspect ratio: 0.3–1.0)
- Apply mild colour jitter (brightness ±10%, contrast ±10%, saturation ±10%, hue ±5%)
- Paste at a different location → binary pixel mask marks the pasted region
- Patch mask for L_seg: max-pool 224×224 pixel mask → 14×14 patch grid (196 values)

### Hyperparameters

| Parameter | Value | Source |
|---|---|---|
| Backbone | CLIP ViT-B/16-quickgelu | Paper Section 4.1 |
| Feature layer | Block 6 (1-indexed) = block 5 (0-indexed) | Paper Section 4.1 |
| Memory bank size | 196 × 768 | Paper Section 4.1 |
| Adapter layers (Table 2 default) | {5,6} = [4,5] | Confirmed from Paper Table 3 |
| Adapter layers (Table 5 ablation) | {1,2} = [0,1] | Paper Section 4.5 |
| Rounds T | 5 | **Not stated in paper** — our assumption |
| Stage 1 epochs E1 | 10 | **Not stated in paper** — our assumption |
| Stage 2 epochs E2 | 10 | **Not stated in paper** — our assumption |
| β (reference score scaling) | 0.9 | **Not stated in paper** — assumed from UCAD [1] |
| Learning rate | 1e-4 (Adam) | **Not stated in paper** — standard default |
| λ_detec, λ_seg | 1.0, 1.0 | **Not stated in paper** — equal weighting |
| Seeds averaged | 3 (seeds: 42, 0, 1) | **Not stated in paper** — practical constraint |

---

## 4. Evaluation

- **I-AUROC:** Image-level AUROC. One anomaly score per test image (MaxPool output s).
- **P-AUROC:** Pixel-wise AUROC. The 14×14 patch score map is **upsampled to 224×224 via bilinear interpolation**, then compared against the full-resolution GT mask pixel-by-pixel.
- **P-AP:** Pixel-wise average precision (precision-recall AUC).

All metrics averaged over 3 random seeds per (category, k) combination.

---

## 5. Results

### Table 2 — Few-shot I-AUROC / P-AUROC on MVTec AD (adapter {5,6})

*All k values use adapter config {5,6} = [4,5], matching the paper's Table 2 default. Averaged over 3 seeds.*

#### I-AUROC

| Category | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|
| bottle | 0.620 | 0.728 | 0.514 | 0.492 |
| cable | 0.515 | 0.567 | 0.595 | 0.475 |
| capsule | 0.538 | 0.530 | 0.538 | 0.604 |
| carpet | 0.712 | 0.683 | 0.641 | 0.534 |
| grid | 0.482 | 0.447 | 0.461 | 0.471 |
| hazelnut | 0.389 | 0.539 | 0.473 | 0.314 |
| leather | 0.873 | 0.794 | 0.686 | 0.669 |
| metal_nut | 0.574 | 0.586 | 0.624 | 0.607 |
| pill | 0.538 | 0.590 | 0.572 | 0.595 |
| screw | 0.463 | 0.392 | 0.388 | 0.868 |
| tile | 0.686 | 0.629 | 0.596 | 0.533 |
| toothbrush | 0.555 | 0.453 | 0.511 | 0.533 |
| transistor | 0.472 | 0.662 | 0.718 | 0.664 |
| wood | 0.557 | 0.626 | 0.622 | 0.496 |
| zipper | 0.621 | 0.636 | 0.734 | 0.622 |
| **mean** | **0.573** | **0.591** | **0.578** | **0.565** |

#### P-AUROC

| Category | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|
| bottle | 0.888 | 0.796 | 0.814 | 0.751 |
| cable | 0.792 | 0.830 | 0.814 | 0.756 |
| capsule | 0.928 | 0.752 | 0.700 | 0.652 |
| carpet | 0.884 | 0.882 | 0.838 | 0.769 |
| grid | 0.575 | 0.621 | 0.613 | 0.603 |
| hazelnut | 0.681 | 0.832 | 0.791 | 0.626 |
| leather | 0.965 | 0.912 | 0.856 | 0.789 |
| metal_nut | 0.873 | 0.875 | 0.799 | 0.786 |
| pill | 0.891 | 0.868 | 0.851 | 0.818 |
| screw | 0.757 | 0.827 | 0.706 | 0.471 |
| tile | 0.816 | 0.762 | 0.762 | 0.722 |
| toothbrush | 0.896 | 0.883 | 0.885 | 0.618 |
| transistor | 0.700 | 0.678 | 0.678 | 0.606 |
| wood | 0.656 | 0.721 | 0.711 | 0.642 |
| zipper | 0.466 | 0.678 | 0.570 | 0.593 |
| **mean** | **0.784** | **0.795** | **0.759** | **0.680** |

#### Comparison with paper (×100)

| k | Paper I-AUROC | Ours I-AUROC | Paper P-AUROC | Ours P-AUROC |
|---|---|---|---|---|
| 1 | 93.0 | 57.3 | 96.3 | 78.4 |
| 2 | 95.0 | 59.1 | 97.0 | 79.5 |
| 4 | 95.9 | 57.8 | 96.7 | 75.9 |
| 8 | 96.2 | 56.5 | 96.3 | 68.0 |

---

### Table 3 — Adapter Layer Ablation (mean I-AUROC / P-AUROC, ×100)

Averaged over 3 seeds, k ∈ {1, 2}.

#### Our results

| Adapter Layers | k=1 I | k=1 P | k=2 I | k=2 P |
|---|---|---|---|---|
| {1,2} (blocks 0,1) | 63.5 | 80.4 | 59.7 | 78.0 |
| {3,4} (blocks 2,3) | 61.7 | 79.7 | 62.4 | 80.0 |
| {5,6} (blocks 4,5) | 57.3 | 78.4 | 59.1 | 79.5 |

#### Paper results (Table 3)

| Adapter Layers | k=1 I | k=1 P | k=2 I | k=2 P |
|---|---|---|---|---|
| All {1,2}+{3,4}+{5,6} | 86.5 | 95.8 | 94.6 | 96.9 |
| {1,2} only | 86.7 | 96.0 | 92.7 | 96.2 |
| {3,4} only | 91.0 | 96.4 | 94.4 | 96.6 |
| {5,6} only | 93.0 | 96.3 | 95.0 | 97.0 |

---

### Table 5 — Component Ablation, k=2 (mean I-AUROC / P-AUROC, ×100)

Averaged over 3 seeds. Adapter config: {1,2} (as stated in paper Section 4.5).

| Variant | Paper I | Paper P | Ours I | Ours P |
|---|---|---|---|---|
| PatchCore (frozen backbone) | 85.43 | 95.87 | 62.0 | 76.1 |
| + FMN only | 88.39 | 95.02 | 58.3 | 76.5 |
| + Adapter only | 91.73 | 96.17 | 62.4 | 82.1 |
| FMN + Adapter (no iterative) | 84.27 | 90.23 | — | — |
| Full DFM (iterative) | 92.73 | 96.20 | 60.1 | 80.2 |


