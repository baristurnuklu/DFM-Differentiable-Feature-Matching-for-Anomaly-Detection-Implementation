"""
Greedy coreset sampling for memory bank construction.

Given N patch feature vectors, selects C representative vectors that
maximally cover the feature space. Used to build the initial memory bank M
before training, following PatchCore / DFM.

Algorithm: greedy farthest-point sampling
  1. Pick a random starting point
  2. min_dist[i] = squared distance from point i to the nearest selected point
  3. Next pick = argmax(min_dist)
  4. Update min_dist after each pick
  5. Repeat until C points selected

Complexity: O(N * C * D) total, O(N * D) per iteration.
"""

import torch


MEMORY_BANK_SIZE = 196  # 14×14 patch grid, fixed by the paper


def greedy_coreset(features: torch.Tensor, target_n: int = MEMORY_BANK_SIZE,
                   seed: int = 0) -> torch.Tensor:
    """
    Select target_n representative vectors from features via greedy farthest-point sampling.

    Args:
        features : (N, D) float tensor — all patch features from training images
        target_n : desired coreset size (default 196)
        seed     : random seed for reproducible starting point (local, no global side effects)

    Returns:
        coreset : (min(N, target_n), D)
    """
    N, D = features.shape

    if N <= target_n:
        return features.clone()

    device = features.device

    # Local generator — does not affect global torch random state
    gen = torch.Generator()
    gen.manual_seed(seed)
    start = int(torch.randint(N, (1,), generator=gen).item())

    selected_indices = [start]
    min_dist = _sq_dists_to_point(features, features[start])  # (N,)

    for _ in range(target_n - 1):
        next_idx = int(min_dist.argmax().item())
        selected_indices.append(next_idx)
        new_dists = _sq_dists_to_point(features, features[next_idx])
        min_dist = torch.minimum(min_dist, new_dists)

    idx_tensor = torch.tensor(selected_indices, dtype=torch.long, device=device)
    return features[idx_tensor]


def _sq_dists_to_point(features: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distance from every row in features to a single query vector."""
    diff = features - query.unsqueeze(0)
    return (diff * diff).sum(dim=-1)


def build_memory_bank(train_features: torch.Tensor,
                      target_n: int = MEMORY_BANK_SIZE,
                      seed: int = 0) -> torch.Tensor:
    """
    Build the initial memory bank from all training patch features.

    Args:
        train_features : (k * 196, D) — stacked patch features from k training images
        target_n       : memory bank size (default 196)
        seed           : random seed

    Returns:
        memory_bank : (target_n, D)
    """
    assert train_features.ndim == 2, \
        f"Expected (N, D) tensor, got shape {train_features.shape}"
    return greedy_coreset(train_features, target_n=target_n, seed=seed)
