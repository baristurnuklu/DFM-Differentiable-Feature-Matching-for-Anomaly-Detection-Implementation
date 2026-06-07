"""
Feature Matching Network (FMN) — the core of DFM.

Replaces PatchCore's non-differentiable nearest-neighbour search with an
equivalent differentiable pipeline:

  PatchCore:  S[i] = min_k ||P[i] - M[k]||²          (discrete argmin)
  FMN:        Sim[i,k] = ||P[i] - M[k]||₂             (full L2 distance matrix)
              Sim'     = modulate1(Sim)                (learnable, identity init)
              S        = MinPool(Sim', dim=-1)          (differentiable)
              S'       = modulate2(S)                  (learnable, identity init)
              s        = MaxPool(S', dim=-1)            (differentiable)

At initialisation FMN == PatchCore exactly (modulate layers are identity,
M is the coreset memory bank).

Shapes throughout (B = batch, N = 196 patches, L = 196 memory slots, D = 768):
  P      (B, N, D)
  M      (L, D)        nn.Parameter
  Sim    (B, N, L)
  Sim'   (B, N, L)
  S      (B, N)
  S'     (B, N)
  s      (B,)
"""

import torch
import torch.nn as nn

from src.coreset import MEMORY_BANK_SIZE
from src.backbone import EMBED_DIM, NUM_PATCHES


class FMN(nn.Module):
    """
    Feature Matching Network.

    Args:
        memory_bank_size : number of memory bank entries L (default 196)
        embed_dim        : feature dimension D (default 768)
    """

    def __init__(
        self,
        memory_bank_size: int = MEMORY_BANK_SIZE,
        embed_dim: int = EMBED_DIM,
    ):
        super().__init__()

        self.memory_bank_size = memory_bank_size
        self.embed_dim = embed_dim

        # Memory bank — treated as a trainable parameter in Stage 2.
        # Initialised to zeros here; call set_memory_bank() before training.
        self.M = nn.Parameter(
            torch.zeros(memory_bank_size, embed_dim), requires_grad=False
        )

        # Modulate layer 1: operates on the memory-bank axis of Sim (B, N, L→L)
        self.modulate1 = nn.Linear(memory_bank_size, memory_bank_size, bias=True)
        nn.init.eye_(self.modulate1.weight)
        nn.init.zeros_(self.modulate1.bias)

        # Modulate layer 2: operates on the spatial axis of S (B, N→N)
        self.modulate2 = nn.Linear(num_patches := NUM_PATCHES, num_patches, bias=True)
        nn.init.eye_(self.modulate2.weight)
        nn.init.zeros_(self.modulate2.bias)

    # ── Memory bank setup ──────────────────────────────────────────────────────

    def set_memory_bank(self, coreset: torch.Tensor) -> None:
        """
        Initialise M from a coreset tensor.

        Args:
            coreset : (L, D) tensor produced by build_memory_bank()
        """
        assert coreset.shape == (self.memory_bank_size, self.embed_dim), (
            f"Expected coreset shape ({self.memory_bank_size}, {self.embed_dim}), "
            f"got {tuple(coreset.shape)}"
        )
        with torch.no_grad():
            self.M.copy_(coreset)

    # ── Stage helpers ──────────────────────────────────────────────────────────

    def freeze_memory_bank(self) -> None:
        self.M.requires_grad_(False)

    def unfreeze_memory_bank(self) -> None:
        self.M.requires_grad_(True)

    def fmn_parameters(self):
        """Yields modulate layer params + M — everything trained in Stage 2."""
        yield from self.modulate1.parameters()
        yield from self.modulate2.parameters()
        yield self.M

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, P: torch.Tensor):
        """
        Args:
            P : (B, N, D) patch features from backbone

        Returns:
            s  : (B,)    image-level anomaly score  (for I-AUROC / detection loss)
            S_prime : (B, N) patch-level scores     (for P-AUROC / segmentation loss)
        """
        # ── Distance matrix ───────────────────────────────────────────────────
        # ||P[i] - M[k]||₂  (paper eq. 3 — actual L2, not squared)
        P_sq = (P * P).sum(dim=-1, keepdim=True)                       # (B, N, 1)
        M_sq = (self.M * self.M).sum(dim=-1)                           # (L,)
        PM   = P @ self.M.T                                            # (B, N, L)
        Sim  = (P_sq + M_sq - 2.0 * PM).clamp(min=1e-6).sqrt()         # (B, N, L)

        # ── Modulate layer 1 → MinPool ────────────────────────────────────────
        Sim_prime = self.modulate1(Sim)                  # (B, N, L)
        S         = Sim_prime.min(dim=-1).values         # (B, N)

        # ── Modulate layer 2 → MaxPool ────────────────────────────────────────
        S_prime = self.modulate2(S)                      # (B, N)
        s       = S_prime.max(dim=-1).values             # (B,)

        return s, S_prime
