"""
CLIP ViT-B/16 backbone with optional adapter insertion.

Paper spec:
  - Backbone  : ViT-B/16 from CLIP (OpenAI pretrained weights)
  - Feature   : output of the 6th block (1-indexed) = index 5 (0-indexed)
                shape (B, 196, 768) — patch tokens only, CLS dropped
  - Adapters  : nn.Linear(768, 768) inserted after chosen blocks,
                initialized as identity (W=I, b=0)
  - Training  : backbone weights are always frozen; only adapters are trained

adapter_layers uses 0-indexed block indices:
  paper {1,2}  → [0, 1]
  paper {3,4}  → [2, 3]
  paper {5,6}  → [4, 5]
  Table 5 default → [0, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from typing import List, Optional


EMBED_DIM    = 768
NUM_PATCHES  = 196   # 14×14 for 224px input with 16px patches
FEATURE_LAYER = 5    # 0-indexed (= 6th block, paper's 1-indexed notation)


# ── Adapter ────────────────────────────────────────────────────────────────────

class Adapter(nn.Module):
    """
    Single linear layer initialized as identity.
    Inserted after a transformer block — at init adapter(x) == x exactly.
    """

    def __init__(self, dim: int = EMBED_DIM):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=True)
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ── Block wrapper ──────────────────────────────────────────────────────────────

class BlockWithAdapter(nn.Module):
    """
    Frozen transformer block followed by a trainable adapter.
    forward(x) = adapter(block(x))
    """

    def __init__(self, block: nn.Module, adapter: Adapter):
        super().__init__()
        self.block   = block    # frozen
        self.adapter = adapter  # trainable

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.adapter(self.block(x, *args, **kwargs))


# ── Backbone ───────────────────────────────────────────────────────────────────

class CLIPViTBackbone(nn.Module):
    """
    CLIP ViT-B/16 backbone that extracts patch features from a chosen intermediate block.

    Args:
        adapter_layers : 0-indexed block indices where adapters are inserted.
                         None or [] means no adapters (pure frozen PatchCore backbone).
        feature_layer  : 0-indexed block whose output is returned as patch features.
                         Default 5 (= 6th block, paper spec).
        device         : 'cpu', 'cuda', or 'mps'
    """

    def __init__(
        self,
        adapter_layers: Optional[List[int]] = None,
        feature_layer: int = FEATURE_LAYER,
        device: str = "cpu",
    ):
        super().__init__()
        self.feature_layer = feature_layer
        self._device = torch.device(device)

        # ── Load pretrained CLIP ViT-B/16 ─────────────────────────────────────
        clip_model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16-quickgelu", pretrained="openai"
        )
        vit = clip_model.visual

        # ── Freeze every backbone parameter ───────────────────────────────────
        for p in vit.parameters():
            p.requires_grad_(False)

        # ── Extract the components we run manually ─────────────────────────────
        # Stored as plain attributes so they move with .to(device) and stay frozen.
        self.conv1               = vit.conv1
        self.class_embedding     = vit.class_embedding      # Parameter (768,)
        self.positional_embedding = vit.positional_embedding # Parameter (197, 768)
        self.ln_pre              = vit.ln_pre
        self.blocks              = nn.ModuleList(list(vit.transformer.resblocks))

        # ── Insert adapters ───────────────────────────────────────────────────
        for idx in (adapter_layers or []):
            if not (0 <= idx < len(self.blocks)):
                raise ValueError(
                    f"adapter_layers index {idx} out of range "
                    f"(model has {len(self.blocks)} blocks, 0-indexed)"
                )
            adapter = Adapter(EMBED_DIM)
            self.blocks[idx] = BlockWithAdapter(self.blocks[idx], adapter)

        self.to(self._device)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 3, 224, 224) normalised input images

        Returns:
            patch_features : (B, 196, 768)
        """
        x = x.to(self._device)

        # Patch embedding
        x = self.conv1(x)                                       # (B, 768, 14, 14)
        x = x.reshape(x.shape[0], x.shape[1], -1)              # (B, 768, 196)
        x = x.permute(0, 2, 1)                                  # (B, 196, 768)

        # Prepend CLS token
        cls = self.class_embedding\
                   .unsqueeze(0).unsqueeze(0)\
                   .expand(x.shape[0], -1, -1)                  # (B, 1, 768)
        x = torch.cat([cls, x], dim=1)                          # (B, 197, 768)

        # Positional embedding + pre-norm
        x = x + self.positional_embedding                        # (B, 197, 768)
        x = self.ln_pre(x)                                       # (B, 197, 768)

        # Transformer blocks — seq-first format required by open_clip
        x = x.permute(1, 0, 2)                                  # (197, B, 768)
        for block in self.blocks[: self.feature_layer + 1]:
            x = block(x)
        x = x.permute(1, 0, 2)                                  # (B, 197, 768)

        # Drop CLS token, L2-normalize along feature dim (PatchCore convention)
        x = x[:, 1:, :]                                          # (B, 196, 768)
        return F.normalize(x, p=2, dim=-1)                       # (B, 196, 768)

    # ── Parameter helpers ─────────────────────────────────────────────────────

    def adapter_parameters(self):
        """Yields only the trainable adapter parameters (for Stage 1 optimizer)."""
        for block in self.blocks:
            if isinstance(block, BlockWithAdapter):
                yield from block.adapter.parameters()

    def freeze_adapters(self):
        for p in self.adapter_parameters():
            p.requires_grad_(False)

    def unfreeze_adapters(self):
        for p in self.adapter_parameters():
            p.requires_grad_(True)
