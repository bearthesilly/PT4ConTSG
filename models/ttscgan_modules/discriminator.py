"""
TTS-CGAN Discriminator Network.

Patch-based Transformer discriminator with auxiliary classifier.

Reference:
    TTS-CGAN: A Transformer Time-Series Conditional GAN for Biosignal Data Augmentation
    https://arxiv.org/abs/2206.13676
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module for discriminator."""

    def __init__(
        self,
        emb_size: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        """
        Initialize multi-head attention.

        Args:
            emb_size: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate for attention weights
        """
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads

        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, N, D)

        Returns:
            Output tensor (B, N, D)
        """
        queries = rearrange(
            self.queries(x), "b n (h d) -> b h n d", h=self.num_heads
        )
        keys = rearrange(
            self.keys(x), "b n (h d) -> b h n d", h=self.num_heads
        )
        values = rearrange(
            self.values(x), "b n (h d) -> b h n d", h=self.num_heads
        )

        # Compute attention scores
        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)
        scaling = self.emb_size ** 0.5
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)

        # Apply attention to values
        out = torch.einsum("bhal, bhlv -> bhav", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)

        return out


class FeedForwardBlock(nn.Sequential):
    """Feed-forward block with GELU activation."""

    def __init__(
        self,
        emb_size: int,
        expansion: int = 4,
        drop_p: float = 0.0,
    ):
        """
        Initialize feed-forward block.

        Args:
            emb_size: Embedding dimension
            expansion: MLP expansion factor
            drop_p: Dropout rate
        """
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class ResidualAdd(nn.Module):
    """Residual connection wrapper."""

    def __init__(self, fn: nn.Module):
        """
        Initialize residual add.

        Args:
            fn: Module to wrap with residual connection
        """
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        """Forward with residual connection."""
        return x + self.fn(x)


class DisTransformerEncoderBlock(nn.Sequential):
    """Transformer encoder block for discriminator."""

    def __init__(
        self,
        emb_size: int = 100,
        num_heads: int = 5,
        drop_p: float = 0.0,
        forward_expansion: int = 4,
        forward_drop_p: float = 0.0,
    ):
        """
        Initialize transformer encoder block.

        Args:
            emb_size: Embedding dimension
            num_heads: Number of attention heads
            drop_p: Attention dropout rate
            forward_expansion: MLP expansion factor
            forward_drop_p: MLP dropout rate
        """
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, num_heads, drop_p),
                    nn.Dropout(drop_p),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(emb_size, forward_expansion, forward_drop_p),
                    nn.Dropout(drop_p),
                )
            ),
        )


class DisTransformerEncoder(nn.Sequential):
    """Stacked transformer encoder for discriminator."""

    def __init__(self, depth: int = 8, **kwargs):
        """
        Initialize transformer encoder stack.

        Args:
            depth: Number of transformer blocks
            **kwargs: Arguments for DisTransformerEncoderBlock
        """
        super().__init__(
            *[DisTransformerEncoderBlock(**kwargs) for _ in range(depth)]
        )


class PatchEmbeddingLinear(nn.Module):
    """
    Patch embedding with linear projection.

    Converts image-like input to sequence of patch embeddings.
    """

    def __init__(
        self,
        in_channels: int = 21,
        patch_size: int = 16,
        emb_size: int = 100,
        seq_length: int = 1024,
    ):
        """
        Initialize patch embedding.

        Args:
            in_channels: Number of input channels
            patch_size: Size of each patch
            emb_size: Output embedding dimension
            seq_length: Input sequence length
        """
        super().__init__()

        self.patch_size = patch_size
        self.emb_size = emb_size
        num_patches = seq_length // patch_size

        # Patch projection
        self.projection = nn.Sequential(
            Rearrange(
                "b c (h s1) (w s2) -> b (h w) (s1 s2 c)",
                s1=1,
                s2=patch_size,
            ),
            nn.Linear(patch_size * in_channels, emb_size),
        )

        # CLS token and position embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn(num_patches + 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, 1, L)

        Returns:
            Patch embeddings with CLS token (B, num_patches + 1, emb_size)
        """
        b = x.shape[0]

        # Project patches
        x = self.projection(x)  # (B, num_patches, emb_size)

        # Prepend CLS token
        cls_tokens = repeat(self.cls_token, "() n e -> b n e", b=b)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches + 1, emb_size)

        # Add position embeddings
        x = x + self.positions

        return x


class ClassificationHead(nn.Module):
    """
    Dual classification head for adversarial and class prediction.

    Outputs:
        - adv_out: Real/Fake score (B, 1)
        - cls_out: Class logits (B, n_classes)
    """

    def __init__(
        self,
        emb_size: int = 100,
        adv_classes: int = 1,
        cls_classes: int = 10,
    ):
        """
        Initialize classification head.

        Args:
            emb_size: Input embedding dimension
            adv_classes: Output dimension for adversarial head (typically 1)
            cls_classes: Number of classes for auxiliary classifier
        """
        super().__init__()

        # Adversarial head (real/fake)
        self.adv_head = nn.Sequential(
            Reduce("b n e -> b e", reduction="mean"),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, adv_classes),
        )

        # Classification head (auxiliary classifier)
        self.cls_head = nn.Sequential(
            Reduce("b n e -> b e", reduction="mean"),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, cls_classes),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.

        Args:
            x: Transformer output (B, N, D)

        Returns:
            Tuple of (adv_out, cls_out):
                - adv_out: Adversarial score (B, 1)
                - cls_out: Class logits (B, n_classes)
        """
        out_adv = self.adv_head(x)
        out_cls = self.cls_head(x)
        return out_adv, out_cls


class Discriminator(nn.Module):
    """
    TTS-CGAN Discriminator.

    Patch-based Transformer with dual output heads:
    - Adversarial head for real/fake classification
    - Auxiliary classifier for class prediction

    Architecture:
        1. Patch embedding with linear projection
        2. Transformer encoder stack
        3. Dual classification heads

    Input format:
        - x: (B, channels, 1, seq_len) - image-like format

    Output format:
        - adv_out: (B, 1) - real/fake score
        - cls_out: (B, n_classes) - class logits
    """

    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 15,
        data_emb_size: int = 50,
        label_emb_size: int = 10,
        seq_length: int = 150,
        depth: int = 3,
        n_classes: int = 9,
        num_heads: int = 5,
        drop_p: float = 0.5,
        forward_drop_p: float = 0.5,
    ):
        """
        Initialize discriminator.

        Args:
            in_channels: Number of input channels (n_var)
            patch_size: Size of each patch
            data_emb_size: Embedding dimension
            label_emb_size: Label embedding dimension (unused, kept for compatibility)
            seq_length: Input sequence length
            depth: Number of transformer blocks
            n_classes: Number of classes for auxiliary classifier
            num_heads: Number of attention heads
            drop_p: Dropout rate
            forward_drop_p: MLP dropout rate
        """
        super().__init__()

        self.in_channels = in_channels
        self.patch_size = patch_size
        self.seq_length = seq_length
        self.n_classes = n_classes

        # Patch embedding
        self.patch_embed = PatchEmbeddingLinear(
            in_channels=in_channels,
            patch_size=patch_size,
            emb_size=data_emb_size,
            seq_length=seq_length,
        )

        # Transformer encoder
        self.transformer = DisTransformerEncoder(
            depth=depth,
            emb_size=data_emb_size,
            num_heads=num_heads,
            drop_p=drop_p,
            forward_drop_p=forward_drop_p,
        )

        # Classification heads
        self.head = ClassificationHead(
            emb_size=data_emb_size,
            adv_classes=1,
            cls_classes=n_classes,
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor (B, channels, 1, seq_len)

        Returns:
            Tuple of (adv_out, cls_out):
                - adv_out: Adversarial score (B, 1)
                - cls_out: Class logits (B, n_classes)
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches + 1, emb_size)

        # Transformer encoding
        x = self.transformer(x)  # (B, num_patches + 1, emb_size)

        # Classification
        adv_out, cls_out = self.head(x)

        return adv_out, cls_out
