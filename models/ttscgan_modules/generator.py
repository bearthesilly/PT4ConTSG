"""
TTS-CGAN Generator Network.

Transformer-based generator with label embedding for conditional time series generation.

Reference:
    TTS-CGAN: A Transformer Time-Series Conditional GAN for Biosignal Data Augmentation
    https://arxiv.org/abs/2206.13676
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module."""

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

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, N, D)
            mask: Optional attention mask

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

        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy = energy.masked_fill(~mask, fill_value)

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

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """Forward with residual connection."""
        return x + self.fn(x, **kwargs)


class GenTransformerEncoderBlock(nn.Sequential):
    """Transformer encoder block for generator."""

    def __init__(
        self,
        emb_size: int,
        num_heads: int = 5,
        drop_p: float = 0.5,
        forward_expansion: int = 4,
        forward_drop_p: float = 0.5,
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


class GenTransformerEncoder(nn.Sequential):
    """Stacked transformer encoder for generator."""

    def __init__(self, depth: int = 3, **kwargs):
        """
        Initialize transformer encoder stack.

        Args:
            depth: Number of transformer blocks
            **kwargs: Arguments for GenTransformerEncoderBlock
        """
        super().__init__(
            *[GenTransformerEncoderBlock(**kwargs) for _ in range(depth)]
        )


class Generator(nn.Module):
    """
    TTS-CGAN Generator.

    Takes noise vector and class label, generates time series in image-like format.

    Architecture:
        1. Label embedding concatenated with noise
        2. Linear projection to sequence of embeddings
        3. Transformer encoder stack
        4. Conv2d to output channels

    Input format:
        - noise: (B, latent_dim)
        - labels: (B,) integer class indices

    Output format:
        - generated: (B, channels, 1, seq_len) - image-like format
    """

    def __init__(
        self,
        seq_len: int = 150,
        channels: int = 3,
        num_classes: int = 9,
        latent_dim: int = 100,
        data_embed_dim: int = 10,
        label_embed_dim: int = 10,
        depth: int = 3,
        num_heads: int = 5,
        forward_drop_rate: float = 0.5,
        attn_drop_rate: float = 0.5,
    ):
        """
        Initialize generator.

        Args:
            seq_len: Output sequence length
            channels: Number of output channels (n_var)
            num_classes: Number of class labels
            latent_dim: Dimension of noise vector
            data_embed_dim: Internal embedding dimension
            label_embed_dim: Label embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            forward_drop_rate: MLP dropout rate
            attn_drop_rate: Attention dropout rate
        """
        super().__init__()

        self.seq_len = seq_len
        self.channels = channels
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.data_embed_dim = data_embed_dim
        self.label_embed_dim = label_embed_dim
        self.depth = depth
        self.num_heads = num_heads

        # Label embedding
        self.label_embedding = nn.Embedding(num_classes, label_embed_dim)

        # Project noise + label to sequence
        self.l1 = nn.Linear(latent_dim + label_embed_dim, seq_len * data_embed_dim)

        # Transformer encoder
        self.blocks = GenTransformerEncoder(
            depth=depth,
            emb_size=data_embed_dim,
            num_heads=num_heads,
            drop_p=attn_drop_rate,
            forward_drop_p=forward_drop_rate,
        )

        # Output projection (Conv2d for compatibility with original)
        self.deconv = nn.Conv2d(data_embed_dim, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, z: Tensor, labels: Tensor) -> Tensor:
        """
        Generate time series from noise and labels.

        Args:
            z: Noise vector (B, latent_dim)
            labels: Class labels (B,)

        Returns:
            Generated time series (B, channels, 1, seq_len)
        """
        # Embed labels and concatenate with noise
        c = self.label_embedding(labels)  # (B, label_embed_dim)
        x = torch.cat([z, c], dim=1)  # (B, latent_dim + label_embed_dim)

        # Project to sequence
        x = self.l1(x)  # (B, seq_len * data_embed_dim)
        x = x.view(-1, self.seq_len, self.data_embed_dim)  # (B, seq_len, data_embed_dim)

        # Transformer encoding
        x = self.blocks(x)  # (B, seq_len, data_embed_dim)

        # Reshape for conv: (B, 1, seq_len, data_embed_dim) -> (B, data_embed_dim, 1, seq_len)
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        x = x.permute(0, 3, 1, 2)  # (B, data_embed_dim, 1, seq_len)

        # Output projection
        output = self.deconv(x)  # (B, channels, 1, seq_len)

        return output
