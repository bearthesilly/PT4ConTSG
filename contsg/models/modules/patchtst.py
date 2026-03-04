"""
PatchTST Modules: Shared components for time series encoding.

This module provides the foundational components for PatchTST-based time series
encoding, including patch embeddings, attention mechanisms, and segment-level
encoders.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# Positional Embeddings
# =============================================================================

class PositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embedding.

    Fixed positional encodings computed in log space for numerical stability.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        """Return positional encoding for sequence length."""
        return self.pe[:, : x.size(1)]


# =============================================================================
# Patch Embedding
# =============================================================================

class PatchEmbedding(nn.Module):
    """
    Patch embedding for time series.

    Converts time series into patches and projects to d_model dimension.
    Supports multi-variate time series with channel-independent patching.

    Args:
        d_model: Model dimension
        patch_len: Length of each patch
        n_var: Number of variables/channels
        stride: Stride for patch extraction
        padding: Padding to add before patching
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int,
        patch_len: int,
        n_var: int,
        stride: int,
        padding: int,
        dropout: float,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.n_var = n_var
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        self.value_embedding = nn.Linear(patch_len * n_var, d_model, bias=False)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, n_var, L) - batch of multivariate time series
        Returns:
            (B, n_patches, d_model) - patch embeddings
        """
        # Pad and unfold into patches
        x = self.padding_patch_layer(x)  # (B, n_var, L + pad)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # (B, n_var, Nl, Pl)

        B, n_var, Nl, Pl = x.shape
        # Reshape: (B, n_var, Nl, Pl) -> (B, Nl, Pl * n_var)
        x = x.permute(0, 2, 3, 1).contiguous().reshape(B, Nl, Pl * n_var)

        # Project and add positional encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


# =============================================================================
# Attention Mechanisms
# =============================================================================

class TriangularCausalMask:
    """Triangular causal attention mask."""

    def __init__(self, B: int, L: int, device: torch.device = "cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)

    @property
    def mask(self) -> Tensor:
        return self._mask


class FullAttention(nn.Module):
    """
    Full scaled dot-product attention.

    Args:
        mask_flag: Whether to apply causal masking
        factor: Attention factor (unused, for API compatibility)
        scale: Custom attention scale (default: 1/sqrt(d_k))
        attention_dropout: Dropout rate for attention weights
        output_attention: Whether to return attention weights
    """

    def __init__(
        self,
        mask_flag: bool = True,
        factor: int = 5,
        scale: Optional[float] = None,
        attention_dropout: float = 0.1,
        output_attention: bool = False,
    ):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        attn_mask: Optional[TriangularCausalMask] = None,
        tau: Optional[float] = None,
        delta: Optional[Tensor] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            queries: (B, L, H, E)
            keys: (B, S, H, E)
            values: (B, S, H, D)
            attn_mask: Optional causal mask
        Returns:
            output: (B, L, H, D)
            attention: Optional (B, H, L, S) attention weights
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / math.sqrt(E)

        # Compute attention scores: (B, H, L, S)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        # Apply causal mask if needed
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        # Softmax and dropout
        A = self.dropout(torch.softmax(scale * scores, dim=-1))

        # Compute output: (B, L, H, D)
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        return V.contiguous(), None


class AttentionLayer(nn.Module):
    """
    Multi-head attention layer with linear projections.

    Args:
        attention: Attention mechanism module
        d_model: Model dimension
        n_heads: Number of attention heads
        d_keys: Dimension of keys (default: d_model // n_heads)
        d_values: Dimension of values (default: d_model // n_heads)
    """

    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        n_heads: int,
        d_keys: Optional[int] = None,
        d_values: Optional[int] = None,
    ):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        attn_mask: Optional[TriangularCausalMask] = None,
        tau: Optional[float] = None,
        delta: Optional[Tensor] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            queries, keys, values: (B, L, D) or (B, S, D)
        Returns:
            output: (B, L, D)
            attention: Optional attention weights
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project and reshape for multi-head attention
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        # Apply attention
        out, attn = self.inner_attention(
            queries, keys, values, attn_mask, tau=tau, delta=delta
        )

        # Reshape and project output
        out = out.view(B, L, -1)
        return self.out_projection(out), attn


# =============================================================================
# Encoder Layers
# =============================================================================

class EncoderLayer(nn.Module):
    """
    Transformer encoder layer with attention and feedforward.

    Args:
        attention: Attention layer
        d_model: Model dimension
        d_ff: Feedforward dimension (default: 4 * d_model)
        dropout: Dropout rate
        activation: Activation function ("relu" or "gelu")
    """

    def __init__(
        self,
        attention: AttentionLayer,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[TriangularCausalMask] = None,
        tau: Optional[float] = None,
        delta: Optional[Tensor] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x: (B, L, D) input tensor
        Returns:
            output: (B, L, D)
            attention: Optional attention weights
        """
        # Self-attention with residual
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)

        # Feedforward with residual
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    """
    Transformer encoder stack.

    Args:
        attn_layers: List of encoder layers
        conv_layers: Optional list of convolution layers
        norm_layer: Optional final normalization layer
    """

    def __init__(
        self,
        attn_layers: list,
        conv_layers: Optional[list] = None,
        norm_layer: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers else None
        self.norm = norm_layer

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[TriangularCausalMask] = None,
        tau: Optional[float] = None,
        delta: Optional[Tensor] = None,
    ) -> tuple[Tensor, list]:
        """
        Args:
            x: (B, L, D) input tensor
        Returns:
            output: (B, L, D)
            attentions: List of attention weights
        """
        attns = []

        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(
                zip(self.attn_layers[:-1], self.conv_layers, strict=False)
            ):
                d = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=d)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class TVEncoder(nn.Module):
    """
    Time-Variable encoder with separate attention for time and variable dimensions.

    Alternates between time-wise and variable-wise attention at each layer.

    Args:
        t_attn_layers: List of time attention layers
        v_attn_layers: List of variable attention layers
        conv_layers: Optional convolution layers
        norm_layer: Optional final normalization
    """

    def __init__(
        self,
        t_attn_layers: list,
        v_attn_layers: list,
        conv_layers: Optional[list] = None,
        norm_layer: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.t_attn_layers = nn.ModuleList(t_attn_layers)
        self.v_attn_layers = nn.ModuleList(v_attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers else None
        self.norm = norm_layer

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[TriangularCausalMask] = None,
        tau: Optional[float] = None,
        delta: Optional[Tensor] = None,
    ) -> tuple[Tensor, list]:
        """
        Args:
            x: (B, L, N, D) input tensor with time and variable dims
        Returns:
            output: (B, L, N, D)
            attentions: List of attention weights
        """
        B, L, N, D = x.shape
        attns = []

        if self.conv_layers is not None:
            for i, (t_attn, v_attn, conv) in enumerate(
                zip(self.t_attn_layers, self.v_attn_layers, self.conv_layers, strict=False)
            ):
                d = delta if i == 0 else None
                # Time attention
                x = x.permute(0, 2, 1, 3).reshape(B * N, L, -1)
                x, t_att = t_attn(x, attn_mask=attn_mask, tau=tau, delta=d)
                attns.append(t_att)

                # Variable attention
                x = x.reshape(B, N, L, -1).permute(0, 2, 1, 3).reshape(B * L, N, -1)
                x, v_att = v_attn(x, attn_mask=attn_mask, tau=tau, delta=d)
                attns.append(v_att)

                x = x.reshape(B, L, N, -1)
                x = conv(x)
        else:
            for t_attn, v_attn in zip(self.t_attn_layers, self.v_attn_layers, strict=False):
                # Time attention
                x = x.permute(0, 2, 1, 3).reshape(B * N, L, -1)
                x, t_att = t_attn(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(t_att)

                # Variable attention
                x = x.reshape(B, N, L, -1).permute(0, 2, 1, 3).reshape(B * L, N, -1)
                x, v_att = v_attn(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(v_att)

                x = x.reshape(B, L, N, -1)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


# =============================================================================
# PatchEncoder & SegmentTSEncoder
# =============================================================================

def _sinusoidal_time_embedding(pos: Tensor, d_model: int, device: torch.device) -> Tensor:
    """Generate sinusoidal time embeddings."""
    pe = torch.zeros(pos.shape[0], pos.shape[1], d_model, device=device)
    position = pos.unsqueeze(2).float()
    div_term = 1 / torch.pow(
        10000.0, torch.arange(0, d_model, 2, device=device).float() / d_model
    )
    pe[:, :, 0::2] = torch.sin(position * div_term)
    pe[:, :, 1::2] = torch.cos(position * div_term)
    return pe


class PatchEncoder(nn.Module):
    """
    Patch-based encoder with time and variable positional embeddings.

    Encodes each variable independently with shared parameters, then adds
    time and variable positional embeddings.

    Args:
        d_model: Model dimension
        patch_len: Patch length
        stride: Patch stride
        padding: Padding for patching
        dropout: Dropout rate
        n_var: Number of variables
        n_heads: Number of attention heads
        d_ff: Feedforward dimension
        e_layers: Number of encoder layers
        seq_len: Sequence length (for computing patch count)
        factor: Attention factor
        activation: Activation function
        output_attention: Whether to output attention
    """

    def __init__(
        self,
        d_model: int,
        patch_len: int,
        stride: int,
        padding: int,
        dropout: float,
        n_var: int,
        n_heads: int,
        d_ff: int,
        e_layers: int,
        seq_len: int,
        factor: int = 5,
        activation: str = "gelu",
        output_attention: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_var = n_var
        self.seq_len = seq_len

        # Patch embedding (single variable)
        self.patch_embedding = PatchEmbedding(
            d_model, patch_len, 1, stride, padding, dropout
        )

        # Transformer encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False, factor,
                            attention_dropout=dropout,
                            output_attention=output_attention,
                        ),
                        d_model, n_heads,
                    ),
                    d_model, d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

        # Compute number of patches
        self.n_patches = int((seq_len + padding - patch_len) / stride + 1)

        # Variable positional embedding
        self.var_pos_emb = nn.Embedding(n_var, d_model)

    def forward(self, ts: Tensor) -> Tensor:
        """
        Args:
            ts: (B, L, N) - batch of multivariate time series
        Returns:
            (B*N, Nl, d_model) - encoded patches per variable
        """
        B, L, N = ts.shape
        device = ts.device

        # Reshape for per-variable encoding: (B, L, N) -> (B*N, 1, L)
        ts = ts.permute(0, 2, 1).reshape(B * N, 1, L)

        # Patch embedding: (B*N, Nl, d_model)
        ts_emb = self.patch_embedding(ts)
        BN, Nl, D = ts_emb.shape

        # Time positional embedding
        tp = torch.arange(Nl, device=device).unsqueeze(0)  # (1, Nl)
        time_pos_emb = _sinusoidal_time_embedding(tp, D, device)
        time_pos_emb = time_pos_emb.expand(BN, -1, -1)  # (BN, Nl, D)

        # Variable positional embedding
        var_pos_emb = self.var_pos_emb(torch.arange(N, device=device))  # (N, D)
        var_pos_emb = var_pos_emb.unsqueeze(0).expand(B, -1, -1)  # (B, N, D)
        var_pos_emb = var_pos_emb.reshape(B * N, 1, -1).expand(-1, Nl, -1)  # (BN, Nl, D)

        # Add positional embeddings
        ts_emb = ts_emb + time_pos_emb + var_pos_emb

        # Encode
        ts_enc_out, _ = self.encoder(ts_emb)
        return ts_enc_out

    def mask_forward(self, ts: Tensor, mask_ratio: float) -> Tensor:
        """
        Forward with random masking (for MAE-style pretraining).

        Args:
            ts: (B, L, N) - batch of time series
            mask_ratio: Fraction of patches to mask
        Returns:
            (B*N, Nl, d_model) - encoded patches with masking
        """
        B, L, N = ts.shape
        device = ts.device

        ts = ts.permute(0, 2, 1).reshape(B * N, 1, L)
        ts_emb = self.patch_embedding(ts)
        BN, Nl, D = ts_emb.shape

        # Random masking
        n_mask = int(Nl * mask_ratio)
        zero_indices = torch.multinomial(
            torch.ones(BN, Nl, device=device), n_mask, replacement=False
        )
        mask = torch.ones(BN, Nl, dtype=torch.float, device=device)
        mask.scatter_(1, zero_indices, 0)
        mask = mask.unsqueeze(-1)  # (BN, Nl, 1)

        ts_emb_masked = ts_emb * mask

        # Add positional embeddings
        tp = torch.arange(Nl, device=device).unsqueeze(0)
        time_pos_emb = _sinusoidal_time_embedding(tp, D, device).expand(BN, -1, -1)

        var_pos_emb = self.var_pos_emb(torch.arange(N, device=device))
        var_pos_emb = var_pos_emb.unsqueeze(0).expand(B, -1, -1)
        var_pos_emb = var_pos_emb.reshape(B * N, 1, -1).expand(-1, Nl, -1)

        ts_emb_masked = ts_emb_masked + time_pos_emb + var_pos_emb

        ts_enc_out, _ = self.encoder(ts_emb_masked)
        return ts_enc_out


class SegmentTSEncoder(nn.Module):
    """
    Segment-level time series encoder.

    Splits the input time series into fixed-length segments and encodes each
    segment independently using PatchTST-style architecture.

    Args:
        segment_len: Length of each segment
        n_segments: Number of segments (K)
        d_model: Model dimension
        patch_len: Patch length for PatchTST
        n_var: Number of variables/channels
        stride: Stride for patching
        padding: Padding for patching
        dropout: Dropout rate
        e_layers: Number of encoder layers
        n_heads: Number of attention heads
        d_ff: Feedforward dimension
        coemb_dim: Output co-embedding dimension
        factor: Attention factor
        activation: Activation function
        output_attention: Whether to output attention weights
    """

    def __init__(
        self,
        segment_len: int,
        n_segments: int,
        d_model: int,
        patch_len: int,
        n_var: int,
        stride: int,
        padding: int,
        dropout: float,
        e_layers: int,
        n_heads: int,
        d_ff: int,
        coemb_dim: int,
        factor: int = 5,
        activation: str = "gelu",
        output_attention: bool = False,
    ):
        super().__init__()
        self.segment_len = segment_len
        self.n_segments = n_segments
        self.d_model = d_model
        self.coemb_dim = coemb_dim

        # Patch embedding (multi-variate for each segment)
        self.patch_embedding = PatchEmbedding(
            d_model, patch_len, n_var, stride, padding, dropout
        )

        # Transformer encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False, factor,
                            attention_dropout=dropout,
                            output_attention=output_attention,
                        ),
                        d_model, n_heads,
                    ),
                    d_model, d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

        # Number of patches per segment
        n_patches_per_segment = int(
            (segment_len + padding - patch_len) / stride + 1
        )
        self.n_patches_per_segment = n_patches_per_segment

        # Projection to segment embedding
        self.segment_proj = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(d_model * n_patches_per_segment, coemb_dim),
        )

    def forward(self, ts: Tensor) -> Tensor:
        """
        Encode time series into segment-level embeddings.

        Args:
            ts: (B, L, C) - batch of time series
        Returns:
            (B, K, D) - segment embeddings
        """
        B, L, C = ts.shape
        K = self.n_segments
        seg_len = self.segment_len

        # Split into segments: (B, L, C) -> (B, K, seg_len, C)
        ts_segments = ts.view(B, K, seg_len, C)

        # Flatten for batch processing: (B*K, seg_len, C)
        ts_flat = ts_segments.view(B * K, seg_len, C)

        # PatchEmbedding expects (B, C, L): (B*K, C, seg_len)
        ts_flat = ts_flat.permute(0, 2, 1)

        # Patch embedding: (B*K, n_patches, d_model)
        patch_emb = self.patch_embedding(ts_flat)

        # Transformer encoding: (B*K, n_patches, d_model)
        enc_out, _ = self.encoder(patch_emb)

        # Project to segment embedding: (B*K, coemb_dim)
        seg_emb = self.segment_proj(enc_out)

        # Reshape: (B, K, coemb_dim)
        return seg_emb.view(B, K, -1)

    def get_segment_ts(self, ts: Tensor, segment_idx: int) -> Tensor:
        """
        Extract a specific segment from time series.

        Args:
            ts: (B, L, C) - batch of time series
            segment_idx: Index of segment to extract
        Returns:
            (B, seg_len, C) - extracted segment
        """
        start = segment_idx * self.segment_len
        end = start + self.segment_len
        return ts[:, start:end, :]
