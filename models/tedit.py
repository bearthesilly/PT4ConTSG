"""
TEdit: Multi-scale Patch Diffusion for Attribute-Conditioned Time Series Generation.

This module implements the TEdit model with multi-scale patch processing,
residual blocks with time/feature transformers, and adaptive conditioning.

Note: TEdit is attribute-based, not text-based. It uses discrete attributes
(batch["attrs"]) for conditioning, similar to TimeWeaver.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from contsg.models.base import BaseGeneratorModule, DiffusionMixin
from contsg.registry import Registry


# =============================================================================
# Helper Functions
# =============================================================================

def get_torch_trans(heads: int = 8, layers: int = 1, channels: int = 64) -> nn.Module:
    """Create a transformer encoder."""
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels,
        nhead=heads,
        dim_feedforward=64,
        activation="gelu",
        batch_first=True,
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels: int, out_channels: int, kernel_size: int) -> nn.Conv1d:
    """Create Conv1d layer with Kaiming initialization."""
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


# =============================================================================
# Attribute Encoder (for discrete attributes)
# =============================================================================

class AttributeEncoder(nn.Module):
    """
    Heterogeneous attribute encoder supporting multiple discrete attributes.
    Each attribute can have a different number of options.

    This is the same implementation as in TimeWeaver, designed for discrete
    attribute conditioning.
    """

    def __init__(self, num_attr_ops: list[int], emb_dim: int, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.emb_dim = emb_dim
        self.n_attr = len(num_attr_ops)
        self.n_ops_list = num_attr_ops

        # Initialize embeddings with shifted indices
        self.attr_emb = self._init_embs(num_attr_ops)
        self.register_buffer(
            "unknown_index",
            torch.tensor(self.n_ops_list, dtype=torch.long) - 1,
        )

        # Empty token embedding for unconditional generation
        self.empty_emb = nn.Embedding(num_embeddings=1, embedding_dim=self.emb_dim)

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.GELU(),
            nn.Linear(self.emb_dim, self.emb_dim)
        )

    def _init_embs(self, n_ops: list[int]) -> nn.Embedding:
        """Initialize embeddings with cumulative shift for heterogeneous attributes."""
        shift = np.cumsum(n_ops)
        shift = np.insert(shift, 0, 0)
        emb = nn.Embedding(num_embeddings=shift[-1], embedding_dim=self.emb_dim)
        shift_tensor = torch.from_numpy(shift[:-1]).unsqueeze(0)
        self.register_buffer("attr_shift", shift_tensor)
        return emb

    def forward(self, attrs: Tensor, replace_with_empty: bool = False) -> Tensor:
        """
        Args:
            attrs: Attribute indices (B, K)
            replace_with_empty: Whether to use empty token embeddings (for CFG)

        Returns:
            emb: Attribute embeddings (B, K, D)
        """
        if attrs.dim() == 1:
            attrs = attrs.unsqueeze(1)
        attrs = attrs.long()

        # Handle unknown/missing attributes (indicated by -1)
        if (attrs < 0).any():
            attrs = attrs.clone()
            unknown_mask = attrs < 0
            unknown_index = self.unknown_index.to(attrs.device).unsqueeze(0).expand_as(attrs)
            attrs[unknown_mask] = unknown_index[unknown_mask]

        if replace_with_empty:
            idx = torch.zeros(attrs.shape, dtype=torch.long, device=attrs.device)
            emb = self.empty_emb(idx)
        else:
            emb = self.attr_emb(attrs + self.attr_shift)

        emb = self.out_proj(emb)
        return emb


class AttrProjectorAvgLinear(nn.Module):
    """Average-pool attribute embeddings and project with a single linear layer."""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, attr: Tensor) -> Tensor:
        h = torch.mean(attr, dim=1, keepdim=True)
        h = h[:, None, :, :]
        return self.proj(h)


# =============================================================================
# Embedding Modules
# =============================================================================

class DiffusionEmbedding(nn.Module):
    """Sinusoidal diffusion timestep embedding."""

    def __init__(self, num_steps: int, embedding_dim: int = 128, projection_dim: Optional[int] = None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim // 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step: Tensor) -> Tensor:
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps: int, dim: int = 64) -> Tensor:
        steps = torch.arange(num_steps).unsqueeze(1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)
        table = steps * frequencies
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class SideEncoder(nn.Module):
    """Side information encoder for time and variable embeddings."""

    def __init__(self, num_var: int, var_emb_dim: int, time_emb_dim: int):
        super().__init__()
        self.num_var = num_var
        self.var_emb_dim = var_emb_dim
        self.time_emb_dim = time_emb_dim
        self.total_emb_dim = var_emb_dim + time_emb_dim

        self.var_emb = nn.Embedding(num_embeddings=num_var, embedding_dim=var_emb_dim)
        self.register_buffer("var_ids", torch.arange(num_var))

    def time_embedding(self, pos: Tensor, d_model: int = 128) -> Tensor:
        """Sinusoidal position embedding."""
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model, device=pos.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2, device=pos.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, tp: Tensor) -> Tensor:
        """
        Args:
            tp: Time positions (B, L)
        Returns:
            Side embeddings (B, D_t + D_v, V, L)
        """
        B, L = tp.shape
        # Time embedding: (B, L, D_t)
        time_emb = self.time_embedding(tp, self.time_emb_dim)
        # Expand to (B, L, V, D_t)
        time_emb = time_emb.unsqueeze(2).expand(-1, -1, self.num_var, -1)

        # Variable embedding: (V, D_v)
        var_emb = self.var_emb(self.var_ids)
        # Expand to (B, L, V, D_v)
        var_emb = var_emb.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        # Concatenate and permute: (B, D_t + D_v, V, L)
        side_emb = torch.cat([time_emb, var_emb], dim=-1)
        side_emb = side_emb.permute(0, 3, 2, 1)
        return side_emb


# =============================================================================
# Patch Embedding Modules
# =============================================================================

class TsPatchEmbedding(nn.Module):
    """Time series patch embedding with learnable projection."""

    def __init__(self, L_patch_len: int, channels: int, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.L_patch_len = L_patch_len
        self.padding_patch_layer = nn.ReplicationPad2d((0, L_patch_len, 0, 0))
        self.value_embedding = nn.Sequential(
            nn.Linear(L_patch_len * channels, d_model),
            nn.ReLU(),
        )

    def forward(self, x_in: Tensor) -> Tensor:
        """
        Args:
            x_in: Input tensor (B, C, n_var, L)
        Returns:
            Patch embeddings (B, D, n_var, Nl)
        """
        if x_in.shape[-1] % self.L_patch_len:
            x = self.padding_patch_layer(x_in)
        else:
            x = x_in
        x = x.unfold(dimension=3, size=self.L_patch_len, step=self.L_patch_len)
        B, C, n_var, Nl, Pl = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous().reshape(B, n_var, Nl, Pl * C)
        x = self.value_embedding(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class SidePatchEmbedding(nn.Module):
    """Side information patch embedding."""

    def __init__(self, L_patch_len: int, channels: int, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.L_patch_len = L_patch_len
        self.padding_patch_layer = nn.ReplicationPad2d((0, L_patch_len, 0, 0))
        self.value_embedding = nn.Linear(L_patch_len * channels, d_model)

    def forward(self, x_in: Tensor) -> Tensor:
        """
        Args:
            x_in: Input tensor (B, C, n_var, L)
        Returns:
            Patch embeddings (B, D, n_var, Nl)
        """
        if x_in.shape[-1] % self.L_patch_len:
            x = self.padding_patch_layer(x_in)
        else:
            x = x_in
        x = x.unfold(dimension=3, size=self.L_patch_len, step=self.L_patch_len)
        B, C, n_var, Nl, Pl = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous().reshape(B, n_var, Nl, Pl * C)
        x = self.value_embedding(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class PatchDecoder(nn.Module):
    """Decode patch embeddings back to time series."""

    def __init__(self, L_patch_len: int, d_model: int, channels: int):
        super().__init__()
        self.L_patch_len = L_patch_len
        self.channels = channels
        self.linear = nn.Linear(d_model, L_patch_len * channels)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Patch embeddings (B, D, n_var, Nl)
        Returns:
            Decoded output (B, C, n_var, Nl * L_patch_len)
        """
        B, D, n_var, Nl = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.linear(x)
        x = x.reshape(B, n_var, Nl, self.L_patch_len, self.channels)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = x.reshape(B, self.channels, n_var, Nl * self.L_patch_len)
        return x


# =============================================================================
# Residual Block
# =============================================================================

class ResidualBlock(nn.Module):
    """Residual block with time and feature transformers."""

    def __init__(
        self,
        side_dim: int,
        channels: int,
        diffusion_embedding_dim: int,
        nheads: int,
        is_linear: bool = False,
    ):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.side_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.is_linear = is_linear
        # Note: Linear attention transformer option is disabled in this migration
        # as it requires external dependency (linear_attention_transformer)
        # Use standard transformer instead
        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y: Tensor, base_shape: Tuple[int, ...], attention_mask: Optional[Tensor] = None) -> Tensor:
        """Apply time-wise attention."""
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(0, 2, 1), mask=attention_mask).permute(0, 2, 1)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y: Tensor, base_shape: Tuple[int, ...], attention_mask: Optional[Tensor] = None) -> Tensor:
        """Apply feature-wise attention."""
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(0, 2, 1), mask=attention_mask).permute(0, 2, 1)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(
        self,
        x: Tensor,
        side_emb: Tensor,
        diffusion_emb: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: Input tensor (B, C, K, L)
            side_emb: Side embeddings (B, side_dim, K, L)
            diffusion_emb: Diffusion step embedding (B, D)
            attention_mask: Optional attention mask
        Returns:
            Output tensor and skip connection
        """
        B, channel, K, L = x.shape
        base_shape = x.shape

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1).unsqueeze(-1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape, attention_mask)
        y = self.forward_feature(y, base_shape, None)

        y = y.reshape(B, channel, K * L)
        y = self.mid_projection(y)

        _, side_dim, _, _ = side_emb.shape
        side_emb = side_emb.reshape(B, side_dim, K * L)
        side_emb = self.side_projection(side_emb)
        y = y + side_emb

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)

        return (x + residual) / math.sqrt(2.0), skip


# =============================================================================
# TEdit Core Module
# =============================================================================

class TEditCore(nn.Module):
    """Core TEdit module for multi-scale patch diffusion."""

    def __init__(
        self,
        channels: int,
        multipatch_num: int,
        num_steps: int,
        diffusion_embedding_dim: int,
        side_config: Dict[str, Any],
        attention_mask_type: str,
        base_patch: int,
        L_patch_len: int,
        layers: int,
        nheads: int,
        is_linear: bool = False,
        input_dim: int = 2,
    ):
        super().__init__()
        self.channels = channels
        self.multipatch_num = multipatch_num
        self.attention_mask_type = attention_mask_type

        # Diffusion embedding
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=num_steps,
            embedding_dim=diffusion_embedding_dim,
        )

        # Side encoder
        self.side_encoder = SideEncoder(
            num_var=side_config["num_var"],
            var_emb_dim=side_config["var_emb"],
            time_emb_dim=side_config["time_emb"],
        )
        side_dim = self.side_encoder.total_emb_dim

        # Output projection
        self.output_projection1 = Conv1d_with_init(channels, channels, 1)

        # Multi-scale patch modules
        self.ts_downsample = nn.ModuleList()
        self.side_downsample = nn.ModuleList()
        self.patch_decoder = nn.ModuleList()

        for i in range(multipatch_num):
            patch_len = base_patch * (L_patch_len ** i)
            self.ts_downsample.append(
                TsPatchEmbedding(
                    L_patch_len=patch_len,
                    channels=input_dim,
                    d_model=channels,
                    dropout=0.0,
                )
            )
            self.side_downsample.append(
                SidePatchEmbedding(
                    L_patch_len=patch_len,
                    channels=side_dim,
                    d_model=side_dim,
                    dropout=0.0,
                )
            )
            self.patch_decoder.append(
                PatchDecoder(
                    L_patch_len=patch_len,
                    d_model=channels,
                    channels=1,
                )
            )

        # Multi-patch mixer
        self.multipatch_mixer = nn.Linear(multipatch_num, 1)

        # Residual layers
        self.residual_layers = nn.ModuleList([
            ResidualBlock(
                side_dim=side_dim,
                channels=channels,
                diffusion_embedding_dim=diffusion_embedding_dim,
                nheads=nheads,
                is_linear=is_linear,
            )
            for _ in range(layers)
        ])

    def get_mask(self, attr_len: int, len_list: list, device: torch.device) -> Tensor:
        """Generate attention mask for parallel patches."""
        total_len = sum(len_list) + attr_len
        mask = torch.zeros(total_len, total_len, device=device) - float("inf")
        mask[:attr_len, :] = 0
        mask[:, :attr_len] = 0
        start_id = attr_len
        for length in len_list:
            mask[start_id:start_id + length, start_id:start_id + length] = 0
            start_id += length
        return mask

    def forward(
        self,
        x_raw: Tensor,
        tp: Tensor,
        attr_emb_raw: Optional[Tensor],
        diffusion_step: Tensor,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Args:
            x_raw: Input time series (B, input_dim, K, L)
            tp: Time positions (B, L)
            attr_emb_raw: Optional attribute embeddings
            diffusion_step: Diffusion timestep (B,)
        Returns:
            Predicted noise (B, K, L) and auxiliary outputs
        """
        B, inputdim, K, L = x_raw.shape

        # Encode side information
        side_emb_raw = self.side_encoder(tp)
        diffusion_emb = self.diffusion_embedding(diffusion_step)

        # Multi-scale patch embedding
        x_list = []
        side_list = []
        for i in range(self.multipatch_num):
            x = self.ts_downsample[i](x_raw)
            side_emb = self.side_downsample[i](side_emb_raw)
            x_list.append(x)
            side_list.append(side_emb)

        # Attention mask
        if self.attention_mask_type == "full" or attr_emb_raw is None:
            attention_mask = None
        elif self.attention_mask_type == "parallel":
            attention_mask = self.get_mask(
                0,
                [x_list[i].shape[-1] for i in range(len(x_list))],
                device=x_raw.device,
            )

        # Concatenate multi-scale patches
        x_in = torch.cat(x_list, dim=-1)
        side_in = torch.cat(side_list, dim=-1)

        # Attribute embedding (if provided)
        if attr_emb_raw is None:
            attr_emb = torch.zeros_like(x_in)
        else:
            attr_emb = attr_emb_raw.permute(0, 3, 1, 2)

        B, _, Nk, Nl = x_in.shape
        _x_in = x_in
        skip = []

        # Residual blocks
        for layer in self.residual_layers:
            input_to_layer = x_in + _x_in + attr_emb
            x_in, skip_connection = layer(
                input_to_layer,
                side_in,
                diffusion_emb,
                attention_mask=attention_mask,
            )
            skip.append(skip_connection)

        # Aggregate skip connections
        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, Nk * Nl)
        x = self.output_projection1(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, Nk, Nl)

        # Decode multi-scale patches
        start_id = 0
        all_out = []
        for i in range(len(x_list)):
            x_out = x[:, :, :, start_id:start_id + x_list[i].shape[-1]]
            x_out = self.patch_decoder[i](x_out)
            x_out = x_out[:, :, :, :L]  # Truncate to original length
            all_out.append(x_out)
            start_id += x_list[i].shape[-1]

        # Mix multi-scale outputs
        all_out = torch.cat(all_out, dim=1)
        all_out = self.multipatch_mixer(all_out.permute(0, 2, 3, 1).contiguous())
        all_out = all_out.reshape(B, K, L)

        return all_out, {}


# =============================================================================
# TEdit Lightning Module
# =============================================================================

@Registry.register_model("tedit", aliases=["ted"])
class TEdit(BaseGeneratorModule, DiffusionMixin):
    """
    TEdit: Multi-scale Patch Diffusion for Attribute-Conditioned Time Series Generation.

    This model uses multi-scale patching with time/feature transformers for
    diffusion-based time series generation.
    """
    SUPPORTED_STAGES = ["finetune"]

    def _build_model(self) -> None:
        """Build the TEdit model architecture."""
        model_cfg = self.config.model
        data_cfg = self.config.data
        cond_cfg = self.config.condition

        # Model hyperparameters with defaults
        channels = getattr(model_cfg, "channels", 64)
        layers = getattr(model_cfg, "layers", 3)
        nheads = getattr(model_cfg, "nheads", 8)
        multipatch_num = getattr(model_cfg, "multipatch_num", 3)
        base_patch = getattr(model_cfg, "base_patch", 4)
        L_patch_len = getattr(model_cfg, "L_patch_len", 3)
        diffusion_steps = getattr(model_cfg, "diffusion_steps", 1000)
        diffusion_embedding_dim = getattr(model_cfg, "diffusion_embedding_dim", 128)
        attention_mask_type = getattr(model_cfg, "attention_mask_type", "full")
        noise_schedule = getattr(model_cfg, "noise_schedule", None)
        if noise_schedule is None:
            noise_schedule = getattr(model_cfg, "schedule", "cosine")
        beta_start = getattr(model_cfg, "beta_start", None)
        beta_end = getattr(model_cfg, "beta_end", None)
        is_linear = getattr(model_cfg, "is_linear", False)

        # Side encoder configuration
        side_var_emb = getattr(model_cfg, "side_var_emb", None)
        side_time_emb = getattr(model_cfg, "side_time_emb", None)
        var_emb_dim = side_var_emb if side_var_emb is not None else getattr(model_cfg, "var_emb", 64)
        time_emb_dim = side_time_emb if side_time_emb is not None else getattr(model_cfg, "time_emb", 64)

        side_config = {
            "num_var": data_cfg.n_var,
            "var_emb": var_emb_dim,
            "time_emb": time_emb_dim,
        }

        # Attribute conditioning configuration (TEdit is attribute-based, not text-based)
        if not cond_cfg.attribute.enabled:
            raise ValueError("TEdit requires attribute conditioning to be enabled")

        attr_dim = getattr(model_cfg, "attr_dim", None)
        if attr_dim is None:
            attr_dim = getattr(model_cfg, "attr_emb", None)
        if attr_dim is None:
            attr_dim = cond_cfg.attribute.output_dim
        self.attr_emb_dim = attr_dim
        self.num_attr_ops = [
            disc["num_classes"] for disc in cond_cfg.attribute.discrete_configs
        ]

        # Build attribute encoder and projector
        self.attr_encoder = AttributeEncoder(
            num_attr_ops=self.num_attr_ops,
            emb_dim=self.attr_emb_dim,
            device=self.config.device,
        )
        self.attr_projector = AttrProjectorAvgLinear(
            dim_in=self.attr_emb_dim,
            dim_out=channels,
        )

        # Core TEdit model
        self.core = TEditCore(
            channels=channels,
            multipatch_num=multipatch_num,
            num_steps=diffusion_steps,
            diffusion_embedding_dim=diffusion_embedding_dim,
            side_config=side_config,
            attention_mask_type=attention_mask_type,
            base_patch=base_patch,
            L_patch_len=L_patch_len,
            layers=layers,
            nheads=nheads,
            is_linear=is_linear,
            input_dim=1,  # noisy ts only (match legacy training input)
        )

        # Diffusion parameters
        self.num_steps = diffusion_steps
        if noise_schedule == "quad":
            beta_start = beta_start if beta_start is not None else 0.0001
            beta_end = beta_end if beta_end is not None else 0.5
            betas = self.quad_beta_schedule(self.num_steps, beta_start, beta_end)
        elif noise_schedule == "linear":
            if beta_start is None and beta_end is None:
                betas = self.linear_beta_schedule(self.num_steps)
            else:
                beta_start = beta_start if beta_start is not None else 0.0001
                beta_end = beta_end if beta_end is not None else 0.02
                betas = torch.linspace(float(beta_start), float(beta_end), self.num_steps)
        elif noise_schedule == "cosine":
            betas = self.cosine_beta_schedule(self.num_steps)
        else:
            raise ValueError(f"Unknown noise schedule: {noise_schedule}")

        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_sqrt = torch.sqrt(alpha_bar)
        one_minus_alpha_bar_sqrt = torch.sqrt(1.0 - alpha_bar)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alphas_cumprod", alpha_bar_sqrt)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", one_minus_alpha_bar_sqrt)
        self.register_buffer("alpha_bar_sqrt", alpha_bar_sqrt)
        self.register_buffer("alpha_bar_sqrt_inverse", 1.0 / alpha_bar_sqrt)
        self.register_buffer("one_minus_alpha_bar_sqrt", one_minus_alpha_bar_sqrt)
        self.register_buffer("reverse_coef1", 1.0 / torch.sqrt(alphas))
        self.register_buffer("reverse_coef2", (1.0 - alphas) / one_minus_alpha_bar_sqrt)

        if self.num_steps > 1:
            numerator = 1.0 - alpha_bar[:-1]
            denominator = 1.0 - alpha_bar[1:]
            sigma_sq = numerator / denominator * betas[1:]
            sigma = torch.sqrt(sigma_sq)
            reverse_coef2_determin = torch.sqrt(1.0 - alpha_bar[:-1])
            reverse_coef2_ddim = torch.sqrt(1.0 - alpha_bar[:-1] - sigma_sq)
            alpha_bar_sqrt_prev = alpha_bar_sqrt[:-1]
        else:
            sigma = torch.zeros(1, device=betas.device)
            reverse_coef2_determin = torch.zeros(1, device=betas.device)
            reverse_coef2_ddim = torch.zeros(1, device=betas.device)
            alpha_bar_sqrt_prev = torch.zeros(1, device=betas.device)

        self.register_buffer("sigma", sigma)
        self.register_buffer("reverse_coef2_determin", reverse_coef2_determin)
        self.register_buffer("reverse_coef2_ddim", reverse_coef2_ddim)
        self.register_buffer("alpha_bar_sqrt_prev", alpha_bar_sqrt_prev)

    def _encode_attributes(self, batch: Dict[str, Tensor]) -> Optional[Tensor]:
        """
        Encode attributes from batch.

        Args:
            batch: Batch dictionary containing "attrs" key

        Returns:
            Attribute embeddings (B, 1, 1, D) or None if not using condition
        """
        if not self.use_condition or "attrs" not in batch:
            return None

        attrs = batch["attrs"]  # (B, K)
        attr_emb = self.attr_encoder(attrs)  # (B, K, D)
        attr_emb = self.attr_projector(attr_emb)  # (B, 1, 1, D)
        return attr_emb

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Forward pass for training.

        Args:
            batch: Dictionary containing:
                - "ts": Time series (B, L, C) or (B, C, L)
                - "attrs": Attribute indices (B, K) for discrete attributes
                - "tp": Time positions (B, L)
        Returns:
            Dictionary with loss
        """
        ts = batch["ts"]  # (B, L, C) or (B, C, L)
        tp = batch.get("tp")

        # Ensure ts is (B, C, L)
        # If ts[-1] == n_var, it's (B, L, C) and needs permute
        # If ts[-1] == seq_len (and ts[-2] == n_var), it's already (B, C, L)
        if ts.dim() == 3 and ts.shape[-1] == self.config.data.n_var:
            ts = ts.permute(0, 2, 1)

        B, C, L = ts.shape

        # Generate time positions if not provided
        if tp is None:
            tp = torch.arange(L, device=ts.device).unsqueeze(0).expand(B, -1).float()

        # Sample diffusion timesteps
        t = torch.randint(0, self.num_steps, (B,), device=ts.device)

        # Forward diffusion
        noise = torch.randn_like(ts)
        noisy_ts, _ = self.q_sample(
            ts, t, noise,
            self.sqrt_alphas_cumprod,
            self.sqrt_one_minus_alphas_cumprod,
        )

        # Prepare input: only noisy ts (match legacy)
        # (B, C, L) -> (B, 1, C, L)
        x_input = noisy_ts.unsqueeze(1)

        # Process attribute condition
        attr_emb = self._encode_attributes(batch)

        # Predict noise
        noise_pred, _ = self.core(x_input, tp, attr_emb, t)

        # Compute loss against original noise
        loss = F.mse_loss(noise_pred, noise)

        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        condition: Tensor,
        n_samples: int = 1,
        sampler: str = "ddim",
        **kwargs: Any,
    ) -> Tensor:
        """
        Generate time series samples using DDPM sampling.

        Args:
            condition: Attribute tensor (B, K) - indices for discrete attributes
            n_samples: Number of samples per condition
        Returns:
            Generated time series (B * n_samples, L, C)
        """
        B = condition.shape[0]
        device = condition.device
        C = self.config.data.n_var
        L = self.config.data.seq_length

        # Encode attribute condition
        attr_emb = None
        if self.use_condition:
            attr_emb = self.attr_encoder(condition)  # (B, K, D)
            attr_emb = self.attr_projector(attr_emb)  # (B, 1, 1, D)

        # Expand for n_samples
        if n_samples > 1:
            if attr_emb is not None:
                attr_emb = attr_emb.repeat_interleave(n_samples, dim=0)

        total_samples = B * n_samples

        # Generate time positions
        tp = torch.arange(L, device=device).unsqueeze(0).expand(total_samples, -1).float()

        # Start from noise
        x = torch.randn(total_samples, C, L, device=device)

        sampler = kwargs.get("sampler", sampler)
        eta = kwargs.get("eta", 0.0)

        # DDPM/DDIM reverse process
        for t in reversed(range(self.num_steps)):
            t_batch = torch.full((total_samples,), t, device=device, dtype=torch.long)

            # Only noisy input channel (match training input)
            x_input = x.unsqueeze(1)

            # Predict noise
            noise_pred, _ = self.core(x_input, tp, attr_emb, t_batch)

            if sampler == "ddpm":
                x = self._ddpm_reverse(x, noise_pred, t_batch)
            else:
                x = self._ddim_reverse(x, noise_pred, t_batch, eta=eta)

        return x.permute(0, 2, 1)  # (B * n_samples, L, C)

    def _ddpm_reverse(self, x_t: Tensor, pred_noise: Tensor, t: Tensor) -> Tensor:
        """DDPM reverse step."""
        coef1 = self.reverse_coef1[t].view(-1, 1, 1)
        coef2 = self.reverse_coef2[t].view(-1, 1, 1)
        x_prev = coef1 * (x_t - coef2 * pred_noise)

        noise = torch.randn_like(x_t)
        t_prev = (t - 1).clamp(min=0)
        sigma = self.sigma[t_prev].view(-1, 1, 1)
        mask = (t > 0).float().view(-1, 1, 1)
        x_prev = x_prev + mask * sigma * noise
        return x_prev

    def _ddim_reverse(self, x_t: Tensor, pred_noise: Tensor, t: Tensor, eta: float = 0.0) -> Tensor:
        """DDIM reverse step (deterministic when eta=0)."""
        coef1 = self.one_minus_alpha_bar_sqrt[t].view(-1, 1, 1)
        coef2 = self.alpha_bar_sqrt_inverse[t].view(-1, 1, 1)
        x0_pred = (x_t - coef1 * pred_noise) * coef2

        t_prev = (t - 1).clamp(min=0)
        coef_prev = self.alpha_bar_sqrt_prev[t_prev].view(-1, 1, 1)

        if eta == 0.0:
            coef2_prev = self.reverse_coef2_determin[t_prev].view(-1, 1, 1)
            coef3 = 0.0
        else:
            coef2_prev = self.reverse_coef2_ddim[t_prev].view(-1, 1, 1)
            coef3 = self.sigma[t_prev].view(-1, 1, 1)

        noise = torch.randn_like(x_t)
        x_prev = coef_prev * x0_pred + coef2_prev * pred_noise + coef3 * noise

        mask = (t == 0).view(-1, 1, 1)
        x_prev = torch.where(mask, x0_pred, x_prev)
        return x_prev
