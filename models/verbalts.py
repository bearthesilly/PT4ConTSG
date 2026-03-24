"""
VerbalTS: Multi-view Noise Estimation for Text-Conditioned Time Series Generation.

This module implements the VerbalTS model with multi-scale patch processing and
adaptive conditioning mechanisms for conditional time series generation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from contsg.models.base import BaseGeneratorModule, DiffusionMixin
from contsg.registry import Registry


# =============================================================================
# Helper Functions and Modules
# =============================================================================


class TextProjectorMVarMScaleMStep(nn.Module):
    """
    Project text embeddings into (n_var, n_scale) grids with diffusion-step context.

    This mirrors the legacy VerbalTS projector behavior:
    - Cross-attention over variable, scale, and diffusion-step embeddings
    - Output shape: (B, n_var, n_scale, dim_out)
    """

    def __init__(
        self,
        n_var: int,
        n_scale: int,
        n_steps: int,
        n_stages: int,
        dim_in: int = 128,
        dim_out: int = 128,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_var = n_var
        self.n_scale = n_scale
        self.n_steps = n_steps
        self.n_stages = n_stages
        self.seg_size = n_steps // n_stages + 1

        self.var_emb = nn.Parameter(torch.zeros((1, n_var, dim_in)))
        self.scale_emb = nn.Parameter(torch.zeros((1, n_scale, dim_in)))
        self.step_emb = nn.Parameter(torch.zeros((1, n_stages, dim_in)))

        var_layer = nn.TransformerDecoderLayer(
            d_model=dim_in, nhead=8, dim_feedforward=64, activation="gelu", batch_first=True
        )
        self.var_cross_attn = nn.TransformerDecoder(var_layer, num_layers=2)

        scale_layer = nn.TransformerDecoderLayer(
            d_model=dim_in, nhead=8, dim_feedforward=64, activation="gelu", batch_first=True
        )
        self.scale_cross_attn = nn.TransformerDecoder(scale_layer, num_layers=2)

        step_layer = nn.TransformerDecoderLayer(
            d_model=dim_in, nhead=8, dim_feedforward=64, activation="gelu", batch_first=True
        )
        self.step_cross_attn = nn.TransformerDecoder(step_layer, num_layers=2)

        self.proj_out = nn.Linear(dim_in, dim_out)

    def forward(self, attr: torch.Tensor, diffusion_step: torch.Tensor) -> torch.Tensor:
        """
        Args:
            attr: Text embeddings, shape (B, S, D) or (B, D)
            diffusion_step: Diffusion timestep indices, shape (B,)

        Returns:
            Tensor of shape (B, n_var, n_scale, dim_out)
        """
        if attr.dim() == 2:
            attr = attr.unsqueeze(1)

        batch_size = attr.shape[0]
        var_emb = self.var_emb.expand([batch_size, -1, -1])
        mvar_attr = self.var_cross_attn(tgt=var_emb, memory=attr)
        mvar_attr = mvar_attr[:, :, None, :]

        scale_emb = self.scale_emb.expand([batch_size, -1, -1])
        mscale_attr = self.scale_cross_attn(tgt=scale_emb, memory=attr)
        mscale_attr = mscale_attr[:, None, :, :].expand([-1, self.n_var, -1, -1])

        step_emb = self.step_emb.expand([batch_size, -1, -1])
        mstep_attr = self.step_cross_attn(tgt=step_emb, memory=attr)
        indices = diffusion_step // self.seg_size
        indices = indices[:, None, None]
        mstep_attr = torch.gather(
            mstep_attr,
            dim=1,
            index=indices.expand([-1, -1, mstep_attr.shape[-1]]),
        )
        mstep_attr = mstep_attr[:, None, :, :].expand([-1, self.n_var, -1, -1])

        mix_attr = mvar_attr + mscale_attr + mstep_attr
        return self.proj_out(mix_attr)


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


def get_torch_cross_trans(heads: int = 8, layers: int = 1, channels: int = 64) -> nn.Module:
    """Create a transformer decoder for cross-attention."""
    decoder_layer = nn.TransformerDecoderLayer(
        d_model=channels,
        nhead=heads,
        dim_feedforward=64,
        activation="gelu",
        batch_first=True,
    )
    return nn.TransformerDecoder(decoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels: int, out_channels: int, kernel_size: int) -> nn.Conv1d:
    """Create Conv1d layer with Kaiming initialization."""
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


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
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step: torch.Tensor) -> torch.Tensor:
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps: int, dim: int = 64) -> torch.Tensor:
        steps = torch.arange(num_steps).unsqueeze(1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)
        table = steps * frequencies
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class SideEncoder(nn.Module):
    """Side information encoder for time and variable embeddings."""

    def __init__(self, num_var: int, var_emb_dim: int, time_emb_dim: int, device: str):
        super().__init__()
        self.num_var = num_var
        self.var_emb_dim = var_emb_dim
        self.time_emb_dim = time_emb_dim
        self.device = device
        self.total_emb_dim = var_emb_dim + time_emb_dim

        self.var_emb = nn.Embedding(num_embeddings=num_var, embedding_dim=var_emb_dim)
        self.register_buffer("var_ids", torch.arange(num_var))

    def time_embedding(self, pos: torch.Tensor, d_model: int = 128) -> torch.Tensor:
        """Sinusoidal position embedding."""
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model, device=pos.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2, device=pos.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, tp: torch.Tensor) -> torch.Tensor:
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


class TsPatchEmbedding(nn.Module):
    """Time series patch embedding."""

    def __init__(self, L_patch_len: int, channels: int, d_model: int, dropout: float):
        super().__init__()
        self.L_patch_len = L_patch_len
        self.padding_patch_layer = nn.ReplicationPad2d((0, L_patch_len, 0, 0))
        self.value_embedding = nn.Sequential(
            nn.Linear(L_patch_len * channels, d_model),
            nn.ReLU(),
        )

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
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

    def __init__(self, L_patch_len: int, channels: int, d_model: int, dropout: float):
        super().__init__()
        self.L_patch_len = L_patch_len
        self.padding_patch_layer = nn.ReplicationPad2d((0, L_patch_len, 0, 0))
        self.value_embedding = nn.Sequential(
            nn.Linear(L_patch_len * channels, d_model),
        )

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
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
    """Patch decoder to reconstruct time series from patch embeddings."""

    def __init__(self, L_patch_len: int, d_model: int, channels: int):
        super().__init__()
        self.L_patch_len = L_patch_len
        self.channels = channels
        self.linear = nn.Linear(d_model, L_patch_len * channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Patch embeddings (B, D, n_var, Nl)
        Returns:
            Time series (B, C, n_var, L)
        """
        B, D, n_var, Nl = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.linear(x)
        x = x.reshape(B, n_var, Nl, self.L_patch_len, self.channels).permute(0, 4, 1, 2, 3).contiguous()
        x = x.reshape(B, self.channels, n_var, Nl * self.L_patch_len)
        return x


# =============================================================================
# Residual Block with Multi-conditioning
# =============================================================================

class ResidualBlock(nn.Module):
    """Residual block with time/feature attention and conditioning."""

    def __init__(
        self,
        side_dim: int,
        channels: int,
        diffusion_embedding_dim: int,
        nheads: int,
        condition_type: str = "add",
    ):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.side_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

        if condition_type == "add":
            pass
        elif condition_type == "cross_attention":
            self.condition_cross_attention = get_torch_cross_trans(heads=nheads, layers=1, channels=channels)
        elif condition_type == "adaLN":
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 3 * channels, bias=True),
            )

    def forward_time(self, y: torch.Tensor, base_shape: tuple, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply temporal attention."""
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(0, 2, 1), mask=attention_mask).permute(0, 2, 1)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y: torch.Tensor, base_shape: tuple, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply feature attention."""
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(0, 2, 1), mask=attention_mask).permute(0, 2, 1)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward_cross_attention(self, y: torch.Tensor, cond: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply cross-attention with condition."""
        B, channel, K, L = y.shape
        y = y.reshape(B, channel, K, L).permute(0, 2, 3, 1).reshape(B * K, L, channel)
        cond = cond.reshape(B, channel, K, L).permute(0, 2, 3, 1).reshape(B * K, L, channel)
        y = self.condition_cross_attention(tgt=y, memory=cond, memory_mask=attention_mask).permute(0, 2, 1)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3)
        return y

    def modulate(self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Apply adaptive layer normalization modulation."""
        return x * (1 + scale) + shift

    def forward(
        self,
        x: torch.Tensor,
        side_emb: torch.Tensor,
        attr_emb: torch.Tensor,
        diffusion_emb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        condition_type: str = "add",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of residual block.

        Args:
            x: Input tensor (B, C, K, L)
            side_emb: Side information (B, D_side, K, L)
            attr_emb: Condition embedding (B, C, K, L)
            diffusion_emb: Diffusion timestep embedding (B, D)
            attention_mask: Optional attention mask
            condition_type: Conditioning mechanism ("add", "cross_attention", "adaLN")

        Returns:
            Tuple of (residual output, skip connection)
        """
        # Apply conditioning
        if condition_type == "add":
            x = x + attr_emb
        elif condition_type == "cross_attention":
            x = self.forward_cross_attention(x, attr_emb, attention_mask=attention_mask)
        elif condition_type == "adaLN":
            gamma, beta, alpha = self.adaLN_modulation(attr_emb.permute(0, 2, 3, 1)).chunk(3, dim=-1)
            gamma, beta, alpha = gamma.permute(0, 3, 1, 2), beta.permute(0, 3, 1, 2), alpha.permute(0, 3, 1, 2)

        B, channel, K, L = x.shape
        base_shape = x.shape

        # Add diffusion timestep
        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1).unsqueeze(-1)
        y = x + diffusion_emb
        if condition_type == "adaLN":
            y = self.modulate(y, gamma, beta)

        # Apply attention
        y = self.forward_time(y, base_shape, attention_mask)
        y = self.forward_feature(y, base_shape, None)

        # Apply alpha scaling for adaLN
        if condition_type == "adaLN":
            y = y.reshape(B, channel, K, L)
            y = alpha * y
            y = y.reshape(B, channel, K * L)

        y = y.reshape(B, channel, K * L)
        y = self.mid_projection(y)

        # Add side information
        _, side_dim, _, _ = side_emb.shape
        side_emb = side_emb.reshape(B, side_dim, K * L)
        side_emb = self.side_projection(side_emb)
        y = y + side_emb

        # Gated activation
        gate, filter_ = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter_)
        y = self.output_projection(y)

        # Split into residual and skip
        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip


# =============================================================================
# VerbalTS Core Module
# =============================================================================

class VerbalTSCore(nn.Module):
    """Core VerbalTS architecture with multi-scale patch processing."""

    def __init__(self, config: Dict[str, Any], inputdim: int = 1):
        super().__init__()
        self.config = config
        self.n_var = config["n_var"]
        self.channels = config["channels"]
        self.multipatch_num = config["multipatch_num"]
        if self.config.get("condition_type") == "cross_attn":
            self.config["condition_type"] = "cross_attention"

        # Diffusion timestep embedding
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        # Side encoder (time + variable)
        self.side_encoder = SideEncoder(
            num_var=self.n_var,
            var_emb_dim=config["side"]["var_emb"],
            time_emb_dim=config["side"]["time_emb"],
            device=config["device"],
        )
        side_dim = self.side_encoder.total_emb_dim

        # Multi-scale patch processing
        self.attention_mask_type = config.get("attention_mask_type", "parallel")
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)

        self.ts_downsample = nn.ModuleList([])
        self.side_downsample = nn.ModuleList([])
        self.patch_decoder = nn.ModuleList([])

        for i in range(self.multipatch_num):
            patch_len = config["base_patch"] * (config["L_patch_len"] ** i)
            self.ts_downsample.append(
                TsPatchEmbedding(L_patch_len=patch_len, channels=inputdim, d_model=self.channels, dropout=0)
            )
            self.patch_decoder.append(
                PatchDecoder(L_patch_len=patch_len, d_model=self.channels, channels=1)
            )
            self.side_downsample.append(
                SidePatchEmbedding(L_patch_len=patch_len, channels=side_dim, d_model=side_dim, dropout=0)
            )

        self.multipatch_mixer = nn.Linear(self.multipatch_num, 1)

        # Residual layers
        self.residual_layers = nn.ModuleList([
            ResidualBlock(
                side_dim=side_dim,
                channels=self.channels,
                diffusion_embedding_dim=config["diffusion_embedding_dim"],
                nheads=config["nheads"],
                condition_type=config.get("condition_type", "add"),
            )
            for _ in range(config["layers"])
        ])

    def forward(
        self,
        x_raw: torch.Tensor,
        tp: torch.Tensor,
        attr_emb_raw: Optional[torch.Tensor],
        diffusion_step: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass of VerbalTS.

        Args:
            x_raw: Input time series (B, C, n_var, L)
            tp: Time positions (B, L)
            attr_emb_raw: Condition embeddings (B, n_var, n_scale, D) or None
            diffusion_step: Diffusion timestep (B,)

        Returns:
            Tuple of (predicted noise, loss dict)
        """
        B_raw, inputdim, n_var, L = x_raw.shape

        # Encode side information
        side_emb_raw = self.side_encoder(tp)
        diffusion_emb = self.diffusion_embedding(diffusion_step)

        # Multi-scale patch embedding
        x_list = []
        side_list = []
        scale_length = []
        for i in range(self.multipatch_num):
            x = self.ts_downsample[i](x_raw)
            side_emb = self.side_downsample[i](side_emb_raw)
            x_list.append(x)
            side_list.append(side_emb)
            scale_length.append(x.shape[-1])

        # Prepare attention mask
        if self.attention_mask_type == "full" or attr_emb_raw is None:
            attention_mask = None
        elif self.attention_mask_type == "parallel":
            attention_mask = self.get_mask(0, [x_list[i].shape[-1] for i in range(len(x_list))], device=x_raw.device)

        # Concatenate multi-scale inputs
        x_in = torch.cat(x_list, dim=-1)
        side_in = torch.cat(side_list, dim=-1)

        # Prepare condition embedding
        if attr_emb_raw is None:
            attr_emb = torch.zeros_like(x_in)
        else:
            if "scale" in self.config.get("text_projector", ""):
                # Multi-scale condition projection
                assert len(scale_length) == attr_emb_raw.shape[2]
                mscale_attr_list = []
                for i in range(len(scale_length)):
                    tmp_scale_attr = attr_emb_raw[:, :, i:i+1, :].expand([-1, -1, scale_length[i], -1])
                    mscale_attr_list.append(tmp_scale_attr)
                attr_emb = torch.cat(mscale_attr_list, dim=2)
            else:
                attr_emb = attr_emb_raw.expand([-1, -1, x_in.shape[-1], -1])
            attr_emb = attr_emb.permute(0, 3, 1, 2)

        B, _, Nk, Nl = x_in.shape
        _x_in = x_in
        skip = []

        # Apply residual blocks
        for layer in self.residual_layers:
            x_in, skip_connection = layer(
                x_in + _x_in,
                side_in,
                attr_emb,
                diffusion_emb,
                attention_mask=attention_mask,
                condition_type=self.config.get("condition_type", "add"),
            )
            skip.append(skip_connection)

        # Aggregate skip connections
        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))

        # Output projection
        x = x.reshape(B, self.channels, Nk * Nl)
        x = self.output_projection1(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, Nk, Nl)

        # Multi-scale patch decoding
        start_id = 0
        all_out = []
        for i in range(len(x_list)):
            x_out = x[:, :, :, start_id:start_id+x_list[i].shape[-1]]
            x_out = self.patch_decoder[i](x_out)
            x_out = x_out[:, :, :, :L]
            all_out.append(x_out)
            start_id += x_list[i].shape[-1]

        # Mix multi-scale outputs
        all_out = torch.cat(all_out, dim=1)
        all_out = self.multipatch_mixer(all_out.permute(0, 2, 3, 1).contiguous())
        all_out = all_out.reshape((B_raw, n_var, L))

        return all_out, {}

    def get_mask(self, attr_len: int, len_list: list, device: str = "cpu") -> torch.Tensor:
        """Create attention mask for parallel multi-scale processing."""
        total_len = sum(len_list) + attr_len
        mask = torch.zeros(total_len, total_len, device=device) - float("inf")
        mask[:attr_len, :] = 0
        mask[:, :attr_len] = 0
        start_id = attr_len
        for i in range(len(len_list)):
            mask[start_id:start_id+len_list[i], start_id:start_id+len_list[i]] = 0
            start_id += len_list[i]
        return mask


# =============================================================================
# VerbalTS Lightning Module
# =============================================================================

@Registry.register_model("verbalts", aliases=["vts"])
class VerbalTSModule(BaseGeneratorModule, DiffusionMixin):
    """VerbalTS: Multi-view noise estimation with adaLN for text-conditioned generation.

    Multi-view noise estimation with adaptive conditioning and multi-scale
    patch processing for text-conditioned time series generation.

    Features:
    - Multi-scale patch embedding for capturing patterns at different granularities
    - Adaptive Layer Normalization (adaLN) or cross-attention conditioning
    - Time and variable side information encoding
    - DDPM/DDIM sampling strategies

    Config Parameters:
        channels: Model hidden channels (default: 64)
        layers: Number of residual blocks (default: 3)
        nheads: Number of attention heads (default: 8)
        L_patch_len: Local patch length multiplier (default: 3)
        multipatch_num: Number of multi-scale patches (default: 3)
        base_patch: Base patch size (default: 4)
        diffusion_steps: Number of diffusion timesteps (default: 1000)
        condition_type: Conditioning mechanism ("adaLN", "cross_attention", "add")
        attention_mask_type: Attention mask type ("full", "parallel")
    """

    SUPPORTED_STAGES = ["finetune"]

    def _build_model(self) -> None:
        """Build VerbalTS architecture."""
        cfg = self.config.model
        data_cfg = self.config.data
        cond_cfg = self.config.condition
        text_projector_name = (
            getattr(cond_cfg.text, "text_projector", None)
            or getattr(cfg, "text_projector", "scale")
        )
        condition_type = getattr(cfg, "condition_type", "adaLN")
        if condition_type == "cross_attn":
            condition_type = "cross_attention"
        attention_mask_type = getattr(cfg, "attention_mask_type", "parallel")
        diffusion_embedding_dim = cfg.diffusion_embedding_dim or (cfg.channels * 2)
        side_var_emb = getattr(cfg, "side_var_emb", getattr(cfg, "var_emb", 16))
        side_time_emb = getattr(cfg, "side_time_emb", getattr(cfg, "time_emb", 112))

        # Build model config dict
        model_config = {
            "n_var": data_cfg.n_var,
            "seq_length": data_cfg.seq_length,
            "channels": cfg.channels,
            "layers": cfg.layers,
            "nheads": cfg.nheads,
            "dropout": cfg.dropout,
            "L_patch_len": getattr(cfg, "L_patch_len", 3),
            "multipatch_num": getattr(cfg, "multipatch_num", 3),
            "base_patch": getattr(cfg, "base_patch", 4),
            "num_steps": getattr(cfg, "diffusion_steps", 1000),
            "diffusion_embedding_dim": diffusion_embedding_dim,
            "condition_type": condition_type,
            "attention_mask_type": attention_mask_type,
            "text_projector": text_projector_name,
            "device": self.config.device,
            "side": {
                "var_emb": side_var_emb,
                "time_emb": side_time_emb,
            },
        }

        # Core VerbalTS model
        self.verbalts = VerbalTSCore(model_config, inputdim=1)

        # Diffusion schedule
        self.num_steps = model_config["num_steps"]
        noise_schedule = getattr(cfg, "noise_schedule", "cosine")
        beta_start = getattr(cfg, "beta_start", None)
        beta_end = getattr(cfg, "beta_end", None)

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
                betas = torch.linspace(beta_start, beta_end, self.num_steps)
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

        # Condition projector (if needed)
        self.cond_projector = None
        self.cond_projector_type = None
        self.text_embedding_key = cond_cfg.text.embedding_key
        if self.use_condition:
            if cond_cfg.text.enabled:
                text_dim = cond_cfg.text.input_dim
                if cond_cfg.text.output_dim is not None:
                    output_dim = cond_cfg.text.output_dim
                else:
                    output_dim = cfg.channels

                if text_projector_name == "var_scale_diffstep_multi":
                    num_stages = cond_cfg.text.num_stages or 1
                    self.cond_projector = TextProjectorMVarMScaleMStep(
                        n_var=data_cfg.n_var,
                        n_scale=model_config["multipatch_num"],
                        n_steps=model_config["num_steps"],
                        n_stages=num_stages,
                        dim_in=text_dim,
                        dim_out=output_dim,
                    )
                    self.cond_projector_type = "diffstep"
                else:
                    # Simple per-scale linear projection
                    self.cond_projector = nn.ModuleList([
                        nn.Linear(text_dim, output_dim)
                        for _ in range(model_config["multipatch_num"])
                    ])
                    self.cond_projector_type = "linear"

    def _project_text_condition(
        self,
        cap_emb: torch.Tensor,
        diffusion_step: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Project raw text embeddings to (B, n_var, n_scale, D)."""
        if self.cond_projector is None:
            raise ValueError("Condition projector is not initialized")

        if self.cond_projector_type == "diffstep":
            if diffusion_step is None:
                raise ValueError("diffusion_step is required for var_scale_diffstep_multi")
            return self.cond_projector(cap_emb, diffusion_step)

        if cap_emb.dim() == 3:
            cap_emb = cap_emb.mean(dim=1)

        attr_emb_list = [proj(cap_emb) for proj in self.cond_projector]
        attr_emb = torch.stack(attr_emb_list, dim=1)
        attr_emb = attr_emb.unsqueeze(1).expand(-1, self.config.data.n_var, -1, -1)
        return attr_emb

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.

        Args:
            batch: Dictionary containing:
                - ts: Time series (B, L, C)
                - tp: Time positions (B, L)
                - cap_emb: Text embeddings (B, D) if use_condition is True

        Returns:
            Dictionary with "loss" key
        """
        # Unpack batch
        ts = batch["ts"].float()  # (B, L, C)
        tp = batch["tp"].float()  # (B, L)

        ts = ts.permute(0, 2, 1).unsqueeze(1)  # (B, 1, C, L)

        B = ts.shape[0]

        # Sample random timesteps
        t = torch.randint(0, self.num_steps, (B,), device=self.device)

        # Add noise
        noise = torch.randn_like(ts)
        noisy_ts, _ = self.q_sample(
            ts,
            t,
            noise=noise,
            sqrt_alphas_cumprod=self.sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod,
        )

        # Get condition embedding
        attr_emb = None
        if self.use_condition and self.cond_projector is not None:
            if self.text_embedding_key in batch:
                cap_emb = batch[self.text_embedding_key].float()
                attr_emb = self._project_text_condition(cap_emb, diffusion_step=t)

        # Predict noise
        pred_noise, _ = self.verbalts(noisy_ts, tp, attr_emb, t)

        # Compute loss
        loss = F.mse_loss(pred_noise, noise.squeeze(1))

        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        condition: torch.Tensor,
        n_samples: int = 1,
        sampler: str = "ddim",
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Generate time series samples.

        Args:
            condition: Text embedding (B, D) or (B, S, D)
            n_samples: Number of samples per condition
            sampler: Sampling strategy ("ddpm" or "ddim")
            **kwargs: Additional parameters (tp, etc.)

        Returns:
            Generated samples (n_samples, B, L, C)
        """
        B = condition.shape[0]
        device = condition.device

        # Get time positions
        if "tp" in kwargs:
            tp = kwargs["tp"]
        else:
            seq_len = self.config.data.seq_length
            tp = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1).float()

        attr_emb = None
        if self.cond_projector is not None and self.cond_projector_type != "diffstep":
            attr_emb = self._project_text_condition(condition)

        samples = []
        for _ in range(n_samples):
            # Start from random noise
            x = torch.randn(B, 1, self.config.data.n_var, self.config.data.seq_length, device=device)

            # Reverse diffusion
            for t_int in reversed(range(self.num_steps)):
                t = torch.full((B,), t_int, device=device, dtype=torch.long)

                # Predict noise
                step_attr_emb = attr_emb
                if self.cond_projector is not None and self.cond_projector_type == "diffstep":
                    step_attr_emb = self._project_text_condition(condition, diffusion_step=t)
                pred_noise, _ = self.verbalts(x, tp, step_attr_emb, t)
                pred_noise = pred_noise.unsqueeze(1)  # (B, 1, C, L)

                # Reverse step
                if sampler == "ddpm":
                    x = self._ddpm_reverse(x, pred_noise, t)
                else:
                    x = self._ddim_reverse(x, pred_noise, t)

            # Reshape: (B, 1, C, L) -> (B, L, C)
            x = x.squeeze(1).permute(0, 2, 1)
            samples.append(x)

        return torch.stack(samples, dim=0)

    def _ddpm_reverse(self, x_t: torch.Tensor, pred_noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """DDPM reverse step."""
        coef1 = self.reverse_coef1[t].view(-1, 1, 1, 1)
        coef2 = self.reverse_coef2[t].view(-1, 1, 1, 1)
        x_prev = coef1 * (x_t - coef2 * pred_noise)

        noise = torch.randn_like(x_t)
        t_prev = (t - 1).clamp(min=0)
        sigma = self.sigma[t_prev].view(-1, 1, 1, 1)
        mask = (t > 0).float().view(-1, 1, 1, 1)
        x_prev = x_prev + mask * sigma * noise

        return x_prev

    def _ddim_reverse(self, x_t: torch.Tensor, pred_noise: torch.Tensor, t: torch.Tensor, eta: float = 0.0) -> torch.Tensor:
        """DDIM reverse step (deterministic when eta=0)."""
        coef1 = self.one_minus_alpha_bar_sqrt[t].view(-1, 1, 1, 1)
        coef2 = self.alpha_bar_sqrt_inverse[t].view(-1, 1, 1, 1)
        x0_pred = (x_t - coef1 * pred_noise) * coef2

        t_prev = (t - 1).clamp(min=0)
        coef_prev = self.alpha_bar_sqrt_prev[t_prev].view(-1, 1, 1, 1)

        if eta == 0.0:
            coef2_prev = self.reverse_coef2_determin[t_prev].view(-1, 1, 1, 1)
            coef3 = 0.0
        else:
            coef2_prev = self.reverse_coef2_ddim[t_prev].view(-1, 1, 1, 1)
            coef3 = self.sigma[t_prev].view(-1, 1, 1, 1)

        noise = torch.randn_like(x_t)
        x_prev = coef_prev * x0_pred + coef2_prev * pred_noise + coef3 * noise

        mask = (t == 0).view(-1, 1, 1, 1)
        x_prev = torch.where(mask, x0_pred, x_prev)

        return x_prev
