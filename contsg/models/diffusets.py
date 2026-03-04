"""
DiffuSETS: Latent Diffusion for Text-Conditioned Time Series Generation.

This module implements the DiffuSETS model with two-stage training:
- Stage 1 (vae_pretrain): Train VAE to compress time series to latent space
- Stage 2 (finetune): Train U-Net diffusion model in latent space with text conditioning
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence

from contsg.models.base import BaseGeneratorModule, DiffusionMixin
from contsg.registry import Registry


# =============================================================================
# VAE Components
# =============================================================================

class VAESelfAttention(nn.Module):
    """Self-attention module for VAE."""

    def __init__(self, n_heads: int, d_embed: int, in_proj_bias: bool = True, out_proj_bias: bool = True):
        super().__init__()
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: Tensor, causal_mask: bool = False) -> Tensor:
        """
        Args:
            x: (B, S, D)
        Returns:
            Output: (B, S, D)
        """
        batch_size, seq_len, d_embed = x.shape
        interim_shape = (batch_size, seq_len, self.n_heads, self.d_head)

        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        output = weight @ v
        output = output.transpose(1, 2).reshape(batch_size, seq_len, d_embed)
        return self.out_proj(output)


class VAEAttentionBlock(nn.Module):
    """Attention block for VAE."""

    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = VAESelfAttention(1, channels)

    def forward(self, x: Tensor) -> Tensor:
        """Args: x (B, C, L). Returns: (B, C, L)."""
        residue = x
        x = x.transpose(-1, -2)  # (B, L, C)
        x = self.attention(x)
        x = x.transpose(-1, -2)  # (B, C, L)
        return x + residue


class VAEResidualBlock(nn.Module):
    """Residual block for VAE."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        residue = x
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)
        return x + self.residual_layer(residue)


class VAEEncoder(nn.Module):
    """VAE Encoder: (B, L, C) -> (B, 4, L/8)."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv1d(in_channels, 128, kernel_size=3, padding=1),
            VAEResidualBlock(128, 128),
            VAEResidualBlock(128, 128),
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=0),
            VAEResidualBlock(128, 256),
            VAEResidualBlock(256, 256),
            nn.Conv1d(256, 256, kernel_size=3, stride=2, padding=0),
            VAEResidualBlock(256, 512),
            VAEResidualBlock(512, 512),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=0),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEAttentionBlock(512),
            VAEResidualBlock(512, 512),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            nn.Conv1d(512, 8, kernel_size=3, padding=1),
            nn.Conv1d(8, 8, kernel_size=1, padding=0),
        ])

    def forward(self, x: Tensor, noise: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            x: (B, L, C)
            noise: Optional (B, 4, L/8)
        Returns:
            z: (B, 4, L/8), mean: (B, 4, L/8), log_var: (B, 4, L/8)
        """
        x = x.transpose(1, 2)  # (B, C, L)

        for layer in self.layers:
            if isinstance(layer, nn.Conv1d) and layer.stride[0] == 2:
                x = F.pad(x, (0, 1))
            x = layer(x)

        mean, log_var = torch.chunk(x, 2, dim=1)
        log_var = torch.clamp(log_var, -30, 20)
        stdev = (log_var * 0.5).exp()

        if noise is None:
            noise = torch.randn_like(stdev)
        z = mean + stdev * noise
        z = z * 0.18215  # Scale factor from Legacy

        return z, mean, log_var


class VAEDecoder(nn.Module):
    """VAE Decoder: (B, 4, L/8) -> (B, L, C)."""

    def __init__(self, out_channels: int):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv1d(4, 4, kernel_size=1, padding=0),
            nn.Conv1d(4, 512, kernel_size=3, padding=1),
            VAEResidualBlock(512, 512),
            VAEAttentionBlock(512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            VAEResidualBlock(512, 256),
            VAEResidualBlock(256, 256),
            VAEResidualBlock(256, 256),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            VAEResidualBlock(256, 128),
            VAEResidualBlock(128, 128),
            VAEResidualBlock(128, 128),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv1d(128, out_channels, kernel_size=3, padding=1),
        ])

    def forward(self, z: Tensor) -> Tensor:
        """Args: z (B, 4, L/8). Returns: (B, L, C)."""
        x = z / 0.18215
        for layer in self.layers:
            x = layer(x)
        return x.transpose(1, 2)  # (B, L, C)


# =============================================================================
# U-Net Diffusion Components
# =============================================================================

class DiffusionTimeEmbedding(nn.Module):
    """Time step embedding for diffusion."""

    def __init__(self, num_steps: int, n_channels: int = 4, dim_embed: int = 64, dim_latent: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(dim_embed, dim_latent)
        self.fc2 = nn.Linear(dim_latent, n_channels)

        # Pre-compute embedding table
        t = torch.arange(num_steps) + 1
        half_dim = dim_embed // 2
        emb = 10.0 / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        self.register_buffer("embeddings", emb)

    def forward(self, t: Tensor) -> Tensor:
        """Args: t (B,). Returns: (B, n_channels)."""
        emb = self.embeddings[t]
        out = self.fc1(emb)
        out = F.mish(out)
        out = self.fc2(out)
        return out


class UNetSelfAttention(nn.Module):
    """Self-attention for U-Net."""

    def __init__(self, channels: int, hidden_dim: int = 16, num_heads: int = 4):
        super().__init__()
        self.to_q = nn.Linear(channels, hidden_dim)
        self.to_k = nn.Linear(channels, hidden_dim)
        self.to_v = nn.Linear(channels, hidden_dim)
        self.to_out = nn.Linear(hidden_dim, channels)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        """Args: x (B, L, C). Returns: (B, L, C)."""
        x_norm = self.norm(x)
        q = self.to_q(x_norm)
        k = self.to_k(x_norm)
        v = self.to_v(x_norm)
        h, _ = self.attention(q, k, v)
        return self.to_out(h) + x


class UNetCrossAttention(nn.Module):
    """Cross-attention with text conditioning."""

    def __init__(self, channels: int, hidden_dim: int = 16, num_heads: int = 4, text_embed_dim: int = 768):
        super().__init__()
        self.to_q = nn.Linear(channels, hidden_dim)
        self.to_k = nn.Linear(channels, hidden_dim)
        self.to_v = nn.Linear(channels, hidden_dim)
        self.to_out = nn.Linear(hidden_dim, channels)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)
        self.value_combiner = nn.Linear(text_embed_dim, channels)
        self.cond_norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor, text_embed: Optional[Tensor]) -> Tensor:
        """Args: x (B, L, C), text_embed (B, 1, D) or None. Returns: (B, L, C)."""
        x_norm = self.norm(x)
        q = self.to_q(x_norm)

        if text_embed is not None and text_embed.numel() > 0:
            combined = text_embed
        else:
            combined = torch.zeros(x.shape[0], 1, self.value_combiner.in_features, device=x.device, dtype=x.dtype)

        combined = self.value_combiner(combined)
        combined = combined.repeat(1, x.shape[1] // combined.shape[1], 1)
        combined = self.cond_norm(combined)

        k = self.to_k(combined)
        v = self.to_v(combined)
        h, _ = self.attention(q, k, v)
        return self.to_out(h) + x


class UNetBlock(nn.Module):
    """U-Net block with time and text conditioning."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        num_steps: int,
        kernel_size: int = 5,
        n_heads: int = 4,
        hidden_dim: int = 16,
        text_embed_dim: int = 768,
    ):
        super().__init__()
        n_shortcut = (n_inputs + n_outputs) // 2

        self.pre_shortcut = nn.Conv1d(n_inputs, n_shortcut, kernel_size, padding="same")
        self.shortcut_conv = nn.Conv1d(n_shortcut, n_shortcut, 1, padding="same")
        self.post_shortcut = nn.Conv1d(n_shortcut, n_outputs, kernel_size, padding="same")

        self.norm1 = nn.GroupNorm(1, n_shortcut)
        self.norm2 = nn.GroupNorm(1, n_outputs)
        self.res_conv = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else nn.Identity()

        self.self_attn = UNetSelfAttention(n_shortcut, hidden_dim, n_heads)
        self.cross_attn = UNetCrossAttention(n_shortcut, hidden_dim, n_heads, text_embed_dim)
        self.attn_norm = nn.LayerNorm(n_shortcut)

        self.time_emb = DiffusionTimeEmbedding(num_steps, n_inputs)

    def forward(self, x: Tensor, t: Tensor, text_embed: Optional[Tensor]) -> Tensor:
        """Args: x (B, C, L), t (B,), text_embed (B, 1, D). Returns: (B, C_out, L)."""
        initial_x = x
        t_emb = self.time_emb(t).unsqueeze(-1)
        x = x + t_emb

        shortcut = self.pre_shortcut(x)
        shortcut = self.norm1(shortcut)
        shortcut = F.mish(shortcut)
        shortcut = self.shortcut_conv(shortcut)

        shortcut = shortcut.transpose(-1, -2)
        shortcut = self.self_attn(shortcut)
        shortcut = self.cross_attn(shortcut, text_embed)
        shortcut = self.attn_norm(shortcut)
        shortcut = shortcut.transpose(-1, -2)

        out = self.post_shortcut(shortcut)
        out = self.norm2(out)
        out = F.mish(out)
        return out + self.res_conv(initial_x)


class DownsamplingBlock(nn.Module):
    """Downsampling block in U-Net."""

    def __init__(self, n_inputs: int, n_outputs: int, num_steps: int, **kwargs):
        super().__init__()
        self.down = nn.Conv1d(n_outputs, n_outputs, 3, stride=2, padding=1)
        self.block = UNetBlock(n_inputs, n_outputs, num_steps, **kwargs)

    def forward(self, x: Tensor, t: Tensor, text_embed: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        h = self.block(x, t, text_embed)
        return h, self.down(h)


class UpsamplingBlock(nn.Module):
    """Upsampling block in U-Net."""

    def __init__(self, n_inputs: int, n_outputs: int, num_steps: int, up_dim: Optional[int] = None, **kwargs):
        super().__init__()
        self.block = UNetBlock(n_inputs, n_outputs, num_steps, **kwargs)
        up_ch = up_dim if up_dim is not None else n_inputs // 2
        self.up = nn.ConvTranspose1d(up_ch, up_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x: Tensor, h: Optional[Tensor], t: Tensor, text_embed: Optional[Tensor]) -> Tensor:
        x = self.up(x)
        if h is not None:
            if x.shape[2] != h.shape[2]:
                target = h.shape[2]
                if x.shape[2] < target:
                    pad = target - x.shape[2]
                    x = F.pad(x, (pad // 2, pad - pad // 2), mode='replicate')
                else:
                    diff = x.shape[2] - target
                    x = x[:, :, diff // 2: diff // 2 + target]
            x = torch.cat([x, h], dim=1)
        return self.block(x, t, text_embed)


class BottleneckNet(nn.Module):
    """Bottleneck layer in U-Net."""

    def __init__(self, n_channels: int, num_steps: int, n_heads: int = 4, hidden_dim: int = 16, text_embed_dim: int = 768):
        super().__init__()
        self.time_emb = DiffusionTimeEmbedding(num_steps, n_channels)
        self.conv1 = nn.Conv1d(n_channels, n_channels, 3, padding="same")
        self.conv1_2 = nn.Conv1d(n_channels, n_channels, 3, padding="same")
        self.conv2 = nn.Conv1d(n_channels, n_channels, 3, padding="same")
        self.norm1 = nn.GroupNorm(1, n_channels)
        self.norm2 = nn.GroupNorm(1, n_channels)
        self.attn_norm = nn.LayerNorm(n_channels)
        self.self_attn = UNetSelfAttention(n_channels, hidden_dim, n_heads)
        self.cross_attn = UNetCrossAttention(n_channels, hidden_dim, n_heads, text_embed_dim)

    def forward(self, x: Tensor, t: Tensor, text_embed: Optional[Tensor]) -> Tensor:
        out = x + self.time_emb(t).unsqueeze(-1)
        out = self.conv1(out)
        out = self.norm1(out)
        out = F.mish(out)
        out = self.conv1_2(out)

        out = out.transpose(-1, -2)
        out = self.self_attn(out)
        out = self.cross_attn(out, text_embed)
        out = self.attn_norm(out)
        out = out.transpose(-1, -2)

        out = self.conv2(out)
        out = self.norm2(out)
        out = F.mish(out)
        return x + out


class DiffuSETSUNet(nn.Module):
    """U-Net for latent diffusion on (B, 4, L/8) latent space."""

    def __init__(
        self,
        num_steps: int = 50,
        kernel_size: int = 3,
        num_levels: int = 5,
        n_channels: int = 4,
        text_embed_dim: int = 768,
        n_heads: int = 8,
    ):
        super().__init__()
        self.num_levels = num_levels
        assert kernel_size % 2 == 1, "Kernel size must be odd"

        # Build channel lists
        input_ch = []
        output_ch = []

        for i in range(num_levels - 1):
            input_ch.append(n_channels * 2**i)
        for i in range(num_levels - 1):
            input_ch.append(2 * input_ch[num_levels - i - 2])

        for i in range(num_levels - 1):
            output_ch.append(2 * input_ch[i])
        for i in range(num_levels - 1):
            output_ch.append(output_ch[num_levels - i - 2] // 2)

        for i in range(num_levels - 2):
            k = 2 * (num_levels - 1) - i - 1
            input_ch[k] += output_ch[i]

        hidden_state = [ch * 2 for ch in input_ch]

        # Build blocks
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        for i in range(num_levels - 1):
            self.down_blocks.append(
                DownsamplingBlock(
                    input_ch[i], output_ch[i], num_steps,
                    kernel_size=kernel_size, n_heads=n_heads,
                    hidden_dim=hidden_state[i], text_embed_dim=text_embed_dim
                )
            )

        self.bottleneck = BottleneckNet(
            input_ch[num_levels], num_steps,
            n_heads=4, hidden_dim=32, text_embed_dim=text_embed_dim
        )

        i = num_levels - 1
        self.up_blocks.append(
            UpsamplingBlock(
                input_ch[i], output_ch[i], num_steps,
                kernel_size=kernel_size, up_dim=input_ch[i],
                n_heads=n_heads, hidden_dim=hidden_state[i], text_embed_dim=text_embed_dim
            )
        )
        for i in range(num_levels, 2 * num_levels - 2):
            self.up_blocks.append(
                UpsamplingBlock(
                    input_ch[i], output_ch[i], num_steps,
                    kernel_size=kernel_size, n_heads=n_heads,
                    hidden_dim=hidden_state[i], text_embed_dim=text_embed_dim
                )
            )

        self.output_conv = nn.Sequential(
            nn.Conv1d(output_ch[-1], n_channels, 3, padding="same"),
            nn.Mish(),
            nn.Conv1d(n_channels, n_channels, 1, padding="same"),
        )

    def forward(self, x: Tensor, t: Tensor, text_embed: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: (B, 4, L/8) - noisy latent
            t: (B,) - timestep
            text_embed: (B, 1, D) or (B, D) - text embeddings
        Returns:
            Predicted noise (B, 4, L/8)
        """
        if x.dim() == 4:
            x = x.squeeze(1)
        if text_embed is not None and text_embed.dim() == 2:
            text_embed = text_embed.unsqueeze(1)

        shortcuts = []
        out = x

        for block in self.down_blocks:
            h, out = block(out, t, text_embed)
            shortcuts.append(h)
        shortcuts.pop()

        out = self.bottleneck(out, t, text_embed)

        out = self.up_blocks[0](out, None, t, text_embed)
        for idx, block in enumerate(self.up_blocks[1:]):
            out = block(out, shortcuts[-1 - idx], t, text_embed)

        return self.output_conv(out)


# =============================================================================
# DiffuSETS Lightning Module
# =============================================================================

def vae_loss_fn(recons: Tensor, x: Tensor, mu: Tensor, log_var: Tensor, kld_weight: float = 1.0) -> Dict[str, Tensor]:
    """Compute VAE loss with KL divergence."""
    mse = F.mse_loss(recons, x, reduction='sum').div(x.size(0))
    q_z = Normal(mu, (log_var * 0.5).exp())
    p_z = Normal(torch.zeros_like(mu), torch.ones_like(log_var))
    kld = kl_divergence(q_z, p_z).sum(1).mean()
    return {"loss": mse + kld_weight * kld, "mse": mse.detach(), "kld": kld.detach()}


@Registry.register_model("diffusets", aliases=["dfs"])
class DiffuSETS(BaseGeneratorModule, DiffusionMixin):
    """
    DiffuSETS: Latent Diffusion for Text-Conditioned Time Series Generation.

    Two-stage training:
    - Stage 'vae_pretrain': Train VAE to compress (B, L, C) -> (B, 4, L/8)
    - Stage 'finetune': Train U-Net diffusion in latent space with text conditioning

    Usage:
        # Stage 1: VAE pretraining
        model = DiffuSETS(config, use_condition=False, current_stage='vae_pretrain')
        trainer.fit(model, datamodule)

        # Stage 2: Latent diffusion
        model = DiffuSETS.load_from_checkpoint(ckpt_path)
        model.set_stage('finetune')
        trainer.fit(model, datamodule)
    """

    SUPPORTED_STAGES = ["vae_pretrain", "finetune"]

    def __init__(
        self,
        config,
        use_condition: bool = True,
        learning_rate: Optional[float] = None,
        current_stage: str = "finetune",
        **kwargs,
    ):
        self._current_stage = current_stage
        super().__init__(config, use_condition, learning_rate, **kwargs)

    def _build_model(self) -> None:
        """Build VAE and U-Net components."""
        cfg = self.config.model
        data_cfg = self.config.data
        cond_cfg = self.config.condition

        self.n_var = data_cfg.n_var
        self.seq_len = data_cfg.seq_length
        self.latent_channels = getattr(cfg, "latent_channels", 4)

        # VAE
        self.encoder = VAEEncoder(self.n_var)
        self.decoder = VAEDecoder(self.n_var)

        # KL annealing
        self.kld_weight_max = getattr(cfg, "kld_weight", 1.0)
        self.kld_annealing = getattr(cfg, "kld_annealing", True)
        self.kld_annealing_type = getattr(cfg, "kld_annealing_type", "linear")
        self.kld_warmup_epochs = getattr(cfg, "kld_warmup_epochs", 300)
        self.kld_start_weight = getattr(cfg, "kld_start_weight", 0.01)

        # Diffusion U-Net
        num_steps = getattr(cfg, "diffusion_steps", 50)
        text_input_dim = cond_cfg.text.input_dim
        text_output_dim = getattr(cond_cfg.text, "output_dim", None) or text_input_dim
        text_embed_dim = text_output_dim
        if text_output_dim != text_input_dim:
            self.text_proj = nn.Linear(text_input_dim, text_output_dim)
        else:
            self.text_proj = nn.Identity()

        self.unet = DiffuSETSUNet(
            num_steps=num_steps,
            kernel_size=getattr(cfg, "kernel_size", 3),
            num_levels=getattr(cfg, "num_levels", 5),
            n_channels=self.latent_channels,
            text_embed_dim=text_embed_dim,
            n_heads=getattr(cfg, "nheads", 8),
        )

        self.num_steps = num_steps

        # Diffusion schedule
        schedule = getattr(cfg, "noise_schedule", None) or getattr(cfg, "schedule", "quad")
        beta_start = getattr(cfg, "beta_start", 0.0001)
        beta_end = getattr(cfg, "beta_end", 0.5)
        if schedule == "linear":
            betas = self.linear_beta_schedule(num_steps, beta_start, beta_end)
        elif schedule == "quad":
            betas = self.quad_beta_schedule(num_steps, beta_start, beta_end)
        elif schedule == "cosine":
            betas = self.cosine_beta_schedule(num_steps)
        else:
            raise ValueError(f"Unknown noise schedule: {schedule}")
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("alpha_bar_sqrt", torch.sqrt(alphas_cumprod))
        self.register_buffer("alpha_bar_sqrt_inverse", 1.0 / torch.sqrt(alphas_cumprod))
        self.register_buffer("reverse_coef1", 1.0 / torch.sqrt(alphas))
        self.register_buffer(
            "reverse_coef2",
            (1.0 - alphas) / torch.sqrt(1.0 - alphas_cumprod),
        )
        sigma_sq = (1.0 - alphas_cumprod[:-1]) / (1.0 - alphas_cumprod[1:]) * betas[1:]
        self.register_buffer("sigma", torch.sqrt(sigma_sq))
        self.register_buffer("reverse_coef2_ddim", torch.sqrt(1.0 - alphas_cumprod[:-1] - sigma_sq))
        self.register_buffer("reverse_coef2_ddim_determin", torch.sqrt(1.0 - alphas_cumprod[:-1]))

    def set_stage(self, stage: str) -> None:
        """Set current training stage."""
        if stage not in self.SUPPORTED_STAGES:
            raise ValueError(f"Unknown stage: {stage}. Supported: {self.SUPPORTED_STAGES}")
        self._current_stage = stage

        # Freeze/unfreeze modules based on stage
        if stage == "vae_pretrain":
            for p in self.unet.parameters():
                p.requires_grad = False
            for p in self.encoder.parameters():
                p.requires_grad = True
            for p in self.decoder.parameters():
                p.requires_grad = True
        elif stage == "finetune":
            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.decoder.parameters():
                p.requires_grad = False
            for p in self.unet.parameters():
                p.requires_grad = True

    def get_kld_weight(self) -> float:
        """Get current KL divergence weight (with optional annealing)."""
        if not self.kld_annealing:
            return self.kld_weight_max
        epoch = self.current_epoch
        if epoch >= self.kld_warmup_epochs:
            return self.kld_weight_max
        progress = epoch / self.kld_warmup_epochs
        if self.kld_annealing_type == "linear":
            weight = self.kld_start_weight + (self.kld_weight_max - self.kld_start_weight) * progress
        elif self.kld_annealing_type == "sigmoid":
            k = 10
            x = (progress - 0.5) * k
            sigmoid = 1.0 / (1.0 + math.exp(-x))
            weight = self.kld_start_weight + (self.kld_weight_max - self.kld_start_weight) * sigmoid
        elif self.kld_annealing_type == "cyclical":
            cycle_length = max(self.kld_warmup_epochs // 4, 1)
            cycle_progress = (epoch % cycle_length) / cycle_length
            weight = self.kld_start_weight + (self.kld_weight_max - self.kld_start_weight) * cycle_progress
        else:
            raise ValueError(f"Unknown annealing type: {self.kld_annealing_type}")
        return weight

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Encode time series to latent space."""
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent to time series."""
        return self.decoder(z)

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Training forward pass."""
        ts = batch["ts"]  # (B, L, C)
        cap_emb = batch.get("cap_emb")

        if self._current_stage == "vae_pretrain":
            return self._vae_forward(ts)
        else:
            return self._diffusion_forward(ts, cap_emb)

    def _vae_forward(self, ts: Tensor) -> Dict[str, Tensor]:
        """VAE training forward pass."""
        # Pad to multiple of 8
        orig_len = ts.shape[1]
        pad = (-orig_len) % 8
        if pad:
            ts = F.pad(ts, (0, 0, 0, pad), mode="replicate")

        z, mean, log_var = self.encoder(ts)
        recons = self.decoder(z)

        # Trim to original length
        if pad:
            recons = recons[:, :orig_len, :]
            ts = ts[:, :orig_len, :]

        loss_dict = vae_loss_fn(recons, ts, mean, log_var, self.get_kld_weight())
        return loss_dict

    def _diffusion_forward(self, ts: Tensor, cap_emb: Optional[Tensor]) -> Dict[str, Tensor]:
        """Latent diffusion training forward pass."""
        B = ts.shape[0]
        device = ts.device
        if self.use_condition and cap_emb is None:
            raise ValueError("DiffuSETS finetune requires 'cap_emb' in batch when use_condition=True.")

        with torch.no_grad():
            z, _, _ = self.encoder(ts)

        # Sample timestep and noise
        t = torch.randint(0, self.num_steps, (B,), device=device)
        noise = torch.randn_like(z)

        # Forward diffusion
        noisy_z, _ = self.q_sample(
            z, t, noise,
            self.sqrt_alphas_cumprod,
            self.sqrt_one_minus_alphas_cumprod,
        )

        # Process condition
        text_embed = None
        if self.use_condition and cap_emb is not None:
            text_embed = self.text_proj(cap_emb)

        # Predict noise
        noise_pred = self.unet(noisy_z, t, text_embed)

        loss = F.mse_loss(noise_pred, noise)
        return {"loss": loss, "diffusion_loss": loss}

    @torch.no_grad()
    def generate(
        self,
        condition: Tensor,
        n_samples: int = 1,
        num_steps: Optional[int] = None,
        **kwargs,
    ) -> Tensor:
        """Generate time series from text condition."""
        B = condition.shape[0]
        device = condition.device

        if num_steps is None:
            num_steps = self.num_steps
        if num_steps != self.num_steps:
            raise ValueError("num_steps must match diffusion_steps for DiffuSETS sampling.")

        # Compute latent spatial size
        latent_len = (self.seq_len + 7) // 8

        # Expand condition
        if n_samples > 1:
            if condition.dim() == 2:
                condition = condition.unsqueeze(1).expand(-1, n_samples, -1)
                condition = condition.reshape(B * n_samples, -1)
            else:
                condition = condition.unsqueeze(1).expand(-1, n_samples, -1, -1)
                condition = condition.reshape(B * n_samples, condition.shape[-2], condition.shape[-1])
        total = B * n_samples
        text_embed = self.text_proj(condition)
        sampler = kwargs.get("sampler", "ddim")
        is_determin = kwargs.get("is_determin", True)

        # Start from noise in latent space
        z = torch.randn(total, self.latent_channels, latent_len, device=device)

        # DDPM/DDIM reverse process (legacy parity)
        for t_idx in reversed(range(num_steps)):
            t = torch.full((total,), t_idx, device=device, dtype=torch.long)

            noise_pred = self.unet(z, t, text_embed)
            if sampler == "ddpm":
                coef1 = self.reverse_coef1[t_idx]
                coef2 = self.reverse_coef2[t_idx]
                z = coef1 * (z - coef2 * noise_pred)
                if t_idx > 0:
                    z = z + self.sigma[t_idx - 1] * torch.randn_like(z)
            elif sampler == "ddim":
                pred_z0 = (
                    z - self.sqrt_one_minus_alphas_cumprod[t_idx] * noise_pred
                ) * self.alpha_bar_sqrt_inverse[t_idx]
                if t_idx == 0:
                    z = pred_z0
                else:
                    coef1 = self.alpha_bar_sqrt[t_idx - 1]
                    if is_determin:
                        coef2 = self.reverse_coef2_ddim_determin[t_idx - 1]
                        coef3 = 0.0
                    else:
                        coef2 = self.reverse_coef2_ddim[t_idx - 1]
                        coef3 = self.sigma[t_idx - 1]
                    z = coef1 * pred_z0 + coef2 * noise_pred + coef3 * torch.randn_like(z)
            else:
                raise ValueError(f"Unknown sampler: {sampler}")

        # Decode
        ts = self.decoder(z)

        # Trim to seq_len
        if ts.shape[1] < self.seq_len:
            ts = F.pad(ts, (0, 0, 0, self.seq_len - ts.shape[1]), mode="replicate")
        elif ts.shape[1] > self.seq_len:
            ts = ts[:, :self.seq_len, :]

        return ts.view(B, n_samples, self.seq_len, self.n_var)
