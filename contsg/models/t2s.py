"""
T2S (Text-to-Series): Text-conditioned time series generation with flow matching and Diffusion Transformer.

This model uses:
- Flow matching (Rectified Flow) for the diffusion process
- Diffusion Transformer architecture for noise prediction
- 2D patching of time series data
- AdaLN (Adaptive Layer Normalization) for conditioning
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Attention, Mlp
from torch import Tensor

from contsg.models.base import BaseGeneratorModule
from contsg.registry import Registry


# =============================================================================
# Utility Functions
# =============================================================================


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Apply adaptive layer normalization modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def get_sinusoidal_positional_embeddings(num_positions: int, d_model: int) -> Tensor:
    """
    Generate sinusoidal positional embeddings.

    Args:
        num_positions: Number of positions
        d_model: Embedding dimension

    Returns:
        Positional embeddings of shape (1, num_positions, d_model)
    """
    position = torch.arange(num_positions).unsqueeze(1)  # (num_positions, 1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
    ).unsqueeze(0)  # (1, d_model/2)

    pos_embedding = torch.zeros(num_positions, d_model)
    pos_embedding[:, 0::2] = torch.sin(position * div_term)
    pos_embedding[:, 1::2] = torch.cos(position * div_term)

    return pos_embedding.unsqueeze(0)  # (1, num_positions, d_model)


# =============================================================================
# AutoEncoder Components
# =============================================================================


class Residual(nn.Module):
    """Residual block for T2S AutoEncoder."""

    def __init__(self, in_channels: int, num_hiddens: int, num_residual_hiddens: int):
        super().__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=num_residual_hiddens,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ReLU(True),
            nn.Conv1d(
                in_channels=num_residual_hiddens,
                out_channels=num_hiddens,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self._block(x)


class ResidualStack(nn.Module):
    """Stack of residual blocks for T2S AutoEncoder."""

    def __init__(
        self,
        in_channels: int,
        num_hiddens: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
    ):
        super().__init__()
        self._layers = nn.ModuleList(
            [
                Residual(in_channels, num_hiddens, num_residual_hiddens)
                for _ in range(num_residual_layers)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self._layers:
            x = layer(x)
        return F.relu(x)


class T2SEncoder(nn.Module):
    """T2S AutoEncoder Encoder.

    Compresses time series via Conv1d downsampling and forces output to fixed
    temporal dimension of 30 via F.interpolate.
    """

    def __init__(
        self,
        in_channels: int,
        num_hiddens: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
        embedding_dim: int,
    ):
        super().__init__()
        self._conv_1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=num_hiddens // 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self._conv_2 = nn.Conv1d(
            in_channels=num_hiddens // 2,
            out_channels=num_hiddens,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self._conv_3 = nn.Conv1d(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )
        self._pre_vq_conv = nn.Conv1d(
            in_channels=num_hiddens,
            out_channels=embedding_dim,
            kernel_size=1,
            stride=1,
        )

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            inputs: (B, C, T) time series in channel-first format

        Returns:
            z: (B, embedding_dim, 30) - latent representation with fixed temporal dim
            before: (B, embedding_dim, T/4) - representation before interpolation (for cross loss)
        """
        x = inputs
        x = self._conv_1(x)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)

        x = self._conv_3(x)
        x = self._residual_stack(x)
        x = self._pre_vq_conv(x)

        before = x  # Save for cross loss
        x = F.interpolate(x, size=30, mode="linear", align_corners=True)

        return x, before


class T2SDecoder(nn.Module):
    """T2S AutoEncoder Decoder.

    Reconstructs time series from latent space via interpolation and Conv1d upsampling.
    """

    def __init__(
        self,
        in_channels: int,
        num_hiddens: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
        out_channels: int,
    ):
        super().__init__()
        self._conv_1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )
        self._conv_trans_1 = nn.ConvTranspose1d(
            in_channels=num_hiddens,
            out_channels=num_hiddens // 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self._conv_trans_2 = nn.ConvTranspose1d(
            in_channels=num_hiddens // 2,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, inputs: Tensor, length: int) -> tuple[Tensor, Tensor]:
        """
        Args:
            inputs: (B, embedding_dim, 30) latent representation
            length: Original time series length for reconstruction

        Returns:
            x: (B, C, length) reconstructed time series
            after: (B, embedding_dim, length//4) - representation after interpolation (for cross loss)
        """
        # Interpolate to length//4 before conv layers
        x = F.interpolate(inputs, size=int(length // 4), mode="linear", align_corners=True)
        after = x  # Save for cross loss

        x = self._conv_1(x)
        x = self._residual_stack(x)
        x = self._conv_trans_1(x)
        x = F.relu(x)
        x = self._conv_trans_2(x)

        return x, after


class T2SAutoEncoder(nn.Module):
    """T2S AutoEncoder for latent space compression.

    Uses Conv1d-based encoder/decoder with F.interpolate to fixed latent dim (30).
    No Vector Quantizer layer is used.

    Loss function:
        total_loss = recon_loss + cross_loss
        where:
            recon_loss = MSE(reconstructed, original)
            cross_loss = MSE(before_interpolate, after_interpolate)
    """

    def __init__(
        self,
        num_hiddens: int = 128,
        num_residual_layers: int = 2,
        num_residual_hiddens: int = 32,
        embedding_dim: int = 64,
        num_input_channels: int = 1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_input_channels = num_input_channels

        self.encoder = T2SEncoder(
            in_channels=num_input_channels,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            embedding_dim=embedding_dim,
        )
        self.decoder = T2SDecoder(
            in_channels=embedding_dim,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            out_channels=num_input_channels,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Full forward pass with loss computation.

        Args:
            x: (B, C, T) input time series

        Returns:
            recon: (B, C, T) reconstructed time series
            z: (B, embedding_dim, 30) latent representation
            before: (B, embedding_dim, T/4) encoder output before interpolation
            after: (B, embedding_dim, T/4) decoder input after interpolation
        """
        z, before = self.encoder(x)
        recon, after = self.decoder(z, length=x.shape[-1])
        return recon, z, before, after

    def encode(self, x: Tensor) -> Tensor:
        """Encode time series to latent space."""
        z, _ = self.encoder(x)
        return z

    def decode(self, z: Tensor, length: int) -> Tensor:
        """Decode latent representation to time series."""
        recon, _ = self.decoder(z, length=length)
        return recon


# =============================================================================
# Time Embedding
# =============================================================================


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        assert dim % 2 == 0, "Dimension must be even"

    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: Timestep tensor (B,) with values in [0, 1]

        Returns:
            Time embeddings (B, dim)
        """
        t = t * 100.0
        t = t.unsqueeze(-1)

        freqs = torch.pow(10000, torch.linspace(0, 1, self.dim // 2, device=t.device))

        sin_emb = torch.sin(t[:, None] / freqs)
        cos_emb = torch.cos(t[:, None] / freqs)
        embedding = torch.cat([sin_emb, cos_emb], dim=-1)
        embedding = embedding.squeeze(1)
        return embedding


# =============================================================================
# Transformer Components
# =============================================================================


class TransformerLayer(nn.Module):
    """
    Diffusion Transformer layer with AdaLN (Adaptive Layer Normalization).

    Uses timestep and text conditioning to modulate the layer normalization
    parameters (shift, scale, gate) for both attention and MLP blocks.
    """

    def __init__(self, d_model: int = 64, num_heads: int = 4, mlp_ratio: float = 2.0):
        super().__init__()

        mlp_hidden_dim = int(d_model * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")

        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(d_model, num_heads=num_heads, qkv_bias=True)
        self.mlp = Mlp(
            in_features=d_model,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

        # AdaLN modulation: produces 6 parameters (shift, scale, gate for attn and mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model, bias=True),
        )

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor (B, num_patches, d_model)
            c: Conditioning tensor (B, d_model) - combined time and text embedding

        Returns:
            Output tensor (B, num_patches, d_model)
        """
        # Get modulation parameters
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )

        # Attention block with AdaLN
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )

        # MLP block with AdaLN
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )

        return x


class DiffusionTransformer(nn.Module):
    """
    Diffusion Transformer for time series generation.

    Architecture:
    1. 2D Patching: Convert (B, C, L) -> patches
    2. Patch Embedding: Linear projection of patches
    3. Positional Encoding: Sinusoidal positional embeddings
    4. Transformer Layers: Multiple layers with AdaLN conditioning
    5. Unpacking: Convert patches back to (B, C, L)
    """

    def __init__(
        self,
        n_var: int = 1,
        seq_len: int = 64,
        patch_size: int = 2,
        emb_size: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()

        # Data dimensions (treat as 2D: H=seq_len, W=n_var)
        self.channel = 1
        self.n_var = n_var
        self.seq_len = seq_len

        # Match legacy orientation: H=seq_len, W=n_var
        self.H = seq_len  # Temporal length
        self.W = n_var  # Latent channels

        # Patching parameters
        self.patch_size = patch_size
        self.patch_count = int((self.H / self.patch_size) * (self.W / self.patch_size))

        # Patch embedding via Conv2d
        self.conv = nn.Conv2d(
            in_channels=self.channel,
            out_channels=self.channel * self.patch_size**2,
            kernel_size=self.patch_size,
            padding=0,
            stride=self.patch_size,
        )
        self.patch_emb = nn.Linear(
            in_features=self.channel * self.patch_size**2, out_features=emb_size
        )

        # Positional embeddings
        pos_embed = get_sinusoidal_positional_embeddings(self.patch_count, emb_size)
        self.register_buffer("pos_embed", pos_embed)

        # Output layers
        self.ln = nn.LayerNorm(emb_size)
        self.linear_emb_to_patch = nn.Linear(emb_size, self.channel * self.patch_size**2)

        # Time and conditioning
        self.time_emb = TimeEmbedding(dim=emb_size)

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                TransformerLayer(d_model=emb_size, num_heads=num_heads, mlp_ratio=mlp_ratio)
                for _ in range(num_layers)
            ]
        )

        self.initialize_weights()

    def forward(self, x: Tensor, t: Tensor, text_emb: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through the diffusion transformer.

        Args:
            x: Input time series (B, C, L) where L=seq_len, C=n_var
            t: Normalized timestep (B,) in range [0, 1]
            text_emb: Text conditioning embedding (B, D) or (B, 1, D)

        Returns:
            Predicted velocity/noise (B, C, L)
        """
        # Reshape to 2D: (B, C, L) -> (B, H, W)
        # where H=seq_len, W=n_var (legacy orientation)
        x = x.permute(0, 2, 1)  # (B, L, C)

        # Add channel dimension and apply patching
        x = x.unsqueeze(1)  # (B, 1, L, C)
        x = self.conv(x)  # (B, new_channel, patch_h, patch_w)
        x = x.permute(0, 2, 3, 1)  # (B, patch_h, patch_w, new_channel)
        x = x.reshape(x.size(0), self.patch_count, -1)  # (B, num_patches, new_channel)

        # Patch embedding and positional encoding
        x = self.patch_emb(x)  # (B, num_patches, emb_size)
        x = x + self.pos_embed  # Add positional embeddings

        # Time embedding
        t_emb = self.time_emb(t)  # (B, emb_size)

        # Combine time and text conditioning
        c = t_emb
        if text_emb is not None:
            if len(text_emb.shape) == 3:
                text_emb = text_emb.squeeze(1)  # (B, D)
            c = t_emb + text_emb  # (B, emb_size)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, c)

        # Output projection
        x = self.ln(x)
        x = self.linear_emb_to_patch(x)  # (B, num_patches, patch_size^2)

        # Unpack patches back to 2D
        x = x.view(
            x.size(0),
            int(self.H / self.patch_size),
            int(self.W / self.patch_size),
            self.channel,
            self.patch_size,
            self.patch_size,
        )
        x = x.permute(0, 3, 1, 2, 4, 5)  # (B, channel, patch_h, patch_w, ps, ps)
        x = x.permute(0, 1, 2, 4, 3, 5)  # (B, channel, patch_h, ps, patch_w, ps)
        x = x.reshape(x.size(0), self.channel, self.H, self.W)  # (B, channel, H, W)

        # Remove channel dimension and permute back
        x = x.squeeze(1)  # (B, H, W)
        x = x.permute(0, 2, 1)  # (B, W, H) -> (B, L, C)

        return x

    def initialize_weights(self):
        """Initialize network weights."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Zero-out adaLN modulation layers in transformer blocks
        for block in self.layers:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)


# =============================================================================
# Rectified Flow Sampler
# =============================================================================


class RectifiedFlow:
    """
    Rectified Flow for time series generation.

    Uses linear interpolation between noise and data:
        x_t = t * x_1 + (1 - t) * x_0
    where x_0 is noise and x_1 is data.
    """

    def __init__(self, num_steps: int):
        assert num_steps is not None, "num_steps must be provided"
        assert num_steps > 1, "num_steps must be greater than 1"
        self.num_steps = num_steps

    def euler(self, x_t: Tensor, v: Tensor, dt: float) -> Tensor:
        """Euler integration step."""
        return x_t + v * dt

    def forward(self, x_1: Tensor, t: Tensor, x_0: Tensor) -> Tensor:
        """
        Forward process: interpolate between noise and data.

        Args:
            x_1: Data (B, C, L)
            t: Timestep indices (B,) in range [0, num_steps-1]
            x_0: Noise (B, C, L)

        Returns:
            Interpolated x_t (B, C, L)
        """
        # Normalize t to [0, 1]
        t_norm = t.float() / float(self.num_steps - 1)

        # Broadcast t to match x_1's shape
        shape = [t_norm.shape[0]] + [1] * (x_1.dim() - 1)
        t_norm = t_norm.view(*shape)

        # Linear interpolation: x_t = t * x_1 + (1 - t) * x_0
        x_t = t_norm * x_1 + (1.0 - t_norm) * x_0
        return x_t

    def loss(self, v_pred: Tensor, v_target: Tensor) -> Tensor:
        """
        Compute flow matching loss.

        Args:
            v_pred: Predicted velocity (B, C, L)
            v_target: Target velocity = x_1 - x_0 (B, C, L)

        Returns:
            MSE loss
        """
        return F.mse_loss(v_pred, v_target)


# =============================================================================
# T2S Model
# =============================================================================


@Registry.register_model("t2s", aliases=["text2series"])
class T2SModule(BaseGeneratorModule):
    """T2S: Two-stage Latent Flow Matching for text-to-series generation.

    This model implements a two-stage training pipeline:

    Stage 1 (ae_pretrain): AutoEncoder Pretraining
        - Train AutoEncoder to compress time series to fixed latent dimension (30)
        - Loss = MSE(recon, x) + MSE(before_interp, after_interp)

    Stage 2 (finetune): Latent Flow Matching
        - Freeze AutoEncoder, train Diffusion Transformer in latent space
        - Uses Rectified Flow for velocity prediction
        - Loss = MSE(v_pred, z - noise)

    Architecture:
        - AutoEncoder: Conv1d-based with F.interpolate(size=30) compression
        - Diffusion Transformer: 2D patching with AdaLN conditioning

    Usage:
        # Stage 1: AE pretraining
        model = T2SModule(config, use_condition=False, current_stage='ae_pretrain')
        trainer.fit(model, datamodule)

        # Stage 2: Latent flow matching
        model = T2SModule.load_from_checkpoint(ckpt_path)
        model.set_stage('finetune')
        trainer.fit(model, datamodule)
    """

    SUPPORTED_STAGES = ["ae_pretrain", "finetune"]
    LATENT_SEQ_LEN = 30  # Fixed latent temporal dimension

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
        """Build AutoEncoder and Diffusion Transformer components."""
        cfg = self.config.model
        data_cfg = self.config.data

        # Save original data dimensions
        self.n_var = data_cfg.n_var
        self.seq_len = data_cfg.seq_length
        self.num_steps = cfg.flow_steps

        # =================================================================
        # AutoEncoder Configuration
        # =================================================================
        ae_cfg = cfg.ae

        self.ae = T2SAutoEncoder(
            num_hiddens=ae_cfg.num_hiddens,
            num_residual_layers=ae_cfg.num_residual_layers,
            num_residual_hiddens=ae_cfg.num_residual_hiddens,
            embedding_dim=ae_cfg.embedding_dim,
            num_input_channels=ae_cfg.num_input_channels or self.n_var,
        )

        # =================================================================
        # Diffusion Transformer Configuration
        # =================================================================
        # Transformer operates in latent space: (B, 30, embedding_dim)
        self.transformer = DiffusionTransformer(
            n_var=ae_cfg.embedding_dim,  # Latent channels as "variates"
            seq_len=self.LATENT_SEQ_LEN,  # Fixed latent temporal dim
            patch_size=cfg.patch_size,
            emb_size=cfg.channels,
            num_layers=cfg.layers,
            num_heads=cfg.nheads,
            mlp_ratio=cfg.mlp_ratio,
        )

        # Build rectified flow sampler
        self.flow = RectifiedFlow(num_steps=self.num_steps)

        # Text embedding projection (if needed)
        text_dim = cfg.text_dim or self.config.condition.text.input_dim
        if text_dim != cfg.channels:
            self.text_proj = nn.Linear(text_dim, cfg.channels)
        else:
            self.text_proj = nn.Identity()

        self.cfg_scale = cfg.cfg_scale
        self.num_infer_steps = cfg.num_infer_steps

        # Apply initial stage configuration
        self.set_stage(self._current_stage)

    def set_stage(self, stage: str) -> None:
        """Set current training stage and freeze/unfreeze parameters accordingly.

        Args:
            stage: Either 'ae_pretrain' or 'finetune'
        """
        if stage not in self.SUPPORTED_STAGES:
            raise ValueError(f"Unknown stage: {stage}. Supported: {self.SUPPORTED_STAGES}")

        self._current_stage = stage

        if stage == "ae_pretrain":
            # Freeze Transformer, train AE
            for p in self.transformer.parameters():
                p.requires_grad = False
            for p in self.ae.parameters():
                p.requires_grad = True
            if hasattr(self, "text_proj") and not isinstance(self.text_proj, nn.Identity):
                for p in self.text_proj.parameters():
                    p.requires_grad = False
            self.ae.train()
            self.transformer.eval()

        elif stage == "finetune":
            # Freeze AE, train Transformer
            for p in self.ae.parameters():
                p.requires_grad = False
            self.ae.eval()
            for p in self.transformer.parameters():
                p.requires_grad = True
            if hasattr(self, "text_proj") and not isinstance(self.text_proj, nn.Identity):
                for p in self.text_proj.parameters():
                    p.requires_grad = True
            self.transformer.train()

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Training forward pass with stage-aware dispatch.

        Args:
            batch: Dictionary containing:
                - "ts": Time series (B, L, C)
                - "cap_emb": Caption embedding (B, D) if use_condition=True (finetune only)

        Returns:
            Dictionary with loss and stage-specific metrics
        """
        if self._current_stage == "ae_pretrain":
            return self._ae_forward(batch)
        else:  # finetune
            return self._flow_forward(batch)

    def _ae_forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """AutoEncoder pretraining forward pass.

        Loss = MSE(recon, x) + MSE(before_interp, after_interp)

        Args:
            batch: Dictionary with "ts" key

        Returns:
            Dictionary with loss, recon_loss, cross_loss
        """
        ts = batch["ts"]  # (B, L, C)

        # Convert to channel-first format for AE: (B, L, C) -> (B, C, L)
        x = ts.permute(0, 2, 1)

        # Forward through AE
        recon, z, before, after = self.ae(x)

        # Compute losses (mathematically equivalent to Legacy)
        recon_loss = F.mse_loss(recon, x)
        cross_loss = F.mse_loss(before, after)
        total_loss = recon_loss + cross_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_loss.detach(),
            "cross_loss": cross_loss.detach(),
        }

    def _flow_forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Latent Flow Matching forward pass.

        Performs flow matching in the AutoEncoder's latent space.

        Args:
            batch: Dictionary with "ts" and optionally "cap_emb"

        Returns:
            Dictionary with loss, flow_loss
        """
        ts = batch["ts"]  # (B, L, C)
        B = ts.shape[0]
        device = ts.device

        # Get text conditioning
        if self.use_condition and "cap_emb" not in batch:
            raise ValueError("T2S finetune requires 'cap_emb' in batch when use_condition=True.")
        if self.use_condition and "cap_emb" in batch:
            text_emb = batch["cap_emb"]  # (B, D)
            if text_emb.dim() == 3:
                text_emb = text_emb.squeeze(1)
            text_emb = self.text_proj(text_emb)  # (B, emb_size)
        else:
            text_emb = None

        # Encode to latent space (with no_grad since AE is frozen)
        x = ts.permute(0, 2, 1)  # (B, C, L)
        with torch.no_grad():
            z, _ = self.ae.encoder(x)  # (B, embedding_dim, 30)

        # Sample timestep and noise
        t = torch.randint(0, self.num_steps, (B,), device=device)
        noise = torch.randn_like(z)

        # Flow forward: x_t = t * z + (1-t) * noise
        z_t = self.flow.forward(z, t, noise)

        # Normalize timestep for network input
        t_norm = t.float() / float(self.num_steps - 1)

        # Predict velocity: v = z - noise
        v_pred = self.transformer(z_t, t_norm, text_emb)

        # Target velocity
        v_target = z - noise

        # Compute loss
        loss = self.flow.loss(v_pred, v_target)

        return {
            "loss": loss,
            "flow_loss": loss,
        }

    @torch.no_grad()
    def generate(
        self,
        condition: Tensor,
        n_samples: int = 1,
        num_steps: Optional[int] = None,
        **kwargs: Any,
    ) -> Tensor:
        """
        Generate time series samples in latent space and decode.

        Sampling process:
        1. Start from noise in latent space: (B*n, embedding_dim, 30)
        2. Euler ODE integration to obtain z_0
        3. Decode z_0 to time series via AE decoder

        Args:
            condition: Text embedding (B, D) or (B, 1, D)
            n_samples: Number of samples to generate per condition
            num_steps: Number of sampling steps (default: self.num_steps)

        Returns:
            Generated time series (B, n_samples, L, C)
        """
        B = condition.shape[0]
        device = condition.device

        if num_steps is None:
            num_steps = self.num_infer_steps or self.num_steps
        if num_steps <= 1:
            raise ValueError("num_steps must be greater than 1 for sampling.")

        # Process text conditioning
        if condition.dim() == 3:
            condition = condition.squeeze(1)
        text_emb = self.text_proj(condition)  # (B, emb_size)

        # Expand for n_samples
        text_emb = text_emb.repeat_interleave(n_samples, dim=0)  # (B*n_samples, emb_size)

        total = B * n_samples

        # Start from noise in latent space: (B*n, embedding_dim, 30)
        z = torch.randn(
            total, self.ae.embedding_dim, self.LATENT_SEQ_LEN, device=device
        )

        # Euler ODE integration in latent space
        # dt = 1/num_steps ensures total integration distance = num_steps * dt = 1
        dt = 1.0 / num_steps
        cfg_scale = kwargs.get("cfg_scale", self.cfg_scale)
        use_cfg = cfg_scale is not None and cfg_scale != 1.0

        for step in range(num_steps):
            # t ranges from 0 to (num_steps-1)/num_steps, never reaching 1
            t = torch.full(
                (total,), step * dt, device=device, dtype=torch.float32
            )

            # Predict velocity in latent space
            if use_cfg:
                v_uncond = self.transformer(z, t, None)
                v_cond = self.transformer(z, t, text_emb)
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = self.transformer(z, t, text_emb)

            # Euler step
            z = self.flow.euler(z, v, dt)

        # Decode to original time series length
        ts_recon = self.ae.decode(z, length=self.seq_len)  # (B*n, C, L)

        # Convert to (B*n, L, C)
        ts_recon = ts_recon.permute(0, 2, 1)

        # Reshape to (B, n_samples, L, C)
        ts_recon = ts_recon.view(B, n_samples, self.seq_len, self.n_var)

        return ts_recon
