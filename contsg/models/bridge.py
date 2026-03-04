"""
Bridge: Domain-agnostic diffusion-based time series generation.

This module implements the Bridge model, which uses a domain-unified prototyper
to extract domain-agnostic latent representations from example time series, then
conditions a diffusion model on these representations along with text embeddings.

Reference: Bridge model from the VerbalTS benchmark framework.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from contsg.models.base import BaseGeneratorModule, DiffusionMixin
from contsg.registry import Registry
from contsg.config.schema import ExperimentConfig


# Import Bridge modules from contsg package
from contsg.models.bridge_modules.unet import UNetModel
from contsg.models.bridge_modules.prototype import DomainUnifiedPrototyper
from contsg.models.bridge_modules.utils import Return


@Registry.register_model("bridge", aliases=["brg"])
class BridgeModule(BaseGeneratorModule, DiffusionMixin):
    """Bridge: Domain-agnostic diffusion with prototype conditioning.

    Key features:
    - Domain-unified prototyper extracts latent representations from example time series
    - UNet backbone with cross-attention and text embedding fusion
    - Supports classifier-free guidance (CFG)
    - Single-stage training: finetune (conditional)

    Architecture:
    - cond_stage_model: DomainUnifiedPrototyper - extracts latent prototypes from examples
    - model: UNetModel - diffusion backbone with cross-attention
    - Text fusion via ConditioningMLP (optional)

    The model requires "bridge_example_ts" in the batch during both training and inference.
    """

    SUPPORTED_STAGES = ["finetune"]
    needs_bridge_example = True

    def _build_model(self) -> None:
        """Build Bridge model architecture."""
        cfg = self.config.model
        data_cfg = self.config.data

        # Data dimensions
        self.n_var = data_cfg.n_var
        self.seq_length = data_cfg.seq_length

        # Diffusion parameters (Bridge uses explicit diffusion_* fields)
        self.num_steps = int(cfg.diffusion_steps)
        self.beta_start = float(cfg.beta_start)
        self.beta_end = float(cfg.beta_end)
        self.schedule = cfg.noise_schedule

        # Bridge-specific parameters
        latent_dim = int(getattr(cfg, "latent_dim", 32))
        repre_emb_channels = getattr(cfg, "repre_emb_channels", None)
        if repre_emb_channels is None:
            repre_emb_channels = latent_dim
        self.repre_emb_channels = int(repre_emb_channels)
        num_latents = getattr(cfg, "num_latents", None)
        if num_latents is None:
            num_latents = 16
        self.num_latents = int(num_latents)
        latent_unit = getattr(cfg, "latent_unit", None)
        if latent_unit is None:
            latent_unit = 1
        self.latent_unit = int(latent_unit)
        self.cond_drop_prob = float(getattr(cfg, "cond_drop_prob", 0.5))
        self.use_cfg = bool(getattr(cfg, "use_cfg", True))

        # UNet architecture parameters
        model_channels = int(getattr(cfg, "model_channels", 64))
        num_res_blocks = int(getattr(cfg, "num_res_blocks", 2))
        attention_resolutions = getattr(cfg, "attention_resolutions", [1, 2, 4])
        channel_mult = getattr(cfg, "channel_mult", [1, 2, 4, 4])
        dropout = float(getattr(cfg, "dropout", 0.0))

        # Conditioning parameters
        context_dim = getattr(cfg, "context_dim", None)
        if context_dim is None or int(context_dim) != self.repre_emb_channels:
            context_dim = self.repre_emb_channels
        text_dim = getattr(cfg, "text_dim", None)  # Text embedding dimension (e.g., 768 or 4096)
        fusion_type = getattr(cfg, "fusion_type", "gated_add")  # Text fusion type

        # Cross-attention parameters
        use_spatial_transformer = getattr(cfg, "use_spatial_transformer", True)
        use_scale_shift_norm = getattr(cfg, "use_scale_shift_norm", True)
        resblock_updown = getattr(cfg, "resblock_updown", True)
        transformer_depth = getattr(cfg, "transformer_depth", 1)
        num_heads = getattr(cfg, "num_heads", 8)
        num_head_channels = getattr(cfg, "num_head_channels", -1)

        # Build cond_stage_model: DomainUnifiedPrototyper
        prototype_dim = getattr(cfg, "prototype_dim", None)
        if prototype_dim is None:
            prototype_dim = self.repre_emb_channels
        prototype_dim = int(prototype_dim)
        num_latents = self.num_latents  # Number of learnable latent vectors

        self.cond_stage_model = DomainUnifiedPrototyper(
            dim=prototype_dim,
            window=self.seq_length,
            num_latents=num_latents,
            num_channels=self.n_var,
            latent_dim=self.repre_emb_channels,
            bn=True,
        )

        # Build UNet model
        self.model = UNetModel(
            seq_len=self.seq_length,
            in_channels=self.n_var,
            model_channels=model_channels,
            out_channels=self.n_var,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            conv_resample=True,
            dims=1,  # 1D time series
            num_classes=None,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=-1,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=False,
            use_spatial_transformer=use_spatial_transformer,
            transformer_depth=transformer_depth,
            context_dim=context_dim,
            text_dim=text_dim,
            fusion_type=fusion_type,
            legacy=True,
            repre_emb_channels=self.repre_emb_channels,
            latent_unit=self.latent_unit,
            use_cfg=self.use_cfg,
            cond_drop_prob=self.cond_drop_prob,
            use_pam=False,
        )

        # Setup diffusion schedule
        self._setup_diffusion_schedule()

    def _setup_diffusion_schedule(self) -> None:
        """Setup diffusion noise schedule."""
        if self.schedule == "linear":
            betas = torch.linspace(self.beta_start, self.beta_end, self.num_steps)
        elif self.schedule == "quad":
            betas = self.quad_beta_schedule(self.num_steps, self.beta_start, self.beta_end)
        elif self.schedule == "cosine":
            betas = self.cosine_beta_schedule(self.num_steps)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # Posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20))
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Forward pass for training.

        Args:
            batch: Dictionary containing:
                - "ts": Time series (B, L, C)
                - "cap_emb": Text embedding (B, D) - optional, only in finetune stage
                - "bridge_example_ts": Example time series for prototype extraction (B, L, C)
                - "tp": Time points (B, L) - optional

        Returns:
            Dictionary with "loss" and other metrics
        """
        # Unpack batch
        ts = batch["ts"]  # (B, L, C)
        B, L, C = ts.shape
        device = ts.device

        # Bridge requires example time series for prototype extraction
        if "bridge_example_ts" not in batch:
            raise ValueError("Bridge model requires 'bridge_example_ts' in batch")

        if self.use_condition and "cap_emb" not in batch:
            raise ValueError("Bridge requires 'cap_emb' in batch when use_condition=True.")

        bridge_example_ts = batch["bridge_example_ts"]  # (B, L, C)

        # Convert to (B, C, L) for model
        ts = ts.permute(0, 2, 1)
        bridge_example_ts = bridge_example_ts.permute(0, 2, 1)

        # Extract latent prototypes from example time series
        # c: (B, num_latents, latent_dim), mask: (B, num_latents)
        c, mask = self.cond_stage_model(bridge_example_ts)

        # Get text embedding if available (finetune stage)
        text_embedding = None
        if self.use_condition and "cap_emb" in batch:
            text_embedding = batch["cap_emb"]  # (B, text_dim)

        # Sample timestep
        t = torch.randint(0, self.num_steps, (B,), device=device).long()

        # Add noise
        noise = torch.randn_like(ts)
        noisy_ts = self._q_sample(ts, t, noise)

        # Predict noise
        model_output = self.model(
            x=noisy_ts,
            timesteps=t,
            context=c,
            mask=mask,
            y=None,
            cond_drop_prob=self.cond_drop_prob if self.training else 0.0,
            text_embedding=text_embedding,
        )

        # Extract predicted noise
        pred_noise = model_output.pred

        # Compute loss
        loss = F.mse_loss(pred_noise, noise)

        return {
            "loss": loss,
            "mse_loss": loss,
        }

    def _q_sample(self, x_start: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """Forward diffusion process: add noise to data."""
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

        # Expand dims for broadcasting
        while sqrt_alphas_cumprod_t.dim() < x_start.dim():
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def _predict_x0_from_noise(self, x_t: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """Predict x_0 from x_t and predicted noise."""
        sqrt_recip_alphas_cumprod_t = self.sqrt_recip_alphas_cumprod[t]
        sqrt_recipm1_alphas_cumprod_t = self.sqrt_recipm1_alphas_cumprod[t]

        while sqrt_recip_alphas_cumprod_t.dim() < x_t.dim():
            sqrt_recip_alphas_cumprod_t = sqrt_recip_alphas_cumprod_t.unsqueeze(-1)
            sqrt_recipm1_alphas_cumprod_t = sqrt_recipm1_alphas_cumprod_t.unsqueeze(-1)

        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise

    def _p_mean_variance(
        self,
        x: Tensor,
        t: Tensor,
        c: Tensor,
        mask: Optional[Tensor],
        text_embedding: Optional[Tensor],
        clip_denoised: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute mean and variance of p(x_{t-1} | x_t)."""
        # Predict noise
        model_output = self.model(
            x=x,
            timesteps=t,
            context=c,
            mask=mask,
            y=None,
            cond_drop_prob=0.0,  # No dropout during inference
            text_embedding=text_embedding,
        )
        pred_noise = model_output.pred

        # Predict x_0
        x_recon = self._predict_x0_from_noise(x, t, pred_noise)

        if clip_denoised:
            x_recon = torch.clamp(x_recon, -1.0, 1.0)

        # Compute posterior mean
        posterior_mean_coef1 = self.posterior_mean_coef1[t]
        posterior_mean_coef2 = self.posterior_mean_coef2[t]

        while posterior_mean_coef1.dim() < x.dim():
            posterior_mean_coef1 = posterior_mean_coef1.unsqueeze(-1)
            posterior_mean_coef2 = posterior_mean_coef2.unsqueeze(-1)

        posterior_mean = posterior_mean_coef1 * x_recon + posterior_mean_coef2 * x

        # Get variance
        posterior_variance = self.posterior_variance[t]
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]

        while posterior_variance.dim() < x.dim():
            posterior_variance = posterior_variance.unsqueeze(-1)
            posterior_log_variance_clipped = posterior_log_variance_clipped.unsqueeze(-1)

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @torch.no_grad()
    def generate(
        self,
        condition: Tensor,
        n_samples: int = 1,
        bridge_example_ts: Optional[Tensor] = None,
        sampler: str = "ddim",
        ddim_steps: Optional[int] = None,
        eta: float = 0.0,
        clip_denoised: bool = True,
        **kwargs: Any,
    ) -> Tensor:
        """
        Generate time series samples.

        Args:
            condition: Text embedding (B, text_dim) or None for unconditional
            n_samples: Number of samples to generate per condition
            bridge_example_ts: Example time series for prototype extraction (B, L, C)
                Must be provided for Bridge model!
            sampler: "ddpm" or "ddim"
            ddim_steps: Number of steps for DDIM sampling (default: num_steps)
            eta: DDIM eta parameter (0 = deterministic, 1 = stochastic like DDPM)
            clip_denoised: Whether to clip denoised samples to [-1, 1]

        Returns:
            Generated time series (B, n_samples, L, C)
        """
        if bridge_example_ts is None:
            raise ValueError("Bridge model requires 'bridge_example_ts' for generation")

        B = bridge_example_ts.shape[0]
        device = self.device

        # Convert to (B, C, L)
        if bridge_example_ts.dim() == 3 and bridge_example_ts.shape[1] != self.n_var:
            bridge_example_ts = bridge_example_ts.permute(0, 2, 1)

        # Extract latent prototypes
        c, mask = self.cond_stage_model(bridge_example_ts)

        # Get text embedding if using condition
        text_embedding = condition if self.use_condition else None

        # Expand for n_samples
        if n_samples > 1:
            c = c.repeat_interleave(n_samples, dim=0)
            if mask is not None:
                mask = mask.repeat_interleave(n_samples, dim=0)
            if text_embedding is not None:
                text_embedding = text_embedding.repeat_interleave(n_samples, dim=0)

        # Initialize from noise
        shape = (B * n_samples, self.n_var, self.seq_length)
        x = torch.randn(shape, device=device)

        # Sampling loop
        if sampler == "ddpm":
            x = self._ddpm_sample(x, c, mask, text_embedding, clip_denoised)
        elif sampler == "ddim":
            steps = ddim_steps if ddim_steps is not None else self.num_steps
            x = self._ddim_sample(x, c, mask, text_embedding, steps, eta, clip_denoised)
        else:
            raise ValueError(f"Unknown sampler: {sampler}")

        # Convert back to (B, L, C)
        x = x.permute(0, 2, 1)

        # Reshape to (B, n_samples, L, C)
        x = x.view(B, n_samples, self.seq_length, self.n_var)

        return x

    def _ddpm_sample(
        self,
        x: Tensor,
        c: Tensor,
        mask: Optional[Tensor],
        text_embedding: Optional[Tensor],
        clip_denoised: bool,
    ) -> Tensor:
        """DDPM sampling loop."""
        B = x.shape[0]
        device = x.device

        for t_int in reversed(range(self.num_steps)):
            t = torch.full((B,), t_int, device=device, dtype=torch.long)

            # Get p(x_{t-1} | x_t)
            mean, variance, _ = self._p_mean_variance(
                x, t, c, mask, text_embedding, clip_denoised
            )

            # Sample x_{t-1}
            noise = torch.randn_like(x) if t_int > 0 else torch.zeros_like(x)
            x = mean + torch.sqrt(variance) * noise

        return x

    def _ddim_sample(
        self,
        x: Tensor,
        c: Tensor,
        mask: Optional[Tensor],
        text_embedding: Optional[Tensor],
        steps: int,
        eta: float,
        clip_denoised: bool,
    ) -> Tensor:
        """DDIM sampling loop with fewer steps."""
        import numpy as np

        B = x.shape[0]
        device = x.device

        # Create timestep schedule - use linspace to get exactly `steps` timesteps
        timesteps = np.linspace(0, self.num_steps - 1, steps, dtype=int).tolist()
        timesteps = list(reversed(timesteps))

        for i, t_int in enumerate(timesteps):
            t = torch.full((B,), t_int, device=device, dtype=torch.long)

            # Predict noise
            model_output = self.model(
                x=x,
                timesteps=t,
                context=c,
                mask=mask,
                y=None,
                cond_drop_prob=0.0,
                text_embedding=text_embedding,
            )
            pred_noise = model_output.pred

            # Predict x_0
            x_0 = self._predict_x0_from_noise(x, t, pred_noise)
            if clip_denoised:
                x_0 = torch.clamp(x_0, -1.0, 1.0)

            # Get next timestep
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
            else:
                t_prev = 0

            # DDIM sampling formula
            alpha = self.alphas_cumprod[t_int]
            alpha_prev = self.alphas_cumprod[t_prev] if t_prev > 0 else torch.tensor(1.0, device=device)

            sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha)) * torch.sqrt(1 - alpha / alpha_prev)

            # Compute x_{t-1}
            dir_xt = torch.sqrt(1 - alpha_prev - sigma ** 2) * pred_noise
            noise = torch.randn_like(x) if sigma > 0 else torch.zeros_like(x)
            x = torch.sqrt(alpha_prev) * x_0 + dir_xt + sigma * noise

        return x

    @property
    def device(self) -> torch.device:
        """Get device of the model."""
        return next(self.parameters()).device
