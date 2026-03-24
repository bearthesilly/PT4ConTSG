"""
TTS-CGAN: Transformer Time-Series Conditional GAN.

Label-conditioned GAN for time series generation using Transformer architecture.

Reference:
    TTS-CGAN: A Transformer Time-Series Conditional GAN for Biosignal Data Augmentation
    https://arxiv.org/abs/2206.13676

Original implementation:
    references/tts-cgan/
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from contsg.models.base import BaseGANModule
from contsg.models.ttscgan_modules import Generator, Discriminator
from contsg.registry import Registry


@Registry.register_model("ttscgan", aliases=["tcg"])
class TTSCGAN(BaseGANModule):
    """
    TTS-CGAN: Transformer-based Conditional GAN for Time Series Generation.

    This model generates time series conditioned on class labels using:
    - Generator: Transformer encoder with label embedding
    - Discriminator: Patch-based Transformer with auxiliary classifier

    Training uses WGAN-GP loss with auxiliary classification loss.

    Data Format:
        - Internal: (B, C, 1, L) image-like format (for conv compatibility)
        - Output: (B, n_samples, L, C) standard ContTSG format

    Condition:
        - Label: (B,) integer class indices

    Example:
        config = ExperimentConfig(
            model=TTSCGANModelConfig(
                num_classes=5,
                latent_dim=100,
            ),
            condition=ConditionConfig(
                label=LabelConditionConfig(enabled=True, num_classes=5),
            ),
        )
        model = TTSCGAN(config)
        samples = model.generate(labels, n_samples=10)
    """

    SUPPORTED_STAGES = ["finetune"]  # Single-stage training

    def _build_generator(self) -> None:
        """Build the Generator network."""
        cfg = self.config.model
        data_cfg = self.config.data
        cond_cfg = self.config.condition

        # Get dimensions
        self.seq_len = data_cfg.seq_length
        self.n_var = data_cfg.n_var
        self.latent_dim = getattr(cfg, "latent_dim", 100)
        self.num_classes = cond_cfg.label.num_classes

        # Build generator
        self.generator = Generator(
            seq_len=self.seq_len,
            channels=self.n_var,
            num_classes=self.num_classes,
            latent_dim=self.latent_dim,
            data_embed_dim=getattr(cfg, "data_embed_dim", 10),
            label_embed_dim=getattr(cfg, "label_embed_dim", 10),
            depth=getattr(cfg, "g_depth", 3),
            num_heads=getattr(cfg, "g_num_heads", 5),
            forward_drop_rate=getattr(cfg, "g_dropout", 0.5),
            attn_drop_rate=getattr(cfg, "g_attn_dropout", 0.5),
        )

    def _build_discriminator(self) -> None:
        """Build the Discriminator network."""
        cfg = self.config.model

        # Compute patch size (default: divide evenly, or use config)
        d_patch_size = getattr(cfg, "d_patch_size", 1)

        # Validate patch size
        if self.seq_len % d_patch_size != 0:
            # Find closest valid patch size
            for ps in range(d_patch_size, 0, -1):
                if self.seq_len % ps == 0:
                    d_patch_size = ps
                    break

        # Build discriminator
        self.discriminator = Discriminator(
            in_channels=self.n_var,
            patch_size=d_patch_size,
            data_emb_size=getattr(cfg, "d_embed_dim", 50),
            label_emb_size=getattr(cfg, "label_embed_dim", 10),
            seq_length=self.seq_len,
            depth=getattr(cfg, "d_depth", 3),
            n_classes=self.num_classes,
            num_heads=getattr(cfg, "d_num_heads", 5),
            drop_p=getattr(cfg, "d_dropout", 0.5),
            forward_drop_p=getattr(cfg, "d_dropout", 0.5),
        )

        # Loss functions
        self.cls_criterion = nn.CrossEntropyLoss()

        # Loss weights
        self.lambda_cls = getattr(cfg, "lambda_cls", 1.0)
        self.lambda_gp = getattr(cfg, "lambda_gp", 10.0)

    def _post_init(self) -> None:
        """Build attr-to-label map for fallback label extraction."""
        self._build_attr_to_label_map()

    def _ts_to_img(self, ts: Tensor) -> Tensor:
        """
        Convert time series to image format.

        Args:
            ts: Time series (B, L, C)

        Returns:
            Image format (B, C, 1, L)
        """
        return ts.permute(0, 2, 1).unsqueeze(2)

    def _img_to_ts(self, img: Tensor) -> Tensor:
        """
        Convert image format to time series.

        Args:
            img: Image format (B, C, 1, L)

        Returns:
            Time series (B, L, C)
        """
        return img.squeeze(2).permute(0, 2, 1)

    def _gradient_penalty(self, real: Tensor, fake: Tensor) -> Tensor:
        """
        Compute WGAN-GP gradient penalty.

        Args:
            real: Real samples (B, C, 1, L)
            fake: Fake samples (B, C, 1, L)

        Returns:
            Gradient penalty (scalar)
        """
        B = real.size(0)
        device = real.device

        # Random interpolation coefficient
        alpha = torch.rand(B, 1, 1, 1, device=device)
        interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

        # Discriminator output on interpolated
        d_interp, _ = self.discriminator(interpolated)

        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interp,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interp),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Compute penalty (use reshape for non-contiguous tensors)
        gradients = gradients.reshape(B, -1)
        gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gp

    def discriminator_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Compute discriminator loss.

        Loss components:
        1. Wasserstein loss: -E[D(real)] + E[D(fake)]
        2. Gradient penalty: lambda_gp * GP
        3. Classification loss: lambda_cls * CE(real)

        Args:
            batch: Batch dictionary with "ts" and "label"

        Returns:
            Dictionary with d_loss and component losses
        """
        ts = batch["ts"]  # (B, L, C)
        labels = self._extract_labels_from_batch(batch)  # (B,)
        B = ts.size(0)
        device = ts.device

        # Convert to image format
        real_imgs = self._ts_to_img(ts)  # (B, C, 1, L)

        # Generate fake samples
        noise = torch.randn(B, self.latent_dim, device=device)
        fake_labels = torch.randint(0, self.num_classes, (B,), device=device)
        fake_imgs = self.generator(noise, fake_labels)  # (B, C, 1, L)

        # Discriminator forward on real and fake
        r_adv, r_cls = self.discriminator(real_imgs)
        f_adv, _ = self.discriminator(fake_imgs.detach())

        # Wasserstein loss
        d_adv_loss = -r_adv.mean() + f_adv.mean()

        # Gradient penalty
        gp = self._gradient_penalty(real_imgs, fake_imgs.detach())

        # Classification loss on real samples
        d_cls_loss = self.cls_criterion(r_cls, labels)

        # Total discriminator loss
        d_loss = d_adv_loss + self.lambda_gp * gp + self.lambda_cls * d_cls_loss

        return {
            "d_loss": d_loss,
            "d_adv_loss": d_adv_loss.detach(),
            "d_gp_loss": gp.detach(),
            "d_cls_loss": d_cls_loss.detach(),
        }

    def generator_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Compute generator loss.

        Loss components:
        1. Adversarial loss: -E[D(fake)]
        2. Classification loss: lambda_cls * CE(fake)

        Args:
            batch: Batch dictionary with "ts"

        Returns:
            Dictionary with g_loss and component losses
        """
        ts = batch["ts"]
        B = ts.size(0)
        device = ts.device

        # Generate fake samples
        noise = torch.randn(B, self.latent_dim, device=device)
        fake_labels = torch.randint(0, self.num_classes, (B,), device=device)
        fake_imgs = self.generator(noise, fake_labels)

        # Discriminator forward on fake
        g_adv, g_cls = self.discriminator(fake_imgs)

        # Generator losses
        g_adv_loss = -g_adv.mean()
        g_cls_loss = self.cls_criterion(g_cls, fake_labels)

        # Total generator loss
        g_loss = g_adv_loss + self.lambda_cls * g_cls_loss

        return {
            "g_loss": g_loss,
            "g_adv_loss": g_adv_loss.detach(),
            "g_cls_loss": g_cls_loss.detach(),
        }

    @torch.no_grad()
    def generate(
        self,
        condition: Tensor,
        n_samples: int = 1,
        use_ema: bool = True,
        **kwargs,
    ) -> Tensor:
        """
        Generate time series samples from label or attribute conditions.

        Args:
            condition: Either:
                - Class labels (B,) as integer indices
                - Attributes (B, A) to be converted to labels
            n_samples: Number of samples to generate per condition
            use_ema: Whether to use EMA weights for generation
            **kwargs: Additional arguments (ignored)

        Returns:
            Generated samples (B, n_samples, L, C)
        """
        self.eval()

        # Handle condition format
        if condition.dim() == 1:
            # Already 1D labels
            labels = condition.long()
        elif condition.dim() == 2:
            # 2D attributes - convert to labels using mixin method
            labels = self._attrs_to_labels_tensor(condition)
        else:
            raise ValueError(
                f"TTS-CGAN expects 1D label tensor (B,) or 2D attribute tensor (B, A), "
                f"got shape {condition.shape}."
            )
        B = labels.size(0)
        device = labels.device

        # Optionally load EMA weights
        saved_weights = None
        if use_ema and hasattr(self, "_ema_params") and len(self._ema_params) > 0:
            saved_weights = self.save_ema_weights()
            self.load_ema_weights()

        try:
            all_samples = []
            for _ in range(n_samples):
                # Generate noise
                noise = torch.randn(B, self.latent_dim, device=device)

                # Generate fake images
                fake_imgs = self.generator(noise, labels)  # (B, C, 1, L)

                # Convert to time series format
                ts = self._img_to_ts(fake_imgs)  # (B, L, C)
                all_samples.append(ts)

            # Stack samples: (n_samples, B, L, C) -> (B, n_samples, L, C)
            result = torch.stack(all_samples, dim=0).permute(1, 0, 2, 3)

        finally:
            # Restore original weights if we loaded EMA
            if saved_weights is not None:
                self.restore_weights(saved_weights)

        return result
