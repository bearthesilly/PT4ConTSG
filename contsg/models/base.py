"""
Base Lightning module for conditional time series generation models.

This module defines the base class that all generation models should inherit from.
It provides standardized training, validation, and generation interfaces.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import Tensor

from contsg.config.label_infer import infer_attr_label_map
from contsg.config.schema import ExperimentConfig


class LabelExtractionMixin:
    """
    Mixin class for label extraction from attributes.

    Provides methods to convert attribute tuples to label indices using
    observed combination mapping. Used by label-conditioned models like
    TTSCGAN and TimeVQVAE.

    Requires:
        self.config.condition.attribute to be properly configured
    """

    def _build_attr_to_label_map(self) -> None:
        """
        Build mapping from attribute tuples to label indices.

        This method should be called in _post_init() for models that need
        to derive labels from attributes (e.g., ttscgan, timevqvae).

        Sets:
            self._attr_to_label_dict: Dict mapping attr tuple -> label index
            self._num_attr_ops: List of num_classes per attribute
        """
        self._attr_to_label_dict: Optional[Dict[Tuple[int, ...], int]] = None
        self._num_attr_ops: Optional[List[int]] = None

        attr_cfg = self.config.condition.attribute
        if not (attr_cfg.enabled and attr_cfg.discrete_configs):
            return

        num_attr_ops = [int(cfg["num_classes"]) for cfg in attr_cfg.discrete_configs]
        self._attr_to_label_dict = infer_attr_label_map(
            self.config.data.data_folder, num_attr_ops
        )
        self._num_attr_ops = num_attr_ops

    def _extract_labels_from_batch(self, batch: Dict[str, Tensor]) -> Tensor:
        """
        Extract labels from batch with fallback to attribute-derived labels.

        Priority:
            1. batch["label"] - direct label
            2. batch["labels"] - alternative key
            3. batch["attrs"] - compute from attributes using Cartesian product

        Args:
            batch: Batch dictionary from dataloader

        Returns:
            Labels tensor of shape (B,)

        Raises:
            ValueError: If no valid label source found
        """
        if "label" in batch:
            labels = batch["label"]
        elif "labels" in batch:
            labels = batch["labels"]
        elif "attrs" in batch and self._attr_to_label_dict is not None:
            labels = self._attrs_to_labels_tensor(batch["attrs"])
        else:
            raise ValueError(
                "Label-conditioned model expects 'label' or 'labels' in batch, "
                "or 'attrs' with attribute conditioning enabled."
            )

        return labels.long()

    def _attrs_to_labels_tensor(self, attrs: Tensor) -> Tensor:
        """
        Convert attribute tensor to label indices.

        Args:
            attrs: Attribute tensor (B, A) where A is number of attributes

        Returns:
            Label indices (B,)

        Raises:
            ValueError: If attribute-to-label mapping not initialized
        """
        if self._attr_to_label_dict is None:
            raise ValueError(
                "Attribute-to-label mapping not initialized. "
                "Ensure condition.attribute.discrete_configs is set."
            )

        attrs = attrs.long()
        if attrs.dim() == 1:
            attrs = attrs.unsqueeze(1)

        # Handle unknown attributes (value -1) by mapping to last class
        num_attr_ops_tensor = torch.tensor(
            self._num_attr_ops, device=attrs.device, dtype=attrs.dtype
        )
        attrs = torch.where(attrs == -1, num_attr_ops_tensor.unsqueeze(0) - 1, attrs)

        # Convert attribute tuples to label indices
        labels = torch.tensor(
            [self._attr_to_label_dict[tuple(attr.tolist())] for attr in attrs],
            device=attrs.device,
        )
        return labels.long()


class BaseGeneratorModule(LabelExtractionMixin, pl.LightningModule):
    """
    Base class for all conditional time series generation models.

    This class provides a standardized interface for:
    - Model construction via `_build_model()`
    - Training via `training_step()`
    - Validation via `validation_step()`
    - Sample generation via `generate()`
    - Optimizer configuration via `configure_optimizers()`

    All model implementations should inherit from this class and implement
    the abstract methods.

    Attributes:
        config: The experiment configuration
        model: The underlying neural network (built in _build_model)
        use_condition: Whether to use conditioning in this stage
        learning_rate: Learning rate (can be overridden per stage)

    Class Attributes:
        SUPPORTED_STAGES: List of supported training stages. If not defined,
            all stages are assumed to be supported.

    Example:
        @Registry.register_model("my_model")
        class MyModel(BaseGeneratorModule):
            SUPPORTED_STAGES = ["pretrain", "finetune"]  # Two-stage support

            def _build_model(self):
                self.encoder = nn.Linear(...)
                self.decoder = nn.Linear(...)

            def forward(self, batch):
                loss = self._compute_loss(batch)
                return {"loss": loss}

            def generate(self, condition, n_samples=1):
                return self._sample(condition, n_samples)
    """

    # Subclasses can override this to specify supported stages
    # If not defined, all stages are supported
    SUPPORTED_STAGES: Optional[List[str]] = None

    def __init__(
        self,
        config: ExperimentConfig,
        use_condition: bool = True,
        learning_rate: Optional[float] = None,
        **kwargs,
    ):
        """
        Initialize the generator module.

        Args:
            config: Complete experiment configuration
            use_condition: Whether to use conditioning in this training stage
            learning_rate: Learning rate override for multi-stage training
            **kwargs: Additional arguments for load_from_checkpoint compatibility
        """
        super().__init__()

        self.config = config
        self.use_condition = use_condition
        self._learning_rate = learning_rate  # Per-stage LR override

        self.save_hyperparameters({
            "config": config.model_dump(mode="json"),
            "use_condition": use_condition,
            "learning_rate": learning_rate,
        })

        # Build the model architecture
        self._build_model()

        # Initialize any model-specific state
        self._post_init()

    @abstractmethod
    def _build_model(self) -> None:
        """
        Build the model architecture.

        This method should create all neural network modules and assign them
        as attributes (e.g., self.encoder, self.decoder, etc.).

        Implementations should use `self.config.model` for model-specific
        parameters and `self.config.data` for data-related parameters.

        Example:
            def _build_model(self):
                cfg = self.config.model
                self.encoder = nn.TransformerEncoder(...)
                self.diffusion = GaussianDiffusion(steps=cfg.diffusion_steps)
        """
        raise NotImplementedError

    def _post_init(self) -> None:
        """
        Optional post-initialization hook.

        Override this method to perform any additional setup after
        `_build_model()` has been called.
        """
        pass

    @abstractmethod
    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Forward pass for training.

        Args:
            batch: Dictionary containing:
                - "ts": Time series tensor (B, L, C)
                - "cap_emb": Caption embedding (B, D) or (B, S, D)
                - "attrs": Attribute tensor (B, A) if applicable
                - Additional model-specific inputs

        Returns:
            Dictionary containing at least:
                - "loss": Total training loss (scalar tensor)
            Optional additional keys for logging (e.g., "mse_loss", "kl_loss")

        Example:
            def forward(self, batch):
                ts = batch["ts"]
                cond = batch["cap_emb"]
                noise_pred, noise = self.diffusion(ts, cond)
                loss = F.mse_loss(noise_pred, noise)
                return {"loss": loss, "mse_loss": loss}
        """
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        condition: Tensor,
        n_samples: int = 1,
        **kwargs: Any,
    ) -> Tensor:
        """
        Generate time series samples given a condition.

        Args:
            condition: Conditioning tensor (e.g., text embedding)
                Shape: (B, D) or (B, S, D)
            n_samples: Number of samples to generate per condition
            **kwargs: Additional generation parameters (e.g., guidance_scale)

        Returns:
            Generated time series tensor
            Shape: (B, n_samples, L, C) or (B * n_samples, L, C)
        """
        raise NotImplementedError

    # =========================================================================
    # Training Methods
    # =========================================================================

    def training_step(
        self,
        batch: Dict[str, Tensor],
        batch_idx: int,
    ) -> Tensor:
        """
        Execute a single training step.

        Args:
            batch: Batch dictionary from dataloader
            batch_idx: Index of the batch

        Returns:
            Training loss
        """
        outputs = self(batch)

        # Log all output values
        for name, value in outputs.items():
            if isinstance(value, Tensor) and value.numel() == 1:
                prog_bar = name == "loss"
                self.log(f"train/{name}", value, prog_bar=prog_bar, on_step=True, on_epoch=True)

        return outputs["loss"]

    def validation_step(
        self,
        batch: Dict[str, Tensor],
        batch_idx: int,
    ) -> Tensor:
        """
        Execute a single validation step.

        Args:
            batch: Batch dictionary from dataloader
            batch_idx: Index of the batch

        Returns:
            Validation loss
        """
        outputs = self(batch)

        # Log all output values
        for name, value in outputs.items():
            if isinstance(value, Tensor) and value.numel() == 1:
                self.log(f"val/{name}", value, sync_dist=True, on_epoch=True)

        return outputs["loss"]

    def test_step(
        self,
        batch: Dict[str, Tensor],
        batch_idx: int,
    ) -> Dict[str, Tensor]:
        """
        Execute a single test step.

        Args:
            batch: Batch dictionary from dataloader
            batch_idx: Index of the batch

        Returns:
            Dictionary with test metrics
        """
        outputs = self(batch)

        for name, value in outputs.items():
            if isinstance(value, Tensor) and value.numel() == 1:
                self.log(f"test/{name}", value, sync_dist=True)

        return outputs

    # =========================================================================
    # Optimizer Configuration
    # =========================================================================

    def configure_optimizers(self) -> Any:
        """
        Configure optimizers and learning rate schedulers.

        Uses per-stage learning rate if provided, otherwise falls back
        to default from config.

        Returns:
            Optimizer or tuple of (optimizer_list, scheduler_list)
        """
        train_cfg = self.config.train

        # Use per-stage LR if provided, otherwise default
        lr = self._learning_rate if self._learning_rate is not None else train_cfg.lr

        # Create optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=train_cfg.weight_decay,
        )

        # Create scheduler
        if train_cfg.scheduler == "none":
            return optimizer

        if train_cfg.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=train_cfg.epochs,
                **train_cfg.scheduler_params,
            )
        elif train_cfg.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=train_cfg.scheduler_params.get("step_size", 100),
                gamma=train_cfg.scheduler_params.get("gamma", 0.5),
            )
        elif train_cfg.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=train_cfg.scheduler_params.get("factor", 0.5),
                patience=train_cfg.scheduler_params.get("patience", 10),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                },
            }
        else:
            return optimizer

        return [optimizer], [scheduler]

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def device_type(self) -> str:
        """Device type string ('cuda' or 'cpu')."""
        return next(self.parameters()).device.type

    def get_progress_bar_dict(self) -> Dict[str, Any]:
        """Customize progress bar display."""
        items = super().get_progress_bar_dict()
        # Remove version number
        items.pop("v_num", None)
        return items


class DiffusionMixin:
    """
    Mixin class for diffusion-based generation models.

    Provides common diffusion operations like noise scheduling,
    forward diffusion, and reverse sampling.
    """

    def linear_beta_schedule(
        self,
        timesteps: int,
        beta_start: Optional[float] = None,
        beta_end: Optional[float] = None,
    ) -> Tensor:
        """Linear noise schedule."""
        if beta_start is None or beta_end is None:
            scale = 1000 / timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
        return torch.linspace(float(beta_start), float(beta_end), timesteps)

    def quad_beta_schedule(self, timesteps: int, beta_start: float, beta_end: float) -> Tensor:
        """Quadratic schedule in sqrt space (legacy)."""
        start = float(beta_start) ** 0.5
        end = float(beta_end) ** 0.5
        return torch.linspace(start, end, timesteps) ** 2

    def cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> Tensor:
        """Cosine noise schedule from improved DDPM."""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def q_sample(
        self,
        x_start: Tensor,
        t: Tensor,
        noise: Optional[Tensor] = None,
        sqrt_alphas_cumprod: Optional[Tensor] = None,
        sqrt_one_minus_alphas_cumprod: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward diffusion: add noise to data.

        Args:
            x_start: Clean data (B, L, C)
            t: Timestep indices (B,)
            noise: Optional pre-generated noise
            sqrt_alphas_cumprod: Precomputed sqrt(alpha_bar)
            sqrt_one_minus_alphas_cumprod: Precomputed sqrt(1 - alpha_bar)

        Returns:
            Noisy data and noise
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha = sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = sqrt_one_minus_alphas_cumprod[t]

        # Expand dimensions for broadcasting
        while sqrt_alpha.dim() < x_start.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

        noisy = sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
        return noisy, noise


class BaseGANModule(LabelExtractionMixin, pl.LightningModule):
    """
    Base class for GAN-based time series generation models.

    This class provides a standardized interface for GAN training with:
    - Two optimizers (Generator and Discriminator)
    - Alternating training steps with manual optimization
    - Exponential Moving Average (EMA) for generator weights
    - Standard generate() interface for evaluation compatibility

    All GAN implementations should inherit from this class and implement
    the abstract methods.

    Attributes:
        config: The experiment configuration
        generator: The generator network (built in _build_generator)
        discriminator: The discriminator network (built in _build_discriminator)
        use_condition: Whether to use conditioning
        automatic_optimization: Set to False for manual G/D optimization

    Example:
        @Registry.register_model("my_gan")
        class MyGAN(BaseGANModule):
            def _build_generator(self):
                self.generator = nn.Sequential(...)

            def _build_discriminator(self):
                self.discriminator = nn.Sequential(...)

            def generator_step(self, batch):
                # Compute generator loss
                return {"g_loss": g_loss}

            def discriminator_step(self, batch):
                # Compute discriminator loss
                return {"d_loss": d_loss}

            def generate(self, condition, n_samples=1):
                return self._sample(condition, n_samples)
    """

    # GANs typically use single-stage training
    SUPPORTED_STAGES: Optional[List[str]] = None

    def __init__(
        self,
        config: ExperimentConfig,
        use_condition: bool = True,
        learning_rate: Optional[float] = None,
        **kwargs,
    ):
        """
        Initialize the GAN module.

        Args:
            config: Complete experiment configuration
            use_condition: Whether to use conditioning
            learning_rate: Learning rate override (applies to both G and D if not
                           using model-specific g_lr/d_lr)
            **kwargs: Additional arguments for load_from_checkpoint compatibility
        """
        super().__init__()

        self.config = config
        self.use_condition = use_condition
        self._learning_rate = learning_rate

        # Disable automatic optimization for manual G/D updates
        self.automatic_optimization = False

        self.save_hyperparameters({
            "config": config.model_dump(mode="json"),
            "use_condition": use_condition,
            "learning_rate": learning_rate,
        })

        # Build generator and discriminator
        self._build_generator()
        self._build_discriminator()

        # Setup EMA for generator
        self._setup_ema()

        # Post-initialization hook
        self._post_init()

    @abstractmethod
    def _build_generator(self) -> None:
        """
        Build the generator network.

        This method should create the generator and assign it to self.generator.

        Example:
            def _build_generator(self):
                self.generator = Generator(
                    latent_dim=self.config.model.latent_dim,
                    ...
                )
        """
        raise NotImplementedError

    @abstractmethod
    def _build_discriminator(self) -> None:
        """
        Build the discriminator network.

        This method should create the discriminator and assign it to
        self.discriminator. It may also set up loss functions.

        Example:
            def _build_discriminator(self):
                self.discriminator = Discriminator(...)
                self.adv_criterion = nn.BCEWithLogitsLoss()
        """
        raise NotImplementedError

    def _post_init(self) -> None:
        """
        Optional post-initialization hook.

        Override this method to perform any additional setup after
        both networks have been built.
        """
        pass

    @abstractmethod
    def generator_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Compute generator loss for one training step.

        Args:
            batch: Batch dictionary from dataloader

        Returns:
            Dictionary containing at least:
                - "g_loss": Generator loss (scalar tensor)
            Optional additional keys for logging (e.g., "g_adv_loss", "g_cls_loss")
        """
        raise NotImplementedError

    @abstractmethod
    def discriminator_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Compute discriminator loss for one training step.

        Args:
            batch: Batch dictionary from dataloader

        Returns:
            Dictionary containing at least:
                - "d_loss": Discriminator loss (scalar tensor)
            Optional additional keys for logging (e.g., "d_adv_loss", "d_gp_loss")
        """
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        condition: Tensor,
        n_samples: int = 1,
        **kwargs: Any,
    ) -> Tensor:
        """
        Generate time series samples given a condition.

        Args:
            condition: Conditioning tensor (e.g., class labels)
                Shape: (B,) for labels or (B, D) for embeddings
            n_samples: Number of samples to generate per condition
            **kwargs: Additional generation parameters

        Returns:
            Generated time series tensor
            Shape: (B, n_samples, L, C)
        """
        raise NotImplementedError

    # =========================================================================
    # Training Methods
    # =========================================================================

    def training_step(
        self,
        batch: Dict[str, Tensor],
        batch_idx: int,
    ) -> None:
        """
        Execute alternating G/D training step.

        Training order:
        1. Train Discriminator
        2. Train Generator (every n_critic steps)
        3. Update EMA weights

        Args:
            batch: Batch dictionary from dataloader
            batch_idx: Index of the batch
        """
        g_opt, d_opt = self.optimizers()

        # Get n_critic from model config (default 1)
        n_critic = getattr(self.config.model, "n_critic", 1)

        # ----- Train Discriminator -----
        d_outputs = self.discriminator_step(batch)
        d_opt.zero_grad()
        self.manual_backward(d_outputs["d_loss"])

        # Gradient clipping if configured
        grad_clip = getattr(self.config.train, "gradient_clip_val", 0.0)
        if grad_clip > 0:
            self.clip_gradients(d_opt, gradient_clip_val=grad_clip)

        d_opt.step()

        # Log discriminator metrics
        for name, value in d_outputs.items():
            if isinstance(value, Tensor) and value.numel() == 1:
                self.log(f"train/{name}", value.detach(), prog_bar=(name == "d_loss"))

        # ----- Train Generator (every n_critic steps) -----
        if (batch_idx + 1) % n_critic == 0:
            g_outputs = self.generator_step(batch)
            g_opt.zero_grad()
            self.manual_backward(g_outputs["g_loss"])

            if grad_clip > 0:
                self.clip_gradients(g_opt, gradient_clip_val=grad_clip)

            g_opt.step()

            # Update EMA
            self._update_ema()

            # Log generator metrics
            for name, value in g_outputs.items():
                if isinstance(value, Tensor) and value.numel() == 1:
                    self.log(f"train/{name}", value.detach(), prog_bar=(name == "g_loss"))

    def validation_step(
        self,
        batch: Dict[str, Tensor],
        batch_idx: int,
    ) -> Dict[str, Tensor]:
        """
        Execute validation step.

        Computes both G and D losses for monitoring.
        Note: Uses torch.enable_grad() to allow gradient penalty computation.

        Args:
            batch: Batch dictionary from dataloader
            batch_idx: Index of the batch

        Returns:
            Dictionary with validation metrics
        """
        # Enable gradients for gradient penalty computation in discriminator_step
        with torch.enable_grad():
            d_outputs = self.discriminator_step(batch)
            g_outputs = self.generator_step(batch)

        # Log all metrics
        for name, value in d_outputs.items():
            if isinstance(value, Tensor) and value.numel() == 1:
                self.log(f"val/{name}", value.detach(), sync_dist=True)

        for name, value in g_outputs.items():
            if isinstance(value, Tensor) and value.numel() == 1:
                self.log(f"val/{name}", value.detach(), sync_dist=True)

        # Log combined loss for ModelCheckpoint compatibility
        combined_loss = d_outputs["d_loss"].detach() + g_outputs["g_loss"].detach()
        self.log("val/loss", combined_loss, sync_dist=True)

        return {**d_outputs, **g_outputs}

    def test_step(
        self,
        batch: Dict[str, Tensor],
        batch_idx: int,
    ) -> Dict[str, Tensor]:
        """Execute test step."""
        with torch.enable_grad():
            d_outputs = self.discriminator_step(batch)
            g_outputs = self.generator_step(batch)

        for name, value in {**d_outputs, **g_outputs}.items():
            if isinstance(value, Tensor) and value.numel() == 1:
                self.log(f"test/{name}", value.detach(), sync_dist=True)

        # Log combined loss for consistency
        combined_loss = d_outputs["d_loss"].detach() + g_outputs["g_loss"].detach()
        self.log("test/loss", combined_loss, sync_dist=True)

        return {**d_outputs, **g_outputs}

    # =========================================================================
    # Optimizer Configuration
    # =========================================================================

    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        """
        Configure optimizers for Generator and Discriminator.

        Returns:
            List of [generator_optimizer, discriminator_optimizer]
        """
        model_cfg = self.config.model
        train_cfg = self.config.train

        # Get learning rates (model-specific or fallback to config)
        base_lr = self._learning_rate if self._learning_rate else train_cfg.lr
        g_lr = getattr(model_cfg, "g_lr", base_lr)
        d_lr = getattr(model_cfg, "d_lr", base_lr)

        # Get Adam betas
        beta1 = getattr(model_cfg, "beta1", 0.0)
        beta2 = getattr(model_cfg, "beta2", 0.9)

        # Create optimizers
        g_opt = torch.optim.AdamW(
            self.generator.parameters(),
            lr=g_lr,
            betas=(beta1, beta2),
            weight_decay=train_cfg.weight_decay,
        )
        d_opt = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=d_lr,
            betas=(beta1, beta2),
            weight_decay=train_cfg.weight_decay,
        )

        return [g_opt, d_opt]

    # =========================================================================
    # EMA Methods
    # =========================================================================

    def _setup_ema(self) -> None:
        """Initialize Exponential Moving Average for generator weights."""
        self._ema_decay = getattr(self.config.model, "ema", 0.995)
        self._ema_params: List[Tensor] = []

        # Store initial generator parameters on CPU
        for p in self.generator.parameters():
            self._ema_params.append(p.data.clone().cpu())

    def _update_ema(self) -> None:
        """Update EMA weights after generator step."""
        with torch.no_grad():
            for ema_p, p in zip(self._ema_params, self.generator.parameters()):
                ema_p.mul_(self._ema_decay).add_(
                    (1 - self._ema_decay) * p.data.cpu()
                )

    def load_ema_weights(self) -> None:
        """Load EMA weights into generator (for evaluation)."""
        with torch.no_grad():
            for ema_p, p in zip(self._ema_params, self.generator.parameters()):
                p.data.copy_(ema_p.to(p.device))

    def save_ema_weights(self) -> List[Tensor]:
        """Save current generator weights before loading EMA."""
        return [p.data.clone() for p in self.generator.parameters()]

    def restore_weights(self, saved_weights: List[Tensor]) -> None:
        """Restore saved generator weights."""
        with torch.no_grad():
            for saved_p, p in zip(saved_weights, self.generator.parameters()):
                p.data.copy_(saved_p)

    # =========================================================================
    # Checkpoint Hooks
    # =========================================================================

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Save EMA parameters in checkpoint."""
        checkpoint["ema_params"] = self._ema_params

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Load EMA parameters from checkpoint."""
        if "ema_params" in checkpoint:
            self._ema_params = checkpoint["ema_params"]

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_generator_parameters(self) -> int:
        """Number of trainable generator parameters."""
        return sum(p.numel() for p in self.generator.parameters() if p.requires_grad)

    @property
    def num_discriminator_parameters(self) -> int:
        """Number of trainable discriminator parameters."""
        return sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad)

    @property
    def device_type(self) -> str:
        """Device type string ('cuda' or 'cpu')."""
        return next(self.parameters()).device.type
