"""
Multi-stage training orchestrator.

Handles sequential training stages (e.g., pretrain → finetune) with
automatic checkpoint passing between stages.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)

from contsg.config.schema import StageConfig, ExperimentConfig
from contsg.train.callbacks import GradientMonitorCallback, SlimProgressBar
from contsg.utils.progress import get_progress_mode, ProgressMode

logger = logging.getLogger(__name__)


class StageCheckpoint:
    """Tracks checkpoints across training stages."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self._checkpoints: Dict[str, Path] = {}

    def get_checkpoint(self, stage_name: str) -> Optional[Path]:
        """Get checkpoint path for a stage."""
        return self._checkpoints.get(stage_name)

    def set_checkpoint(self, stage_name: str, path: Path):
        """Set checkpoint path for a stage."""
        self._checkpoints[stage_name] = path

    @property
    def latest(self) -> Optional[Path]:
        """Get the most recently saved checkpoint."""
        if not self._checkpoints:
            return None
        return list(self._checkpoints.values())[-1]


class MultiStageTrainer:
    """
    Orchestrates multi-stage training.

    Handles:
    - Sequential stage execution
    - Checkpoint passing between stages
    - Stage-specific configurations (LR, epochs, freezing)
    - Validation of model stage support
    """

    def __init__(
        self,
        config: ExperimentConfig,
        model_class: Type[pl.LightningModule],
        datamodule: pl.LightningDataModule,
        checkpoint_root: Optional[Path] = None,
    ):
        self.config = config
        self.model_class = model_class
        self.datamodule = datamodule
        self.checkpoint_root = Path(checkpoint_root) if checkpoint_root else config.output_dir
        self.checkpoint_tracker = StageCheckpoint(self.checkpoint_root)

    def validate_stages(self) -> bool:
        """
        Validate that model supports configured stages.

        Returns:
            True if valid, raises ValueError otherwise.
        """
        # Check if model has SUPPORTED_STAGES attribute
        if hasattr(self.model_class, "SUPPORTED_STAGES"):
            supported = self.model_class.SUPPORTED_STAGES
            if supported is None:
                return True
            for stage in self.config.train.stages:
                if stage.name not in supported:
                    raise ValueError(
                        f"Model {self.model_class.__name__} does not support "
                        f"stage '{stage.name}'. Supported stages: {supported}"
                    )
        return True

    def _build_callbacks(self, stage: StageConfig) -> List[pl.Callback]:
        """Build callbacks for a training stage."""
        callbacks = []

        # Checkpoint callback
        stage_dir = self.checkpoint_root / "checkpoints" / stage.name
        stage_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(
            dirpath=stage_dir,
            filename=f"{stage.name}-{{epoch:02d}}-{{val/loss:.4f}}",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        )
        callbacks.append(checkpoint_callback)

        # Early stopping
        if stage.early_stopping_patience > 0:
            early_stop = EarlyStopping(
                monitor="val/loss",
                patience=stage.early_stopping_patience,
                mode="min",
                verbose=True,
            )
            callbacks.append(early_stop)

        # LR Monitor
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

        # Gradient/Parameter Norm Monitor
        if self.config.train.log_grad_norm or self.config.train.log_param_norm:
            callbacks.append(GradientMonitorCallback(
                log_every_n_steps=self.config.train.log_norm_every_n_steps,
                log_grad_norm=self.config.train.log_grad_norm,
                log_param_norm=self.config.train.log_param_norm,
            ))

        return callbacks

    def _freeze_modules(self, model: pl.LightningModule, module_names: List[str]):
        """Freeze specified modules in model."""
        for name in module_names:
            module = getattr(model, name, None)
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = False
                logger.info(f"Frozen module: {name}")
            else:
                logger.warning(f"Module '{name}' not found in model, skipping freeze")

    def _create_trainer(
        self, stage: StageConfig, use_gradient_clip: bool = True
    ) -> pl.Trainer:
        """Create PyTorch Lightning trainer for a stage.

        Args:
            stage: Stage configuration
            use_gradient_clip: Whether to enable gradient clipping.
                Set to False for models using manual optimization (e.g., GANs).
        """
        callbacks = self._build_callbacks(stage)

        # Gradient clipping is not supported for manual optimization
        gradient_clip_val = (
            self.config.train.gradient_clip_val if use_gradient_clip else None
        )

        # Determine progress bar behavior based on mode
        progress_mode = get_progress_mode()
        from contsg.utils.progress import is_interactive

        if progress_mode == ProgressMode.OFF:
            enable_progress_bar = False
        elif progress_mode == ProgressMode.LOG:
            # SlimProgressBar is a ProgressBar subclass, so enable_progress_bar must be True
            enable_progress_bar = True
            callbacks.append(SlimProgressBar())
        elif progress_mode == ProgressMode.TQDM:
            enable_progress_bar = True
        else:  # AUTO
            if is_interactive():
                enable_progress_bar = True
            else:
                # SlimProgressBar is a ProgressBar subclass, so enable_progress_bar must be True
                enable_progress_bar = True
                callbacks.append(SlimProgressBar())

        trainer = pl.Trainer(
            max_epochs=stage.epochs,
            accelerator="auto",
            devices=1,
            callbacks=callbacks,
            gradient_clip_val=gradient_clip_val,
            accumulate_grad_batches=self.config.train.accumulate_grad_batches,
            val_check_interval=self.config.train.val_check_interval,
            limit_train_batches=self.config.train.limit_train_batches,
            limit_val_batches=self.config.train.limit_val_batches,
            limit_test_batches=self.config.train.limit_test_batches,
            num_sanity_val_steps=self.config.train.num_sanity_val_steps,
            default_root_dir=self.checkpoint_root,
            enable_progress_bar=enable_progress_bar,
            logger=self._create_logger(stage),
        )

        return trainer

    def _create_logger(self, stage: StageConfig):
        """Create logger for a stage."""
        from pytorch_lightning.loggers import TensorBoardLogger

        return TensorBoardLogger(
            save_dir=self.checkpoint_root / "logs",
            name=stage.name,
        )

    def _create_model(
        self,
        stage: StageConfig,
        checkpoint_path: Optional[Path] = None,
    ) -> pl.LightningModule:
        """
        Create model for a stage.

        Args:
            stage: Stage configuration
            checkpoint_path: Optional checkpoint to load from

        Returns:
            Configured model instance
        """
        # Build model kwargs from config
        model_kwargs = {
            "config": self.config,  # Pass full ExperimentConfig
            "learning_rate": stage.lr,
            "use_condition": stage.use_condition,
        }

        if checkpoint_path is not None:
            if stage.name == "finetune" and hasattr(self.model_class, "load_vqvae_from_checkpoint"):
                logger.info(
                    "Loading VQ-VAE weights from checkpoint for finetune stage: %s",
                    checkpoint_path,
                )
                model = self.model_class(
                    **model_kwargs,
                    current_stage="pretrain",
                )
                model.load_vqvae_from_checkpoint(checkpoint_path, strict=False)
            else:
                # Load from checkpoint
                logger.info(f"Loading model from checkpoint: {checkpoint_path}")
                model = self.model_class.load_from_checkpoint(
                    checkpoint_path,
                    **model_kwargs,
                )
        else:
            model = self.model_class(**model_kwargs)

        # Call model's set_stage method if it exists (for two-stage models)
        if hasattr(model, 'set_stage'):
            model.set_stage(stage.name)
            logger.info(f"Called model.set_stage('{stage.name}')")

        # Freeze modules if specified (additional freeze from config)
        if stage.freeze_modules:
            self._freeze_modules(model, stage.freeze_modules)

        return model

    def run_stage(self, stage: StageConfig) -> Path:
        """
        Run a single training stage.

        Args:
            stage: Stage configuration

        Returns:
            Path to best checkpoint from this stage
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting Stage: {stage.name}")
        logger.info(f"  Epochs: {stage.epochs}")
        logger.info(f"  LR: {stage.lr}")
        logger.info(f"  Use Condition: {stage.use_condition}")
        logger.info(f"  Freeze Modules: {stage.freeze_modules}")
        logger.info(f"{'='*60}\n")

        # Resolve checkpoint from previous stage
        checkpoint_path = None
        if stage.load_from_stage:
            checkpoint_path = self.checkpoint_tracker.get_checkpoint(stage.load_from_stage)
            if checkpoint_path is None:
                raise ValueError(
                    f"Stage '{stage.name}' requires checkpoint from "
                    f"'{stage.load_from_stage}' but it wasn't found. "
                    f"Available: {list(self.checkpoint_tracker._checkpoints.keys())}"
                )

        # Create model and trainer
        model = self._create_model(stage, checkpoint_path)

        # Check if model uses manual optimization (e.g., GANs)
        # Gradient clipping is not supported for manual optimization in Lightning
        use_gradient_clip = getattr(model, "automatic_optimization", True)
        trainer = self._create_trainer(stage, use_gradient_clip=use_gradient_clip)

        # Train
        trainer.fit(model, datamodule=self.datamodule)

        # Get best checkpoint path
        best_ckpt = trainer.checkpoint_callback.best_model_path
        if best_ckpt:
            best_ckpt = Path(best_ckpt)
            self.checkpoint_tracker.set_checkpoint(stage.name, best_ckpt)
            logger.info(f"Stage '{stage.name}' completed. Best checkpoint: {best_ckpt}")
        else:
            # Fallback to last checkpoint
            stage_dir = self.checkpoint_root / "checkpoints" / stage.name
            last_ckpt = stage_dir / "last.ckpt"
            if last_ckpt.exists():
                self.checkpoint_tracker.set_checkpoint(stage.name, last_ckpt)
                best_ckpt = last_ckpt

        return best_ckpt

    def train(self) -> Path:
        """
        Run all training stages.

        Returns:
            Path to final checkpoint
        """
        # Validate stages
        self.validate_stages()

        stages = self.config.train.stages
        logger.info(f"Starting multi-stage training with {len(stages)} stage(s)")

        # Run each stage
        final_checkpoint = None
        for i, stage in enumerate(stages):
            logger.info(f"\n[Stage {i+1}/{len(stages)}]")
            final_checkpoint = self.run_stage(stage)

        logger.info(f"\n{'='*60}")
        logger.info("All training stages completed!")
        logger.info(f"Final checkpoint: {final_checkpoint}")
        logger.info(f"{'='*60}")

        return final_checkpoint


def run_training(
    config: ExperimentConfig,
    model_class: Type[pl.LightningModule],
    datamodule: pl.LightningDataModule,
) -> Path:
    """
    Convenience function to run multi-stage training.

    Args:
        config: Experiment configuration
        model_class: Lightning module class
        datamodule: Data module

    Returns:
        Path to final checkpoint
    """
    trainer = MultiStageTrainer(config, model_class, datamodule)
    return trainer.train()
