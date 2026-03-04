"""
Custom callbacks for ConTSG training.

Provides additional logging and monitoring capabilities.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ProgressBar
from tqdm import tqdm


class LoggingCallback(Callback):
    """
    Custom logging callback for experiment tracking.

    Logs additional information during training and validation.
    """

    def __init__(self, log_every_n_steps: int = 100):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_train_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Log epoch start."""
        pass

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Log epoch end with summary statistics."""
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics

        # Log learning rate
        if hasattr(trainer, "optimizers") and trainer.optimizers:
            lr = trainer.optimizers[0].param_groups[0]["lr"]
            pl_module.log("train/lr", lr, on_epoch=True)

    def on_validation_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Log validation summary."""
        pass


class GradientMonitorCallback(Callback):
    """
    Monitor gradient and parameter statistics during training.

    Logs:
    - train/grad_norm: Total gradient L2 norm
    - train/param_norm: Total parameter L2 norm
    - train/grad_norm_max: Maximum per-parameter gradient norm

    Useful for debugging training instabilities.
    """

    def __init__(
        self,
        log_every_n_steps: int = 50,
        log_grad_norm: bool = True,
        log_param_norm: bool = True,
    ):
        """
        Initialize the gradient monitor callback.

        Args:
            log_every_n_steps: Log statistics every N steps
            log_grad_norm: Whether to log gradient norms
            log_param_norm: Whether to log parameter norms
        """
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.log_grad_norm = log_grad_norm
        self.log_param_norm = log_param_norm

    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: Any,
    ) -> None:
        """Log gradient and parameter statistics before optimizer step."""
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        import torch

        # Compute gradient norms
        if self.log_grad_norm:
            grad_norms = []
            total_grad_norm_sq = 0.0

            for param in pl_module.parameters():
                if param.grad is not None:
                    param_grad_norm = param.grad.data.norm(2).item()
                    grad_norms.append(param_grad_norm)
                    total_grad_norm_sq += param_grad_norm ** 2

            if grad_norms:
                total_grad_norm = total_grad_norm_sq ** 0.5
                pl_module.log("train/grad_norm", total_grad_norm, on_step=True, on_epoch=False)
                pl_module.log("train/grad_norm_max", max(grad_norms), on_step=True, on_epoch=False)

        # Compute parameter norms
        if self.log_param_norm:
            total_param_norm_sq = 0.0
            for param in pl_module.parameters():
                if param.requires_grad:
                    total_param_norm_sq += param.data.norm(2).item() ** 2

            total_param_norm = total_param_norm_sq ** 0.5
            pl_module.log("train/param_norm", total_param_norm, on_step=True, on_epoch=False)


class SlimProgressBar(ProgressBar):
    """
    Slim progress bar for non-interactive environments (Slurm, pipes).

    Instead of tqdm progress bars that spam logs, outputs one line per epoch:
        [Epoch 1/700] train_loss=0.2345, val_loss=0.1890, lr=1e-3 (2m 30s)

    Usage:
        trainer = pl.Trainer(
            callbacks=[SlimProgressBar()],
            enable_progress_bar=True,  # Must be True since SlimProgressBar is a ProgressBar
        )
    """

    def __init__(self, log_every_n_steps: int = 50):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self._epoch_start_time: float = 0.0
        self._train_batch_count: int = 0

    def enable(self) -> bool:
        """Always enabled (we handle our own output)."""
        return True

    def disable(self) -> None:
        """Cannot be disabled."""
        pass

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Print training start info."""
        max_epochs = trainer.max_epochs or 0
        print(f"\n[Training] Starting {max_epochs} epochs...", flush=True)

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Record epoch start time."""
        self._epoch_start_time = time.time()
        self._train_batch_count = 0

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Optionally log batch progress."""
        self._train_batch_count += 1

        # Log every N steps for long epochs
        if self.log_every_n_steps > 0 and self._train_batch_count % self.log_every_n_steps == 0:
            total_batches = trainer.num_training_batches
            if total_batches:
                pct = self._train_batch_count / total_batches * 100
                print(f"  [Epoch {trainer.current_epoch + 1}] Batch {self._train_batch_count}/{total_batches} ({pct:.0f}%)", flush=True)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Print epoch summary."""
        epoch = trainer.current_epoch + 1
        max_epochs = trainer.max_epochs or 0
        elapsed = time.time() - self._epoch_start_time

        # Format elapsed time
        if elapsed < 60:
            elapsed_str = f"{elapsed:.0f}s"
        elif elapsed < 3600:
            elapsed_str = f"{elapsed / 60:.1f}m"
        else:
            elapsed_str = f"{elapsed / 3600:.1f}h"

        # Get metrics
        metrics = trainer.callback_metrics
        train_loss = metrics.get("train/loss", metrics.get("train_loss"))
        val_loss = metrics.get("val/loss", metrics.get("val_loss"))

        # Get learning rate
        lr = None
        if trainer.optimizers:
            lr = trainer.optimizers[0].param_groups[0]["lr"]

        # Build message
        parts = [f"[Epoch {epoch}/{max_epochs}]"]
        if train_loss is not None:
            parts.append(f"train_loss={train_loss:.4f}")
        if val_loss is not None:
            parts.append(f"val_loss={val_loss:.4f}")
        if lr is not None:
            parts.append(f"lr={lr:.2e}")
        parts.append(f"({elapsed_str})")

        print(" ".join(parts), flush=True)

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Skip validation progress output."""
        pass

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Skip validation end output (already included in epoch summary)."""
        pass

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Print training completion."""
        print(f"\n[Training] Completed!", flush=True)

    # Disable tqdm-related methods by returning disabled tqdm bars
    def init_train_tqdm(self):
        return tqdm(disable=True)

    def init_validation_tqdm(self):
        return tqdm(disable=True)

    def init_test_tqdm(self):
        return tqdm(disable=True)

    def init_predict_tqdm(self):
        return tqdm(disable=True)

    def on_sanity_check_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        pass

    def on_sanity_check_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        pass
