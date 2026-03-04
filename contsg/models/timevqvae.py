"""
TimeVQVAE: VQ-VAE + MaskGIT for label-conditioned time series generation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import Tensor

from contsg.models.base import BaseGeneratorModule
from contsg.models.timevqvae_modules.vqvae import TimeVQVAE
from contsg.models.timevqvae_modules.generators.maskgit import MaskGIT
from contsg.registry import Registry

logger = logging.getLogger(__name__)


@Registry.register_model("timevqvae", aliases=["tvq"])
class TimeVQVAEModule(BaseGeneratorModule):
    """TimeVQVAE with two-stage training (VQ-VAE pretrain + MaskGIT finetune)."""

    SUPPORTED_STAGES = ["pretrain", "finetune"]

    def __init__(
        self,
        config,
        use_condition: bool = True,
        learning_rate: Optional[float] = None,
        current_stage: Optional[str] = None,
        **kwargs,
    ):
        if current_stage is None:
            current_stage = "finetune" if use_condition else "pretrain"
        self._current_stage = current_stage
        super().__init__(config, use_condition, learning_rate, **kwargs)

    def _build_model(self) -> None:
        data_cfg = self.config.data
        self.n_var = data_cfg.n_var
        self.seq_len = data_cfg.seq_length

        self.model_cfg = self._build_timevqvae_config()
        missing = [key for key in ["VQ-VAE", "encoder", "decoder"] if key not in self.model_cfg]
        if missing:
            raise ValueError(f"TimeVQVAE requires config sections: {missing}")

        self.vqvae = TimeVQVAE(self.model_cfg)
        self.maskgit: Optional[MaskGIT] = None

        self.num_classes = self._resolve_num_classes()
        self._build_attr_to_label_map()  # Use shared method from base class

        if self._current_stage == "finetune":
            self._build_maskgit()

        self.set_stage(self._current_stage)

    def _build_timevqvae_config(self) -> dict:
        # Use by_alias=True to get "VQ-VAE" instead of "vqvae"
        cfg_dict = self.config.model.model_dump(mode="python", by_alias=True)
        cfg_dict.pop("name", None)

        # Ensure VQ-VAE key exists (fallback for compatibility)
        if "VQ-VAE" not in cfg_dict:
            if "vqvae" in cfg_dict:
                cfg_dict["VQ-VAE"] = cfg_dict.pop("vqvae")
            elif "vqvae_config" in cfg_dict:
                cfg_dict["VQ-VAE"] = cfg_dict.pop("vqvae_config")

        # Ensure MaskGIT key exists
        if "MaskGIT" not in cfg_dict and "maskgit" in cfg_dict:
            cfg_dict["MaskGIT"] = cfg_dict.pop("maskgit")

        # Fill MaskGIT prior settings from unified `prior` if legacy keys are missing.
        maskgit_cfg = cfg_dict.get("MaskGIT")
        prior_cfg = cfg_dict.get("prior")
        if isinstance(maskgit_cfg, dict) and isinstance(prior_cfg, dict):
            if "prior_model_l" not in maskgit_cfg:
                maskgit_cfg["prior_model_l"] = dict(prior_cfg)
            if "prior_model_h" not in maskgit_cfg:
                maskgit_cfg["prior_model_h"] = dict(prior_cfg)

        cfg_dict["timepoint"] = self.seq_len
        cfg_dict["variable_num"] = self.n_var
        return cfg_dict

    def _resolve_num_classes(self) -> int:
        label_cfg = self.config.condition.label
        if label_cfg.enabled:
            return label_cfg.num_classes

        if self._current_stage == "finetune":
            raise ValueError(
                "TimeVQVAE finetune requires label conditioning enabled to infer num_classes."
            )
        return 1

    def _build_maskgit(self) -> None:
        if self.maskgit is not None:
            return

        maskgit_cfg = self.model_cfg.get("MaskGIT")
        if maskgit_cfg is None:
            raise ValueError("TimeVQVAE requires MaskGIT configuration for finetune stage.")

        self._ensure_vqvae_buffers()

        pretrained_path = self._resolve_pretrained_path()
        self.maskgit = MaskGIT(
            choice_temperatures=maskgit_cfg["choice_temperatures"],
            T=maskgit_cfg["T"],
            config=self.model_cfg,
            n_classes=self.num_classes,
            vqvae=self.vqvae,
            pretrained_model_path=pretrained_path,
        )

    def _resolve_pretrained_path(self) -> Optional[str]:
        pretrained_path = self.model_cfg.get("pretrained_model_path", None)
        if not pretrained_path or pretrained_path == "auto":
            return None
        if "{run_id}" in pretrained_path and "run_id" in self.model_cfg:
            return pretrained_path.format(run_id=self.model_cfg["run_id"])
        return pretrained_path

    def _extract_vqvae_state_dict(self, state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        prefixes = (
            "vqvae.",
            "model.vqvae.",
            "module.vqvae.",
            "model.module.vqvae.",
        )
        vqvae_state = {}
        for key, value in state_dict.items():
            for prefix in prefixes:
                if key.startswith(prefix):
                    vqvae_state[key[len(prefix):]] = value
                    break
        return vqvae_state

    def load_vqvae_from_checkpoint(self, checkpoint_path: str | Path, strict: bool = False) -> None:
        """Load only VQ-VAE weights from a Lightning checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state_dict = state["state_dict"]
        elif isinstance(state, dict):
            state_dict = state
        else:
            raise ValueError(f"Unsupported checkpoint format at {checkpoint_path}")

        vqvae_state = self._extract_vqvae_state_dict(state_dict)
        if not vqvae_state:
            raise ValueError(
                f"No VQ-VAE weights found in checkpoint: {checkpoint_path}"
            )

        incompatible = self.vqvae.load_state_dict(vqvae_state, strict=strict)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            logger.info(
                "Loaded VQ-VAE with missing=%d unexpected=%d from %s",
                len(incompatible.missing_keys),
                len(incompatible.unexpected_keys),
                checkpoint_path,
            )

        self._ensure_vqvae_buffers()

    def _ensure_vqvae_buffers(self) -> None:
        if self.vqvae.encoder_l.num_tokens.item() != 0 and self.vqvae.encoder_h.num_tokens.item() != 0:
            return
        device = next(self.vqvae.parameters()).device
        dummy = torch.zeros(1, self.n_var, self.seq_len, device=device)
        with torch.no_grad():
            _ = self.vqvae.encoder_l(dummy)
            _ = self.vqvae.encoder_h(dummy)

    def set_stage(self, stage: str) -> None:
        if stage not in self.SUPPORTED_STAGES:
            raise ValueError(f"Unknown stage: {stage}. Supported: {self.SUPPORTED_STAGES}")

        self._current_stage = stage

        if stage == "pretrain":
            for p in self.vqvae.parameters():
                p.requires_grad = True
            self.vqvae.train()
            if self.maskgit is not None:
                for p in self.maskgit.parameters():
                    p.requires_grad = False
                self.maskgit.eval()
        else:
            self._build_maskgit()
            for p in self.vqvae.parameters():
                p.requires_grad = False
            self.vqvae.eval()
            if self.maskgit is not None:
                for p in self.maskgit.parameters():
                    p.requires_grad = True
                self.maskgit.train()

    def _extract_labels(self, batch: Dict[str, Tensor]) -> Tensor:
        """Extract labels using shared method, then reshape for TimeVQVAE."""
        labels = self._extract_labels_from_batch(batch)
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)
        return labels

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        ts = batch["ts"].permute(0, 2, 1)

        if self._current_stage == "pretrain":
            loss = self.vqvae(ts)
            return {"loss": loss, "vqvae_loss": loss}

        labels = self._extract_labels(batch)
        if self.maskgit is None:
            self._build_maskgit()
        mask_pred_loss, (mask_pred_loss_l, mask_pred_loss_h) = self.maskgit(ts, labels)
        return {
            "loss": mask_pred_loss,
            "mask_pred_loss": mask_pred_loss,
            "mask_pred_loss_l": mask_pred_loss_l,
            "mask_pred_loss_h": mask_pred_loss_h,
        }

    @torch.no_grad()
    def generate(
        self,
        condition: Tensor,
        n_samples: int = 1,
        **kwargs: Any,
    ) -> Tensor:
        if condition is None:
            raise ValueError("TimeVQVAE generation requires label conditions.")
        if self.maskgit is None:
            self._build_maskgit()

        labels = condition
        if labels.dim() > 1 and labels.shape[-1] > 1 and self._attr_to_label_dict is not None:
            labels = self._extract_labels({"attrs": labels})
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)

        device = next(self.parameters()).device
        sample_batch_size = int(kwargs.get("batch_size", 32))

        all_samples = []
        for label in labels:
            class_index = int(label.item())
            samples = self._sample_class(class_index, n_samples, sample_batch_size, device)
            all_samples.append(samples)

        samples = torch.stack(all_samples, dim=0)
        samples = samples.permute(0, 1, 3, 2)
        return samples

    def _sample_class(self, class_index: int, n_samples: int, batch_size: int, device: torch.device) -> Tensor:
        n_iters = n_samples // batch_size
        is_residual_batch = False
        if n_samples % batch_size > 0:
            n_iters += 1
            is_residual_batch = True

        x_new = []
        for i in range(n_iters):
            b = batch_size
            if (i + 1 == n_iters) and is_residual_batch:
                b = n_samples - ((n_iters - 1) * batch_size)

            embed_ind_l, embed_ind_h = self.maskgit.iterative_decoding(
                num=b, device=device, class_index=class_index
            )
            x_l = self.maskgit.decode_token_ind_to_timeseries(embed_ind_l, "lf")
            x_h = self.maskgit.decode_token_ind_to_timeseries(embed_ind_h, "hf")
            x_new.append(x_l + x_h)

        return torch.cat(x_new, dim=0)
