"""
WaveStitch core diffusion model integration for ConTSG.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from contsg.config.schema import ExperimentConfig
from contsg.models.base import BaseGeneratorModule
from contsg.registry import Registry


class DiscreteAttributeEncoder(nn.Module):
    def __init__(self, num_attr_ops: List[int], emb_dim: int) -> None:
        super().__init__()
        if not num_attr_ops:
            raise ValueError("num_attr_ops cannot be empty")
        self.emb_dim = emb_dim
        self.num_attr_ops = [int(x) for x in num_attr_ops]

        shift_vals = [0]
        for count in self.num_attr_ops:
            shift_vals.append(shift_vals[-1] + count)
        shift = torch.tensor(shift_vals, dtype=torch.long)
        self.register_buffer("attr_shift", shift[:-1].unsqueeze(0))
        self.register_buffer(
            "unknown_index",
            torch.tensor(self.num_attr_ops, dtype=torch.long) - 1,
        )
        self.attr_emb = nn.Embedding(int(shift[-1].item()), emb_dim)
        self.empty_emb = nn.Embedding(1, emb_dim)

        self.out_proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, attrs: Tensor, replace_with_empty: bool = False) -> Tensor:
        if attrs.dim() == 1:
            attrs = attrs.unsqueeze(1)
        attrs = attrs.long()
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
        return self.out_proj(emb)


@Registry.register_model("wavestitch", aliases=["wst"])
class WaveStitchModule(BaseGeneratorModule):
    """
    WaveStitch core diffusion model with SSSDS4Imputer backbone.
    """

    SUPPORTED_STAGES = ["finetune"]

    def __init__(
        self,
        config: ExperimentConfig,
        use_condition: bool = True,
        learning_rate: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        self.num_attr_ops: List[int] = []
        super().__init__(config, use_condition=use_condition, learning_rate=learning_rate, **kwargs)

    def _build_model(self) -> None:
        cfg = self.config.model
        data_cfg = self.config.data
        cond_cfg = self.config.condition

        if not cond_cfg.attribute.enabled:
            raise ValueError("WaveStitch requires condition.attribute.enabled = True")
        if not cond_cfg.attribute.discrete_configs:
            raise ValueError("WaveStitch requires condition.attribute.discrete_configs to be set")

        self.n_var = data_cfg.n_var
        self.seq_length = data_cfg.seq_length

        self.num_attr_ops = [disc["num_classes"] for disc in cond_cfg.attribute.discrete_configs]
        self.attr_embed_dim = int(getattr(cfg, "attr_embed_dim", cond_cfg.attribute.output_dim))
        self.cond_channels = int(getattr(cfg, "cond_channels", 16))

        self.attr_encoder = DiscreteAttributeEncoder(
            num_attr_ops=self.num_attr_ops,
            emb_dim=self.attr_embed_dim,
        )
        self.attr_projector = nn.Sequential(
            nn.Linear(self.attr_embed_dim, self.cond_channels),
            nn.GELU(),
        )

        try:
            from contsg.models.wavestitch_modules.sssd_s4 import SSSDS4Imputer
        except ImportError as exc:
            raise ImportError(
                "WaveStitch requires optional dependencies; install with `pip install -e \".[full]\"`."
            ) from exc

        s4_lmax = int(getattr(cfg, "s4_lmax", 100))
        if s4_lmax < self.seq_length:
            s4_lmax = self.seq_length

        self.model = SSSDS4Imputer(
            in_channels=self.n_var,
            res_channels=int(getattr(cfg, "res_channels", 64)),
            skip_channels=int(getattr(cfg, "skip_channels", 64)),
            out_channels=self.n_var,
            num_res_layers=int(getattr(cfg, "num_res_layers", 4)),
            diffusion_step_embed_dim_in=int(getattr(cfg, "diff_step_embed_in", 32)),
            diffusion_step_embed_dim_mid=int(getattr(cfg, "diff_step_embed_mid", 64)),
            diffusion_step_embed_dim_out=int(getattr(cfg, "diff_step_embed_out", 64)),
            cond_channels=self.cond_channels,
            s4_lmax=s4_lmax,
            s4_d_state=int(getattr(cfg, "s4_dstate", 64)),
            s4_dropout=float(getattr(cfg, "s4_dropout", 0.0)),
            s4_bidirectional=bool(getattr(cfg, "s4_bidirectional", True)),
            s4_layernorm=bool(getattr(cfg, "s4_layernorm", True)),
        )

        self.num_steps = int(getattr(cfg, "diffusion_steps", 200))
        beta_start = float(getattr(cfg, "beta_start", 0.0001))
        beta_end = float(getattr(cfg, "beta_end", 0.02))

        betas = torch.linspace(beta_start, beta_end, self.num_steps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        beta_tilde = betas.clone()
        if self.num_steps > 1:
            beta_tilde[1:] = betas[1:] * (1.0 - alpha_bar[:-1]) / (1.0 - alpha_bar[1:])

        alpha_bar_prev = torch.cat([torch.ones(1, device=alpha_bar.device), alpha_bar[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("alpha_bar_prev", alpha_bar_prev)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))
        self.register_buffer("sigma", torch.sqrt(beta_tilde))

    def _encode_attrs(self, attrs: Optional[Tensor], batch_size: int, seq_len: int) -> Tensor:
        device = self.device
        if not self.use_condition:
            return torch.zeros(batch_size, seq_len, self.cond_channels, device=device)
        if attrs is None:
            raise ValueError("WaveStitch expects attrs in batch for conditional training")

        attrs = attrs.long()
        if attrs.dim() == 1:
            attrs = attrs.unsqueeze(1)
        if attrs.shape[1] != len(self.num_attr_ops):
            raise ValueError(
                f"attrs has {attrs.shape[1]} columns, expected {len(self.num_attr_ops)}"
            )

        attr_emb = self.attr_encoder(attrs)
        pooled = attr_emb.mean(dim=1)
        cond = self.attr_projector(pooled)
        return cond.unsqueeze(1).expand(batch_size, seq_len, self.cond_channels)

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        ts = batch["ts"].float()
        attrs = batch.get("attrs")
        if attrs is None:
            attrs = batch.get("attrs_discrete")

        batch_size, seq_len, _ = ts.shape
        cond = self._encode_attrs(attrs, batch_size, seq_len)

        t = torch.randint(0, self.num_steps, (batch_size,), device=ts.device)
        t_embed = t.float().view(-1, 1) / self.num_steps
        noise = torch.randn_like(ts)
        sqrt_alpha = self.sqrt_alpha_bar[t].view(-1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)

        noisy_ts = sqrt_alpha * ts + sqrt_one_minus_alpha * noise
        pred_noise = self.model(noisy_ts, t_embed, cond)
        loss = F.mse_loss(pred_noise, noise.permute(0, 2, 1))
        return {"loss": loss, "mse_loss": loss}

    @torch.no_grad()
    def generate(self, condition: Tensor, n_samples: int = 1, **kwargs: Any) -> Tensor:
        sampler = kwargs.get("sampler", "ddim")
        eta = float(kwargs.get("eta", 0.0))

        if condition.dim() == 1:
            condition = condition.unsqueeze(1)

        batch_size = condition.shape[0]
        device = condition.device
        seq_len = self.seq_length
        tp = kwargs.get("tp")
        if tp is not None:
            seq_len = tp.shape[1]

        attrs = condition.long()
        attr_emb = self.attr_encoder(attrs)
        pooled = attr_emb.mean(dim=1)
        cond = self.attr_projector(pooled)
        cond = cond.repeat_interleave(n_samples, dim=0)
        cond = cond.unsqueeze(1).expand(batch_size * n_samples, seq_len, self.cond_channels)

        x = torch.randn(batch_size * n_samples, seq_len, self.n_var, device=device)

        for t_int in reversed(range(self.num_steps)):
            t = torch.full((x.shape[0],), t_int, device=device, dtype=torch.long)
            t_embed = t.float().view(-1, 1) / self.num_steps
            pred_noise = self.model(x, t_embed, cond)
            pred_noise = pred_noise.permute(0, 2, 1)

            if sampler == "ddpm":
                x = self._ddpm_reverse(x, pred_noise, t)
            else:
                x = self._ddim_reverse(x, pred_noise, t, eta)

        x = x.view(batch_size, n_samples, seq_len, self.n_var)
        return x

    def _ddpm_reverse(self, x_t: Tensor, pred_noise: Tensor, t: Tensor) -> Tensor:
        alpha_t = self.alphas[t].view(-1, 1, 1)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1)
        beta_t = self.betas[t].view(-1, 1, 1)
        sigma_t = self.sigma[t].view(-1, 1, 1)

        coef = beta_t / torch.sqrt(1.0 - alpha_bar_t)
        mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - coef * pred_noise)

        noise = torch.randn_like(x_t)
        mask = (t > 0).float().view(-1, 1, 1)
        return mean + mask * sigma_t * noise

    def _ddim_reverse(self, x_t: Tensor, pred_noise: Tensor, t: Tensor, eta: float) -> Tensor:
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1)
        alpha_bar_prev = self.alpha_bar_prev[t].view(-1, 1, 1)

        x0_pred = (x_t - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)

        if eta == 0.0:
            return (
                torch.sqrt(alpha_bar_prev) * x0_pred
                + torch.sqrt(1.0 - alpha_bar_prev) * pred_noise
            )

        sigma = eta * torch.sqrt(
            (1.0 - alpha_bar_prev)
            / (1.0 - alpha_bar_t)
            * (1.0 - alpha_bar_t / alpha_bar_prev)
        )
        noise = torch.randn_like(x_t)
        coeff = torch.sqrt(1.0 - alpha_bar_prev - sigma**2)
        return torch.sqrt(alpha_bar_prev) * x0_pred + coeff * pred_noise + sigma * noise
