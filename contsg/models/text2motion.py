"""
Text-to-Motion (CVPR'22) backbone adapted for conditional time series generation.

This implementation reuses the core ideas from references/text-to-motion:
- Movement conv encoder/decoder (downsample by factor 4)
- Autoregressive latent generator with (prior, posterior) and attention

Differences vs the original:
- Uses ConTSG batch format: batch["ts"] (B, L, C) and batch["cap_emb"] (B, D)
- Does not require token-level word/POS inputs (no spacy/glove dependency)
- Supports ConTSG multi-stage training: unconditional AE pretrain -> conditional finetune
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from contsg.config.schema import ExperimentConfig, Text2MotionModelConfig
from contsg.models.base import BaseGeneratorModule
from contsg.registry import Registry


def _init_weight(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.xavier_uniform_(module.weight)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)


def _reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def _kl_gauss(mu1: Tensor, logvar1: Tensor, mu2: Tensor, logvar2: Tensor) -> Tensor:
    sigma1 = torch.exp(0.5 * logvar1)
    sigma2 = torch.exp(0.5 * logvar2)
    kld = torch.log(sigma2 / sigma1) + (torch.exp(logvar1) + (mu1 - mu2) ** 2) / (
        2 * torch.exp(logvar2)
    ) - 0.5
    return kld.sum() / mu1.shape[0]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 300) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, pos: Tensor) -> Tensor:
        return self.pe[pos]


class MovementConvEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_size, output_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)
        self.apply(_init_weight)

    def forward(self, inputs: Tensor) -> Tensor:
        x = inputs.permute(0, 2, 1)
        y = self.main(x).permute(0, 2, 1)
        return self.out_net(y)


class MovementConvDecoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose1d(input_size, hidden_size, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(hidden_size, output_size, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)
        self.apply(_init_weight)

    def forward(self, inputs: Tensor) -> Tensor:
        x = inputs.permute(0, 2, 1)
        y = self.main(x).permute(0, 2, 1)
        return self.out_net(y)


class TextDecoder(nn.Module):
    def __init__(self, text_size: int, input_size: int, output_size: int, hidden_size: int, n_layers: int):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.n_layers = int(n_layers)
        self.emb = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.gru = nn.ModuleList([nn.GRUCell(hidden_size, hidden_size) for _ in range(self.n_layers)])
        self.z2init = nn.Linear(text_size, hidden_size * n_layers)
        self.positional_encoder = PositionalEncoding(hidden_size)
        self.mu_net = nn.Linear(hidden_size, output_size)
        self.logvar_net = nn.Linear(hidden_size, output_size)
        self.apply(_init_weight)

    def get_init_hidden(self, latent: Tensor) -> List[Tensor]:
        hidden = self.z2init(latent)
        hidden = torch.split(hidden, self.hidden_size, dim=-1)
        return list(hidden)

    def forward(self, inputs: Tensor, hidden: List[Tensor], p: Tensor) -> Tuple[Tensor, Tensor, Tensor, List[Tensor]]:
        x_in = self.emb(inputs)
        x_in = x_in + self.positional_encoder(p).to(inputs.device).detach()
        h_in = x_in
        for i in range(self.n_layers):
            hidden[i] = self.gru[i](h_in, hidden[i])
            h_in = hidden[i]
        mu = self.mu_net(h_in)
        logvar = self.logvar_net(h_in)
        z = _reparameterize(mu, logvar)
        return z, mu, logvar, hidden


class TextVAEDecoder(nn.Module):
    def __init__(self, text_size: int, input_size: int, output_size: int, hidden_size: int, n_layers: int):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.n_layers = int(n_layers)
        self.emb = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.z2init = nn.Linear(text_size, hidden_size * n_layers)
        self.gru = nn.ModuleList([nn.GRUCell(hidden_size, hidden_size) for _ in range(self.n_layers)])
        self.positional_encoder = PositionalEncoding(hidden_size)
        self.output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size),
        )
        self.apply(_init_weight)

    def get_init_hidden(self, latent: Tensor) -> List[Tensor]:
        hidden = self.z2init(latent)
        hidden = torch.split(hidden, self.hidden_size, dim=-1)
        return list(hidden)

    def forward(self, inputs: Tensor, last_pred: Tensor, hidden: List[Tensor], p: Tensor) -> Tuple[Tensor, List[Tensor]]:
        h_in = self.emb(inputs)
        h_in = h_in + self.positional_encoder(p).to(inputs.device).detach()
        for i in range(self.n_layers):
            hidden[i] = self.gru[i](h_in, hidden[i])
            h_in = hidden[i]
        pose_pred = self.output(h_in)
        _ = last_pred
        return pose_pred, hidden


class AttLayer(nn.Module):
    def __init__(self, query_dim: int, key_dim: int, value_dim: int) -> None:
        super().__init__()
        self.W_q = nn.Linear(query_dim, value_dim)
        self.W_k = nn.Linear(key_dim, value_dim, bias=False)
        self.W_v = nn.Linear(key_dim, value_dim)
        self.softmax = nn.Softmax(dim=1)
        self.dim = int(value_dim)
        self.apply(_init_weight)

    def forward(self, query: Tensor, key_mat: Tensor) -> Tuple[Tensor, Tensor]:
        query_vec = self.W_q(query).unsqueeze(-1)
        val_set = self.W_v(key_mat)
        key_set = self.W_k(key_mat)
        weights = torch.matmul(key_set, query_vec) / math.sqrt(self.dim)
        co_weights = self.softmax(weights)
        values = val_set * co_weights
        pred = values.sum(dim=1)
        return pred, co_weights


@Registry.register_model("text2motion", aliases=["t2m"])
class Text2Motion(BaseGeneratorModule):
    """
    Text2Motion backbone adapted to ConTSG.

    Training behavior:
    - If `use_condition=False` (e.g. stage 'pretrain'): trains movement autoencoder only.
    - If `use_condition=True` (e.g. stage 'finetune'): trains conditional generator with KL + recon losses.
    """

    SUPPORTED_STAGES = ["pretrain", "finetune"]

    def __init__(self, config: ExperimentConfig, **kwargs: Any) -> None:
        super().__init__(config=config, **kwargs)

    @property
    def model_cfg(self) -> Text2MotionModelConfig:
        cfg = self.config.model
        if not isinstance(cfg, Text2MotionModelConfig):
            raise TypeError(f"Expected Text2MotionModelConfig, got {type(cfg)}")
        return cfg

    def _build_model(self) -> None:
        cfg = self.model_cfg
        data_cfg = self.config.data

        self.text_proj = nn.Linear(self.config.condition.text.input_dim, cfg.text_latent_dim)

        self.mov_enc = MovementConvEncoder(
            input_size=data_cfg.n_var,
            hidden_size=cfg.dim_movement_enc_hidden,
            output_size=cfg.dim_movement_latent,
        )
        self.mov_dec = MovementConvDecoder(
            input_size=cfg.dim_movement_latent,
            hidden_size=cfg.dim_movement_dec_hidden,
            output_size=data_cfg.n_var,
        )

        self.att_layer = AttLayer(
            query_dim=cfg.dim_dec_hidden,
            key_dim=cfg.text_latent_dim,
            value_dim=cfg.dim_att_vec,
        )

        self.seq_pri = TextDecoder(
            text_size=cfg.text_latent_dim,
            input_size=cfg.dim_movement_latent + cfg.dim_att_vec,
            output_size=cfg.dim_z,
            hidden_size=cfg.dim_pri_hidden,
            n_layers=cfg.n_layers_pri,
        )
        self.seq_post = TextDecoder(
            text_size=cfg.text_latent_dim,
            input_size=cfg.dim_movement_latent + cfg.dim_movement_latent + cfg.dim_att_vec,
            output_size=cfg.dim_z,
            hidden_size=cfg.dim_pos_hidden,
            n_layers=cfg.n_layers_pos,
        )
        self.seq_dec = TextVAEDecoder(
            text_size=cfg.text_latent_dim,
            input_size=cfg.dim_movement_latent + cfg.dim_att_vec + cfg.dim_z,
            output_size=cfg.dim_movement_latent,
            hidden_size=cfg.dim_dec_hidden,
            n_layers=cfg.n_layers_dec,
        )

    def set_stage(self, stage: str) -> None:
        """Set current training stage and freeze/unfreeze parameters accordingly.

        Args:
            stage: Training stage ('pretrain' or 'finetune')
        """
        if stage not in self.SUPPORTED_STAGES:
            raise ValueError(f"Unknown stage: {stage}. Supported: {self.SUPPORTED_STAGES}")

        if stage == "pretrain":
            # Train Movement AutoEncoder only
            for p in self.mov_enc.parameters():
                p.requires_grad = True
            for p in self.mov_dec.parameters():
                p.requires_grad = True
            # Freeze conditional generator modules
            for module in [self.text_proj, self.att_layer, self.seq_pri, self.seq_post, self.seq_dec]:
                for p in module.parameters():
                    p.requires_grad = False

        elif stage == "finetune":
            # Freeze Movement AutoEncoder
            for p in self.mov_enc.parameters():
                p.requires_grad = False
            for p in self.mov_dec.parameters():
                p.requires_grad = False
            # Train conditional generator modules
            for module in [self.text_proj, self.att_layer, self.seq_pri, self.seq_post, self.seq_dec]:
                for p in module.parameters():
                    p.requires_grad = True

    def _validate_shapes(self, ts: Tensor) -> Tuple[int, int, int]:
        if ts.dim() != 3:
            raise ValueError(f"Expected batch['ts'] to be (B, L, C), got {ts.shape}")
        B, L, C = ts.shape
        cfg = self.model_cfg
        if cfg.unit_length != 4:
            raise ValueError(
                f"text2motion backbone requires unit_length=4 (conv down/up by 4), got {cfg.unit_length}"
            )
        if L % cfg.unit_length != 0:
            raise ValueError(
                f"seq_length must be divisible by unit_length={cfg.unit_length}, got L={L}"
            )
        return B, L, C

    def _encode_text(self, cap_emb: Tensor) -> Tuple[Tensor, Tensor]:
        text_latent = self.text_proj(cap_emb.float())
        word_hids = text_latent.unsqueeze(1)
        return word_hids, text_latent

    def _initial_movement_token(self, batch_size: int, channels: int, device: torch.device) -> Tensor:
        cfg = self.model_cfg
        zeros = torch.zeros((batch_size, cfg.unit_length, channels), device=device)
        init = self.mov_enc(zeros)
        if init.shape[1] != 1:
            raise ValueError(f"Expected init movement length=1, got {init.shape}")
        return init.squeeze(1)

    def _generate_movements(
        self,
        word_hids: Tensor,
        text_latent: Tensor,
        mov_len: int,
        m_lens: Tensor,
        use_posterior: bool,
        movements_gt: Optional[Tensor] = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> Dict[str, Tensor]:
        cfg = self.model_cfg
        B = text_latent.shape[0]
        device = text_latent.device

        hidden_pri = self.seq_pri.get_init_hidden(text_latent)
        hidden_dec = self.seq_dec.get_init_hidden(text_latent)

        hidden_pos: Optional[List[Tensor]] = None
        if use_posterior:
            hidden_pos = self.seq_post.get_init_hidden(text_latent)
            if movements_gt is None:
                raise ValueError("movements_gt must be provided when use_posterior=True")

        mov_in = self._initial_movement_token(B, self.config.data.n_var, device)

        teacher_force = use_posterior and (random.random() < teacher_forcing_ratio)

        mus_pri: List[Tensor] = []
        logvars_pri: List[Tensor] = []
        mus_pos: List[Tensor] = []
        logvars_pos: List[Tensor] = []
        fake_mov_batch: List[Tensor] = []

        for i in range(mov_len):
            att_vec, _ = self.att_layer(hidden_dec[-1], word_hids)
            tta = m_lens // cfg.unit_length - i

            pri_in = torch.cat([mov_in, att_vec], dim=-1)
            z_pri, mu_pri, logvar_pri, hidden_pri = self.seq_pri(pri_in, hidden_pri, tta)
            mus_pri.append(mu_pri)
            logvars_pri.append(logvar_pri)

            if use_posterior:
                assert movements_gt is not None and hidden_pos is not None
                mov_tgt = movements_gt[:, i]
                pos_in = torch.cat([mov_in, mov_tgt, att_vec], dim=-1)
                z_pos, mu_pos, logvar_pos, hidden_pos = self.seq_post(pos_in, hidden_pos, tta)
                mus_pos.append(mu_pos)
                logvars_pos.append(logvar_pos)
                z = z_pos
            else:
                z = z_pri
                mov_tgt = None

            dec_in = torch.cat([mov_in, att_vec, z], dim=-1)
            fake_mov, hidden_dec = self.seq_dec(dec_in, mov_in, hidden_dec, tta)
            fake_mov_batch.append(fake_mov.unsqueeze(1))

            if teacher_force and mov_tgt is not None:
                mov_in = mov_tgt.detach()
            else:
                mov_in = fake_mov.detach()

        out: Dict[str, Tensor] = {
            "fake_movements": torch.cat(fake_mov_batch, dim=1),
            "mu_pri": torch.cat(mus_pri, dim=0),
            "logvar_pri": torch.cat(logvars_pri, dim=0),
        }
        if use_posterior:
            out["mu_pos"] = torch.cat(mus_pos, dim=0)
            out["logvar_pos"] = torch.cat(logvars_pos, dim=0)
        return out

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        cfg = self.model_cfg
        ts = batch["ts"].float()
        B, L, C = self._validate_shapes(ts)

        movements = self.mov_enc(ts)
        recon_ts = self.mov_dec(movements)

        if not self.use_condition:
            loss_rec = F.smooth_l1_loss(recon_ts, ts)
            return {"loss": loss_rec, "loss_rec": loss_rec}

        if "cap_emb" not in batch:
            raise ValueError("text2motion requires batch['cap_emb'] when use_condition=True")

        cap_emb = batch["cap_emb"]
        word_hids, text_latent = self._encode_text(cap_emb)

        mov_len = movements.shape[1]
        m_lens = torch.full((B,), L, device=ts.device, dtype=torch.long)

        movements_gt = movements.detach() if cfg.detach_movement_latents else movements

        outputs = self._generate_movements(
            word_hids=word_hids,
            text_latent=text_latent,
            mov_len=mov_len,
            m_lens=m_lens,
            use_posterior=True,
            movements_gt=movements_gt,
            teacher_forcing_ratio=cfg.teacher_forcing_ratio,
        )

        fake_movements = outputs["fake_movements"]
        fake_ts = self.mov_dec(fake_movements)

        loss_rec_ts = F.smooth_l1_loss(fake_ts, ts)
        loss_rec_mov = F.smooth_l1_loss(fake_movements, movements_gt)
        loss_kld = _kl_gauss(outputs["mu_pos"], outputs["logvar_pos"], outputs["mu_pri"], outputs["logvar_pri"])

        loss = cfg.lambda_rec_ts * loss_rec_ts + cfg.lambda_rec_mov * loss_rec_mov + cfg.lambda_kld * loss_kld

        return {
            "loss": loss,
            "loss_rec_ts": loss_rec_ts,
            "loss_rec_mov": loss_rec_mov,
            "loss_kld": loss_kld,
        }

    @torch.no_grad()
    def generate(self, condition: Tensor, n_samples: int = 1, **kwargs: Any) -> Tensor:
        _ = kwargs
        cfg = self.model_cfg
        if condition.dim() != 2:
            raise ValueError(f"Expected condition to be (B, D), got {condition.shape}")
        B = condition.shape[0]
        device = condition.device

        seq_len = self.config.data.seq_length
        if seq_len % cfg.unit_length != 0:
            raise ValueError(f"seq_length must be divisible by unit_length={cfg.unit_length}, got {seq_len}")
        mov_len = seq_len // cfg.unit_length

        word_hids, text_latent = self._encode_text(condition)
        m_lens = torch.full((B,), seq_len, device=device, dtype=torch.long)

        samples: List[Tensor] = []
        for _ in range(n_samples):
            gen = self._generate_movements(
                word_hids=word_hids,
                text_latent=text_latent,
                mov_len=mov_len,
                m_lens=m_lens,
                use_posterior=False,
            )
            fake_ts = self.mov_dec(gen["fake_movements"])
            samples.append(fake_ts.unsqueeze(0))

        return torch.cat(samples, dim=0)
