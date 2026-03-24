import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from contsg.models.base import BaseGeneratorModule
from contsg.registry import Registry


# ============================================================
# Utilities
# ============================================================


def default(value, d):
    return d if value is None else value


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B]
        half = self.dim // 2
        device = t.device
        freq = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / max(half - 1, 1)
        )
        emb = t.float().unsqueeze(1) * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class MLP(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AbsNormalize(nn.Module):
    def __init__(self, dim: int = -1, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(x.float())
        x = F.normalize(x, p=1, dim=self.dim, eps=self.eps)
        return x


# ============================================================
# Condition encoder + factor generators
# ============================================================


class ConditionEncoder(nn.Module):
    """
    Shared condition encoder.
    Accepts pre-computed condition embedding from benchmark, then mixes it with timestep.
    """

    def __init__(self, cond_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(hidden_dim)
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.cond_proj(cond) + self.time_embed(t)
        return self.mlp(h)


class BasisBinaryFactorGenerator(nn.Module):
    """
    Parameter-efficient binary factor generator for G-node topic modeling.

    Instead of predicting a full [dim_g, dim_z] matrix from condition,
    we learn K shared basis matrices and only predict K coefficients.
    """

    def __init__(
        self,
        cond_hidden_dim: int,
        dim_g: int,
        dim_z: int,
        num_basis: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim_g = dim_g
        self.dim_z = dim_z
        self.num_basis = num_basis

        self.base = nn.Parameter(torch.randn(dim_g, dim_z) * 0.02)
        self.bases = nn.Parameter(torch.randn(num_basis, dim_g, dim_z) * 0.02)
        self.coeff_head = MLP(cond_hidden_dim, cond_hidden_dim, num_basis, dropout)
        self.gate_head = MLP(cond_hidden_dim, cond_hidden_dim, dim_g, dropout)

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        # h: [B, H]
        coeff = torch.tanh(self.coeff_head(h))  # [B, K]
        # [B, G, D] = [G, D] + sum_k coeff_k * basis_k
        delta = torch.einsum("bk,kgd->bgd", coeff, self.bases)
        binary = self.base.unsqueeze(0) + delta
        # additional row gate: cheap but strong
        row_gate = 1.0 + 0.1 * torch.tanh(self.gate_head(h)).unsqueeze(-1)
        binary = binary * row_gate
        return {"binary": binary, "coeff": coeff}


class StructuredTernaryFactorGenerator(nn.Module):
    """
    Generates effective ternary U/V factors with low-dimensional control.

    We avoid full hypernetwork generation by combining:
      1) shared base matrices,
      2) row scaling, 3) column scaling,
      4) a tiny basis residual.
    """

    def __init__(
        self,
        cond_hidden_dim: int,
        out_dim: int,
        in_dim: int,
        num_basis: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.num_basis = num_basis

        self.base = nn.Parameter(torch.randn(out_dim, in_dim) * 0.02)
        self.bases = nn.Parameter(torch.randn(num_basis, out_dim, in_dim) * 0.02)

        self.row_head = MLP(cond_hidden_dim, cond_hidden_dim, out_dim, dropout)
        self.col_head = MLP(cond_hidden_dim, cond_hidden_dim, in_dim, dropout)
        self.coeff_head = MLP(cond_hidden_dim, cond_hidden_dim, num_basis, dropout)

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        row_scale = 1.0 + 0.1 * torch.tanh(self.row_head(h))  # [B, O]
        col_scale = 1.0 + 0.1 * torch.tanh(self.col_head(h))  # [B, I]
        coeff = torch.tanh(self.coeff_head(h))                # [B, K]

        delta = torch.einsum("bk,koi->boi", coeff, self.bases)
        weight = self.base.unsqueeze(0) + delta
        weight = weight * row_scale.unsqueeze(-1) * col_scale.unsqueeze(1)
        return {
            "weight": weight,
            "row_scale": row_scale,
            "col_scale": col_scale,
            "coeff": coeff,
        }


class UnaryConditioner(nn.Module):
    def __init__(self, cond_hidden_dim: int, model_dim: int, patch_num: int, dropout: float = 0.0):
        super().__init__()
        self.scale = MLP(cond_hidden_dim, cond_hidden_dim, model_dim, dropout)
        self.shift = MLP(cond_hidden_dim, cond_hidden_dim, model_dim, dropout)
        self.segment_gate = MLP(cond_hidden_dim, cond_hidden_dim, patch_num, dropout)

    def forward(self, unary: torch.Tensor, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        # unary: [B, C, P, D]
        scale = 1.0 + 0.1 * torch.tanh(self.scale(h)).unsqueeze(1).unsqueeze(1)
        shift = self.shift(h).unsqueeze(1).unsqueeze(1)
        seg_gate = torch.sigmoid(self.segment_gate(h)).unsqueeze(1).unsqueeze(-1)  # [B,1,P,1]
        unary = unary * scale + shift
        unary = unary * (0.5 + seg_gate)
        return {"unary": unary, "segment_gate": seg_gate}


class PTDynamicFactors(nn.Module):
    def __init__(
        self,
        cond_hidden_dim: int,
        model_dim: int,
        ff_dim: int,
        num_heads: int,
        patch_num: int,
        num_basis_binary: int = 8,
        num_basis_ternary: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        rank = model_dim // num_heads
        out_dim = num_heads * rank

        self.unary = UnaryConditioner(cond_hidden_dim, model_dim, patch_num, dropout)
        self.topic = BasisBinaryFactorGenerator(
            cond_hidden_dim, ff_dim, model_dim, num_basis=num_basis_binary, dropout=dropout
        )

        self.time_u = StructuredTernaryFactorGenerator(
            cond_hidden_dim, out_dim, model_dim, num_basis=num_basis_ternary, dropout=dropout
        )
        self.time_v = StructuredTernaryFactorGenerator(
            cond_hidden_dim, out_dim, model_dim, num_basis=num_basis_ternary, dropout=dropout
        )
        self.channel_u = StructuredTernaryFactorGenerator(
            cond_hidden_dim, out_dim, model_dim, num_basis=num_basis_ternary, dropout=dropout
        )
        self.channel_v = StructuredTernaryFactorGenerator(
            cond_hidden_dim, out_dim, model_dim, num_basis=num_basis_ternary, dropout=dropout
        )

    def forward(self, unary: torch.Tensor, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = {}
        out.update(self.unary(unary, h))
        out["topic"] = self.topic(h)
        out["time_u"] = self.time_u(h)
        out["time_v"] = self.time_v(h)
        out["channel_u"] = self.channel_u(h)
        out["channel_v"] = self.channel_v(h)
        return out


# ============================================================
# PT-style operator with dynamic factors
# ============================================================


class DynamicPTHeadSelection(nn.Module):
    """
    PT/ST-PT-style head dependency with condition-instantiated ternary factors.
    This is the operator-level innovation: condition affects the factor weights,
    not only the hidden activations.
    """

    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.rank = model_dim // num_heads

    def _batched_linear(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # x: [B, ..., D], weight: [B, O, D] -> [B, ..., O]
        return torch.einsum("b...d,bod->b...o", x, weight)

    def _time_messages(
        self,
        qz: torch.Tensor,
        u_weight: torch.Tensor,
        v_weight: torch.Tensor,
    ) -> torch.Tensor:
        # qz: [B, C, P, D], weight: [B, H*R, D]
        B, C, P, _ = qz.shape
        q_u = self._batched_linear(qz, u_weight).view(B, C, P, self.num_heads, self.rank).permute(0, 1, 3, 2, 4)
        q_v = self._batched_linear(qz, v_weight).view(B, C, P, self.num_heads, self.rank).permute(0, 1, 3, 2, 4)
        attn = torch.einsum("bchpr,bchqr->bchpq", q_u, q_v) / math.sqrt(self.rank)
        attn = F.softmax(attn, dim=-1)
        msg = torch.einsum("bchpq,bcq d->bchpd", attn, qz)
        msg = msg.mean(dim=2).permute(0, 1, 3, 2)  # [B,C,D,P]
        return msg.permute(0, 1, 3, 2)             # [B,C,P,D]

    def _channel_messages(
        self,
        qz: torch.Tensor,
        u_weight: torch.Tensor,
        v_weight: torch.Tensor,
    ) -> torch.Tensor:
        # cross-channel dependency at each patch index
        B, C, P, _ = qz.shape
        q_u = self._batched_linear(qz, u_weight).view(B, C, P, self.num_heads, self.rank).permute(0, 2, 3, 1, 4)
        q_v = self._batched_linear(qz, v_weight).view(B, C, P, self.num_heads, self.rank).permute(0, 2, 3, 1, 4)
        attn = torch.einsum("bphcr,bphdr->bphcd", q_u, q_v) / math.sqrt(self.rank)
        attn = F.softmax(attn, dim=-1)
        qz_patch = qz.permute(0, 2, 1, 3)  # [B,P,C,D]
        msg = torch.einsum("bphcd,bpcd->bphcd", attn, qz_patch)
        msg = msg.mean(dim=2).permute(0, 2, 1, 3)  # [B,C,P,D]
        return msg

    def forward(self, qz: torch.Tensor, factors: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        m_t = self._time_messages(qz, factors["time_u"]["weight"], factors["time_v"]["weight"])
        m_c = self._channel_messages(qz, factors["channel_u"]["weight"], factors["channel_v"]["weight"])
        return m_t, m_c


class DynamicPTTopicModeling(nn.Module):
    def __init__(self, model_dim: int, ff_dim: int):
        super().__init__()
        self.model_dim = model_dim
        self.ff_dim = ff_dim
        self.act = AbsNormalize(dim=-1)

    def forward(self, qz: torch.Tensor, binary_weight: torch.Tensor) -> torch.Tensor:
        # qz: [B,C,P,D], binary_weight: [B,G,D]
        qg = torch.einsum("bcpd,bgd->bcpg", qz, binary_weight)
        qg = self.act(qg)
        msg = torch.einsum("bcpg,bgd->bcpd", qg, binary_weight)
        return msg


class DynamicPTBlock(nn.Module):
    def __init__(self, model_dim: int, ff_dim: int, num_heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(model_dim)
        self.head = DynamicPTHeadSelection(model_dim, num_heads)
        self.topic = DynamicPTTopicModeling(model_dim, ff_dim)
        self.out = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, qz: torch.Tensor, unary: torch.Tensor, dyn_factors: Dict[str, torch.Tensor]) -> torch.Tensor:
        old = qz
        qz_n = self.norm(qz)
        m_t, m_c = self.head(qz_n, dyn_factors)
        m_g = self.topic(qz_n, dyn_factors["topic"]["binary"])
        qz = unary + m_t + m_c + m_g
        qz = 0.5 * (qz + old)
        qz = qz + self.out(self.norm(qz))
        return qz


class PTFactorNoisePredictor(nn.Module):
    """
    ST-PT-inspired diffusion noise predictor with condition-instantiated factors.
    """

    def __init__(self, cfg, data_cfg):
        super().__init__()
        self.seq_len = data_cfg.seq_length
        self.n_var = data_cfg.n_var
        self.patch_len = cfg.patch_len
        assert self.seq_len % self.patch_len == 0, "seq_length must be divisible by patch_len"
        self.patch_num = self.seq_len // self.patch_len

        self.model_dim = cfg.d_model
        self.ff_dim = cfg.d_ff
        self.num_heads = cfg.n_heads
        self.num_layers = cfg.e_layers
        self.cond_dim = cfg.cond_dim
        self.cond_hidden_dim = default(getattr(cfg, "cond_hidden_dim", None), cfg.d_model)

        self.patch_embed = nn.Sequential(
            nn.Linear(self.patch_len, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.cond_encoder = ConditionEncoder(self.cond_dim, self.cond_hidden_dim, getattr(cfg, "dropout", 0.0))
        self.factor_generator = PTDynamicFactors(
            self.cond_hidden_dim,
            self.model_dim,
            self.ff_dim,
            self.num_heads,
            self.patch_num,
            num_basis_binary=default(getattr(cfg, "num_basis_binary", None), 8),
            num_basis_ternary=default(getattr(cfg, "num_basis_ternary", None), 4),
            dropout=getattr(cfg, "dropout", 0.0),
        )
        self.blocks = nn.ModuleList(
            [DynamicPTBlock(self.model_dim, self.ff_dim, self.num_heads) for _ in range(self.num_layers)]
        )
        self.out_proj = nn.Linear(self.model_dim, self.patch_len)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        # [B,L,C] -> [B,C,P,patch]
        B, L, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, self.patch_num, self.patch_len)
        return x

    def _unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        # [B,C,P,patch] -> [B,L,C]
        B, C, P, p = x.shape
        x = x.reshape(B, C, P * p).transpose(1, 2)
        return x

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x_patch = self._patchify(x_t)
        unary = self.patch_embed(x_patch)                       # [B,C,P,D]
        h = self.cond_encoder(cond, t)                          # [B,H]
        dyn = self.factor_generator(unary, h)
        unary = dyn["unary"]

        qz = unary
        for blk in self.blocks:
            qz = blk(qz, unary, dyn)

        eps_patch = self.out_proj(qz)
        eps = self._unpatchify(eps_patch)
        return eps


# ============================================================
# Diffusion wrapper following benchmark template
# ============================================================


@dataclass
class DiffusionBuffers:
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    alphas_cumprod_prev: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    sqrt_recip_alphas_cumprod: torch.Tensor
    sqrt_recipm1_alphas_cumprod: torch.Tensor


def make_beta_schedule(T: int, beta_start: float, beta_end: float) -> DiffusionBuffers:
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)
    return DiffusionBuffers(
        betas=betas,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        alphas_cumprod_prev=alphas_cumprod_prev,
        sqrt_alphas_cumprod=torch.sqrt(alphas_cumprod),
        sqrt_one_minus_alphas_cumprod=torch.sqrt(1.0 - alphas_cumprod),
        sqrt_recip_alphas_cumprod=torch.sqrt(1.0 / alphas_cumprod),
        sqrt_recipm1_alphas_cumprod=torch.sqrt(1.0 / alphas_cumprod - 1),
    )


@Registry.register_model("pt_factor_generator")
class PTFactorGeneratorModule(BaseGeneratorModule):
    """
    ConTSG-compatible generator module.

    Main idea:
      - Replace static ST-PT factor matrices by condition-generated dynamic factors.
      - Avoid parameter explosion via basis mixing + structured scaling.
      - Optional classifier-free guidance in generate().
    """

    def _build_model(self):
        cfg = self.config.model
        data_cfg = self.config.data

        self.net = PTFactorNoisePredictor(cfg, data_cfg)
        self.num_steps = int(default(getattr(cfg, "num_steps", None), 1000))
        self.cfg_dropout = float(default(getattr(cfg, "cfg_dropout", None), 0.1))
        self.use_cfg = bool(default(getattr(cfg, "use_cfg", None), True))

        bufs = make_beta_schedule(
            self.num_steps,
            float(default(getattr(cfg, "beta_start", None), 1e-4)),
            float(default(getattr(cfg, "beta_end", None), 0.02)),
        )
        for k, v in bufs.__dict__.items():
            self.register_buffer(k, v)

        self.null_condition = nn.Parameter(torch.zeros(1, cfg.cond_dim))

    # -------------------------
    # conditioning helpers
    # -------------------------
    def _get_condition_from_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "cap_emb" in batch:
            return batch["cap_emb"]
        if "condition" in batch:
            return batch["condition"]
        if "cond" in batch:
            return batch["cond"]
        raise KeyError("Expected one of batch['cap_emb'], batch['condition'], batch['cond']'] for conditioning.")

    def _maybe_drop_condition(self, cond: torch.Tensor) -> torch.Tensor:
        if not self.training or not self.use_cfg or self.cfg_dropout <= 0.0:
            return cond
        keep = (torch.rand(cond.size(0), device=cond.device) > self.cfg_dropout).float().unsqueeze(-1)
        null = self.null_condition.expand(cond.size(0), -1)
        return keep * cond + (1.0 - keep) * null

    # -------------------------
    # q(x_t | x_0)
    # -------------------------
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        noise = default(noise, torch.randn_like(x0))
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_alpha * x0 + sqrt_one_minus * noise

    def predict_eps(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.net(x_t, t, cond)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ts = batch["ts"]  # [B,L,C]
        cond = self._get_condition_from_batch(batch)
        cond = self._maybe_drop_condition(cond)

        B = ts.size(0)
        t = torch.randint(0, self.num_steps, (B,), device=ts.device)
        noise = torch.randn_like(ts)
        x_t = self.q_sample(ts, t, noise)
        pred = self.predict_eps(x_t, t, cond)
        loss = F.mse_loss(pred, noise)
        return {"loss": loss}

    # -------------------------
    # DDIM + optional CFG
    # -------------------------
    @torch.no_grad()
    def _predict_eps_cfg(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, guidance_scale: float) -> torch.Tensor:
        if (not self.use_cfg) or guidance_scale <= 1.0:
            return self.predict_eps(x_t, t, cond)
        null = self.null_condition.expand(cond.size(0), -1)
        eps_uncond = self.predict_eps(x_t, t, null)
        eps_cond = self.predict_eps(x_t, t, cond)
        return eps_uncond + guidance_scale * (eps_cond - eps_uncond)

    @torch.no_grad()
    def _ddim_step(self, x_t: torch.Tensor, t: int, t_prev: int, cond: torch.Tensor, guidance_scale: float = 1.0) -> torch.Tensor:
        B = x_t.size(0)
        t_tensor = torch.full((B,), t, device=x_t.device, dtype=torch.long)
        eps = self._predict_eps_cfg(x_t, t_tensor, cond, guidance_scale)

        a_t = self.alphas_cumprod[t]
        a_prev = self.alphas_cumprod_prev[t] if t_prev < 0 else self.alphas_cumprod[t_prev]

        x0_pred = (x_t - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)
        dir_xt = torch.sqrt(1 - a_prev) * eps
        x_prev = torch.sqrt(a_prev) * x0_pred + dir_xt
        return x_prev

    @torch.no_grad()
    def generate(self, condition: torch.Tensor, n_samples: int = 1, **kwargs) -> torch.Tensor:
        """
        condition: [B,D]
        return: [B, n_samples, L, C]
        """
        device = condition.device
        B = condition.size(0)
        L = self.config.data.seq_length
        C = self.config.data.n_var

        steps = int(kwargs.get("steps", default(getattr(self.config.model, "sample_steps", None), 50)))
        guidance_scale = float(kwargs.get("guidance_scale", default(getattr(self.config.model, "guidance_scale", None), 1.5)))

        cond = condition.unsqueeze(1).repeat(1, n_samples, 1).reshape(B * n_samples, -1)
        x = torch.randn(B * n_samples, L, C, device=device)

        # uniform DDIM stride
        schedule = torch.linspace(self.num_steps - 1, 0, steps, device=device).long().tolist()
        for i, t in enumerate(schedule):
            t_prev = -1 if i == len(schedule) - 1 else schedule[i + 1]
            x = self._ddim_step(x, t, t_prev, cond, guidance_scale=guidance_scale)

        x = x.view(B, n_samples, L, C)
        return x
