# -*- coding: utf-8 -*-
"""
PTFactorGenerator V2 -- Dynamic-factor PT denoiser for ConTSG.

Key improvements over pt4contsg.py (V1)
----------------------------------------
  [1] Dynamic factor architecture: condition instantiates factor matrices
      (U/V ternary, W binary) so the full graph topology changes per sample.
      Per-patch segment gate enables fine-grained temporal control.
  [2] Benchmark-standard diffusion schedule: quad, T=50, beta_end=0.5 --
      matching all baselines (VerbalTS, WaveStitch, TEdit, TimeWeaver, …).
      CFG disabled (cfg_scale=1.0) for fair comparison.
  [3] Scaled dimensions: d_model=128, d_ff=256, e_layers=4,
      cond_hidden_dim=256.
  [4] More bases: num_basis_binary=16, num_basis_ternary=8. AttrEncoder
      for discrete attribute conditioning (synth-m, airquality).
  [5] Learned residual gate replaces fixed 0.5 MFVI damping to prevent
      mean-field over-smoothing of noise predictions.
  [6] CtxTokenizer + per-block cross-attention: global cap_emb is expanded
      into n_ctx pseudo-tokens that patch tokens can selectively attend to,
      enabling fine-grained text-semantic conditioning (text mode only).

V2 additions
------------
  [7] Dual RoPE (time + channel): restores positional awareness that was
      missing in V1; critical for temporal shape fidelity (DTW, CRPS).
  [8] H-variable regularizer changed from 1/sqrt(d_head) to 1/d_head,
      matching the original PT paper's mean-field scaling.
  [9] v-prediction parameterization: predicts v = sqrt(a_bar)*eps - sqrt(1-a_bar)*x0
      instead of ε; more numerically stable across all noise levels.
  [10] Spectral auxiliary loss: FFT-domain constraint preserving
       autocorrelation structure (fixes ACD gap).
  [11] Self-conditioning: feeds the model's own x0 estimate back as input,
       improving sample quality at minimal compute cost.
  [12] Per-patch condition modulation of ternary factors: condition biases
       U projections differently per temporal position for fine-grained
       structural controllability.
  [13] Inter-attribute cross-talk: lightweight MHA lets attribute embeddings
       interact before factor generation, enabling synergistic conditioning.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from contsg.models.base import BaseGeneratorModule
from contsg.registry import Registry


# ============================================================
# Utilities
# ============================================================

def _default(value, d):
    return d if value is None else value


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1)
        )
        emb = t.float().unsqueeze(1) * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class MLP(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AbsNormalize(nn.Module):
    def __init__(self, dim: int = -1, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim, self.eps = dim, eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(x.float())
        return F.normalize(x, p=1, dim=self.dim, eps=self.eps)


# ============================================================
# [V2-7] Rotary Position Embedding (ported from pt4contsg.py)
# ============================================================

def _rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Minimal rotary positional embedding.

    Returns (cos, sin) of shape [1, seq_len, head_dim].
    """

    def __init__(self, head_dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", pos, self.inv_freq.to(device))  # [S, head_dim//2]
        emb = torch.cat([freqs, freqs], dim=-1)                          # [S, head_dim]
        return emb.cos().unsqueeze(0).to(dtype), emb.sin().unsqueeze(0).to(dtype)


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply RoPE to x of shape [..., S, D]."""
    return x * cos + _rotate_half(x) * sin


# ============================================================
# Condition encoders
# ============================================================

class ConditionEncoder(nn.Module):
    """Projects pre-computed condition embedding + timestep → hidden vector."""

    def __init__(self, cond_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(hidden_dim)
        self.cond_proj  = nn.Linear(cond_dim, hidden_dim)
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


class AttrEncoder(nn.Module):
    """Embeds discrete attribute indices → [B, cond_dim] vector.

    Each attribute field gets its own nn.Embedding; all embeddings are
    concatenated and projected to cond_dim.  Used when condition.attribute
    is the primary conditioning modality (synth-m, airquality_beijing).

    [V2-13] When attr_cross_talk=True and there are multiple attribute fields,
    a lightweight MHA lets the per-field embeddings interact before projection.
    """

    def __init__(
        self,
        attr_configs: List[Dict],
        cond_dim: int,
        attr_embed_dim: int = 16,
        dropout: float = 0.0,
        cross_talk: bool = True,
    ) -> None:
        super().__init__()
        self.embeds = nn.ModuleList([
            nn.Embedding(int(cfg["num_classes"]), attr_embed_dim)
            for cfg in attr_configs
        ])
        total_dim = len(attr_configs) * attr_embed_dim

        # [V2-13] Inter-attribute cross-talk
        self.cross_talk: Optional[nn.MultiheadAttention] = None
        if cross_talk and len(attr_configs) > 1:
            self.cross_talk = nn.MultiheadAttention(
                attr_embed_dim, num_heads=max(1, attr_embed_dim // 8),
                dropout=dropout, batch_first=True,
            )
            self.cross_talk_norm = nn.LayerNorm(attr_embed_dim)

        self.proj = nn.Sequential(
            nn.Linear(total_dim, cond_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(cond_dim, cond_dim),
        )
        self.null = nn.Parameter(torch.zeros(cond_dim))

    def forward(self, attrs: torch.Tensor, cond_drop_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """attrs: [B] or [B, A]  →  [B, cond_dim]"""
        if attrs.dim() == 1:
            attrs = attrs.unsqueeze(1)
        parts = [self.embeds[k](attrs[:, k].long()) for k in range(len(self.embeds))]

        # [V2-13] Inter-attribute cross-talk: let attribute embeddings interact
        if self.cross_talk is not None and len(parts) > 1:
            stacked = torch.stack(parts, dim=1)  # [B, K, attr_embed_dim]
            stacked_norm = self.cross_talk_norm(stacked)
            interacted, _ = self.cross_talk(stacked_norm, stacked_norm, stacked_norm)
            stacked = stacked + interacted  # residual
            parts = [stacked[:, k] for k in range(stacked.shape[1])]

        c = self.proj(torch.cat(parts, dim=-1))
        if cond_drop_mask is not None:
            null = self.null.unsqueeze(0).expand(c.shape[0], -1)
            c = torch.where(cond_drop_mask.unsqueeze(1), null, c)
        return c


# ============================================================
# Dynamic factor generators
# ============================================================

class BasisBinaryFactorGenerator(nn.Module):
    """
    Parameter-efficient binary factor:  W_eff = W_base + Σ_k coeff_k * Basis_k
    plus a per-row gate for additional conditioning flexibility.
    """

    def __init__(
        self,
        cond_hidden_dim: int,
        dim_g: int,
        dim_z: int,
        num_basis: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim_g, self.dim_z = dim_g, dim_z
        self.base   = nn.Parameter(torch.randn(dim_g, dim_z) * 0.02)
        self.bases  = nn.Parameter(torch.randn(num_basis, dim_g, dim_z) * 0.02)
        self.coeff_head = MLP(cond_hidden_dim, cond_hidden_dim, num_basis, dropout)
        self.gate_head  = MLP(cond_hidden_dim, cond_hidden_dim, dim_g,    dropout)

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        coeff = torch.tanh(self.coeff_head(h))                     # [B, K]
        delta = torch.einsum("bk,kgd->bgd", coeff, self.bases)    # [B, G, D]
        binary = self.base.unsqueeze(0) + delta
        row_gate = 1.0 + 0.1 * torch.tanh(self.gate_head(h)).unsqueeze(-1)
        binary = binary * row_gate
        return {"binary": binary, "coeff": coeff}


class StructuredTernaryFactorGenerator(nn.Module):
    """
    Generates ternary U/V factors via:
      W_eff = (W_base + Σ_k coeff_k * Basis_k) ⊙ row_scale ⊗ col_scale
    """

    def __init__(
        self,
        cond_hidden_dim: int,
        out_dim: int,
        in_dim: int,
        num_basis: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base   = nn.Parameter(torch.randn(out_dim, in_dim) * 0.02)
        self.bases  = nn.Parameter(torch.randn(num_basis, out_dim, in_dim) * 0.02)
        self.row_head   = MLP(cond_hidden_dim, cond_hidden_dim, out_dim,   dropout)
        self.col_head   = MLP(cond_hidden_dim, cond_hidden_dim, in_dim,    dropout)
        self.coeff_head = MLP(cond_hidden_dim, cond_hidden_dim, num_basis, dropout)

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        row   = 1.0 + 0.1 * torch.tanh(self.row_head(h))
        col   = 1.0 + 0.1 * torch.tanh(self.col_head(h))
        coeff = torch.tanh(self.coeff_head(h))
        delta  = torch.einsum("bk,koi->boi", coeff, self.bases)
        weight = (self.base.unsqueeze(0) + delta) * row.unsqueeze(-1) * col.unsqueeze(1)
        return {"weight": weight, "row_scale": row, "col_scale": col, "coeff": coeff}


class UnaryConditioner(nn.Module):
    """
    AdaLN-style scale/shift + per-patch segment gate.
    """

    def __init__(
        self, cond_hidden_dim: int, model_dim: int, patch_num: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.scale        = MLP(cond_hidden_dim, cond_hidden_dim, model_dim, dropout)
        self.shift        = MLP(cond_hidden_dim, cond_hidden_dim, model_dim, dropout)
        self.segment_gate = MLP(cond_hidden_dim, cond_hidden_dim, patch_num, dropout)

    def forward(self, unary: torch.Tensor, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        scale    = 1.0 + 0.1 * torch.tanh(self.scale(h)).unsqueeze(1).unsqueeze(1)
        shift    = self.shift(h).unsqueeze(1).unsqueeze(1)
        seg_gate = torch.sigmoid(self.segment_gate(h)).unsqueeze(1).unsqueeze(-1)  # [B,1,P,1]
        unary    = unary * scale + shift
        unary    = unary * (0.5 + seg_gate)
        return {"unary": unary, "segment_gate": seg_gate}


class CtxTokenizer(nn.Module):
    """Expands a global condition vector into N_ctx latent pseudo-tokens."""

    def __init__(self, cond_dim: int, model_dim: int, n_ctx: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.n_ctx = n_ctx
        self.proj  = nn.Linear(cond_dim, n_ctx * model_dim)
        self.norm  = nn.LayerNorm(model_dim)
        self.drop  = nn.Dropout(dropout)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """cond: [B, cond_dim]  →  [B, n_ctx, model_dim]"""
        B = cond.shape[0]
        tokens = self.proj(cond).view(B, self.n_ctx, -1)
        return self.drop(self.norm(tokens))


# ============================================================
# [V2-12] Per-patch condition modulation of ternary factors
# ============================================================

class PatchConditionModulator(nn.Module):
    """Produces per-patch bias for U projections in the time dimension.

    Takes the condition hidden vector h [B, cond_hidden_dim] and generates
    a per-patch, per-head bias [B, P, H, R] that is added to the query (qu)
    before computing attention. This allows the condition to steer *which
    patches attend to which neighbours* differently at each temporal position.
    """

    def __init__(
        self, cond_hidden_dim: int, patch_num: int, num_heads: int, rank: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_num = patch_num
        self.num_heads = num_heads
        self.rank = rank
        out_dim = patch_num * num_heads * rank
        self.proj = nn.Sequential(
            nn.Linear(cond_hidden_dim, cond_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(cond_hidden_dim, out_dim),
        )
        # Small init so it starts near identity
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, cond_hidden_dim] → [B, P, H, R]"""
        B = h.shape[0]
        return self.proj(h).view(B, self.patch_num, self.num_heads, self.rank)


class PTDynamicFactors(nn.Module):
    """Assembles all dynamic factor generators -- one call computes all factors."""

    def __init__(
        self,
        cond_hidden_dim: int,
        model_dim: int,
        ff_dim: int,
        num_heads: int,
        patch_num: int,
        num_basis_binary: int = 16,
        num_basis_ternary: int = 8,
        dropout: float = 0.0,
        patch_cond_modulate: bool = True,
    ) -> None:
        super().__init__()
        rank    = model_dim // num_heads
        out_dim = num_heads * rank

        self.unary    = UnaryConditioner(cond_hidden_dim, model_dim, patch_num, dropout)
        self.topic    = BasisBinaryFactorGenerator(
            cond_hidden_dim, ff_dim, model_dim,
            num_basis=num_basis_binary, dropout=dropout,
        )
        self.time_u   = StructuredTernaryFactorGenerator(
            cond_hidden_dim, out_dim, model_dim, num_basis=num_basis_ternary, dropout=dropout
        )
        self.time_v   = StructuredTernaryFactorGenerator(
            cond_hidden_dim, out_dim, model_dim, num_basis=num_basis_ternary, dropout=dropout
        )
        self.channel_u = StructuredTernaryFactorGenerator(
            cond_hidden_dim, out_dim, model_dim, num_basis=num_basis_ternary, dropout=dropout
        )
        self.channel_v = StructuredTernaryFactorGenerator(
            cond_hidden_dim, out_dim, model_dim, num_basis=num_basis_ternary, dropout=dropout
        )

        # [V2-12] Per-patch condition modulation
        self.patch_mod: Optional[PatchConditionModulator] = None
        if patch_cond_modulate:
            self.patch_mod = PatchConditionModulator(
                cond_hidden_dim, patch_num, num_heads, rank, dropout
            )

    def forward(self, unary: torch.Tensor, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = {}
        out.update(self.unary(unary, h))
        out["topic"]     = self.topic(h)
        out["time_u"]    = self.time_u(h)
        out["time_v"]    = self.time_v(h)
        out["channel_u"] = self.channel_u(h)
        out["channel_v"] = self.channel_v(h)
        # [V2-12]
        out["patch_query_bias"] = self.patch_mod(h) if self.patch_mod is not None else None
        return out


# ============================================================
# PT operator with dynamic factors
# ============================================================

class DynamicPTHeadSelection(nn.Module):
    """ST-PT ternary factors where U/V are condition-instantiated per sample.

    [V2-7] Dual RoPE applied to projections before attention.
    [V2-8] Regularizer: 1/d_head instead of 1/sqrt(d_head).
    [V2-12] Per-patch query bias for temporal attention.
    """

    def __init__(self, model_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.rank      = model_dim // num_heads
        # [V2-7] Dual RoPE
        self.rope_time = RotaryEmbedding(self.rank)
        self.rope_chan = RotaryEmbedding(self.rank)

    @staticmethod
    def _bl(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum("b...d,bod->b...o", x, w)

    def _time_msg(
        self,
        qz: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        patch_query_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, C, P, _ = qz.shape
        H, R = self.num_heads, self.rank
        qu = self._bl(qz, u).view(B, C, P, H, R).permute(0, 1, 3, 2, 4)  # [B, C, H, P, R]
        qv = self._bl(qz, v).view(B, C, P, H, R).permute(0, 1, 3, 2, 4)

        # [V2-12] Per-patch condition modulation of query
        if patch_query_bias is not None:
            # patch_query_bias: [B, P, H, R] → permute to [B, H, P, R] → unsqueeze to [B, 1, H, P, R]
            qu = qu + patch_query_bias.permute(0, 2, 1, 3).unsqueeze(1)

        # [V2-7] Apply RoPE along time dimension
        cos, sin = self.rope_time(P, qz.device, qz.dtype)  # [1, P, R]
        # Broadcast over B, C, H dims
        qu = _apply_rope(qu, cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0))
        qv = _apply_rope(qv, cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0))

        # [V2-8] Regularizer: 1/d_head instead of 1/sqrt(d_head)
        attn = torch.einsum("bchpr,bchqr->bchpq", qu, qv) / R
        attn = F.softmax(attn, dim=-1)
        msg  = torch.einsum("bchpq,bcqd->bchpd", attn, qz)
        return msg.mean(dim=2)                               # [B, C, P, D]

    def _chan_msg(self, qz: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, C, P, D = qz.shape
        H, R = self.num_heads, self.rank
        qu = self._bl(qz, u).view(B, C, P, H, R).permute(0, 2, 3, 1, 4)  # [B,P,H,C,R]
        qv = self._bl(qz, v).view(B, C, P, H, R).permute(0, 2, 3, 1, 4)

        # [V2-7] Apply RoPE along channel dimension
        cos, sin = self.rope_chan(C, qz.device, qz.dtype)  # [1, C, R]
        qu = _apply_rope(qu, cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0))
        qv = _apply_rope(qv, cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0))

        # [V2-8] Regularizer: 1/d_head
        attn = torch.einsum("bphcr,bphfr->bphcf", qu, qv) / R
        attn = F.softmax(attn, dim=-1)                       # [B, P, H, C, C]
        qzp  = qz.permute(0, 2, 1, 3)                       # [B, P, C, D]
        msg  = torch.einsum("bphcf,bpfd->bphcd", attn, qzp).mean(dim=2)  # [B, P, C, D]
        return msg.permute(0, 2, 1, 3)                       # [B, C, P, D]

    def forward(
        self,
        qz: torch.Tensor,
        factors: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        m_t = self._time_msg(
            qz,
            factors["time_u"]["weight"],
            factors["time_v"]["weight"],
            patch_query_bias=factors.get("patch_query_bias"),
        )
        m_c = self._chan_msg(qz, factors["channel_u"]["weight"], factors["channel_v"]["weight"])
        return m_t, m_c


class DynamicPTTopicModeling(nn.Module):
    """Binary factor with condition-instantiated weight matrix W."""

    def __init__(self, model_dim: int, ff_dim: int) -> None:
        super().__init__()
        self.act = AbsNormalize(dim=-1)

    def forward(self, qz: torch.Tensor, binary_weight: torch.Tensor) -> torch.Tensor:
        qg  = torch.einsum("bcpd,bgd->bcpg", qz, binary_weight)
        qg  = self.act(qg)
        msg = torch.einsum("bcpg,bgd->bcpd", qg, binary_weight)
        return msg


class DynamicPTBlock(nn.Module):
    """One MFVI iteration with condition-instantiated factors + residual MLP.

    - Learned residual gate replaces fixed 0.5 damping.
    - Optional cross-attention to ctx_tokens (text pseudo-tokens).
    """

    def __init__(self, model_dim: int, ff_dim: int, num_heads: int, n_ctx: int = 0) -> None:
        super().__init__()
        self.norm       = nn.LayerNorm(model_dim)
        self.head       = DynamicPTHeadSelection(model_dim, num_heads)
        self.topic      = DynamicPTTopicModeling(model_dim, ff_dim)
        # Learned residual gate
        self.gate_proj  = nn.Linear(model_dim, model_dim, bias=True)
        self.out        = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        # Optional cross-attention to text pseudo-tokens
        if n_ctx > 0:
            self.cross_norm = nn.LayerNorm(model_dim)
            self.cross_attn = nn.MultiheadAttention(
                model_dim, num_heads, dropout=0.0, batch_first=True
            )

    def forward(
        self,
        qz: torch.Tensor,
        unary: torch.Tensor,
        dyn: Dict[str, torch.Tensor],
        ctx_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        old  = qz
        qz_n = self.norm(qz)
        m_t, m_c = self.head(qz_n, dyn)
        m_g  = self.topic(qz_n, dyn["topic"]["binary"])

        # Learned residual gate
        gate = torch.sigmoid(self.gate_proj(old))       # [B, C, P, D]
        new_candidate = unary + m_t + m_c + m_g
        qz   = gate * old + (1.0 - gate) * new_candidate

        # Cross-attention to text pseudo-tokens
        if ctx_tokens is not None and hasattr(self, "cross_attn"):
            B, C, P, D = qz.shape
            q  = self.cross_norm(qz).reshape(B * C, P, D)
            k  = ctx_tokens.unsqueeze(1).expand(-1, C, -1, -1).reshape(B * C, -1, D)
            ctx_out, _ = self.cross_attn(q, k, k)
            qz = qz + ctx_out.reshape(B, C, P, D)

        qz = qz + self.out(self.norm(qz))
        return qz


# ============================================================
# Full denoiser backbone
# ============================================================

class PTFactorNoisePredictor(nn.Module):
    """
    ST-PT backbone as DDPM predictor with dynamic factor conditioning.

    [V2-11] Self-conditioning: accepts optional x0_self_cond input that is
    concatenated with x_t along the patch dimension before embedding.

    I/O:  x_t [B, L, C]  →  prediction [B, L, C]
    """

    def __init__(self, cfg, data_cfg) -> None:
        super().__init__()
        self.seq_len  = data_cfg.seq_length
        self.n_var    = data_cfg.n_var
        self.patch_len = int(getattr(cfg, "patch_len", 8))
        assert self.seq_len % self.patch_len == 0, (
            f"seq_len ({self.seq_len}) must be divisible by patch_len ({self.patch_len})"
        )
        self.patch_num = self.seq_len // self.patch_len

        model_dim        = int(getattr(cfg, "d_model",         128))
        ff_dim           = int(getattr(cfg, "d_ff",            256))
        n_heads          = int(getattr(cfg, "n_heads",          8))
        n_layers         = int(getattr(cfg, "e_layers",         4))
        cond_dim         = int(getattr(cfg, "cond_dim",         64))
        cond_hidden_dim  = int(getattr(cfg, "cond_hidden_dim",  256))
        num_basis_bin    = int(getattr(cfg, "num_basis_binary",  16))
        num_basis_tern   = int(getattr(cfg, "num_basis_ternary",  8))
        dropout          = float(getattr(cfg, "dropout",        0.1))
        self.n_ctx       = int(getattr(cfg, "n_ctx",             0))
        patch_cond_mod   = bool(_default(getattr(cfg, "patch_cond_modulate", None), True))

        # [V2-11] Self-conditioning: input is [x_t; x0_hat] concatenated along patch dim
        self.self_cond   = bool(_default(getattr(cfg, "self_cond", None), True))
        embed_input_dim  = self.patch_len * 2 if self.self_cond else self.patch_len

        self.patch_embed = nn.Sequential(
            nn.Linear(embed_input_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.cond_encoder = ConditionEncoder(cond_dim, cond_hidden_dim, dropout)
        self.factor_gen   = PTDynamicFactors(
            cond_hidden_dim, model_dim, ff_dim, n_heads, self.patch_num,
            num_basis_binary=num_basis_bin,
            num_basis_ternary=num_basis_tern,
            dropout=dropout,
            patch_cond_modulate=patch_cond_mod,
        )
        # Pseudo-token cross-attention: only when n_ctx > 0
        self.ctx_tok: Optional[CtxTokenizer] = None
        if self.n_ctx > 0:
            self.ctx_tok = CtxTokenizer(cond_dim, model_dim, self.n_ctx, dropout)

        self.blocks = nn.ModuleList([
            DynamicPTBlock(model_dim, ff_dim, n_heads, n_ctx=self.n_ctx)
            for _ in range(n_layers)
        ])
        self.out_proj = nn.Linear(model_dim, self.patch_len)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        return x.transpose(1, 2).reshape(B, C, self.patch_num, self.patch_len)

    def _unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        B, C, P, p = x.shape
        return x.reshape(B, C, P * p).transpose(1, 2)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        x0_self_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_patch = self._patchify(x_t)  # [B, C, P, patch_len]

        # [V2-11] Self-conditioning: concatenate x0_hat along patch dim
        if self.self_cond:
            if x0_self_cond is None:
                x0_patch = torch.zeros_like(x_patch)
            else:
                x0_patch = self._patchify(x0_self_cond)
            x_patch = torch.cat([x_patch, x0_patch], dim=-1)  # [B, C, P, 2*patch_len]

        unary   = self.patch_embed(x_patch)          # [B, C, P, D]
        h       = self.cond_encoder(cond, t)          # [B, H]
        dyn     = self.factor_gen(unary, h)
        unary   = dyn["unary"]

        # Pseudo-tokens for cross-attention (text conditioning only)
        ctx_tokens: Optional[torch.Tensor] = None
        if self.ctx_tok is not None:
            ctx_tokens = self.ctx_tok(cond)           # [B, n_ctx, model_dim]

        qz = unary
        for blk in self.blocks:
            qz = blk(qz, unary, dyn, ctx_tokens=ctx_tokens)
        return self._unpatchify(self.out_proj(qz))    # [B, L, C]


# ============================================================
# Diffusion schedule helper
# ============================================================

def _make_schedule(
    T: int,
    beta_start: float,
    beta_end: float,
    noise_schedule: str = "quad",
) -> Dict[str, torch.Tensor]:
    """Build DDPM diffusion buffers."""
    if noise_schedule == "quad":
        betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, T) ** 2
    else:  # linear
        betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    abar   = torch.cumprod(alphas, dim=0)
    abar_p = torch.cat([torch.ones(1), abar[:-1]])
    return dict(
        betas=betas,
        alphas=alphas,
        alphas_cumprod=abar,
        alphas_cumprod_prev=abar_p,
        sqrt_alphas_cumprod=abar.sqrt(),
        sqrt_one_minus_alphas_cumprod=(1.0 - abar).sqrt(),
        sqrt_recip_alphas_cumprod=(1.0 / abar).sqrt(),
        sqrt_recipm1_alphas_cumprod=(1.0 / abar - 1).sqrt(),
        # [V2-9] Additional buffers for v-prediction
        sqrt_alphas=alphas.sqrt(),
    )


# ============================================================
# [V2-10] Spectral auxiliary loss
# ============================================================

def _spectral_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """FFT-domain loss to preserve autocorrelation structure.

    Computes L1 loss on the magnitude spectrum along the time axis.
    """
    # pred, target: [B, L, C]
    pred_fft   = torch.fft.rfft(pred, dim=1)
    target_fft = torch.fft.rfft(target, dim=1)
    # Magnitude spectrum loss
    mag_loss = F.l1_loss(pred_fft.abs(), target_fft.abs())
    return mag_loss


# ============================================================
# Lightning Module
# ============================================================

@Registry.register_model("pt_factor_generator", aliases=["ptfg"])
class PTFactorGeneratorModule(BaseGeneratorModule):
    """
    ConTSG generator: dynamic factor PT denoiser (V2).

    Config parameters (model.*):
        d_model (int):           Hidden dimension.                Default: 128
        d_ff (int):              Binary factor / FFN dimension.   Default: 256
        n_heads (int):           Ternary factor heads.            Default: 8
        e_layers (int):          Number of DynamicPTBlock layers. Default: 4
        patch_len (int):         Patch length (seq_len % == 0).   Default: 8
        cond_dim (int):          Input condition dimension.        Default: 64
        cond_hidden_dim (int):   Condition encoder width.          Default: 256
        num_basis_binary (int):  Binary factor basis count.        Default: 16
        num_basis_ternary (int): Ternary factor basis count.       Default: 8
        dropout (float):         Dropout rate.                     Default: 0.1
        num_steps (int):         DDPM training steps.              Default: 50
        beta_start (float):      Schedule start beta.              Default: 1e-4
        beta_end (float):        Schedule end beta.                Default: 0.5
        noise_schedule (str):    'quad' or 'linear'.               Default: quad
        sample_steps (int):      DDIM inference steps.             Default: 50
        cfg_dropout (float):     CFG training dropout rate.        Default: 0.0
        guidance_scale (float):  CFG guidance at inference.        Default: 1.0
        use_cfg (bool):          Enable classifier-free guidance.  Default: False
        attr_embed_dim (int):    Per-attribute embedding dim.      Default: 16
        n_ctx (int):             Pseudo-token count for text cross-attn.  Default: 0
        prediction_type (str):   'v' or 'eps'.                    Default: v   [V2-9]
        spectral_loss_weight (float): Weight for spectral loss.   Default: 0.1 [V2-10]
        self_cond (bool):        Enable self-conditioning.         Default: true [V2-11]
        patch_cond_modulate (bool): Per-patch ternary modulation. Default: true [V2-12]
        attr_cross_talk (bool):  Inter-attribute cross-talk.       Default: true [V2-13]
    """

    def _build_model(self) -> None:
        cfg      = self.config.model
        data_cfg = self.config.data
        cond_cfg = self.config.condition

        # ── Denoiser backbone ──────────────────────────────────────────────────
        self.net = PTFactorNoisePredictor(cfg, data_cfg)

        # ── Diffusion schedule (benchmark-standard: quad, T=50, beta_end=0.5) ──
        self.num_steps     = int(_default(getattr(cfg, "num_steps",      None),  50))
        beta_start         = float(_default(getattr(cfg, "beta_start",   None),  1e-4))
        beta_end           = float(_default(getattr(cfg, "beta_end",     None),  0.5))
        noise_schedule     = str(_default(getattr(cfg, "noise_schedule", None),  "quad"))
        bufs = _make_schedule(self.num_steps, beta_start, beta_end, noise_schedule)
        for k, v in bufs.items():
            self.register_buffer(k, v)

        # ── CFG ───────────────────────────────────────────────────────────────
        self.cfg_dropout    = float(_default(getattr(cfg, "cfg_dropout",    None), 0.0))
        self.guidance_scale = float(_default(getattr(cfg, "guidance_scale", None), 1.0))
        self.use_cfg        = bool(_default(getattr(cfg, "use_cfg",         None), False))
        self.sample_steps   = int(_default(getattr(cfg, "sample_steps",     None), 50))

        # ── [V2-9] Prediction type ───────────────────────────────────────────
        self.prediction_type    = str(_default(getattr(cfg, "prediction_type",    None), "v"))
        # ── [V2-10] Spectral loss ────────────────────────────────────────────
        self.spectral_loss_weight = float(_default(getattr(cfg, "spectral_loss_weight", None), 0.1))

        cond_dim = int(getattr(cfg, "cond_dim", 64))
        self.null_condition = nn.Parameter(torch.zeros(1, cond_dim))

        # ── Attribute encoder ────────────────────────────────────────────────
        self._attr_encoder: Optional[AttrEncoder] = None
        attr_cross_talk = bool(_default(getattr(cfg, "attr_cross_talk", None), True))
        if cond_cfg.attribute.enabled and cond_cfg.attribute.discrete_configs:
            attr_embed_dim = int(_default(getattr(cfg, "attr_embed_dim", None), 16))
            dropout        = float(getattr(cfg, "dropout", 0.1))
            self._attr_encoder = AttrEncoder(
                cond_cfg.attribute.discrete_configs,
                cond_dim,
                attr_embed_dim=attr_embed_dim,
                dropout=dropout,
                cross_talk=attr_cross_talk,
            )

        self._text_enabled  = cond_cfg.text.enabled
        self._attr_enabled  = cond_cfg.attribute.enabled
        self._label_enabled = cond_cfg.label.enabled

        # ── [V2-14] Text+Attr fusion projection ────────────────────────────
        self._cond_fusion: Optional[str] = None
        self._fusion_proj: Optional[nn.Module] = None
        if self._text_enabled and self._attr_enabled and self._attr_encoder is not None:
            text_dim = int(getattr(cond_cfg.text, "input_dim", cond_dim))
            self._cond_fusion = str(getattr(cond_cfg, "fusion", "concat"))
            self._fusion_proj = nn.Sequential(
                nn.Linear(text_dim + cond_dim, cond_dim),
                nn.SiLU(),
                nn.Linear(cond_dim, cond_dim),
            )

    # ---------------------------------------------------------------
    # Condition extraction
    # ---------------------------------------------------------------

    def _encode_condition(
        self,
        batch: Dict[str, torch.Tensor],
        force_null: bool = False,
    ) -> torch.Tensor:
        B = batch["ts"].shape[0]
        device = batch["ts"].device

        if force_null or not self.use_condition:
            return self.null_condition.expand(B, -1).to(device)

        # [V2-14] Text+Attr fusion: concat both modalities when both enabled
        if (self._cond_fusion is not None
                and self._fusion_proj is not None
                and "cap_emb" in batch
                and "attrs" in batch
                and self._attr_encoder is not None):
            text_cond = batch["cap_emb"].float().to(device)
            attr_cond = self._attr_encoder(batch["attrs"].to(device))
            cond = self._fusion_proj(torch.cat([text_cond, attr_cond], dim=-1))
        elif self._text_enabled and "cap_emb" in batch:
            cond = batch["cap_emb"].float().to(device)
        elif self._attr_enabled and "attrs" in batch and self._attr_encoder is not None:
            cond = self._attr_encoder(batch["attrs"].to(device))
        elif "condition" in batch:
            cond = batch["condition"].float().to(device)
        elif "label" in batch:
            return self.null_condition.expand(B, -1).to(device)
        else:
            return self.null_condition.expand(B, -1).to(device)

        # CFG training dropout
        if self.training and self.use_cfg and self.cfg_dropout > 0.0:
            keep = (torch.rand(B, device=device) > self.cfg_dropout).bool()
            null = self.null_condition.expand(B, -1).to(device)
            cond = torch.where(keep.unsqueeze(1), cond, null)

        return cond

    # ---------------------------------------------------------------
    # Forward diffusion
    # ---------------------------------------------------------------

    def _q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        sa = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)         # type: ignore[attr-defined]
        sm = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)  # type: ignore[attr-defined]
        return sa * x0 + sm * noise

    # ---------------------------------------------------------------
    # [V2-9] v-prediction helpers
    # ---------------------------------------------------------------

    def _compute_v_target(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """v = sqrt(a_bar) * eps - sqrt(1 - a_bar) * x0"""
        sa = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)          # type: ignore[attr-defined]
        sm = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)  # type: ignore[attr-defined]
        return sa * noise - sm * x0

    def _v_to_eps_x0(
        self, v: torch.Tensor, x_t: torch.Tensor, t_val: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recover eps and x0 from v-prediction.

        v   = sqrt(a_bar) * eps - sqrt(1 - a_bar) * x0
        x_t = sqrt(a_bar) * x0  + sqrt(1 - a_bar) * eps

        eps = sqrt(a_bar) * v   + sqrt(1 - a_bar) * x_t
        x0  = sqrt(a_bar) * x_t - sqrt(1 - a_bar) * v
        """
        sa = self.sqrt_alphas_cumprod[t_val]                      # type: ignore[attr-defined]
        sm = self.sqrt_one_minus_alphas_cumprod[t_val]             # type: ignore[attr-defined]
        eps = sa * v + sm * x_t
        x0  = sa * x_t - sm * v
        return eps, x0

    # ---------------------------------------------------------------
    # Training forward
    # ---------------------------------------------------------------

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ts   = batch["ts"].float()
        B    = ts.shape[0]
        t    = torch.randint(0, self.num_steps, (B,), device=self.device)
        noise = torch.randn_like(ts)
        x_t  = self._q_sample(ts, t, noise)
        cond  = self._encode_condition(batch)

        # [V2-9] Compute prediction target
        if self.prediction_type == "v":
            target = self._compute_v_target(ts, noise, t)
        else:
            target = noise

        # [V2-11] Self-conditioning: 50% of the time, pass the model's own x0 estimate
        x0_self_cond = None
        if self.net.self_cond and self.training:
            if torch.rand(1).item() > 0.5:
                with torch.no_grad():
                    pred_first = self.net(x_t, t, cond, x0_self_cond=None)
                    if self.prediction_type == "v":
                        # Recover x0 from v for each sample in batch
                        sa = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)    # type: ignore[attr-defined]
                        sm = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)  # type: ignore[attr-defined]
                        x0_self_cond = (sa * x_t - sm * pred_first).detach()
                    else:
                        sa = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)    # type: ignore[attr-defined]
                        sm = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)  # type: ignore[attr-defined]
                        x0_self_cond = ((x_t - sm * pred_first) / sa.clamp(min=1e-8)).detach()

        pred  = self.net(x_t, t, cond, x0_self_cond=x0_self_cond)
        loss  = F.mse_loss(pred, target)

        # [V2-10] Spectral auxiliary loss
        if self.spectral_loss_weight > 0.0:
            loss = loss + self.spectral_loss_weight * _spectral_loss(pred, target)

        return {"loss": loss}

    # ---------------------------------------------------------------
    # DDIM inference with CFG
    # ---------------------------------------------------------------

    @torch.no_grad()
    def _ddim_step(
        self,
        x_t: torch.Tensor,
        t: int,
        t_prev: int,
        cond: torch.Tensor,
        x0_self_cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (x_{t-1}, x0_pred) -- the latter is used for self-conditioning."""
        B  = x_t.shape[0]
        tv = torch.full((B,), t, device=x_t.device, dtype=torch.long)

        pred = self.net(x_t, tv, cond, x0_self_cond=x0_self_cond)

        if self.use_cfg and self.guidance_scale > 1.0 and self.use_condition:
            null = self.null_condition.expand(B, -1).to(x_t.device)
            pred_u = self.net(x_t, tv, null, x0_self_cond=x0_self_cond)
            pred   = pred_u + self.guidance_scale * (pred - pred_u)

        # [V2-9] Convert prediction to eps and x0
        if self.prediction_type == "v":
            eps, x0_pred = self._v_to_eps_x0(pred, x_t, t)
        else:
            eps = pred
            a_t = self.alphas_cumprod[t]                           # type: ignore[attr-defined]
            x0_pred = (x_t - (1.0 - a_t).sqrt() * eps) / a_t.sqrt()

        a_t    = self.alphas_cumprod[t]                            # type: ignore[attr-defined]
        a_prev = (
            self.alphas_cumprod_prev[t]                            # type: ignore[attr-defined]
            if t_prev < 0
            else self.alphas_cumprod[t_prev]                       # type: ignore[attr-defined]
        )
        x_prev = a_prev.sqrt() * x0_pred + (1.0 - a_prev).sqrt() * eps
        return x_prev, x0_pred

    @torch.no_grad()
    def generate(
        self,
        condition: torch.Tensor,
        n_samples: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """
        Returns [B, n_samples, L, C]  (evaluator normalises both orderings).
        """
        device = condition.device
        B      = condition.shape[0]
        L, C   = self.config.data.seq_length, self.config.data.n_var

        steps = int(kwargs.get("steps", self.sample_steps))

        # Encode condition once
        # [V2-14] Text+Attr fusion: when both enabled, condition is cap_emb
        # and attrs comes via kwargs
        attrs = kwargs.get("attrs", None)
        if (self._cond_fusion is not None
                and self._fusion_proj is not None
                and self._attr_encoder is not None
                and attrs is not None
                and condition.is_floating_point()):
            text_cond = condition.float().to(device)
            attr_cond = self._attr_encoder(attrs.to(device))
            cond = self._fusion_proj(torch.cat([text_cond, attr_cond], dim=-1))
        elif self._attr_enabled and self._attr_encoder is not None and not condition.is_floating_point():
            cond = self._attr_encoder(condition.to(device))
        else:
            cond = condition.float().to(device)
            if self._text_enabled and cond.shape[-1] != self.null_condition.shape[-1]:
                cond = cond  # raw cap_emb
            elif not self._text_enabled:
                cond = self.null_condition.expand(B, -1).to(device)

        cond_rep = cond.unsqueeze(1).repeat(1, n_samples, 1).reshape(B * n_samples, -1)
        x        = torch.randn(B * n_samples, L, C, device=device)

        schedule = torch.linspace(
            self.num_steps - 1, 0, steps, device=device
        ).long().tolist()

        # [V2-11] Self-conditioning: carry x0_pred across steps
        x0_self_cond = None

        for i, t in enumerate(schedule):
            t_prev = -1 if i == len(schedule) - 1 else schedule[i + 1]
            x, x0_pred = self._ddim_step(x, t, t_prev, cond_rep, x0_self_cond=x0_self_cond)
            # Update self-conditioning estimate for next step
            if self.net.self_cond:
                x0_self_cond = x0_pred

        return x.view(B, n_samples, L, C)
