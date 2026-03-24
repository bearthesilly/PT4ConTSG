"""
PT4ConTSG: Probabilistic Transformer for Conditional Time Series Generation.

Adapts the ST-PT (Spatio-Temporal Probabilistic Transformer) from forecasting
to conditional DDPM-based generation via the following key modifications:

  1. No instance normalization  — caller/dataset handles normalization;
     diffusion SNR is preserved across all timesteps.
  2. Timestep embedding        — sinusoidal + MLP, injected additively into
     every patch's unary potential so the denoiser is noise-level-aware.
  3. Multi-modal condition enc  — unified text / attribute / label encoder
     that produces a single [B, dim_z] vector injected into unary potentials,
     persisting across all MFVI iterations as a per-patch prior energy.
  4. Compositional attr topics  — per-attribute independent binary factor
     modules; each attribute's message propagates through its own isolated
     pathway during MFVI, enabling compositional generalization to novel
     attribute combinations at test time (Design Item 5 of the plan).
  5. Per-patch decoder          — symmetric inverse of the unary MLP;
     reconstructs the full time-series sequence from Z variable outputs.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from contsg.models.base import BaseGeneratorModule, DiffusionMixin
from contsg.registry import Registry


# =============================================================================
# Utilities: RoPE (self-contained, no transformers dependency)
# =============================================================================

def _rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class _RotaryEmbedding(nn.Module):
    """Minimal rotary positional embedding.

    Returns (cos, sin) of shape [1, seq_len, head_dim] — same convention
    as LlamaRotaryEmbedding used in the original PT_forecast_v15.py.
    """

    def __init__(self, head_dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: Tensor, position_ids: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x:            [B, S, D] — used for device/dtype only.
            position_ids: [1, S]    — position indices.
        Returns:
            cos, sin: [1, S, head_dim] float tensors.
        """
        inv_freq = self.inv_freq.to(x.device)  # type: ignore[attr-defined]
        freqs = torch.einsum(
            "i,j->ij",
            position_ids.float().squeeze(0),
            inv_freq,
        )                                          # [S, head_dim//2]
        emb = torch.cat([freqs, freqs], dim=-1)    # [S, head_dim]
        return (
            emb.cos().unsqueeze(0).to(x.dtype),
            emb.sin().unsqueeze(0).to(x.dtype),
        )


class _RopeApplier:
    """Applies (or un-applies) rotary embedding to [B, H, S, D] tensors."""

    def __init__(self, cos: Tensor, sin: Tensor, unsqueeze_dim: int = 1) -> None:
        # cos/sin arrive as [1, S, D]; unsqueeze to [1, 1, S, D] for broadcasting.
        self.cos = cos.unsqueeze(unsqueeze_dim)
        self.sin = sin.unsqueeze(unsqueeze_dim)

    def apply(self, x: Tensor) -> Tensor:
        return x * self.cos + _rotate_half(x) * self.sin

    def apply_o(self, x: Tensor) -> Tensor:
        return x * self.cos - _rotate_half(x) * self.sin


# =============================================================================
# Potential functions  (unchanged from PT_forecast_v15.py)
# =============================================================================

class _SquaredSoftmax(nn.Module):
    def __init__(self, dim: int = -1, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim, self.eps = dim, eps

    def forward(self, x: Tensor) -> Tensor:
        x = x.to(torch.float32).pow(2)
        return F.normalize(x, p=1, dim=self.dim, eps=self.eps).to(x.dtype)


class _AbsNormalization(nn.Module):
    def __init__(self, dim: int = -1, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim, self.eps = dim, eps

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(x.to(torch.float32))
        return F.normalize(x, p=1, dim=self.dim, eps=self.eps).to(x.dtype)


# =============================================================================
# PT Factor Graph — Head Selection  (ternary factors T^time and T^channel)
# Adapted from PT_forecast_v15.py; self-contained, no external PT imports.
# =============================================================================

class _PtHeadSelection(nn.Module):
    """Spatio-temporal multi-channel head selection via MFVI.

    Computes messages F (→ H variables) and G (→ Z variables) for both the
    temporal and channel dimensions simultaneously.  A single joint softmax
    normalises the H distribution over the combined time+channel candidate set,
    so the two dimensions compete for "attention budget" — the same design as
    PT_forecast_v15.py.
    """

    def __init__(
        self,
        dim_z: int,
        n_heads: int,
        ternary_init_std: float = 0.2,
        ternary_scaling: float = 1.0,
        dropout_h: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim_z = dim_z
        self.n_heads = n_heads
        self.ternary_rank = dim_z // n_heads
        self.regularize_h = 1.0 / dim_z
        self.ternary_scaling = ternary_scaling
        self.dropout = nn.Dropout(dropout_h)

        size = n_heads * self.ternary_rank
        self.u_time = nn.Parameter(torch.empty(size, dim_z))
        self.v_time = nn.Parameter(torch.empty(size, dim_z))
        self.u_chan = nn.Parameter(torch.empty(size, dim_z))
        self.v_chan = nn.Parameter(torch.empty(size, dim_z))

        head_dim = self.ternary_rank
        self.rope_time = _RotaryEmbedding(head_dim)
        self.rope_chan = _RotaryEmbedding(head_dim)

        for p in [self.u_time, self.v_time, self.u_chan, self.v_chan]:
            nn.init.normal_(p, std=ternary_init_std)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_F(
        self,
        qz: Tensor,
        dep_mask: Optional[Tensor],
        u: nn.Parameter,
        v: nn.Parameter,
        rope: _RotaryEmbedding,
    ) -> Tuple[Tensor, Tensor, Tensor, _RopeApplier, int, int]:
        """
        Compute message F (toward H) from Z marginals.

        Returns:
            F_mat:   [B, H, S, S]  — attention logits
            qu_o:    [B, H, S, R]  — RoPE-inverse-applied Q (for G)
            qv_rot:  [B, H, S, R]  — RoPE-applied V (for G)
            applier: _RopeApplier  — to be reused in _compute_G
            B, S:    ints
        """
        B, S, _ = qz.shape
        pos = torch.arange(S, device=qz.device).unsqueeze(0)  # [1, S]
        cos, sin = rope(qz, pos)
        applier = _RopeApplier(cos, sin)

        qu = F.linear(qz, u) * self.ternary_scaling            # [B, S, H*R]
        qv = F.linear(qz, v) * self.ternary_scaling

        qu = qu.view(B, S, self.n_heads, self.ternary_rank).transpose(1, 2)  # [B, H, S, R]
        qv = qv.view(B, S, self.n_heads, self.ternary_rank).transpose(1, 2)

        qu_rot = applier.apply(qu)
        qu_o   = applier.apply_o(qu)  # inverse RoPE → used in G
        qv_rot = applier.apply(qv)

        F_mat = torch.matmul(qu_rot, qv_rot.transpose(2, 3))  # [B, H, S, S]
        if dep_mask is not None:
            F_mat = F_mat + dep_mask
        return F_mat, qu_o, qv_rot, applier, B, S

    def _compute_G(
        self,
        qh: Tensor,
        qu_o: Tensor,
        qv_rot: Tensor,
        applier: _RopeApplier,
        u: nn.Parameter,
        v: nn.Parameter,
        B: int,
        S: int,
    ) -> Tensor:
        """
        Compute message G (toward Z) from H marginals.

        Args:
            qh:    [B, H, S, S]  — H posterior
            qu_o:  [B, H, S, R]  — from _compute_F
            qv_rot:[B, H, S, R]  — from _compute_F
            applier: _RopeApplier — reused from _compute_F
        Returns:
            [B, S, dim_z]
        """
        g1 = torch.matmul(qh, qv_rot)               # [B, H, S, R]
        g2 = torch.matmul(qh.transpose(2, 3), qu_o) # [B, H, S, R]

        g1 = applier.apply_o(g1)
        g2 = applier.apply(g2)

        g1 = g1.transpose(1, 2).reshape(B, S, self.n_heads * self.ternary_rank)
        g2 = g2.transpose(1, 2).reshape(B, S, self.n_heads * self.ternary_rank)
        return (g1 @ u + g2 @ v) * self.ternary_scaling  # [B, S, dim_z]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        qz: Tensor,
        dep_mask_time: Optional[Tensor],
        dep_mask_chan: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            qz:            [B, C, P, dim_z]
            dep_mask_time: [B*C, 1, P, P]  or None
            dep_mask_chan: [B*P, 1, C, C]  or None
        Returns:
            m_time, m_chan: [B, C, P, dim_z] each
        """
        B, C, P, D = qz.shape

        # ── Temporal messages (each channel treated independently) ──
        qz_t = qz.reshape(B * C, P, D)
        F_t, qu_o_t, qv_t, app_t, _, _ = self._compute_F(
            qz_t, dep_mask_time, self.u_time, self.v_time, self.rope_time
        )

        # ── Channel messages (each time-step patch treated independently) ──
        qz_c = qz.transpose(1, 2).reshape(B * P, C, D)
        F_c, qu_o_c, qv_c, app_c, _, _ = self._compute_F(
            qz_c, dep_mask_chan, self.u_chan, self.v_chan, self.rope_chan
        )

        # ── Joint H normalisation over time + channel candidates ──
        # F_t: [B*C, H, P, P] → [B, C, H, P, P] → [B, H, C, P, P]
        F_t_r = F_t.view(B, C, self.n_heads, P, P).permute(0, 2, 1, 3, 4)
        # F_c: [B*P, H, C, C] → [B, P, H, C, C] → [B, H, C, P, C]
        F_c_r = F_c.view(B, P, self.n_heads, C, C).permute(0, 2, 3, 1, 4)

        logits = torch.cat([F_t_r, F_c_r], dim=-1)          # [B, H, C, P, P+C]
        qh_joint = F.softmax(
            logits / self.regularize_h, dim=-1, dtype=torch.float32
        )
        qh_t_part, qh_c_part = qh_joint.split([P, C], dim=-1)

        # ── Back to per-dimension format ──
        # time: [B, H, C, P, P] → [B, C, H, P, P] → [B*C, H, P, P]
        qh_time = qh_t_part.permute(0, 2, 1, 3, 4).reshape(B * C, self.n_heads, P, P)
        qh_time = qh_time.to(qz.dtype)
        # channel: [B, H, C, P, C] → [B, P, H, C, C] → [B*P, H, C, C]
        qh_chan = qh_c_part.permute(0, 3, 1, 2, 4).reshape(B * P, self.n_heads, C, C)
        qh_chan = qh_chan.to(qz.dtype)

        m_time = self._compute_G(qh_time, qu_o_t, qv_t, app_t,
                                  self.u_time, self.v_time, B * C, P)
        m_chan = self._compute_G(qh_chan, qu_o_c, qv_c, app_c,
                                  self.u_chan, self.v_chan, B * P, C)

        # Reshape back to [B, C, P, D]
        m_time = m_time.view(B, C, P, D)
        m_chan  = m_chan.view(B, P, C, D).transpose(1, 2)
        return m_time, m_chan


# =============================================================================
# PT Factor Graph — Topic Modeling  (binary factor / G variables)
# =============================================================================

class _PtTopicModeling(nn.Module):
    """Global bottleneck binary factor — equivalent to the PT FFN.

    Retained for text/label conditioning where condition is injected
    globally via unary potentials rather than per-attribute pathways.
    """

    def __init__(
        self,
        dim_z: int,
        dim_g: int,
        init_std: float = 0.2,
        scaling: float = 1.0,
        regularize_g: float = 1.0,
    ) -> None:
        super().__init__()
        self.scaling = scaling
        self.regularize_g = regularize_g
        self.act = _AbsNormalization(dim=-1)
        self.W = nn.Parameter(torch.empty(dim_g, dim_z))
        nn.init.normal_(self.W, std=init_std)

    def forward(self, qz: Tensor) -> Tensor:
        qg = F.linear(qz, self.W) * self.scaling        # [B, C, P, dim_g]
        qg = self.act(qg / self.regularize_g)
        return qg @ self.W * self.scaling                # [B, C, P, dim_z]


# =============================================================================
# New — Per-Attribute Compositional Binary Factor  (Design Item 5)
# =============================================================================

class _AttributeTopicModule(nn.Module):
    """Per-attribute independent binary factor for compositional conditioning.

    Each discrete attribute field gets its own independent binary factor.
    The factor matrix is formed as:
        W_eff = W_base + u(attr) ⊗ v(attr)     (rank-1 residual perturbation)

    where W_base is the shared base factor (learned) and u, v are small MLPs
    that map the attribute embedding to the perturbation vectors.

    Because each attribute's message propagates through its own isolated
    pathway and the perturbation is computed independently, novel attribute
    combinations at test time are handled compositionally: the model needs
    only to have seen each attribute *individually*, not their joint product.
    """

    def __init__(
        self,
        num_classes: int,
        attr_embed_dim: int,
        dim_z: int,
        dim_g: int,
        init_std: float = 0.2,
        scaling: float = 1.0,
        regularize_g: float = 1.0,
    ) -> None:
        super().__init__()
        self.scaling = scaling
        self.regularize_g = regularize_g
        self.act = _AbsNormalization(dim=-1)

        # Base factor (dataset-wide shared component)
        self.W_base = nn.Parameter(torch.empty(dim_g, dim_z))
        nn.init.normal_(self.W_base, std=init_std)

        # Attribute embedding + rank-1 residual projectors
        self.embed = nn.Embedding(num_classes, attr_embed_dim)
        self.to_u = nn.Linear(attr_embed_dim, dim_g, bias=False)
        self.to_v = nn.Linear(attr_embed_dim, dim_z, bias=False)

    def forward(self, qz: Tensor, attr_idx: Tensor) -> Tensor:
        """
        Args:
            qz:       [B, C, P, dim_z]
            attr_idx: [B] — integer attribute value for this field
        Returns:
            message:  [B, C, P, dim_z]
        """
        a_emb = self.embed(attr_idx.long())     # [B, attr_embed_dim]
        u = self.to_u(a_emb)                    # [B, dim_g]
        v = self.to_v(a_emb)                    # [B, dim_z]

        # W_eff[b] = W_base + u[b] ⊗ v[b] — [B, dim_g, dim_z]
        W = self.W_base.unsqueeze(0) + u.unsqueeze(2) * v.unsqueeze(1)

        # Forward pass with sample-specific W
        # qz: [B, C, P, dim_z]  →  einsum over d (dim_z)
        qg = torch.einsum("bcpd,bgd->bcpg", qz, W) * self.scaling  # [B, C, P, dim_g]
        qg = self.act(qg / self.regularize_g)
        return torch.einsum("bcpg,bgd->bcpd", qg, W) * self.scaling # [B, C, P, dim_z]


# =============================================================================
# MFVI Iterator  (one full message-passing round)
# =============================================================================

class _PtEncoderIterator(nn.Module):
    """One iteration of MFVI over the 2D spatio-temporal factor graph.

    Accepts an optional *external* condition message `m_cond` — the sum of
    all per-attribute topic module outputs — so Design Items 4 and 5 can
    both contribute to each Z update without altering the core MFVI math.
    """

    def __init__(
        self,
        dim_z: int,
        n_heads: int,
        dim_g: int,
        regularize_z: float = 1.0,
        **head_sel_kwargs: Any,
    ) -> None:
        super().__init__()
        self.regularize_z = regularize_z
        self.head_sel = _PtHeadSelection(dim_z, n_heads, **head_sel_kwargs)
        self.topic    = _PtTopicModeling(dim_z, dim_g)
        self.norm     = _SquaredSoftmax(dim=-1)

    def forward(
        self,
        unary: Tensor,
        qz: Tensor,
        dep_mask_time: Optional[Tensor],
        dep_mask_chan: Optional[Tensor],
        m_cond: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            unary:         [B, C, P, dim_z] — persistent prior (content+cond+t)
            qz:            [B, C, P, dim_z] — current Z marginals
            dep_mask_time: [B*C, 1, P, P]
            dep_mask_chan: [B*P, 1, C, C]
            m_cond:        [B, C, P, dim_z] optional attribute topic messages
        Returns:
            qz_new: [B, C, P, dim_z]
        """
        old_qz = qz
        qz_norm = self.norm(qz)

        m_t, m_c = self.head_sel(qz_norm, dep_mask_time, dep_mask_chan)
        m_g = self.topic(qz_norm)

        total = m_t + m_c + m_g + unary
        if m_cond is not None:
            total = total + m_cond

        qz_new = total / self.regularize_z
        return (qz_new + old_qz) * 0.5     # damping (residual connection)


# =============================================================================
# New — Timestep Embedding  (Design Item 2)
# =============================================================================

class _TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding + two-layer MLP → [B, dim_z]."""

    def __init__(self, dim_z: int, freq_dim: int = 256) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, dim_z * 2),
            nn.SiLU(),
            nn.Linear(dim_z * 2, dim_z),
        )

    def forward(self, t: Tensor) -> Tensor:
        """t: [B] integer timesteps → [B, dim_z]"""
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / (half - 1)
        )                                                    # [half]
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # [B, half]
        emb  = torch.cat([args.cos(), args.sin()], dim=-1)  # [B, freq_dim]
        return self.mlp(emb)


# =============================================================================
# New — Multi-Modal Condition Encoder  (Design Item 3)
# =============================================================================

class _ConditionEncoder(nn.Module):
    """Unified encoder for text / attribute (global) / label conditions.

    Each modality has an independent encoder branch.  All branches produce
    a single [B, dim_z] vector.  A learnable null embedding supports
    classifier-free guidance (CFG) training.

    Condition dtype dispatch:
      - floating-point → text branch (cap_emb [B, D])
      - integer        → attribute branch ([B, A]) if attr_enabled,
                         else label branch ([B])
    """

    def __init__(
        self,
        dim_z: int,
        text_input_dim: int = 0,
        attr_embed_dim: int = 16,
        attr_discrete_configs: Optional[List[Dict]] = None,
        label_num_classes: int = 0,
        cond_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cond_dropout   = cond_dropout
        self.text_enabled   = text_input_dim > 0
        self.attr_enabled   = bool(attr_discrete_configs)
        self.label_enabled  = label_num_classes > 0

        if self.text_enabled:
            self.text_proj = nn.Sequential(
                nn.Linear(text_input_dim, dim_z * 2),
                nn.SiLU(),
                nn.Linear(dim_z * 2, dim_z),
            )

        if self.attr_enabled and attr_discrete_configs:
            self.attr_embeds = nn.ModuleList([
                nn.Embedding(cfg["num_classes"], attr_embed_dim)
                for cfg in attr_discrete_configs
            ])
            total_dim = len(attr_discrete_configs) * attr_embed_dim
            self.attr_proj = nn.Sequential(
                nn.Linear(total_dim, dim_z),
                nn.SiLU(),
                nn.Linear(dim_z, dim_z),
            )

        if self.label_enabled:
            self.label_emb = nn.Embedding(label_num_classes, dim_z)

        # Learnable null embedding for classifier-free guidance
        self.null_emb = nn.Parameter(torch.zeros(dim_z))

    def forward(
        self,
        condition: Tensor,
        use_cond: bool = True,
        force_null: bool = False,
    ) -> Tensor:
        """
        Args:
            condition:  [B, D] float (text) | [B, A] long (attr) | [B] long (label)
            use_cond:   If False, return null embedding.
            force_null: If True, return null embedding (used at CFG inference).
        Returns:
            c_emb: [B, dim_z]
        """
        B = condition.shape[0]
        device = condition.device

        if not use_cond or force_null:
            return self.null_emb.unsqueeze(0).expand(B, -1).to(device)

        if self.text_enabled and condition.is_floating_point():
            c = self.text_proj(condition.float())
        elif self.attr_enabled and not condition.is_floating_point():
            if condition.dim() == 1:
                condition = condition.unsqueeze(1)
            parts = [
                self.attr_embeds[k](condition[:, k].long())   # type: ignore[index]
                for k in range(len(self.attr_embeds))         # type: ignore[arg-type]
            ]
            c = self.attr_proj(torch.cat(parts, dim=-1))
        elif self.label_enabled:
            c = self.label_emb(condition.long().squeeze(-1))
        else:
            return self.null_emb.unsqueeze(0).expand(B, -1).to(device)

        # Classifier-free guidance: randomly drop condition during training
        if self.training and self.cond_dropout > 0.0:
            drop_mask = torch.bernoulli(
                torch.full((B,), self.cond_dropout, device=device)
            ).bool()
            null = self.null_emb.unsqueeze(0).expand(B, -1)
            c = torch.where(drop_mask.unsqueeze(1), null, c)

        return c


# =============================================================================
# PT Denoiser  (full backbone)
# =============================================================================

class _PtDenoiser(nn.Module):
    """ST-PT factor graph assembled as a DDPM denoising backbone.

    I/O contract:
        Input:  x_t  [B, L, C] — noisy time series (dataset-normalized scale)
                t_emb [B, dim_z] — diffusion timestep embedding
                c_emb [B, dim_z] — condition embedding
                attrs [B, A] long  — optional, for per-attribute topic modules
        Output: noise_pred [B, L, C] — predicted Gaussian noise
    """

    def __init__(
        self,
        n_var: int,
        seq_len: int,
        dim_z: int,
        dim_g: int,
        n_heads: int,
        n_iter: int,
        patch_len: int,
        attr_discrete_configs: Optional[List[Dict]] = None,
        attr_embed_dim: int = 16,
        regularize_z: float = 1.0,
        ternary_init_std: float = 0.2,
        ternary_scaling: float = 1.0,
    ) -> None:
        super().__init__()
        assert seq_len % patch_len == 0, (
            f"seq_len ({seq_len}) must be divisible by patch_len ({patch_len})"
        )
        self.n_var      = n_var
        self.seq_len    = seq_len
        self.patch_len  = patch_len
        self.n_patches  = seq_len // patch_len

        # Unary factor: patch_len → dim_z  (Design Item 1: no instance norm here)
        self.unary_mlp = nn.Sequential(
            nn.Linear(patch_len, dim_z),
            nn.GELU(),
            nn.Linear(dim_z, dim_z),
        )

        # MFVI iterations
        self.iters = nn.ModuleList([
            _PtEncoderIterator(
                dim_z=dim_z,
                n_heads=n_heads,
                dim_g=dim_g,
                regularize_z=regularize_z,
                ternary_init_std=ternary_init_std,
                ternary_scaling=ternary_scaling,
            )
            for _ in range(n_iter)
        ])

        # Per-attribute compositional topic modules (Design Item 5)
        self.attr_topics: Optional[nn.ModuleList] = None
        if attr_discrete_configs:
            self.attr_topics = nn.ModuleList([
                _AttributeTopicModule(
                    num_classes=cfg["num_classes"],
                    attr_embed_dim=attr_embed_dim,
                    dim_z=dim_z,
                    dim_g=dim_g,
                )
                for cfg in attr_discrete_configs
            ])

        # Per-patch reconstruction decoder (Design Item 6)
        self.decoder_mlp = nn.Sequential(
            nn.Linear(dim_z, dim_z),
            nn.GELU(),
            nn.Linear(dim_z, patch_len),
        )

    @staticmethod
    def _make_dep_mask(B: int, S: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """4D attention mask with -inf on the diagonal (no self-head)."""
        mask = torch.zeros(B, 1, S, S, device=device, dtype=dtype)
        diag = torch.eye(S, device=device, dtype=torch.bool)
        mask.masked_fill_(diag.unsqueeze(0).unsqueeze(0), float("-inf"))
        return mask

    def forward(
        self,
        x_t: Tensor,
        t_emb: Tensor,
        c_emb: Tensor,
        attrs: Optional[Tensor] = None,
    ) -> Tensor:
        B, L, C = x_t.shape
        P = self.n_patches

        # Patch: [B, L, C] → [B, C, P, patch_len]
        x = x_t.transpose(1, 2).reshape(B, C, P, self.patch_len)

        # Unary potentials + condition + timestep injection (Design Items 2+3+4)
        unary = self.unary_mlp(x)                              # [B, C, P, dim_z]
        unary = unary + c_emb.reshape(B, 1, 1, -1) \
                      + t_emb.reshape(B, 1, 1, -1)

        # Build dependency masks (no self-loop)
        dep_mask_time = self._make_dep_mask(B * C, P, x_t.device, unary.dtype)
        dep_mask_chan = self._make_dep_mask(B * P, C, x_t.device, unary.dtype)

        # Per-attribute compositional topic messages (Design Item 5)
        m_cond: Optional[Tensor] = None
        if self.attr_topics is not None and attrs is not None:
            if attrs.dim() == 1:
                attrs = attrs.unsqueeze(1)
            m_cond = sum(
                self.attr_topics[k](unary, attrs[:, k])   # type: ignore[index]
                for k in range(len(self.attr_topics))     # type: ignore[arg-type]
            )

        # MFVI message-passing loop
        qz = unary
        for it in self.iters:
            qz = it(unary, qz, dep_mask_time, dep_mask_chan, m_cond)

        # Per-patch decode → [B, L, C]  (Design Item 6)
        out = self.decoder_mlp(qz)              # [B, C, P, patch_len]
        out = out.reshape(B, C, L).transpose(1, 2)
        return out


# =============================================================================
# Lightning Module
# =============================================================================

@Registry.register_model("pt4contsg", aliases=["pt_gen"])
class PT4ConTSGModule(BaseGeneratorModule, DiffusionMixin):
    """PT4ConTSG: Probabilistic Transformer for Conditional Time Series Generation.

    ST-PT backbone adapted as a DDPM denoiser with structured condition
    injection and per-attribute compositional binary factors.

    Config parameters (model.*):
        dim_z (int):           Z-variable dimension.         Default: 64
        dim_g (int):           G-variable (binary) dimension. Default: dim_z*2
        n_heads (int):         Ternary factor channels.       Default: 8
        n_iter (int):          MFVI iterations.               Default: 4
        patch_len (int):       Patch size. seq_len % patch_len == 0.  Default: 8
        attr_embed_dim (int):  Per-attribute embedding dim.   Default: 16
        diffusion_steps (int): DDPM timesteps.                Default: 50
        noise_schedule (str):  "quad" | "cosine" | "linear". Default: "quad"
        beta_start (float):    Used for quad/linear schedule.  Default: 0.0001
        beta_end (float):      Used for quad/linear schedule.  Default: 0.5

    Config parameters (condition.text.*):
        cfg_scale (float):     Classifier-free guidance scale at inference.
    """

    SUPPORTED_STAGES = ["finetune"]

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_model(self) -> None:
        cfg      = self.config.model
        data_cfg = self.config.data
        cond_cfg = self.config.condition

        # ── Hyperparameters ──
        dim_z          = int(getattr(cfg, "dim_z", 64))
        dim_g          = int(getattr(cfg, "dim_g", dim_z * 2))
        n_heads        = int(getattr(cfg, "n_heads", 8))
        n_iter         = int(getattr(cfg, "n_iter", 4))
        patch_len      = int(getattr(cfg, "patch_len", 8))
        attr_embed_dim = int(getattr(cfg, "attr_embed_dim", 16))
        self.cfg_scale = float(getattr(cond_cfg.text, "cfg_scale", 1.0))

        # ── Noise schedule ──
        self.num_steps    = int(getattr(cfg, "diffusion_steps", 50))
        noise_schedule    = str(getattr(cfg, "noise_schedule", "quad"))
        beta_start        = float(getattr(cfg, "beta_start", 0.0001))
        beta_end          = float(getattr(cfg, "beta_end", 0.5))

        if noise_schedule == "quad":
            betas = self.quad_beta_schedule(self.num_steps, beta_start, beta_end)
        elif noise_schedule == "cosine":
            betas = self.cosine_beta_schedule(self.num_steps)
        else:  # linear
            betas = self.linear_beta_schedule(self.num_steps, beta_start, beta_end)

        alphas    = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alpha_bar)
        self.register_buffer("sqrt_alphas_cumprod",         alpha_bar.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alpha_bar).sqrt())

        # ── Condition configuration ──
        text_input_dim = cond_cfg.text.input_dim   if cond_cfg.text.enabled   else 0
        attr_configs   = cond_cfg.attribute.discrete_configs \
                         if cond_cfg.attribute.enabled else []
        label_classes  = cond_cfg.label.num_classes if cond_cfg.label.enabled else 0
        cond_dropout   = cond_cfg.condition_dropout

        # ── Modules ──
        self.t_embed = _TimestepEmbedding(dim_z)

        self.cond_encoder = _ConditionEncoder(
            dim_z=dim_z,
            text_input_dim=text_input_dim,
            attr_embed_dim=attr_embed_dim,
            attr_discrete_configs=attr_configs or None,
            label_num_classes=label_classes,
            cond_dropout=cond_dropout,
        )

        self.denoiser = _PtDenoiser(
            n_var=data_cfg.n_var,
            seq_len=data_cfg.seq_length,
            dim_z=dim_z,
            dim_g=dim_g,
            n_heads=n_heads,
            n_iter=n_iter,
            patch_len=patch_len,
            attr_discrete_configs=attr_configs or None,
            attr_embed_dim=attr_embed_dim,
        )

    # ------------------------------------------------------------------
    # Condition helper
    # ------------------------------------------------------------------

    def _encode_condition(
        self,
        batch: Dict[str, Tensor],
        force_null: bool = False,
    ) -> Tensor:
        cond_cfg = self.config.condition
        B = batch["ts"].shape[0]

        if not self.use_condition or force_null:
            return self.cond_encoder.null_emb.unsqueeze(0).expand(B, -1)

        if cond_cfg.text.enabled and "cap_emb" in batch:
            cond = batch["cap_emb"].float()
        elif cond_cfg.attribute.enabled and "attrs" in batch:
            cond = batch["attrs"]
        elif cond_cfg.label.enabled:
            cond = batch.get("label", batch.get("labels"))
            if cond is None:
                return self.cond_encoder.null_emb.unsqueeze(0).expand(B, -1)
        else:
            return self.cond_encoder.null_emb.unsqueeze(0).expand(B, -1)

        return self.cond_encoder(cond, use_cond=True)

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Training step: sample t, add noise, predict noise (ε-parameterisation)."""
        ts = batch["ts"].float()    # [B, L, C], dataset-normalised
        B  = ts.shape[0]

        t = torch.randint(0, self.num_steps, (B,), device=self.device)

        noise = torch.randn_like(ts)
        x_t, _ = self.q_sample(
            ts, t,
            noise=noise,
            sqrt_alphas_cumprod=self.sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod,
        )

        t_emb  = self.t_embed(t)
        c_emb  = self._encode_condition(batch)
        attrs  = batch.get("attrs") if self.config.condition.attribute.enabled else None

        noise_pred = self.denoiser(x_t, t_emb, c_emb, attrs=attrs)
        loss = F.mse_loss(noise_pred, noise)
        return {"loss": loss}

    # ------------------------------------------------------------------
    # Generation (DDIM / DDPM reverse process)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        condition: Tensor,
        n_samples: int = 1,
        sampler: str = "ddim",
        attrs: Optional[Tensor] = None,
        ddim_steps: int = 50,
        **kwargs: Any,
    ) -> Tensor:
        """Generate samples via reverse diffusion.

        Args:
            condition:  [B, D] float (text) | [B, A] long (attr) | [B] long (label)
            n_samples:  Samples to generate per condition.
            sampler:    "ddim" (fast) or "ddpm" (stochastic).
            attrs:      [B, A] long — passed separately for compositional topic
                        modules when text conditioning is the primary modality.
            ddim_steps: DDIM sub-step count (ignored for ddpm).
        Returns:
            [n_samples, B, L, C]
        """
        data_cfg = self.config.data
        B = condition.shape[0]
        L, C = data_cfg.seq_length, data_cfg.n_var
        device = self.device

        # Encode condition once; reuse across all n_samples
        c_emb  = self.cond_encoder(condition, use_cond=self.use_condition)
        c_null = self.cond_encoder.null_emb.unsqueeze(0).expand(B, -1).to(device)

        # Attribute tensor: use `attrs` kwarg if given, else `condition` itself
        # when attribute conditioning is the primary modality
        if attrs is None and not condition.is_floating_point() and condition.dim() > 1:
            attrs = condition

        # Build reverse timestep sequence
        T = self.num_steps
        if sampler == "ddim":
            step_ratio = max(1, T // ddim_steps)
            timesteps  = list(reversed(range(0, T, step_ratio)))
        else:
            timesteps = list(reversed(range(T)))

        all_samples = []
        for _ in range(n_samples):
            x = torch.randn(B, L, C, device=device)

            for idx, t_val in enumerate(timesteps):
                t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)
                t_emb   = self.t_embed(t_batch)

                eps = self.denoiser(x, t_emb, c_emb, attrs=attrs)

                # Classifier-free guidance
                if self.cfg_scale != 1.0 and self.use_condition:
                    eps_u = self.denoiser(x, t_emb, c_null, attrs=None)
                    eps   = eps_u + self.cfg_scale * (eps - eps_u)

                # Predict x0 from ε
                alpha_bar_t = self.alphas_cumprod[t_val]
                x0_pred = (x - (1.0 - alpha_bar_t).sqrt() * eps) / alpha_bar_t.sqrt()
                x0_pred = x0_pred.clamp(-10.0, 10.0)

                if idx == len(timesteps) - 1:
                    x = x0_pred
                else:
                    t_prev       = timesteps[idx + 1]
                    alpha_bar_p  = self.alphas_cumprod[t_prev]

                    if sampler == "ddim":
                        x = alpha_bar_p.sqrt() * x0_pred + (1.0 - alpha_bar_p).sqrt() * eps
                    else:  # DDPM stochastic
                        beta_t  = self.betas[t_val]
                        alpha_t = self.alphas[t_val]
                        mean    = (1.0 / alpha_t.sqrt()) * (
                            x - beta_t / (1.0 - alpha_bar_t).sqrt() * eps
                        )
                        var = beta_t * (1.0 - alpha_bar_p) / (1.0 - alpha_bar_t)
                        x   = mean + var.clamp(min=1e-20).sqrt() * torch.randn_like(x)

            all_samples.append(x)

        return torch.stack(all_samples, dim=0)   # [n_samples, B, L, C]
