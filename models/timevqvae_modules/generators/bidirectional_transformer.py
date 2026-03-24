"""Bidirectional Transformer used by MaskGIT prior."""

from __future__ import annotations

import math
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from torch import Tensor
from torch.nn.utils import weight_norm


def calculate_padding(kernel_size, stride, dilation):
    effective_kernel_size = dilation * (kernel_size - 1) + 1
    padding = math.floor((effective_kernel_size - stride) / 2)
    return padding


class Upscale(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, h_dim: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            weight_norm(
                nn.Conv1d(
                    in_channels,
                    h_dim,
                    kernel_size=7,
                    stride=1,
                    dilation=1,
                    padding=calculate_padding(7, 1, 1),
                )
            ),
            nn.GELU(),
            nn.BatchNorm1d(h_dim),
            weight_norm(
                nn.Conv1d(
                    h_dim,
                    out_channels,
                    kernel_size=7,
                    stride=1,
                    dilation=2,
                    padding=calculate_padding(7, 1, 2),
                )
            ),
        )

    def forward(self, x, upscale_size: int):
        x = rearrange(x, "b n d -> b d n")
        x = F.interpolate(x, size=(upscale_size,), mode="nearest")
        x = self.conv(x)
        x = rearrange(x, "b d m -> b m d")
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        inner_dim = int(dim * mult * 2 / 3)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            weight_norm(nn.Linear(dim, inner_dim, bias=False)),
            nn.GELU(),
            weight_norm(nn.LayerNorm(inner_dim)),
            nn.Linear(inner_dim, dim, bias=False),
        )

    def forward(self, x):
        return self.net(x)


def exists(val):
    return val is not None


def l2norm(t):
    return F.normalize(t, dim=-1)


class DiffAttn(nn.Module):
    def __init__(self, embedding_dim: int = 128, nhead: int = 4, depth: int = 1, dropout: float = 0.2):
        super().__init__()
        self.nhead = nhead
        self.head_dim = embedding_dim // nhead
        self.scale = self.head_dim ** -0.5
        self.weight = weight_norm(nn.Linear(embedding_dim, embedding_dim * 3, bias=True))
        self.out_proj = weight_norm(nn.Linear(embedding_dim, embedding_dim, bias=True))
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.depth = depth

    def forward(self, x: Tensor) -> Tensor:
        b, n, d = x.shape
        qkv = self.weight(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.nhead), qkv)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        scores = scores.softmax(dim=-1)
        scores = self.dropout(scores)
        out = torch.matmul(scores, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.out_proj(out)
        out = self.layer_norm(out + x)
        return out


class MultiheadDiffAttn(nn.Module):
    def __init__(self, dim: int, depth: int, nhead: int = 4, dim_head: int = 32):
        super().__init__()
        self.nhead = nhead
        self.dim = dim
        self.dim_head = dim_head
        self.depth = depth
        self.scale = dim_head ** -0.5

        inner_dim = dim_head * nhead
        self.to_qkv = weight_norm(nn.Linear(dim, inner_dim * 3, bias=False))
        self.to_out = weight_norm(nn.Linear(inner_dim, dim, bias=False))
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.nhead), qkv)

        sim = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = sim.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        out = self.layer_norm(out + x)
        return out


class Attention(nn.Module):
    def __init__(self, dim: int, dim_head: int = 32, heads: int = 8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = weight_norm(nn.Linear(dim, hidden_dim * 3, bias=False))
        self.to_out = weight_norm(nn.Linear(hidden_dim, dim, bias=False))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        sim = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = sim.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class TransformerBlocks(nn.Module):
    def __init__(self, dim: int, depth: int = 8, heads: int = 8, dim_head: int = 64, ff_mult: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = x + attn(x)
            x = x + ff(x)
        return x


class BidirectionalTransformer(nn.Module):
    def __init__(
        self,
        kind: str,
        num_tokens: int,
        codebook_sizes: dict,
        embed_dim: int,
        hidden_dim: int,
        n_layers: int,
        heads: int,
        ff_mult: int,
        use_rmsnorm: bool,
        p_unconditional: float,
        n_classes: int,
        model_dropout: float = 0.3,
        emb_dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__()
        kind = kind.lower()
        assert kind in ["lf", "hf"], "invalid `kind`."
        self.kind = kind
        self.num_tokens = num_tokens
        self.n_classes = n_classes
        self.p_unconditional = p_unconditional
        in_dim = embed_dim if kind == "lf" else 2 * embed_dim
        self.emb_dropout = emb_dropout
        self.mask_token_ind = {"lf": codebook_sizes["lf"], "hf": codebook_sizes["hf"]}

        self.tok_emb_l = nn.Embedding(codebook_sizes["lf"] + 1, embed_dim)
        if kind == "hf":
            self.tok_emb_h = nn.Embedding(codebook_sizes["hf"] + 1, embed_dim)

        initial_pos_emb_size = max(self.num_tokens + 1, 64)
        self.pos_emb = nn.Embedding(initial_pos_emb_size, in_dim)
        self.class_condition_emb = nn.Embedding(n_classes + 1, in_dim)
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        self.blocks = TransformerBlocks(dim=hidden_dim, depth=n_layers, dim_head=64, heads=heads, ff_mult=ff_mult)
        codebook_size = codebook_sizes["lf"] if kind == "lf" else codebook_sizes["hf"]
        self.pred_head = nn.Sequential(
            weight_norm(nn.Linear(in_features=hidden_dim, out_features=hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(hidden_dim, eps=1e-12),
            weight_norm(nn.Linear(in_features=hidden_dim, out_features=codebook_size)),
        )

        if kind == "hf":
            self.projector = Upscale(embed_dim, embed_dim, 2 * embed_dim)

    def _maybe_expand_position_embeddings(self, required_length: int, device: torch.device):
        if required_length <= self.pos_emb.num_embeddings:
            return

        new_num_embeddings = max(required_length, self.pos_emb.num_embeddings * 2)
        new_pos_emb = nn.Embedding(
            new_num_embeddings,
            self.pos_emb.embedding_dim,
            device=device,
            dtype=self.pos_emb.weight.dtype,
        )
        with torch.no_grad():
            new_pos_emb.weight[: self.pos_emb.num_embeddings] = self.pos_emb.weight
            nn.init.normal_(new_pos_emb.weight[self.pos_emb.num_embeddings :], mean=0.0, std=0.02)

        self.pos_emb = new_pos_emb

    def class_embedding(self, class_condition: Union[None, torch.Tensor], batch_size: int, device):
        cond_type = "uncond" if isinstance(class_condition, type(None)) else "class-cond"

        if cond_type == "uncond":
            class_uncondition = repeat(
                torch.Tensor([self.n_classes]).long().to(device), "i -> b i", b=batch_size
            )
            cls_emb = self.class_condition_emb(class_uncondition)
            return cls_emb
        if self.training:
            ind = torch.rand(class_condition.shape).to(device) > self.p_unconditional
        else:
            ind = torch.ones_like(class_condition, dtype=torch.bool).to(device)
        class_condition = torch.where(ind, class_condition.long(), self.n_classes)
        cls_emb = self.class_condition_emb(class_condition)
        return cls_emb

    def _token_emb_dropout(self, s: torch.LongTensor, token_emb: torch.FloatTensor, freq_type: str, p: float):
        mask_ind = (s == self.mask_token_ind[freq_type])[:, :, None]
        token_emb_dropout = F.dropout(token_emb, p=p)
        token_emb = torch.where(mask_ind, token_emb, token_emb_dropout)
        return token_emb

    def forward_lf(self, s_M_l, class_condition: Union[None, torch.Tensor] = None):
        device = s_M_l.device

        token_embeddings = self.tok_emb_l(s_M_l)
        if self.training:
            token_embeddings = self._token_emb_dropout(s_M_l, token_embeddings, "lf", p=self.emb_dropout)

        cls_emb = self.class_embedding(class_condition, s_M_l.shape[0], device)

        n = token_embeddings.shape[1]
        self._maybe_expand_position_embeddings(n, device)
        position_embeddings = self.pos_emb.weight[:n, :]
        embed = token_embeddings + position_embeddings
        embed = torch.cat((cls_emb, embed), dim=1)
        embed = self.proj_in(embed)
        embed = self.blocks(embed)
        logits = self.pred_head(embed)[:, 1:, :]
        return logits

    def forward_hf(self, s_l, s_M_h, class_condition=None):
        device = s_l.device

        token_embeddings_l = self.tok_emb_l(s_l)
        token_embeddings_h = self.tok_emb_h(s_M_h)

        if self.training:
            token_embeddings_l = self._token_emb_dropout(s_l, token_embeddings_l, "lf", p=self.emb_dropout)
            token_embeddings_h = self._token_emb_dropout(s_M_h, token_embeddings_h, "hf", p=self.emb_dropout)

        token_embeddings_l = self.projector(token_embeddings_l, upscale_size=token_embeddings_h.shape[1])
        token_embeddings = torch.cat((token_embeddings_l, token_embeddings_h), dim=-1)

        cls_emb = self.class_embedding(class_condition, s_l.shape[0], device)

        n = token_embeddings.shape[1]
        self._maybe_expand_position_embeddings(n, device)
        position_embeddings = self.pos_emb.weight[:n, :]
        embed = token_embeddings + position_embeddings
        embed = torch.cat((cls_emb, embed), dim=1)
        embed = self.proj_in(embed)
        embed = self.blocks(embed)
        logits = self.pred_head(embed)[:, 1:, :]
        return logits

    def forward(self, s_M_l, s_M_h=None, class_condition: Union[None, torch.Tensor] = None):
        if self.kind == "lf":
            logits = self.forward_lf(s_M_l, class_condition)
        elif self.kind == "hf":
            logits = self.forward_hf(s_M_l, s_M_h, class_condition)
        else:
            raise ValueError
        return logits
