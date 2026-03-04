"""
CTTP: Contrastive Text-to-Time Series Pretraining.

This module implements the CTTP model for learning joint text-time series embeddings
through contrastive learning. CTTP is a **representation learning model**, NOT a
generative model.

Key Features:
- Instance Mode: Global TS vs Global Text alignment
- Segment Mode: Dual-task learning (Segment + Global alignment)
- Online Text Encoding: Uses HuggingFace CLIP/BERT models directly with raw text
- Pre-computed Embedding Mode: Uses pre-extracted embeddings for efficiency
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from contsg.models.modules.patchtst import (
    Encoder,
    EncoderLayer,
    AttentionLayer,
    FullAttention,
    PatchEmbedding,
    SegmentTSEncoder,
)
from contsg.registry import Registry


# =============================================================================
# Loss Functions
# =============================================================================

class ContrastiveLoss(nn.Module):
    """
    Margin-based contrastive loss.

    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    Args:
        margin: Margin for negative pairs
    """

    def __init__(self, margin: float = 3.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1: Tensor, emb2: Tensor, label: Tensor) -> Tensor:
        """
        Args:
            emb1, emb2: (B, D) embeddings
            label: (B,) - 0 for positive pairs, 1 for negative pairs
        Returns:
            Scalar loss
        """
        dist = F.pairwise_distance(emb1, emb2, keepdim=True)
        loss = torch.mean(
            (1 - label) * torch.pow(dist, 2)
            + label * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        )
        return loss


# =============================================================================
# Text Encoders
# =============================================================================

class HuggingFaceTextEncoder(nn.Module):
    """
    Text encoder using HuggingFace CLIP model with online tokenization.

    Supports both single-segment and multi-segment (Segment Level) modes.

    Args:
        pretrain_model_path: Path to HuggingFace model
        pretrain_model_dim: Dimension of pretrained model output
        textemb_hidden_dim: Hidden dimension for projection MLP
        coemb_dim: Output co-embedding dimension
        output_type: "cls" for CLS token, "all" for mean pooling
        max_length: Maximum token length (None = use model default)
    """

    def __init__(
        self,
        pretrain_model_path: str,
        pretrain_model_dim: int,
        textemb_hidden_dim: int,
        coemb_dim: int,
        output_type: str = "cls",
        max_length: int | None = None,
    ):
        super().__init__()
        self.output_type = output_type

        # Lazy import to avoid HuggingFace dependency at module level
        from transformers import AutoTokenizer, CLIPTextModelWithProjection, CLIPTextConfig

        # Handle LongCLIP with extended positions
        if "Longclip" in pretrain_model_path:
            clip_config = CLIPTextConfig.from_pretrained(pretrain_model_path)
            clip_config.max_position_embeddings = 248
            self.model = CLIPTextModelWithProjection.from_pretrained(
                pretrain_model_path, config=clip_config
            )
        else:
            self.model = CLIPTextModelWithProjection.from_pretrained(pretrain_model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_model_path)
        self.max_length = max_length or self.model.config.max_position_embeddings

        # Freeze pretrained model
        for param in self.model.parameters():
            param.requires_grad = False

        # Projection MLP
        self.text_enc = nn.Sequential(
            nn.Linear(pretrain_model_dim, textemb_hidden_dim),
            nn.LayerNorm(textemb_hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(textemb_hidden_dim, coemb_dim),
        )

    def forward(
        self,
        text: list[str] | list[list[str]],
        return_all_segments: bool = False,
    ) -> Tensor:
        """
        Encode text to embeddings.

        Args:
            text: List[str] for instance mode, List[List[str]] for segment mode
            return_all_segments: If True in segment mode, return (B, K, D)
        Returns:
            Embeddings (B, D) or (B, K, D) or (B*K, D)
        """
        device = next(self.model.parameters()).device

        # Check if multi-segment mode
        if isinstance(text[0], list):
            return self._encode_segments(text, return_all_segments, device)
        else:
            return self._encode_single(text, device)

    def _encode_single(self, text: list[str], device: torch.device) -> Tensor:
        """Encode single-segment text."""
        inputs = self.tokenizer(
            text, padding=True, truncation=True, return_tensors="pt"
        )["input_ids"].to(device)

        # Handle long texts by chunking
        chunks = inputs.split(self.max_length, dim=1)

        if self.output_type == "cls":
            embeddings = [self.model(input_ids=chunk).text_embeds for chunk in chunks]
            text_emb = torch.mean(torch.stack(embeddings), dim=0)
        else:  # "all"
            embeddings = [
                torch.mean(self.model(input_ids=chunk).last_hidden_state, dim=1)
                for chunk in chunks
            ]
            text_emb = torch.mean(torch.stack(embeddings), dim=0)

        return self.text_enc(text_emb)

    def _encode_segments(
        self,
        text: list[list[str]],
        return_all_segments: bool,
        device: torch.device,
    ) -> Tensor:
        """Encode multi-segment text."""
        batch_size = len(text)
        max_k = len(text[0])  # Assumes already padded

        # Flatten: (B, K) -> (B*K,)
        flat_texts = []
        for sample_segments in text:
            flat_texts.extend(sample_segments)

        # Encode all segments
        inputs = self.tokenizer(
            flat_texts, padding=True, truncation=True, return_tensors="pt"
        )["input_ids"].to(device)

        chunks = inputs.split(self.max_length, dim=1)

        if self.output_type == "cls":
            embeddings = [self.model(input_ids=chunk).text_embeds for chunk in chunks]
            text_emb = torch.mean(torch.stack(embeddings), dim=0)
        else:
            embeddings = [
                torch.mean(self.model(input_ids=chunk).last_hidden_state, dim=1)
                for chunk in chunks
            ]
            text_emb = torch.mean(torch.stack(embeddings), dim=0)

        text_co_emb = self.text_enc(text_emb)  # (B*K, coemb_dim)

        if return_all_segments:
            return text_co_emb.view(batch_size, max_k, -1)  # (B, K, D)
        return text_co_emb  # (B*K, D)


# =============================================================================
# Time Series Encoders
# =============================================================================

def get_torch_trans(heads: int = 8, layers: int = 1, channels: int = 64) -> nn.Module:
    """Create a simple transformer encoder."""
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64,
        activation="gelu", batch_first=True
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


class PatchTSTEncoder(nn.Module):
    """
    PatchTST-based time series encoder for instance-level CTTP.

    Args:
        d_model: Model dimension
        patch_len: Patch length
        n_var: Number of variables
        stride: Patch stride
        padding: Padding for patching
        dropout: Dropout rate
        seq_len: Sequence length
        n_heads: Number of attention heads
        d_ff: Feedforward dimension
        e_layers: Number of encoder layers
        coemb_dim: Output co-embedding dimension
        factor: Attention factor
        activation: Activation function
        output_attention: Whether to output attention
    """

    def __init__(
        self,
        d_model: int,
        patch_len: int,
        n_var: int,
        stride: int,
        padding: int,
        dropout: float,
        seq_len: int,
        n_heads: int,
        d_ff: int,
        e_layers: int,
        coemb_dim: int,
        factor: int = 5,
        activation: str = "gelu",
        output_attention: bool = False,
    ):
        super().__init__()
        self.patch_embedding = PatchEmbedding(
            d_model, patch_len, n_var, stride, padding, dropout
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False, factor,
                            attention_dropout=dropout,
                            output_attention=output_attention,
                        ),
                        d_model, n_heads,
                    ),
                    d_model, d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

        # Compute flattened dimension
        n_patches = int((seq_len + padding - patch_len) / stride + 1)
        self.head_nf = d_model * n_patches

        self.post_proj = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(self.head_nf, coemb_dim),
        )

    def forward(self, ts: Tensor) -> Tensor:
        """
        Args:
            ts: (B, L, C) time series
        Returns:
            (B, coemb_dim) embeddings
        """
        ts = ts.permute(0, 2, 1)  # (B, C, L)
        ts_emb = self.patch_embedding(ts)  # (B, Nl, d_model)
        ts_enc_out, _ = self.encoder(ts_emb)  # (B, Nl, d_model)
        return self.post_proj(ts_enc_out)  # (B, coemb_dim)


class PatchTSTMAEEncoder(nn.Module):
    """
    PatchTST with per-variable encoding and dual transformer aggregation.

    This is the MAE-style encoder used in CLIPModel_PatchTST.

    Args:
        d_model: Model dimension
        patch_len: Patch length
        n_var: Number of variables
        stride: Patch stride
        padding: Padding for patching
        dropout: Dropout rate
        seq_len: Sequence length
        n_heads: Number of attention heads
        d_ff: Feedforward dimension
        e_layers: Number of encoder layers
        coemb_dim: Output co-embedding dimension
        factor: Attention factor
        activation: Activation function
        output_attention: Whether to output attention
        pretrain_encoder_path: Optional path to pretrained encoder
    """

    def __init__(
        self,
        d_model: int,
        patch_len: int,
        n_var: int,
        stride: int,
        padding: int,
        dropout: float,
        seq_len: int,
        n_heads: int,
        d_ff: int,
        e_layers: int,
        coemb_dim: int,
        factor: int = 5,
        activation: str = "gelu",
        output_attention: bool = False,
        pretrain_encoder_path: str = "",
    ):
        super().__init__()
        from contsg.models.modules.patchtst import PatchEncoder

        self.n_var = n_var

        # Per-variable patch encoder
        self.patch_encoder = PatchEncoder(
            d_model=d_model,
            patch_len=patch_len,
            stride=stride,
            padding=padding,
            dropout=dropout,
            n_var=n_var,
            n_heads=n_heads,
            d_ff=d_ff,
            e_layers=e_layers,
            seq_len=seq_len,
            factor=factor,
            activation=activation,
            output_attention=output_attention,
        )

        # Load pretrained encoder if provided
        if pretrain_encoder_path:
            state_dict = torch.load(pretrain_encoder_path, map_location="cpu")
            self.patch_encoder.load_state_dict(state_dict)

        # Aggregation transformers
        self.time_transformer = get_torch_trans(heads=n_heads, layers=1, channels=d_model)
        self.var_transformer = get_torch_trans(heads=n_heads, layers=1, channels=d_model)

        # Output projection
        self.out_projector = nn.Linear(d_model, coemb_dim)

    def forward(self, ts: Tensor) -> Tensor:
        """
        Args:
            ts: (B, L, N) time series
        Returns:
            (B, coemb_dim) embeddings
        """
        B, L, N = ts.shape

        # Per-variable encoding: (B*N, Nl, d_model)
        ts_var_emb = self.patch_encoder(ts)

        # Time aggregation: take first token per variable
        var_emb = self.time_transformer(ts_var_emb)[:, :1, :].reshape(B, N, -1)  # (B, N, d_model)

        # Variable aggregation: take first token
        co_emb = self.var_transformer(var_emb)[:, :1, :].reshape(B, -1)  # (B, d_model)

        return self.out_projector(co_emb)  # (B, coemb_dim)


# =============================================================================
# CTTP Lightning Module
# =============================================================================

@Registry.register_model("cttp", aliases=["ctp"])
class CTTPModel(pl.LightningModule):
    """
    CTTP: Contrastive Text-to-Time Series Pretraining.

    A representation learning model for learning joint text-TS embeddings.
    NOT a generative model.

    Modes:
    - instance: Global TS vs Global Text alignment
    - segment: Dual-task (Segment-level + Global alignment)

    Text Encoding:
    - online: Uses HuggingFace model with raw text strings

    Args:
        config: ExperimentConfig or dict with model configuration
        mode: "instance" or "segment"
        text_encoding: "online"
        learning_rate: Learning rate for optimizer
    """

    def __init__(
        self,
        config: Any,
        mode: Literal["instance", "segment"] = "instance",
        text_encoding: Literal["online"] = "online",
        learning_rate: float = 1e-4,
        **kwargs,
    ):
        super().__init__()
        # Resolve mode/text_encoding from config when callers rely on defaults.
        # This keeps YAML-driven training behavior consistent with config.model.*.
        cfg = getattr(config, "model", None)
        if cfg is not None:
            cfg_mode = getattr(cfg, "mode", None)
            if cfg_mode is not None and mode == "instance":
                mode = cfg_mode
            cfg_text_encoding = getattr(cfg, "text_encoding", None)
            if cfg_text_encoding is not None and text_encoding == "online":
                text_encoding = cfg_text_encoding

        self.save_hyperparameters(ignore=["config"])

        self.config = config
        self.mode = mode
        self.text_encoding = text_encoding
        self.lr = learning_rate

        if self.text_encoding != "online":
            raise ValueError(
                "CTTPModel only supports online text encoding; "
                "precomputed text embeddings are not supported."
            )

        self._build_model()

    def _build_model(self) -> None:
        """Build CTTP model components."""
        cfg = self.config.model
        data_cfg = self.config.data

        # Common parameters
        d_model = getattr(cfg, "d_model", 128)
        coemb_dim = getattr(cfg, "coemb_dim", 256)
        n_heads = getattr(cfg, "nheads", 8)
        patch_len = getattr(cfg, "patch_len", 16)
        stride = getattr(cfg, "stride", 8)
        padding = getattr(cfg, "padding", 0)
        dropout = getattr(cfg, "dropout", 0.1)
        d_ff = getattr(cfg, "d_ff", 256)
        e_layers = getattr(cfg, "e_layers", 3)
        factor = getattr(cfg, "factor", 5)
        activation = getattr(cfg, "activation", "gelu")
        textemb_hidden_dim = getattr(cfg, "textemb_hidden_dim", 512)

        # Loss configuration
        self.loss_type = getattr(cfg, "loss_type", "contrastive")
        self.margin = getattr(cfg, "margin", 3.0)
        self.temperature = getattr(cfg, "temperature", 0.07)
        self.normalize_embeddings = getattr(cfg, "normalize_embeddings", False)

        # Segment mode parameters
        self.segment_loss_weight = getattr(cfg, "segment_loss_weight", 1.0)
        self.global_loss_weight = getattr(cfg, "global_loss_weight", 0.5)

        # Build TS encoder
        if self.mode == "segment":
            segment_len = getattr(cfg, "segment_len", data_cfg.seq_length // 4)
            n_segments = getattr(cfg, "n_segments", 4)
            self.n_segments = n_segments

            self.ts_enc = SegmentTSEncoder(
                segment_len=segment_len,
                n_segments=n_segments,
                d_model=d_model,
                patch_len=patch_len,
                n_var=data_cfg.n_var,
                stride=stride,
                padding=padding,
                dropout=dropout,
                e_layers=e_layers,
                n_heads=n_heads,
                d_ff=d_ff,
                coemb_dim=coemb_dim,
                factor=factor,
                activation=activation,
            )
        else:
            # Instance mode: use PatchTSTMAE encoder
            ts_encoder_type = getattr(cfg, "ts_encoder_type", "patchtst_mae")

            if ts_encoder_type == "patchtst_mae":
                self.ts_enc = PatchTSTMAEEncoder(
                    d_model=d_model,
                    patch_len=patch_len,
                    n_var=data_cfg.n_var,
                    stride=stride,
                    padding=padding,
                    dropout=dropout,
                    seq_len=data_cfg.seq_length,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    e_layers=e_layers,
                    coemb_dim=coemb_dim,
                    factor=factor,
                    activation=activation,
                    pretrain_encoder_path=getattr(cfg, "pretrain_encoder_path", ""),
                )
            else:
                self.ts_enc = PatchTSTEncoder(
                    d_model=d_model,
                    patch_len=patch_len,
                    n_var=data_cfg.n_var,
                    stride=stride,
                    padding=padding,
                    dropout=dropout,
                    seq_len=data_cfg.seq_length,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    e_layers=e_layers,
                    coemb_dim=coemb_dim,
                    factor=factor,
                    activation=activation,
                )

        # Build text encoder (online only)
        pretrain_model_path = getattr(cfg, "pretrain_model_path", "zer0int/LongCLIP-GmP-ViT-L-14")
        pretrain_model_dim = getattr(cfg, "pretrain_model_dim", 512)

        self.text_enc = HuggingFaceTextEncoder(
            pretrain_model_path=pretrain_model_path,
            pretrain_model_dim=pretrain_model_dim,
            textemb_hidden_dim=textemb_hidden_dim,
            coemb_dim=coemb_dim,
        )

        # Loss functions
        self.contrastive_loss = ContrastiveLoss(margin=self.margin)
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

    def _maybe_normalize(self, emb: Tensor) -> Tensor:
        """Optionally L2-normalize embeddings along the feature dimension.

        Args:
            emb: (B, D) or (B, K, D) embedding tensor
        Returns:
            (B, D) or (B, K, D) tensor with optional L2 normalization
        """
        if not self.normalize_embeddings:
            return emb
        return F.normalize(emb, p=2, dim=-1)

    @staticmethod
    def _build_text_labels(texts: list[str], device: torch.device) -> Tensor:
        """Build batch-local class labels from raw text strings.

        Args:
            texts: List[str] with length B
            device: Torch device for label tensor
        Returns:
            (B,) integer labels where identical texts share the same label
        """
        label_map: dict[str, int] = {}
        labels: list[int] = []
        for text in texts:
            if text not in label_map:
                label_map[text] = len(label_map)
            labels.append(label_map[text])
        return torch.tensor(labels, device=device, dtype=torch.long)

    def _supcon_loss(self, emb_a: Tensor, emb_b: Tensor, labels: Tensor) -> Tensor:
        """Supervised contrastive loss with cross-modal paired views.

        Args:
            emb_a: (B, D) embeddings for modality A (e.g., time series)
            emb_b: (B, D) embeddings for modality B (e.g., text)
            labels: (B,) class labels aligned with emb_a/emb_b
        Returns:
            Scalar loss tensor
        """
        features = torch.cat([emb_a, emb_b], dim=0)
        labels = torch.cat([labels, labels], dim=0)
        if self.normalize_embeddings:
            features = F.normalize(features, p=2, dim=1)

        logits = torch.matmul(features, features.t()) / self.temperature
        self_mask = torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)
        pos_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & ~self_mask

        logits = logits.masked_fill(self_mask, float("-inf"))
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        log_prob = log_prob.masked_fill(~pos_mask, 0.0)

        pos_count = pos_mask.sum(dim=1).clamp(min=1)
        loss = -log_prob.sum(dim=1) / pos_count
        return loss.mean()

    def forward(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        """
        Forward pass computing loss.

        For instance mode:
            batch: {ts: (B,L,C), cap: List[str]}

        For segment mode:
            batch: {ts: (B,L,C), segment_texts: List[List[str]], summary: List[str]}
        """
        if self.mode == "segment":
            return self._forward_segment(batch)
        else:
            return self._forward_instance(batch)

    def _forward_instance(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        """Instance-level forward pass."""
        ts = batch["ts"]  # (B, L, C)
        B = ts.shape[0]
        device = ts.device

        # Get TS embedding
        ts_co_emb = self.ts_enc(ts)  # (B, coemb_dim)
        ts_co_emb = self._maybe_normalize(ts_co_emb)

        # Get text embedding
        text = batch["cap"]  # List[str]
        text_co_emb = self.text_enc(text)
        text_co_emb = self._maybe_normalize(text_co_emb)

        loss_dict = {}

        if self.loss_type == "contrastive":
            # Positive pairs
            pos_labels = torch.zeros(B, device=device)
            loss_dict["positive"] = self.contrastive_loss(text_co_emb, ts_co_emb, pos_labels)

            # Negative pairs (shift indices)
            neg_labels = torch.ones(B, device=device)
            shift = np.random.randint(1, max(B, 2))
            new_idx = (np.arange(B) + shift) % B
            mis_ts_co_emb = ts_co_emb[new_idx]
            loss_dict["negative"] = self.contrastive_loss(text_co_emb, mis_ts_co_emb, neg_labels)
            loss_dict["loss"] = loss_dict["positive"] + loss_dict["negative"]

        elif self.loss_type == "supcon":
            labels = self._build_text_labels(text, device)
            loss_dict["supcon"] = self._supcon_loss(ts_co_emb, text_co_emb, labels)
            loss_dict["loss"] = loss_dict["supcon"]

        else:  # CE / InfoNCE
            sim = torch.mm(ts_co_emb, text_co_emb.t())  # (B, B)
            labels = torch.arange(B, device=device)
            loss_dict["ts2text"] = self.ce_loss(sim, labels).mean()
            loss_dict["text2ts"] = self.ce_loss(sim.t(), labels).mean()
            loss_dict["loss"] = (loss_dict["ts2text"] + loss_dict["text2ts"]) / 2

        return loss_dict

    def _forward_segment(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        """Segment-level forward pass with dual-task loss."""
        ts = batch["ts"]  # (B, L, C)
        segment_texts = batch["segment_texts"]  # List[List[str]]
        summary = batch["summary"]  # List[str]

        B = ts.shape[0]
        K = self.n_segments
        device = ts.device

        # Default collate may transpose to [K][B]; restore [B][K] for encoder.
        if (
            isinstance(segment_texts, list)
            and len(segment_texts) == K
            and all(isinstance(seg, list) and len(seg) == B for seg in segment_texts)
        ):
            segment_texts = [list(items) for items in zip(*segment_texts)]

        # Encode TS segments: (B, K, D)
        ts_seg_emb = self.ts_enc(ts)
        ts_seg_emb = self._maybe_normalize(ts_seg_emb)

        # Encode text segments: (B, K, D)
        text_seg_emb = self.text_enc(segment_texts, return_all_segments=True)
        text_seg_emb = self._maybe_normalize(text_seg_emb)

        # Encode summary: (B, D)
        summary_emb = self.text_enc(summary)
        summary_emb = self._maybe_normalize(summary_emb)

        # Global TS embedding: (B, D)
        ts_global_emb = ts_seg_emb.mean(dim=1)
        ts_global_emb = self._maybe_normalize(ts_global_emb)

        loss_dict = {}

        # === Segment-Level Loss ===
        ts_seg_flat = ts_seg_emb.view(B * K, -1)
        text_seg_flat = text_seg_emb.view(B * K, -1)

        if self.loss_type == "ce":
            sim = torch.mm(ts_seg_flat, text_seg_flat.t())
            labels = torch.arange(B * K, device=device)
            loss_dict["segment_ts2text"] = self.ce_loss(sim, labels).mean()
            loss_dict["segment_text2ts"] = self.ce_loss(sim.t(), labels).mean()
            loss_dict["segment"] = (loss_dict["segment_ts2text"] + loss_dict["segment_text2ts"]) / 2
        elif self.loss_type == "supcon":
            flat_segment_texts: list[str] = []
            for sample_segments in segment_texts:
                flat_segment_texts.extend(sample_segments)
            segment_labels = self._build_text_labels(flat_segment_texts, device)
            loss_dict["segment"] = self._supcon_loss(ts_seg_flat, text_seg_flat, segment_labels)
        else:
            pos_labels = torch.zeros(B * K, device=device)
            loss_dict["segment_pos"] = self.contrastive_loss(text_seg_flat, ts_seg_flat, pos_labels)

            shift = np.random.randint(1, max(B * K, 2))
            new_idx = (torch.arange(B * K) + shift) % (B * K)
            neg_labels = torch.ones(B * K, device=device)
            loss_dict["segment_neg"] = self.contrastive_loss(
                text_seg_flat, ts_seg_flat[new_idx], neg_labels
            )
            loss_dict["segment"] = loss_dict["segment_pos"] + loss_dict["segment_neg"]

        # === Global-Level Loss ===
        if self.loss_type == "ce":
            sim_global = torch.mm(ts_global_emb, summary_emb.t())
            labels_global = torch.arange(B, device=device)
            loss_dict["global"] = (
                self.ce_loss(sim_global, labels_global).mean()
                + self.ce_loss(sim_global.t(), labels_global).mean()
            ) / 2
        elif self.loss_type == "supcon":
            global_labels = self._build_text_labels(summary, device)
            loss_dict["global"] = self._supcon_loss(ts_global_emb, summary_emb, global_labels)
        else:
            pos_labels = torch.zeros(B, device=device)
            loss_dict["global"] = self.contrastive_loss(summary_emb, ts_global_emb, pos_labels)

        # === Total Loss ===
        loss_dict["loss"] = (
            self.segment_loss_weight * loss_dict["segment"]
            + self.global_loss_weight * loss_dict["global"]
        )

        return loss_dict

    # =========================================================================
    # Lightning Training Methods
    # =========================================================================

    def configure_optimizers(self):
        """Configure optimizer (AdamW with weight decay)."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=1e-6,
        )
        return optimizer

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        """Training step."""
        loss_dict = self(batch)
        self.log("train/loss", loss_dict["loss"], prog_bar=True)

        for key, value in loss_dict.items():
            if key != "loss":
                self.log(f"train/{key}", value)

        return loss_dict["loss"]

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Tensor]:
        """Validation step with retrieval accuracy."""
        loss_dict = self(batch)
        self.log("val/loss", loss_dict["loss"], prog_bar=True)

        # Compute retrieval accuracy
        ts = batch["ts"]
        B = ts.shape[0]

        if self.mode == "segment":
            sim = self.compute_similarity(
                ts, batch["segment_texts"], batch["summary"], return_global=True
            )
        else:
            text = batch["cap"]
            ts_emb = self.get_ts_embedding(ts)
            text_emb = self.get_text_embedding(text)
            sim = F.softmax(torch.mm(ts_emb, text_emb.t()), dim=-1)

        pred = torch.argmax(sim, dim=-1).cpu().numpy()
        if self.loss_type == "supcon":
            if self.mode == "segment":
                labels = self._build_text_labels(batch["summary"], ts.device).cpu().numpy()
            else:
                labels = self._build_text_labels(text, ts.device).cpu().numpy()
            acc = (labels[pred] == labels).sum() / B
        else:
            gt = np.arange(B)
            acc = (pred == gt).sum() / B

        self.log("val/acc", acc, prog_bar=True)

        return {"loss": loss_dict["loss"], "acc": acc}

    # =========================================================================
    # Inference Methods
    # =========================================================================

    def get_ts_embedding(self, ts: Tensor) -> Tensor:
        """
        Get time series embedding.

        Args:
            ts: (B, L, C) time series

        Returns:
            For instance mode: (B, coemb_dim)
            For segment mode: (B, K, coemb_dim) segment embeddings
        """
        emb = self.ts_enc(ts)
        return self._maybe_normalize(emb)

    def get_global_ts_embedding(self, ts: Tensor) -> Tensor:
        """
        Get global time series embedding.

        For segment mode, returns mean-pooled segment embeddings.
        For instance mode, returns standard embedding.

        Args:
            ts: (B, L, C)
        Returns:
            (B, coemb_dim)
        """
        emb = self.ts_enc(ts)
        if self.mode == "segment":
            emb = emb.mean(dim=1)
        return self._maybe_normalize(emb)

    def get_text_embedding(
        self,
        text: list[str] | list[list[str]],
    ) -> Tensor:
        """
        Get text embedding.

        Args:
            text: Raw text strings
        Returns:
            (B, coemb_dim) or (B, K, coemb_dim) for segments
        """
        emb = self.text_enc(text)
        return self._maybe_normalize(emb)

    def compute_similarity(
        self,
        ts: Tensor,
        text: list[str] | list[list[str]],
        summary: list[str] | None = None,
        return_global: bool = True,
    ) -> Tensor:
        """
        Compute similarity matrix for retrieval.

        Args:
            ts: (B, L, C) time series
            text: Text input (mode-dependent)
            summary: Summary text (segment mode only)
            return_global: If True, return global similarity; else segment-level

        Returns:
            (B, B) similarity matrix
        """
        if self.mode == "segment":
            ts_seg_emb = self.ts_enc(ts)  # (B, K, D)
            ts_seg_emb = self._maybe_normalize(ts_seg_emb)

            if return_global:
                ts_global = ts_seg_emb.mean(dim=1)  # (B, D)
                ts_global = self._maybe_normalize(ts_global)
                if not isinstance(summary, list):
                    raise TypeError("summary must be a list of strings in segment mode")
                summary_emb = self.text_enc(summary)
                summary_emb = self._maybe_normalize(summary_emb)
                sim = torch.mm(ts_global, summary_emb.t())
            else:
                B, K, D = ts_seg_emb.shape
                ts_flat = ts_seg_emb.view(B * K, -1)
                if not isinstance(text, list):
                    raise TypeError("text must be a list of string lists in segment mode")
                text_emb = self.text_enc(text, return_all_segments=True)
                text_emb = self._maybe_normalize(text_emb)
                text_flat = text_emb.view(B * K, -1)
                sim = torch.mm(ts_flat, text_flat.t())

            return F.softmax(sim, dim=-1)

        else:
            ts_emb = self.ts_enc(ts)
            ts_emb = self._maybe_normalize(ts_emb)
            if not isinstance(text, list):
                raise TypeError("text must be a list of strings in instance mode")
            text_emb = self.text_enc(text)
            text_emb = self._maybe_normalize(text_emb)
            sim = torch.mm(ts_emb, text_emb.t())
            return F.softmax(sim, dim=-1)

    def generate(self, *args, **kwargs):
        """CTTP is not a generative model."""
        raise NotImplementedError(
            "CTTP is a representation learning model, not a generative model. "
            "Use get_ts_embedding(), get_text_embedding(), or compute_similarity() instead."
        )
