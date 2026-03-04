"""
Pydantic configuration schemas for ConTSG.

This module defines type-safe configuration models using Pydantic v2.
All configuration validation happens automatically through Pydantic's type system.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Tuple

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from contsg.config.label_infer import infer_num_classes_from_attrs

# =============================================================================
# Stage Configuration (for multi-stage training)
# =============================================================================

class StageConfig(BaseModel):
    """Single training stage configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Stage name (e.g., 'pretrain', 'finetune')")
    epochs: int = Field(100, ge=1, description="Epochs for this stage")
    lr: float = Field(1e-3, gt=0, description="Learning rate for this stage")
    use_condition: bool = Field(True, description="Whether to use condition in this stage")
    freeze_modules: list[str] = Field(
        default_factory=list,
        description="Module names to freeze (e.g., ['encoder', 'text_encoder'])"
    )
    load_from_stage: str | None = Field(
        None, description="Load weights from this stage's checkpoint"
    )
    early_stopping_patience: int = Field(
        50, ge=0, description="Early stopping patience for this stage"
    )


# Stage presets for common training patterns
STAGE_PRESETS: dict[str, list[dict[str, Any]]] = {
    "single": [
        {"name": "finetune", "epochs": 700, "use_condition": True}
    ],
    "two_stage": [
        {"name": "pretrain", "epochs": 200, "lr": 1e-3, "use_condition": False},
        {"name": "finetune", "epochs": 500, "lr": 1e-3, "use_condition": True, "load_from_stage": "pretrain"},
    ],
    "pretrain_freeze": [
        {"name": "pretrain", "epochs": 200, "lr": 1e-3, "use_condition": False},
        {"name": "finetune", "epochs": 500, "lr": 1e-3, "use_condition": True,
         "load_from_stage": "pretrain", "freeze_modules": ["encoder"]},
    ],
}


# =============================================================================
# Condition Configuration
# =============================================================================

class TextConditionConfig(BaseModel):
    """Text condition encoder configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(True, description="Enable text conditioning")
    input_dim: int = Field(1024, ge=1, description="Input embedding dimension (Qwen3: 1024)")
    output_dim: int | None = Field(None, description="Output dimension after projection")
    embedding_key: str = Field("cap_emb", description="Key in batch for embeddings")
    dropout: float = Field(0.0, ge=0, le=1, description="Dropout rate")
    text_projector: str | None = Field(
        None, description="Text projector type for model-specific conditioning"
    )
    num_stages: int | None = Field(
        None, ge=1, description="Number of stages for diffusion-step projectors"
    )
    cfg_scale: float = Field(
        1.0, ge=0, description="Classifier-free guidance scale (models may ignore)"
    )


class AttributeConditionConfig(BaseModel):
    """Attribute condition encoder configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(False, description="Enable attribute conditioning")
    continuous_dim: int = Field(0, ge=0, description="Number of continuous attributes")
    discrete_configs: list[dict[str, int]] = Field(
        default_factory=list,
        description="List of {'num_classes': N, 'embed_dim': D} for each discrete attribute"
    )
    output_dim: int = Field(128, ge=1, description="Output dimension")
    dropout: float = Field(0.0, ge=0, le=1, description="Dropout rate")


class LabelConditionConfig(BaseModel):
    """Label condition encoder configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(False, description="Enable label conditioning")
    num_classes: int = Field(10, ge=1, description="Number of classes")
    output_dim: int = Field(64, ge=1, description="Embedding dimension")


class ConditionConfig(BaseModel):
    """Complete condition system configuration."""

    model_config = ConfigDict(extra="forbid")

    text: TextConditionConfig = Field(default_factory=lambda: TextConditionConfig())  # pyright: ignore[reportCallIssue]
    attribute: AttributeConditionConfig = Field(default_factory=lambda: AttributeConditionConfig())  # pyright: ignore[reportCallIssue]
    label: LabelConditionConfig = Field(default_factory=lambda: LabelConditionConfig())  # pyright: ignore[reportCallIssue]

    fusion: Literal["concat", "sum", "attention"] = Field(
        "concat", description="Multi-condition fusion strategy"
    )
    output_dim: int | None = Field(
        None, description="Final condition embedding dimension"
    )
    condition_dropout: float = Field(
        0.0, ge=0, le=1, description="Dropout rate for classifier-free guidance"
    )


# =============================================================================
# Training Configuration
# =============================================================================

class TrainConfig(BaseModel):
    """Training configuration with multi-stage support."""

    model_config = ConfigDict(extra="forbid")

    # Multi-stage configuration
    stages: list[StageConfig] = Field(
        default_factory=lambda: [StageConfig(name="finetune", epochs=700)],  # pyright: ignore[reportCallIssue]
        description="Training stages (use 'single', 'two_stage', or custom)"
    )
    stages_preset: str | None = Field(
        None, description="Use preset stages: 'single', 'two_stage', 'pretrain_freeze'"
    )

    # Default parameters (used when stages not specified explicitly)
    epochs: int = Field(700, ge=1, description="Number of training epochs")
    batch_size: int = Field(256, ge=1, description="Training batch size")
    lr: float = Field(1e-3, gt=0, description="Learning rate")
    weight_decay: float = Field(1e-4, ge=0, description="Weight decay for optimizer")
    scheduler: Literal["cosine", "step", "plateau", "none"] = Field(
        "cosine", description="Learning rate scheduler type"
    )
    scheduler_params: dict[str, Any] = Field(
        default_factory=dict, description="Scheduler-specific parameters"
    )
    early_stopping_patience: int = Field(
        50, ge=0, description="Early stopping patience (0 to disable)"
    )
    gradient_clip_val: float = Field(1.0, ge=0, description="Gradient clipping value")
    accumulate_grad_batches: int = Field(1, ge=1, description="Gradient accumulation steps")
    val_check_interval: float = Field(1.0, gt=0, description="Validation check interval")
    num_workers: int = Field(4, ge=0, description="DataLoader workers")
    pin_memory: bool = Field(True, description="Pin memory in DataLoader")
    limit_train_batches: int | float = Field(
        1.0, ge=0, description="Limit training batches (int or fraction)"
    )
    limit_val_batches: int | float = Field(
        1.0, ge=0, description="Limit validation batches (int or fraction)"
    )
    limit_test_batches: int | float = Field(
        1.0, ge=0, description="Limit test batches (int or fraction)"
    )
    num_sanity_val_steps: int = Field(2, ge=0, description="Sanity validation steps")

    # Gradient/Parameter monitoring
    log_grad_norm: bool = Field(True, description="Log gradient norm to TensorBoard")
    log_param_norm: bool = Field(True, description="Log parameter norm to TensorBoard")
    log_norm_every_n_steps: int = Field(50, ge=1, description="Log norms every N steps")

    @model_validator(mode="after")
    def resolve_stages_preset(self) -> TrainConfig:
        """Resolve stages preset if specified, and sync CLI params to stages."""
        if self.stages_preset and self.stages_preset in STAGE_PRESETS:
            preset_stages = STAGE_PRESETS[self.stages_preset]
            object.__setattr__(
                self,
                "stages",
                [StageConfig(**s) for s in preset_stages]
            )
        else:
            # Sync CLI-provided params to the first (default) stage
            # This ensures `--epochs 3` actually trains for 3 epochs
            if len(self.stages) == 1 and self.stages[0].name == "finetune":
                updated_stage = self.stages[0].model_copy(update={
                    "epochs": self.epochs,
                    "lr": self.lr,
                    "early_stopping_patience": self.early_stopping_patience,
                })
                object.__setattr__(self, "stages", [updated_stage])
        return self


# =============================================================================
# Data Configuration
# =============================================================================

class DataConfig(BaseModel):
    """Dataset configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Dataset name (e.g., 'synth-m', 'ettm1')")
    data_folder: Path = Field(..., description="Path to dataset folder")
    n_var: int = Field(..., ge=1, description="Number of variables/channels")
    seq_length: int = Field(128, ge=1, description="Sequence length")
    batch_size: int | None = Field(None, description="Override train.batch_size")

    # Optional dataset-specific settings
    normalize: bool = Field(True, description="Normalize time series")
    train_split: float = Field(0.8, ge=0, le=1, description="Train split ratio")
    val_split: float = Field(0.1, ge=0, le=1, description="Validation split ratio")

    @field_validator("data_folder", mode="before")
    @classmethod
    def convert_to_path(cls, v: Any) -> Path:
        return Path(v) if isinstance(v, str) else v


# =============================================================================
# Model Configuration
# =============================================================================

class ModelConfig(BaseModel):
    """Base model configuration."""

    model_config = ConfigDict(extra="allow")  # Allow extra fields for model-specific params

    name: str = Field(..., description="Model name")
    channels: int = Field(64, ge=1, description="Model hidden channels")
    layers: int = Field(3, ge=1, description="Number of layers")
    nheads: int = Field(8, ge=1, description="Number of attention heads")
    dropout: float = Field(0.1, ge=0, le=1, description="Dropout rate")


class VerbalTSModelConfig(ModelConfig):
    """VerbalTS model configuration."""

    name: Literal["verbalts"] = "verbalts"  # pyright: ignore[reportIncompatibleVariableOverride]
    condition_type: Literal["adaLN", "cross_attention", "cross_attn", "add"] = Field(
        "adaLN", description="Conditioning mechanism type"
    )
    L_patch_len: int = Field(3, ge=1, description="Local patch length")
    multipatch_num: int = Field(3, ge=1, description="Number of multi-patches")
    base_patch: int = Field(4, ge=1, description="Base patch size")
    diffusion_steps: int = Field(50, ge=1, description="Diffusion timesteps")
    diffusion_embedding_dim: int | None = Field(
        None, ge=1, description="Diffusion embedding dimension override"
    )
    attention_mask_type: Literal["full", "parallel"] = Field(
        "parallel", description="Attention mask type"
    )
    noise_schedule: Literal["linear", "cosine", "quad"] = Field(
        "cosine", description="Noise schedule type"
    )
    beta_start: float | None = Field(
        None, ge=0, description="Beta schedule start (for linear/quad schedules)"
    )
    beta_end: float | None = Field(
        None, ge=0, description="Beta schedule end (for linear/quad schedules)"
    )
    side_var_emb: int = Field(16, ge=1, description="Side variable embedding dimension")
    side_time_emb: int = Field(112, ge=1, description="Side time embedding dimension")


class BridgeModelConfig(ModelConfig):
    """Bridge model configuration."""

    name: Literal["bridge"] = "bridge"  # pyright: ignore[reportIncompatibleVariableOverride]
    latent_dim: int = Field(32, ge=1, description="Latent dimension")
    num_latents: int = Field(16, ge=1, description="Number of latent vectors")
    latent_unit: int = Field(1, ge=1, description="Latent unit count for UNet conditioning")
    repre_emb_channels: int | None = Field(
        None, ge=1, description="Prototype embedding channels (defaults to latent_dim)"
    )
    prototype_dim: int | None = Field(
        None, ge=1, description="Prototype encoder hidden dim (defaults to repre_emb_channels)"
    )
    context_dim: int | None = Field(
        None, ge=1, description="Cross-attention context dim (defaults to repre_emb_channels)"
    )
    text_dim: int | None = Field(None, ge=1, description="Text embedding dimension")
    fusion_type: Literal["gated_add", "add"] = Field(
        "gated_add", description="Text-prototype fusion type"
    )
    cond_drop_prob: float = Field(0.5, ge=0, le=1, description="Condition dropout probability")
    use_cfg: bool = Field(True, description="Enable classifier-free guidance dropout")

    # UNet architecture
    model_channels: int = Field(64, ge=1, description="UNet base channels")
    num_res_blocks: int = Field(2, ge=1, description="Residual blocks per level")
    attention_resolutions: list[int] = Field(
        default_factory=lambda: [1, 2, 4], description="Attention resolutions"
    )
    channel_mult: list[int] = Field(
        default_factory=lambda: [1, 2, 4, 4], description="Channel multiplier per level"
    )
    num_heads: int = Field(8, ge=1, description="Attention heads")
    num_head_channels: int = Field(-1, description="Head channels (-1 to infer from num_heads)")
    use_scale_shift_norm: bool = Field(True, description="Use scale-shift norm in ResBlocks")
    resblock_updown: bool = Field(True, description="Use ResBlock up/downsampling")
    use_spatial_transformer: bool = Field(True, description="Enable spatial transformer")
    transformer_depth: int = Field(1, ge=1, description="Transformer block depth")
    dropout: float = Field(0.0, ge=0, le=1, description="UNet dropout rate")

    # Diffusion schedule
    diffusion_steps: int = Field(50, ge=1, description="Diffusion timesteps")
    noise_schedule: Literal["linear", "cosine", "quad"] = Field(
        "quad", description="Noise schedule type"
    )
    beta_start: float = Field(0.0001, ge=0, description="Beta schedule start")
    beta_end: float = Field(0.5, ge=0, description="Beta schedule end")
    bridge_type: Literal["vp", "ve"] = Field("vp", description="Bridge SDE type")


class T2SAutoEncoderConfig(BaseModel):
    """T2S AutoEncoder configuration."""

    model_config = ConfigDict(extra="forbid")

    num_hiddens: int = Field(128, ge=1, description="Conv block hidden channels")
    num_residual_layers: int = Field(2, ge=1, description="Number of residual layers")
    num_residual_hiddens: int = Field(256, ge=1, description="Residual block hidden channels")
    embedding_dim: int = Field(64, ge=1, description="Latent embedding dimension")
    num_input_channels: int | None = Field(
        None, ge=1, description="Input channels (default: data.n_var)"
    )


class T2SModelConfig(ModelConfig):
    """T2S (Text-to-Series) model configuration."""

    name: Literal["t2s"] = "t2s"  # pyright: ignore[reportIncompatibleVariableOverride]

    # Transformer configuration
    channels: int = Field(64, ge=1, description="Transformer embedding size")
    layers: int = Field(4, ge=1, description="Transformer layers")
    nheads: int = Field(4, ge=1, description="Transformer heads")
    patch_size: int = Field(2, ge=1, description="Patch size for 2D patching")
    mlp_ratio: float = Field(2.0, ge=1.0, description="MLP expansion ratio")

    # Flow matching configuration
    flow_steps: int = Field(10, ge=1, description="Flow matching steps")
    num_infer_steps: int | None = Field(
        None, ge=1, description="Sampling steps (default: flow_steps)"
    )
    cfg_scale: float = Field(1.0, ge=0, description="CFG scale for sampling")

    # Text conditioning
    text_dim: int | None = Field(None, ge=1, description="Text embedding input dim override")

    # AutoEncoder configuration
    ae: T2SAutoEncoderConfig = Field(
        default_factory=lambda: T2SAutoEncoderConfig(),  # pyright: ignore[reportCallIssue]
        description="AutoEncoder configuration",
    )


class TimeVQVAEVQConfig(BaseModel):
    """VQ-VAE specific configuration for TimeVQVAE."""

    model_config = ConfigDict(extra="allow")

    n_fft: int = Field(4, ge=1, description="FFT size for time-frequency transform")
    codebook_sizes: dict[str, int] = Field(
        default_factory=lambda: {"lf": 1024, "hf": 1024},
        description="Codebook sizes for LF and HF"
    )
    codebook_dim: int = Field(8, ge=1, description="Codebook dimension")


class TimeVQVAEEncoderConfig(BaseModel):
    """Encoder configuration for TimeVQVAE."""

    model_config = ConfigDict(extra="allow")

    init_dim: int = Field(4, ge=1, description="Initial conv dimension")
    hid_dim: int = Field(128, ge=1, description="Hidden dimension")
    n_resnet_blocks: int = Field(2, ge=1, description="Number of ResNet blocks")
    downsampled_width: dict[str, int] = Field(
        default_factory=lambda: {"lf": 8, "hf": 32},
        description="Downsampled width for LF and HF"
    )


class TimeVQVAEDecoderConfig(BaseModel):
    """Decoder configuration for TimeVQVAE."""

    model_config = ConfigDict(extra="allow")

    n_resnet_blocks: int = Field(2, ge=1, description="Number of ResNet blocks")


class TimeVQVAEESSConfig(BaseModel):
    """ESS (Enhanced Sampling Scheme) configuration for TimeVQVAE MaskGIT."""

    model_config = ConfigDict(extra="allow")

    use: bool = Field(False, description="Whether to use ESS (under maintenance)")
    error_ratio_ma_rate: float = Field(0.3, ge=0, le=1, description="Error ratio moving average rate")


class TimeVQVAEMaskGITConfig(BaseModel):
    """MaskGIT configuration for TimeVQVAE."""

    model_config = ConfigDict(extra="allow")

    choice_temperatures: dict[str, float] = Field(
        default_factory=lambda: {"lf": 10.0, "hf": 0.0},
        description="Temperature for LF and HF (higher temp -> higher sample diversity)"
    )
    T: dict[str, int] = Field(
        default_factory=lambda: {"lf": 10, "hf": 10},
        description="Number of iterations for LF and HF"
    )
    cfg_scale: float = Field(
        1.0,
        description="Classifier-free guidance scale (1.0 = no CFG)"
    )
    ESS: TimeVQVAEESSConfig = Field(
        default_factory=lambda: TimeVQVAEESSConfig(),  # pyright: ignore[reportCallIssue]
        description="ESS (Enhanced Sampling Scheme) configuration"
    )


class TimeVQVAEPriorConfig(BaseModel):
    """Prior configuration for TimeVQVAE."""

    model_config = ConfigDict(extra="allow")

    hidden_dim: int = Field(128, ge=1, description="Hidden dimension")
    n_layers: int = Field(4, ge=1, description="Number of layers")
    heads: int = Field(2, ge=1, description="Number of attention heads")
    ff_mult: float = Field(1.0, ge=0.1, description="Feed-forward multiplier")
    use_rmsnorm: bool = Field(True, description="Use RMSNorm")
    p_unconditional: float = Field(0.2, ge=0, le=1, description="Unconditional probability")


class TimeVQVAEModelConfig(ModelConfig):
    """TimeVQVAE model configuration."""

    name: Literal["timevqvae"] = "timevqvae"  # pyright: ignore[reportIncompatibleVariableOverride]
    timepoint: int | None = Field(
        None, ge=1, description="Sequence length (overridden by data.seq_length)"
    )
    variable_num: int | None = Field(
        None, ge=1, description="Number of variables (overridden by data.n_var)"
    )
    pretrained_model_path: str | None = Field(
        None, description="Path to pretrained VQ-VAE checkpoint (optional)"
    )

    # Nested configurations (aliased for compatibility)
    vqvae: TimeVQVAEVQConfig = Field(
        default_factory=lambda: TimeVQVAEVQConfig(),  # pyright: ignore[reportCallIssue]
        alias="VQ-VAE",
        description="VQ-VAE configuration"
    )
    encoder: TimeVQVAEEncoderConfig = Field(
        default_factory=lambda: TimeVQVAEEncoderConfig(),  # pyright: ignore[reportCallIssue]
        description="Encoder configuration"
    )
    decoder: TimeVQVAEDecoderConfig = Field(
        default_factory=lambda: TimeVQVAEDecoderConfig(),  # pyright: ignore[reportCallIssue]
        description="Decoder configuration"
    )
    maskgit: TimeVQVAEMaskGITConfig = Field(
        default_factory=lambda: TimeVQVAEMaskGITConfig(),  # pyright: ignore[reportCallIssue]
        alias="MaskGIT",
        description="MaskGIT configuration"
    )
    prior: TimeVQVAEPriorConfig = Field(
        default_factory=lambda: TimeVQVAEPriorConfig(),  # pyright: ignore[reportCallIssue]
        description="Prior configuration"
    )


class TimeWeaverModelConfig(ModelConfig):
    """TimeWeaver model configuration."""

    name: Literal["timeweaver", "tw"] = "timeweaver"  # pyright: ignore[reportIncompatibleVariableOverride]
    attr_dim: int = Field(64, ge=1, description="Attribute embedding dimension")
    num_attr_heads: int = Field(4, ge=1, description="Attribute attention heads")
    diffusion_steps: int = Field(50, ge=1, description="Diffusion timesteps")
    noise_schedule: Literal["linear", "cosine", "quad"] = Field(
        "quad", description="Noise schedule type"
    )
    beta_start: float = Field(0.0001, ge=0, description="Beta schedule start")
    beta_end: float = Field(0.5, ge=0, description="Beta schedule end")


class WaveStitchModelConfig(ModelConfig):
    """WaveStitch model configuration."""

    name: Literal["wavestitch"] = "wavestitch"  # pyright: ignore[reportIncompatibleVariableOverride]
    diffusion_steps: int = Field(200, ge=1, description="Diffusion timesteps")
    beta_start: float = Field(0.0001, ge=0, description="Beta schedule start")
    beta_end: float = Field(0.02, ge=0, description="Beta schedule end")
    res_channels: int = Field(64, ge=1, description="Residual channels")
    skip_channels: int = Field(64, ge=1, description="Skip channels")
    num_res_layers: int = Field(4, ge=1, description="Number of residual layers")
    diff_step_embed_in: int = Field(32, ge=1, description="Diffusion step input embedding")
    diff_step_embed_mid: int = Field(64, ge=1, description="Diffusion step mid embedding")
    diff_step_embed_out: int = Field(64, ge=1, description="Diffusion step output embedding")
    s4_lmax: int = Field(100, ge=1, description="S4 sequence length limit")
    s4_dstate: int = Field(64, ge=1, description="S4 state dimension")
    s4_dropout: float = Field(0.0, ge=0, le=1, description="S4 dropout")
    s4_bidirectional: bool = Field(True, description="Use bidirectional S4")
    s4_layernorm: bool = Field(True, description="Enable S4 layer norm")
    attr_embed_dim: int = Field(128, ge=1, description="Attribute embedding dimension")
    cond_channels: int = Field(16, ge=1, description="Condition channels appended to input")


class RetrievalModelConfig(ModelConfig):
    """Retrieval baseline configuration."""

    name: Literal["retrieval"] = "retrieval"  # pyright: ignore[reportIncompatibleVariableOverride]
    similarity_metric: Literal["cosine", "euclidean"] = Field(
        "cosine", description="Similarity metric"
    )
    top_k: int = Field(1, ge=1, description="Number of nearest neighbors")


class TEditModelConfig(ModelConfig):
    """TEdit model configuration."""

    name: Literal["tedit"] = "tedit"  # pyright: ignore[reportIncompatibleVariableOverride]
    multipatch_num: int = Field(3, ge=1, description="Number of multi-patches")
    base_patch: int = Field(4, ge=1, description="Base patch size")
    L_patch_len: int = Field(3, ge=1, description="Patch length multiplier")
    diffusion_steps: int = Field(1000, ge=1, description="Diffusion timesteps")
    diffusion_embedding_dim: int = Field(128, ge=1, description="Diffusion embedding dimension")
    attention_mask_type: Literal["full", "parallel"] = Field(
        "full", description="Attention mask type"
    )
    noise_schedule: Literal["linear", "cosine", "quad"] = Field(
        "cosine", description="Noise schedule type"
    )
    beta_start: float | None = Field(
        None, ge=0, description="Beta schedule start (for linear/quad schedules)"
    )
    beta_end: float | None = Field(
        None, ge=0, description="Beta schedule end (for linear/quad schedules)"
    )
    is_linear: bool = Field(False, description="Use linear attention")
    attr_dim: int | None = Field(None, ge=1, description="Attribute embedding dimension override")
    side_var_emb: int | None = Field(None, ge=1, description="Side variable embedding dimension")
    side_time_emb: int | None = Field(None, ge=1, description="Side time embedding dimension")
    var_emb: int = Field(64, ge=1, description="Variable embedding dimension")
    time_emb: int = Field(64, ge=1, description="Time embedding dimension")


class DiffuSETSModelConfig(ModelConfig):
    """DiffuSETS model configuration (two-stage latent diffusion)."""

    name: Literal["diffusets"] = "diffusets"  # pyright: ignore[reportIncompatibleVariableOverride]
    latent_channels: int = Field(4, ge=1, description="Latent space channels")
    diffusion_steps: int = Field(50, ge=1, description="Diffusion timesteps")
    noise_schedule: Literal["quad", "linear", "cosine"] = Field(
        "quad", description="Noise schedule type"
    )
    beta_start: float = Field(0.0001, ge=0, description="Beta schedule start")
    beta_end: float = Field(0.5, ge=0, description="Beta schedule end")
    kernel_size: int = Field(3, ge=1, description="U-Net kernel size")
    num_levels: int = Field(5, ge=1, description="U-Net depth levels")
    kld_weight: float = Field(1.0, ge=0, description="KL divergence weight")
    kld_annealing: bool = Field(True, description="Enable KL annealing")
    kld_annealing_type: Literal["linear", "sigmoid", "cyclical"] = Field(
        "linear", description="KL annealing type"
    )
    kld_warmup_epochs: int = Field(300, ge=1, description="KL annealing warmup epochs")
    kld_start_weight: float = Field(0.01, ge=0, description="KL annealing start weight")


class CTTPModelConfig(ModelConfig):
    """CTTP model configuration (contrastive representation learning model)."""

    name: Literal["cttp"] = "cttp"  # pyright: ignore[reportIncompatibleVariableOverride]

    # Common encoder parameters
    d_model: int = Field(128, ge=1, description="Model dimension")
    coemb_dim: int = Field(256, ge=1, description="Co-embedding dimension")
    patch_len: int = Field(16, ge=1, description="Patch length")
    stride: int = Field(8, ge=1, description="Patch stride")
    padding: int = Field(0, ge=0, description="Patch padding")
    d_ff: int = Field(256, ge=1, description="Feed-forward dimension")
    e_layers: int = Field(3, ge=1, description="Encoder layers")
    factor: int = Field(5, ge=1, description="Attention factor")
    activation: Literal["relu", "gelu"] = Field("gelu", description="Activation function")

    # Text encoder parameters
    textemb_hidden_dim: int = Field(512, ge=1, description="Text embedding hidden dimension")
    pretrain_model_path: str = Field(
        "openai/clip-vit-base-patch32", description="HuggingFace model path for online encoding"
    )
    pretrain_model_dim: int = Field(512, ge=1, description="Pretrained model output dimension")

    # Loss parameters
    loss_type: Literal["contrastive", "ce", "supcon"] = Field(
        "contrastive", description="Loss type (contrastive, cross-entropy, or supcon)"
    )
    margin: float = Field(3.0, gt=0, description="Contrastive loss margin")
    temperature: float = Field(0.07, gt=0, description="Temperature for supcon/InfoNCE")
    normalize_embeddings: bool = Field(
        False, description="L2-normalize embeddings before similarity computations"
    )

    # Mode parameters
    mode: Literal["instance", "segment"] = Field(
        "instance", description="CTTP mode (instance-level or segment-level)"
    )
    text_encoding: Literal["online", "precomputed"] = Field(
        "online", description="Text encoding mode"
    )

    # Segment-level parameters (only used when mode='segment')
    segment_len: int | None = Field(
        None, description="Segment length (auto-computed if None)"
    )
    n_segments: int = Field(4, ge=1, description="Number of segments for segment mode")
    segment_loss_weight: float = Field(1.0, ge=0, description="Weight for segment-level loss (alpha)")
    global_loss_weight: float = Field(0.5, ge=0, description="Weight for global-level loss (beta)")

    # TS encoder type
    ts_encoder_type: Literal["patchtst", "patchtst_mae"] = Field(
        "patchtst_mae", description="Time series encoder type"
    )
    pretrain_encoder_path: str = Field("", description="Path to pretrained TS encoder")


class Text2MotionModelConfig(ModelConfig):
    """Text-to-Motion (CVPR'22) backbone adapted for time series generation."""

    name: Literal["text2motion"] = "text2motion"  # pyright: ignore[reportIncompatibleVariableOverride]

    # Motion (time series) latent autoencoder
    unit_length: int = Field(
        4,
        ge=1,
        description="Downsample factor used by movement conv encoder/decoder (must be 4 for this backbone).",
    )
    dim_movement_latent: int = Field(512, ge=1, description="Movement latent dimension")
    dim_movement_enc_hidden: int = Field(512, ge=1, description="Movement encoder conv hidden channels")
    dim_movement_dec_hidden: int = Field(512, ge=1, description="Movement decoder conv hidden channels")

    # Text-conditioned latent generator
    text_latent_dim: int = Field(
        1024,
        ge=1,
        description="Projected text latent dimension (typically equals condition.text.input_dim).",
    )
    dim_att_vec: int = Field(512, ge=1, description="Attention vector dimension")
    dim_z: int = Field(128, ge=1, description="Stochastic latent z dimension")

    n_layers_pri: int = Field(1, ge=1, description="GRU layers in prior network")
    n_layers_pos: int = Field(1, ge=1, description="GRU layers in posterior network")
    n_layers_dec: int = Field(1, ge=1, description="GRU layers in movement decoder")

    dim_pri_hidden: int = Field(1024, ge=1, description="Prior GRU hidden size")
    dim_pos_hidden: int = Field(1024, ge=1, description="Posterior GRU hidden size")
    dim_dec_hidden: int = Field(1024, ge=1, description="Decoder GRU hidden size")

    # Loss & training behavior
    lambda_rec_ts: float = Field(1.0, ge=0, description="Weight for time-series reconstruction loss")
    lambda_rec_mov: float = Field(1.0, ge=0, description="Weight for movement-latent reconstruction loss")
    lambda_kld: float = Field(0.01, ge=0, description="Weight for KL divergence loss")
    teacher_forcing_ratio: float = Field(
        0.9, ge=0, le=1, description="Teacher forcing ratio during training"
    )
    detach_movement_latents: bool = Field(
        True,
        description="Detach movement latents when training conditioned generator (matches original 2-stage recipe).",
    )


class TTSCGANModelConfig(ModelConfig):
    """TTS-CGAN: Transformer Time-Series Conditional GAN configuration."""

    name: Literal["ttscgan"] = "ttscgan"  # pyright: ignore[reportIncompatibleVariableOverride]

    # Latent space
    latent_dim: int = Field(100, ge=1, description="Latent noise dimension")

    # Generator architecture
    data_embed_dim: int = Field(10, ge=1, description="Data embedding dimension in generator")
    label_embed_dim: int = Field(10, ge=1, description="Label embedding dimension")
    g_depth: int = Field(3, ge=1, description="Generator Transformer depth")
    g_num_heads: int = Field(5, ge=1, description="Generator attention heads")
    g_dropout: float = Field(0.5, ge=0, le=1, description="Generator dropout rate")
    g_attn_dropout: float = Field(0.5, ge=0, le=1, description="Generator attention dropout rate")

    # Discriminator architecture
    d_patch_size: int = Field(1, ge=1, description="Discriminator patch size")
    d_embed_dim: int = Field(50, ge=1, description="Discriminator embedding dimension")
    d_depth: int = Field(3, ge=1, description="Discriminator Transformer depth")
    d_num_heads: int = Field(5, ge=1, description="Discriminator attention heads")
    d_dropout: float = Field(0.5, ge=0, le=1, description="Discriminator dropout rate")

    # Training hyperparameters
    g_lr: float = Field(0.0002, gt=0, description="Generator learning rate")
    d_lr: float = Field(0.0002, gt=0, description="Discriminator learning rate")
    beta1: float = Field(0.0, ge=0, le=1, description="Adam beta1")
    beta2: float = Field(0.9, ge=0, le=1, description="Adam beta2")
    n_critic: int = Field(1, ge=1, description="Discriminator steps per generator step")

    # Loss weights
    lambda_cls: float = Field(1.0, ge=0, description="Classification loss weight")
    lambda_gp: float = Field(10.0, ge=0, description="Gradient penalty weight")

    # EMA
    ema: float = Field(0.995, ge=0, le=1, description="EMA decay rate for generator")


# Union type for all model configs
# Note: Each model config must have a Literal type for 'name' field to work with discriminator
ModelConfigUnion = Annotated[
    VerbalTSModelConfig | BridgeModelConfig | T2SModelConfig | TimeVQVAEModelConfig | TimeWeaverModelConfig | WaveStitchModelConfig | RetrievalModelConfig | TEditModelConfig | DiffuSETSModelConfig | CTTPModelConfig | Text2MotionModelConfig | TTSCGANModelConfig,
    Field(discriminator="name"),
]


# =============================================================================
# Evaluation Configuration
# =============================================================================


class DiscAUCConfig(BaseModel):
    """Configuration for Real vs Fake Discriminator AUC metric."""

    model_config = ConfigDict(extra="forbid")

    k_folds: int = Field(5, ge=2, description="Number of cross-validation folds")
    epochs: int = Field(5, ge=1, description="Training epochs per fold")
    batch_size: int = Field(128, ge=1, description="Training batch size")
    lr: float = Field(1e-3, gt=0, description="Learning rate")
    weight_decay: float = Field(1e-4, ge=0, description="L2 regularization weight")
    hidden_channels: int = Field(128, ge=1, description="CNN hidden channels")
    num_layers: int = Field(3, ge=1, description="Number of conv layers")
    dropout: float = Field(0.1, ge=0, le=1, description="Dropout rate")
    seed: int = Field(42, ge=0, description="Random seed")
    num_workers: int = Field(0, ge=0, description="DataLoader workers")


class PRDCConfig(BaseModel):
    """Configuration for PRDC-F1 and Joint-PRDC-F1 metrics."""

    model_config = ConfigDict(extra="forbid")

    k: int = Field(5, ge=1, description="Number of nearest neighbors")
    max_samples: int | None = Field(5000, description="Maximum samples to use (None for all)")
    seed: int = Field(0, ge=0, description="Random seed")
    metric: str = Field("euclidean", description="Distance metric")
    backend: str = Field("auto", description="Backend selection (auto, faiss, torch, sklearn)")
    knn_chunk_size: int = Field(4096, ge=1, description="Chunk size for batched kNN computation")

    # Joint PRDC-F1 specific
    joint_enable: bool = Field(True, description="Enable Joint PRDC-F1 metric")
    joint_weights: dict[str, float] = Field(
        default_factory=lambda: {"ts": 1.0, "text": 1.0},
        description="Modality weights for joint embedding"
    )
    joint_normalize: str = Field("standard", description="Normalization method (standard, none)")


class SegmentClassifierConfig(BaseModel):
    """Configuration for segment-level parameter accuracy metric."""

    model_config = ConfigDict(extra="forbid")

    enable: bool = Field(False, description="Enable segment classifier metric")
    checkpoint_dir: Path | None = Field(
        None, description="Directory containing classifier checkpoints"
    )
    segment_len: int = Field(128, ge=1, description="Total sequence length")
    n_segments: int = Field(3, ge=1, description="Number of segments")
    seed: int = Field(42, ge=0, description="Random seed")


class VizConfig(BaseModel):
    """Case study visualization configuration."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    enable: bool = Field(False, description="Enable case study visualization")
    k_cases: int = Field(5, ge=1, alias="n_cases", description="Number of cases to visualize")
    max_vars: int = Field(8, ge=1, description="Max variables to display per case")
    ncols: int = Field(5, ge=1, description="Number of columns in subplot grid")
    seed: int = Field(42, ge=0, description="Random seed for case selection")
    dpi: int = Field(200, ge=50, description="Output figure DPI")
    alpha: float = Field(0.25, ge=0.0, le=1.0, description="Std band transparency")
    figsize_per_subplot: Tuple[float, float] = Field(
        (3.0, 2.0), description="Size per subplot (width, height) in inches"
    )
    output_type: Literal["png", "pdf"] = Field("png", description="Output figure format")


class EvalConfig(BaseModel):
    """Evaluation configuration."""

    model_config = ConfigDict(extra="forbid")

    # Basic settings
    n_samples: int = Field(10, ge=1, description="Number of generation samples per condition")
    metrics: list[str] = Field(
        default_factory=lambda: ["dtw", "fid", "cttp"],
        description="Metrics to compute",
    )
    batch_size: int = Field(64, ge=1, description="Evaluation batch size")
    save_samples: bool = Field(True, description="Save generated samples")
    sampler: str = Field("ddim", description="Sampling method (ddpm, ddim)")
    display_interval: int = Field(10, ge=1, description="Progress display interval")

    # CLIP configuration
    clip_config_path: Path | None = Field(
        None, description="Path to CLIP config YAML"
    )
    clip_model_path: Path | None = Field(
        None, description="Path to CLIP model checkpoint"
    )
    cache_folder: Path | None = Field(
        None, description="Path to statistics cache folder"
    )
    clip_normalize_embeddings: bool | None = Field(
        None,
        description="Override CTTP embedding normalization during eval (None=use model setting)",
    )
    use_longalign: bool = Field(False, description="Use LongAlign (multi-segment averaged) mode")

    # Output configuration
    output_folder: Path | None = Field(None, description="Output folder for results")

    # Reference statistics
    reference_split: Literal["train", "test"] = Field(
        "train", description="Split to use for reference statistics"
    )

    # Metric-specific configurations
    disc_auc: DiscAUCConfig = Field(default_factory=lambda: DiscAUCConfig())  # pyright: ignore[reportCallIssue]
    prdc: PRDCConfig = Field(default_factory=lambda: PRDCConfig())  # pyright: ignore[reportCallIssue]
    segment_classifier: SegmentClassifierConfig = Field(default_factory=lambda: SegmentClassifierConfig())  # pyright: ignore[reportCallIssue]

    # Statistical metrics configuration
    acd_max_lag: int = Field(50, ge=1, description="Maximum lag for autocorrelation")
    mdd_bins: int = Field(32, ge=2, description="Number of bins for marginal distribution")

    # Prediction caching
    use_cache: bool = Field(False, description="Use prediction cache for repeated evaluation")
    cache_file: str = Field("predictions_cache.pkl", description="Prediction cache filename")

    # Visualization
    viz: VizConfig = Field(default_factory=VizConfig, description="Case study visualization config")


# =============================================================================
# Main Experiment Configuration
# =============================================================================

class ExperimentConfig(BaseModel):
    """Complete experiment configuration."""

    model_config = ConfigDict(extra="forbid")

    # Experiment metadata
    name: str = Field(default="", description="Experiment name (auto-generated if empty)")
    description: str = Field(default="", description="Experiment description")
    seed: int = Field(42, ge=0, description="Random seed")
    device: str = Field("cuda:0", description="Device to use")

    # Sub-configurations
    train: TrainConfig = Field(default_factory=lambda: TrainConfig())  # pyright: ignore[reportCallIssue]
    data: DataConfig
    model: ModelConfig
    condition: ConditionConfig = Field(default_factory=lambda: ConditionConfig())  # pyright: ignore[reportCallIssue]
    eval: EvalConfig = Field(default_factory=lambda: EvalConfig())  # pyright: ignore[reportCallIssue]

    # Experiment management
    output_dir: Path = Field(Path("experiments"), description="Output directory")
    resume_from: Path | None = Field(None, description="Checkpoint path to resume from")

    # Timestamps
    created_at: datetime | None = Field(None, description="Creation timestamp")

    @model_validator(mode="after")
    def set_defaults(self) -> ExperimentConfig:
        """Set default values after validation."""
        # Auto-generate experiment name
        if not self.name:
            object.__setattr__(self, "name", f"{self.data.name}_{self.model.name}")

        # Set creation timestamp
        if not self.created_at:
            object.__setattr__(self, "created_at", datetime.now())

        self._auto_infer_attribute_config()
        self._auto_infer_label_config()
        return self

    def _auto_infer_attribute_config(self, infer_for_label: bool = False) -> None:
        """Auto-infer discrete attribute configs from dataset meta when missing."""
        # Only infer by default for attribute-conditioned runs.
        # Label-conditioned runs can request fallback inference explicitly.
        if not self.condition.attribute.enabled and not infer_for_label:
            return

        attr_cfg = self.condition.attribute
        if attr_cfg.discrete_configs:
            return

        data_folder = Path(self.data.data_folder)
        meta_path = data_folder / "meta.json"
        if not meta_path.exists():
            return

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        attr_n_ops = meta.get("attr_n_ops")
        if not attr_n_ops:
            attr_list = meta.get("attr_list")
            attr_value_maps = meta.get("attr_value_maps")
            if attr_list and attr_value_maps:
                attr_n_ops = [len(attr_value_maps.get(name, {})) for name in attr_list]

        if not attr_n_ops:
            return

        num_classes = [int(n) for n in attr_n_ops]
        unknown_mask = None

        for split in ("train", "valid", "test"):
            attrs_path = data_folder / f"{split}_attrs_idx.npy"
            if not attrs_path.exists():
                continue
            import numpy as np

            attrs = np.load(attrs_path, mmap_mode="r")
            if attrs.ndim == 1:
                attrs = attrs.reshape(-1, 1)
            if attrs.shape[1] != len(num_classes):
                raise ValueError(
                    f"meta attr_n_ops has {len(num_classes)} entries, but {attrs_path} has "
                    f"{attrs.shape[1]} columns"
                )
            if unknown_mask is None:
                unknown_mask = np.zeros(attrs.shape[1], dtype=bool)
            unknown_mask |= (attrs < 0).any(axis=0)

        if unknown_mask is not None:
            num_classes = [
                int(n + (1 if has_unknown else 0))
                for n, has_unknown in zip(num_classes, unknown_mask, strict=False)
            ]

        attr_cfg.discrete_configs = [{"num_classes": n} for n in num_classes]

    def _auto_infer_label_config(self) -> None:
        """Auto-infer label num_classes from dataset for label-conditioned models."""
        label_cfg = self.condition.label
        if not label_cfg.enabled:
            return

        data_folder = Path(self.data.data_folder)

        # Try to infer num_classes from train_labels.npy
        labels_path = data_folder / "train_labels.npy"
        if labels_path.exists():
            import numpy as np

            labels = np.load(labels_path)
            num_classes = int(labels.max()) + 1
            label_cfg.num_classes = num_classes
            return

        # Fallback to attribute-derived labels only when label files are unavailable.
        # This keeps label-only workflows independent from attrs/meta unless needed.
        attr_cfg = self.condition.attribute
        if not attr_cfg.discrete_configs:
            self._auto_infer_attribute_config(infer_for_label=True)

        # Use discrete_configs from attribute config (already handles -1 unknown values)
        attr_cfg = self.condition.attribute
        if attr_cfg.discrete_configs:
            num_attr_ops = [int(cfg["num_classes"]) for cfg in attr_cfg.discrete_configs]
            num_classes = infer_num_classes_from_attrs(data_folder, num_attr_ops)
            label_cfg.num_classes = num_classes
            return

        raise ValueError(
            "Label-conditioned model requires labels.npy or attribute discrete_configs "
            f"with attrs_idx files to infer num_classes in {data_folder}."
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict and handle special types
        data = self.model_dump(mode="json", exclude_none=True)

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    def model_dump_yaml(self) -> str:
        """Dump configuration as YAML string."""
        data = self.model_dump(mode="json", exclude_none=True)
        return yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)


# =============================================================================
# Dataset Presets (for easy configuration)
# =============================================================================

DATASET_PRESETS: dict[str, dict[str, Any]] = {
    "debug": {
        "n_var": 2,
        "seq_length": 32,
    },
    "synth-m": {
        "n_var": 2,
        "seq_length": 128,
    },
    "synth-u": {
        "n_var": 1,
        "seq_length": 128,
    },
    "ettm1": {
        "n_var": 1,
        "seq_length": 120,
    },
    "weather": {
        "n_var": 21,
        "seq_length": 36,
    },
    "telecomts": {
        "n_var": 16,
        "seq_length": 128,
    },
    "telecomts_segment": {
        "n_var": 2,
        "seq_length": 128,
    },
    "ptx": {
        "n_var": 12,
        "seq_length": 1000,
    },
    "blindways": {
        "n_var": 72,
        "seq_length": 600,
    },
    "istanbul_traffic": {
        "n_var": 1,
        "seq_length": 144,
    },
    "airquality_beijing": {
        "n_var": 6,
        "seq_length": 24,
    },
    "weather_concept": {
        "n_var": 10,
        "seq_length": 36,
    },
    "weather_morphology": {
        "n_var": 10,
        "seq_length": 36,
    },
    "ptbxl_concept": {
        "n_var": 12,
        "seq_length": 1000,
    },
    "ptbxl_morphology": {
        "n_var": 12,
        "seq_length": 1000,
    },
}


def get_dataset_preset(name: str) -> dict[str, Any]:
    """Get dataset preset configuration."""
    if name not in DATASET_PRESETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_PRESETS.keys())}")
    return DATASET_PRESETS[name].copy()
