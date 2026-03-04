# Extending ConTSG-Bench

This guide explains how to add new **models** and **datasets** to ConTSG-Bench.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Adding a New Model](#2-adding-a-new-model)
3. [Adding a New Dataset](#3-adding-a-new-dataset)
4. [Configuration Reference](#4-configuration-reference)
5. [Multi-Stage Training](#5-multi-stage-training)
6. [Tips and Best Practices](#6-tips-and-best-practices)

---

## 1. Architecture Overview

ConTSG-Bench uses a **decorator-based registry** pattern. Models, datasets, and metrics are registered via decorators and discovered automatically at runtime:

```python
from contsg.registry import Registry

@Registry.register_model("my_model")
class MyModel(BaseGeneratorModule): ...

@Registry.register_dataset("my_dataset")
class MyDataset(BaseDataModule): ...
```

Once registered, they are immediately available via the CLI:

```bash
contsg train -d my_dataset -m my_model
```

### Registry API

| Method | Description |
|--------|-------------|
| `@Registry.register_model(name, aliases=None, config_class=None)` | Register a generation model (optional model-specific schema) |
| `@Registry.register_dataset(name, aliases=None)` | Register a dataset |
| `@Registry.register_metric(name, aliases=None)` | Register an evaluation metric |
| `Registry.get_model(name)` | Look up model class by name or alias |
| `Registry.get_dataset(name)` | Look up dataset class by name or alias |
| `Registry.list_models()` | List all registered model names |
| `Registry.list_datasets()` | List all registered dataset names |
| `Registry.list_metrics()` | List all registered metric names |

---

## 2. Adding a New Model

### 2.1 Overview

To add a new model, you need to:

1. Create a file in `contsg/models/`
2. Subclass `BaseGeneratorModule` (or `BaseGANModule` for GAN-based models)
3. Implement three required methods: `_build_model()`, `forward()`, `generate()`
4. Register the model with `@Registry.register_model()` (optional: attach `config_class`)

### Model Config Validation (Hybrid)

ConTSG-Bench now uses a **hybrid schema strategy**:

- **Relaxed mode (default)**: only `model.name` and base fields are required.  
  Third-party models can be added without editing `contsg/config/schema.py`.
- **Strict mode** (`--strict-schema` or `CONTSG_STRICT_SCHEMA=1`): requires a model-specific schema.

You can optionally provide a schema when registering your model:

```python
from typing import Literal
from pydantic import Field

from contsg.config.schema import ModelConfig
from contsg.registry import Registry


class MyModelConfig(ModelConfig):
    name: Literal["my_model"] = "my_model"
    hidden_dim: int = Field(128, ge=1)


@Registry.register_model("my_model", config_class=MyModelConfig)
class MyModelModule(BaseGeneratorModule):
    ...
```

If you do **not** provide `config_class`, your model still works in relaxed mode.

### 2.2 Base Class: `BaseGeneratorModule`

```python
class BaseGeneratorModule(pl.LightningModule):
    """Base class for all generation models."""

    SUPPORTED_STAGES: Optional[List[str]] = None  # e.g., ["pretrain", "finetune"]

    def __init__(
        self,
        config: ExperimentConfig,
        use_condition: bool = True,
        learning_rate: Optional[float] = None,
        **kwargs,
    ): ...
```

**Constructor parameters:**
- `config` — Full experiment configuration (accessible as `self.config`)
- `use_condition` — Whether conditioning is used in this training stage
- `learning_rate` — Per-stage LR override (for multi-stage training)

The constructor automatically calls `_build_model()` and then `_post_init()`.

### 2.3 Required Methods

#### `_build_model(self) -> None`

Build all neural network modules. Access configuration through `self.config`:

```python
def _build_model(self) -> None:
    cfg = self.config.model       # ModelConfig (channels, layers, etc.)
    data_cfg = self.config.data   # DataConfig (seq_length, n_var, etc.)
    cond_cfg = self.config.condition  # ConditionConfig

    self.encoder = nn.TransformerEncoder(...)
    self.decoder = nn.Linear(cfg.channels, data_cfg.n_var)
```

#### `forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]`

Training forward pass. Receives a batch dictionary and returns a dict containing at least `"loss"`:

```python
def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
    ts = batch["ts"]           # (B, L, C) — time series
    cap_emb = batch["cap_emb"] # (B, D)   — text embedding
    # ... compute loss ...
    return {
        "loss": loss,          # Required: scalar tensor
        "mse_loss": mse_loss,  # Optional: logged automatically
        "kl_loss": kl_loss,    # Optional: logged automatically
    }
```

All returned values are automatically logged by the base class's `training_step()`.

#### `generate(self, condition: Tensor, n_samples: int = 1, **kwargs) -> Tensor`

Generate time series samples from a condition:

```python
def generate(
    self,
    condition: Tensor,   # (B, D) or (B, S, D) — conditioning tensor
    n_samples: int = 1,  # Number of samples per condition
    **kwargs,            # Additional params (sampler, guidance_scale, tp, etc.)
) -> Tensor:
    # ... generation logic (e.g., reverse diffusion) ...
    return samples       # (B, n_samples, L, C) or (B * n_samples, L, C)
```

### 2.4 Batch Dictionary Keys

The dataloader provides these keys in each batch:

| Key | Shape | Description | Always Present |
|-----|-------|-------------|----------------|
| `ts` | `(B, L, C)` | Time series | Yes |
| `tp` | `(B, L)` | Time positions `[0, 1, ..., L-1]` | Yes |
| `cap_emb` | `(B, D)` | Pre-computed text embedding | If text condition enabled |
| `cap` | `(B,)` | Raw caption strings | If available |
| `attrs` | `(B, A)` | Attribute indices | If attribute condition enabled |
| `label` | `(B,)` | Class label | If label condition enabled |
| `idx` | `(B,)` | Sample index | Yes |

### 2.5 Optional Overrides

| Method | Description |
|--------|-------------|
| `_post_init()` | Additional setup after `_build_model()` |
| `configure_optimizers()` | Custom optimizer/scheduler (default: AdamW + cosine) |
| `SUPPORTED_STAGES` | Class attribute listing valid training stages |

### 2.6 Available Mixins

| Mixin | Provides |
|-------|----------|
| `DiffusionMixin` | `linear_beta_schedule()`, `cosine_beta_schedule()`, `quad_beta_schedule()`, `q_sample()` |
| `LabelExtractionMixin` | `_build_attr_to_label_map()`, `_extract_labels_from_batch()` |

### 2.7 Complete Example: Text-Conditioned Diffusion Model

```python
# contsg/models/my_diffusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional

from contsg.models.base import BaseGeneratorModule, DiffusionMixin
from contsg.registry import Registry


@Registry.register_model("my_diffusion", aliases=["mydiff"])
class MyDiffusionModule(BaseGeneratorModule, DiffusionMixin):
    """Example: a simple text-conditioned diffusion model."""

    def _build_model(self) -> None:
        cfg = self.config.model
        data_cfg = self.config.data
        cond_cfg = self.config.condition

        # Condition projection
        self.cond_proj = nn.Linear(cond_cfg.text.input_dim, cfg.channels)

        # Denoising network
        self.denoiser = nn.Sequential(
            nn.Linear(data_cfg.n_var * data_cfg.seq_length + cfg.channels, cfg.channels),
            nn.GELU(),
            nn.Linear(cfg.channels, cfg.channels),
            nn.GELU(),
            nn.Linear(cfg.channels, data_cfg.n_var * data_cfg.seq_length),
        )

        # Timestep embedding
        self.time_embed = nn.Embedding(1000, cfg.channels)

        # Diffusion schedule
        self.num_steps = 1000
        betas = self.cosine_beta_schedule(self.num_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ts = batch["ts"]           # (B, L, C)
        cap_emb = batch["cap_emb"] # (B, D)
        B, L, C = ts.shape

        # Sample random timesteps
        t = torch.randint(0, self.num_steps, (B,), device=ts.device)

        # Add noise (q_sample returns a tuple: noisy_data, noise)
        noise = torch.randn_like(ts)
        noisy_ts, noise = self.q_sample(
            ts, t, noise,
            sqrt_alphas_cumprod=self.sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod,
        )

        # Predict noise
        ts_flat = noisy_ts.reshape(B, -1)              # (B, L*C)
        cond = self.cond_proj(cap_emb)                  # (B, channels)
        t_emb = self.time_embed(t)                      # (B, channels)
        pred_noise = self.denoiser(
            torch.cat([ts_flat, cond + t_emb], dim=-1)
        ).reshape(B, L, C)

        loss = F.mse_loss(pred_noise, noise)
        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        condition: torch.Tensor,
        n_samples: int = 1,
        **kwargs: Any,
    ) -> torch.Tensor:
        B = condition.shape[0]
        L = self.config.data.seq_length
        C = self.config.data.n_var
        device = condition.device
        cond = self.cond_proj(condition)                # (B, channels)

        all_samples = []
        for _ in range(n_samples):
            x = torch.randn(B, L, C, device=device)

            for t_int in reversed(range(self.num_steps)):
                t = torch.full((B,), t_int, device=device, dtype=torch.long)
                t_emb = self.time_embed(t)
                x_flat = x.reshape(B, -1)
                pred_noise = self.denoiser(
                    torch.cat([x_flat, cond + t_emb], dim=-1)
                ).reshape(B, L, C)

                # Simplified DDPM reverse step
                alpha = 1.0 - self.betas[t_int]
                alpha_bar = self.alphas_cumprod[t_int]
                x = (1.0 / alpha.sqrt()) * (
                    x - (self.betas[t_int] / (1.0 - alpha_bar).sqrt()) * pred_noise
                )
                if t_int > 0:
                    x = x + self.betas[t_int].sqrt() * torch.randn_like(x)

            all_samples.append(x)

        return torch.stack(all_samples, dim=1)  # (B, n_samples, L, C)
```

### 2.8 GAN-Based Models

For GAN-based models, inherit from `BaseGANModule` instead:

```python
from contsg.models.base import BaseGANModule

@Registry.register_model("my_gan")
class MyGANModule(BaseGANModule):
    """GAN-based generation model."""

    def _build_generator(self) -> None:
        self.generator = MyGenerator(...)

    def _build_discriminator(self) -> None:
        self.discriminator = MyDiscriminator(...)

    def generator_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # Returns {"g_loss": ..., ...}
        ...

    def discriminator_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # Returns {"d_loss": ..., ...}
        ...

    def generate(self, condition, n_samples=1, **kwargs) -> Tensor:
        ...
```

`BaseGANModule` handles alternating G/D optimization steps and optional EMA for generator weights.

### 2.9 YAML Configuration

Create a config file in `configs/generators/`:

```yaml
# configs/generators/my_diffusion_synth-m.yaml
seed: 42
device: cuda:0

data:
  name: synth-m
  data_folder: ./datasets/synth-m
  n_var: 2
  seq_length: 128

model:
  name: my_diffusion
  channels: 128
  layers: 4

condition:
  text:
    enabled: true
    input_dim: 1024

train:
  epochs: 700
  batch_size: 256
  lr: 1.0e-3
  scheduler: cosine
  early_stopping_patience: 50

eval:
  metrics: [dtw, fid, cttp]
  n_samples: 10
  save_samples: true
```

---

## 3. Adding a New Dataset

### 3.1 Standard Dataset (Recommended)

If your dataset follows the standard file format, registration is a single line:

```python
# contsg/data/datasets/standard.py (or a new file)
from contsg.data.datamodule import BaseDataModule
from contsg.registry import Registry

@Registry.register_dataset("my_dataset")
class MyDatasetDataModule(BaseDataModule):
    """Short description of the dataset."""
    pass
```

#### Expected File Structure

Place your data in `datasets/my_dataset/`:

```
datasets/my_dataset/
├── meta.json              # Dataset metadata (recommended)
├── train_ts.npy           # Training time series       — shape (N_train, L, C)
├── train_cap_emb.npy      # Training text embeddings   — shape (N_train, D)
├── train_caps.npy         # Training raw captions       — shape (N_train,)    [optional]
├── train_attrs_idx.npy    # Training attribute indices  — shape (N_train, A)  [optional]
├── train_labels.npy       # Training class labels       — shape (N_train,)    [optional]
├── valid_ts.npy           # Validation time series
├── valid_cap_emb.npy      # Validation text embeddings
├── test_ts.npy            # Test time series
└── test_cap_emb.npy       # Test text embeddings
```

**Time series shape:** `(N, L, C)` where N = number of samples, L = sequence length, C = number of variates.

**Text embeddings:** Pre-computed using [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) (dimension 1024). Use `contsg`'s precompute pipeline or generate your own.

#### `meta.json` Format

```json
{
  "name": "my_dataset",
  "n_samples": 10000,
  "seq_length": 128,
  "n_var": 2,
  "attr_list": ["trend", "volatility"],
  "attr_value_maps": {
    "trend": {"up": 0, "down": 1, "stable": 2},
    "volatility": {"low": 0, "medium": 1, "high": 2}
  },
  "attr_n_ops": [3, 3]
}
```

The `attr_n_ops` field (or `attr_list` + `attr_value_maps`) is used for automatic `discrete_configs` inference when training attribute-conditioned models. If `attr_n_ops` is not present, the framework falls back to computing it from `attr_value_maps`.

### 3.2 Custom Dataset

For non-standard data formats, override `_create_dataset()`:

```python
import json
import numpy as np
import torch
from pathlib import Path
from typing import Any, Optional

from torch.utils.data import Dataset
from contsg.data.datamodule import BaseDataModule
from contsg.registry import Registry


class MyCustomDataset(Dataset):
    """Custom dataset that loads data from a non-standard format."""

    def __init__(self, data_folder: Path, split: str = "train"):
        self.data_folder = Path(data_folder)
        self.split = split
        self._load_data()

    def _load_data(self) -> None:
        prefix = {"train": "train", "valid": "valid", "test": "test"}[self.split]

        # Load time series (required)
        self.ts = np.load(self.data_folder / f"{prefix}_ts.npy")

        # Load text embeddings (required for text-conditioned models)
        cap_emb_path = self.data_folder / f"{prefix}_cap_emb.npy"
        self.cap_emb = np.load(cap_emb_path) if cap_emb_path.exists() else None

        # Load any custom data
        custom_path = self.data_folder / f"{prefix}_custom.json"
        if custom_path.exists():
            with open(custom_path) as f:
                self.custom_data = json.load(f)

    def __len__(self) -> int:
        return len(self.ts)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "ts": torch.from_numpy(self.ts[idx]).float(),      # (L, C)
            "tp": torch.arange(self.ts.shape[1]).float(),       # (L,)
            "idx": idx,
        }
        if self.cap_emb is not None:
            item["cap_emb"] = torch.from_numpy(self.cap_emb[idx]).float()
        return item


@Registry.register_dataset("my_custom")
class MyCustomDataModule(BaseDataModule):
    """Dataset with custom loading logic."""

    def _create_dataset(self, split: str) -> Dataset:
        return MyCustomDataset(
            data_folder=self.config.data_folder,
            split=split,
        )
```

### 3.3 Text Embedding Precomputation

If your dataset has raw text captions but not pre-computed embeddings, use the precompute pipeline:

```python
from pathlib import Path
from contsg.data.precompute.dataset import precompute_dataset_embeddings
from contsg.data.precompute.sentence_transformer import SentenceTransformerPrecomputer

precomputer = SentenceTransformerPrecomputer(
    model_path="Qwen/Qwen3-Embedding-0.6B",
    embed_dim=1024,   # default dimension for Qwen3-Embedding-0.6B
)
precompute_dataset_embeddings(
    dataset_dir=Path("datasets/my_dataset"),
    precomputer=precomputer,
    splits=["train", "valid", "test"],
)
```

This generates `{split}_cap_emb.npy` files from `{split}_caps.npy`.

---

## 4. Configuration Reference

### 4.1 Condition Types

ConTSG-Bench supports three conditioning modalities. Enable them in the config:

#### Text Conditioning

```yaml
condition:
  text:
    enabled: true
    input_dim: 1024            # Qwen3-Embedding-0.6B dimension
    embedding_key: cap_emb     # Batch key for embeddings
```

Used by: `verbalts`, `t2s`, `bridge`, `diffusets`, `text2motion`, `retrieval`

#### Attribute Conditioning

```yaml
condition:
  attribute:
    enabled: true
    discrete_configs:          # Auto-inferred from meta.json if omitted
      - { num_classes: 4, embed_dim: 32 }
      - { num_classes: 3, embed_dim: 32 }
    output_dim: 128
```

Used by: `timeweaver`, `wavestitch`, `tedit`

#### Label Conditioning

```yaml
condition:
  label:
    enabled: true
    num_classes: 10            # Auto-inferred from data if omitted
    output_dim: 64
```

Used by: `timevqvae`, `ttscgan`

### 4.2 Key Configuration Fields

#### `ExperimentConfig` (Top-Level)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `seed` | `int` | `42` | Random seed |
| `device` | `str` | `"cuda:0"` | Training device |
| `output_dir` | `Path` | `"experiments"` | Experiment output directory |

#### `DataConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | — | Dataset registry name |
| `data_folder` | `Path` | — | Path to dataset folder |
| `n_var` | `int` | — | Number of variates |
| `seq_length` | `int` | `128` | Sequence length |
| `normalize` | `bool` | `True` | Normalize time series |

#### `TrainConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `epochs` | `int` | `700` | Training epochs |
| `batch_size` | `int` | `256` | Batch size |
| `lr` | `float` | `1e-3` | Learning rate |
| `scheduler` | `str` | `"cosine"` | LR scheduler (`cosine`, `step`, `plateau`, `none`) |
| `early_stopping_patience` | `int` | `50` | Early stopping patience |
| `gradient_clip_val` | `float` | `1.0` | Gradient clipping |
| `num_workers` | `int` | `4` | DataLoader workers |

#### `EvalConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `metrics` | `list[str]` | `["dtw", "fid", "cttp"]` | Evaluation metrics |
| `n_samples` | `int` | `10` | Samples per condition |
| `save_samples` | `bool` | `True` | Save generated samples |

---

## 5. Multi-Stage Training

Some models require two-stage training (e.g., pretrain an autoencoder, then train a generator in the latent space).

### 5.1 Configuration

Use `stages_preset` for common patterns:

```yaml
train:
  stages_preset: two_stage     # Shortcut for pretrain → finetune
```

Or define stages explicitly:

```yaml
train:
  stages:
    - name: pretrain
      epochs: 200
      lr: 1.0e-3
      use_condition: false       # Stage 1 often does not use conditions
    - name: finetune
      epochs: 500
      lr: 5.0e-4
      use_condition: true
      load_from_stage: pretrain  # Load weights from stage 1
      freeze_modules: [encoder]  # Optionally freeze modules
```

### 5.2 Model Implementation

Set `SUPPORTED_STAGES` and branch logic per stage:

```python
@Registry.register_model("my_two_stage")
class MyTwoStageModule(BaseGeneratorModule):
    SUPPORTED_STAGES = ["pretrain", "finetune"]

    def __init__(self, config, use_condition=True, learning_rate=None,
                 current_stage="finetune", **kwargs):
        self._current_stage = current_stage
        super().__init__(config, use_condition, learning_rate, **kwargs)

    def _build_model(self) -> None:
        self.encoder = nn.Linear(...)
        self.decoder = nn.Linear(...)
        self.generator = nn.Linear(...)  # Only used in finetune

    def set_stage(self, stage: str) -> None:
        """Called by MultiStageTrainer to set the current stage."""
        self._current_stage = stage

    def forward(self, batch):
        if self._current_stage == "pretrain":
            # Autoencoder reconstruction
            z = self.encoder(batch["ts"])
            recon = self.decoder(z)
            return {"loss": F.mse_loss(recon, batch["ts"])}
        else:
            # Latent generation
            z = self.encoder(batch["ts"]).detach()
            z_gen = self.generator(batch["cap_emb"])
            return {"loss": F.mse_loss(z_gen, z)}
```

### 5.3 Existing Multi-Stage Models

| Model | Stage 1 | Stage 2 | Weight Transfer |
|-------|---------|---------|-----------------|
| `timevqvae` | `pretrain` (VQ-VAE) | `finetune` (MaskGIT) | `load_from_stage: pretrain` |
| `diffusets` | `vae_pretrain` (VAE) | `finetune` (Latent DDPM) | `load_from_stage: vae_pretrain` |
| `t2s` | `ae_pretrain` (AE) | `finetune` (Flow Matching) | `load_from_stage: ae_pretrain` |
| `text2motion` | `pretrain` (Movement AE) | `finetune` (Latent VAE) | `load_from_stage: pretrain` |

---

## 6. Tips and Best Practices

### Smoke Testing

Use the `--smoke` flag to quickly validate your implementation:

```bash
# With in-memory debug data (no data files needed)
contsg train -d debug -m my_model --smoke

# With a real dataset (minimal batches)
contsg train -d synth-m -m my_model --smoke
```

`--smoke` uses tiny batch sizes, minimal train/val/test batches, and a lightweight metric (`ed`) so the full pipeline completes quickly.

### Development Workflow

1. **Start with `debug` dataset** — No data files needed, validates the full train → eval pipeline
2. **Check batch keys** — Print `batch.keys()` in `forward()` to verify available data
3. **Return informative losses** — Return sub-losses (e.g., `"mse_loss"`, `"kl_loss"`) for monitoring
4. **Test generation shape** — Ensure `generate()` output shape is `(B, n_samples, L, C)`

### Condition Type Matching

Make sure your model's expected condition matches the config:

| Your model uses | Enable in config | Available batch key |
|-----------------|-----------------|---------------------|
| Text embeddings | `condition.text.enabled: true` | `batch["cap_emb"]` |
| Attributes | `condition.attribute.enabled: true` | `batch["attrs"]` |
| Labels | `condition.label.enabled: true` | `batch["label"]` |

### File Placement

- **Model code** → `contsg/models/my_model.py`
- **Model sub-modules** → `contsg/models/my_model_modules/` (for complex architectures)
- **Dataset code** → `contsg/data/datasets/my_dataset.py`
- **Config file** → `configs/generators/my_model_datasetname.yaml`

Models and datasets are auto-discovered — no manual import registration is needed.
