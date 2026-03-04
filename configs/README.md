# Configuration Files

This directory contains configuration files for training and evaluation.

## Directory Structure

```
configs/
├── cttp/           # CTTP (Contrastive Text-Time Series Pre-training) model configs
├── generators/     # Generator model configs (Bridge, DiffuSETS, T2S, etc.)
└── README.md
```

## Placeholder Paths

Some configuration files contain placeholder paths that need to be replaced with your actual paths before use:

### 1. `pretrain_model_path` (in CTTP configs)

```yaml
pretrain_model_path: ./checkpoints/longclip
```

This should point to your LongCLIP pretrained model directory. Download from the official LongCLIP repository and update this path.

### 2. `clip_config_path` (in Generator configs)

```yaml
clip_config_path: ./configs/cttp/cttp_<dataset>_<mode>.yaml
```

Points to the corresponding CTTP config file. Some configs already provide a concrete path,
while others keep `<CTTP_CONFIG>` as an explicit placeholder.

### 3. `clip_model_path` (in Generator configs)

```yaml
clip_model_path: ./checkpoints/cttp/<CTTP_CHECKPOINT>.ckpt
```

**Replace `<CTTP_CHECKPOINT>` with your trained CTTP model checkpoint filename.**

Example:
```yaml
clip_model_path: ./checkpoints/cttp/cttp_ettm1_instance_best.ckpt
```

## Dataset Naming Convention

| Dataset Name | Description |
|-------------|-------------|
| `ettm1` | ETT-m1 time series dataset |
| `istanbul_traffic` | Istanbul traffic flow dataset |
| `ptbxl_concept` | PTB-XL ECG dataset (concept-level descriptions) |
| `ptbxl_morphology` | PTB-XL ECG dataset (morphology-level descriptions) |
| `weather_concept` | Weather dataset (concept-level descriptions) |
| `weather_morphology` | Weather dataset (morphology-level descriptions) |
| `telecomts_segment` | Telecom time series dataset (segment-level) |
| `airquality_beijing` | Beijing air quality dataset |

CTTP naming convention: use `cttp_<dataset>.yaml` by default. Only `telecomts_segment`
keeps two explicit modes: `cttp_telecomts_instance.yaml` and
`cttp_telecomts_segment.yaml`.

## Quick Start

1. **Run a smoke test first (no data files needed):**

   ```bash
   contsg train -d debug -m verbalts --smoke
   ```

2. **Prepare datasets**: Place your datasets in `./datasets/<dataset_name>/`

3. **Download LongCLIP**: Download pretrained LongCLIP weights to `./checkpoints/longclip/`

4. **Train CTTP model**:
   ```bash
   contsg train --config configs/cttp/cttp_ettm1.yaml
   ```

5. **Update generator config**:
   - Replace `clip_model_path: ./checkpoints/cttp/<CTTP_CHECKPOINT>.ckpt`
   - If present, replace `clip_config_path: ./configs/cttp/<CTTP_CONFIG>.yaml`

6. **Train generator model**:
   ```bash
   contsg train --config configs/generators/bridge_ettm1.yaml
   ```
