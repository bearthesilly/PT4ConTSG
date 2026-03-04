# Reproducibility Submission Template

This folder provides a minimal template for model submitters to make
leaderboard numbers reproducible.

## Required Deliverables

For each new model submission, please provide:

1. `submission.example.yaml`-style metadata in `submissions/<model>.yaml`
2. Checkpoint links (recommended via Hugging Face model repo)
3. A one-click reproduction script (`reproduce.sh` style)

## Current Official Weight Scope

ConTSG official released checkpoints currently cover:

- `synth-u`
- `synth-m`

Other dataset checkpoints are not publicly released yet.

## Suggested Repository Layout (Submitter Side)

```text
your-repo/
├── scripts/
│   └── reproduce_contsg.sh
├── checkpoints/  # optional if not using remote links
└── README.md
```

## Quick Contract

The reproduction script should:

1. Download or locate the exact checkpoint used for reported numbers.
2. Run ConTSG evaluation with fixed config and seed.
3. Produce a machine-readable metric output file.

