#!/usr/bin/env bash
set -euo pipefail

# Example one-click reproducibility script for ConTSG submissions.
# Submitters should adapt download URLs, config path, and output path.

DATASET=""
SEED="0"
EXP_DIR=""
CHECKPOINT=""
OUT_JSON=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --exp-dir) EXP_DIR="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --out-json) OUT_JSON="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$DATASET" || -z "$EXP_DIR" || -z "$CHECKPOINT" || -z "$OUT_JSON" ]]; then
  cat <<'USAGE'
Usage:
  bash scripts/reproduce_contsg.sh \
    --dataset synth-u \
    --seed 0 \
    --exp-dir ./artifacts/my_model/synth-u/seed0 \
    --checkpoint ./artifacts/my_model/synth-u/seed0/checkpoints/finetune/best.ckpt \
    --out-json ./artifacts/my_model/synth-u/seed0/results/eval_results.json
USAGE
  exit 2
fi

echo "[INFO] dataset=$DATASET seed=$SEED"
echo "[INFO] exp_dir=$EXP_DIR"
echo "[INFO] checkpoint=$CHECKPOINT"

# Optional: download checkpoint from a remote registry before evaluation.
# Example:
# huggingface-cli download <repo_id> <path/in/repo> --local-dir "$(dirname "$CHECKPOINT")"

# Ensure summary points to the selected checkpoint for `--checkpoint best`.
mkdir -p "$EXP_DIR"
cat > "$EXP_DIR/summary.json" <<EOF
{
  "status": "completed",
  "best_checkpoint": "$CHECKPOINT"
}
EOF

# Run ConTSG evaluation.
# You can switch `--checkpoint best` to `--checkpoint finetune/best.ckpt`
# depending on your experiment folder layout.
contsg evaluate "$EXP_DIR" --checkpoint best

# Normalize output path for downstream verification.
mkdir -p "$(dirname "$OUT_JSON")"
cp "$EXP_DIR/results/eval_results.json" "$OUT_JSON"
echo "[INFO] wrote $OUT_JSON"

