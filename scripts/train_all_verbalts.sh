#!/usr/bin/env bash
# Train VerbalTS on all benchmark datasets with the same data paths, train splits,
# and CTTP checkpoint paths as configs/generators/ptfg_*.yaml.
#
# Usage (from repo root):
#   bash scripts/train_all_verbalts.sh
#
# Logs default to log/verbalts_runs/<config_stem>.log. Override with LOGDIR.
# To continue after a failed run: CONTINUE_ON_ERROR=1 bash scripts/train_all_verbalts.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export LOGDIR="${LOGDIR:-$ROOT/log/verbalts_runs}"
mkdir -p "$LOGDIR"

CONFIGS=(
  configs/generators/verbalts_synth-m.yaml
  configs/generators/verbalts_synth-u.yaml
  configs/generators/verbalts_ettm1.yaml
  configs/generators/verbalts_airquality_beijing.yaml
  configs/generators/verbalts_istanbul_traffic.yaml
  configs/generators/verbalts_telecomts.yaml
  configs/generators/verbalts_ptb_concept.yaml
  configs/generators/verbalts_ptb_morphology.yaml
  configs/generators/verbalts_weather_concept.yaml
  configs/generators/verbalts_weather_morphology.yaml
)

CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"

for cfg in "${CONFIGS[@]}"; do
  tag="$(basename "$cfg" .yaml)"
  echo "========================================"
  echo "Training: $tag"
  echo "========================================"
  set +o pipefail
  contsg train --config "$cfg" 2>&1 | tee "$LOGDIR/${tag}.log"
  status="${PIPESTATUS[0]}"
  set -o pipefail
  if [[ "$status" -ne 0 ]]; then
    echo "FAILED: $tag (exit $status)" >&2
    if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
      exit "$status"
    fi
  fi
done
echo "All training jobs finished."
