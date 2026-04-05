#!/bin/bash
# ==============================================================================
# RQ4: Compositional Generalization Experiment Pipeline
#
# Uses a STRUCTURED compositional split of synth-m:
#   - 3 attributes each have a "novel value" excluded from training
#   - Test set naturally stratified by Hamming distance (1, 2, 3)
#   - Head = dist 1 (close), Tail = dist 2-3 (OOD)
#
# Usage: bash scripts/run_rq4.sh
# ==============================================================================

set -e

LOG_DIR="log/rq_experiments"
mkdir -p "$LOG_DIR"

# ==============================================================================
# Phase 0: Create Structured Compositional Split
# ==============================================================================

echo "================================================================="
echo " Phase 0: Creating Structured Compositional Split of synth-m"
echo "================================================================="

python scripts/rq4_create_compositional_split.py \
    --src ./datasets/synth-m \
    --dst ./datasets/synth-m-compo \
    --seed 42

echo ""

# ==============================================================================
# Phase 1: Train Models
# ==============================================================================

echo "================================================================="
echo " Phase 1: Training Ablation Models on synth-m-compo"
echo "================================================================="

echo "[1/2] Training PTFG-best (with cross-talk)..."
contsg train --config configs/ablation/ptfg_synth-m-compo_best.yaml \
    2>&1 | tee "$LOG_DIR/ptfg_synth-m-compo_best.log"

echo "[2/2] Training PTFG-no-cross-talk..."
contsg train --config configs/ablation/ptfg_synth-m-compo_no_cross_talk.yaml \
    2>&1 | tee "$LOG_DIR/ptfg_synth-m-compo_no_cross_talk.log"

echo ""
echo "================================================================="
echo " Phase 1 Complete. Finding experiment directories..."
echo "================================================================="

# Auto-detect the two most recent experiment directories (created today)
TODAY=$(date +%Y%m%d)
PTFG_NOCT=$(ls -td experiments/${TODAY}*synth-m*pt_factor* 2>/dev/null | head -1)
PTFG_BEST=$(ls -td experiments/${TODAY}*synth-m*pt_factor* 2>/dev/null | head -2 | tail -1)

echo "  PTFG_BEST = $PTFG_BEST"
echo "  PTFG_NOCT = $PTFG_NOCT"
echo ""

# ==============================================================================
# Phase 2: Cache Predictions (with CTTP embeddings)
# ==============================================================================

echo "================================================================="
echo " Phase 2: Caching Predictions"
echo "================================================================="

echo "[1/2] Evaluating PTFG-best..."
contsg evaluate "$PTFG_BEST" --use-cache \
    2>&1 | tee "$LOG_DIR/eval_ptfg_synth-m-compo_best.log"

echo "[2/2] Evaluating PTFG-no-cross-talk..."
contsg evaluate "$PTFG_NOCT" --use-cache \
    2>&1 | tee "$LOG_DIR/eval_ptfg_synth-m-compo_no_cross_talk.log"

echo ""

# ==============================================================================
# Phase 3: RQ4 Head-Tail Analysis
# ==============================================================================

echo "================================================================="
echo " Phase 3: RQ4 Compositional Generalization Analysis"
echo "================================================================="

python scripts/rq4_compositional_eval.py --eval \
    --data-folder ./datasets/synth-m-compo \
    --clip-config ./configs/cttp/cttp_synth-m.yaml \
    --clip-model ./checkpoints/cttp/clip_model_synth-m.pth \
    --experiments "$PTFG_BEST" "$PTFG_NOCT" \
    --labels "PTFG-best (cross-talk)" "PTFG-no-cross-talk" \
    --k 5 \
    --output results_rq4.json

echo ""
echo "================================================================="
echo " Done! Results saved to results_rq4.json"
echo "================================================================="
