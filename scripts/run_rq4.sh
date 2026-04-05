#!/bin/bash
# ==============================================================================
# RQ4: Compositional Generalization Experiment Pipeline
#
# Uses a compositional split of synth-m where ~25% of attribute combinations
# are held out for testing, ensuring meaningful Head-Tail analysis.
#
# Pipeline:
#   Phase 0: Create compositional split dataset
#   Phase 1: Train ablation models (best vs no-cross-talk)
#   Phase 2: Generate & cache predictions (with CTTP embeddings)
#   Phase 3: Run RQ4 Head-Tail analysis
#
# Usage: bash scripts/run_rq4.sh
# ==============================================================================

set -e

LOG_DIR="log/rq_experiments"
mkdir -p "$LOG_DIR"

# ==============================================================================
# Phase 0: Create Compositional Split
# ==============================================================================

echo "================================================================="
echo " Phase 0: Creating Compositional Split of synth-m"
echo "================================================================="

python scripts/rq4_create_compositional_split.py \
    --src ./datasets/synth-m \
    --dst ./datasets/synth-m-compo \
    --holdout-ratio 0.25 \
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
echo " Phase 1 Complete. Now find experiment directories:"
echo "================================================================="
echo ""
echo "  ls -td experiments/*synth-m*pt_factor* | head -2"
echo ""
echo "  Then set these and run Phase 2-3:"
echo ""

# ==============================================================================
# Phase 2-3: Uncomment after Phase 1, fill in experiment paths
# ==============================================================================

# PTFG_BEST="experiments/YYYYMMDD_..._synth-m_pt_factor_generator"
# PTFG_NOCT="experiments/YYYYMMDD_..._synth-m_pt_factor_generator"

# --- Phase 2: Cache predictions ---
# contsg evaluate "$PTFG_BEST" --use-cache
# contsg evaluate "$PTFG_NOCT" --use-cache

# --- Phase 3: RQ4 analysis ---
# python scripts/rq4_compositional_eval.py --eval \
#     --data-folder ./datasets/synth-m-compo \
#     --clip-config ./configs/cttp/cttp_synth-m.yaml \
#     --clip-model ./checkpoints/cttp/clip_model_synth-m.pth \
#     --experiments "$PTFG_BEST" "$PTFG_NOCT" \
#     --labels "PTFG-best (cross-talk)" "PTFG-no-cross-talk" \
#     --k 5 \
#     --output results_rq4.json
