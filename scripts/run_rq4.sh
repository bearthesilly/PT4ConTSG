#!/bin/bash
# ==============================================================================
# RQ4: Compositional Generalization Experiment Pipeline
#
# Methodology (aligned with ConTSG paper):
#   - k-NN average Hamming distance for Head/Tail split
#   - CTTP retrieval accuracy (not classifier-based)
#   - Acc_norm = Acc_gen / Acc_ref
#
# Pipeline:
#   Phase 1: Train ablation models (best vs no-cross-talk)
#   Phase 2: Generate & cache predictions (with CTTP embeddings)
#   Phase 3: Run RQ4 analysis
#
# Usage: bash scripts/run_rq4.sh
# ==============================================================================

set -e

LOG_DIR="log/rq_experiments"
mkdir -p "$LOG_DIR"

# ==============================================================================
# Phase 1: Train Models
# ==============================================================================

echo "================================================================="
echo " Phase 1: Training Ablation Models for RQ4"
echo "================================================================="

echo "[1/2] Training PTFG-best on synth-m..."
contsg train --config configs/ablation/ptfg_synth-m_best.yaml \
    2>&1 | tee "$LOG_DIR/ptfg_synth-m_best.log"

echo "[2/2] Training PTFG-no-cross-talk on synth-m..."
contsg train --config configs/ablation/ptfg_synth-m_no_cross_talk.yaml \
    2>&1 | tee "$LOG_DIR/ptfg_synth-m_no_cross_talk.log"

echo ""
echo "================================================================="
echo " Phase 1 Complete."
echo "================================================================="
echo ""
echo "Next steps:"
echo "  1. Find experiment directories:"
echo "     ls -td experiments/*synth-m*pt_factor* | head -2"
echo ""
echo "  2. Set these variables and run Phase 2-3 below:"
echo "     PTFG_BEST_M=experiments/YYYYMMDD_..._synth-m_pt_factor_generator"
echo "     PTFG_NOCT_M=experiments/YYYYMMDD_..._synth-m_pt_factor_generator"
echo ""

# ==============================================================================
# Phase 2-3: Fill in experiment paths after Phase 1 completes
# ==============================================================================

# Uncomment and fill in after Phase 1:

# PTFG_BEST_M="experiments/YYYYMMDD_..._synth-m_pt_factor_generator"
# PTFG_NOCT_M="experiments/YYYYMMDD_..._synth-m_pt_factor_generator"

# --- Phase 2: Cache predictions (CTTP embeddings included automatically) ---
# contsg evaluate "$PTFG_BEST_M" --use-cache
# contsg evaluate "$PTFG_NOCT_M" --use-cache

# --- Phase 3: Run RQ4 analysis ---
# python scripts/rq4_compositional_eval.py --eval \
#     --data-folder ./datasets/synth-m \
#     --clip-config ./configs/cttp/cttp_synth-m.yaml \
#     --clip-model ./checkpoints/cttp/clip_model_synth-m.pth \
#     --experiments "$PTFG_BEST_M" "$PTFG_NOCT_M" \
#     --labels "PTFG-best" "PTFG-no-cross-talk" \
#     --k 5 \
#     --output results_rq4.json
