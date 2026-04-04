#!/bin/bash
# ==============================================================================
# RQ3 + RQ4 Experiment Launcher
#
# This script runs the full pipeline:
#   Phase 1: Train ablation models (best, no-patch-mod, no-cross-talk)
#   Phase 2: Generate & cache predictions for each model
#   Phase 3: Train evaluation classifiers
#   Phase 4: Run RQ3 and RQ4 analysis scripts
#
# Usage: bash scripts/run_rq3_rq4.sh
# ==============================================================================

set -e

LOG_DIR="log/rq_experiments"
mkdir -p "$LOG_DIR"

echo "================================================================="
echo " Phase 1: Training Ablation Models"
echo "================================================================="

# --- RQ3: PTFG-best on synth-u (with segment_param_acc enabled) ---
echo "[1/4] Training PTFG-best on synth-u..."
contsg train --config configs/ablation/ptfg_synth-u_best.yaml \
    2>&1 | tee "$LOG_DIR/ptfg_synth-u_best.log"

# --- RQ3: PTFG-no-patch-mod on synth-u ---
echo "[2/4] Training PTFG-no-patch-mod on synth-u..."
contsg train --config configs/ablation/ptfg_synth-u_no_patch_mod.yaml \
    2>&1 | tee "$LOG_DIR/ptfg_synth-u_no_patch_mod.log"

# --- RQ4: PTFG-best on synth-m ---
echo "[3/4] Training PTFG-best on synth-m..."
contsg train --config configs/ablation/ptfg_synth-m_best.yaml \
    2>&1 | tee "$LOG_DIR/ptfg_synth-m_best.log"

# --- RQ4: PTFG-no-cross-talk on synth-m ---
echo "[4/4] Training PTFG-no-cross-talk on synth-m..."
contsg train --config configs/ablation/ptfg_synth-m_no_cross_talk.yaml \
    2>&1 | tee "$LOG_DIR/ptfg_synth-m_no_cross_talk.log"

echo ""
echo "================================================================="
echo " Phase 1 Complete. Check logs in $LOG_DIR/"
echo "================================================================="
echo ""
echo "After training, fill in the experiment directories below and"
echo "run Phase 2-4 manually (or uncomment the lines below)."
echo ""

# ==============================================================================
# Phase 2-4: Fill in experiment paths after Phase 1 completes
# ==============================================================================

# After Phase 1, find the experiment directories:
#   ls -td experiments/*synth-u*pt_factor* | head -2
#   ls -td experiments/*synth-m*pt_factor* | head -2
#
# Then set these variables:
# PTFG_BEST_U="experiments/YYYYMMDD_..._synth-u_pt_factor_generator"
# PTFG_NOMOD_U="experiments/YYYYMMDD_..._synth-u_pt_factor_generator"
# PTFG_BEST_M="experiments/YYYYMMDD_..._synth-m_pt_factor_generator"
# PTFG_NOCT_M="experiments/YYYYMMDD_..._synth-m_pt_factor_generator"

# --- Phase 2: Cache predictions (needed for RQ3/RQ4 scripts) ---
# contsg evaluate "$PTFG_BEST_U" --use-cache
# contsg evaluate "$PTFG_NOMOD_U" --use-cache
# contsg evaluate "$PTFG_BEST_M" --use-cache
# contsg evaluate "$PTFG_NOCT_M" --use-cache

# --- Phase 3: Train evaluation classifiers ---
# python scripts/rq3_segment_eval.py --train-classifiers \
#     --data-folder ./datasets/synth-u --epochs 100
# python scripts/rq4_compositional_eval.py --train-classifier \
#     --data-folder ./datasets/synth-m --epochs 100

# --- Phase 4: Run analysis ---
# python scripts/rq3_segment_eval.py --eval --retrieval-baseline \
#     --data-folder ./datasets/synth-u \
#     --experiments "$PTFG_BEST_U" "$PTFG_NOMOD_U" \
#     --labels "PTFG-best" "PTFG-no-patch-mod" \
#     --output results_rq3.json

# python scripts/rq4_compositional_eval.py --eval \
#     --data-folder ./datasets/synth-m \
#     --experiments "$PTFG_BEST_M" "$PTFG_NOCT_M" \
#     --labels "PTFG-best" "PTFG-no-cross-talk" \
#     --output results_rq4.json
