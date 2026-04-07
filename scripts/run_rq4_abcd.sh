#!/bin/bash
# ==============================================================================
# RQ4 ABCD: Compositional Generalization -- Full Ablation Suite
#
# Four experiments isolating the effect of condition modality & cross-talk:
#   A: Attr-only + cross-talk ON    (structured, entangled)
#   B: Attr-only + cross-talk OFF   (structured, modular)
#   C: Text-only                    (unstructured baseline)
#   D: Text+Attr fusion + cross-talk ON  (both modalities, V2-14)
#
# Key comparison:
#   A vs B  -> Does cross-talk help or hurt compositional generalization?
#   A vs C  -> Structured attr vs unstructured text for compositionality?
#   D vs A  -> Does adding text on top of attr help?
#   D vs C  -> Does adding attr on top of text help?
#
# Usage: bash scripts/run_rq4_abcd.sh [--skip-data] [--skip-train] [--skip-eval]
# ==============================================================================

set -e

SKIP_DATA=false
SKIP_TRAIN=false
SKIP_EVAL=false

for arg in "$@"; do
    case "$arg" in
        --skip-data)  SKIP_DATA=true ;;
        --skip-train) SKIP_TRAIN=true ;;
        --skip-eval)  SKIP_EVAL=true ;;
    esac
done

LOG_DIR="log/rq4_abcd"
mkdir -p "$LOG_DIR"

# ==============================================================================
# Phase 0: Create Structured Compositional Split (if needed)
# ==============================================================================

if [ "$SKIP_DATA" = false ]; then
    echo "================================================================="
    echo " Phase 0: Creating Structured Compositional Split of synth-m"
    echo "================================================================="

    python scripts/rq4_create_compositional_split.py \
        --src ./datasets/synth-m \
        --dst ./datasets/synth-m-compo \
        --seed 42

    echo ""
fi

# ==============================================================================
# Phase 1: Train 4 Models
# ==============================================================================

if [ "$SKIP_TRAIN" = false ]; then
    echo "================================================================="
    echo " Phase 1: Training 4 Ablation Models (A/B/C/D)"
    echo "================================================================="

    echo ""
    echo "[A] Attr-only + cross-talk ON ..."
    contsg train --config configs/ablation/rq4_A_attr_crosstalk.yaml \
        2>&1 | tee "$LOG_DIR/train_A_attr_crosstalk.log"

    echo ""
    echo "[B] Attr-only + cross-talk OFF ..."
    contsg train --config configs/ablation/rq4_B_attr_no_crosstalk.yaml \
        2>&1 | tee "$LOG_DIR/train_B_attr_no_crosstalk.log"

    echo ""
    echo "[C] Text-only ..."
    contsg train --config configs/ablation/rq4_C_text_only.yaml \
        2>&1 | tee "$LOG_DIR/train_C_text_only.log"

    echo ""
    echo "[D] Text+Attr fusion ..."
    contsg train --config configs/ablation/rq4_D_text_attr_fusion.yaml \
        2>&1 | tee "$LOG_DIR/train_D_text_attr_fusion.log"

    echo ""
    echo "Phase 1 Complete."
    echo ""
fi

# ==============================================================================
# Phase 2: Detect experiment directories & cache predictions
# ==============================================================================

echo "================================================================="
echo " Phase 2: Detecting experiments & caching predictions"
echo "================================================================="

# Auto-detect the 4 most recent experiment directories (created today)
TODAY=$(date +%Y%m%d)
ALL_EXPS=($(ls -td experiments/${TODAY}*synth-m*pt_factor* 2>/dev/null | head -4))

if [ ${#ALL_EXPS[@]} -lt 4 ]; then
    echo "WARNING: Found only ${#ALL_EXPS[@]} experiments from today."
    echo "Expected 4. Listing all found:"
    for e in "${ALL_EXPS[@]}"; do echo "  $e"; done
    echo ""
    echo "If you ran experiments on a different day, set directories manually:"
    echo "  EXP_A=... EXP_B=... EXP_C=... EXP_D=... bash scripts/run_rq4_abcd.sh --skip-data --skip-train"
    if [ -z "${EXP_A}" ]; then
        echo "Aborting."
        exit 1
    fi
fi

# Experiments are listed newest-first; training order was A, B, C, D
# So D is newest (index 0), A is oldest (index 3)
EXP_D="${EXP_D:-${ALL_EXPS[0]}}"
EXP_C="${EXP_C:-${ALL_EXPS[1]}}"
EXP_B="${EXP_B:-${ALL_EXPS[2]}}"
EXP_A="${EXP_A:-${ALL_EXPS[3]}}"

echo "  EXP_A (attr+crosstalk)   = $EXP_A"
echo "  EXP_B (attr-no-crosstalk)= $EXP_B"
echo "  EXP_C (text-only)        = $EXP_C"
echo "  EXP_D (text+attr fusion) = $EXP_D"
echo ""

if [ "$SKIP_EVAL" = false ]; then
    echo "[A] Evaluating attr+crosstalk..."
    contsg evaluate "$EXP_A" --use-cache \
        2>&1 | tee "$LOG_DIR/eval_A_attr_crosstalk.log"

    echo "[B] Evaluating attr-no-crosstalk..."
    contsg evaluate "$EXP_B" --use-cache \
        2>&1 | tee "$LOG_DIR/eval_B_attr_no_crosstalk.log"

    echo "[C] Evaluating text-only..."
    contsg evaluate "$EXP_C" --use-cache \
        2>&1 | tee "$LOG_DIR/eval_C_text_only.log"

    echo "[D] Evaluating text+attr fusion..."
    contsg evaluate "$EXP_D" --use-cache \
        2>&1 | tee "$LOG_DIR/eval_D_text_attr_fusion.log"

    echo ""
fi

# ==============================================================================
# Phase 3: RQ4 Head-Tail Analysis (all 4 models)
# ==============================================================================

echo "================================================================="
echo " Phase 3: RQ4 Compositional Generalization Analysis (ABCD)"
echo "================================================================="

python scripts/rq4_compositional_eval.py --eval \
    --data-folder ./datasets/synth-m-compo \
    --clip-config ./configs/cttp/cttp_synth-m.yaml \
    --clip-model ./checkpoints/cttp/clip_model_synth-m.pth \
    --experiments "$EXP_A" "$EXP_B" "$EXP_C" "$EXP_D" \
    --labels \
        "A: Attr+CrossTalk" \
        "B: Attr-NoCrossTalk" \
        "C: Text-Only" \
        "D: Text+Attr Fusion" \
    --k 5 \
    --output results_rq4_abcd.json

echo ""
echo "================================================================="
echo " Done! Results saved to results_rq4_abcd.json"
echo " Logs in $LOG_DIR/"
echo "================================================================="
