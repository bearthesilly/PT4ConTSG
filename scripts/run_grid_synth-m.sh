#!/usr/bin/env bash
set -euo pipefail

# Grid search for PTFG V2 on synth-m
# Search space: lr=[1e-3,1e-4] x batch_size=[32,64,128,256] x self_cond=[T,F]
# Total: 16 combinations

LOGDIR="log/grid_search"
mkdir -p $LOGDIR

# [1/16] lr1e-3_bs32_scT
echo "[1/16] Running: lr1e-3_bs32_scT"
contsg train --config configs/grid_search/ptfg_synth-m_lr1e-3_bs32_scT.yaml 2>&1 | tee $LOGDIR/ptfg_synth-m_lr1e-3_bs32_scT.log

# [2/16] lr1e-3_bs32_scF
echo "[2/16] Running: lr1e-3_bs32_scF"
contsg train --config configs/grid_search/ptfg_synth-m_lr1e-3_bs32_scF.yaml 2>&1 | tee $LOGDIR/ptfg_synth-m_lr1e-3_bs32_scF.log

# [3/16] lr1e-4_bs32_scT
echo "[3/16] Running: lr1e-4_bs32_scT"
contsg train --config configs/grid_search/ptfg_synth-m_lr1e-4_bs32_scT.yaml 2>&1 | tee $LOGDIR/ptfg_synth-m_lr1e-4_bs32_scT.log

# [4/16] lr1e-4_bs32_scF
echo "[4/16] Running: lr1e-4_bs32_scF"
contsg train --config configs/grid_search/ptfg_synth-m_lr1e-4_bs32_scF.yaml 2>&1 | tee $LOGDIR/ptfg_synth-m_lr1e-4_bs32_scF.log

# [5/16] lr1e-3_bs64_scT
echo "[5/16] Running: lr1e-3_bs64_scT"
contsg train --config configs/grid_search/ptfg_synth-m_lr1e-3_bs64_scT.yaml 2>&1 | tee $LOGDIR/ptfg_synth-m_lr1e-3_bs64_scT.log

# [6/16] lr1e-3_bs64_scF
echo "[6/16] Running: lr1e-3_bs64_scF"
contsg train --config configs/grid_search/ptfg_synth-m_lr1e-3_bs64_scF.yaml 2>&1 | tee $LOGDIR/ptfg_synth-m_lr1e-3_bs64_scF.log

# [7/16] lr1e-4_bs64_scT
echo "[7/16] Running: lr1e-4_bs64_scT"
contsg train --config configs/grid_search/ptfg_synth-m_lr1e-4_bs64_scT.yaml 2>&1 | tee $LOGDIR/ptfg_synth-m_lr1e-4_bs64_scT.log

# [8/16] lr1e-4_bs64_scF
echo "[8/16] Running: lr1e-4_bs64_scF"
contsg train --config configs/grid_search/ptfg_synth-m_lr1e-4_bs64_scF.yaml 2>&1 | tee $LOGDIR/ptfg_synth-m_lr1e-4_bs64_scF.log

# [9/16] lr1e-3_bs128_scT
echo "[9/16] Running: lr1e-3_bs128_scT"
contsg train --config configs/grid_search/ptfg_synth-m_lr1e-3_bs128_scT.yaml 2>&1 | tee $LOGDIR/ptfg_synth-m_lr1e-3_bs128_scT.log

# [10/16] lr1e-3_bs128_scF
echo "[10/16] Running: lr1e-3_bs128_scF"
contsg train --config configs/grid_search/ptfg_synth-m_lr1e-3_bs128_scF.yaml 2>&1 | tee $LOGDIR/ptfg_synth-m_lr1e-3_bs128_scF.log

# [11/16] lr1e-4_bs128_scT
echo "[11/16] Running: lr1e-4_bs128_scT"
contsg train --config configs/grid_search/ptfg_synth-m_lr1e-4_bs128_scT.yaml 2>&1 | tee $LOGDIR/ptfg_synth-m_lr1e-4_bs128_scT.log

# [12/16] lr1e-4_bs128_scF
echo "[12/16] Running: lr1e-4_bs128_scF"
contsg train --config configs/grid_search/ptfg_synth-m_lr1e-4_bs128_scF.yaml 2>&1 | tee $LOGDIR/ptfg_synth-m_lr1e-4_bs128_scF.log

# [13/16] lr1e-3_bs256_scT
echo "[13/16] Running: lr1e-3_bs256_scT"
contsg train --config configs/grid_search/ptfg_synth-m_lr1e-3_bs256_scT.yaml 2>&1 | tee $LOGDIR/ptfg_synth-m_lr1e-3_bs256_scT.log

# [14/16] lr1e-3_bs256_scF
echo "[14/16] Running: lr1e-3_bs256_scF"
contsg train --config configs/grid_search/ptfg_synth-m_lr1e-3_bs256_scF.yaml 2>&1 | tee $LOGDIR/ptfg_synth-m_lr1e-3_bs256_scF.log

# [15/16] lr1e-4_bs256_scT
echo "[15/16] Running: lr1e-4_bs256_scT"
contsg train --config configs/grid_search/ptfg_synth-m_lr1e-4_bs256_scT.yaml 2>&1 | tee $LOGDIR/ptfg_synth-m_lr1e-4_bs256_scT.log

# [16/16] lr1e-4_bs256_scF
echo "[16/16] Running: lr1e-4_bs256_scF"
contsg train --config configs/grid_search/ptfg_synth-m_lr1e-4_bs256_scF.yaml 2>&1 | tee $LOGDIR/ptfg_synth-m_lr1e-4_bs256_scF.log

echo "Grid search complete! 16 experiments finished."
echo "Logs in: $LOGDIR"
