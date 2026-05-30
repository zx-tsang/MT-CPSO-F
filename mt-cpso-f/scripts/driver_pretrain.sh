#!/bin/bash
# Stage 1: masked-reconstruction pre-training of the MT-CPSO-F backbone.
# Reads hyper-parameters from scripts/params_main.json and writes logs to logs/.
set -e
cd "$(dirname "$0")/.."
mkdir -p logs
SEED=${SEED:-42}
PARAMS="scripts/params_main.json"
LOG="logs/pretrain_T115_4_anchorK10_ws50_seed${SEED}.log"
echo "[$(date)] >>> START pretrain (anchorK=10, curriculum-cosine) ws=50 seed=$SEED"
python -u stepb_train.py \
    --dataset T115_4_all_place \
    --params-json $PARAMS \
    --seed $SEED 2>&1 | tee $LOG
echo "[$(date)] <<< DONE"
