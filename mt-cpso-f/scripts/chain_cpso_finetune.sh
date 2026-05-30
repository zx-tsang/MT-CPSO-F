#!/bin/bash
# Stages 2-3: CPSO sensor search + targeted fine-tuning, for K = 2..20.
# Reuses the pretrained backbone produced by driver_pretrain.sh.
# Fast settings: n_particles=40, early_stop=10, 4 parallel workers.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

DATASET=T115_4_all_place_ws50_ss10_pss50
PARAMS=scripts/params_main.json
CKPT=results_T115_4_anchorK10/ws50/pretrain/seed_42/model/mrm_T115_4_Transformer_anchorK10_d0.1_rmin0.5_rmax0.996_ws50_nl2_seed42/checkpoint.pt
NORM=data/${DATASET}/data_Norm_global.npy
TOTAL=500
BASE_OUT=cpso_T115_4_ws50
PYTHON=${PYTHON:-python}
PARALLEL=4
SEED=42

ALL_KS="2 10 20 4 6 8"

echo "[$(date)] ========== CPSO PHASE (FAST: n_particles=40, ES=10, P=$PARALLEL) =========="
run_one_K() {
  local K=$1
  local OUT_DIR=${BASE_OUT}/K${K}
  local LOG=logs/cpso_T115_4_K${K}.log
  echo "[$(date)] >>> CPSO K=$K -> $OUT_DIR"
  $PYTHON -u stepc_cpso_multi_seed.py \
    --dataset $DATASET \
    --params-json $PARAMS \
    --ckpt $CKPT \
    --norm-file $NORM \
    --total-sensors $TOTAL \
    --select-num $K \
    --n-particles 40 \
    --max-iter 200 \
    --early-stop 10 \
    --seeds $SEED \
    --max-parallel 1 \
    --base-out $OUT_DIR > $LOG 2>&1
  echo "[$(date)] <<< CPSO K=$K done"
}
export -f run_one_K
export DATASET PARAMS CKPT NORM TOTAL BASE_OUT PYTHON SEED

echo $ALL_KS | tr " " "\n" | xargs -n 1 -P $PARALLEL -I {} bash -c "run_one_K {}"

echo "[$(date)] ========== ALL CPSO DONE; STARTING FINETUNE =========="

run_one_ft() {
  local K=$1
  local MS_OUT=${BASE_OUT}/K${K}
  local LOG=logs/finetune_T115_4_K${K}.log
  echo "[$(date)] >>> FINETUNE K=$K"
  $PYTHON -u stepd_finetune.py \
    --dataset $DATASET \
    --params-json $PARAMS \
    --ckpt $CKPT \
    --norm-file $NORM \
    --ms-out $MS_OUT \
    --total-sensors $TOTAL \
    --seeds $SEED \
    --epochs 40 \
    --lr 1e-5 \
    --early-stop-patience 8 \
    --max-workers 1 > $LOG 2>&1
  echo "[$(date)] <<< FINETUNE K=$K done"
}
export -f run_one_ft

echo $ALL_KS | tr " " "\n" | xargs -n 1 -P $PARALLEL -I {} bash -c "run_one_ft {}"

echo "[$(date)] ========== ALL DONE =========="
