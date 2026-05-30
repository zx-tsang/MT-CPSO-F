#!/bin/bash
# mrDMD-QR-F: finetune the pretrained Masked Transformer (from mt-cpso-f)
# on the QR-selected sensors of this baseline (from ../idx/mrDMD-qr/).
# Reproduces the "mrDMD-QR-F" row in the paper's Table.
#
# Usage:
#   cd baselines/mrdmd_qr
#   bash mrdmd_qr_f.sh                       # K = 2..20 (default)
#   bash mrdmd_qr_f.sh "2 10 20"             # custom K list
#
# Prereq: mt-cpso-f/scripts/driver_pretrain.sh must have produced the
# Stage-1 pretrain ckpt at the path encoded in $CKPT below.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"          # baselines/mrdmd_qr/
REPO="$(cd "$HERE/../.." && pwd)"              # repository root
MTC="$REPO/mt-cpso-f"                          # mt-cpso-f/

IDX_DIR="$REPO/baselines/idx/mrDMD-qr"         # source of QR sensor indices
OUT_PREFIX="mrdmd_qr_ft"                       # outputs under baselines/mrdmd_qr/mrdmd_qr_ft/

KS="${1:-2 4 6 8 10 12 14 16 18 20}"
SEED=42

# ------------------------------------------------------------------
# Step 1: pre-stage each K's sensor list at the path stepd_finetune.py
# expects:  <ms-out>/seed_<N>/optimized_sensor_indices.txt
# ------------------------------------------------------------------
echo "[$(date)] preparing mrDMD-QR sensor lists ..."
for K in $KS; do
    SENSOR_FILE="$IDX_DIR/sensors_n${K}.txt"
    if [[ ! -f "$SENSOR_FILE" ]]; then
        echo "[ERROR] missing $SENSOR_FILE" >&2
        echo "Re-run 'python ../idx/extract_idx.py' from the repo root" >&2
        echo "to regenerate the per-K indices." >&2
        exit 2
    fi
    SEED_DIR="$HERE/$OUT_PREFIX/K${K}/seed_${SEED}"
    mkdir -p "$SEED_DIR"
    cp "$SENSOR_FILE" "$SEED_DIR/optimized_sensor_indices.txt"
done

# ------------------------------------------------------------------
# Step 2: finetune. stepd_finetune.py uses paths relative to mt-cpso-f/,
# so we cd there and pass absolute --ms-out paths back to this folder.
# ------------------------------------------------------------------
cd "$MTC"
mkdir -p logs

DATASET=T115_4_all_place_ws50_ss10_pss50
PARAMS=scripts/params_main.json
CKPT=results_T115_4_anchorK10/ws50/pretrain/seed_42/model/mrm_T115_4_Transformer_anchorK10_d0.1_rmin0.5_rmax0.996_ws50_nl2_seed42/checkpoint.pt
NORM=data/${DATASET}/data_Norm_global.npy
PYTHON=${PYTHON:-python}

echo "[$(date)] ========== START mrdmd-qr-f finetune (K = $KS) =========="
for K in $KS; do
    OUT_DIR="$HERE/$OUT_PREFIX/K${K}"
    LOG="logs/finetune_${OUT_PREFIX}_K${K}.log"
    echo "[$(date)] >>> mrdmd-qr-f K=$K"
    $PYTHON -u stepd_finetune.py \
        --dataset $DATASET --params-json $PARAMS \
        --ckpt $CKPT --norm-file $NORM \
        --ms-out "$OUT_DIR" --total-sensors 500 \
        --seeds $SEED --epochs 40 --lr 1e-5 \
        --early-stop-patience 8 --max-workers 1 > $LOG 2>&1
    echo "[$(date)] <<< mrdmd-qr-f K=$K done"
done
echo "[$(date)] ========== mrdmd-qr-f ALL DONE =========="
