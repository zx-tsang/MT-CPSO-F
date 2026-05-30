#!/bin/bash
# Wrapper: finetune the pretrained Masked Transformer on QR-selected
# sensors from baselines/idx/, producing the "SVD-QR-F" or "mrDMD-QR-F"
# variant reported in the paper. Reuses stepd_finetune.py by pre-placing
# the QR sensor list at the path stepd expects:
#   <ms-out>/seed_<N>/optimized_sensor_indices.txt
#
# Usage:
#   bash scripts/chain_qr_finetune.sh svd-qr   "2 4 6 8 10 12 14 16 18 20"
#   bash scripts/chain_qr_finetune.sh mrdmd-qr "2 4 6 8 10 12 14 16 18 20"
#
# Prereq: scripts/driver_pretrain.sh has produced the pretrain ckpt at the
# path encoded in $CKPT below.
set -euo pipefail
cd "$(dirname "$0")/.."                          # mt-cpso-f/
mkdir -p logs

WHICH="${1:-svd-qr}"
case "$WHICH" in
    svd-qr)   IDX_SUBDIR="svd-qr";   OUT_PREFIX="svd_qr_ft" ;;
    mrdmd-qr) IDX_SUBDIR="mrDMD-qr"; OUT_PREFIX="mrdmd_qr_ft" ;;
    *) echo "Usage: $0 {svd-qr|mrdmd-qr} [\"K1 K2 ...\"]"; exit 1 ;;
esac

KS="${2:-2 4 6 8 10 12 14 16 18 20}"
SEED=42

# QR sensor indices live in baselines/idx/<IDX_SUBDIR>/sensors_n<K>.txt
# (one tap-id per line, 0-based). Resolved relative to the repo root,
# which is one level up from mt-cpso-f/.
PROJECT_ROOT="$(cd .. && pwd)"
IDX_DIR="$PROJECT_ROOT/baselines/idx/$IDX_SUBDIR"

# Same dataset / pretrain ckpt / params as the CPSO-F chain so the
# comparison stays apples-to-apples.
DATASET=T115_4_all_place_ws50_ss10_pss50
PARAMS=scripts/params_main.json
CKPT=results_T115_4_anchorK10/ws50/pretrain/seed_42/model/mrm_T115_4_Transformer_anchorK10_d0.1_rmin0.5_rmax0.996_ws50_nl2_seed42/checkpoint.pt
NORM=data/${DATASET}/data_Norm_global.npy
TOTAL=500
PYTHON=${PYTHON:-python}

echo "[$(date)] ========== START $WHICH finetune (K = $KS) =========="
for K in $KS; do
    SENSOR_FILE="$IDX_DIR/sensors_n${K}.txt"
    if [[ ! -f "$SENSOR_FILE" ]]; then
        echo "[ERROR] sensor index file missing: $SENSOR_FILE" >&2
        echo "Run 'python baselines/idx/extract_idx.py' from the repo" >&2
        echo "root to regenerate the per-K indices." >&2
        exit 2
    fi

    OUT_DIR="${OUT_PREFIX}/K${K}"
    SEED_DIR="${OUT_DIR}/seed_${SEED}"
    LOG="logs/finetune_${OUT_PREFIX}_K${K}.log"

    # Pre-populate the layout stepd_finetune.py reads from.
    mkdir -p "$SEED_DIR"
    cp "$SENSOR_FILE" "$SEED_DIR/optimized_sensor_indices.txt"

    echo "[$(date)] >>> $WHICH FT K=$K"
    $PYTHON -u stepd_finetune.py \
        --dataset $DATASET \
        --params-json $PARAMS \
        --ckpt $CKPT \
        --norm-file $NORM \
        --ms-out $OUT_DIR \
        --total-sensors $TOTAL \
        --seeds $SEED \
        --epochs 40 \
        --lr 1e-5 \
        --early-stop-patience 8 \
        --max-workers 1 > $LOG 2>&1
    echo "[$(date)] <<< $WHICH FT K=$K done"
done

echo "[$(date)] ========== $WHICH FINETUNE ALL DONE =========="
