#!/usr/bin/env bash
# Single-GPU multi-K parallel training, then sequential evaluation.
# Args: $1 = "small" or "large" (selects params.yaml vs params_large.yaml)
#       $2 = KS as space-separated string (default "2 4 6 8 10 12 14 16 18 20")
#
# Environment overrides:
#   PYBIN  - python interpreter to use (default: `python` on PATH)
set -euo pipefail

SIZE="${1:-small}"
KS="${2:-2 4 6 8 10 12 14 16 18 20}"
ROOT="$(cd "$(dirname "$0")" && pwd)"            # baselines/podtfm/
PROJECT_ROOT="$(cd "$ROOT/../.." && pwd)"        # repository root (shared raw_data here)
cd "$ROOT"
PYBIN="${PYBIN:-python}"
case "$SIZE" in
    small) PARAMS="$ROOT/params.yaml" ;;
    large) PARAMS="$ROOT/params_large.yaml" ;;
    *) echo "first arg must be 'small' or 'large' (got: $SIZE)"; exit 2 ;;
esac
LOG_DIR=$ROOT/logs/podtfm_sweep_${SIZE}_par
mkdir -p "$LOG_DIR"
SUMMARY=$LOG_DIR/summary.tsv
echo -e "K\tMAE_vs_raw_all\tMAE_vs_raw_unknown\tMAE_vs_truncPOD_unknown\tPOD_floor_unknown" > "$SUMMARY"

echo "================ [$SIZE] parallel steprb KS=$KS ================"
$PYBIN stepb_train_parallel.py --params "$PARAMS" --ks $KS \
    > "$LOG_DIR/stepb_train.log" 2>&1 || {
    echo "[$SIZE] stepb_train FAILED — tail:"
    tail -30 "$LOG_DIR/stepb_train.log"
    exit 1
}
echo "[$SIZE] parallel training done. tail:"
tail -15 "$LOG_DIR/stepb_train.log"

for K in $KS; do
    TAG="podtfm_p${K}_k${K}"
    DATA="$PROJECT_ROOT/raw_data/$TAG"
    LOG="$LOG_DIR/K${K}_steprc.log"
    EVAL_DIR="$ROOT/output_sensor/$TAG/eval"

    $PYBIN stepc_evaluate.py --params "$PARAMS" \
        --data-dir "$DATA" --tag "$TAG" --split test --k "$K" \
        > "$LOG" 2>&1

    M_JSON="$EVAL_DIR/metrics.json"
    if [[ -f "$M_JSON" ]]; then
        ALL=$($PYBIN -c "import json; print(json.load(open('$M_JSON'))['MAE_vs_raw'])")
        UNK=$($PYBIN -c "import json; print(json.load(open('$M_JSON'))['MAE_vs_raw_unknown'])")
        MOD=$($PYBIN -c "import json; print(json.load(open('$M_JSON'))['MAE_vs_truncPOD_unknown'])")
        FLR=$($PYBIN -c "import json; print(json.load(open('$M_JSON'))['POD_floor_MAE_unknown'])")
        echo -e "${K}\t${ALL}\t${UNK}\t${MOD}\t${FLR}" >> "$SUMMARY"
        echo "[$SIZE K=$K] MAE_unknown=$UNK  POD_floor=$FLR"
    else
        echo -e "${K}\tNA\tNA\tNA\tNA" >> "$SUMMARY"
        echo "[$SIZE K=$K] EVAL FAILED — see $LOG"
    fi
done

echo
echo "================ [$SIZE] SWEEP DONE ================"
cat "$SUMMARY"
