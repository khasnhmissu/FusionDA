#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# Variant 06 — "grl size sweep"
# ─────────────────────────────────────────────────────────────────
# Re-runs the BEST setting from 05_grl.sh across all YOLO26 sizes
# (n, s, m, l, x) to check whether the GRL gain is consistent
# across model capacities.
#
# Same hyper-params as 05_grl.sh; only --weights changes.
# Each run is written to runs/ablation/06_grl_<size>/.
# ─────────────────────────────────────────────────────────────────
set -e

DATA="configs/data/data.yaml"
PROJECT="runs/ablation"
EPOCHS=40
BATCH=1
SIZES=(m l x)

for s in "${SIZES[@]}"; do
  WEIGHTS="yolo26${s}.pt"
  NAME="06_grl_${s}"
  OUT="$PROJECT/$NAME"
  mkdir -p "$OUT"

  echo ""
  echo "════════════════════════════════════════════════════════════"
  echo "  [06_grl_size_sweep]  size=${s}   weights=${WEIGHTS}"
  echo "════════════════════════════════════════════════════════════"

  python train.py \
    --weights "$WEIGHTS" \
    --data    "$DATA" \
    --epochs  "$EPOCHS" \
    --batch   "$BATCH" \
    --project "$PROJECT" \
    --name    "$NAME" \
    --use-grl \
    --eval-source \
    2>&1 | tee "$OUT/train.log"
done

# ─── Summary ────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  GRL SIZE-SWEEP SUMMARY  (last validation line of each run)"
echo "════════════════════════════════════════════════════════════"
printf "%-12s | %-15s | %-15s\n" "size" "Source mAP@50" "Target mAP@50"
printf "%-12s-+-%-15s-+-%-15s\n" "$(printf '─%.0s' {1..12})" "$(printf '─%.0s' {1..15})" "$(printf '─%.0s' {1..15})"

for s in "${SIZES[@]}"; do
  log="$PROJECT/06_grl_${s}/train.log"
  if [[ -f "$log" ]]; then
    line=$(grep -E "(Source mAP|Student mAP|Target mAP)" "$log" | tail -1)
    src=$(echo "$line" | grep -oP 'Source mAP@50=\K[0-9.]+' || echo "—")
    tgt=$(echo "$line" | grep -oP 'Target mAP@50=\K[0-9.]+' || \
          echo "$line" | grep -oP 'Student mAP@50=\K[0-9.]+' || echo "—")
    printf "%-12s | %-15s | %-15s\n" "yolo26${s}" "$src" "$tgt"
  else
    printf "%-12s | %-15s | %-15s\n" "yolo26${s}" "NO LOG" "NO LOG"
  fi
done
