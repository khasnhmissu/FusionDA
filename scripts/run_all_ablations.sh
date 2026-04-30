#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# Sequentially launches the 5-variant ablation that produced the
# numbers reported in the README.  Run from the project root:
#     bash scripts/run_all_ablations.sh
# ─────────────────────────────────────────────────────────────────
set -e

bash scripts/01_baseline.sh
bash scripts/02_teacher_only.sh
bash scripts/03_source_fake_no_consistency.sh
bash scripts/04_consistency.sh
bash scripts/05_grl.sh

PROJECT="runs/ablation"
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ABLATION SUMMARY  (last validation line of each run)"
echo "════════════════════════════════════════════════════════════"
printf "%-32s | %-15s | %-15s\n" "variant" "Source mAP@50" "Target mAP@50"
printf "%-32s-+-%-15s-+-%-15s\n" "$(printf '─%.0s' {1..32})" "$(printf '─%.0s' {1..15})" "$(printf '─%.0s' {1..15})"

for v in 01_baseline 02_teacher_only 03_source_fake_no_consistency 04_consistency 05_grl; do
  log="$PROJECT/$v/train.log"
  if [[ -f "$log" ]]; then
    line=$(grep -E "(Source mAP|Student mAP|Target mAP)" "$log" | tail -1)
    src=$(echo "$line" | grep -oP 'Source mAP@50=\K[0-9.]+' || echo "—")
    tgt=$(echo "$line" | grep -oP 'Target mAP@50=\K[0-9.]+' || \
          echo "$line" | grep -oP 'Student mAP@50=\K[0-9.]+' || echo "—")
    printf "%-32s | %-15s | %-15s\n" "$v" "$src" "$tgt"
  else
    printf "%-32s | %-15s | %-15s\n" "$v" "NO LOG" "NO LOG"
  fi
done
