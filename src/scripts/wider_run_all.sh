#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# Run all WIDERFACE ablation variants sequentially.
#   01_baseline    — pure detection, source only
#   04_consistency — FDA without GRL
#   05_grl         — full FDA (GRL + teacher + distillation + consistency)
# ─────────────────────────────────────────────────────────────────
set -e

bash scripts/wider_01_baseline.sh
bash scripts/wider_04_consistency.sh
bash scripts/wider_05_grl.sh

echo ""
echo "=== All WIDER ablations finished ==="
echo "Results: runs/wider_ablation/{01_baseline,04_consistency,05_grl}/"
