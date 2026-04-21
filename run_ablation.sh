#!/bin/bash
# ══════════════════════════════════════════════════════════════════════
# FDA Ablation Study — Isolate which DA component causes mAP collapse
# ══════════════════════════════════════════════════════════════════════
#
# Baseline ✅ DONE: detection only → Target mAP stable ~0.47
# Now test each DA component incrementally:
#
#   A: + source_fake           (is CycleGAN data causing issues?)
#   B: + consistency           (is feature consistency loss harmful?)
#   C: + GRL                   (is adversarial training destabilizing?)
#   D: + distillation (no GRL) (is pseudo-label loss the problem?)
#   E: full pipeline           (all together with fixed config)
#
# Each experiment: 30 epochs, batch 4, val every 5 epochs
# Logs saved to runs/ablation/<name>/train.log
# ══════════════════════════════════════════════════════════════════════

set -e

WEIGHTS="yolo26s.pt"
DATA="data.yaml"
EPOCHS=50
BATCH=4
PROJECT="runs/ablation"
COMMON="--data $DATA --weights $WEIGHTS --epochs $EPOCHS --batch $BATCH --project $PROJECT --eval-source"

echo "══════════════════════════════════════════════════════════════"
echo "  FDA Ablation Study — 5 experiments × $EPOCHS epochs"
echo "══════════════════════════════════════════════════════════════"
echo ""

# ──────────────────────────────────────────────────────────────────
# A: Detection + Source_fake ONLY (no consistency, no GRL, no distill)
#    Tests: Does CycleGAN source_fake data cause mAP drop?
# ──────────────────────────────────────────────────────────────────
echo "[A] Source + Source_fake (no consistency, no GRL, no distill)"
mkdir -p $PROJECT/A_source_fake
python train.py $COMMON \
    --name A_source_fake \
    --burn-in-epochs $EPOCHS \
    --no-consistency \
    2>&1 | tee $PROJECT/A_source_fake/train.log
echo ""
echo "[A] Done. Check: $PROJECT/A_source_fake/train.log"
echo ""

# ──────────────────────────────────────────────────────────────────
# B: Detection + Source_fake + Consistency (no GRL, no distill)
#    Tests: Does consistency loss hurt or help?
# ──────────────────────────────────────────────────────────────────
echo "[B] Source + Source_fake + Consistency (no GRL, no distill)"
mkdir -p $PROJECT/B_with_consistency
python train.py $COMMON \
    --name B_with_consistency \
    --burn-in-epochs $EPOCHS \
    2>&1 | tee $PROJECT/B_with_consistency/train.log
echo ""
echo "[B] Done. Check: $PROJECT/B_with_consistency/train.log"
echo ""

# ──────────────────────────────────────────────────────────────────
# C: Detection + Source_fake + Consistency + GRL (no distill)
#    Tests: Does GRL adversarial training destabilize?
# ──────────────────────────────────────────────────────────────────
echo "[C] Source + Source_fake + Consistency + GRL (no distill)"
mkdir -p $PROJECT/C_with_grl
python train.py $COMMON \
    --name C_with_grl \
    --burn-in-epochs $EPOCHS \
    --use-grl \
    --grl-warmup 5 \
    2>&1 | tee $PROJECT/C_with_grl/train.log
echo ""
echo "[C] Done. Check: $PROJECT/C_with_grl/train.log"
echo ""

# ──────────────────────────────────────────────────────────────────
# D: Detection + Source_fake + Consistency + Distillation (NO GRL)
#    Tests: Does pseudo-label distillation cause the collapse?
#    (This is the most suspected cause: distill=23 in previous run)
# ──────────────────────────────────────────────────────────────────
echo "[D] Source + Source_fake + Consistency + Distillation (NO GRL)"
mkdir -p $PROJECT/D_with_distill
python train.py $COMMON \
    --name D_with_distill \
    --burn-in-epochs 10 \
    2>&1 | tee $PROJECT/D_with_distill/train.log
echo ""
echo "[D] Done. Check: $PROJECT/D_with_distill/train.log"
echo ""

# ──────────────────────────────────────────────────────────────────
# E: Full pipeline (all components with fixed config)
#    Tests: Does the full pipeline work with better hyperparams?
# ──────────────────────────────────────────────────────────────────
echo "[E] Full pipeline (all DA components)"
mkdir -p $PROJECT/E_full_pipeline
python train.py $COMMON \
    --name E_full_pipeline \
    --burn-in-epochs 10 \
    --use-grl \
    --grl-warmup 5 \
    2>&1 | tee $PROJECT/E_full_pipeline/train.log
echo ""
echo "[E] Done. Check: $PROJECT/E_full_pipeline/train.log"
echo ""

# ══════════════════════════════════════════════════════════════════════
# Summary — Extract Source + Target mAP from all experiments
# Goal: Source mAP stable/↑  AND  Target mAP ↑ (both matter!)
# ══════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  ABLATION RESULTS SUMMARY"
echo "  Goal: Source mAP ≥ baseline | Target mAP ↑ from baseline"
echo "  Baseline: Source ~0.62, Target ~0.47 (pure detection)"
echo "══════════════════════════════════════════════════════════════"
echo ""
printf "%-25s | %-15s | %-15s\n" "Experiment" "Source mAP@50" "Target mAP@50"
printf "%-25s-+-%-15s-+-%-15s\n" "-------------------------" "---------------" "---------------"

for exp in A_source_fake B_with_consistency C_with_grl D_with_distill E_full_pipeline; do
    logfile="$PROJECT/$exp/train.log"
    if [ -f "$logfile" ]; then
        # Extract LAST validation line with mAP values
        last_line=$(grep -E "(Source mAP|Student mAP)" "$logfile" | tail -1)
        
        # Parse source mAP
        src_map=$(echo "$last_line" | grep -oP 'Source mAP@50=\K[0-9.]+' || echo "—")
        
        # Parse target mAP (either "Target mAP@50=" or the first mAP@50= for Student)
        tgt_map=$(echo "$last_line" | grep -oP 'Target mAP@50=\K[0-9.]+' || \
                  echo "$last_line" | grep -oP 'Student mAP@50=\K[0-9.]+' || echo "—")
        
        printf "%-25s | %-15s | %-15s\n" "$exp" "$src_map" "$tgt_map"
    else
        printf "%-25s | %-15s | %-15s\n" "$exp" "NO LOG" "NO LOG"
    fi
done

echo ""
echo "──────────────────────────────────────────────────────────────"
echo "  Full mAP history per experiment (all validation epochs):"
echo "──────────────────────────────────────────────────────────────"

for exp in A_source_fake B_with_consistency C_with_grl D_with_distill E_full_pipeline; do
    logfile="$PROJECT/$exp/train.log"
    if [ -f "$logfile" ]; then
        echo ""
        echo "[$exp]"
        grep -E "(Source mAP|Student mAP)" "$logfile"
    fi
done

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Done. Full logs: $PROJECT/*/train.log"
echo "  Copy the output above and share for analysis."
echo "══════════════════════════════════════════════════════════════"
