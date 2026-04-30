#!/usr/bin/env bash
# ------------------------------------------------------------------
# Run all explainability scripts in one shot.
#
# Assumes:
#   - baseline model:    runs/ablation/baseline/weights/baseline.pt
#   - paired full:       runs/ablation/paired_full_40ep/weights/best.pt
#                        + checkpoint_ep*.pt (for RQ1)
#   - paired + GRL:      runs/ablation/paired_grl_full_40ep/weights/best.pt
#                        + checkpoint_ep*.pt (for RQ1)
#   - target val:        datasets/target_real/target_real/val/{images,labels}
#   - source val:        datasets/source_real/source_real/val/images
#
# Run from repo root:
#   bash explain/run_all.sh
#
# Logs: explain_out/logs/{rq1,rq2a,rq2b,rq3,det_diff}.log
# ------------------------------------------------------------------

set -eu
set -o pipefail

# ─── Config ──────────────────────────────────────────────────────
ARCH_WEIGHTS="yolo26s.pt"
DATA_YAML="configs/data/data.yaml"

BASELINE_CKPT="runs/ablation/baseline/weights/baseline.pt"
PAIRED_RUN_DIR="runs/ablation/paired_full_40ep"
PAIRED_GRL_RUN_DIR="runs/ablation/paired_grl_full_40ep"

PAIRED_BEST="${PAIRED_RUN_DIR}/weights/best.pt"
PAIRED_GRL_BEST="${PAIRED_GRL_RUN_DIR}/weights/best.pt"

TARGET_IMGS="datasets/target_real/target_real/val/images"
TARGET_LBLS="datasets/target_real/target_real/val/labels"
SOURCE_IMGS="datasets/source_real/source_real/val/images"

IMGSZ=1024
DEVICE="${DEVICE:-0}"                 # allow `DEVICE=1 bash explain/run_all.sh`
N_IMAGES_ATTN=6
N_IMAGES_DRISE=6
N_MASKS_DRISE=1500
N_PER_DOMAIN=500

LOG_DIR="explain_out/logs"
mkdir -p "$LOG_DIR"

# ─── Preflight: verify files exist ───────────────────────────────
fail=0
for f in "$ARCH_WEIGHTS" "$DATA_YAML" "$BASELINE_CKPT" "$PAIRED_BEST" "$PAIRED_GRL_BEST"; do
  if [[ ! -e "$f" ]]; then
    echo "❌ MISSING: $f" >&2
    fail=1
  fi
done
for d in "$TARGET_IMGS" "$TARGET_LBLS" "$SOURCE_IMGS" "$PAIRED_RUN_DIR/weights" "$PAIRED_GRL_RUN_DIR/weights"; do
  if [[ ! -d "$d" ]]; then
    echo "❌ MISSING DIR: $d" >&2
    fail=1
  fi
done
if [[ "$fail" -ne 0 ]]; then
  echo "Fix missing paths above (edit variables at top of $0) then re-run." >&2
  exit 1
fi

# ─── Helpers ─────────────────────────────────────────────────────
run_step() {
  local name="$1"; shift
  local logfile="$LOG_DIR/$name.log"
  echo ""
  echo "════════════════════════════════════════════════════════════"
  echo " ▶ $name"
  echo "   log → $logfile"
  echo "════════════════════════════════════════════════════════════"
  local t0=$(date +%s)
  # tee so we see progress live AND keep a file.
  if "$@" 2>&1 | tee "$logfile"; then
    local t1=$(date +%s)
    echo "   ✓ $name done in $((t1 - t0))s"
  else
    local rc=${PIPESTATUS[0]}
    echo "   ✗ $name FAILED (exit $rc) — see $logfile" >&2
    exit "$rc"
  fi
}

overall_t0=$(date +%s)

# ─── RQ1: pseudo-label quality over epochs (2 paired runs) ───────
# Uses checkpoint_ep*.pt files for teacher_state_dict.
run_step "rq1_pseudo_label_quality" \
  python explain/rq1_pseudo_label_quality.py \
    --compare-runs "$PAIRED_RUN_DIR" "$PAIRED_GRL_RUN_DIR" \
    --names "Paired Full" "Paired GRL" \
    --weights "$ARCH_WEIGHTS" \
    --data "$DATA_YAML" \
    --target-images "$TARGET_IMGS" \
    --target-labels "$TARGET_LBLS" \
    --conf-thres 0.5 --iou-thres 0.5 --imgsz "$IMGSZ" \
    --device "$DEVICE" \
    --output explain_out/rq1_compare

# ─── RQ2a: feature space UMAP + MMD (baseline vs 2 DA) ───────────
run_step "rq2a_feature_umap" \
  python explain/rq2_feature_umap.py \
    --checkpoint "$BASELINE_CKPT" "$PAIRED_BEST" "$PAIRED_GRL_BEST" \
    --names "Baseline" "Paired Full" "Paired GRL" \
    --weights "$ARCH_WEIGHTS" \
    --source-images "$SOURCE_IMGS" \
    --target-images "$TARGET_IMGS" \
    --n-per-domain "$N_PER_DOMAIN" \
    --imgsz "$IMGSZ" --batch 4 \
    --device "$DEVICE" \
    --tsne \
    --output explain_out/rq2_feature

# ─── RQ2b: C2PSA self-attention (3 models, same target images) ──
run_step "rq2b_c2psa_attention" \
  python explain/rq2_c2psa_attention.py \
    --checkpoint "$BASELINE_CKPT" "$PAIRED_BEST" "$PAIRED_GRL_BEST" \
    --names "Baseline" "Paired Full" "Paired GRL" \
    --weights "$ARCH_WEIGHTS" \
    --data "$DATA_YAML" \
    --images "$TARGET_IMGS" \
    --n-images "$N_IMAGES_ATTN" \
    --imgsz "$IMGSZ" --conf-thres 0.25 --max-queries 4 \
    --device "$DEVICE" \
    --output explain_out/rq2_attention

# ─── RQ3: D-RISE saliency (3 models, target foggy images) ────────
run_step "rq3_d_rise" \
  python explain/rq3_d_rise.py \
    --checkpoint "$BASELINE_CKPT" "$PAIRED_BEST" "$PAIRED_GRL_BEST" \
    --names "Baseline" "Paired Full" "Paired GRL" \
    --weights "$ARCH_WEIGHTS" \
    --data "$DATA_YAML" \
    --images "$TARGET_IMGS" \
    --labels "$TARGET_LBLS" \
    --n-images "$N_IMAGES_DRISE" \
    --n-masks "$N_MASKS_DRISE" \
    --mask-grid 8 --mask-p 0.5 --max-targets 3 \
    --conf-thres 0.3 --imgsz "$IMGSZ" --batch 8 \
    --device "$DEVICE" --seed 42 \
    --output explain_out/rq3_drise

# ─── Supporting: detection diff (TP/FP/FN) ───────────────────────
run_step "det_diff_all" \
  python explain/detection_diff.py \
    --checkpoint "$BASELINE_CKPT" "$PAIRED_BEST" "$PAIRED_GRL_BEST" \
    --names "Baseline" "Paired Full" "Paired GRL" \
    --weights "$ARCH_WEIGHTS" \
    --data "$DATA_YAML" \
    --images "$TARGET_IMGS" \
    --labels "$TARGET_LBLS" \
    --n-images 8 --iou-thres 0.5 --conf-thres 0.25 --imgsz "$IMGSZ" \
    --best-matches \
    --device "$DEVICE" \
    --output explain_out/detection_diff_all

run_step "det_diff_small" \
  python explain/detection_diff.py \
    --checkpoint "$BASELINE_CKPT" "$PAIRED_BEST" "$PAIRED_GRL_BEST" \
    --names "Baseline" "Paired Full" "Paired GRL" \
    --weights "$ARCH_WEIGHTS" \
    --data "$DATA_YAML" \
    --images "$TARGET_IMGS" \
    --labels "$TARGET_LBLS" \
    --n-images 8 --iou-thres 0.5 --conf-thres 0.25 --imgsz "$IMGSZ" \
    --best-matches --size-filter small \
    --device "$DEVICE" \
    --output explain_out/detection_diff_small

# ─── Summary ─────────────────────────────────────────────────────
overall_t1=$(date +%s)
mins=$(( (overall_t1 - overall_t0) / 60 ))
secs=$(( (overall_t1 - overall_t0) % 60 ))

echo ""
echo "════════════════════════════════════════════════════════════"
echo " ✓ ALL EXPLAIN STEPS DONE  (total ${mins}m ${secs}s)"
echo "════════════════════════════════════════════════════════════"
echo " Outputs:"
echo "   RQ1  → explain_out/rq1_compare/"
echo "   RQ2a → explain_out/rq2_feature/"
echo "   RQ2b → explain_out/rq2_attention/"
echo "   RQ3  → explain_out/rq3_drise/"
echo "   TP/FP/FN (all sizes)   → explain_out/detection_diff_all/"
echo "   TP/FP/FN (small only)  → explain_out/detection_diff_small/"
echo "   Logs → $LOG_DIR/"
echo "════════════════════════════════════════════════════════════"
