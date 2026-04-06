#!/bin/bash
# ================================================================================
# RUN EXPLAINABILITY — Complete Pipeline
# ================================================================================
# Chạy từ thư mục gốc FusionDA trên server Linux
#
# Bước 1: Train lại 4 options (có checkpoint intermediate)
# Bước 2: Chạy 3 explainability scripts
# Bước 3: Thu thập kết quả
# ================================================================================

set -e  # Dừng khi có lỗi

# ============================================================================
# CẤU HÌNH - Điều chỉnh đường dẫn theo server
# ============================================================================

DEVICE="0"
WEIGHTS="yolo26s.pt"
DATA="data.yaml"
EPOCHS=200
BATCH=4

# Đường dẫn target images/labels (cho explainability)
TARGET_IMAGES="datasets/target_real/target_real/val/images"
TARGET_LABELS="datasets/target_real/target_real/val/labels"

# Output directories
RESULTS_DIR="results"

echo "============================================"
echo "  FusionDA Explainability Pipeline"
echo "============================================"

# # ============================================================================
# # BƯỚC 1: TRAIN LẠI 4 OPTIONS (với checkpoint mỗi 50 epoch)
# # ============================================================================

echo ""
echo "========================================"
echo " STEP 1: Training 4 options"
echo "========================================"

# 1a. Full (GRL) — EMA + GRL
echo ">>> Training: Full (GRL)"
python train.py \
    --weights $WEIGHTS \
    --data $DATA \
    --epochs $EPOCHS \
    --batch $BATCH \
    --device $DEVICE \
    --use-grl \
    --enable-monitoring \
    --project $RESULTS_DIR \
    --name full

# 1b. NoGRL — EMA + No GRL
# echo ">>> Training: NoGRL"
# python train.py \
#     --weights $WEIGHTS \
#     --data $DATA \
#     --epochs $EPOCHS \
#     --batch $BATCH \
#     --device $DEVICE \
#     --enable-monitoring \
#     --project $RESULTS_DIR \
#     --name nogrl


echo ""
echo "✅ All trained!"

# ============================================================================
# BƯỚC 2: PSEUDO-LABEL QUALITY (RQ1)
# ============================================================================

echo ""
echo "========================================"
echo " STEP 2: Pseudo-Label Quality Analysis"
echo "========================================"

python pseudo_label_quality.py --compare \
    --compare-dirs \
        $RESULTS_DIR/full/weights \
    --compare-names "Full (GRL)" \
    --weights $WEIGHTS \
    --target-images $TARGET_IMAGES \
    --target-labels $TARGET_LABELS \
    --output $RESULTS_DIR/pseudo_label_quality \
    --device $DEVICE

echo "✅ Pseudo-label quality analysis complete!"

# ============================================================================
# BƯỚC 3: EIGENCAM VISUALIZATION (RQ2)
# ============================================================================

echo ""
echo "========================================"
echo " STEP 3: EigenCAM Visualization"
echo "========================================"

python gradcam_explain.py \
    --weights $WEIGHTS \
    --checkpoints \
        $RESULTS_DIR/full/weights/best.pt \
    --names "Full (GRL)" \
    --images $TARGET_IMAGES \
    --data $DATA \
    --output $RESULTS_DIR/gradcam \
    --n-images 20 \
    --conf-thresh 0.20 \
    --global-heatmap \
    --device $DEVICE

echo "✅ EigenCAM visualization complete!"

# ============================================================================
# BƯỚC 4: DETECTION DIFF VISUALIZATION
# ============================================================================

# echo ""
# echo "========================================"
# echo " STEP 4: Detection Diff Visualization"
# echo "========================================"

python detection_diff.py \
    --weights $WEIGHTS \
    --checkpoints \
        $RESULTS_DIR/full/weights/best.pt \
    --names "Full (GRL)" \
    --images $TARGET_IMAGES \
    --labels $TARGET_LABELS \
    --output $RESULTS_DIR/detection_diff \
    --n-images 20 \
    --best-matches \
    --device $DEVICE

echo "✅ Detection diff visualization complete!"

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "============================================"
echo "  ALL EXPLAINABILITY COMPLETE!"
echo "============================================"
echo ""
echo "Outputs:"
echo "  📁 $RESULTS_DIR/full-yolov8/explainability/     — UMAP, MMD, Domain Acc"
echo "  📁 $RESULTS_DIR/nogrl-yolov8/explainability/    — UMAP, MMD (no GRL)"
echo "  📁 $RESULTS_DIR/freeze_teacher-yolov8/...       — UMAP, MMD, Domain Acc"
echo "  📁 $RESULTS_DIR/freeze_teacher-nogrl-yolov8/... — UMAP, MMD (no GRL)"
echo "  📁 $RESULTS_DIR/pseudo_label_quality/           — Quality curves + Confidence dist"
echo "  📁 $RESULTS_DIR/gradcam/                        — EigenCAM heatmaps"
echo "  📁 $RESULTS_DIR/detection_diff/                 — TP/FP/FN comparison"
echo ""
echo "Key files for paper:"
echo "  📊 pseudo_label_quality.png        → RQ1: EMA vs Frozen quality"
echo "  📊 confidence_distribution.png     → RQ1: Confidence comparison"
echo "  📊 eigencam_comparison.png         → RQ2: Where models look"
echo "  📊 detection_diff_comparison.png   → Visual: 4-model comparison"
echo "  📊 umap_epoch_*.png               → Feature alignment over time"
echo "============================================"
