#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# Variant 12 — Full FusionDA on Faster R-CNN R-50-FPN
# ─────────────────────────────────────────────────────────────────
set -e

DATA="configs/data/data.yaml"
PROJECT="runs/ablation"
NAME="12_fusionda_fasterrcnn"
EPOCHS=40
BATCH=4
IMGSZ=640

mkdir -p "$PROJECT/$NAME"
python train_fasterrcnn.py \
  --data           "$DATA" \
  --epochs         "$EPOCHS" \
  --batch          "$BATCH" \
  --imgsz          "$IMGSZ" \
  --project        "$PROJECT" \
  --name           "$NAME" \
  --use-grl \
  --amp \
  --val-interval   5 \
  --save-interval  10 \
  --teacher-alpha  0.9999 \
  --conf-thres     0.5 \
  --lambda-weight  0.2 \
  --consistency-weight 0.5 \
  --source-fake-weight 0.1 \
  --grl-warmup     5 \
  --grl-max-alpha  1.0 \
  --grl-weight     0.05 \
  --burn-in-epochs 5 \
  --use-progressive-lambda \
  2>&1 | tee "$PROJECT/$NAME/train.log"
