#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# Variant 09 — Full FusionDA on YOLOv5m
# ─────────────────────────────────────────────────────────────────
set -e

WEIGHTS="yolov5mu.pt"
DATA="configs/data/data.yaml"
PROJECT="runs/ablation"
NAME="09_fusionda_yolov5m"
EPOCHS=40
BATCH=4

mkdir -p "$PROJECT/$NAME"

python train_yolov5m.py \
  --weights "$WEIGHTS" \
  --data    "$DATA" \
  --epochs  "$EPOCHS" \
  --batch   "$BATCH" \
  --project "$PROJECT" \
  --name    "$NAME" \
  --imgsz 640 \
  --use-grl \
  --eval-source \
  2>&1 | tee "$PROJECT/$NAME/train.log"
