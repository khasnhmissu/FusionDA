#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# Variant 08 — Baseline (source-only) on YOLOv5m
# ─────────────────────────────────────────────────────────────────
set -e

WEIGHTS="yolov5mu.pt"
DATA="configs/data/data.yaml"
PROJECT="runs/ablation"
NAME="08_baseline_yolov5m"
EPOCHS=40
BATCH=4

mkdir -p "$PROJECT/$NAME"

python train_yolov5m.py \
  --weights      "$WEIGHTS" \
  --data         "$DATA" \
  --epochs       "$EPOCHS" \
  --batch        "$BATCH" \
  --project      "$PROJECT" \
  --name         "$NAME" \
  --imgsz 640 \
  --baseline \
  --eval-source \
  2>&1 | tee "$PROJECT/$NAME/train.log"
