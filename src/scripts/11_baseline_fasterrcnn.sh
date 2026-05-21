#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# Variant 11 — Baseline (source-only) on Faster R-CNN R-50-FPN
# ─────────────────────────────────────────────────────────────────
set -e

DATA="configs/data/data.yaml"
PROJECT="runs/ablation"
NAME="11_baseline_fasterrcnn"
EPOCHS=40
BATCH=4
IMGSZ=640

mkdir -p "$PROJECT/$NAME"
python train_fasterrcnn.py \
  --data         "$DATA" \
  --epochs       "$EPOCHS" \
  --batch        "$BATCH" \
  --imgsz        "$IMGSZ" \
  --project      "$PROJECT" \
  --name         "$NAME" \
  --baseline \
  --amp \
  --val-interval 5 \
  --save-interval 10 \
  2>&1 | tee "$PROJECT/$NAME/train.log"
