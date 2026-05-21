#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# WIDER Variant 01 — Baseline (pure detection, no DA)
# ─────────────────────────────────────────────────────────────────
# loss = L_src   (only wider source_real + GT)
# Val:  source_real (wider, data_wider_clear) + target_real (data_wider)
# ─────────────────────────────────────────────────────────────────
set -e

WEIGHTS="yolo26s.pt"
DATA="configs/data/data_wider.yaml"
SOURCE_DATA="configs/data/data_wider_clear.yaml"
PROJECT="runs/wider_ablation"
NAME="01_baseline"
EPOCHS=50
BATCH=4

mkdir -p "$PROJECT/$NAME"
python train.py \
  --weights      "$WEIGHTS" \
  --data         "$DATA" \
  --source-data  "$SOURCE_DATA" \
  --epochs       "$EPOCHS" \
  --batch        "$BATCH" \
  --project      "$PROJECT" \
  --name         "$NAME" \
  --baseline \
  --eval-source \
  2>&1 | tee "$PROJECT/$NAME/train.log"
