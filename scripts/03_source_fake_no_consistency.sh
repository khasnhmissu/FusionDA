#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# Variant 03 — "source_fake, no consistency"
# ─────────────────────────────────────────────────────────────────
# loss = L_src + L_src_fake + λ·L_distill
# Consistency loss DISABLED (--no-consistency)
# source_fake detection forward enabled (default weight = 0.1)
# Teacher EMA + pseudo-label distillation are active.
#
# Reference numbers (val/test, batch=4):
#   teacher : mAP50=0.591  mAP50-95=0.374
#   student : mAP50=0.500  mAP50-95=0.327
# ─────────────────────────────────────────────────────────────────
set -e

WEIGHTS="yolo26s.pt"
DATA="configs/data/data.yaml"
PROJECT="runs/ablation"
NAME="03_source_fake_no_consistency"
EPOCHS=50
BATCH=4

mkdir -p "$PROJECT/$NAME"
python train.py \
  --weights         "$WEIGHTS" \
  --data            "$DATA" \
  --epochs          "$EPOCHS" \
  --batch           "$BATCH" \
  --project         "$PROJECT" \
  --name            "$NAME" \
  --no-consistency \
  --eval-source \
  2>&1 | tee "$PROJECT/$NAME/train.log"
