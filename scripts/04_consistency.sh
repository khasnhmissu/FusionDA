#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# Variant 04 — "consistency" (default DA, no GRL)
# ─────────────────────────────────────────────────────────────────
# loss = L_src + L_src_fake + λ·L_distill + β·L_consistency
# All DA terms active EXCEPT GRL.
# This is the default invocation of train.py for the FDA pipeline.
#
# Reference numbers (val/test, batch=4):
#   teacher : mAP50=0.606  mAP50-95=0.388
#   student : mAP50=0.536  mAP50-95=0.356
# ─────────────────────────────────────────────────────────────────
set -e

WEIGHTS="yolo26s.pt"
DATA="configs/data/data.yaml"
PROJECT="runs/ablation"
NAME="04_consistency"
EPOCHS=50
BATCH=4

mkdir -p "$PROJECT/$NAME"
python train.py \
  --weights "$WEIGHTS" \
  --data    "$DATA" \
  --epochs  "$EPOCHS" \
  --batch   "$BATCH" \
  --project "$PROJECT" \
  --name    "$NAME" \
  --eval-source \
  2>&1 | tee "$PROJECT/$NAME/train.log"
