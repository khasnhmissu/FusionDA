#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# Variant 02 — "no source_fake, only teacher"
# ─────────────────────────────────────────────────────────────────
# loss = L_src + λ·L_distill + β·L_consistency
# source_fake forward DISABLED (--source-fake-weight 0)
# Teacher EMA + pseudo-label distillation are active.
# Consistency is active by construction (source_fake feats unused
# but consistency loss still wired — see train.py).
#
# Reference numbers (val/test, batch=4):
#   teacher : mAP50=0.601  mAP50-95=0.380
#   student : mAP50=0.440  mAP50-95=0.304
# ─────────────────────────────────────────────────────────────────
set -e

WEIGHTS="yolo26s.pt"
DATA="configs/data/data.yaml"
PROJECT="runs/ablation"
NAME="02_teacher_only"
EPOCHS=50
BATCH=4

mkdir -p "$PROJECT/$NAME"
python train.py \
  --weights              "$WEIGHTS" \
  --data                 "$DATA" \
  --epochs               "$EPOCHS" \
  --batch                "$BATCH" \
  --project              "$PROJECT" \
  --name                 "$NAME" \
  --source-fake-weight   0 \
  --eval-source \
  2>&1 | tee "$PROJECT/$NAME/train.log"
