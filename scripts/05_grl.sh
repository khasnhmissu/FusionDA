#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# Variant 05 — "grl" (full FDA pipeline)
# ─────────────────────────────────────────────────────────────────
# loss = L_src + L_src_fake + λ·L_distill + β·L_consistency + L_domain
# GRL adversarial domain alignment ENABLED (--use-grl)
# Single-scale discriminator at backbone end (C2PSA / SPPF).
#
# Reference numbers (val/test, batch=4):
#   teacher : mAP50=0.613  mAP50-95=0.396
#   student : mAP50=0.540  mAP50-95=0.361
# ─────────────────────────────────────────────────────────────────
set -e

WEIGHTS="yolo26s.pt"
DATA="configs/data/data.yaml"
PROJECT="runs/ablation"
NAME="05_grl"
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
  --use-grl \
  --eval-source \
  2>&1 | tee "$PROJECT/$NAME/train.log"
