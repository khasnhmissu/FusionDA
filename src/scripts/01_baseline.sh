#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# Variant 01 — Baseline (pure detection, no DA)
# ─────────────────────────────────────────────────────────────────
# loss = L_src                          (only source_real + GT)
# Disabled: teacher, GRL, distillation, consistency, source_fake
# Validation: source_real (data_clear) + target_real (data.yaml)
#
# Reference numbers from the latest run (val/test, batch=4):
#   source_real : Box(P=0.769, R=0.564, mAP50=0.640, mAP50-95=0.420)
#   target_real : Box(P=0.788, R=0.365, mAP50=0.418, mAP50-95=0.290)
# ─────────────────────────────────────────────────────────────────
set -e

WEIGHTS="yolo26s.pt"
DATA="configs/data/data.yaml"
PROJECT="runs/ablation"
NAME="01_baseline"
EPOCHS=50
BATCH=4

mkdir -p "$PROJECT/$NAME"
python train.py \
  --weights      "$WEIGHTS" \
  --data         "$DATA" \
  --epochs       "$EPOCHS" \
  --batch        "$BATCH" \
  --project      "$PROJECT" \
  --name         "$NAME" \
  --baseline \
  --eval-source \
  2>&1 | tee "$PROJECT/$NAME/train.log"
