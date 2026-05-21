#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# WIDER Variant 04 — consistency (default DA, no GRL)
# ─────────────────────────────────────────────────────────────────
# loss = L_src + L_src_fake + λ·L_distill + β·L_consistency
# All DA terms active EXCEPT GRL.
# Uses configs/train_config_wider.yaml (class_mapping = {0:0}, nc=1).
# ─────────────────────────────────────────────────────────────────
set -e

CONFIG="configs/train_config_wider.yaml"
DATA="configs/data/data_wider.yaml"
SOURCE_DATA="configs/data/data_wider_clear.yaml"
PROJECT="runs/wider_ablation"
NAME="04_consistency"
EPOCHS=50
BATCH=4

mkdir -p "$PROJECT/$NAME"
python train.py \
  --config       "$CONFIG" \
  --data         "$DATA" \
  --source-data  "$SOURCE_DATA" \
  --epochs       "$EPOCHS" \
  --batch        "$BATCH" \
  --project      "$PROJECT" \
  --name         "$NAME" \
  --eval-source \
  2>&1 | tee "$PROJECT/$NAME/train.log"
