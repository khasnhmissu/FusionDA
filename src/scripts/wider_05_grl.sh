#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# WIDER Variant 05 — grl (full FDA pipeline)
# ─────────────────────────────────────────────────────────────────
# loss = L_src + L_src_fake + λ·L_distill + β·L_consistency + L_domain
# GRL adversarial domain alignment ENABLED (--use-grl).
# Uses configs/train_config_wider.yaml (class_mapping = {0:0}, nc=1).
# ─────────────────────────────────────────────────────────────────
set -e

CONFIG="configs/train_config_wider.yaml"
DATA="configs/data/data_wider.yaml"
SOURCE_DATA="configs/data/data_wider_clear.yaml"
PROJECT="runs/wider_ablation"
NAME="05_grl"
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
  --use-grl \
  --eval-source \
  2>&1 | tee "$PROJECT/$NAME/train.log"
