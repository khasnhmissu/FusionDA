#!/bin/bash

# ===========================================
# Baseline Training Script
# ===========================================

# Train on target (foggy) dataset only
echo "========================================="
echo "Training baseline on TARGET (foggy) data"
echo "========================================="
python train_baseline.py --data data_foggy.yaml --weights yolov8l.pt --epochs 100 --name baseline_target_only_100 --batch 8

echo "========================================="
echo "Training baseline on SOURCE (clear) data"
echo "========================================="
python train_baseline.py --data data_clear.yaml --weights yolov8l.pt --epochs 200 --name baseline_source_only_200 --batch 8

echo "========================================="
echo "Training baseline on TARGET (foggy) data"
echo "========================================="
python train_baseline.py --data data_foggy.yaml --weights yolov8l.pt --epochs 200 --name baseline_target_only_200 --batch 8

echo "========================================="
echo "All baseline training completed!"
echo "========================================="
