"""
Baseline Training Script
========================
Train YOLOv8 on single domain data (no domain adaptation).

Usage:
    # Train on clear (source) data
    python train_baseline.py --data data_clear.yaml --weights yolov8n.pt --epochs 100 --name baseline_clear
    
    # Train on foggy (target) data  
    python train_baseline.py --data data_foggy.yaml --weights yolov8n.pt --epochs 100 --name baseline_foggy
    
    # Fine-tune clear model on foggy data
    python train_baseline.py --data data_foggy.yaml --weights runs/baseline_clear/weights/best.pt --epochs 50 --name finetune_foggy
"""

import argparse
from ultralytics import YOLO


def train(opt):
    """Train YOLOv8 baseline model."""
    
    # Load model
    model = YOLO(opt.weights)
    
    # Train
    results = model.train(
        data=opt.data,
        epochs=opt.epochs,
        batch=opt.batch,
        imgsz=opt.imgsz,
        device=opt.device,
        workers=opt.workers,
        project=opt.project,
        name=opt.name,
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',
        lr0=opt.lr0,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        close_mosaic=10,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        verbose=True,
    )
    
    # Validate
    metrics = model.val()
    print(f"\nFinal Results:")
    print(f"  mAP@50:    {metrics.box.map50:.4f}")
    print(f"  mAP@50-95: {metrics.box.map:.4f}")
    
    return results


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv8 Baseline Training')
    
    # Model
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='Pretrained weights')
    parser.add_argument('--data', type=str, required=True, help='Dataset YAML file')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='0', help='CUDA device')
    parser.add_argument('--workers', type=int, default=8, help='Dataloader workers')
    parser.add_argument('--lr0', type=float, default=0.01, help='Initial learning rate')
    
    # Output
    parser.add_argument('--project', type=str, default='runs/baseline', help='Project directory')
    parser.add_argument('--name', type=str, default='exp', help='Experiment name')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    print("=" * 60)
    print("YOLOv8 Baseline Training")
    print("=" * 60)
    print(f"Data:    {args.data}")
    print(f"Weights: {args.weights}")
    print(f"Epochs:  {args.epochs}")
    print(f"Device:  cuda:{args.device}")
    print("=" * 60)
    
    train(args)
