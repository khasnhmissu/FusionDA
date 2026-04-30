"""
Baseline Training Script (Dual-Domain Supervised)
===================================================
Train YOLO26/YOLOv8 on both source (clear) and target (foggy) data with labels.
No domain adaptation techniques — just standard supervised training on both domains.

This serves as a baseline to compare against FDA (train.py).

Usage:
    # Train on both source + target (default, uses configs/data/data.yaml)
    python train_baseline.py --data configs/data/data.yaml --weights yolo26s.pt --epochs 100 --name baseline_combined

    # Train on source only
    python train_baseline.py --data configs/data/data_clear.yaml --weights yolo26s.pt --epochs 100 --name baseline_source_only

    # Train on target only
    python train_baseline.py --data configs/data/data_foggy.yaml --weights yolo26s.pt --epochs 100 --name baseline_target_only
"""

import argparse
import gc
import math
import time
import yaml
from pathlib import Path
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda import amp

from ultralytics import YOLO
from ultralytics.utils import LOGGER, colorstr


def train(opt):
    """Train YOLO26/YOLOv8 baseline model on source + target domains (both labeled)."""

    device = torch.device(f'cuda:{opt.device}' if opt.device.isdigit() else opt.device)
    save_dir = Path(opt.project) / opt.name
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / 'weights').mkdir(exist_ok=True)

    # Load data config
    with open(opt.data, encoding='utf-8') as f:
        data_dict = yaml.safe_load(f)

    nc = data_dict['nc']
    names = data_dict['names']
    root = Path(data_dict.get('path', ''))

    # Check if this is a multi-domain config (has train_source_real / train_target_real)
    has_source = 'train_source_real' in data_dict or 'train' in data_dict
    has_target = 'train_target_real' in data_dict

    if has_target:
        LOGGER.info(colorstr('Baseline: ') + 'Dual-domain training (source + target, both labeled)')
    else:
        LOGGER.info(colorstr('Baseline: ') + 'Single-domain training')

    # Load model
    yolo = YOLO(opt.weights)
    model = yolo.model.to(device)
    for param in model.parameters():
        param.requires_grad = True

    LOGGER.info(f'Model params: {sum(p.numel() for p in model.parameters()):,}')

    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=opt.lr0, weight_decay=0.0005)
    lf = lambda x: ((1 - math.cos(x * math.pi / opt.epochs)) / 2) * (opt.lrf - 1) + 1
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # Build dataloaders
    from ultralytics.data import YOLODataset
    from torch.utils.data import DataLoader, ConcatDataset
    from ultralytics.cfg import get_cfg

    gs = max(int(model.stride.max()), 32)
    default_cfg = get_cfg()

    def build_yolo_dataset(img_path, augment=True):
        """Build a YOLODataset for a given image path."""
        return YOLODataset(
            img_path=img_path,
            imgsz=opt.imgsz,
            augment=augment,
            hyp=default_cfg,
            data=data_dict,
            stride=gs,
            rect=False,
            batch_size=opt.batch,
        )

    # --- Build training datasets ---
    train_datasets = []

    if has_target:
        # Multi-domain: source_real + target_real
        source_path = data_dict.get('train_source_real', data_dict.get('train', ''))
        if isinstance(source_path, list):
            source_path = source_path[0]
        source_path = str(root / source_path)

        target_path = data_dict.get('train_target_real', '')
        if isinstance(target_path, list):
            target_path = target_path[0]
        target_path = str(root / target_path)

        LOGGER.info(f'Source train: {source_path}')
        LOGGER.info(f'Target train: {target_path}')

        ds_source = build_yolo_dataset(source_path, augment=True)
        ds_target = build_yolo_dataset(target_path, augment=True)
        train_datasets = [ds_source, ds_target]

        LOGGER.info(f'Source samples: {len(ds_source)}, Target samples: {len(ds_target)}')
    else:
        # Single-domain
        train_path = data_dict.get('train', '')
        if isinstance(train_path, list):
            train_path = train_path[0]
        train_path = str(root / train_path)

        LOGGER.info(f'Train path: {train_path}')
        ds_train = build_yolo_dataset(train_path, augment=True)
        train_datasets = [ds_train]
        LOGGER.info(f'Train samples: {len(ds_train)}')

    # Combine datasets
    if len(train_datasets) > 1:
        combined_dataset = ConcatDataset(train_datasets)
    else:
        combined_dataset = train_datasets[0]

    train_loader = DataLoader(
        combined_dataset,
        batch_size=opt.batch,
        shuffle=True,
        num_workers=opt.workers,
        pin_memory=True,
        collate_fn=YOLODataset.collate_fn,
    )

    nb = len(train_loader)
    LOGGER.info(colorstr('Baseline: ') + f'{len(combined_dataset)} total samples, {nb} batches')

    # --- Build validation dataset (evaluate on target for DA comparison) ---
    val_path = data_dict.get('test', data_dict.get('val', ''))
    if isinstance(val_path, list):
        val_path = val_path[0]
    val_path_full = str(root / val_path)
    LOGGER.info(f'Validation/Test: {val_path_full}')

    # Loss function (use E2ELoss for YOLO26 NMS-free, fallback to v8DetectionLoss for YOLOv8/11)
    # v8DetectionLoss requires model.args.dfl (checked at line 450 of ultralytics/utils/loss.py)
    model.args = default_cfg
    try:
        from ultralytics.utils.loss import E2ELoss
        compute_loss = E2ELoss(model)
    except ImportError:
        from ultralytics.utils.loss import v8DetectionLoss
        compute_loss = v8DetectionLoss(model)

    # AMP
    use_amp = opt.amp and device.type != 'cpu'
    scaler = amp.GradScaler(enabled=use_amp)
    LOGGER.info(colorstr('AMP: ') + ('Enabled' if use_amp else 'Disabled'))

    # Training loop
    LOGGER.info(f'\n{colorstr("Starting training:")} {opt.epochs} epochs...\n')

    best_fitness = 0.0
    t0 = time.time()

    for epoch in range(opt.epochs):
        model.train()

        mloss = torch.zeros(3, device=device)  # box, cls, dfl
        pbar = tqdm(enumerate(train_loader), total=nb, desc=f'Epoch {epoch}/{opt.epochs-1}')
        optimizer.zero_grad()

        for i, batch in pbar:
            # Prepare images
            imgs = batch['img'].to(device).float() / 255.0
            imgs = imgs.clamp(0, 1)

            with torch.amp.autocast('cuda', enabled=use_amp):
                # Forward
                pred = model(imgs)

                # Compute detection loss (returns multi-element tensor)
                loss_raw, loss_items = compute_loss(pred, batch)
                loss = loss_raw.sum()

                if torch.isnan(loss).any().item() or torch.isinf(loss).any().item():
                    LOGGER.warning(f'[NaN] Loss is NaN/Inf at epoch {epoch}, iter {i}. Skipping.')
                    optimizer.zero_grad()
                    continue

            # Backward
            scaler.scale(loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Logging
            mloss = (mloss * i + loss_items[:3]) / (i + 1)

            mem = f'{torch.cuda.memory_reserved() / 1e9:.1f}G' if torch.cuda.is_available() else 'CPU'
            pbar.set_postfix({
                'mem': mem,
                'box': f'{mloss[0]:.4f}',
                'cls': f'{mloss[1]:.4f}',
                'dfl': f'{mloss[2]:.4f}',
            })

        # End of epoch
        scheduler.step()

        gc.collect()
        torch.cuda.empty_cache()

        # Validation (every val_interval epochs + last epoch)
        if (epoch + 1) % opt.val_interval == 0 or epoch == opt.epochs - 1:
            orig_model = yolo.model
            try:
                val_yolo = YOLO(opt.weights)
                val_model = val_yolo.model.to(device)
                val_model.load_state_dict(model.state_dict())
                val_model.eval()
                yolo.model = val_model

                metrics = yolo.val(data=opt.data, split='test', verbose=False)
                current_map50 = metrics.box.map50
                current_map = metrics.box.map

                LOGGER.info(
                    f'Epoch {epoch}: mAP@50={current_map50:.4f}, '
                    f'mAP@50-95={current_map:.4f}, '
                    f'P={metrics.box.mp:.4f}, R={metrics.box.mr:.4f}'
                )

                if current_map50 > best_fitness:
                    best_fitness = current_map50
                    torch.save({
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_fitness': best_fitness,
                    }, save_dir / 'weights' / 'best.pt')
                    LOGGER.info(f'  ★ New best mAP@50: {best_fitness:.4f}')

            except Exception as e:
                LOGGER.warning(f'Validation failed: {e}')
            finally:
                yolo.model = orig_model
                del val_model, val_yolo
                gc.collect()
                torch.cuda.empty_cache()

        # Save last
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_fitness': best_fitness,
        }, save_dir / 'weights' / 'last.pt')

    # Summary
    hours = (time.time() - t0) / 3600
    LOGGER.info(f'\n{"=" * 60}')
    LOGGER.info(f'{opt.epochs} epochs completed in {hours:.2f} hours.')
    LOGGER.info(f'Best mAP@50: {best_fitness:.4f}')
    LOGGER.info(f'Results saved to {save_dir}')
    LOGGER.info(f'{"=" * 60}')

    return best_fitness


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO26/YOLOv8 Baseline Training (Dual-Domain)')

    # Model
    parser.add_argument('--weights', type=str, default='yolo26s.pt', help='Pretrained weights')
    parser.add_argument('--data', type=str, default='configs/data/data.yaml', help='Dataset YAML file (supports multi-domain)')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')

    # Training
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='0', help='CUDA device')
    parser.add_argument('--workers', type=int, default=8, help='Dataloader workers')
    parser.add_argument('--lr0', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01, help='Final lr factor (lr0 * lrf)')

    # Validation
    parser.add_argument('--val-interval', type=int, default=5, help='Validate every N epochs')

    # Output
    parser.add_argument('--project', type=str, default='runs/baseline', help='Project directory')
    parser.add_argument('--name', type=str, default='exp', help='Experiment name')

    # Misc
    parser.add_argument('--amp', action='store_true', help='Enable AMP (mixed precision)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print("=" * 60)
    print("YOLO26/YOLOv8 Baseline Training (Source + Target Supervised)")
    print("=" * 60)
    print(f"Data:    {args.data}")
    print(f"Weights: {args.weights}")
    print(f"Epochs:  {args.epochs}")
    print(f"Batch:   {args.batch}")
    print(f"LR:      {args.lr0}")
    print(f"Device:  cuda:{args.device}")
    print(f"AMP:     {'Enabled' if args.amp else 'Disabled'}")
    print("=" * 60)

    train(args)
