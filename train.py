import argparse
import gc
import math
import time
import yaml
from pathlib import Path
from copy import deepcopy
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda import amp

from ultralytics import YOLO
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.nms import non_max_suppression

from domain_adaptation import (
    DomainDiscriminator, YOLOv8FeatureHook,
    compute_domain_loss, get_grl_alpha, get_domain_accuracy
)

from fusion_da import (FDALoss, WeightEMA)
from utils.FDA_helpers import (
    get_adaptive_conf_thres,
    filter_pseudo_labels_by_uncertainty,
    get_progressive_lambda,
)
from utils.training_logger import TrainingLogger
from utils.domain_monitor import DomainMonitor

def train(opt):
    """Main training function"""
    
    # Setup
    device = torch.device(f'cuda:{opt.device}' if opt.device.isdigit() else opt.device)
    save_dir = Path(opt.project) / opt.name
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / 'weights').mkdir(exist_ok=True)
    
    # Load data config
    with open(opt.data) as f:
        data_dict = yaml.safe_load(f)
    
    nc = data_dict['nc']
    names = data_dict['names']
    
    LOGGER.info(colorstr('FDA: ') + f'Training with {nc} classes')
    
    # ========================================================================
    # MODEL SETUP
    # ========================================================================
    # Load student model
    yolo_student = YOLO(opt.weights)
    model_student = yolo_student.model.to(device)
    
    # CRITICAL: Unfreeze student model parameters (YOLO loads frozen by default)
    for param in model_student.parameters():
        param.requires_grad = True
    
    # Create teacher model (deep copy)
    model_teacher = deepcopy(model_student).to(device)
    
    # Freeze teacher parameters
    for param in model_teacher.parameters():
        param.requires_grad = False
    
    # Setup WeightEMA for teacher updates
    student_params = [p for p in model_student.parameters() if p.requires_grad]
    teacher_params = list(model_teacher.parameters())
    teacher_ema = WeightEMA(teacher_params, student_params, alpha=opt.teacher_alpha)
    
    LOGGER.info(f'Student params: {sum(p.numel() for p in student_params):,}')
    LOGGER.info(f'Teacher params: {sum(p.numel() for p in teacher_params):,}')
    
    # ========================================================================
    # GRL SETUP (Phase 3)
    # ========================================================================
    domain_discriminator = None
    feature_hook = None
    grl_optimizer = None
    
    if opt.use_grl:
        # Auto-detect in_channels from actual forward pass
        feature_hook = YOLOv8FeatureHook(model_student, layer_idx=9)
        
        with torch.no_grad():
            test_img = torch.zeros(1, 3, opt.imgsz, opt.imgsz, device=device)
            _ = model_student(test_img)
            test_feat = feature_hook.get_features()
            in_channels = test_feat.shape[1] if test_feat is not None else 256
        
        domain_discriminator = DomainDiscriminator(
            in_channels=in_channels,
            hidden_dim=opt.grl_hidden_dim,
            dropout=0.3
        ).to(device)
        
        grl_optimizer = optim.Adam(domain_discriminator.parameters(), lr=1e-4)
        
        LOGGER.info(f'{colorstr("GRL:")} Initialized with {in_channels} channels (auto-detected)')
    
    # ========================================================================
    # OPTIMIZER & SCHEDULER
    # ========================================================================
    optimizer = optim.AdamW(model_student.parameters(), lr=opt.lr0, weight_decay=0.0005)
    
    # Cosine annealing scheduler
    lf = lambda x: ((1 - math.cos(x * math.pi / opt.epochs)) / 2) * (opt.lrf - 1) + 1
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    # ========================================================================
    # DATALOADER SETUP
    # ========================================================================
    root = Path(data_dict.get('path', ''))
    
    # ========================================================================
    # TRAINING PATHS (use train split)
    # ========================================================================
    # Source domain paths (real + fake) - TRAIN split
    source_real = data_dict.get('train_source_real', data_dict.get('train', []))
    if isinstance(source_real, str):
        source_real = [source_real]
    source_real = [str(root / p) for p in source_real]
    
    source_fake = data_dict.get('train_source_fake', [])
    if isinstance(source_fake, str):
        source_fake = [source_fake]
    source_fake = [str(root / p) for p in source_fake] if source_fake else []
    
    # Target domain paths (real + fake) - TRAIN split
    target_real = data_dict.get('train_target_real', [])
    if isinstance(target_real, str):
        target_real = [target_real]
    target_real = [str(root / p) for p in target_real]
    
    target_fake = data_dict.get('train_target_fake', [])
    if isinstance(target_fake, str):
        target_fake = [target_fake]
    target_fake = [str(root / p) for p in target_fake] if target_fake else []
    
    # ========================================================================
    # VALIDATION PATHS (use val split, fallback to train if not specified)
    # ========================================================================
    val_source_real = data_dict.get('val_source_real', data_dict.get('val', []))
    if isinstance(val_source_real, str):
        val_source_real = [val_source_real]
    val_source_real = [str(root / p) for p in val_source_real] if val_source_real else source_real
    
    val_target_real = data_dict.get('val_target_real', data_dict.get('test', []))
    if isinstance(val_target_real, str):
        val_target_real = [val_target_real]
    val_target_real = [str(root / p) for p in val_target_real] if val_target_real else target_real
    
    LOGGER.info(f'Source Real (train): {source_real}')
    LOGGER.info(f'Source Fake (train): {source_fake}')
    LOGGER.info(f'Target Real (train): {target_real}')
    LOGGER.info(f'Target Fake (train): {target_fake}')
    LOGGER.info(f'Validation Source: {val_source_real}')
    LOGGER.info(f'Validation Target: {val_target_real}')
    
    # Build dataloaders using Ultralytics
    from ultralytics.data import YOLODataset
    from torch.utils.data import DataLoader
    from ultralytics.cfg import get_cfg
    
    gs = max(int(model_student.stride.max()), 32)
    
    # Get default hyperparameters from ultralytics
    default_cfg = get_cfg()
    
    # Common dataset config
    dataset_args = {
        'imgsz': opt.imgsz,
        'batch_size': opt.batch,
        'augment': True,
        'hyp': default_cfg,  # Use ultralytics full default config
        'rect': False,
        'cache': False,
        'single_cls': False,
        'stride': gs,
        'pad': 0.5,
        'classes': None,
        'data': data_dict,
        'task': 'detect',
    }
    
    # Source Real dataloader
    source_dataset = YOLODataset(
        img_path=source_real[0] if source_real else '',
        **dataset_args
    )
    source_loader = DataLoader(
        source_dataset, batch_size=opt.batch, shuffle=True,
        num_workers=opt.workers, pin_memory=True, collate_fn=YOLODataset.collate_fn
    )
    
    # Target Real dataloader
    if target_real and target_real[0]:  # Check if path exists and is not empty
        target_dataset = YOLODataset(img_path=target_real[0], **dataset_args)
        target_loader = DataLoader(
            target_dataset, batch_size=opt.batch, shuffle=True,
            num_workers=opt.workers, pin_memory=True, collate_fn=YOLODataset.collate_fn
        )
    else:
        LOGGER.warning(colorstr('yellow', 'WARNING: ') + 'train_target_real not found, using source_real as target!')
        target_loader = source_loader
    
    # Source Fake dataloader
    if source_fake and source_fake[0]:
        source_fake_dataset = YOLODataset(img_path=source_fake[0], **dataset_args)
        source_fake_loader = DataLoader(
            source_fake_dataset, batch_size=opt.batch, shuffle=True,
            num_workers=opt.workers, pin_memory=True, collate_fn=YOLODataset.collate_fn
        )
    else:
        LOGGER.warning(colorstr('yellow', 'WARNING: ') + 'train_source_fake not found, using source_real!')
        source_fake_loader = source_loader
    
    # Target Fake dataloader
    if target_fake and target_fake[0]:
        target_fake_dataset = YOLODataset(img_path=target_fake[0], **dataset_args)
        target_fake_loader = DataLoader(
            target_fake_dataset, batch_size=opt.batch, shuffle=True,
            num_workers=opt.workers, pin_memory=True, collate_fn=YOLODataset.collate_fn
        )
    else:
        LOGGER.warning(colorstr('yellow', 'WARNING: ') + 'train_target_fake not found, using target_real! Distillation will NOT work correctly!')
        target_fake_loader = target_loader
    
    nb = max(len(source_loader), len(target_loader), len(source_fake_loader) if source_fake else 0, len(target_fake_loader) if target_fake else 0)
    
    # ========================================================================
    # LOSS FUNCTION
    # ========================================================================
    # Class mapping: COCO class ID -> Dataset class ID
    # IMPORTANT: Include both COCO IDs (early training) AND dataset IDs (after EMA adaptation)
    class_mapping = getattr(opt, 'class_mapping', {0: 0, 1: 1, 2: 1})
    compute_loss = FDALoss(model_student, class_mapping=class_mapping)
    scaler = amp.GradScaler(enabled=device.type != 'cpu')
    # Đảm bảo scaler được enabled khi dùng CUDA
    if device.type == 'cpu':
        print("[WARNING] AMP disabled on CPU")
    # ========================================================================
    # LOGGER SETUP
    # ========================================================================
    logger = TrainingLogger(
        save_dir=str(save_dir),
        project_name='FusionDA',
        use_tensorboard=True,
        verbose=True,
    )
    logger.log_config({
        'weights': opt.weights,
        'data': opt.data,
        'epochs': opt.epochs,
        'batch_size': opt.batch,
        'imgsz': opt.imgsz,
        'lr0': opt.lr0,
        'teacher_alpha': opt.teacher_alpha,
        'conf_thres': opt.conf_thres,
        'lambda_weight': opt.lambda_weight,
        'use_grl': opt.use_grl,
        'grl_weight': opt.grl_weight if opt.use_grl else 0,
        'nc': nc,
        'names': names,
    })
    
    # ========================================================================
    # DOMAIN MONITOR SETUP (Explainability)
    # ========================================================================
    domain_monitor = None
    if opt.enable_monitoring:
        # Determine UMAP epochs based on total
        if opt.epochs <= 50:
            umap_epochs = [0, opt.epochs // 2, opt.epochs - 1]
            tsne_epochs = []  # Disabled - causes OOM
        else:
            step = opt.epochs // 4
            umap_epochs = [0, step, step * 2, step * 3, opt.epochs - 1]
            tsne_epochs = []  # Disabled - causes OOM
        
        domain_monitor = DomainMonitor(
            save_dir=str(save_dir),
            umap_epochs=umap_epochs,
            tsne_epochs=tsne_epochs,
            verbose=True,
        )
        domain_monitor.set_total_epochs(opt.epochs)
        LOGGER.info(f'{colorstr("DomainMonitor:")} Enabled with UMAP at epochs {umap_epochs}')
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    LOGGER.info(f'\n{colorstr("Starting training:")} {opt.epochs} epochs...\n')
    
    best_fitness = 0.0
    t0 = time.time()
    global_step = 0
    
    for epoch in range(opt.epochs):
        model_student.train()
        model_teacher.eval()
        
        if opt.use_grl:
            domain_discriminator.train()
            current_grl_alpha = get_grl_alpha(epoch, opt.epochs, opt.grl_warmup, opt.grl_max_alpha)
        
        # Dynamic parameters
        if opt.use_progressive_lambda:
            current_lambda = get_progressive_lambda(epoch, opt.epochs, opt.warmup_epochs, 0.0, opt.lambda_weight)
        else:
            current_lambda = opt.lambda_weight
        
        # Initialize loss tracking: box, cls, dfl, distill, domain (5 components)
        mloss = torch.zeros(5, device=device)
        
        # Create iterators for all 4 dataloaders
        source_iter = iter(source_loader)
        source_fake_iter = iter(source_fake_loader)
        target_iter = iter(target_loader)
        target_fake_iter = iter(target_fake_loader)
        
        pbar = tqdm(range(nb), desc=f'Epoch {epoch}/{opt.epochs-1}')
        
        optimizer.zero_grad()
        
        for i in pbar:
            # ================================================================
            # LOAD DATA (4 inputs: SR, SF, TR, TF)
            # ================================================================
            # Source Real
            try:
                batch_source = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                batch_source = next(source_iter)
            
            # Source Fake
            try:
                batch_source_fake = next(source_fake_iter)
            except StopIteration:
                source_fake_iter = iter(source_fake_loader)
                batch_source_fake = next(source_fake_iter)
            
            # Target Real
            try:
                batch_target = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                batch_target = next(target_iter)
            
            # Target Fake
            try:
                batch_target_fake = next(target_fake_iter)
            except StopIteration:
                target_fake_iter = iter(target_fake_loader)
                batch_target_fake = next(target_fake_iter)
            
            # Normalize all inputs
            imgs_source = batch_source['img'].to(device).float() / 255.0
            imgs_source_fake = batch_source_fake['img'].to(device).float() / 255.0
            imgs_target = batch_target['img'].to(device).float() / 255.0
            imgs_target_fake = batch_target_fake['img'].to(device).float() / 255.0
            
            # ================================================================
            # FORWARD PASS (4 PHASES)
            # ================================================================
            with torch.amp.autocast('cuda', enabled=device.type != 'cpu'):
                
                # ============================================================
                # PHASE 1: SOURCE REAL FORWARD
                # ============================================================
                pred_source = model_student(imgs_source)
                loss_source, loss_items_source = compute_loss(pred_source, batch_source)
                
                # Extract source features (ALWAYS for monitoring, used for GRL after warmup)
                source_features = None
                if feature_hook:
                    source_features = feature_hook.get_features()
                
                # ============================================================
                # PHASE 1B: SOURCE FAKE FORWARD (same labels as source real)
                # ============================================================
                pred_source_fake = model_student(imgs_source_fake)
                loss_source_fake, _ = compute_loss(pred_source_fake, batch_source)  # Use source real labels!
                
                # Clear GRL hook buffer
                if opt.use_grl and feature_hook:
                    _ = feature_hook.get_features()
                
                # ============================================================
                # PHASE 2: TARGET REAL FORWARD
                # ============================================================
                pred_target = model_student(imgs_target)
                
                # Extract target features (ALWAYS for monitoring, used for GRL after warmup)
                target_features = None
                if feature_hook:
                    target_features = feature_hook.get_features()
                
                # ============================================================
                # PHASE 3: DOMAIN LOSS (GRL)
                # ============================================================
                loss_domain = torch.tensor(0.0, device=device, requires_grad=False)
                domain_acc = 0.0
                
                if opt.use_grl and epoch >= opt.grl_warmup:
                    if source_features is not None and target_features is not None:
                        domain_pred_source = domain_discriminator(source_features, current_grl_alpha)
                        domain_pred_target = domain_discriminator(target_features, current_grl_alpha)
                        loss_domain = compute_domain_loss(domain_pred_source, domain_pred_target) * opt.grl_weight
                        domain_acc = get_domain_accuracy(domain_pred_source, domain_pred_target)
                        
                        # Track domain accuracy for explainability
                        if domain_monitor:
                            domain_monitor.update_domain_accuracy(
                                domain_pred_source, domain_pred_target, epoch, i
                            )
                
                # ============================================================
                # PHASE 4: PSEUDO-LABELS & DISTILLATION
                # ============================================================
                # Teacher generates pseudo-labels from target-fake
                # Student predictions on target-real will be compared against these
                
                with torch.no_grad():
                    # Get teacher predictions on target FAKE domain
                    pred_teacher = model_teacher(imgs_target_fake)
                    
                    # Extract predictions tensor - format [B, 4+nc, N]
                    # YOLOv8 outputs: xywh (decoded) + class_scores (sigmoid applied)
                    # NMS handles: transpose, xywh2xyxy, filtering internally
                    pred_tensor = pred_teacher[0] if isinstance(pred_teacher, (list, tuple)) else pred_teacher
                    
                    # Adaptive confidence threshold (curriculum: starts low, increases)
                    adaptive_conf = get_adaptive_conf_thres(epoch, opt.epochs, opt.conf_thres)
                    
                    # NMS to get pseudo-labels
                    # Output format: list of [N, 6] tensors with [x1,y1,x2,y2,conf,cls]
                    pseudo_labels = non_max_suppression(
                        pred_tensor,
                        conf_thres=adaptive_conf,
                        iou_thres=opt.iou_thres,
                        max_det=300,
                    )
                    
                    # Clip bboxes to valid range [0, imgsz] and filter invalid
                    # Use target-real dimensions (same size as target-fake)
                    img_h, img_w = imgs_target.shape[2:]
                    for idx, preds in enumerate(pseudo_labels):
                        if preds is not None and len(preds) > 0:
                            # Clip coordinates
                            preds[:, 0].clamp_(0, img_w)  # x1
                            preds[:, 1].clamp_(0, img_h)  # y1
                            preds[:, 2].clamp_(0, img_w)  # x2
                            preds[:, 3].clamp_(0, img_h)  # y2
                            # Filter valid boxes (x2 > x1, y2 > y1, min size)
                            valid = (preds[:, 2] > preds[:, 0] + 2) & (preds[:, 3] > preds[:, 1] + 2)
                            pseudo_labels[idx] = preds[valid]
                    
                    # Count pseudo-labels
                    n_pseudo = sum(len(p) if p is not None else 0 for p in pseudo_labels)
                    
                    # Debug log (every 100 iterations)
                    if i % 100 == 0:
                        # Count predictions before class filtering (in fusion_da)
                        n_raw = sum(len(p) if p is not None else 0 for p in pseudo_labels)
                        if n_raw > 0:
                            all_preds = torch.cat([p for p in pseudo_labels if p is not None and len(p) > 0])
                            unique_cls = all_preds[:, 5].unique().tolist() if len(all_preds) > 0 else []
                            LOGGER.info(f'[Pseudo-Labels] conf={adaptive_conf:.3f}, count={n_pseudo}, classes={unique_cls}')
                        else:
                            # Debug: check raw teacher output
                            max_conf = pred_tensor[0, 4:].max().item() if len(pred_tensor.shape) == 3 else 0
                            LOGGER.info(f'[Pseudo-Labels] conf={adaptive_conf:.3f}, count=0, teacher_max_conf={max_conf:.4f}')
                
                # Compute distillation loss: Student(target-real) vs Teacher(target-fake) pseudo-labels
                # This transfers knowledge from style-transferred domain back to real domain
                loss_distillation = compute_loss.compute_distillation_loss(
                    pred_target, pseudo_labels, (img_h, img_w)  # pred_target = Student on target-real
                )
                
                # ============================================================
                # TOTAL LOSS (Công thức 4.7)
                # L = Ldet(Is) + Ldet(Isf) + α·Ldis + β·Lcon + Ldomain
                # ============================================================
                # Ensure all losses are scalars
                def ensure_scalar(x):
                    if isinstance(x, torch.Tensor) and x.numel() > 1:
                        return x.sum()
                    return x
                
                loss_source = ensure_scalar(loss_source)
                loss_source_fake = ensure_scalar(loss_source_fake)
                loss_distillation = ensure_scalar(loss_distillation)
                loss_domain = ensure_scalar(loss_domain)
                
                # ============================================================
                # CONSISTENCY LOSS (Công thức 4.5)
                # Lcon = ‖Ldet(Isource) - Ldet(Isource-fake)‖₂
                # Ensures model predictions are consistent between real and fake
                # 
                # NOTE: We detach loss_source_fake to avoid conflicting gradients.
                # This way, consistency loss only affects source_fake predictions
                # to become more like source_real predictions.
                # ============================================================
                loss_consistency = (loss_source.detach() - loss_source_fake) ** 2  # L2 distance
                consistency_weight = 2.0  # β = 2.0
                
                # Total Loss (Công thức 4.7):
                # L = Ldet(Is) + Ldet(Isf) + α·Ldis + β·Lcon + Ldomain(GRL)
                loss = (loss_source + loss_source_fake 
                        + loss_distillation * current_lambda 
                        + loss_consistency * consistency_weight
                        + loss_domain)
                
                # NaN detection - skip batch if loss is invalid
                if torch.isnan(loss) or torch.isinf(loss):
                    LOGGER.warning(f'[NaN] Loss is NaN/Inf at epoch {epoch}, iter {i}. Skipping batch.')
                    optimizer.zero_grad()
                    if grl_optimizer:
                        grl_optimizer.zero_grad()
                    continue
            
            # ================================================================
            # BACKWARD
            # ================================================================
            scaler.scale(loss).backward()

            # ================================================================
            # OPTIMIZER STEP (with proper scaler handling)
            # ================================================================
            # Always unscale the main optimizer first
            scaler.unscale_(optimizer)
            
            # Handle GRL optimizer separately if it exists and is active
            grl_optimizer_active = opt.use_grl and epoch >= opt.grl_warmup and grl_optimizer is not None
            if grl_optimizer_active:
                scaler.unscale_(grl_optimizer)

            # Gradient clipping (after unscale, before step)
            torch.nn.utils.clip_grad_norm_(model_student.parameters(), max_norm=5.0)  # Reduced from 10
            if opt.use_grl and domain_discriminator:
                torch.nn.utils.clip_grad_norm_(domain_discriminator.parameters(), max_norm=5.0)
            
            # Check for NaN in gradients - skip step if found
            grad_nan = False
            for p in model_student.parameters():
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    grad_nan = True
                    break
            
            if grad_nan:
                LOGGER.warning(f'[NaN Grad] Gradients contain NaN/Inf at epoch {epoch}, iter {i}. Skipping step.')
                optimizer.zero_grad()
                if grl_optimizer:
                    grl_optimizer.zero_grad()
                # Must update scaler to reset state after unscale_()
                scaler.update()
                continue

            # Step main optimizer
            scaler.step(optimizer)
            
            # Step GRL optimizer if active
            if grl_optimizer_active:
                scaler.step(grl_optimizer)

            # Update scaler ONCE after all steps
            scaler.update()

            # Zero gradients
            optimizer.zero_grad()
            if grl_optimizer_active:
                grl_optimizer.zero_grad()
            
            # Update teacher with EMA every iteration
            # NaN protection is handled inside teacher_ema.step()
            teacher_ema.step()
            
            # Clear CUDA cache more frequently to prevent OOM
            if i % 100 == 0:
                torch.cuda.empty_cache()
            
            # ================================================================
            # LOGGING
            # ================================================================
            loss_items = torch.tensor([
                loss_items_source[0].item() if isinstance(loss_items_source, (list, tuple)) else loss_source.item(),
                loss_items_source[1].item() if isinstance(loss_items_source, (list, tuple)) and len(loss_items_source) > 1 else 0,
                loss_items_source[2].item() if isinstance(loss_items_source, (list, tuple)) and len(loss_items_source) > 2 else 0,
                loss_distillation.item(),
                loss_domain.item(),
            ], device=device)
            
            mloss = (mloss * i + loss_items) / (i + 1)
            
            # Log to TrainingLogger
            logger.log_iteration(epoch, i, {
                'loss_sr': loss_source.item(),
                'loss_sf': loss_source_fake.item(),
                'loss_distill': loss_distillation.item(),
                'loss_consistency': loss_consistency.item(),
                'loss_domain': loss_domain.item(),
                'loss_total': loss.item(),
            }, extra={
                'lr': optimizer.param_groups[0]['lr'],
                'grl_alpha': current_grl_alpha if opt.use_grl else 0,
                'conf_thres': adaptive_conf,
                'lambda': current_lambda,
            }, global_step=global_step)
            global_step += 1
            
            # Collect features for explainability (every 50 iterations to save memory)
            # Limit collection to first 10 batches per epoch to avoid OOM
            if domain_monitor and i % 50 == 0 and i < 500:
                domain_monitor.collect_features(
                    features_sr=source_features,
                    features_tr=target_features,
                )
            
            mem = f'{torch.cuda.memory_reserved() / 1e9:.1f}G' if torch.cuda.is_available() else 'CPU'
            pbar.set_postfix({
                'mem': mem,
                'box': f'{mloss[0]:.4f}',
                'distill': f'{mloss[3]:.4f}',
                'domain': f'{mloss[4]:.4f}',
            })
        
        # ================================================================
        # END OF EPOCH
        # ================================================================
        scheduler.step()
        
        # CRITICAL: Process and clear collected features to prevent OOM
        if domain_monitor:
            try:
                domain_monitor.end_epoch(epoch)
            except Exception as e:
                LOGGER.warning(f'DomainMonitor end_epoch failed: {e}')
            # Force clear features even if end_epoch failed
            domain_monitor._epoch_features.clear()
            domain_monitor._epoch_labels.clear()
        
        # Aggressive memory cleanup at end of each epoch
        gc.collect()
        torch.cuda.empty_cache()

        # Validation every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == opt.epochs - 1:
            # Validate using state_dict copy (deepcopy fails with hooks)
            orig_model = yolo_student.model
            try:
                # Create a fresh model and load weights
                val_yolo = YOLO(opt.weights)  # Use module-level import
                val_model = val_yolo.model.to(device)
                val_model.load_state_dict(model_student.state_dict())
                val_model.eval()
                
                # Swap model for validation
                yolo_student.model = val_model
                
                metrics = yolo_student.val(data=opt.data, split='test', verbose=False)
                current_map50 = metrics.box.map50
                current_map = metrics.box.map

                LOGGER.info(f'Epoch {epoch}: mAP@50={current_map50:.4f}, mAP@50-95={current_map:.4f}')

                # Log epoch metrics
                logger.log_epoch(epoch, {
                    'mAP50': current_map50,
                    'mAP50-95': current_map,
                    'precision': metrics.box.mp,
                    'recall': metrics.box.mr,
                })

                # Save best
                if current_map50 > best_fitness:
                    best_fitness = current_map50
                    torch.save({
                        'epoch': epoch,
                        'model': model_student.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_fitness': best_fitness,
                    }, save_dir / 'weights' / 'best.pt')
            except Exception as e:
                LOGGER.warning(f'Validation failed (epoch {epoch}): {e}')
            finally:
                yolo_student.model = orig_model
                del val_model, val_yolo
                # Force garbage collection after validation
                gc.collect()
                torch.cuda.empty_cache()

        # Save last
        torch.save({
            'epoch': epoch,
            'model': model_student.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_fitness': best_fitness,
        }, save_dir / 'weights' / 'last.pt')
    
    # ========================================================================
    # CLEANUP
    # ========================================================================
    if feature_hook:
        feature_hook.remove()
    
    # Finalize logger
    logger.finalize()
    
    # Finalize domain monitor
    if domain_monitor:
        domain_monitor.finalize()
    
    hours = (time.time() - t0) / 3600
    LOGGER.info(f'\n{opt.epochs} epochs completed in {hours:.2f} hours.')
    LOGGER.info(f'Best mAP@50: {best_fitness:.4f}')
    LOGGER.info(f'Results saved to {save_dir}')
    LOGGER.info(f'Run visualization: python utils/visualize_training.py --log-dir {save_dir}')
    
    return best_fitness
# ============================================================================
# ARGUMENT PARSER
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='FDA Training')
    
    # Config file (optional - will be loaded first, then CLI args override)
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file (e.g., configs/train_config.yaml)')
    
    # Model
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='Pretrained weights')
    parser.add_argument('--data', type=str, default='data_v8.yaml', help='Dataset config')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    
    # Training
    parser.add_argument('--epochs', type=int, default=200, help='Total epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='0', help='CUDA device')
    parser.add_argument('--workers', type=int, default=8, help='Dataloader workers')
    
    # Optimizer
    parser.add_argument('--lr0', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01, help='Final learning rate factor')
    
    # Teacher-Student (OPTIMIZED)
    # teacher_alpha: 0.995 cân bằng giữa stability và adaptivity
    # conf_thres: 0.15 bắt đầu thấp (curriculum learning)
    # iou_thres: 0.45 standard NMS threshold
    # lambda_weight: 0.1 đủ mạnh để distillation có tác dụng
    parser.add_argument('--teacher-alpha', type=float, default=0.995, help='Teacher EMA decay (lower = faster adaptation)')
    parser.add_argument('--conf-thres', type=float, default=0.15, help='Base pseudo-label confidence (adaptive)')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--lambda-weight', type=float, default=0.1, help='Distillation loss weight')
    
    # Progressive training
    parser.add_argument('--use-progressive-lambda', action='store_true', help='Progressive lambda')
    parser.add_argument('--warmup-epochs', type=int, default=10, help='Warmup epochs for lambda')
    
    # GRL (OPTIMIZED for stability)
    # grl_warmup: 10 epochs để detection loss ổn định trước
    # grl_max_alpha: 0.5 tránh gradient explosion
    # grl_weight: 0.05 nhẹ nhàng để domain alignment không phá detection
    parser.add_argument('--use-grl', action='store_true', help='Enable GRL')
    parser.add_argument('--grl-warmup', type=int, default=10, help='GRL warmup epochs')
    parser.add_argument('--grl-max-alpha', type=float, default=0.5, help='Max GRL alpha (lower = more stable)')
    parser.add_argument('--grl-weight', type=float, default=0.05, help='GRL loss weight')
    parser.add_argument('--grl-hidden-dim', type=int, default=256, help='Discriminator hidden dim')
    
    # Output
    parser.add_argument('--project', type=str, default='runs/fda', help='Project dir')
    parser.add_argument('--name', type=str, default='exp', help='Experiment name')
    
    # Monitoring & Explainability
    parser.add_argument('--enable-monitoring', action='store_true', help='Enable domain monitoring (UMAP, MMD, etc.)')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # Load config from file if provided
    if args.config:
        from utils.config_loader import load_config, config_to_namespace, merge_cli_args
        config = load_config(args.config)
        config = merge_cli_args(config, args)
        args = config_to_namespace(config)
    
    print("=" * 70)
    print("FDA: Semi-Supervised Domain Adaptive Training")
    print("=" * 70)
    print(f"Config:  {args.config if hasattr(args, 'config') and args.config else 'CLI only'}")
    print(f"Model:   {args.weights}")
    print(f"Data:    {args.data}")
    print(f"Epochs:  {args.epochs}")
    print(f"Batch:   {args.batch}")
    print(f"Device:  cuda:{args.device}")
    print(f"GRL:     {'Enabled' if args.use_grl else 'Disabled'}")
    print(f"Warmup:  {args.warmup_epochs} epochs")
    print("=" * 70)
    
    train(args)