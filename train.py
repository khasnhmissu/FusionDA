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
    
    # Setup WeightEMA for teacher (timm-style implementation)
    # CRITICAL: WeightEMA creates a deepcopy internally, so we only pass student
    # The EMA model (teacher) starts as an exact copy of pretrained student
    # and is updated via EMA during training
    teacher_ema = WeightEMA(model_student, decay=opt.teacher_alpha, device=device)
    
    # Reference to teacher model for convenience (always in eval mode)
    model_teacher = teacher_ema.module
    
    LOGGER.info(f'Student params: {sum(p.numel() for p in model_student.parameters()):,}')
    LOGGER.info(f'Teacher (EMA) params: {sum(p.numel() for p in model_teacher.parameters()):,}')
    
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
    # COCO: 0=person, 2=car
    # Dataset: 0=person, 1=car
    # IMPORTANT: After EMA adaptation, teacher may output Dataset IDs (0, 1) instead of COCO IDs (0, 2)
    # So we include BOTH:
    #   - COCO IDs: 0->0, 2->1 (early training when teacher uses pretrained weights)
    #   - Dataset IDs: 0->0, 1->1 (after EMA adapts teacher to student's class output)
    class_mapping = getattr(opt, 'class_mapping', {0: 0, 1: 1, 2: 1})
    LOGGER.info(f'Class mapping (COCO/Dataset->Dataset): {class_mapping}')
    compute_loss = FDALoss(model_student, class_mapping=class_mapping)
    
    # CRITICAL: Disable AMP due to gradient underflow causing NaN in early layers
    # FP16 cannot represent very small gradients → underflow → scaler overflow → NaN
    # Re-enable with --amp flag after training stabilizes
    use_amp = getattr(opt, 'amp', False) and device.type != 'cpu'
    scaler = amp.GradScaler(enabled=use_amp)
    if not use_amp:
        LOGGER.info(colorstr('AMP: ') + 'Disabled for stability. Use --amp to enable.')
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
        # Teacher (EMA model) is ALWAYS in eval mode (enforced by WeightEMA)
        # This ensures correct output format (decoded boxes + sigmoid scores)
        # The call below is redundant but kept for clarity
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
            
            # VALIDATION: Ensure inputs are in valid range [0, 1]
            # Some augmented images may have values outside this range
            imgs_source = imgs_source.clamp(0, 1)
            imgs_source_fake = imgs_source_fake.clamp(0, 1)
            imgs_target = imgs_target.clamp(0, 1)
            imgs_target_fake = imgs_target_fake.clamp(0, 1)
            
            # ================================================================
            # FORWARD PASS (4 PHASES)
            # ================================================================
            with torch.amp.autocast('cuda', enabled=use_amp):
                
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
                    # DEBUG: Add hooks to find NaN-producing layer
                    nan_layer_info = {'first_nan_layer': None, 'input_stats': None}
                    hooks = []
                    
                    def make_nan_check_hook(name):
                        def hook(module, input, output):
                            if nan_layer_info['first_nan_layer'] is None:
                                out_tensor = output[0] if isinstance(output, tuple) else output
                                if isinstance(out_tensor, torch.Tensor) and torch.isnan(out_tensor).any():
                                    inp_tensor = input[0] if isinstance(input, tuple) else input
                                    inp_has_nan = torch.isnan(inp_tensor).any().item() if isinstance(inp_tensor, torch.Tensor) else False
                                    nan_layer_info['first_nan_layer'] = name
                                    nan_layer_info['input_has_nan'] = inp_has_nan
                                    if isinstance(inp_tensor, torch.Tensor):
                                        nan_layer_info['input_stats'] = f'min={inp_tensor.min():.4f}, max={inp_tensor.max():.4f}'
                        return hook
                    
                    # Only register hooks occasionally to avoid overhead
                    if i % 100 == 0:
                        for name, module in model_teacher.named_modules():
                            if len(list(module.children())) == 0:  # Leaf modules only
                                hooks.append(module.register_forward_hook(make_nan_check_hook(name)))
                    
                    # Get teacher predictions on target FAKE domain
                    pred_teacher = model_teacher(imgs_target_fake)
                    
                    # Remove hooks
                    for h in hooks:
                        h.remove()
                    
                    # Extract predictions tensor - format [B, 4+nc, N]
                    # YOLOv8 outputs: xywh (decoded) + class_scores (sigmoid applied)
                    # In train() mode, YOLOv8 may return dict with 'one2one' key
                    # NMS handles: transpose, xywh2xyxy, filtering internally
                    
                    # DEBUG: Log output format once at start
                    if i == 0 and epoch == 0:
                        LOGGER.info(f'[DEBUG] pred_teacher type: {type(pred_teacher).__name__}')
                        if isinstance(pred_teacher, dict):
                            LOGGER.info(f'[DEBUG] pred_teacher keys: {list(pred_teacher.keys())}')
                            for k, v in pred_teacher.items():
                                if isinstance(v, torch.Tensor):
                                    LOGGER.info(f'[DEBUG]   {k}: Tensor shape={v.shape}, dtype={v.dtype}')
                                elif isinstance(v, (list, tuple)) and len(v) > 0:
                                    LOGGER.info(f'[DEBUG]   {k}: {type(v).__name__} len={len(v)}, first={type(v[0]).__name__}')
                                    if isinstance(v[0], torch.Tensor):
                                        LOGGER.info(f'[DEBUG]     first tensor shape={v[0].shape}')
                        elif isinstance(pred_teacher, (list, tuple)):
                            LOGGER.info(f'[DEBUG] pred_teacher is {type(pred_teacher).__name__} len={len(pred_teacher)}')
                            if len(pred_teacher) > 0 and isinstance(pred_teacher[0], torch.Tensor):
                                LOGGER.info(f'[DEBUG]   first tensor shape={pred_teacher[0].shape}')
                    
                    if isinstance(pred_teacher, dict):
                        # Dict output (shouldn't happen in eval mode, but handle it)
                        # In train mode this has 'boxes', 'scores', 'feats' - not usable for NMS
                        # Try common keys first
                        pred_tensor = pred_teacher.get('one2one', pred_teacher.get('one2many', None))
                        if pred_tensor is not None:
                            pred_tensor = pred_tensor[0] if isinstance(pred_tensor, (list, tuple)) else pred_tensor
                        else:
                            # Fallback for train mode: cannot use this format for NMS
                            LOGGER.warning('[Teacher] Dict output detected - switching to eval mode may be needed')
                            pred_tensor = torch.zeros(1, 84, 8400, device=device)
                    elif isinstance(pred_teacher, (list, tuple)):
                        # Eval mode typically returns (tensor,) or [tensor]
                        pred_tensor = pred_teacher[0]
                    else:
                        # Direct tensor output
                        pred_tensor = pred_teacher
                    
                    # CRITICAL: Handle NaN/Inf in teacher output
                    # This can happen due to numerical instability with certain target-fake images
                    # Rather than letting NaN propagate, we skip distillation for this batch
                    teacher_output_valid = True
                    if pred_tensor is not None and isinstance(pred_tensor, torch.Tensor):
                        if torch.isnan(pred_tensor).any() or torch.isinf(pred_tensor).any():
                            teacher_output_valid = False
                            # Replace with zeros so NMS returns empty (no pseudo-labels)
                            pred_tensor = torch.zeros_like(pred_tensor)
                    else:
                        teacher_output_valid = False
                        pred_tensor = torch.zeros(1, 6, 8400, device=device)  # Dummy tensor
                    
                    # CRITICAL: In train mode, output is raw logits, NMS needs probabilities
                    # Apply sigmoid to class scores (indices 4:) if values are outside [0,1]
                    if teacher_output_valid and len(pred_tensor.shape) == 3:
                        cls_scores = pred_tensor[:, 4:, :]  # [B, nc, N]
                        if cls_scores.max() > 1.0 or cls_scores.min() < 0.0:
                            # Apply sigmoid to convert logits to probabilities
                            pred_tensor = torch.cat([
                                pred_tensor[:, :4, :],  # Keep box coords as-is
                                torch.sigmoid(cls_scores)  # Sigmoid for class scores
                            ], dim=1)
                    
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
                    
                    # CRITICAL: Get valid COCO class IDs from class_mapping
                    # Teacher outputs COCO classes (0-79), we only want mapped ones
                    valid_coco_classes = set(class_mapping.keys())  # e.g., {0, 2} for person, car
                    
                    for idx, preds in enumerate(pseudo_labels):
                        if preds is not None and len(preds) > 0:
                            # Clip coordinates
                            preds[:, 0].clamp_(0, img_w)  # x1
                            preds[:, 1].clamp_(0, img_h)  # y1
                            preds[:, 2].clamp_(0, img_w)  # x2
                            preds[:, 3].clamp_(0, img_h)  # y2
                            
                            # Filter valid boxes (x2 > x1, y2 > y1, min size)
                            valid_box = (preds[:, 2] > preds[:, 0] + 2) & (preds[:, 3] > preds[:, 1] + 2)
                            
                            # Filter by valid classes (only keep person/car from COCO)
                            cls_ids = preds[:, 5].long()
                            valid_cls = torch.zeros(len(preds), dtype=torch.bool, device=device)
                            for coco_cls in valid_coco_classes:
                                valid_cls |= (cls_ids == coco_cls)
                            
                            # Apply both filters
                            valid = valid_box & valid_cls
                            filtered_preds = preds[valid]
                            
                            # Map COCO class IDs to dataset class IDs
                            if len(filtered_preds) > 0:
                                for coco_cls, dataset_cls in class_mapping.items():
                                    mask = (filtered_preds[:, 5].long() == coco_cls)
                                    filtered_preds[mask, 5] = dataset_cls
                            
                            pseudo_labels[idx] = filtered_preds
                    
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
                            # pred_tensor shape: [B, 4+nc, N] 
                            has_nan = torch.isnan(pred_tensor).any().item()
                            has_inf = torch.isinf(pred_tensor).any().item()
                            
                            if has_nan or has_inf:
                                # CRITICAL: Teacher forward pass has numerical issues
                                LOGGER.warning(f'[TEACHER DEBUG] pred_tensor has NaN={has_nan}, Inf={has_inf}')
                                
                                # Check input image
                                img_nan = torch.isnan(imgs_target_fake).any().item()
                                LOGGER.warning(f'[TEACHER DEBUG] imgs_target_fake has NaN={img_nan}')
                                
                                # Check first conv layer weights specifically
                                first_conv = None
                                for name, module in model_teacher.named_modules():
                                    if 'model.0.conv' in name and hasattr(module, 'weight'):
                                        first_conv = module
                                        w = module.weight.data
                                        LOGGER.warning(f'[TEACHER DEBUG] model.0.conv weight stats: '
                                                      f'min={w.min():.6f}, max={w.max():.6f}, '
                                                      f'mean={w.mean():.6f}, std={w.std():.6f}')
                                        break
                                
                                # Check teacher weights (PARAMETERS)
                                teacher_nan_count = 0
                                teacher_inf_count = 0
                                for name, param in model_teacher.named_parameters():
                                    if torch.isnan(param.data).any():
                                        teacher_nan_count += 1
                                        if teacher_nan_count <= 3:
                                            LOGGER.warning(f'[TEACHER DEBUG] NaN in param: {name}')
                                    if torch.isinf(param.data).any():
                                        teacher_inf_count += 1
                                        if teacher_inf_count <= 3:
                                            LOGGER.warning(f'[TEACHER DEBUG] Inf in param: {name}')
                                
                                LOGGER.warning(f'[TEACHER DEBUG] Total NaN params: {teacher_nan_count}, Inf params: {teacher_inf_count}')
                                
                                # Check teacher BUFFERS (BatchNorm running_mean, running_var)
                                # These are NOT updated by EMA and can accumulate NaN!
                                buffer_nan_count = 0
                                buffer_inf_count = 0
                                for name, buf in model_teacher.named_buffers():
                                    if torch.isnan(buf).any():
                                        buffer_nan_count += 1
                                        if buffer_nan_count <= 3:
                                            LOGGER.warning(f'[TEACHER DEBUG] NaN in BUFFER: {name}')
                                    if torch.isinf(buf).any():
                                        buffer_inf_count += 1
                                        if buffer_inf_count <= 3:
                                            LOGGER.warning(f'[TEACHER DEBUG] Inf in BUFFER: {name}')
                                
                                LOGGER.warning(f'[TEACHER DEBUG] Total NaN buffers: {buffer_nan_count}, Inf buffers: {buffer_inf_count}')
                                LOGGER.warning(f'[TEACHER DEBUG] EMA step_count: {teacher_ema.step_count}, nan_count: {teacher_ema.nan_count}')
                                
                                # Log which layer first produced NaN
                                if nan_layer_info['first_nan_layer']:
                                    LOGGER.warning(f"[TEACHER DEBUG] FIRST NaN LAYER: {nan_layer_info['first_nan_layer']}")
                                    LOGGER.warning(f"[TEACHER DEBUG] Input to that layer had NaN: {nan_layer_info.get('input_has_nan', 'unknown')}")
                                    LOGGER.warning(f"[TEACHER DEBUG] Input stats: {nan_layer_info.get('input_stats', 'unknown')}")
                                
                                max_conf = float('nan')
                            else:
                                # Teacher output is valid but no person/car detections above threshold
                                # Show max confidence for person (class 0) and car (class 2) after sigmoid
                                if len(pred_tensor.shape) == 3:
                                    # pred_tensor already has sigmoid applied if it was logits
                                    person_conf = pred_tensor[0, 4, :].max().item()  # class 0 = person
                                    car_conf = pred_tensor[0, 6, :].max().item() if pred_tensor.shape[1] > 6 else 0  # class 2 = car
                                    max_conf = max(person_conf, car_conf)
                                else:
                                    max_conf = 0
                            
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
                # Clamp difference to prevent squared overflow
                loss_diff = torch.clamp(loss_source.detach() - loss_source_fake, -10, 10)
                loss_consistency = loss_diff ** 2  # L2 distance
                consistency_weight = 1.0  # β = 1.0 (reduced from 2.0 for stability)
                
                # Total Loss (Công thức 4.7):
                # L = Ldet(Is) + Ldet(Isf) + α·Ldis + β·Lcon + Ldomain(GRL)
                loss = (loss_source + loss_source_fake 
                        + loss_distillation * current_lambda 
                        + loss_consistency * consistency_weight
                        + loss_domain)
                
                # Final safety clamp to prevent extreme values
                loss = torch.clamp(loss, 0, 500)
                
                # ============================================================
                # DEBUG: Detailed NaN detection for each loss component
                # ============================================================
                nan_sources = []
                if torch.isnan(loss_source) or torch.isinf(loss_source):
                    nan_sources.append(f'loss_source={loss_source.item():.4f}')
                if torch.isnan(loss_source_fake) or torch.isinf(loss_source_fake):
                    nan_sources.append(f'loss_source_fake={loss_source_fake.item():.4f}')
                if torch.isnan(loss_distillation) or torch.isinf(loss_distillation):
                    nan_sources.append(f'loss_distillation={loss_distillation.item():.4f}')
                if torch.isnan(loss_consistency) or torch.isinf(loss_consistency):
                    nan_sources.append(f'loss_consistency={loss_consistency.item():.4f}')
                if torch.isnan(loss_domain) or torch.isinf(loss_domain):
                    nan_sources.append(f'loss_domain={loss_domain.item():.4f}')
                
                # Check input images for NaN
                if torch.isnan(imgs_source).any():
                    nan_sources.append('imgs_source')
                if torch.isnan(imgs_target).any():
                    nan_sources.append('imgs_target')
                if torch.isnan(imgs_source_fake).any():
                    nan_sources.append('imgs_source_fake')
                if torch.isnan(imgs_target_fake).any():
                    nan_sources.append('imgs_target_fake')
                
                # Check model predictions for NaN
                if isinstance(pred_source, dict):
                    pred_source_tensor = pred_source.get('one2one', pred_source.get('one2many', None))
                    if pred_source_tensor is not None:
                        pred_source_tensor = pred_source_tensor[0] if isinstance(pred_source_tensor, (list, tuple)) else pred_source_tensor
                elif isinstance(pred_source, (list, tuple)):
                    pred_source_tensor = pred_source[0]
                else:
                    pred_source_tensor = pred_source
                
                if pred_source_tensor is not None and isinstance(pred_source_tensor, torch.Tensor):
                    if torch.isnan(pred_source_tensor).any():
                        nan_sources.append('pred_source')
                
                # Log NaN sources if any found
                if nan_sources:
                    LOGGER.warning(f'[NaN DEBUG] iter {i}: NaN found in: {", ".join(nan_sources)}')
                    LOGGER.warning(f'[NaN DEBUG] Loss values: SR={loss_source.item():.4f}, SF={loss_source_fake.item():.4f}, '
                                   f'Dist={loss_distillation.item():.4f}, Con={loss_consistency.item():.4f}, Dom={loss_domain.item():.4f}')
                
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
            torch.nn.utils.clip_grad_norm_(model_student.parameters(), max_norm=2.0)  # Reduced from 5.0 for stability
            if opt.use_grl and domain_discriminator:
                torch.nn.utils.clip_grad_norm_(domain_discriminator.parameters(), max_norm=2.0)
            
            # Check for NaN in gradients - skip step if found
            grad_nan = False
            nan_layer = None
            max_grad = 0.0
            for name, p in model_student.named_parameters():
                if p.grad is not None:
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        grad_nan = True
                        nan_layer = name
                        break
                    grad_max = p.grad.abs().max().item()
                    if grad_max > max_grad:
                        max_grad = grad_max
            
            if grad_nan:
                LOGGER.warning(f'[NaN Grad] Gradients contain NaN/Inf at epoch {epoch}, iter {i}. Layer: {nan_layer}')
                LOGGER.warning(f'[NaN Grad] Last valid max_grad={max_grad:.4f}')
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
            
            # Update teacher with EMA - but only after warmup period
            # Student needs to learn basic patterns before EMA starts
            # Otherwise student's random weights will corrupt pretrained teacher
            ema_warmup_epochs = getattr(opt, 'ema_warmup_epochs', 1)  # Default: wait 1 epoch
            if epoch >= ema_warmup_epochs:
                # timm-style EMA update - pass student model directly
                teacher_ema.update(model_student)
            elif i == 0:  # Log once per epoch during warmup
                LOGGER.info(f'[EMA] Warmup: keeping pretrained teacher until epoch {ema_warmup_epochs}')
            
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
    parser.add_argument('--lr0', type=float, default=0.0001, help='Initial learning rate (reduced for stability)')
    parser.add_argument('--lrf', type=float, default=0.01, help='Final learning rate factor')
    
    # Teacher-Student (OPTIMIZED)
    # teacher_alpha: 0.995 cân bằng giữa stability và adaptivity
    # conf_thres: 0.15 bắt đầu thấp (curriculum learning)
    # iou_thres: 0.45 standard NMS threshold
    # lambda_weight: 0.1 đủ mạnh để distillation có tác dụng
    parser.add_argument('--teacher-alpha', type=float, default=0.995, help='Teacher EMA decay (lower = faster adaptation)')
    parser.add_argument('--conf-thres', type=float, default=0.35, help='Base pseudo-label confidence (adaptive)')
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
    parser.add_argument('--grl-max-alpha', type=float, default=0.3, help='Max GRL alpha (lower = more stable)')
    parser.add_argument('--grl-weight', type=float, default=0.05, help='GRL loss weight')
    parser.add_argument('--grl-hidden-dim', type=int, default=256, help='Discriminator hidden dim')
    
    # Output
    parser.add_argument('--project', type=str, default='runs/fda', help='Project dir')
    parser.add_argument('--name', type=str, default='exp', help='Experiment name')
    
    # Monitoring & Explainability
    parser.add_argument('--enable-monitoring', action='store_true', help='Enable domain monitoring (UMAP, MMD, etc.)')
    parser.add_argument('--amp', action='store_true', help='Enable AMP (Mixed Precision) - may cause NaN with complex losses')
    
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