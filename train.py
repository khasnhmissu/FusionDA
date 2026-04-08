"""FDA Training - Semi-Supervised Domain Adaptive Training for YOLO26/YOLOv8"""
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
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.nms import non_max_suppression

from domain_adaptation import (
    DomainDiscriminator, YOLOv8FeatureHook, find_last_backbone_layer,
    MultiScaleFusedDiscriminator, MultiScaleFeatureHook, find_multiscale_layers,
    compute_domain_loss,
    get_grl_alpha, get_domain_accuracy,
)
from fusion_da import FDALoss, WeightEMA, PairedMultiDomainDataset
from utils.FDA_helpers import (
    get_adaptive_conf_thres,
    filter_pseudo_labels_by_uncertainty,
    get_progressive_lambda,
)
from utils.training_logger import TrainingLogger
from utils.domain_monitor import DomainMonitor
import cv2
import numpy as np


def save_debug_image(img_tensor, pseudo_labels, save_path, names, conf_thres=0.5):
    """Save debug image with bounding boxes."""
    if isinstance(img_tensor, torch.Tensor):
        img = img_tensor.detach().cpu().numpy()
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img = img_tensor
    
    img_draw = img.copy()
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
    total_count = 0
    
    if pseudo_labels is not None:
        for preds in pseudo_labels:
            if preds is not None and len(preds) > 0:
                for pred in preds:
                    x1, y1, x2, y2, conf, cls = pred[:6].cpu().numpy()
                    if conf < conf_thres:
                        continue
                    total_count += 1
                    cls = int(cls)
                    color = colors[cls % len(colors)]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
                    cls_name = names[cls] if cls < len(names) else f'cls{cls}'
                    label = f'{cls_name}: {conf:.2f}'
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(img_draw, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
                    cv2.putText(img_draw, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.putText(img_draw, f'Count: {total_count} | Conf >= {conf_thres:.2f}', 
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), img_draw)
    return total_count


def train(opt):
    """Main training function."""
    device = torch.device(f'cuda:{opt.device}' if opt.device.isdigit() else opt.device)
    save_dir = Path(opt.project) / opt.name
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / 'weights').mkdir(exist_ok=True)
    
    # Load data config
    with open(opt.data, encoding='utf-8') as f:
        data_dict = yaml.safe_load(f)
    
    nc = data_dict['nc']
    names = data_dict['names']
    LOGGER.info(colorstr('FDA: ') + f'Training with {nc} classes')
    
    # Student model
    if getattr(opt, 'cfg', None):
        yolo_student = YOLO(opt.cfg)
        if opt.weights and opt.weights != 'None':
            yolo_student.load(opt.weights)
    else:
        yolo_student = YOLO(opt.weights)
        
    model_student = yolo_student.model.to(device)
    for param in model_student.parameters():
        param.requires_grad = True
    
    # Teacher model (EMA)
    # θ_teacher = α * θ_teacher + (1-α) * θ_student
    # Update called AFTER optimizer.step()
    # update_after_step=500: Let student learn from GT before EMA updates
    teacher_ema = WeightEMA(
        model_student,
        alpha=opt.teacher_alpha,
        freeze_teacher=getattr(opt, 'freeze_teacher', False),
        device=device,
        update_after_step=getattr(opt, 'update_after_step', 2000),
        alpha_rampup_steps=getattr(opt, 'alpha_rampup_steps', 5000),
    )
    model_teacher = teacher_ema.module
    
    LOGGER.info(f'Student params: {sum(p.numel() for p in model_student.parameters()):,}')
    LOGGER.info(f'Teacher params: {sum(p.numel() for p in model_teacher.parameters()):,}')
    
    # ── Feature MSE Distillation hook (Teacher backbone end, độc lập với GRL) ──
    teacher_feature_hook = None
    if getattr(opt, 'use_feature_distill', False):
        teacher_backbone_idx = find_last_backbone_layer(model_teacher)
        teacher_feature_hook = YOLOv8FeatureHook(model_teacher, layer_idx=teacher_backbone_idx)
        LOGGER.info(
            f'{colorstr("FeatureDistill:")} Teacher hook tại layer {teacher_backbone_idx} '
            f'(lambda_feature={getattr(opt, "lambda_feature", 0.05)})'
        )
    
    # ── GRL / Domain-Adaptation setup ────────────────────────────────────
    domain_discriminator = None   # single-scale  DomainDiscriminator
    ms_discriminator    = None    # multi-scale   MultiScaleDomainDiscriminator
    feature_hook        = None    # single-scale  YOLOv8FeatureHook
    ms_feature_hook     = None    # multi-scale   MultiScaleFeatureHook
    grl_optimizer       = None
    monitor_hook        = None    # monitoring-only hook (UMAP/MMD)
    use_multiscale_grl  = getattr(opt, 'use_multiscale_grl', False)

    # Always create a monitoring hook for feature collection (UMAP/MMD)
    if opt.enable_monitoring:
        backbone_layer_idx = find_last_backbone_layer(model_student)
        monitor_hook = YOLOv8FeatureHook(model_student, layer_idx=backbone_layer_idx)
        LOGGER.info(f'{colorstr("Monitor:")} Feature hook for UMAP/MMD (layer {backbone_layer_idx})')

    if opt.use_grl:
        with torch.no_grad():
            test_img = torch.zeros(1, 3, opt.imgsz, opt.imgsz, device=device)

        if use_multiscale_grl:
            # ── Multi-scale discriminator  (P3 / P4 / P5) ──────────────────
            ms_layer_indices = find_multiscale_layers(model_student)
            LOGGER.info(f'{colorstr("GRL [multi-scale]:")} hooks at layers {ms_layer_indices}')

            ms_feature_hook = MultiScaleFeatureHook(model_student, ms_layer_indices)

            # Probe channel sizes
            with torch.no_grad():
                _ = model_student(test_img)
                probe_feats = ms_feature_hook.get_features()
                in_ch_list = [
                    f.shape[1] if f is not None else 256
                    for f in probe_feats
                ]
            LOGGER.info(f'{colorstr("GRL [multi-scale]:")} in_channels per scale = {in_ch_list}')
            LOGGER.info(
                f'{colorstr("GRL [multi-scale]:")} P3=layer{ms_layer_indices[0]}, '
                f'P4=layer{ms_layer_indices[1]}, P5=layer{ms_layer_indices[2]} '
                f'(PAN outputs — post-FPN fusion)'
            )

            ms_discriminator = MultiScaleFusedDiscriminator(
                in_channels_list=in_ch_list,
                hidden_dim=opt.grl_hidden_dim,
                dropout=getattr(opt, 'grl_dropout', 0.3),
            ).to(device)
            grl_optimizer = optim.Adam(
                ms_discriminator.parameters(),
                lr=getattr(opt, 'grl_lr', 1e-4),
            )
        else:
            # ── Single-scale discriminator  (backbone end) ─────────────────
            if monitor_hook:
                feature_hook = monitor_hook   # reuse monitoring hook
            else:
                backbone_layer_idx = find_last_backbone_layer(model_student)
                feature_hook = YOLOv8FeatureHook(model_student, layer_idx=backbone_layer_idx)

            with torch.no_grad():
                _ = model_student(test_img)
                test_feat = feature_hook.get_features()
                in_channels = test_feat.shape[1] if test_feat is not None else 256

            domain_discriminator = DomainDiscriminator(
                in_channels=in_channels,
                hidden_dim=opt.grl_hidden_dim,
                dropout=getattr(opt, 'grl_dropout', 0.3),
            ).to(device)
            grl_optimizer = optim.Adam(
                domain_discriminator.parameters(),
                lr=getattr(opt, 'grl_lr', 1e-4),
            )
            LOGGER.info(f'{colorstr("GRL [single-scale]:")} in_channels={in_channels}')
    
    # Checkpoint epochs for pseudo-label quality analysis
    checkpoint_epochs = set(getattr(opt, 'checkpoint_epochs', [0, 50, 100, 150]))
    checkpoint_epochs.add(opt.epochs - 1)  # Always save final
    LOGGER.info(f'Saving intermediate checkpoints at epochs: {sorted(checkpoint_epochs)}')
    
    # Optimizer & Scheduler
    optimizer = optim.AdamW(model_student.parameters(), lr=opt.lr0, weight_decay=0.0005)
    lf = lambda x: ((1 - math.cos(x * math.pi / opt.epochs)) / 2) * (opt.lrf - 1) + 1
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    # Data paths
    root = Path(data_dict.get('path', ''))
    
    def get_paths(key, fallback=[]):
        paths = data_dict.get(key, fallback)
        if isinstance(paths, str):
            paths = [paths]
        return [str(root / p) for p in paths] if paths else []
    
    source_real = get_paths('train_source_real', data_dict.get('train', []))
    source_fake = get_paths('train_source_fake')
    target_real = get_paths('train_target_real')
    target_fake = get_paths('train_target_fake')
    
    LOGGER.info(f'Source Real: {source_real}')
    LOGGER.info(f'Source Fake: {source_fake}')
    LOGGER.info(f'Target Real: {target_real}')
    LOGGER.info(f'Target Fake: {target_fake}')
    
    # Dataloader
    from ultralytics.data import YOLODataset
    from torch.utils.data import DataLoader
    from ultralytics.cfg import get_cfg
    
    gs = max(int(model_student.stride.max()), 32)
    default_cfg = get_cfg()
    
    paired_dataset = PairedMultiDomainDataset(
        source_real_path=source_real[0] if source_real else '',
        source_fake_path=source_fake[0] if source_fake else None,
        target_real_path=target_real[0] if target_real else None,
        target_fake_path=target_fake[0] if target_fake else None,
        imgsz=opt.imgsz,
        augment=False,
        hyp=default_cfg,
        data=data_dict,
        stride=gs,
    )
    
    paired_loader = DataLoader(
        paired_dataset,
        batch_size=opt.batch,
        shuffle=True,
        num_workers=opt.workers,
        pin_memory=True,
        collate_fn=PairedMultiDomainDataset.collate_fn,
    )
    
    nb = len(paired_loader)
    LOGGER.info(colorstr('FDA: ') + f'{len(paired_dataset)} samples, {nb} batches')
    
    # Loss function
    class_mapping = getattr(opt, 'class_mapping', {0: 0, 1: 1, 2: 1})
    LOGGER.info(f'Class mapping: {class_mapping}')
    compute_loss = FDALoss(
        model_student,
        class_mapping         = class_mapping,
        box_gain              = getattr(opt, 'box_gain', 7.5),
        cls_gain              = getattr(opt, 'cls_gain', 0.5),
    )
    
    # AMP
    use_amp = getattr(opt, 'amp', False) and device.type != 'cpu'
    scaler = amp.GradScaler(enabled=use_amp)
    if not use_amp:
        LOGGER.info(colorstr('AMP: ') + 'Disabled')
    
    # Logger
    tuning_mode = getattr(opt, 'tuning_mode', False)
    logger = TrainingLogger(
        save_dir=str(save_dir),
        project_name='FusionDA',
        use_tensorboard=not tuning_mode,  # Disable TB during tuning
        verbose=not tuning_mode,
    )
    logger.log_config({
        'weights': opt.weights, 'data': opt.data, 'epochs': opt.epochs,
        'batch_size': opt.batch, 'imgsz': opt.imgsz, 'lr0': opt.lr0,
        'teacher_alpha': opt.teacher_alpha, 'conf_thres': opt.conf_thres,
        'lambda_weight': opt.lambda_weight, 'use_grl': opt.use_grl,
        'grl_weight': opt.grl_weight if opt.use_grl else 0,
        'nc': nc, 'names': names,
    })
    
    # Domain Monitor
    domain_monitor = None
    if opt.enable_monitoring:
        if opt.epochs <= 50:
            umap_epochs = [0, opt.epochs // 2, opt.epochs - 1]
        else:
            step = opt.epochs // 4
            umap_epochs = [0, step, step * 2, step * 3, opt.epochs - 1]
        
        domain_monitor = DomainMonitor(
            save_dir=str(save_dir),
            umap_epochs=umap_epochs,
            tsne_epochs=[],
            verbose=True,
        )
        domain_monitor.set_total_epochs(opt.epochs)
        LOGGER.info(f'{colorstr("DomainMonitor:")} UMAP at epochs {umap_epochs}')
    
    # Training loop
    LOGGER.info(f'\n{colorstr("Starting training:")} {opt.epochs} epochs...\n')
    
    best_fitness = 0.0
    t0 = time.time()
    global_step = 0
    burn_in_epochs = getattr(opt, 'burn_in_epochs', 10)
    source_loss_baseline = None  # Track source loss baseline for teacher validation
    prev_domain_acc = 0.5        # Track domain accuracy across iterations for adaptive GRL weight
    
    for epoch in range(opt.epochs):
        model_student.train()
        model_teacher.eval()
        
        if opt.use_grl:
            if use_multiscale_grl and ms_discriminator is not None:
                ms_discriminator.train()
            elif domain_discriminator is not None:
                domain_discriminator.train()
            current_grl_alpha = get_grl_alpha(epoch, opt.epochs, opt.grl_warmup, opt.grl_max_alpha)
        
        if opt.use_progressive_lambda:
            current_lambda = get_progressive_lambda(epoch, opt.epochs, opt.warmup_epochs, 0.0, opt.lambda_weight)
        else:
            current_lambda = opt.lambda_weight
        
        mloss = torch.zeros(5, device=device)
        paired_iter = iter(paired_loader)
        pbar = tqdm(range(nb), desc=f'Epoch {epoch}/{opt.epochs-1}')
        optimizer.zero_grad()
        
        for i in pbar:
            try:
                batch = next(paired_iter)
            except StopIteration:
                paired_iter = iter(paired_loader)
                batch = next(paired_iter)
            
            # Load all 4 domains
            imgs_source = batch['source_real']['img'].to(device).float() / 255.0
            imgs_source_fake = batch['source_fake']['img'].to(device).float() / 255.0
            imgs_target = batch['target_real']['img'].to(device).float() / 255.0
            imgs_target_fake = batch['target_fake']['img'].to(device).float() / 255.0
            batch_source = batch['source_real']
            
            imgs_source = imgs_source.clamp(0, 1)
            imgs_source_fake = imgs_source_fake.clamp(0, 1)
            imgs_target = imgs_target.clamp(0, 1)
            imgs_target_fake = imgs_target_fake.clamp(0, 1)
            
            # Init multi-scale feature placeholders (avoids NameError before warmup)
            ms_src_feats      = [None] * (len(ms_feature_hook.hooks) if ms_feature_hook else 0)
            ms_src_fake_feats = [None] * (len(ms_feature_hook.hooks) if ms_feature_hook else 0)
            ms_tgt_feats      = [None] * (len(ms_feature_hook.hooks) if ms_feature_hook else 0)
            
            # Feature MSE distillation placeholder
            teacher_feat_for_mse = None

            with torch.amp.autocast('cuda', enabled=use_amp):
                # Source forward
                pred_source = model_student(imgs_source)
                loss_source, loss_items_source = compute_loss(pred_source, batch_source)

                # ── Collect source features ────────────────────────────────
                source_features = None
                if use_multiscale_grl and ms_feature_hook:
                    ms_src_feats = ms_feature_hook.get_features()
                elif feature_hook:
                    source_features = feature_hook.get_features()
                elif monitor_hook:
                    source_features = monitor_hook.get_features()

                # Source-fake forward
                pred_source_fake = model_student(imgs_source_fake)
                loss_source_fake, _ = compute_loss(pred_source_fake, batch_source)

                source_fake_features = None
                if use_multiscale_grl and ms_feature_hook:
                    ms_src_fake_feats = ms_feature_hook.get_features()
                elif feature_hook:
                    source_fake_features = feature_hook.get_features()
                elif monitor_hook:
                    source_fake_features = monitor_hook.get_features()

                # Target forward
                pred_target = model_student(imgs_target)

                target_features = None
                if use_multiscale_grl and ms_feature_hook:
                    ms_tgt_feats = ms_feature_hook.get_features()
                elif feature_hook:
                    target_features = feature_hook.get_features()
                elif monitor_hook:
                    target_features = monitor_hook.get_features()

                # ── Domain adversarial loss (GRL) ─────────────────────────
                loss_domain = torch.tensor(0.0, device=device, requires_grad=False)
                domain_acc  = 0.0

                if opt.use_grl and epoch >= opt.grl_warmup:
                    # Use prev iteration's domain_acc to decide GRL weight boost.
                    # (domain_acc is reset to 0.0 each step before being computed below,
                    #  so we must look at the previous iteration's value.)
                    effective_grl_weight = opt.grl_weight
                    if prev_domain_acc > 0.75:
                        # Discriminator winning: boost GRL adversarial signal.
                        # Trigger at 0.75 (earlier than 0.85) so backbone gets
                        # pushed before discriminator fully dominates.
                        boost = 1.0 + 2.0 * (prev_domain_acc - 0.75) / 0.25
                        effective_grl_weight = min(opt.grl_weight * boost, opt.grl_weight * 3.0)

                    if use_multiscale_grl and ms_discriminator is not None:
                        # ── Multi-scale fused path  (MF-DANN) ─────────────────
                        # MultiScaleFusedDiscriminator: GAP each backbone scale →
                        # concat → ONE shared MLP → [B, 1] scalar.
                        # Backbone only needs to fool a single discriminator
                        # → domain confusion is achievable, domain_acc → ~0.5.
                        if ms_feature_hook.all_available(ms_src_feats) and \
                           ms_feature_hook.all_available(ms_tgt_feats):
                            ms_discriminator.set_alpha(current_grl_alpha)
                            fused_src_logit = ms_discriminator(ms_src_feats, current_grl_alpha)  # [B,1]
                            fused_tgt_logit = ms_discriminator(ms_tgt_feats, current_grl_alpha)  # [B,1]
                            # Single unified loss — no / n_scales needed (fused output is already scalar)
                            loss_domain = compute_domain_loss(
                                fused_src_logit, fused_tgt_logit
                            ) * effective_grl_weight
                            domain_acc = get_domain_accuracy(fused_src_logit, fused_tgt_logit)
                            if domain_monitor:
                                domain_monitor.update_domain_accuracy(
                                    fused_src_logit, fused_tgt_logit, epoch, i
                                )
                    else:
                        # Single-scale path
                        if source_features is not None and target_features is not None:
                            domain_pred_source = domain_discriminator(source_features, current_grl_alpha)
                            domain_pred_target = domain_discriminator(target_features, current_grl_alpha)
                            loss_domain = compute_domain_loss(
                                domain_pred_source, domain_pred_target
                            ) * effective_grl_weight
                            domain_acc = get_domain_accuracy(domain_pred_source, domain_pred_target)
                            if domain_monitor:
                                domain_monitor.update_domain_accuracy(
                                    domain_pred_source, domain_pred_target, epoch, i
                                )
                
                # ── Burn-in check: no pseudo labels during early epochs ──
                img_h, img_w = imgs_target.shape[2:]
                
                if epoch < burn_in_epochs:
                    # During burn-in: student learns ONLY from GT (source)
                    # No pseudo labels to prevent confirmation bias
                    pseudo_labels = [torch.zeros(0, 6, device=device)
                                     for _ in range(imgs_target.shape[0])]
                    n_pseudo = 0
                    adaptive_conf = 1.0  # For logging
                    
                    if i == 0 and epoch == 0:
                        LOGGER.info(f'[BURN-IN] No pseudo labels for first {burn_in_epochs} epochs')
                else:
                    # ── Self-validation gating (every 5 epochs, first batch) ──
                    # Check if teacher still produces reasonable output on source
                    if epoch % 5 == 0 and i == 0:
                        with torch.no_grad():
                            teacher_pred_source = model_teacher(imgs_source)
                            teacher_source_loss, _ = compute_loss(teacher_pred_source, batch_source)
                            teacher_source_loss_val = teacher_source_loss.item() if isinstance(teacher_source_loss, torch.Tensor) else teacher_source_loss
                            
                            if source_loss_baseline is None:
                                source_loss_baseline = teacher_source_loss_val
                                LOGGER.info(f'[GATE] Teacher source loss baseline: {source_loss_baseline:.4f}')
                            elif source_loss_baseline > 0 and teacher_source_loss_val > source_loss_baseline * 2.0:
                                LOGGER.warning(
                                    f'[GATE] Teacher degraded! Loss {teacher_source_loss_val:.4f} > '
                                    f'2x baseline {source_loss_baseline:.4f}. Pausing EMA for 500 steps.'
                                )
                                teacher_ema.pause_updates(steps=500)
                            else:
                                LOGGER.info(
                                    f'[GATE] Teacher OK: loss={teacher_source_loss_val:.4f} '
                                    f'(baseline={source_loss_baseline:.4f})'
                                )
                    
                    # ── Pseudo-labels from teacher ──
                    with torch.no_grad():
                        pred_teacher = model_teacher(imgs_target_fake)
                        
                        # Thu thập feature của Teacher trên ảnh target-fake
                        # (dùng cho Feature MSE Distillation: MSE(Student(src_real), Teacher(tgt_fake)))
                        if teacher_feature_hook is not None:
                            teacher_feat_for_mse = teacher_feature_hook.get_features()
                        
                        if isinstance(pred_teacher, dict):
                            # Ưu tiên lấy 'one2many' cho Knowledge Distillation (dùng kèm NMS)
                            # 'one2one' (end-to-end) thường rất sparse và confidence thấp do Hungarian Matching.
                            pred_tensor = pred_teacher.get('one2many', pred_teacher.get('one2one', None))
                            if pred_tensor is not None:
                                pred_tensor = pred_tensor[0] if isinstance(pred_tensor, (list, tuple)) else pred_tensor
                            else:
                                pred_tensor = torch.zeros(1, 84, 10, device=device)
                        elif isinstance(pred_teacher, (list, tuple)):
                            pred_tensor = pred_teacher[0]
                        else:
                            pred_tensor = pred_teacher
                        
                        teacher_output_valid = True
                        if pred_tensor is not None and isinstance(pred_tensor, torch.Tensor):
                            if torch.isnan(pred_tensor).any() or torch.isinf(pred_tensor).any():
                                teacher_output_valid = False
                                pred_tensor = torch.zeros_like(pred_tensor)
                        else:
                            teacher_output_valid = False
                            # Fallback tensor small length (10) since 8400 is no longer valid for P2
                            pred_tensor = torch.zeros(1, 6, 10, device=device)
                        
                        # Tensor trước khi vào NMS phải ở chuẩn Ultralytics: [B, 4+nc, N]
                        # Nếu đang ở dạng [B, N, 4+nc] (ví dụ: [1, 8400, 84]), ta cần permute lại
                        if teacher_output_valid and len(pred_tensor.shape) == 3:
                            if pred_tensor.shape[-1] != 6:  # Bỏ qua nếu đã là End-to-End inference format [B, 300, 6]
                                # N > 4+nc => shape[1] > shape[-1]
                                if pred_tensor.shape[1] > pred_tensor.shape[-1]:
                                    pred_tensor = pred_tensor.permute(0, 2, 1)
                                
                                # Apply sigmoid nếu cần
                                cls_scores = pred_tensor[:, 4:, :]
                                if cls_scores.max() > 1.0 or cls_scores.min() < 0.0:
                                    pred_tensor = torch.cat([
                                        pred_tensor[:, :4, :],
                                        torch.sigmoid(cls_scores)
                                    ], dim=1)
                        
                        # Adaptive confidence threshold (curriculum learning)
                        # After burn-in: start strict (max_conf) and relax over time
                        adaptive_conf = get_adaptive_conf_thres(
                            epoch, opt.epochs, opt.conf_thres,
                            max_conf=getattr(opt, 'conf_thres_max', 0.7),
                            burn_in_epochs=burn_in_epochs,
                        )
                        
                        # NMS with reduced max_det to prevent box explosion
                        pseudo_labels = non_max_suppression(
                            pred_tensor,
                            conf_thres=adaptive_conf,
                            iou_thres=opt.iou_thres,
                            max_det=20,
                        )
                        
                        valid_coco_classes = set(class_mapping.keys())
                        
                        for idx, preds in enumerate(pseudo_labels):
                            if preds is not None and len(preds) > 0:
                                preds[:, 0].clamp_(0, img_w)
                                preds[:, 1].clamp_(0, img_h)
                                preds[:, 2].clamp_(0, img_w)
                                preds[:, 3].clamp_(0, img_h)
                                
                                box_w = preds[:, 2] - preds[:, 0]
                                box_h = preds[:, 3] - preds[:, 1]
                                box_area = box_w * box_h
                                img_area = img_w * img_h
                                
                                # Track exploded boxes for logging
                                large_boxes = (box_area >= 0.8 * img_area).sum().item()
                                if large_boxes > 0 and i == 0:
                                    LOGGER.warning(f"[WARNING] Epoch {epoch}: Discarded {large_boxes} exploded pseudo-labels (>80% image area). Teacher diverging!")
                                
                                # Enforce min dimensions (5x5) and max area (80% of image)
                                valid_box = (box_w > 5) & (box_h > 5) & (box_area < 0.8 * img_area)
                                
                                cls_ids = preds[:, 5].long()
                                valid_cls = torch.zeros(len(preds), dtype=torch.bool, device=device)
                                for coco_cls in valid_coco_classes:
                                    valid_cls |= (cls_ids == coco_cls)
                                
                                valid = valid_box & valid_cls
                                filtered_preds = preds[valid]
                                
                                if len(filtered_preds) > 0:
                                    for coco_cls, dataset_cls in class_mapping.items():
                                        mask = (filtered_preds[:, 5].long() == coco_cls)
                                        filtered_preds[mask, 5] = dataset_cls
                                
                                pseudo_labels[idx] = filtered_preds
                        
                        n_pseudo = sum(len(p) if p is not None else 0 for p in pseudo_labels)
                        
                        # Debug images (skip in tuning mode to save disk)
                        tuning_mode = getattr(opt, 'tuning_mode', False)
                        if not tuning_mode and i % 100 == 0:
                            debug_dir = save_dir / 'debug_images'
                            debug_dir.mkdir(exist_ok=True)
                            n_boxes = save_debug_image(
                                imgs_target_fake[0],
                                [pseudo_labels[0]] if pseudo_labels else None,
                                debug_dir / f'epoch{epoch:03d}_iter{epoch * nb + i:06d}_teacher.jpg',
                                names,
                                conf_thres=adaptive_conf
                            )
                            LOGGER.info(f'[DEBUG] Epoch {epoch}: Teacher detections={n_boxes}, conf={adaptive_conf:.3f}')
                
                # Distillation loss (pseudo-box labels)
                loss_distillation = compute_loss.compute_distillation_loss(
                    pred_target, pseudo_labels, (img_h, img_w)
                )
                
                # ── Feature MSE Distillation ──────────────────────────────
                # Ép Student(source-real features) gần với Teacher(target-fake features)
                # Gradient chỉ chảy vào Student; teacher_feat được detach.
                loss_feature_mse = torch.tensor(0.0, device=device, requires_grad=False)
                if (getattr(opt, 'use_feature_distill', False)
                        and teacher_feat_for_mse is not None
                        and source_features is not None
                        and epoch >= burn_in_epochs):
                    tf = teacher_feat_for_mse.detach()  # không cập nhật Teacher
                    sf = source_features
                    # Align spatial size nếu 2 feature map khác nhau (hiếm, nhưng an toàn)
                    if sf.shape != tf.shape:
                        tf = torch.nn.functional.interpolate(
                            tf, size=sf.shape[2:], mode='bilinear', align_corners=False
                        )
                    loss_feature_mse = torch.nn.functional.mse_loss(sf, tf)
                
                # Total loss
                # L = Ldet(src) + Ldet(src_fake) + λ_box·Ldistill + λ_feat·Lfeat_mse + β·Lcons + Ldomain
                def ensure_scalar(x):
                    if isinstance(x, torch.Tensor) and x.numel() > 1:
                        return x.sum()
                    return x
                
                loss_source       = ensure_scalar(loss_source)
                loss_source_fake  = ensure_scalar(loss_source_fake)
                loss_distillation = ensure_scalar(loss_distillation)
                loss_feature_mse  = ensure_scalar(loss_feature_mse)
                loss_domain       = ensure_scalar(loss_domain)
                
                # Consistency loss: L2(Ldet(source) - Ldet(source_fake))
                loss_diff = torch.clamp(loss_source.detach() - loss_source_fake, -10, 10)
                loss_consistency = loss_diff ** 2
                consistency_weight = 1.0
                
                lambda_feature = getattr(opt, 'lambda_feature', 0.05)
                loss = (loss_source + loss_source_fake
                        + loss_distillation * current_lambda
                        + loss_feature_mse  * lambda_feature
                        + loss_consistency  * consistency_weight
                        + loss_domain)
                loss = torch.clamp(loss, 0, 500)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    LOGGER.warning(f'[NaN] Loss is NaN/Inf at epoch {epoch}, iter {i}. Skipping.')
                    optimizer.zero_grad()
                    if grl_optimizer:
                        grl_optimizer.zero_grad()
                    continue
            
            # Backward
            scaler.scale(loss).backward()
            
            # Optimizer step
            scaler.unscale_(optimizer)
            grl_optimizer_active = opt.use_grl and epoch >= opt.grl_warmup and grl_optimizer is not None
            if grl_optimizer_active:
                scaler.unscale_(grl_optimizer)

            grad_clip = getattr(opt, 'gradient_clip', 2.0)
            torch.nn.utils.clip_grad_norm_(model_student.parameters(), max_norm=grad_clip)
            if opt.use_grl:
                if use_multiscale_grl and ms_discriminator is not None:
                    torch.nn.utils.clip_grad_norm_(ms_discriminator.parameters(), max_norm=grad_clip)
                elif domain_discriminator is not None:
                    torch.nn.utils.clip_grad_norm_(domain_discriminator.parameters(), max_norm=grad_clip)

            # Check for NaN gradients
            grad_nan = False
            for name, p in model_student.named_parameters():
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    grad_nan = True
                    LOGGER.warning(f'[NaN Grad] at epoch {epoch}, iter {i}, layer: {name}')
                    break

            if grad_nan:
                optimizer.zero_grad()
                if grl_optimizer:
                    grl_optimizer.zero_grad()
                scaler.update()
                continue

            scaler.step(optimizer)
            # Always update discriminator — stopping it when it's winning is backwards
            # (the GRL still pushes the backbone but discriminator can't adapt → stays dominant)
            if grl_optimizer_active:
                scaler.step(grl_optimizer)
            scaler.update()

            optimizer.zero_grad()
            if grl_optimizer_active:
                grl_optimizer.zero_grad()
            
            # EMA update (AFTER optimizer.step!)
            teacher_ema.update(model_student)
            prev_domain_acc = domain_acc  # carry forward for next iteration's GRL weight boost
            
            if i % 100 == 0:
                torch.cuda.empty_cache()
            
            # Logging
            loss_items = torch.tensor([
                loss_items_source[0].item() if isinstance(loss_items_source, (list, tuple)) else loss_source.item(),
                loss_items_source[1].item() if isinstance(loss_items_source, (list, tuple)) and len(loss_items_source) > 1 else 0,
                loss_items_source[2].item() if isinstance(loss_items_source, (list, tuple)) and len(loss_items_source) > 2 else 0,
                loss_distillation.item(),
                loss_domain.item(),
            ], device=device)
            
            mloss = (mloss * i + loss_items) / (i + 1)
            
            log_dict = {
                'loss_sr':           loss_source.item(),
                'loss_sf':           loss_source_fake.item(),
                'loss_distill':      loss_distillation.item(),
                'loss_consistency':  loss_consistency.item(),
                'loss_domain':       loss_domain.item(),
                'loss_total':        loss.item(),
            }
            if getattr(opt, 'use_feature_distill', False):
                log_dict['loss_feat_mse'] = loss_feature_mse.item()
            
            logger.log_iteration(epoch, i, log_dict, extra={
                'lr':         optimizer.param_groups[0]['lr'],
                'grl_alpha':  current_grl_alpha if opt.use_grl else 0,
                'conf_thres': adaptive_conf,
                'lambda':     current_lambda,
            }, global_step=global_step)
            global_step += 1
            
            if domain_monitor and i % 50 == 0 and i < 500:
                domain_monitor.collect_features(
                    features_sr=source_features,
                    features_sf=source_fake_features,
                    features_tr=target_features,
                    max_batches=50,  # More samples for meaningful UMAP
                )
            
            mem = f'{torch.cuda.memory_reserved() / 1e9:.1f}G' if torch.cuda.is_available() else 'CPU'
            feat_mse_str = f'{loss_feature_mse.item():.4f}' if getattr(opt, 'use_feature_distill', False) else '-'
            pbar.set_postfix({
                'mem':     mem,
                'box':     f'{mloss[0]:.4f}',
                'distill': f'{mloss[3]:.4f}',
                'feat':    feat_mse_str,
                'domain':  f'{mloss[4]:.4f}',
            })
        
        # End of epoch
        scheduler.step()
        
        if domain_monitor:
            try:
                domain_monitor.end_epoch(epoch)
            except Exception as e:
                LOGGER.warning(f'DomainMonitor failed: {e}')
            domain_monitor._epoch_features.clear()
            domain_monitor._epoch_labels.clear()
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # Validation
        if (epoch + 1) % 10 == 0 or epoch == opt.epochs - 1:
            orig_model = yolo_student.model
            try:
                val_yolo = YOLO(opt.weights)
                val_model = val_yolo.model.to(device)
                val_model.load_state_dict(model_student.state_dict())
                val_model.eval()
                yolo_student.model = val_model
                
                metrics = yolo_student.val(data=opt.data, split='test', verbose=False)
                current_map50 = metrics.box.map50
                current_map = metrics.box.map
                
                LOGGER.info(f'Epoch {epoch}: mAP@50={current_map50:.4f}, mAP@50-95={current_map:.4f}')
                
                logger.log_epoch(epoch, {
                    'mAP50': current_map50,
                    'mAP50-95': current_map,
                    'precision': metrics.box.mp,
                    'recall': metrics.box.mr,
                })
                
                if current_map50 > best_fitness:
                    best_fitness = current_map50
                    torch.save({
                        'epoch': epoch,
                        'model': model_student.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_fitness': best_fitness,
                    }, save_dir / 'weights' / 'best.pt')
            except Exception as e:
                LOGGER.warning(f'Validation failed: {e}')
            finally:
                yolo_student.model = orig_model
                del val_model, val_yolo
                gc.collect()
                torch.cuda.empty_cache()
        
        # Save last (skip in tuning mode to save disk)
        if not getattr(opt, 'tuning_mode', False):
            torch.save({
                'epoch': epoch,
                'model': model_student.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_fitness': best_fitness,
            }, save_dir / 'weights' / 'last.pt')
        
        # Save intermediate checkpoints for pseudo-label quality analysis
        if epoch in checkpoint_epochs and not getattr(opt, 'tuning_mode', False):
            ckpt_data = {
                'epoch': epoch,
                'model': model_student.state_dict(),
                'teacher_state_dict': model_teacher.state_dict(),
                'best_fitness': best_fitness,
            }
            torch.save(ckpt_data, save_dir / 'weights' / f'checkpoint_ep{epoch:03d}.pt')
            LOGGER.info(f'💾 Saved checkpoint: checkpoint_ep{epoch:03d}.pt (student + teacher)')
    
    # Cleanup
    if ms_feature_hook:
        ms_feature_hook.remove()
    if feature_hook and feature_hook is not monitor_hook:
        feature_hook.remove()
    if monitor_hook:
        monitor_hook.remove()
    if teacher_feature_hook:
        teacher_feature_hook.remove()
    logger.finalize()
    if domain_monitor:
        domain_monitor.finalize()
    
    hours = (time.time() - t0) / 3600
    LOGGER.info(f'\n{opt.epochs} epochs completed in {hours:.2f} hours.')
    LOGGER.info(f'Best mAP@50: {best_fitness:.4f}')
    LOGGER.info(f'Results saved to {save_dir}')
    
    return best_fitness


def parse_args():
    parser = argparse.ArgumentParser(description='FDA Training')
    
    parser.add_argument('--config', type=str, default=None, help='YAML config file')
    
    # Model
    parser.add_argument('--cfg', type=str, default=None, help='custom model.yaml file (e.g. yolov8-p2.yaml)')
    parser.add_argument('--weights', type=str, default='yolo26s.pt')
    parser.add_argument('--data', type=str, default='data_v8.yaml')
    parser.add_argument('--imgsz', type=int, default=640)
    
    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--lr0', type=float, default=0.0001)
    parser.add_argument('--lrf', type=float, default=0.01)
    
    # Teacher-Student
    parser.add_argument('--teacher-alpha', type=float, default=0.9999)
    parser.add_argument('--conf-thres', type=float, default=0.1)
    parser.add_argument('--conf-thres-max', type=float, default=0.7)
    parser.add_argument('--iou-thres', type=float, default=0.45)
    parser.add_argument('--lambda-weight', type=float, default=0.1)
    parser.add_argument('--freeze-teacher', action='store_true', default=None)
    
    # Progressive
    parser.add_argument('--use-progressive-lambda', action='store_true')
    parser.add_argument('--warmup-epochs', type=int, default=10)
    parser.add_argument('--burn-in-epochs', type=int, default=50)
    
    # GRL
    parser.add_argument('--use-grl', action='store_true')
    parser.add_argument('--use-multiscale-grl', action='store_true',
                        help='Use multi-scale domain discriminator (P3+P4+P5). '
                             'More SOTA but ~15%% more compute. Requires --use-grl.')
    parser.add_argument('--grl-warmup', type=int, default=20,
                        help='Epochs before GRL activates (increased from 10 for stability)')
    parser.add_argument('--grl-max-alpha', type=float, default=1.0)
    parser.add_argument('--grl-weight', type=float, default=0.05,
                        help='GRL loss weight (increased to 0.05 to match detection loss)')
    parser.add_argument('--grl-hidden-dim', type=int, default=512,
                        help='Hidden dim for discriminator MLP (single-scale=512, multi-scale=256)')
    parser.add_argument('--grl-dropout', type=float, default=0.1,
                        help='Dropout for discriminator MLP')
    parser.add_argument('--grl-lr', type=float, default=0.00005,
                        help='Learning rate for discriminator')
    
    # Output
    parser.add_argument('--project', type=str, default='runs/fda')
    parser.add_argument('--name', type=str, default='exp')
    
    # Misc
    parser.add_argument('--enable-monitoring', action='store_true')
    parser.add_argument('--amp', action='store_true')
    
    # ── Feature MSE Distillation ─────────────────────────────────────────
    parser.add_argument(
        '--use-feature-distill', action='store_true',
        help='Bật Feature MSE Distillation: ép Student(source-real) features '
             'gần với Teacher(target-fake) features ở layer cuối backbone.'
    )
    parser.add_argument(
        '--lambda-feature', type=float, default=0.05,
        help='Hệ số nhân cho Feature MSE Loss (default 0.05). '
             'Chỉ có tác dụng khi --use-feature-distill được bật.'
    )

    # ── Small-object loss improvements ──────────────────────────────────
    parser.add_argument(
        '--use-small-object-loss', action='store_true',
        help='Replace CIoU with Inner-CIoU + ScaleAware TAL for better small-object mAP'
    )
    parser.add_argument(
        '--inner-ratio', type=float, default=0.7,
        help='Auxiliary box scale for Inner-CIoU (0 < r <= 1, default 0.7)'
    )
    parser.add_argument(
        '--use-wise-iou', action='store_true',
        help='Enable WiseIoU v3 reweighting on top of Inner-CIoU'
    )
    parser.add_argument(
        '--box-gain', type=float, default=7.5,
        help='Box/IoU loss weight (default 7.5)'
    )
    parser.add_argument(
        '--cls-gain', type=float, default=0.5,
        help='Classification loss weight (default 0.5)'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    if args.config:
        from utils.config_loader import load_config, config_to_namespace, merge_cli_args
        config = load_config(args.config)
        config = merge_cli_args(config, args)
        args = config_to_namespace(config)
    
    print("=" * 70)
    print("FDA: Semi-Supervised Domain Adaptive Training")
    print("=" * 70)
    print(f"Model:   {args.weights}")
    print(f"Data:    {args.data}")
    print(f"Epochs:  {args.epochs}")
    print(f"Batch:   {args.batch}")
    print(f"Device:  cuda:{args.device}")
    grl_mode = 'Disabled'
    if args.use_grl:
        grl_mode = 'Multi-Scale (P3+P4+P5)' if getattr(args, 'use_multiscale_grl', False) else 'Single-Scale'
    print(f"GRL:     {grl_mode}")
    freeze_teacher = getattr(args, 'freeze_teacher', False)
    print(f"Teacher: {'FROZEN' if freeze_teacher else 'EMA'}")
    print("=" * 70)
    
    train(args)