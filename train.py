import argparse
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
from ultralytics.utils.ops import non_max_suppression
from ultralytics.utils import LOGGER, colorstr

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
    
    # Source domain paths (real + fake)
    source_real = data_dict.get('train_source_real', data_dict.get('train', []))
    if isinstance(source_real, str):
        source_real = [source_real]
    source_real = [str(root / p) for p in source_real]
    
    source_fake = data_dict.get('train_source_fake', [])
    if isinstance(source_fake, str):
        source_fake = [source_fake]
    source_fake = [str(root / p) for p in source_fake] if source_fake else []
    
    # Target domain paths (real + fake)
    target_real = data_dict.get('train_target_real', [])
    if isinstance(target_real, str):
        target_real = [target_real]
    target_real = [str(root / p) for p in target_real]
    
    target_fake = data_dict.get('train_target_fake', [])
    if isinstance(target_fake, str):
        target_fake = [target_fake]
    target_fake = [str(root / p) for p in target_fake] if target_fake else []
    
    # Test path
    test_real = data_dict.get('test_target_real', data_dict.get('test', []))
    if isinstance(test_real, str):
        test_real = [test_real]
    test_real = [str(root / p) for p in test_real]
    
    LOGGER.info(f'Source Real: {source_real}')
    LOGGER.info(f'Source Fake: {source_fake}')
    LOGGER.info(f'Target Real: {target_real}')
    LOGGER.info(f'Target Fake: {target_fake}')
    
    # Build dataloaders using Ultralytics
    from ultralytics.data import build_dataloader, build_yolo_dataset
    
    gs = max(int(model_student.stride.max()), 32)
    
    # Source dataloader
    source_dataset = build_yolo_dataset(
        yolo_student.overrides,
        img_path=source_real[0] if source_real else '',
        batch=opt.batch,
        data=data_dict,
        mode='train',
        stride=gs,
    )
    source_loader = build_dataloader(source_dataset, opt.batch, opt.workers, shuffle=True)
    
    # Target Real dataloader
    if target_real:
        target_dataset = build_yolo_dataset(
            yolo_student.overrides,
            img_path=target_real[0],
            batch=opt.batch,
            data=data_dict,
            mode='train',
            stride=gs,
        )
        target_loader = build_dataloader(target_dataset, opt.batch, opt.workers, shuffle=True)
    else:
        target_loader = source_loader  # Fallback
    
    # Source Fake dataloader
    if source_fake:
        source_fake_dataset = build_yolo_dataset(
            yolo_student.overrides,
            img_path=source_fake[0],
            batch=opt.batch,
            data=data_dict,
            mode='train',
            stride=gs,
        )
        source_fake_loader = build_dataloader(source_fake_dataset, opt.batch, opt.workers, shuffle=True)
    else:
        source_fake_loader = source_loader  # Fallback to source real
    
    # Target Fake dataloader
    if target_fake:
        target_fake_dataset = build_yolo_dataset(
            yolo_student.overrides,
            img_path=target_fake[0],
            batch=opt.batch,
            data=data_dict,
            mode='train',
            stride=gs,
        )
        target_fake_loader = build_dataloader(target_fake_dataset, opt.batch, opt.workers, shuffle=True)
    else:
        target_fake_loader = target_loader  # Fallback to target real
    
    nb = max(len(source_loader), len(target_loader), len(source_fake_loader) if source_fake else 0, len(target_fake_loader) if target_fake else 0)
    
    # ========================================================================
    # LOSS FUNCTION
    # ========================================================================
    compute_loss = FDALoss(model_student)
    scaler = amp.GradScaler(enabled=device.type != 'cpu')
    
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
            tsne_epochs = [0, opt.epochs - 1]
        else:
            step = opt.epochs // 4
            umap_epochs = [0, step, step * 2, step * 3, opt.epochs - 1]
            tsne_epochs = [0, opt.epochs // 2, opt.epochs - 1]
        
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
            with amp.autocast(enabled=device.type != 'cpu'):
                
                # ============================================================
                # PHASE 1: SOURCE REAL FORWARD
                # ============================================================
                pred_source = model_student(imgs_source)
                loss_source, loss_items_source = compute_loss(pred_source, batch_source)
                
                # Extract source features for GRL
                source_features = None
                if opt.use_grl and epoch >= opt.grl_warmup and feature_hook:
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
                
                # Extract target features for GRL
                target_features = None
                if opt.use_grl and epoch >= opt.grl_warmup and feature_hook:
                    target_features = feature_hook.get_features()
                
                # ============================================================
                # PHASE 3: DOMAIN LOSS (GRL)
                # ============================================================
                loss_domain = torch.tensor(0.0, device=device)
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
                with torch.no_grad():
                    # Get teacher predictions on target FAKE domain
                    pred_teacher = model_teacher(imgs_target_fake)
                    
                    # Extract predictions tensor
                    pred_tensor = pred_teacher[0] if isinstance(pred_teacher, (list, tuple)) else pred_teacher
                    
                    # Adaptive confidence threshold
                    adaptive_conf = get_adaptive_conf_thres(epoch, opt.epochs, opt.conf_thres)
                    
                    # NMS to get pseudo-labels
                    pseudo_labels = non_max_suppression(
                        pred_tensor,
                        conf_thres=adaptive_conf,
                        iou_thres=opt.iou_thres,
                        max_det=300,
                        multi_label=True,
                    )
                    
                    # Filter by uncertainty
                    pseudo_labels = filter_pseudo_labels_by_uncertainty(pseudo_labels, 0.25)
                
                # Compute distillation loss
                img_h, img_w = imgs_target.shape[2:]
                loss_distillation = compute_loss.compute_distillation_loss(
                    pred_target, pseudo_labels, (img_h, img_w)
                )
                
                # ============================================================
                # TOTAL LOSS (SR + SF + Distill + Domain)
                # ============================================================
                loss = loss_source + loss_source_fake + loss_distillation * current_lambda + loss_domain
            
            # ================================================================
            # BACKWARD
            # ================================================================
            scaler.scale(loss).backward()
            
            # Gradient clipping for stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model_student.parameters(), max_norm=10.0)
            
            # ================================================================
            # OPTIMIZER STEP (correct order: all step() before update())
            # ================================================================
            scaler.step(optimizer)
            
            # Update GRL optimizer (with scaler for mixed precision)
            if opt.use_grl and epoch >= opt.grl_warmup and grl_optimizer:
                scaler.unscale_(grl_optimizer)
                torch.nn.utils.clip_grad_norm_(domain_discriminator.parameters(), max_norm=10.0)
                scaler.step(grl_optimizer)
            
            # Update scaler AFTER all step() calls
            scaler.update()
            
            # Zero gradients
            optimizer.zero_grad()
            if opt.use_grl and grl_optimizer:
                grl_optimizer.zero_grad()
            
            # Update teacher with EMA
            teacher_ema.step()
            
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
                'loss_domain': loss_domain.item(),
                'loss_total': loss.item(),
            }, extra={
                'lr': optimizer.param_groups[0]['lr'],
                'grl_alpha': current_grl_alpha if opt.use_grl else 0,
                'conf_thres': adaptive_conf,
                'lambda': current_lambda,
            }, global_step=global_step)
            global_step += 1
            
            # Collect features for explainability (every 10 iterations)
            if domain_monitor and i % 10 == 0:
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
        
        # Domain monitoring end-of-epoch analysis
        if domain_monitor:
            monitor_result = domain_monitor.end_epoch(epoch)
            LOGGER.info(f'[Monitor] {domain_monitor.mmd_tracker.get_interpretation()}')
        
        # Validation every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == opt.epochs - 1:
            # Use Ultralytics validation
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
    parser.add_argument('--lr0', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01, help='Final learning rate factor')
    
    # Teacher-Student
    parser.add_argument('--teacher-alpha', type=float, default=0.999, help='Teacher EMA decay')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='Pseudo-label confidence')
    parser.add_argument('--iou-thres', type=float, default=0.3, help='NMS IoU threshold')
    parser.add_argument('--lambda-weight', type=float, default=0.005, help='Distillation weight')
    
    # Progressive training
    parser.add_argument('--use-progressive-lambda', action='store_true', help='Progressive lambda')
    parser.add_argument('--warmup-epochs', type=int, default=20, help='Warmup epochs')
    
    # GRL
    parser.add_argument('--use-grl', action='store_true', help='Enable GRL')
    parser.add_argument('--grl-warmup', type=int, default=20, help='GRL warmup epochs')
    parser.add_argument('--grl-max-alpha', type=float, default=1.0, help='Max GRL alpha')
    parser.add_argument('--grl-weight', type=float, default=0.1, help='GRL loss weight')
    parser.add_argument('--grl-hidden-dim', type=int, default=256, help='Discriminator hidden dim')
    
    # Output
    parser.add_argument('--project', type=str, default='runs/fda', help='Project dir')
    parser.add_argument('--name', type=str, default='exp', help='Experiment name')
    
    # Monitoring & Explainability
    parser.add_argument('--enable-monitoring', action='store_true', help='Enable domain monitoring (UMAP, MMD, etc.)')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    print("=" * 70)
    print("FDA: Semi-Supervised Domain Adaptive Training")
    print("=" * 70)
    print(f"Model:   {args.weights}")
    print(f"Data:    {args.data}")
    print(f"Epochs:  {args.epochs}")
    print(f"Device:  cuda:{args.device}")
    print(f"GRL:     {'Enabled' if args.use_grl else 'Disabled'}")
    print("=" * 70)
    
    train(args)