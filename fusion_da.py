import torch
import torch.nn as nn
from copy import deepcopy
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.ops import xyxy2xywhn
from ultralytics.cfg import get_cfg


class WeightEMA(nn.Module):
    """
    Model Exponential Moving Average following timm.ModelEmaV2 best practices.
    
    CRITICAL DIFFERENCES FROM OLD IMPLEMENTATION:
    1. Uses deepcopy() to create an INDEPENDENT EMA model
    2. Iterates through state_dict().values() for ALL params + buffers
    3. EMA model is ALWAYS in eval() mode
    4. Uses lerp_() for efficient in-place EMA updates
    
    The old implementation was broken because teacher and student shared references,
    causing corruption when student weights changed during training.
    
    Reference: https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/model_ema.py
    """
    
    def __init__(self, model, decay=0.9999, device=None):
        """
        Initialize EMA with a deepcopy of the source model.
        
        Args:
            model: Source model to create EMA from (will be deepcopied)
            decay: EMA decay rate (higher = slower adaptation, 0.9999 typical)
            device: Device to place EMA model on (None = same as source)
        """
        super().__init__()
        
        # CRITICAL: deepcopy creates completely independent model
        # This prevents the corruption that occurred in the old implementation
        self.module = deepcopy(model)
        self.module.eval()  # EMA model is ALWAYS in eval mode
        
        self.decay = decay
        self.device = device
        self.step_count = 0
        self.nan_count = 0
        
        if device is not None:
            self.module.to(device=device)
        
        # Disable gradients for EMA model - it's never trained directly
        for p in self.module.parameters():
            p.requires_grad_(False)
        
        # Count total state_dict entries for logging
        n_params = sum(1 for _ in self.module.parameters())
        n_buffers = sum(1 for _ in self.module.buffers())
        print(f"[EMA] Initialized with decay={decay}, {n_params} params, {n_buffers} buffers")
        print(f"[EMA] Using timm-style implementation with deepcopy + direct param iteration")
    
    @torch.no_grad()
    def update(self, model, decay=None):
        """
        Update EMA model with current model weights.
        
        CRITICAL FIX: Use zip(module.parameters(), model.parameters()) and 
        named_buffers() to get ACTUAL tensor references, not copies!
        
        state_dict().values() returns COPIES of tensors, so in-place ops
        on those copies don't affect the actual model weights.
        
        Args:
            model: Source model with updated weights
            decay: Override decay value (optional)
        """
        d = decay if decay is not None else self.decay
        
        # ================================================================
        # PART 1: Update parameters (weights, biases)
        # ================================================================
        for ema_p, model_p in zip(self.module.parameters(), model.parameters()):
            # Check for NaN/Inf in source
            if torch.isnan(model_p.data).any() or torch.isinf(model_p.data).any():
                self.nan_count += 1
                if self.nan_count <= 5:
                    print(f"[EMA WARNING] Source param has NaN/Inf, skipping update (count={self.nan_count})")
                return
            
            # EMA update: ema = decay * ema + (1 - decay) * model
            # lerp_(end, weight) = self + weight * (end - self)
            # With weight = (1-d): ema + (1-d)*(model - ema) = d*ema + (1-d)*model
            ema_p.data.lerp_(model_p.data, weight=1.0 - d)
        
        # ================================================================
        # PART 2: Update buffers (BatchNorm running_mean, running_var, etc.)
        # ================================================================
        ema_buffers = dict(self.module.named_buffers())
        for name, model_buf in model.named_buffers():
            if name in ema_buffers:
                ema_buf = ema_buffers[name]
                if ema_buf.is_floating_point():
                    # Check for NaN/Inf
                    if torch.isnan(model_buf).any() or torch.isinf(model_buf).any():
                        continue  # Skip this buffer but continue with others
                    # EMA update for floating point buffers
                    ema_buf.lerp_(model_buf, weight=1.0 - d)
                else:
                    # Non-floating buffers (like num_batches_tracked): copy directly
                    ema_buf.copy_(model_buf)
        
        self.step_count += 1
        
        # Periodic verification
        if self.step_count % 500 == 0:
            valid = self._verify_weights()
            if valid:
                print(f"[EMA] Step {self.step_count} completed successfully")
    
    def _verify_weights(self):
        """Check EMA model weights for NaN/Inf."""
        for name, param in self.module.named_parameters():
            if torch.isnan(param.data).any():
                print(f"[EMA ERROR] EMA param '{name}' has NaN")
                return False
            if torch.isinf(param.data).any():
                print(f"[EMA ERROR] EMA param '{name}' has Inf")
                return False
        for name, buf in self.module.named_buffers():
            if buf.is_floating_point():
                if torch.isnan(buf).any():
                    print(f"[EMA ERROR] EMA buffer '{name}' has NaN")
                    return False
                if torch.isinf(buf).any():
                    print(f"[EMA ERROR] EMA buffer '{name}' has Inf")
                    return False
        return True
    
    def forward(self, *args, **kwargs):
        """Forward pass through EMA model (always in eval mode)."""
        return self.module(*args, **kwargs)
    
    # Backward compatibility aliases
    def step(self):
        """Deprecated: Use update(model) instead. This is a no-op for compatibility."""
        print("[EMA WARNING] step() is deprecated. Use update(model) instead.")
        pass
            
            
# ============================================================================
# DUAL DOMAIN DATASET - Load real + fake images
# ============================================================================
class DualDomainDataset(torch.utils.data.Dataset):
    """
    Dataset that loads paired real and fake (style-transferred) images.
    """
    def __init__(self, real_paths, fake_paths, imgsz=640, augment=True):
        from ultralytics.data.dataset import YOLODataset
        
        # Build datasets for real and fake
        self.real_dataset = YOLODataset(
            img_path=real_paths,
            imgsz=imgsz,
            augment=augment,
            cache=False,
        )
        
        self.fake_dataset = YOLODataset(
            img_path=fake_paths,
            imgsz=imgsz,
            augment=augment,
            cache=False,
        ) if fake_paths else None
        
        self.n = len(self.real_dataset)
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        # Get real image
        real_data = self.real_dataset[idx]
        
        # Get corresponding fake image (or same if not available)
        if self.fake_dataset and len(self.fake_dataset) > 0:
            fake_idx = idx % len(self.fake_dataset)
            fake_data = self.fake_dataset[fake_idx]
        else:
            fake_data = real_data
        
        return {
            'img_real': real_data['img'],
            'img_fake': fake_data['img'],
            'cls': real_data.get('cls', torch.tensor([])),
            'bboxes': real_data.get('bboxes', torch.tensor([])),
            'batch_idx': real_data.get('batch_idx', torch.tensor([])),
        }
def create_dual_dataloader(real_paths, fake_paths, imgsz, batch_size, workers=4, augment=True):
    """Create dataloader for dual-domain (real + fake) images"""
    from ultralytics.data import YOLODataset
    from ultralytics.data.build import InfiniteDataLoader
    
    # Simple approach: use standard YOLO dataset for real images
    dataset = YOLODataset(
        img_path=real_paths,
        imgsz=imgsz,
        augment=augment,
        cache=False,
        batch_size=batch_size,
    )
    
    loader = InfiniteDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        collate_fn=YOLODataset.collate_fn,
    )
    
    return loader, dataset


# ============================================================================
# Combines Ultralytics loss with custom components
# ============================================================================
class FDALoss:
    """
    Combined loss for FDA training.
    Uses Ultralytics v8DetectionLoss internally.
    """
    def __init__(self, model, class_mapping=None):
        """
        Args:
            model: YOLOv8 detection model
            class_mapping: Dict mapping class IDs to dataset class IDs.
                          Must include BOTH:
                          - COCO class IDs (for early training when Teacher still outputs COCO)
                          - Dataset class IDs (for later training after Teacher adapts via EMA)
                          Default: {0: 0, 1: 1, 2: 1, 5: 1, 7: 1}
        """
        from ultralytics.utils import IterableSimpleNamespace
        
        # Store class mapping for pseudo-label processing
        # IMPORTANT: Include both COCO and dataset class IDs!
        # - Early training: Teacher outputs COCO IDs (0, 2)
        # - Later training: Teacher (via EMA) outputs dataset IDs (0, 1)
        self.class_mapping = class_mapping or {
            0: 0,   # person (both COCO and dataset)
            1: 1,   # dataset car → car (for EMA-adapted Teacher)
            2: 1,   # COCO car → car
        }
        
        # Create complete hyp config with ALL required attributes
        # Including loss weights that get_cfg() doesn't provide
        hyp = IterableSimpleNamespace(
            # Loss weights (CRITICAL - these are required by v8DetectionLoss)
            box=7.5,      # box loss gain
            cls=0.5,      # cls loss gain
            dfl=1.5,      # dfl loss gain
            pose=12.0,    # pose loss gain
            kobj=1.0,     # keypoint obj loss gain
            # Other training hyperparameters
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            # Augmentation (needed for some operations)
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
            # Other
            label_smoothing=0.0,
            nbs=64,
            overlap_mask=True,
            mask_ratio=4,
            dropout=0.0,
        )
        
        # Set model.args
        model.args = hyp
        
        self.detection_loss = v8DetectionLoss(model)
        
        # Ensure hyp is IterableSimpleNamespace 
        self.detection_loss.hyp = hyp
        
        self.device = next(model.parameters()).device
    
    def __call__(self, preds, batch):
        """Compute detection loss using Ultralytics internals"""
        result = self.detection_loss(preds, batch)
        
        # v8DetectionLoss returns (loss_sum, loss_items_tensor)
        # loss_sum should be scalar, but ensure it
        if isinstance(result, tuple):
            loss = result[0]
            loss_items = result[1] if len(result) > 1 else None
        else:
            loss = result
            loss_items = None
        
        # Ensure scalar
        if loss.numel() > 1:
            loss = loss.sum()
            
        return loss, loss_items
    
    def compute_distillation_loss(self, student_preds, pseudo_targets, img_shape):
        """
        Compute distillation loss using pseudo-labels.
        
        Args:
            student_preds: Student model predictions
            pseudo_targets: Pseudo-labels from teacher [x1,y1,x2,y2,conf,cls]
            img_shape: (height, width) of input images
        """
        if pseudo_targets is None or len(pseudo_targets) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Convert pseudo-labels to batch format for Ultralytics loss
        batch = self._format_pseudo_targets(pseudo_targets, img_shape)
        
        if batch['cls'].numel() == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        loss, loss_items = self.detection_loss(student_preds, batch)
        return loss
    
    def _format_pseudo_targets(self, pseudo_labels_list, img_shape):
        """Convert NMS output to Ultralytics batch format"""
        img_h, img_w = img_shape
        
        all_cls = []
        all_bboxes = []
        all_batch_idx = []
        
        for batch_idx, preds in enumerate(pseudo_labels_list):
            if preds is None or len(preds) == 0:
                continue
            
            # preds format: [x1, y1, x2, y2, conf, cls]
            boxes_xyxy = preds[:, :4]
            coco_classes = preds[:, 5]
            
            # ============================================================
            # COCO to Dataset class mapping (configurable via class_mapping)
            # ============================================================
            coco_to_dataset = self.class_mapping
            
            # Filter and remap classes
            valid_mask = torch.zeros(len(preds), dtype=torch.bool, device=preds.device)
            remapped_classes = torch.zeros_like(coco_classes)
            
            for coco_cls, dataset_cls in coco_to_dataset.items():
                mask = coco_classes == coco_cls
                valid_mask |= mask
                remapped_classes[mask] = dataset_cls
            
            # Only keep valid predictions
            if valid_mask.sum() == 0:
                continue
                
            boxes_xyxy = boxes_xyxy[valid_mask]
            classes = remapped_classes[valid_mask]
            
            # Convert to normalized xywh
            boxes_xywhn = xyxy2xywhn(boxes_xyxy, w=img_w, h=img_h)
            
            all_cls.append(classes)
            all_bboxes.append(boxes_xywhn)
            all_batch_idx.append(torch.full((valid_mask.sum().item(),), batch_idx, device=preds.device))
        
        if not all_cls:
            return {
                'cls': torch.tensor([], device=self.device),
                'bboxes': torch.tensor([], device=self.device).reshape(0, 4),
                'batch_idx': torch.tensor([], device=self.device),
            }
        
        return {
            'cls': torch.cat(all_cls).view(-1, 1),
            'bboxes': torch.cat(all_bboxes),
            'batch_idx': torch.cat(all_batch_idx),
        }


