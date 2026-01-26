import torch
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.ops import xyxy2xywhn
from ultralytics.cfg import get_cfg

class WeightEMA:
    """
    Exponential Moving Average for teacher model.
    """
    def __init__(self, teacher_params, student_params, alpha=0.999):
        self.teacher_params = list(teacher_params)
        self.student_params = list(student_params)
        self.alpha = alpha
        
        # Initialize teacher = student
        for t_param, s_param in zip(self.teacher_params, self.student_params):
            t_param.data.copy_(s_param.data)
    
    def step(self):
        """Update teacher params with EMA of student params"""
        one_minus_alpha = 1.0 - self.alpha
        for t_param, s_param in zip(self.teacher_params, self.student_params):
            t_param.data.mul_(self.alpha)
            t_param.data.add_(s_param.data * one_minus_alpha)
            
            
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
    def __init__(self, model):
        from ultralytics.utils import IterableSimpleNamespace
        
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
            return torch.tensor(0.0, device=self.device)
        
        # Convert pseudo-labels to batch format for Ultralytics loss
        batch = self._format_pseudo_targets(pseudo_targets, img_shape)
        
        if batch['cls'].numel() == 0:
            return torch.tensor(0.0, device=self.device)
        
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
            classes = preds[:, 5]
            
            # Convert to normalized xywh
            boxes_xywhn = xyxy2xywhn(boxes_xyxy, w=img_w, h=img_h)
            
            all_cls.append(classes)
            all_bboxes.append(boxes_xywhn)
            all_batch_idx.append(torch.full((len(preds),), batch_idx, device=preds.device))
        
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


