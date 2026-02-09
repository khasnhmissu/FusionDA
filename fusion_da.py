import torch
import torch.nn as nn
from copy import deepcopy
from typing import Optional
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.ops import xyxy2xywhn
from ultralytics.cfg import get_cfg


class WeightEMA(nn.Module):
    """
    Exponential Moving Average for Mean Teacher.
    
    θ_teacher = α * θ_teacher + (1-α) * θ_student
    
    Key design:
    - Only parameters updated, buffers (BatchNorm) stay frozen
    - Teacher always in eval mode
    - Call update() AFTER optimizer.step()
    - update_after_step: Delay EMA updates to let student stabilize
    """
    
    def __init__(
            self,
            model,
            alpha: float = 0.999,
            freeze_teacher: bool = False,
            device: Optional[torch.device] = None,
            update_after_step: int = 500,  # Delay EMA updates
    ):
        super().__init__()
        
        # Create independent copy
        self.module = deepcopy(model)
        self.module.eval()
        
        self.alpha = alpha
        self.freeze_teacher = freeze_teacher
        self.device = device
        self.update_after_step = update_after_step
        self.updates = 0
        self.step_count = 0
        self.nan_count = 0
        
        if device is not None:
            self.module.to(device=device)
        
        # Disable gradients - teacher is never trained directly
        for p in self.module.parameters():
            p.requires_grad_(False)
        
        n_params = sum(1 for _ in self.module.parameters())
        n_buffers = sum(1 for _ in self.module.buffers())
        
        if freeze_teacher:
            print(f"[WeightEMA] FREEZE: Teacher stays pretrained")
        else:
            print(f"[WeightEMA] alpha={alpha}, {n_params} params (EMA), {n_buffers} buffers (frozen)")
            print(f"[WeightEMA] EMA updates start after step {update_after_step}")
    
    @torch.no_grad()
    def update(self, model):
        """Update teacher with EMA. Call AFTER optimizer.step()."""
        self.step_count += 1
        
        if self.freeze_teacher:
            self.updates += 1
            return
        
        # Delay EMA updates to let student learn from GT first
        if self.step_count < self.update_after_step:
            if self.step_count % 100 == 0:
                print(f"[EMA] Step {self.step_count}: waiting until step {self.update_after_step}")
            return
        
        teacher_params = list(self.module.parameters())
        student_params = list(model.parameters())
        
        # Check for NaN in student
        for sp in student_params:
            if torch.isnan(sp.data).any() or torch.isinf(sp.data).any():
                self.nan_count += 1
                if self.nan_count <= 5:
                    print(f"[WeightEMA] Student has NaN/Inf, skipping")
                return
        
        # EMA: θ = α*θ + (1-α)*θ_student
        one_minus_alpha = 1.0 - self.alpha
        for tp, sp in zip(teacher_params, student_params):
            tp.data.mul_(self.alpha)
            tp.data.add_(sp.data * one_minus_alpha)
        
        self.updates += 1
        
        if self.updates % 500 == 0:
            print(f"[EMA] Update {self.updates}: α={self.alpha:.6f}")
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class PairedMultiDomainDataset(torch.utils.data.Dataset):
    """
    Dataset loading 4 matched images: source_real, source_fake, target_real, target_fake.
    
    Matching by filename ensures:
    - Consistency Loss: source_real ↔ source_fake (same scene)
    - Distillation: target_real ↔ target_fake (same scene)
    """
    
    def __init__(self, source_real_path, source_fake_path, target_real_path, target_fake_path,
                 imgsz=640, augment=False, hyp=None, data=None, stride=32):
        from ultralytics.data.dataset import YOLODataset
        from pathlib import Path
        
        self.imgsz = imgsz
        self.augment = augment
        
        dataset_args = {
            'imgsz': imgsz, 'augment': augment, 'cache': False,
            'hyp': hyp, 'data': data, 'task': 'detect', 'stride': stride,
            'rect': True, 'single_cls': False, 'pad': 0.5, 'classes': None,
        }
        
        self.source_real_ds = YOLODataset(img_path=source_real_path, **dataset_args)
        self.source_fake_ds = YOLODataset(img_path=source_fake_path, **dataset_args) if source_fake_path else None
        self.target_real_ds = YOLODataset(img_path=target_real_path, **dataset_args) if target_real_path else None
        self.target_fake_ds = YOLODataset(img_path=target_fake_path, **dataset_args) if target_fake_path else None
        
        # Build filename → index mappings
        self.source_real_file_to_idx = self._build_file_index(self.source_real_ds)
        self.source_fake_file_to_idx = self._build_file_index(self.source_fake_ds) if self.source_fake_ds else {}
        self.target_real_file_to_idx = self._build_file_index(self.target_real_ds) if self.target_real_ds else {}
        self.target_fake_file_to_idx = self._build_file_index(self.target_fake_ds) if self.target_fake_ds else {}
        
        self.n = len(self.source_real_ds)
        self._log_matching_stats()
    
    def _build_file_index(self, dataset):
        from pathlib import Path
        if dataset is None:
            return {}
        return {Path(f).stem: i for i, f in enumerate(dataset.im_files)}
    
    def _log_matching_stats(self):
        sr = set(self.source_real_file_to_idx.keys())
        sf = set(self.source_fake_file_to_idx.keys())
        tr = set(self.target_real_file_to_idx.keys())
        tf = set(self.target_fake_file_to_idx.keys())
        print(f"[PairedDataset] Source: {len(sr)} real, {len(sf)} fake, {len(sr & sf)} matched")
        print(f"[PairedDataset] Target: {len(tr)} real, {len(tf)} fake, {len(tr & tf)} matched")
    
    def _get_matched_item(self, primary_ds, primary_idx, other_ds, other_idx_map):
        if other_ds is None or not other_idx_map:
            return primary_ds[primary_idx]
        
        from pathlib import Path
        filename = Path(primary_ds.im_files[primary_idx]).stem
        
        if filename in other_idx_map:
            return other_ds[other_idx_map[filename]]
        return other_ds[primary_idx % len(other_ds)]
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        source_real = self.source_real_ds[idx]
        source_fake = self._get_matched_item(self.source_real_ds, idx, self.source_fake_ds, self.source_fake_file_to_idx)
        
        target_idx = idx % len(self.target_real_ds) if self.target_real_ds else idx
        target_real = self.target_real_ds[target_idx] if self.target_real_ds else source_real
        target_fake = self._get_matched_item(
            self.target_real_ds, target_idx, self.target_fake_ds, self.target_fake_file_to_idx
        ) if self.target_real_ds else source_fake
        
        return {
            'source_real': source_real, 'source_fake': source_fake,
            'target_real': target_real, 'target_fake': target_fake,
        }
    
    @staticmethod
    def collate_fn(batch):
        from ultralytics.data.dataset import YOLODataset
        return {
            'source_real': YOLODataset.collate_fn([b['source_real'] for b in batch]),
            'source_fake': YOLODataset.collate_fn([b['source_fake'] for b in batch]),
            'target_real': YOLODataset.collate_fn([b['target_real'] for b in batch]),
            'target_fake': YOLODataset.collate_fn([b['target_fake'] for b in batch]),
        }


class DualDomainDataset(torch.utils.data.Dataset):
    """Dataset loading paired real and fake (style-transferred) images."""
    
    def __init__(self, real_paths, fake_paths, imgsz=640, augment=True):
        from ultralytics.data.dataset import YOLODataset
        
        self.real_dataset = YOLODataset(img_path=real_paths, imgsz=imgsz, augment=augment, cache=False)
        self.fake_dataset = YOLODataset(img_path=fake_paths, imgsz=imgsz, augment=augment, cache=False) if fake_paths else None
        self.n = len(self.real_dataset)
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        real_data = self.real_dataset[idx]
        
        if self.fake_dataset and len(self.fake_dataset) > 0:
            fake_data = self.fake_dataset[idx % len(self.fake_dataset)]
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
    """Create dataloader for dual-domain images."""
    from ultralytics.data import YOLODataset
    from ultralytics.data.build import InfiniteDataLoader
    
    dataset = YOLODataset(img_path=real_paths, imgsz=imgsz, augment=augment, cache=False, batch_size=batch_size)
    
    return InfiniteDataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True, collate_fn=YOLODataset.collate_fn,
    ), dataset


class FDALoss:
    """Combined loss for FDA training using Ultralytics v8DetectionLoss."""
    
    def __init__(self, model, class_mapping=None):
        from ultralytics.utils import IterableSimpleNamespace
        
        # Class mapping: COCO ID → Dataset ID
        self.class_mapping = class_mapping or {0: 0, 1: 1, 2: 1}
        
        hyp = IterableSimpleNamespace(
            box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0,
            lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
            warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
            degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
            flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0,
            label_smoothing=0.0, nbs=64, overlap_mask=True, mask_ratio=4, dropout=0.0,
        )
        
        model.args = hyp
        self.detection_loss = v8DetectionLoss(model)
        self.detection_loss.hyp = hyp
        self.device = next(model.parameters()).device
    
    def __call__(self, preds, batch):
        result = self.detection_loss(preds, batch)
        
        if isinstance(result, tuple):
            loss = result[0]
            loss_items = result[1] if len(result) > 1 else None
        else:
            loss = result
            loss_items = None
        
        if loss.numel() > 1:
            loss = loss.sum()
            
        return loss, loss_items
    
    def compute_distillation_loss(self, student_preds, pseudo_targets, img_shape):
        """Compute loss using pseudo-labels from teacher."""
        if pseudo_targets is None or len(pseudo_targets) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        batch = self._format_pseudo_targets(pseudo_targets, img_shape)
        
        if batch['cls'].numel() == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        loss, _ = self.detection_loss(student_preds, batch)
        return loss
    
    def _format_pseudo_targets(self, pseudo_labels_list, img_shape):
        """Convert NMS output to Ultralytics batch format."""
        img_h, img_w = img_shape
        
        all_cls, all_bboxes, all_batch_idx = [], [], []
        
        for batch_idx, preds in enumerate(pseudo_labels_list):
            if preds is None or len(preds) == 0:
                continue
            
            boxes_xyxy = preds[:, :4]
            coco_classes = preds[:, 5]
            
            # Filter and remap classes
            valid_mask = torch.zeros(len(preds), dtype=torch.bool, device=preds.device)
            remapped = torch.zeros_like(coco_classes)
            
            for coco_cls, dataset_cls in self.class_mapping.items():
                mask = coco_classes == coco_cls
                valid_mask |= mask
                remapped[mask] = dataset_cls
            
            if valid_mask.sum() == 0:
                continue
            
            boxes_xyxy = boxes_xyxy[valid_mask]
            classes = remapped[valid_mask]
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
