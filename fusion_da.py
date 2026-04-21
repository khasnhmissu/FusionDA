import math
import torch
import torch.nn as nn
from copy import deepcopy
from typing import Optional
# YOLO26 là NMS-free end-to-end model → dùng E2ELoss
# YOLOv8/11 (có NMS) → dùng v8DetectionLoss
try:
    from ultralytics.utils.loss import E2ELoss as _DetLoss
    _IS_E2E = True
except ImportError:
    from ultralytics.utils.loss import v8DetectionLoss as _DetLoss
    _IS_E2E = False
from ultralytics.utils.ops import xyxy2xywhn
from ultralytics.cfg import get_cfg

# custom_loss.py giữ nguyên trong repo nhưng không import nữa (dùng loss chuẩn YOLO26)


class WeightEMA(nn.Module):
    """
    Exponential Moving Average for Mean Teacher with anti-collapse safeguards.
    
    θ_teacher = α_eff * θ_teacher + (1-α_eff) * θ_student
    
    Key design:
    - Only parameters updated, buffers (BatchNorm) stay frozen
    - Teacher always in eval mode
    - Call update() AFTER optimizer.step()
    - update_after_step: Delay EMA updates to let student stabilize
    - Cosine alpha ramp-up: α_eff starts at 1.0 (no update) and slowly
      decreases to target α, preventing early noise from poisoning teacher
    - pause_updates(): External quality gating to halt EMA when teacher degrades
    """
    
    def __init__(
            self,
            model,
            alpha: float = 0.999,
            freeze_teacher: bool = False,
            device: Optional[torch.device] = None,
            update_after_step: int = 500,       # Delay EMA updates
            alpha_rampup_steps: int = 2000,      # Cosine ramp-up period
    ):
        super().__init__()
        
        # Create independent copy
        self.module = deepcopy(model)
        self.module.eval()
        
        self.alpha = alpha
        self.freeze_teacher = freeze_teacher
        self.device = device
        self.update_after_step = update_after_step
        self.alpha_rampup_steps = alpha_rampup_steps
        self.updates = 0
        self.step_count = 0
        self.nan_count = 0
        self._pause_remaining = 0  # Steps to pause updates
        
        if device is not None:
            self.module.to(device=device)
        
        # Disable gradients - teacher is never trained directly
        for p in self.module.parameters():
            p.requires_grad_(False)
        
        # Save initial teacher state for divergence monitoring
        self._initial_param_norm = sum(
            p.data.norm().item() for p in self.module.parameters()
        )
        
        n_params = sum(1 for _ in self.module.parameters())
        n_buffers = sum(1 for _ in self.module.buffers())
        
        if freeze_teacher:
            print(f"[WeightEMA] FREEZE: Teacher stays pretrained")
        else:
            print(f"[WeightEMA] alpha={alpha}, {n_params} params (EMA), {n_buffers} buffers (EMA, slow-updated)")
            print(f"[WeightEMA] EMA updates start after step {update_after_step}")
            print(f"[WeightEMA] Alpha ramp-up over {alpha_rampup_steps} steps after delay")
    
    def pause_updates(self, steps: int = 500):
        """Pause EMA updates for N steps (called when teacher quality degrades)."""
        self._pause_remaining = steps
        print(f"[WeightEMA] ⚠️ Pausing EMA updates for {steps} steps")
    
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
        
        # Honor pause requests from quality gating
        if self._pause_remaining > 0:
            self._pause_remaining -= 1
            if self._pause_remaining % 100 == 0:
                print(f"[EMA] Paused, {self._pause_remaining} steps remaining")
            return
        
        # Get full state dicts to include BOTH parameters AND buffers (like BatchNorm stats)
        teacher_state = self.module.state_dict()
        student_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        
        # Check for NaN in student
        for _, sp in student_state.items():
            if sp.dtype.is_floating_point and (torch.isnan(sp).any() or torch.isinf(sp).any()):
                self.nan_count += 1
                if self.nan_count <= 5:
                    print(f"[WeightEMA] Student has NaN/Inf, skipping iteration")
                return
        
        # Cosine alpha ramp-up: effective_alpha starts at 1.0 (no update)
        # and slowly decreases to self.alpha over alpha_rampup_steps.
        # This prevents early noisy student updates from poisoning the teacher.
        steps_since_start = self.step_count - self.update_after_step
        ramp_progress = min(steps_since_start / max(self.alpha_rampup_steps, 1), 1.0)
        # Cosine schedule: 1.0 → self.alpha
        effective_alpha = 1.0 - (1.0 - self.alpha) * (1 - math.cos(math.pi * ramp_progress)) / 2
        
        # EMA: θ = α_eff*θ + (1-α_eff)*θ_student
        one_minus_alpha = 1.0 - effective_alpha
        for k, tp in teacher_state.items():
            if k not in student_state:
                continue
                
            sp = student_state[k]
            
            if tp.dtype.is_floating_point:
                # Continuous parameters and buffers (Conv weights, running_mean, running_var)
                tp.data.mul_(effective_alpha)
                tp.data.add_(sp.data.detach() * one_minus_alpha)
            else:
                # Integer buffers (e.g. num_batches_tracked)
                tp.data.copy_(sp.data)
        
        self.updates += 1
        
        if self.updates % 500 == 0:
            # Monitor parameter divergence
            current_norm = sum(p.data.norm().item() for p in self.module.parameters())
            drift = abs(current_norm - self._initial_param_norm) / max(self._initial_param_norm, 1e-8)
            print(f"[EMA] Update {self.updates}: α_eff={effective_alpha:.6f} "
                  f"(target α={self.alpha:.6f}), drift={drift:.4f}")
    
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
                 imgsz=640, augment=False, hyp=None, data=None, stride=32,
                 copy_paste_small=False, copy_paste_max_copies=3, copy_paste_small_thr=32.0):
        from ultralytics.data.dataset import YOLODataset
        from pathlib import Path
        
        self.imgsz = imgsz
        self.augment = augment
        self.copy_paste_small = copy_paste_small
        self.copy_paste_max_copies = copy_paste_max_copies
        self.copy_paste_small_thr = copy_paste_small_thr
        
        if copy_paste_small:
            print(f'[PairedDataset] CopyPaste-Small enabled: thr={copy_paste_small_thr}px, max_copies={copy_paste_max_copies}')
        
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
        
        # Copy-paste small objects for source data (has GT labels)
        if self.copy_paste_small:
            from utils.copy_paste_small import apply_small_copy_paste
            source_real = apply_small_copy_paste(
                source_real,
                small_thr=self.copy_paste_small_thr,
                max_copies=self.copy_paste_max_copies,
            )
            source_fake = apply_small_copy_paste(
                source_fake,
                small_thr=self.copy_paste_small_thr,
                max_copies=self.copy_paste_max_copies,
            )
        
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
    
    # Minimal data dict required by YOLODataset (channels, names are accessed in cache_labels/build_transforms)
    _MINIMAL_DATA = {'channels': 3, 'names': {0: 'object'}, 'nc': 1}

    def __init__(self, real_paths, fake_paths, imgsz=640, augment=True, data=None):
        from ultralytics.data.dataset import YOLODataset
        
        _data = data or self._MINIMAL_DATA
        self.real_dataset = YOLODataset(img_path=real_paths, imgsz=imgsz, augment=augment, cache=False, data=_data)
        self.fake_dataset = YOLODataset(img_path=fake_paths, imgsz=imgsz, augment=augment, cache=False, data=_data) if fake_paths else None
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


def create_dual_dataloader(real_paths, fake_paths, imgsz, batch_size, workers=4, augment=True, data=None):
    """Create dataloader for dual-domain images."""
    from ultralytics.data import YOLODataset
    from ultralytics.data.build import InfiniteDataLoader
    
    _data = data or {'channels': 3, 'names': {0: 'object'}, 'nc': 1}
    dataset = YOLODataset(img_path=real_paths, imgsz=imgsz, augment=augment, cache=False, batch_size=batch_size, data=_data)
    
    return InfiniteDataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True, collate_fn=YOLODataset.collate_fn,
    ), dataset


class FDALoss:
    """
    Combined loss for FDA training.

    Loss selection dựa trên architecture:
    - YOLO26 (NMS-free, end-to-end): sử dụng E2ELoss (one2many + one2one objectives)
    - YOLOv8/11 (NMS): sử dụng v8DetectionLoss (fallback)

    Args:
        model:         De-paralleled student model.
        class_mapping: COCO class ID → Dataset class ID mapping.
        box_gain:      Hyperparameter weight for box/IoU loss.
        cls_gain:      Hyperparameter weight for classification loss.
        dfl_gain:      DFL loss weight (v8DetectionLoss cần trường này).
    """
    
    def __init__(
        self,
        model,
        class_mapping=None,
        box_gain:  float = 7.5,
        cls_gain:  float = 0.5,
        dfl_gain:  float = 1.5,   # v8DetectionLoss bắt buộc có hyp.dfl
        use_small_object_loss: bool = False,
        inner_ratio:  float = 0.7,
        use_wise_iou: bool  = False,
    ):
        from ultralytics.utils import IterableSimpleNamespace

        # Class mapping: COCO ID → Dataset ID
        self.class_mapping = class_mapping or {0: 0, 1: 1, 2: 1}

        hyp = IterableSimpleNamespace(
            box=box_gain, cls=cls_gain, dfl=dfl_gain, pose=12.0, kobj=1.0,
            lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
            warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
            degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
            flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0,
            label_smoothing=0.0, nbs=64, overlap_mask=True, mask_ratio=4, dropout=0.0,
            epochs=200,   # cần cho E2ELoss.decay()
        )

        model.args = hyp

        # Select detection loss function
        if use_small_object_loss:
            from functools import partial
            from custom_loss import SmallObjectDetectionLoss
            loss_fn = partial(
                SmallObjectDetectionLoss,
                inner_ratio=inner_ratio,
                use_wise_iou=use_wise_iou,
            )
            print(f'[FDALoss] Using SmallObjectDetectionLoss '
                  f'(inner_ratio={inner_ratio}, wise_iou={use_wise_iou})')
        else:
            from ultralytics.utils.loss import v8DetectionLoss
            loss_fn = v8DetectionLoss

        if _IS_E2E:
            from ultralytics.utils.loss import E2ELoss
            self.detection_loss = E2ELoss(model, loss_fn=loss_fn)
        else:
            self.detection_loss = loss_fn(model)
        self.detection_loss.hyp = hyp
        self.device = next(model.parameters()).device
        
        # Separate single-head loss for distillation (one2many only).
        # E2ELoss trains both one2many + one2one heads, which doubles
        # noise amplification from pseudo-labels. For distillation,
        # only use v8DetectionLoss on one2many predictions.
        if _IS_E2E:
            from ultralytics.utils.loss import v8DetectionLoss as _V8Loss
            self.distill_loss = _V8Loss(model)
            self.distill_loss.hyp = hyp
            self._is_e2e = True
        else:
            self.distill_loss = self.detection_loss
            self._is_e2e = False
    
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
        """Compute loss using pseudo-labels from teacher.
        
        Uses v8DetectionLoss (one2many head only) instead of full E2ELoss
        to avoid double-head noise amplification from noisy pseudo-labels.
        """
        if pseudo_targets is None or len(pseudo_targets) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        batch = self._format_pseudo_targets(pseudo_targets, img_shape)
        
        if batch['cls'].numel() == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # For E2E models: extract one2many predictions only (skip one2one)
        if self._is_e2e and isinstance(student_preds, dict) and 'one2many' in student_preds:
            preds = student_preds['one2many']
        else:
            preds = student_preds
        
        loss, _ = self.distill_loss(preds, batch)
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
