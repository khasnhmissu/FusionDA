"""Format adapters: Ultralytics YOLODataset batch ↔ torchvision detection.

    Aspect              Ultralytics                         torchvision
    ------              ------------------------------      ------------------------------
    Image batching      Tensor[B, 3, H, W]                  List[Tensor[3, H, W]]
    Box parameterisation [cx, cy, w, h]  normalised [0,1]   [x1, y1, x2, y2]  pixels
    Class id            0-indexed (0 = person)              1-indexed (0 = background)
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch


def yolo_batch_to_torchvision(
    batch: dict,
    *,
    drop_degenerate: bool = True,
) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    """Convert one Ultralytics-style batch dict into torchvision detection inputs.

    batch must contain:
      - "img"       : Tensor[B, 3, H, W] in [0, 1]
      - "batch_idx" : Tensor[N] — image index per box
      - "cls"       : Tensor[N, 1] — YOLO 0-indexed class id
      - "bboxes"    : Tensor[N, 4] — normalised [cx, cy, w, h]

    Returns (images, targets) where images is a zero-copy list of slices and
    targets is List[{"boxes": [Ki, 4] xyxy abs, "labels": [Ki] int64 1-indexed}].
    """
    img = batch["img"]
    if img.dim() != 4:
        raise ValueError(
            f"batch['img'] must be [B, 3, H, W]; got shape {tuple(img.shape)}"
        )
    B, _, H, W = img.shape
    device = img.device

    bidx = batch["batch_idx"]
    if bidx.dim() > 1:
        bidx = bidx.view(-1)
    bidx = bidx.long()

    cls = batch["cls"]
    if cls.dim() > 1:
        cls = cls.view(-1)
    cls = cls.long()

    bboxes = batch["bboxes"]
    if bboxes.dim() != 2 or bboxes.shape[-1] != 4:
        raise ValueError(
            f"batch['bboxes'] must be [N, 4]; got shape {tuple(bboxes.shape)}"
        )

    images: List[torch.Tensor] = list(img)   # zero-copy along dim 0
    targets: List[Dict[str, torch.Tensor]] = []

    for b in range(B):
        mask = (bidx == b)
        if not mask.any():
            targets.append({
                "boxes":  torch.zeros((0, 4), dtype=torch.float32, device=device),
                "labels": torch.zeros((0,),   dtype=torch.int64,   device=device),
            })
            continue

        bxywhn = bboxes[mask]
        cx, cy, bw, bh = bxywhn.unbind(1)
        x1 = (cx - bw / 2.0) * W
        y1 = (cy - bh / 2.0) * H
        x2 = (cx + bw / 2.0) * W
        y2 = (cy + bh / 2.0) * H
        boxes = torch.stack([x1, y1, x2, y2], dim=1).to(torch.float32)

        # Clamp to image bounds — cleaner numerics for the proposal sampler.
        boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0.0, max=float(W))
        boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0.0, max=float(H))

        labels = (cls[mask] + 1).to(torch.int64)   # 0-indexed YOLO → 1-indexed torchvision

        if drop_degenerate:
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[valid]
            labels = labels[valid]

        targets.append({"boxes": boxes, "labels": labels})

    return images, targets


def torchvision_predictions_to_pseudo_targets(
    predictions: List[Dict[str, torch.Tensor]],
    *,
    conf_thres: float = 0.5,
) -> List[Dict[str, torch.Tensor]]:
    """Convert teacher eval-mode predictions into student-mode pseudo-targets.

    Class indices are kept 1-indexed (they came from a torchvision model) — no
    shift needed when feeding back into torchvision train mode.
    """
    targets: List[Dict[str, torch.Tensor]] = []
    for p in predictions:
        keep = p["scores"] >= conf_thres
        targets.append({
            "boxes":  p["boxes"][keep].detach(),
            "labels": p["labels"][keep].detach(),
        })
    return targets


def count_pseudo(targets: List[Dict[str, torch.Tensor]]) -> int:
    """Total pseudo-label count across a batch — used to short-circuit distillation."""
    return int(sum(int(t["boxes"].shape[0]) for t in targets))
