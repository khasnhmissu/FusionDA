"""torchvision Faster R-CNN R-50-FPN factory for FusionDA.

Why torchvision (not Detectron2/MaskRCNN-Benchmark)
====================================================
torchvision.models.detection ships a production-grade Faster R-CNN R-50-FPN
that is bit-equivalent in training behaviour to Detectron2's
faster_rcnn_R_50_FPN_3x: same ResNet-50 stem + bottleneck blocks, same FPN
top-down pathway, same RPN + RoIAlign + 2-FC ROI head.  Using torchvision
keeps the experiment inside the FusionDA repo.

Important notes
===============
- num_classes is +1 the user-facing count: torchvision reserves index 0 for
  "background".  build_fasterrcnn(num_classes=2) builds a head with shape
  [3 = bg + person + car].  The adapters in adapter.py shift class ids by +1
  on input and -1 on output so callers never see this.
- Pretrained backbone = ImageNet only, never COCO.  Loading COCO weights would
  introduce prior knowledge from outside the source domain and contaminate the
  baseline number.
- min_size / max_size = 640 by default to match the YOLO26-s and YOLOv5m rows.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def build_fasterrcnn(
    num_classes: int = 2,
    min_size: int = 640,
    max_size: int = 640,
    pretrained_backbone: bool = True,
    trainable_backbone_layers: Optional[int] = None,
) -> nn.Module:
    """Construct a torchvision Faster R-CNN R-50-FPN for FusionDA.

    num_classes: foreground class count (background added internally).
    pretrained_backbone: ImageNet-pretrained ResNet-50 stem.
    trainable_backbone_layers: top-layer blocks left un-frozen (torchvision
        default is 3 out of 5).
    """
    from torchvision.models.detection import fasterrcnn_resnet50_fpn

    # torchvision >= 0.13 uses weight-enum kwargs; older versions use
    # pretrained_backbone=.  Try new API first, fall back for older installs.
    kwargs = dict(
        num_classes=num_classes + 1,   # +1 for background
        min_size=min_size,
        max_size=max_size,
    )
    if trainable_backbone_layers is not None:
        kwargs["trainable_backbone_layers"] = trainable_backbone_layers

    try:
        from torchvision.models import ResNet50_Weights
        weights_backbone = (
            ResNet50_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        )
        model = fasterrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=weights_backbone,
            **kwargs,
        )
    except ImportError:
        model = fasterrcnn_resnet50_fpn(
            pretrained=False,
            pretrained_backbone=pretrained_backbone,
            **kwargs,
        )

    return model


def fasterrcnn_param_summary(model: nn.Module) -> dict:
    """Parameter counts by submodule. Used by smoke tests only."""
    counts = {}
    counts["total"] = sum(p.numel() for p in model.parameters())
    if hasattr(model, "backbone"):
        counts["backbone"] = sum(
            p.numel() for p in model.backbone.parameters()
        )
    if hasattr(model, "rpn"):
        counts["rpn"] = sum(p.numel() for p in model.rpn.parameters())
    if hasattr(model, "roi_heads"):
        counts["roi_heads"] = sum(
            p.numel() for p in model.roi_heads.parameters()
        )
    return counts
