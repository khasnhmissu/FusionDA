"""FusionDA — torchvision Faster R-CNN R-50-FPN backend.

Adds a two-stage Faster R-CNN training path without modifying existing FusionDA
code, enabling apples-to-apples comparison against ALDI's R-50-FPN.

Public API:
  build_fasterrcnn                        — model factory (ImageNet backbone, no COCO leakage)
  FasterRCNNLoss                          — equivalent of fusion_da.FDALoss
  FPNFeatureHook                          — equivalent of YOLOv8FeatureHook (captures FPN level)
  yolo_batch_to_torchvision               — Ultralytics batch dict → torchvision list-of-dicts
  torchvision_predictions_to_pseudo_targets — teacher predictions → student pseudo-targets
"""

from .adapter import (
    yolo_batch_to_torchvision,
    torchvision_predictions_to_pseudo_targets,
)
from .hooks import FPNFeatureHook
from .loss import FasterRCNNLoss
from .model import build_fasterrcnn

__all__ = [
    "build_fasterrcnn",
    "FasterRCNNLoss",
    "FPNFeatureHook",
    "yolo_batch_to_torchvision",
    "torchvision_predictions_to_pseudo_targets",
]
