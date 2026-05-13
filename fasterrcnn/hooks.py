"""Feature hooks on torchvision Faster R-CNN's FPN output.

Where to hook
=============
torchvision composes the backbone as:

    FasterRCNN.backbone : BackboneWithFPN
        ├─ body : IntermediateLayerGetter wrapping ResNet-50
        │           returns OrderedDict {"0": C2, "1": C3, "2": C4, "3": C5}
        └─ fpn  : FeaturePyramidNetwork
                    returns OrderedDict {"0": P2, "1": P3, "2": P4, "3": P5,
                                         "pool": P6}

We register on the whole BackboneWithFPN module so get_features() returns the
post-FPN pyramid — exactly what the RPN and ROI heads see.

Which FPN level
===============
Default level="2" (P4, stride 16):

    P2 stride  4  → high-resolution, low semantic strength, expensive memory
    P3 stride  8  → small-object detail
    P4 stride 16  → balanced, comparable to YOLO C2PSA in role  ← default
    P5 stride 32  → strongest semantic, lowest spatial resolution
    P6 stride 64  → too coarse for cosine consistency on Cityscapes
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


DEFAULT_LEVEL = "2"   # FPN P4, stride 16


class FPNFeatureHook:
    """Capture a single FPN level after each forward pass.

    Mirrors the public API of YOLOv8FeatureHook so train_fasterrcnn.py can
    reuse the same consumption pattern (get_features() to snapshot, remove()
    at teardown).
    """

    def __init__(self, model: nn.Module, level: str = DEFAULT_LEVEL):
        if not hasattr(model, "backbone"):
            raise AttributeError(
                "FPNFeatureHook expects a torchvision-style detection model "
                "with a `.backbone` attribute (BackboneWithFPN)."
            )
        self.model = model
        self.level = str(level)
        self.features: Optional[torch.Tensor] = None
        self._handle = model.backbone.register_forward_hook(self._capture)

    def _capture(self, module, args, output):  # noqa: ARG002
        # output is an OrderedDict keyed by string level.
        if self.level not in output:
            available = list(output.keys())
            self._handle.remove()
            raise KeyError(
                f"FPN level '{self.level}' not found in backbone output. "
                f"Available levels: {available}.  "
                f"Pass an explicit ``level=`` to FPNFeatureHook(...)."
            )
        self.features = output[self.level]

    def get_features(self) -> Optional[torch.Tensor]:
        """Return most recently captured [B, 256, H, W] tensor, or None."""
        return self.features

    def remove(self) -> None:
        """De-register the forward hook. Safe to call multiple times."""
        try:
            self._handle.remove()
        except Exception:
            pass
        self.features = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: ARG002
        self.remove()
