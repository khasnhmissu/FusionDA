"""FasterRCNNLoss — equivalent of fusion_da.FDALoss for torchvision.

Mirrors FDALoss's public API: __call__ returns (total, loss_items) and
compute_distillation_loss accepts (images, pseudo_targets, img_shape).
This lets train_fasterrcnn.py share the same loop shape as train.py.

torchvision Faster R-CNN loss dict (model.train() mode):
    {
        "loss_classifier":   ROI head classification loss  (cross-entropy)
        "loss_box_reg":      ROI head box regression        (smooth L1)
        "loss_objectness":   RPN objectness                 (BCE)
        "loss_rpn_box_reg":  RPN box regression             (smooth L1)
    }

compute_distillation_loss short-circuits with 0-loss when the teacher returned
no usable pseudo-labels for the entire batch.  torchvision tolerates per-image
empties internally, but a whole-batch empty creates degenerate inputs to the
proposal sampler.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .adapter import count_pseudo


# Chosen to mirror the RPN-then-ROI conceptual order so logs read top-down
# through the detector.
LOSS_COMPONENT_ORDER = (
    "loss_objectness",
    "loss_rpn_box_reg",
    "loss_classifier",
    "loss_box_reg",
)


class FasterRCNNLoss:
    """Aggregator and distillation helper for torchvision Faster R-CNN."""

    def __init__(
        self,
        model: nn.Module,
        gain_classifier: float = 1.0,
        gain_box_reg: float = 1.0,
        gain_objectness: float = 1.0,
        gain_rpn_box_reg: float = 1.0,
    ):
        self.model = model
        self.gains: Dict[str, float] = {
            "loss_classifier":  float(gain_classifier),
            "loss_box_reg":     float(gain_box_reg),
            "loss_objectness":  float(gain_objectness),
            "loss_rpn_box_reg": float(gain_rpn_box_reg),
        }
        # Cache device for zero-loss short-circuits when no pseudo-labels exist.
        try:
            self._device = next(model.parameters()).device
        except StopIteration:
            self._device = torch.device("cpu")

    def __call__(
        self,
        images: List[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run model in train mode and return (weighted_total, loss_items[4]).

        images:  List[Tensor[3, H, W]] in [0, 1] float.
        targets: List[{"boxes": [Ki, 4] xyxy abs, "labels": [Ki] int64 1-indexed}].
        loss_items: detached float[4] in LOSS_COMPONENT_ORDER for logging.
        """
        if not self.model.training:
            raise RuntimeError(
                "FasterRCNNLoss expects model in train() mode."
            )
        loss_dict = self.model(images, targets)
        # Tolerates missing keys — some forks omit objectness when RPN is frozen.
        total = sum(
            loss_dict[k] * self.gains[k]
            for k in self.gains
            if k in loss_dict
        )
        if not isinstance(total, torch.Tensor):
            total = torch.tensor(0.0, device=self._device, requires_grad=True)
        loss_items = torch.tensor(
            [
                float(loss_dict[k].detach()) if k in loss_dict else 0.0
                for k in LOSS_COMPONENT_ORDER
            ],
            dtype=torch.float32,
        )
        return total, loss_items

    def compute_distillation_loss(
        self,
        images: List[torch.Tensor],
        pseudo_targets: List[Dict[str, torch.Tensor]],
        img_shape: Optional[Tuple[int, int]] = None,  # noqa: ARG002 — kept for parity with FDALoss
    ) -> torch.Tensor:
        """Run model on student inputs with teacher pseudo-labels as GT.

        img_shape is accepted but unused — torchvision recovers image sizes
        from the input tensors.  Returns 0 (with grad) if all pseudo-targets
        are empty.
        """
        n = count_pseudo(pseudo_targets)
        if n == 0:
            return torch.tensor(0.0, device=self._device, requires_grad=True)

        if not self.model.training:
            raise RuntimeError(
                "FasterRCNNLoss.compute_distillation_loss expects model in train() mode."
            )
        loss_dict = self.model(images, pseudo_targets)
        return sum(
            loss_dict[k] * self.gains[k]
            for k in self.gains
            if k in loss_dict
        )
