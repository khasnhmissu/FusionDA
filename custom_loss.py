"""
custom_loss.py — FusionDA custom detection loss components.

Replaces CIoU with Inner-CIoU (arXiv:2311.02877) and optionally adds
WiseIoU v3 reweighting (arXiv:2301.10051).  Also provides a scale-aware
TaskAlignedAssigner that gives MORE topk to small GT boxes.

Drop-in replacement for v8DetectionLoss: just swap FDALoss to use
SmallObjectDetectionLoss instead of v8DetectionLoss.

Usage in fusion_da.py:
    from custom_loss import SmallObjectDetectionLoss
    self.detection_loss = SmallObjectDetectionLoss(
        model,
        inner_ratio=0.7,
        use_wise_iou=False,   # set True to enable WiseIoU reweighting
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import v8DetectionLoss, BboxLoss
from ultralytics.utils.tal import TaskAlignedAssigner, make_anchors, dist2bbox, bbox2dist
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.ops import xywh2xyxy


# ---------------------------------------------------------------------------
# Utility: Inner-box scaling
# ---------------------------------------------------------------------------

def _scale_box_to_center(boxes: torch.Tensor, ratio: float) -> torch.Tensor:
    """
    Scale xyxy boxes inward toward their center by *ratio*.

    ratio=1.0  → no change (original box)
    ratio=0.7  → 70% of original width/height, centred on same centre
    ratio=0.5  → half-size auxiliary box

    Args:
        boxes: (..., 4) in xyxy format
        ratio: scalar in (0, 1]
    Returns:
        scaled boxes (..., 4) in xyxy format
    """
    cx = (boxes[..., 0] + boxes[..., 2]) * 0.5
    cy = (boxes[..., 1] + boxes[..., 3]) * 0.5
    half_w = (boxes[..., 2] - boxes[..., 0]) * 0.5 * ratio
    half_h = (boxes[..., 3] - boxes[..., 1]) * 0.5 * ratio
    return torch.stack([cx - half_w, cy - half_h, cx + half_w, cy + half_h], dim=-1)


# ---------------------------------------------------------------------------
# Inner-CIoU loss computation
# ---------------------------------------------------------------------------

def inner_ciou_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    ratio: float = 0.7,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Inner-CIoU: uses auxiliary (inner) boxes for the IoU signal, but keeps
    the full-box CIoU penalty terms (centre distance + aspect-ratio).

    Paper: "Inner-IoU: More Effective Intersection over Union Loss with
            Auxiliary Bounding Box" (arXiv:2311.02877).

    Args:
        pred, target: (..., 4) xyxy
        ratio: auxiliary box scale factor. 0.7 works well for mixed datasets;
               use 0.5–0.65 if your small objects are very tiny.
        eps: numerical stability epsilon
    Returns:
        loss tensor (same leading shape as pred/target), range [0, 2+]
    """
    # ---- Inner-IoU term (auxiliary boxes) --------------------------------
    inner_pred   = _scale_box_to_center(pred,   ratio)
    inner_target = _scale_box_to_center(target, ratio)

    # intersect
    inter_x1 = torch.max(inner_pred[..., 0], inner_target[..., 0])
    inter_y1 = torch.max(inner_pred[..., 1], inner_target[..., 1])
    inter_x2 = torch.min(inner_pred[..., 2], inner_target[..., 2])
    inter_y2 = torch.min(inner_pred[..., 3], inner_target[..., 3])
    inter_w  = (inter_x2 - inter_x1).clamp(min=0)
    inter_h  = (inter_y2 - inter_y1).clamp(min=0)
    inter    = inter_w * inter_h

    pred_area   = (inner_pred[..., 2]   - inner_pred[..., 0]).clamp(min=0)   \
                * (inner_pred[..., 3]   - inner_pred[..., 1]).clamp(min=0)
    target_area = (inner_target[..., 2] - inner_target[..., 0]).clamp(min=0) \
                * (inner_target[..., 3] - inner_target[..., 1]).clamp(min=0)
    union = pred_area + target_area - inter + eps
    iou   = inter / union                                  # Inner-IoU

    # ---- CIoU penalty terms (FULL boxes) ----------------------------------
    pw = (pred[..., 2]   - pred[..., 0]).clamp(min=0)
    ph = (pred[..., 3]   - pred[..., 1]).clamp(min=0)
    tw = (target[..., 2] - target[..., 0]).clamp(min=0)
    th = (target[..., 3] - target[..., 1]).clamp(min=0)

    # Centre distance
    pcx = (pred[..., 0]   + pred[..., 2])   * 0.5
    pcy = (pred[..., 1]   + pred[..., 3])   * 0.5
    tcx = (target[..., 0] + target[..., 2]) * 0.5
    tcy = (target[..., 1] + target[..., 3]) * 0.5

    # Enclosing box diagonal²
    enc_x1 = torch.min(pred[..., 0], target[..., 0])
    enc_y1 = torch.min(pred[..., 1], target[..., 1])
    enc_x2 = torch.max(pred[..., 2], target[..., 2])
    enc_y2 = torch.max(pred[..., 3], target[..., 3])
    c2     = (enc_x2 - enc_x1).pow(2) + (enc_y2 - enc_y1).pow(2) + eps

    rho2 = (pcx - tcx).pow(2) + (pcy - tcy).pow(2)       # centre dist²
    v    = (4 / (torch.pi ** 2)) * (torch.atan(tw / (th + eps)) - torch.atan(pw / (ph + eps))).pow(2)

    with torch.no_grad():
        alpha_ciou = v / (1 - iou + v + eps)              # CIoU α

    ciou_penalty = rho2 / c2 + v * alpha_ciou

    # ---- Final Inner-CIoU loss -------------------------------------------
    return 1.0 - iou + ciou_penalty                        # same sign convention as 1-CIoU


# ---------------------------------------------------------------------------
# WiseIoU v3 reweighting wrapper
# ---------------------------------------------------------------------------

class WiseIoUReweighter:
    """
    WiseIoU v3 non-monotonic focusing coefficient.

    Paper: "Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing
            Mechanism" (arXiv:2301.10051, v3).

    The reweighting factor R is computed per-sample and DETACHED before
    multiplication so it does not contribute to second-order gradients.

    Usage::
        wise = WiseIoUReweighter()
        loss = wise(iou_loss_per_sample)   # returns scalar
    """

    def __init__(self, momentum: float = 0.02, delta: float = 3.0, alpha: float = 1.0):
        """
        Args:
            momentum: EMA momentum for the moving-average IoU loss (L̄).
            delta:    Controls sharpness of the non-monotonic curve.
            alpha:    Scaling base in the exponent.
        """
        self.momentum = momentum
        self.delta    = delta
        self.alpha    = alpha
        self._moving_avg: float | None = None   # scalar moving average

    def __call__(self, iou_loss_per_sample: torch.Tensor) -> torch.Tensor:
        """
        Args:
            iou_loss_per_sample: 1-D tensor of per-anchor IoU losses.
        Returns:
            Weighted sum (scalar).
        """
        # Update moving average L̄
        batch_mean = iou_loss_per_sample.detach().mean().item()
        if self._moving_avg is None:
            self._moving_avg = batch_mean
        else:
            self._moving_avg = (1 - self.momentum) * self._moving_avg + self.momentum * batch_mean

        # Outlier degree β = L / L̄
        beta = iou_loss_per_sample.detach() / (self._moving_avg + 1e-9)

        # Non-monotonic focusing coefficient R
        # R = exp((β - β*) / (δ · α^(β - β*)))
        # where β* = mean(β) acts as the "ordinary quality" reference
        beta_star = beta.mean()
        diff      = beta - beta_star
        r         = torch.exp(diff / (self.delta * self.alpha ** diff.clamp(-10, 10)))

        # Detach so R is a pure weighting factor
        return (r.detach() * iou_loss_per_sample).sum()


# ---------------------------------------------------------------------------
# Custom BboxLoss: Inner-CIoU  (+optionally WiseIoU)
# ---------------------------------------------------------------------------

class InnerCIoUBboxLoss(BboxLoss):
    """
    BboxLoss with Inner-CIoU replacing the standard CIoU.

    Inherits DFL computation unchanged from the parent BboxLoss.
    Optionally wraps the per-sample IoU losses with WiseIoU v3.

    Args:
        reg_max:       Inherited from parent (= m.reg_max - 1).
        use_dfl:       Inherited from parent.
        inner_ratio:   Auxiliary box scale factor for Inner-IoU (0 < r ≤ 1).
        use_wise_iou:  If True, apply WiseIoU v3 gradient reweighting.
        wise_momentum: EMA momentum for WiseIoU moving-average tracker.
    """

    def __init__(
        self,
        reg_max: int,
        use_dfl: bool = False,   # kept for API compat; ignored — inferred from reg_max
        inner_ratio: float = 0.7,
        use_wise_iou: bool = False,
        wise_momentum: float = 0.02,
    ):
        # Modern Ultralytics BboxLoss only accepts reg_max; use_dfl is gone.
        super().__init__(reg_max)
        self.inner_ratio  = inner_ratio
        self.use_wise_iou = use_wise_iou
        self._wise        = WiseIoUReweighter(momentum=wise_momentum) if use_wise_iou else None

    def forward(
        self,
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
        imgsz=None,    # new Ultralytics arg (used for L1 normalisation when DFL off)
        stride=None,   # new Ultralytics arg
    ):
        """Compute Inner-CIoU loss (+ optional WiseIoU) and DFL loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)   # (N_fg, 1)

        # Per-sample Inner-CIoU loss  (shape: N_fg)
        per_sample_loss = inner_ciou_loss(
            pred_bboxes[fg_mask],
            target_bboxes[fg_mask],
            ratio=self.inner_ratio,
        ).squeeze(-1)                                            # ensure 1-D

        if self.use_wise_iou and self._wise is not None:
            # WiseIoU: weighted sum, already normalised inside the reweighter
            loss_iou = self._wise(per_sample_loss * weight.squeeze(-1)) / target_scores_sum
        else:
            loss_iou = (per_sample_loss * weight.squeeze(-1)).sum() / target_scores_sum

        # DFL — mirrors new BboxLoss.forward() logic exactly
        if self.dfl_loss is not None:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(
                pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                target_ltrb[fg_mask],
            ) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            # No DFL: L1 loss normalised by image size (same as parent's else-branch)
            if imgsz is not None and stride is not None:
                target_ltrb = bbox2dist(anchor_points, target_bboxes)
                target_ltrb = target_ltrb * stride
                target_ltrb[..., 0::2] /= imgsz[1]
                target_ltrb[..., 1::2] /= imgsz[0]
                _pred = pred_dist * stride
                _pred[..., 0::2] /= imgsz[1]
                _pred[..., 1::2] /= imgsz[0]
                loss_dfl = (
                    F.l1_loss(_pred[fg_mask], target_ltrb[fg_mask], reduction="none")
                    .mean(-1, keepdim=True) * weight
                )
                loss_dfl = loss_dfl.sum() / target_scores_sum
            else:
                loss_dfl = torch.tensor(0.0, device=pred_dist.device)

        return loss_iou, loss_dfl


# ---------------------------------------------------------------------------
# Scale-Aware TaskAlignedAssigner
# ---------------------------------------------------------------------------

class ScaleAwareTaskAlignedAssigner(TaskAlignedAssigner):
    """
    TAL assigner with scale-adaptive topk.

    Key insight (opposite to naive intuition):
    - SMALL GT boxes → HIGHER topk
        Small objects lie in very few anchor cells.  Topk=10 might select
        anchors that never truly overlap the GT, giving near-zero IoU signal.
        Raising topk for small GTs ensures enough positive candidates are
        considered from the neighbourhood, even when strict overlap is sparse.

    - LARGE GT boxes  → LOWER topk (or same as default)
        Large objects already generate tens of overlapping anchors; the
        standard topk=10 selection already finds good candidates.
        Raising topk for large objects just adds redundant positives.

    Size thresholds (in units of anchor stride × grid size = pixels at input
    resolution, i.e., the product of the stride of the finest scale head).
    You can tune small_thr / large_thr via the constructor.

    Args:
        topk_small:  topk used when GT area is below small_thr².
        topk_medium: topk used for medium GT boxes (default).
        topk_large:  topk used when GT area is above large_thr².
        small_thr:   GT side threshold (pixels) below which → topk_small.
        large_thr:   GT side threshold (pixels) above which → topk_large.
        num_classes, alpha, beta, eps: forwarded to parent.
    """

    def __init__(
        self,
        topk_small:  int   = 20,
        topk_medium: int   = 13,
        topk_large:  int   = 7,
        small_thr:   float = 32.0,
        large_thr:   float = 96.0,
        num_classes: int   = 80,
        alpha:       float = 0.5,
        beta:        float = 6.0,
        eps:         float = 1e-9,
    ):
        # Initialise with the MAXIMUM topk so that parent buffers are large
        # enough; we will override select_topk_candidates per GT.
        super().__init__(
            topk=topk_small,          # largest value
            num_classes=num_classes,
            alpha=alpha,
            beta=beta,
            eps=eps,
        )
        self.topk_small  = topk_small
        self.topk_medium = topk_medium
        self.topk_large  = topk_large
        self.small_thr   = small_thr
        self.large_thr   = large_thr

    # ------------------------------------------------------------------
    # Override: replace the fixed-topk selection with a per-GT one
    # ------------------------------------------------------------------
    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """Same as parent but with scale-dependent topk per GT."""
        # New Ultralytics signature requires mask_gt to skip padded GTs early.
        try:
            mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, mask_gt)
        except TypeError:
            # Fallback for older Ultralytics versions that don't take mask_gt.
            mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes)
        align_metric, overlaps = self.get_box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt
        )

        # --- scale-adaptive topk -------------------------------------------
        # gt_bboxes: (bs, n_max_boxes, 4) in xyxy (pixel coords)
        gt_w   = (gt_bboxes[..., 2] - gt_bboxes[..., 0]).clamp(min=0)   # (bs, n_max)
        gt_h   = (gt_bboxes[..., 3] - gt_bboxes[..., 1]).clamp(min=0)
        gt_side = torch.sqrt(gt_w * gt_h + 1e-9)                          # geometric mean side

        # Per-GT topk tensor: (bs, n_max_boxes)
        topk_per_gt = torch.where(
            gt_side < self.small_thr,
            torch.full_like(gt_side, self.topk_small,  dtype=torch.long),
            torch.where(
                gt_side > self.large_thr,
                torch.full_like(gt_side, self.topk_large,  dtype=torch.long),
                torch.full_like(gt_side, self.topk_medium, dtype=torch.long),
            )
        )   # (bs, n_max_boxes)  — long tensor

        # We cap at the actual number of anchors to avoid topk OOB
        n_anchors = align_metric.shape[-1]
        topk_per_gt = topk_per_gt.clamp(max=n_anchors)

        mask_topk = self._select_topk_per_gt(align_metric, topk_per_gt, mask_gt)
        mask_pos  = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    @staticmethod
    def _select_topk_per_gt(
        metrics:     torch.Tensor,    # (bs, n_max_boxes, n_anchors)
        topk_per_gt: torch.Tensor,    # (bs, n_max_boxes) int
        mask_gt:     torch.Tensor,    # (bs, n_max_boxes, 1)
    ) -> torch.Tensor:
        """
        Select top-k anchors per GT where k varies per GT box.

        Returns:
            count_tensor: (bs, n_max_boxes, n_anchors) float mask
        """
        bs, n_max, n_anc = metrics.shape
        count_tensor = torch.zeros_like(metrics)    # float

        for b in range(bs):
            for g in range(n_max):
                if mask_gt[b, g, 0] == 0:
                    continue                         # padded GT
                k = int(topk_per_gt[b, g].item())
                k = max(1, min(k, n_anc))
                _, topk_idxs = torch.topk(metrics[b, g], k, dim=-1, largest=True)
                count_tensor[b, g].scatter_(0, topk_idxs, 1.0)

        # If an anchor is assigned to multiple GTs, keep the one with
        # the highest alignment metric (same as parent's logic).
        # We reuse the parent's static helper via argmax resolution below.
        return count_tensor


# ---------------------------------------------------------------------------
# SmallObjectDetectionLoss  (drop-in for v8DetectionLoss in FDALoss)
# ---------------------------------------------------------------------------

class SmallObjectDetectionLoss(v8DetectionLoss):
    """
    v8DetectionLoss with:
      1. Inner-CIoU replacing CIoU  (arXiv:2311.02877)
      2. Optional WiseIoU v3 gradient reweighting  (arXiv:2301.10051)
      3. Scale-aware TaskAlignedAssigner (higher topk → small GT boxes)

    Drop-in replacement: the returned (loss_sum, loss_items) tuple is
    identical to v8DetectionLoss, so the rest of the training pipeline
    (FDALoss, distillation path, logging) requires NO changes.

    Args:
        model:         De-paralleled student model (forwarded to parent).
        inner_ratio:   Inner-box scale for Inner-CIoU  (default 0.7).
        use_wise_iou:  Enable WiseIoU v3 reweighting  (default False).
        topk_small:    topk for GT side < small_thr   (default 20).
        topk_medium:   topk for medium GT             (default 13).
        topk_large:    topk for GT side > large_thr   (default 7).
        small_thr:     Pixel threshold for small GT   (default 32).
        large_thr:     Pixel threshold for large GT   (default 96).
    """

    def __init__(
        self,
        model,
        inner_ratio:   float = 0.7,
        use_wise_iou:  bool  = False,
        topk_small:    int   = 20,
        topk_medium:   int   = 13,
        topk_large:    int   = 7,
        small_thr:     float = 32.0,
        large_thr:     float = 96.0,
    ):
        super().__init__(model)   # sets up self.bce, self.hyp, strides, etc.

        # --- Replace BboxLoss with Inner-CIoU version ---
        self.bbox_loss = InnerCIoUBboxLoss(
            reg_max       = self.reg_max,   # new API: pass full reg_max (e.g. 16), not reg_max-1
            use_dfl       = self.use_dfl,
            inner_ratio   = inner_ratio,
            use_wise_iou  = use_wise_iou,
        ).to(self.device)

        # --- Replace TAL assigner with scale-aware version ---
        self.assigner = ScaleAwareTaskAlignedAssigner(
            topk_small  = topk_small,
            topk_medium = topk_medium,
            topk_large  = topk_large,
            small_thr   = small_thr,
            large_thr   = large_thr,
            num_classes = self.nc,
            alpha       = 0.5,
            beta        = 6.0,
        )

        _wise_str = "WiseIoU+InnerCIoU" if use_wise_iou else "InnerCIoU"
        print(
            f"[SmallObjectDetectionLoss] {_wise_str} "
            f"(ratio={inner_ratio}), "
            f"ScaleAwareTAL topk: small={topk_small}, mid={topk_medium}, large={topk_large} "
            f"thresholds: <{small_thr}px / >{large_thr}px"
        )
