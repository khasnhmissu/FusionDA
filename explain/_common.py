"""Shared utilities for explainability scripts.

All explain/rq*.py scripts import from here to avoid code duplication and to
ensure consistent checkpoint loading, YOLO26 E2E output parsing, matching
logic, and preprocessing.

Key design choices that differ from the old root-level scripts:
  * Checkpoints saved by train.py are dicts with keys 'model' (student) and
    optionally 'teacher_state_dict' (only in checkpoint_ep*.pt files). The
    load helper takes a `which='student'|'teacher'` argument.
  * YOLO26 E2E model in eval mode returns `(postproc_one2one [B,300,6],
    raw_dict)`. We prefer the postproc tensor (already decoded + NMS-free
    by design) over one2many which needs manual NMS.
  * The user's `yolo26s.pt` is a 2-class model (person=0, car=1). No COCO
    class remapping is needed anywhere; the old `class_mapping={0:0, 2:1}`
    in detection_diff was silently dropping class 1 (car) predictions.
  * Per-size buckets follow COCO: small <32², medium 32²-96², large >96².
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import yaml


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

# COCO size thresholds on bbox area in ORIGINAL image pixel coords.
AREA_SMALL = 32 ** 2    # < 1024 px²
AREA_MEDIUM = 96 ** 2   # 1024-9216 px². Above 9216 = large.


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def letterbox(img, new_shape=640, stride=32, color=(114, 114, 114)):
    """Resize with unchanged aspect ratio + pad. Returns (img, ratio, (dw, dh))."""
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, (r, r), (dw, dh)


def scale_boxes_to_original(boxes_xyxy, img_shape_orig, ratio, pad):
    """Undo letterbox on xyxy boxes. In-place on `boxes_xyxy`."""
    dw, dh = pad
    r = ratio[0]
    boxes_xyxy[:, 0] -= dw
    boxes_xyxy[:, 1] -= dh
    boxes_xyxy[:, 2] -= dw
    boxes_xyxy[:, 3] -= dh
    boxes_xyxy[:, :4] /= r
    h_orig, w_orig = img_shape_orig
    boxes_xyxy[:, 0].clamp_(0, w_orig)
    boxes_xyxy[:, 1].clamp_(0, h_orig)
    boxes_xyxy[:, 2].clamp_(0, w_orig)
    boxes_xyxy[:, 3].clamp_(0, h_orig)
    return boxes_xyxy


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_model_from_ckpt(arch_weights, ckpt_path, device, which='student'):
    """Load YOLO architecture from `arch_weights`, overwrite weights from `ckpt_path`.

    Parameters
    ----------
    arch_weights : str
        Path to a YOLO .pt file whose architecture matches the checkpoint
        (e.g. 'yolo26s.pt'). Only the architecture is taken; state_dict is
        replaced by the checkpoint.
    ckpt_path : str
        Path to the training checkpoint.
    device : torch.device | str
    which : {'student', 'teacher'}
        Which state to load:
          - 'student': 'model' key (best.pt, last.pt, checkpoint_ep*.pt — all have it)
          - 'teacher': 'teacher_state_dict' key (only checkpoint_ep*.pt)

    Returns
    -------
    model : nn.Module in eval mode.
    """
    from ultralytics import YOLO

    yolo = YOLO(arch_weights)
    model = yolo.model.to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict):
        if which == 'teacher':
            if 'teacher_state_dict' not in ckpt:
                raise KeyError(
                    f"'teacher_state_dict' not found in {ckpt_path}. "
                    f"Only checkpoint_ep*.pt files contain teacher state. "
                    f"Keys present: {list(ckpt.keys())}"
                )
            state = ckpt['teacher_state_dict']
        else:
            state = ckpt.get('model', ckpt.get('state_dict', ckpt))
    else:
        state = ckpt

    # nn.Module wrapper — unwrap
    if hasattr(state, 'state_dict'):
        state = state.float().state_dict() if hasattr(state, 'float') else state.state_dict()

    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(missing) > 10 or len(unexpected) > 10:
        print(f"[LoadCkpt] ⚠ many mismatched keys when loading '{which}' "
              f"from {Path(ckpt_path).name}: "
              f"missing={len(missing)}, unexpected={len(unexpected)}. "
              f"Check that --weights architecture matches the checkpoint.")
    model.eval()
    return model


def load_class_names(data_yaml):
    """Return {int: str} from data.yaml's `names` field."""
    with open(data_yaml, encoding='utf-8') as f:
        d = yaml.safe_load(f)
    names = d.get('names', {})
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    return {int(k): v for k, v in names.items()}


# ---------------------------------------------------------------------------
# YOLO26 E2E output parsing
# ---------------------------------------------------------------------------

def parse_yolo26_output(pred):
    """Normalise a YOLO26/YOLOv8 raw forward output to [B, N, 6] (or raw [B, 4+nc, N]).

    YOLO26 E2E eval returns `(postproc_one2one [B, 300, 6], raw_dict)`.
    The postprocessed tensor is decoded + NMS-free by design — that is what
    we prefer.

    For non-E2E YOLOv8 we get raw logits that still need non_max_suppression
    downstream.
    """
    if isinstance(pred, tuple) and len(pred) == 2:
        postproc, raw = pred
        if isinstance(postproc, torch.Tensor) and postproc.ndim == 3 and postproc.shape[-1] == 6:
            return postproc
        pred = raw

    if isinstance(pred, torch.Tensor) and pred.ndim == 3 and pred.shape[-1] == 6:
        return pred

    # Raw tensor or dict path
    if isinstance(pred, dict):
        # Prefer one2one over one2many for E2E (sparse, cleaner).
        cand = pred.get('one2one', pred.get('one2many'))
        if isinstance(cand, dict):
            # {'scores': ..., 'boxes': ...} raw — caller must decode. Return dict.
            return pred
        pred = cand

    if isinstance(pred, (list, tuple)):
        pred = pred[0]

    if not isinstance(pred, torch.Tensor):
        raise ValueError(f"Cannot parse prediction: got {type(pred)}")

    if pred.ndim == 3 and pred.shape[-1] == 6:
        return pred

    # Raw [B, N, 4+nc] or [B, 4+nc, N]
    if pred.ndim == 3 and pred.shape[-1] > pred.shape[1]:
        pred = pred.permute(0, 2, 1)
    if pred.ndim == 3:
        cls = pred[:, 4:, :]
        if cls.max() > 1.0 or cls.min() < 0.0:
            pred = torch.cat([pred[:, :4, :], torch.sigmoid(cls)], dim=1)
    return pred


@torch.no_grad()
def run_inference_single(model, img_path, device, imgsz=640,
                         conf_thres=0.25, iou_thres=0.45, max_det=300):
    """Run model on a single image path, return detections in ORIGINAL image coords.

    Returns
    -------
    np.ndarray [N, 6] — columns [x1, y1, x2, y2, conf, cls]
    """
    from ultralytics.utils.nms import non_max_suppression

    gs = max(int(model.stride.max()), 32)
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return np.array([]).reshape(0, 6)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = img_rgb.shape[:2]

    img_lb, ratio, (dw, dh) = letterbox(img_rgb, new_shape=imgsz, stride=gs)
    img_t = torch.from_numpy(img_lb.transpose(2, 0, 1)).to(device).float() / 255.0
    img_t = img_t.unsqueeze(0)

    pred = parse_yolo26_output(model(img_t))

    if isinstance(pred, torch.Tensor) and pred.ndim == 3 and pred.shape[-1] == 6:
        det = pred[0]
        det = det[det[:, 4] > conf_thres]
    else:
        dets = non_max_suppression(pred, conf_thres=conf_thres,
                                    iou_thres=iou_thres, max_det=max_det)
        det = dets[0]

    if det is None or len(det) == 0:
        return np.array([]).reshape(0, 6)

    det = det.clone()
    det[:, :4] = scale_boxes_to_original(det[:, :4], (h_orig, w_orig), ratio, (dw, dh))
    return det.cpu().numpy()


# ---------------------------------------------------------------------------
# GT + matching
# ---------------------------------------------------------------------------

def load_gt_labels(label_path, img_w, img_h):
    """Parse a YOLO-format .txt and return np.ndarray [N, 5] = [cls, x1, y1, x2, y2]
    in pixel coordinates. Missing file → empty array."""
    boxes = []
    if not os.path.exists(label_path):
        return np.array([]).reshape(0, 5)
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            xc, yc, w, h = map(float, parts[1:5])
            x1 = (xc - w / 2) * img_w
            y1 = (yc - h / 2) * img_h
            x2 = (xc + w / 2) * img_w
            y2 = (yc + h / 2) * img_h
            boxes.append([cls_id, x1, y1, x2, y2])
    return np.array(boxes) if boxes else np.array([]).reshape(0, 5)


def box_area_xyxy(box):
    """Area of an xyxy box. Works on 1D or 2D np arrays."""
    a = np.atleast_2d(box)
    return np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)


def size_bucket(area):
    """COCO size category: 'small' | 'medium' | 'large'."""
    if area < AREA_SMALL:
        return 'small'
    if area < AREA_MEDIUM:
        return 'medium'
    return 'large'


def compute_iou(b1, b2):
    """IoU of two xyxy boxes (1D arrays)."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    u = a1 + a2 - inter
    return inter / u if u > 0 else 0.0


def match_predictions_to_gt(preds, gt_boxes, iou_thres=0.5):
    """Greedy confidence-ranked matching.

    Parameters
    ----------
    preds : np.ndarray [N, 6]   — [x1, y1, x2, y2, conf, cls]
    gt_boxes : np.ndarray [M, 5] — [cls, x1, y1, x2, y2]
    iou_thres : float

    Returns
    -------
    matched : list[(pred_idx, gt_idx, iou)]
    unmatched_preds : list[int]   # FPs
    unmatched_gt    : list[int]   # FNs
    """
    if len(preds) == 0 and len(gt_boxes) == 0:
        return [], [], []
    if len(preds) == 0:
        return [], [], list(range(len(gt_boxes)))
    if len(gt_boxes) == 0:
        return [], list(range(len(preds))), []

    matched = []
    matched_gt = set()
    unmatched_preds = []

    order = np.argsort(-preds[:, 4])  # descending conf
    for pi in order:
        pred_cls = int(preds[pi, 5])
        best_iou, best_gi = 0.0, -1
        for gi, gt in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            if int(gt[0]) != pred_cls:
                continue
            iou = compute_iou(preds[pi, :4], gt[1:5])
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_iou >= iou_thres and best_gi >= 0:
            matched.append((int(pi), int(best_gi), float(best_iou)))
            matched_gt.add(best_gi)
        else:
            unmatched_preds.append(int(pi))

    unmatched_gt = [i for i in range(len(gt_boxes)) if i not in matched_gt]
    return matched, unmatched_preds, unmatched_gt


def list_images(d):
    return sorted([f for f in Path(d).iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS])
