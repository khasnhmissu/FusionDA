"""Standalone validator for Faster R-CNN R-50-FPN checkpoints.

Why standalone (and not Ultralytics YOLO.val)
=============================================
torchvision Faster R-CNN cannot be loaded into Ultralytics YOLO().  Its head
and prediction format are different, so model.val() would crash before reading
the val set.

This module:
  1. Builds the architecture via fasterrcnn.model.build_fasterrcnn.
  2. Loads a FusionDA-style state-dict checkpoint.
  3. Runs the model in eval mode, collecting COCO-format predictions.
  4. Delegates to eval_v5m.run_coco_size_eval for GT-building and COCO eval.
     That function handles zero-detection accounting, +1 category-id shift
     detection, and GT-category filtering — nothing is re-implemented here.

Output format is identical to yolo26eval.py and eval_v5m.py so chapter rows
can be filled from the same console output.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from PIL import Image as PILImage

from .adapter import torchvision_predictions_to_pseudo_targets  # noqa: F401
from .model import build_fasterrcnn


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _find_val_images_dir(data_dir: Path) -> Optional[Path]:
    """Locate images/ supporting three layouts:
    1. Direct:      data_dir/images  + data_dir/labels
    2. Nested val:  data_dir/.../val/images
    3. Double-wrap: data_dir/<name>/images
    """
    data_path = Path(data_dir)
    direct = data_path / "images"
    if direct.is_dir() and (data_path / "labels").is_dir():
        return direct
    nested_val = next((p for p in data_path.rglob("val/images") if p.is_dir()), None)
    if nested_val:
        return nested_val
    for p in sorted(data_path.rglob("images")):
        if p.is_dir() and (p.parent / "labels").is_dir():
            return p
    return None


def _load_image_as_tensor(img_path: Path, device: torch.device) -> torch.Tensor:
    """Load image as [3, H, W] float in [0, 1].

    Raw RGB in [0, 1] is required — torchvision's GeneralizedRCNNTransform
    applies ImageNet normalisation internally.
    """
    with PILImage.open(img_path) as im:
        im = im.convert("RGB")
    arr = torch.frombuffer(im.tobytes(), dtype=torch.uint8).view(
        im.size[1], im.size[0], 3
    ).clone()
    return (arr.permute(2, 0, 1).float() / 255.0).to(device)


def _ckpt_state_dict(weights_path: str) -> dict:
    """Extract model state-dict from a FusionDA-style checkpoint.

    Accepted shapes:
        torch.save({"model": state_dict, ...})   ← train_fasterrcnn.py
        torch.save(state_dict)                   ← raw save
    """
    obj = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
        raise ValueError(
            f"Could not find a state_dict under the loaded checkpoint. "
            f"Keys: {list(obj.keys())[:8]}"
        )
    raise ValueError(f"Loaded object is not a dict: {type(obj)}")


def load_fasterrcnn_for_eval(
    weights_path: str,
    num_classes: int = 2,
    min_size: int = 640,
    max_size: int = 640,
    device: str = "cpu",
) -> torch.nn.Module:
    """Build architecture and load checkpoint.

    No backbone pretrain at eval time — all weights are overwritten from the
    checkpoint anyway, and skipping the download is faster.
    """
    device_t = torch.device(device)
    model = build_fasterrcnn(
        num_classes=num_classes,
        min_size=min_size,
        max_size=max_size,
        pretrained_backbone=False,
    )
    sd = _ckpt_state_dict(weights_path)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[eval_fasterrcnn] WARNING: {len(missing)} missing keys (first 3): "
              f"{missing[:3]}")
    if unexpected:
        print(f"[eval_fasterrcnn] WARNING: {len(unexpected)} unexpected keys (first 3): "
              f"{unexpected[:3]}")
    model = model.to(device_t).eval()
    return model


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    img_dir: Path,
    device: torch.device,
    score_thresh: float = 0.001,
) -> list[dict]:
    """Forward every image in img_dir and collect COCO-format prediction dicts.

    score_thresh=0.001 mirrors Ultralytics' conf=0.001 validation default —
    keeping a low floor preserves the full PR curve tail for faithful AP.

    Returns predictions compatible with Ultralytics save_json=True output:
    same key names, same xywh bbox layout, same +1-shifted category_id.
    This makes it a drop-in input for eval_v5m.run_coco_size_eval.
    """
    img_files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in _IMG_EXTS)
    if not img_files:
        print(f"[eval_fasterrcnn] No images found under {img_dir}.")
        return []

    predictions: list[dict] = []
    n_total = len(img_files)
    log_every = max(1, n_total // 20)
    for i, img_path in enumerate(img_files):
        img = _load_image_as_tensor(img_path, device)
        out = model([img])[0]
        # image_id: int if numeric stem, else string stem — matches Ultralytics
        # save_json convention so eval_v5m.run_coco_size_eval can map it.
        stem = img_path.stem
        image_id = int(stem) if stem.isnumeric() else stem
        boxes = out["boxes"].cpu()
        labels = out["labels"].cpu()
        scores = out["scores"].cpu()
        for b, lbl, s in zip(boxes, labels, scores):
            score_f = float(s.item())
            if score_f < score_thresh:
                continue
            x1, y1, x2, y2 = b.tolist()
            predictions.append({
                "image_id":    image_id,
                "category_id": int(lbl.item()),   # 1-indexed (torchvision)
                "bbox":        [x1, y1, x2 - x1, y2 - y1],
                "score":       score_f,
            })
        if (i + 1) % log_every == 0 or (i + 1) == n_total:
            print(f"[eval_fasterrcnn] {i + 1}/{n_total} images, "
                  f"cum predictions={len(predictions)}")
    return predictions


def evaluate(
    weights_path: str,
    data_dir: str,
    dataset_name: str,
    *,
    num_classes: int = 2,
    names: Optional[dict] = None,
    min_size: int = 640,
    max_size: int = 640,
    device: str = "cpu",
    score_thresh: float = 0.001,
    save_json_dir: Optional[Path] = None,
) -> dict:
    """End-to-end evaluation: load → infer → COCO eval (size split included).

    Returns {"map50", "map50_95", "ap_small", "ap_medium", "ap_large"}.
    """
    print("=" * 60)
    print(f"Đánh giá {dataset_name} trên '{data_dir}'".center(60))
    print("=" * 60)

    device_t = torch.device(
        device if (not device.startswith("cuda") or torch.cuda.is_available()) else "cpu"
    )
    if device_t.type != device.split(":")[0]:
        print(f"[eval_fasterrcnn] WARNING: requested {device}, using {device_t}")

    model = load_fasterrcnn_for_eval(
        weights_path,
        num_classes=num_classes,
        min_size=min_size,
        max_size=max_size,
        device=str(device_t),
    )

    img_dir = _find_val_images_dir(Path(data_dir))
    if img_dir is None:
        print(f"[eval_fasterrcnn] ERROR: could not locate images/ under {data_dir}")
        return {}

    predictions = run_inference(model, img_dir, device_t, score_thresh=score_thresh)

    if not predictions:
        print("[eval_fasterrcnn] No predictions produced — bỏ qua COCO eval.")
        return {}

    save_dir = Path(save_json_dir) if save_json_dir else Path(data_dir) / "_fasterrcnn_eval"
    save_dir.mkdir(parents=True, exist_ok=True)
    pred_path = save_dir / "predictions.json"
    pred_path.write_text(json.dumps(predictions))
    print(f"[eval_fasterrcnn] Saved {len(predictions)} predictions → {pred_path}")

    if names is None:
        names = {0: "person", 1: "car"} if num_classes == 2 else \
                {i: f"class_{i}" for i in range(num_classes)}

    # Reuse eval_v5m's GT-builder + COCO summariser (zero-detection accounting,
    # +1 shift detection, GT-cat filtering) — nothing to re-implement.
    # Late import to avoid pulling in pycocotools at module-import time.
    import sys as _sys
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in _sys.path:
        _sys.path.insert(0, str(repo_root))
    from eval_v5m import run_coco_size_eval

    run_coco_size_eval(save_dir, Path(data_dir), names)
    print("=" * 60)

    return _extract_summary_stats(save_dir, Path(data_dir), names)


def _extract_summary_stats(
    save_dir: Path,
    data_dir: Path,
    names: dict,
) -> dict:
    """Re-run minimal COCO eval to return metrics as a dict instead of printing."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    pred_path = save_dir / "predictions.json"
    dt_list = json.loads(pred_path.read_text())
    if not dt_list:
        return {}

    img_dir = _find_val_images_dir(data_dir)
    if img_dir is None:
        return {}
    label_dir = img_dir.parent / "labels"

    img_files = [p for p in img_dir.iterdir() if p.suffix.lower() in _IMG_EXTS]
    ul_id_to_filepath = {
        (int(p.stem) if p.stem.isnumeric() else p.stem): p
        for p in img_files
    }

    images, annotations = [], []
    ann_id = 1
    ul_id_to_safe_id = {}
    next_safe_id = 1
    for ul_id in sorted(ul_id_to_filepath.keys(), key=lambda x: str(x)):
        safe_id = next_safe_id
        next_safe_id += 1
        ul_id_to_safe_id[ul_id] = safe_id
        with PILImage.open(ul_id_to_filepath[ul_id]) as im:
            W, H = im.size
        images.append({"id": safe_id, "file_name": ul_id_to_filepath[ul_id].name,
                       "width": W, "height": H})
        lbl_path = label_dir / (ul_id_to_filepath[ul_id].stem + ".txt")
        if lbl_path.exists():
            for line in lbl_path.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                abs_w, abs_h = bw * W, bh * H
                annotations.append({
                    "id": ann_id, "image_id": safe_id, "category_id": cls_id,
                    "bbox": [cx * W - abs_w / 2, cy * H - abs_h / 2, abs_w, abs_h],
                    "area": abs_w * abs_h, "iscrowd": 0,
                })
                ann_id += 1

    gt_cat_ids_present = {int(ann["category_id"]) for ann in annotations}
    dt_cat_ids = {d.get("category_id", -1) for d in dt_list}
    shift_cat = (
        bool(dt_cat_ids) and bool(gt_cat_ids_present)
        and min(dt_cat_ids) >= 1 and 0 in gt_cat_ids_present
    )

    new_dt_list = []
    for d in dt_list:
        ul_id = d["image_id"]
        if ul_id not in ul_id_to_safe_id:
            continue
        cat_id = d["category_id"] - 1 if shift_cat else d["category_id"]
        if cat_id not in gt_cat_ids_present:
            continue
        d2 = dict(d)
        d2["image_id"] = ul_id_to_safe_id[ul_id]
        d2["category_id"] = cat_id
        new_dt_list.append(d2)

    coco_categories = [
        {"id": cid, "name": names.get(cid, f"class_{cid}")}
        for cid in sorted(gt_cat_ids_present)
    ]
    gt_dict = {
        "images": images, "annotations": annotations,
        "categories": coco_categories,
    }
    gt_path = save_dir / "_coco_gt_for_summary.json"
    gt_path.write_text(json.dumps(gt_dict))

    coco_gt = COCO(str(gt_path))
    coco_dt = coco_gt.loadRes(new_dt_list)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()

    # Skip the pretty printout — eval_v5m already did that above.
    s = coco_eval.eval["precision"]   # [T, R, K, A, M]

    def _mean_filtered(arr):
        flat = arr[arr > -1]
        return float(flat.mean()) if flat.size else float("nan")

    summary = {
        "map50_95":  _mean_filtered(s[:, :, :, 0, -1]),
        "map50":     _mean_filtered(s[0, :, :, 0, -1]),
        "ap_small":  _mean_filtered(s[:, :, :, 1, -1]),
        "ap_medium": _mean_filtered(s[:, :, :, 2, -1]),
        "ap_large":  _mean_filtered(s[:, :, :, 3, -1]),
    }
    gt_path.unlink(missing_ok=True)
    return summary


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Evaluate a Faster R-CNN R-50-FPN checkpoint.")
    p.add_argument("--weights",     required=True)
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--source-dir",  default="source_test")
    p.add_argument("--target-dir",  default="target_test")
    p.add_argument("--real-fog-dir", default=None)
    p.add_argument("--imgsz",       type=int, default=640)
    p.add_argument("--device",      default="cuda:0")
    args = p.parse_args()

    targets = [(args.source_dir, "Source Test"),
               (args.target_dir, "Target Test")]
    if args.real_fog_dir:
        targets.append((args.real_fog_dir, "Real Fog Test"))

    for d, name in targets:
        if Path(d).exists():
            evaluate(
                args.weights, d, name,
                num_classes=args.num_classes,
                min_size=args.imgsz, max_size=args.imgsz,
                device=args.device,
            )
        else:
            print(f"[!] Không tìm thấy thư mục '{d}'")
