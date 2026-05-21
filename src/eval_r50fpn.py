"""eval_r50fpn.py — COCO-standard evaluation for Faster R-CNN R-50-FPN checkpoints.
============
For each test directory (source / target / real-fog), in order:

    1. Load the FusionDA checkpoint with ``fasterrcnn.eval.load_fasterrcnn_for_eval``.
    2. Run inference, save Ultralytics-format ``predictions.json``.
    3. Build a pycocotools COCO GT + DT pair using the same logic
       ``eval_v5m.run_coco_size_eval`` already uses (zero-detection FN
       accounting, +1 cat_id shift detection, GT-only category filtering).
    4. Run ``pycocotools.COCOeval``

==================================================================

Usage
=====
    python eval_r50fpn.py \\
        --weights      runs/ablation/12_fusionda_fasterrcnn/weights/best.pt \\
        --num-classes  2 \\
        --imgsz        640 \\
        --device       cuda:0 \\
        --source-dir   datasets/source_real/source_real/val \\
        --target-dir   datasets/target_real/target_real/val \\
        --real-fog-dir datasets/RTTS_yolo
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import torch
from PIL import Image as PILImage
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Reused from fasterrcnn.eval — no duplication of model loading / inference logic.
from fasterrcnn.eval import (
    _find_val_images_dir,
    load_fasterrcnn_for_eval,
    run_inference,
)


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _build_coco_objects(
    predictions: list[dict],
    data_dir: Path,
    names: dict,
):
    """Build pycocotools COCO + DT structures with zero-detection FN accounting.

    Unlike run_coco_size_eval, exposes pycocotools objects so the caller
    can extract per-class AP.
    Returns (coco_gt, coco_dt, gt_cat_ids_present, n_total, n_zero_pred).
    """
    img_dir = _find_val_images_dir(data_dir)
    if img_dir is None:
        return None, None, set(), 0, 0
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

    # CRITICAL: register every val image, including zero-detection ones, so
    # their GT objects (often small) are correctly counted as FNs.  This is
    # the same correctness guarantee that yolo26eval / eval_v5m already
    # provide; we re-implement here to keep the COCO objects accessible.
    all_val_ids = sorted(ul_id_to_filepath.keys(), key=lambda x: str(x))
    for ul_id in all_val_ids:
        safe_id = next_safe_id
        next_safe_id += 1
        ul_id_to_safe_id[ul_id] = safe_id

        img_path = ul_id_to_filepath[ul_id]
        with PILImage.open(img_path) as im:
            W, H = im.size
        images.append({
            "id": safe_id, "file_name": img_path.name,
            "width": W, "height": H,
        })

        lbl_path = label_dir / (img_path.stem + ".txt")
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
                    "bbox": [cx * W - abs_w / 2, cy * H - abs_h / 2,
                             abs_w, abs_h],
                    "area": abs_w * abs_h, "iscrowd": 0,
                })
                ann_id += 1

    used_dt_ids = {d["image_id"] for d in predictions}
    n_total = len(all_val_ids)
    n_with_preds = len(used_dt_ids & set(all_val_ids))
    n_zero_pred = n_total - n_with_preds
    print(f"[COCO] Val images: {n_total} total, "
          f"{n_with_preds} with predictions, "
          f"{n_zero_pred} zero-detection")
    if n_zero_pred > 0:
        print(f"[COCO] {n_zero_pred} zero-detection image(s) contribute "
              f"only FNs — their GT objects (often small) are now correctly "
              f"counted.")

    # Robust shift detection — same heuristic eval_v5m.py uses.
    # torchvision Faster R-CNN's eval-mode predictions are 1-indexed
    # (label 0 = background); GT YOLO labels are 0-indexed.  Detect via
    # ``min(dt) >= 1 and 0 in gt``.
    gt_cat_ids_present = {int(ann["category_id"]) for ann in annotations}
    dt_cat_ids = {d.get("category_id", -1) for d in predictions}
    shift_cat = (
        bool(dt_cat_ids) and bool(gt_cat_ids_present)
        and min(dt_cat_ids) >= 1 and 0 in gt_cat_ids_present
    )
    if shift_cat:
        print(f"[COCO] Auto-detected category_id +1 shift "
              f"(torchvision Faster R-CNN labels 1-indexed, GT 0-indexed). "
              f"dt cat range=[{min(dt_cat_ids)}, {max(dt_cat_ids)}], "
              f"gt cats present={sorted(gt_cat_ids_present)}.")

    # Register only categories present in GT — avoids the 80-cat mAP-collapse
    # bug eval_v5m.py was patched against.
    def _name_for(cid):
        n = None
        if isinstance(names, dict):
            n = names.get(cid, names.get(str(cid)))
        elif isinstance(names, (list, tuple)) and 0 <= cid < len(names):
            n = names[cid]
        return n if n is not None else f"class_{cid}"

    coco_categories = [
        {"id": cid, "name": _name_for(cid)}
        for cid in sorted(gt_cat_ids_present)
    ]
    print(f"[COCO] Registering {len(coco_categories)} GT categories: "
          f"{[(c['id'], c['name']) for c in coco_categories]}")

    # Filter DT — drop predictions on images we don't have, drop predictions
    # on categories not in GT (these come from dormant head slots if a
    # multi-class checkpoint was loaded onto a 2-class dataset; for a true
    # 2-class FasterRCNN this is a no-op).
    new_dt_list = []
    n_drop_imgmiss = n_drop_cat_oor = 0
    for d in predictions:
        ul_id = d["image_id"]
        if ul_id not in ul_id_to_safe_id:
            n_drop_imgmiss += 1
            continue
        cat_id = d["category_id"] - 1 if shift_cat else d["category_id"]
        if cat_id not in gt_cat_ids_present:
            n_drop_cat_oor += 1
            continue
        d2 = dict(d)
        d2["image_id"] = ul_id_to_safe_id[ul_id]
        d2["category_id"] = cat_id
        new_dt_list.append(d2)

    if n_drop_cat_oor > 0:
        print(f"[COCO] Dropped {n_drop_cat_oor}/{len(predictions)} DT "
              f"predictions on cats not present in GT.")
    if n_drop_imgmiss > 0:
        print(f"[COCO] Dropped {n_drop_imgmiss} DT predictions whose "
              f"image_id is not in the val image set.")

    coco_gt_dict = {
        "images": images,
        "annotations": annotations,
        "categories": coco_categories,
    }

    # pycocotools requires file paths, not in-memory dicts — write to a
    # short-lived tempfile and unlink right after construction.
    fd, tmp_gt_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    Path(tmp_gt_path).write_text(json.dumps(coco_gt_dict))
    try:
        coco_gt = COCO(tmp_gt_path)
        coco_dt = coco_gt.loadRes(new_dt_list) if new_dt_list else None
    finally:
        try:
            Path(tmp_gt_path).unlink()
        except OSError:
            pass

    return coco_gt, coco_dt, gt_cat_ids_present, n_total, n_zero_pred


def _per_class_aps(coco_eval: COCOeval, gt_cat_ids: set):
    """Extract per-class AP@0.5 and AP@0.5:0.95 from the COCOeval result.

    ``coco_eval.eval['precision']`` has shape ``[T, R, K, A, M]``:
        T=10 IoU thresholds (0.50, 0.55, ..., 0.95)
        R=101 recall thresholds
        K=number of registered categories (sorted by id)
        A=4 area ranges (0=all, 1=small, 2=medium, 3=large)
        M=3 max-detection thresholds (0=1, 1=10, 2=100)

    Per-class numbers use ``A=0`` (all areas) and ``M=2`` (maxDets=100),
    which is the COCO mAP convention.
    """
    p = coco_eval.eval["precision"]
    cat_ids_sorted = sorted(gt_cat_ids)
    out = {}
    for k, cat_id in enumerate(cat_ids_sorted):
        a50 = p[0, :, k, 0, 2]      # T=0 → IoU=0.5
        a5095 = p[:, :, k, 0, 2]    # all 10 IoU thresholds
        ap50 = float(a50[a50 > -1].mean()) if (a50 > -1).any() else 0.0
        ap5095 = float(a5095[a5095 > -1].mean()) if (a5095 > -1).any() else 0.0
        out[cat_id] = (ap50, ap5095)
    return out


def _print_per_class_table(coco_gt: COCO, coco_eval: COCOeval, gt_cat_ids: set):
    """Print Ultralytics-style per-class table.

    Format::

                     Class    Images  Instances    mAP50  mAP50-95
                       all       N         M       X.XXX    X.XXX
                    person      N1        M1       X.XXX    X.XXX
                       car      N2        M2       X.XXX    X.XXX
    """
    aps = _per_class_aps(coco_eval, gt_cat_ids)
    cat_ids_sorted = sorted(gt_cat_ids)

    print(f"\n  {'Class':>15}  {'Images':>6}  {'Instances':>9}  "
          f"{'mAP50':>7}  {'mAP50-95':>9}")

    all_n_images = len(coco_gt.getImgIds())
    all_n_inst = len(coco_gt.getAnnIds())
    map50_avg = sum(v[0] for v in aps.values()) / max(1, len(aps))
    map5095_avg = sum(v[1] for v in aps.values()) / max(1, len(aps))
    print(f"  {'all':>15}  {all_n_images:>6}  {all_n_inst:>9}  "
          f"{map50_avg:>7.4f}  {map5095_avg:>9.4f}")

    for cat_id in cat_ids_sorted:
        cat = coco_gt.loadCats(cat_id)[0]
        name = cat["name"]
        n_images = len(coco_gt.getImgIds(catIds=[cat_id]))
        n_inst = len(coco_gt.getAnnIds(catIds=[cat_id]))
        ap50, ap5095 = aps[cat_id]
        print(f"  {name:>15}  {n_images:>6}  {n_inst:>9}  "
              f"{ap50:>7.4f}  {ap5095:>9.4f}")

    return map50_avg, map5095_avg


def _print_compact_summary(dataset_name: str, map50: float, map5095: float):
    """Mirrors eval_v5m.py's compact summary block."""
    print(f"\n{'─' * 40}")
    print(f" Faster R-CNN R-50-FPN – {dataset_name}")
    print("─" * 40)
    print(f" mAP@50    : {map50:.4f}")
    print(f" mAP@50-95 : {map5095:.4f}")
    print("─" * 40)


def _print_coco_size_table(coco_eval: COCOeval):
    """Match eval_v5m.run_coco_size_eval's final pretty-printed block,
    so output diff between R-50-FPN row and YOLO row is purely numeric."""
    stats = coco_eval.stats
    rows = [
        ("AP",        "0.50:0.95", stats[0]),
        ("AP@50",     "0.50",      stats[1]),
        ("AP@75",     "0.75",      stats[2]),
        ("AP_small",  "0.50:0.95", stats[3]),
        ("AP_medium", "0.50:0.95", stats[4]),
        ("AP_large",  "0.50:0.95", stats[5]),
        ("AR@1",      "0.50:0.95", stats[6]),
        ("AR@10",     "0.50:0.95", stats[7]),
        ("AR@100",    "0.50:0.95", stats[8]),
    ]
    print("\n" + "─" * 44)
    print(f"  {'Metric':<18}  {'IoU':>10}  {'Value':>8}")
    print("─" * 44)
    for name, iou, val in rows:
        print(f"  {name:<18}  {iou:>10}  {val:>8.4f}")
    print("─" * 44 + "\n")


def evaluate(
    weights_path: str,
    data_dir: str,
    dataset_name: str,
    *,
    num_classes: int = 2,
    names: Optional[dict] = None,
    imgsz: int = 640,
    device: str = "cuda:0",
    score_thresh: float = 0.001,
    save_json_dir: Optional[Path] = None,
) -> dict:
    """Load → infer → COCO eval. Returns {map50, map50_95, ap_small, ap_medium, ap_large}."""
    print("=" * 60)
    print(f"Đánh giá {dataset_name} trên '{data_dir}'".center(60))
    print("=" * 60)

    if names is None:
        names = ({0: "person", 1: "car"} if num_classes == 2
                 else {i: f"class_{i}" for i in range(num_classes)})

    # Resolve device with same fallback as fasterrcnn.eval.evaluate.
    device_t = torch.device(
        device if (not device.startswith("cuda") or torch.cuda.is_available())
        else "cpu"
    )
    if device_t.type != device.split(":")[0]:
        print(f"[!] Requested {device}, falling back to {device_t}")

    model = load_fasterrcnn_for_eval(
        weights_path,
        num_classes=num_classes,
        min_size=imgsz, max_size=imgsz,
        device=str(device_t),
    )

    img_dir = _find_val_images_dir(Path(data_dir))
    if img_dir is None:
        print(f"[!] Không tìm thấy images/ dưới '{data_dir}'")
        return {}

    predictions = run_inference(
        model, img_dir, device_t, score_thresh=score_thresh,
    )
    if not predictions:
        print(f"[!] Không có predictions nào — bỏ qua COCO eval.")
        return {}

    save_dir = (Path(save_json_dir) if save_json_dir else
                Path(data_dir) / "_r50fpn_eval")
    save_dir.mkdir(parents=True, exist_ok=True)
    pred_path = save_dir / "predictions.json"
    pred_path.write_text(json.dumps(predictions))
    print(f"[*] Saved {len(predictions)} predictions → {pred_path}")

    coco_gt, coco_dt, gt_cat_ids, n_total, n_zero = _build_coco_objects(
        predictions, Path(data_dir), names,
    )
    if coco_gt is None or coco_dt is None:
        print(f"[!] COCO build failed — không thể eval.")
        return {}

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    map50_per_class_avg, map5095_per_class_avg = _print_per_class_table(
        coco_gt, coco_eval, gt_cat_ids,
    )

    # Use coco_eval.stats[0/1], not per-class average — can differ by ~0.001
    # due to averaging order; stats is the authoritative COCO mAP.
    _print_compact_summary(
        dataset_name,
        float(coco_eval.stats[1]),    # AP@50
        float(coco_eval.stats[0]),    # AP@50:95
    )

    _print_coco_size_table(coco_eval)

    return {
        "map50":     float(coco_eval.stats[1]),
        "map50_95":  float(coco_eval.stats[0]),
        "ap_small":  float(coco_eval.stats[3]),
        "ap_medium": float(coco_eval.stats[4]),
        "ap_large":  float(coco_eval.stats[5]),
        "per_class": _per_class_aps(coco_eval, gt_cat_ids),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="COCO-standard evaluation for Faster R-CNN R-50-FPN.",
    )
    parser.add_argument("--weights",      default="best.pt")
    parser.add_argument("--num-classes",  type=int, default=2)
    parser.add_argument("--imgsz",        type=int, default=640)
    parser.add_argument("--device",       default="cuda:0")
    parser.add_argument("--source-dir",   default=None)
    parser.add_argument("--target-dir",   default=None)
    parser.add_argument(
        "--real-fog-dirs", nargs="+", default=[], metavar="DIR",
        help="Một hoặc nhiều folder real-fog, mỗi folder chứa images/ và labels/ "
             "(chuẩn YOLO). Ví dụ: --real-fog-dirs RTTS_yolo DAWN FoggyDriving",
    )
    parser.add_argument("--score-thresh", type=float, default=0.001,
                        help="Score floor for predictions.json (matches "
                             "Ultralytics conf=0.001 default).")
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        print(f"[LỖI] Không tìm thấy: '{args.weights}'")
        sys.exit(1)

    eval_targets = []
    if args.source_dir:
        eval_targets.append((args.source_dir, "Source Test"))
    if args.target_dir:
        eval_targets.append((args.target_dir, "Target Test"))
    for fog_dir in args.real_fog_dirs:
        eval_targets.append((fog_dir, f"Real Fog – {Path(fog_dir).name}"))

    for dir_path, name in eval_targets:
        if os.path.exists(dir_path):
            evaluate(
                args.weights, dir_path, name,
                num_classes=args.num_classes,
                imgsz=args.imgsz,
                device=args.device,
                score_thresh=args.score_thresh,
            )
        else:
            print(f"[!] Không tìm thấy thư mục '{dir_path}'")
