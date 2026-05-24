"""eval_v5m.py — universal evaluation for FusionDA checkpoints.
Usage
=====
Auto-detect base weights from the checkpoint's head-layer index:

    python eval_v5m.py \\
        --weights    runs/fda/exp/weights/best.pt \\
        --target-dir datasets/target_real/target_real/val

Manual override (still supported):

    python eval_v5m.py \\
        --weights      runs/fda/exp/weights/best.pt \\
        --base-weights yolov5mu.pt \\
        --target-dir   datasets/target_real/target_real/val

The ``--source-dir`` / ``--target-dir`` / ``--real-fog-dir`` flags behave
identically to ``yolo26eval.py``.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
from pathlib import Path

import torch
import yaml
from PIL import Image as PILImage
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO


def _find_images_dir(data_dir):
    """Locate the images subfolder. Supports three layouts:
       1. Direct:       data_dir/images  + data_dir/labels
       2. Nested val:   data_dir/.../val/images
       3. Double-wrap:  data_dir/<name>/images (zip giải nén thêm 1 cấp)
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


def create_yaml(data_dir, model_names):
    yaml_path = Path(data_dir) / "eval_data.yaml"
    data_path = Path(data_dir).absolute()
    img_dir = _find_images_dir(data_path)
    if img_dir is None:
        val_dir = "val/images"
    else:
        val_dir = str(img_dir.relative_to(data_path)).replace("\\", "/")
    dataset_yaml = {
        "path": str(data_path),
        "train": val_dir, "val": val_dir, "test": val_dir,
        "nc": len(model_names), "names": model_names,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, sort_keys=False)
    return str(yaml_path)


# Pattern matches ``model.<idx>.cv3.0.2.bias`` and the YOLO26-E2E variant
# ``model.<idx>.one2one_cv3.0.2.bias``.  Both carry the per-class bias tensor
# whose first dimension equals nc.
_HEAD_BIAS_PATTERN = re.compile(
    r"^model\.(\d+)\.(?:one2one_)?cv3\.0\.2\.bias$"
)


def _detect_head_layer_idx(sd):
    """Scan a state_dict for the Detect-head layer index and nc.

    Returns
    -------
    (layer_idx, nc) or (None, None) if no recognisable head bias is found.
    """
    candidates = []
    for key, tensor in sd.items():
        m = _HEAD_BIAS_PATTERN.match(key)
        if m is None:
            continue
        candidates.append((int(m.group(1)), tensor.shape[0], key))
    if not candidates:
        return None, None
    # Multiple candidates can exist (E2E head has both cv3 and one2one_cv3 at
    # the same layer); they MUST agree on nc.  Pick the highest layer index
    # and assert agreement to surface any state_dict inconsistencies early.
    candidates.sort(key=lambda x: x[0])
    head_layer = candidates[-1][0]
    head_nc = candidates[-1][1]
    for idx, nc, key in candidates:
        if idx == head_layer and nc != head_nc:
            raise RuntimeError(
                f"Inconsistent head layer: {key} has nc={nc}, "
                f"expected {head_nc} (other keys at layer {head_layer})."
            )
    return head_layer, head_nc


def _detect_nc_from_state_dict(sd):
    """nc-only accessor — preserves the original yolo26eval.py signature."""
    _, nc = _detect_head_layer_idx(sd)
    return nc


# Map detected head-layer-index → matching pretrained base weights file.
# YOLO26-s places its Detect head at layer 23; Ultralytics' yolov5mu (the
# anchor-free YOLOv5m port used by FusionDA) places it at layer 24.
_HEAD_IDX_TO_BASE = {
    23: "yolo26s.pt",
    24: "yolov5mu.pt",
}


def _auto_select_base_weights(head_idx):
    return _HEAD_IDX_TO_BASE.get(head_idx)


def load_model_custom(weight_path, base_weights=None):
    """Load a FusionDA checkpoint into an Ultralytics YOLO wrapper.

    base_weights is optional — auto-selected from the checkpoint's head-layer
    index (layer 23 → yolo26s.pt, layer 24 → yolov5mu.pt).
    """
    print(f"Loading weights từ {weight_path}...")
    try:
        ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)

        # Case 1: checkpoint contains state_dict (from train.py)
        if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
            sd = ckpt["model"]
            head_idx, ckpt_nc = _detect_head_layer_idx(sd)
            print(f"[*] Checkpoint state_dict: head layer={head_idx}, nc={ckpt_nc}")

            # Resolve base_weights: explicit arg wins; else auto-select.
            if base_weights is None:
                auto_base = _auto_select_base_weights(head_idx)
                if auto_base is None:
                    raise RuntimeError(
                        f"Cannot auto-detect base weights for head at layer "
                        f"{head_idx}. Pass --base-weights explicitly. "
                        f"Recognised heads: {list(_HEAD_IDX_TO_BASE.keys())}"
                    )
                base_weights = auto_base
                print(f"[*] Auto-selected base-weights: {base_weights} "
                      f"(head idx={head_idx})")

            # Detect nc on the base architecture for the rebuild decision.
            base_model = None
            base_nc = None
            try:
                base_model = YOLO(base_weights)
                base_nc = base_model.model.model[-1].nc
            except Exception:
                pass

            if ckpt_nc is not None and base_nc is not None and ckpt_nc != base_nc:
                # nc khác nhau → rebuild model với đúng nc
                print(f"[!] nc mismatch: checkpoint={ckpt_nc}, base={base_nc}")
                print(f"[*] Rebuild model với nc={ckpt_nc} từ base architecture...")

                from ultralytics.nn.tasks import DetectionModel
                base_cfg = base_model.model.yaml
                custom_cfg = copy.deepcopy(base_cfg)
                custom_cfg["nc"] = ckpt_nc

                new_model = DetectionModel(custom_cfg, nc=ckpt_nc)
                new_model.load_state_dict(sd)
                # Sanity: rebuild produced the head we expected
                rebuilt_idx, rebuilt_nc = _detect_head_layer_idx(new_model.state_dict())
                assert rebuilt_idx == head_idx and rebuilt_nc == ckpt_nc, (
                    f"Rebuild head mismatch: expected (layer={head_idx}, "
                    f"nc={ckpt_nc}), got (layer={rebuilt_idx}, nc={rebuilt_nc})"
                )
                print(f"[✓] Load state_dict thành công! "
                      f"(layer={head_idx}, nc={ckpt_nc})")

                base_model.model = new_model
                base_model.model.args = (
                    base_model.model.args
                    if hasattr(base_model.model, "args") else {}
                )
                # Propagate class names if the checkpoint stored them.
                # NB: ``YOLO.names`` is a read-only property that proxies to
                # ``YOLO.model.names``; assigning directly to the wrapper
                # raises AttributeError.  Set on the inner DetectionModel.
                ckpt_names = ckpt.get("names")
                if isinstance(ckpt_names, dict):
                    base_model.model.names = {int(k): v for k, v in ckpt_names.items()}
                elif isinstance(ckpt_names, (list, tuple)):
                    base_model.model.names = {i: n for i, n in enumerate(ckpt_names)}
                return base_model

            # nc match (or undetectable on either side) → load straight onto base.
            print(f"[*] Mở base-weights: {base_weights}")
            model = YOLO(base_weights) if base_model is None else base_model
            try:
                model.model.load_state_dict(sd)
                print("[✓] Load state_dict thành công!")
            except Exception as e:
                print(f"[!] strict=True thất bại: {e}")
                model.model.load_state_dict(sd, strict=False)
                print("[✓] Load state_dict (strict=False) thành công!")
            ckpt_names = ckpt.get("names")
            if isinstance(ckpt_names, dict):
                model.model.names = {int(k): v for k, v in ckpt_names.items()}
            elif isinstance(ckpt_names, (list, tuple)):
                model.model.names = {i: n for i, n in enumerate(ckpt_names)}
            return model

        # Case 2: full model object (YOLO save format)
        return YOLO(weight_path)
    except Exception as e:
        print(f"Lỗi khi mở checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return YOLO(weight_path)


def run_coco_size_eval(save_dir, data_dir, model_names):
    """Read predictions.json and run pycocotools COCOeval for AP_small / AP_medium / AP_large."""
    save_dir = Path(save_dir)
    data_dir = Path(data_dir)
    dt_path = save_dir / "predictions.json"
    if not dt_path.exists():
        print(f"[!] Không tìm thấy {dt_path} — bỏ qua COCO size eval.")
        return

    dt_list = json.loads(dt_path.read_text())
    if not dt_list:
        print("[!] predictions.json rỗng — bỏ qua.")
        return
    print(f"[COCO] Đọc {len(dt_list)} detections từ {dt_path}")

    img_dir = _find_images_dir(data_dir)
    if img_dir is None:
        print("[!] Không tìm thấy folder images — bỏ qua.")
        return
    label_dir = img_dir.parent / "labels"

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    img_files = [p for p in img_dir.iterdir() if p.suffix.lower() in img_exts]

    ul_id_to_filepath = {
        (int(p.stem) if p.stem.isnumeric() else p.stem): p
        for p in img_files
    }

    used_dt_ids = {d["image_id"] for d in dt_list}

    # ⚠ CRITICAL: register every val image in the COCO GT — including images
    # where the model produced zero detections.  Otherwise pycocotools silently
    # drops their GT objects (they exist as FNs but are never counted).
    images, annotations = [], []
    ann_id = 1
    ul_id_to_safe_id = {}
    next_safe_id = 1

    all_val_ids = sorted(ul_id_to_filepath.keys(), key=lambda x: str(x))

    for ul_id in all_val_ids:
        safe_id = next_safe_id
        next_safe_id += 1
        ul_id_to_safe_id[ul_id] = safe_id

        img_path = ul_id_to_filepath[ul_id]
        with PILImage.open(img_path) as im:
            W, H = im.size
        images.append({"id": safe_id, "file_name": img_path.name, "width": W, "height": H})

        lbl_path = label_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            for line in lbl_path.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                abs_w, abs_h = bw * W, bh * H
                abs_x = cx * W - abs_w / 2
                abs_y = cy * H - abs_h / 2
                annotations.append({
                    "id": ann_id,
                    "image_id": safe_id,
                    "category_id": cls_id,
                    "bbox": [abs_x, abs_y, abs_w, abs_h],
                    "area": abs_w * abs_h,
                    "iscrowd": 0,
                })
                ann_id += 1

    n_total = len(all_val_ids)
    n_with_preds = len(used_dt_ids & set(all_val_ids))
    n_zero_pred = n_total - n_with_preds
    print(f"[COCO] Val images: {n_total} total, {n_with_preds} with predictions, "
          f"{n_zero_pred} zero-detection")
    if n_zero_pred > 0:
        print(f"[COCO] {n_zero_pred} zero-detection image(s) contribute only FNs — "
              f"their GT objects (often small) are now correctly counted.")

    # Ultralytics 8.4 save_json uses class_map = range(1, nc+1), so DT cat_ids
    # are 1-indexed while GT YOLO labels are 0-indexed.
    # Detect shift via: min(dt_cat_ids) >= 1 AND 0 in gt_cat_ids_present.
    # Driving from actual GT cats (not model.names) handles the 80-head+2-class
    # case where dt_cat_ids ⊆ model.names.keys() passes by coincidence.
    gt_cat_ids_present = {int(ann["category_id"]) for ann in annotations}
    dt_cat_ids = {d.get("category_id", -1) for d in dt_list}

    shift_cat = (
        bool(dt_cat_ids) and bool(gt_cat_ids_present)
        and min(dt_cat_ids) >= 1 and 0 in gt_cat_ids_present
    )
    if shift_cat:
        print(f"[COCO] Auto-detected category_id +1 shift "
              f"(Ultralytics class_map=range(1,N+1) convention). "
              f"dt cat range=[{min(dt_cat_ids)}, {max(dt_cat_ids)}], "
              f"gt cats present={sorted(gt_cat_ids_present)}.")

    # COCO GT must list only the categories that actually appear in GT.
    # Listing all of model.names (which can be COCO-80) registers many
    # empty-GT cats and adds noise to the per-cat AP averaging.
    def _name_for_cat(cid):
        n = None
        if isinstance(model_names, dict):
            # dict keys may be int or str depending on yaml load path.
            n = model_names.get(cid, model_names.get(str(cid)))
        elif isinstance(model_names, (list, tuple)) and 0 <= cid < len(model_names):
            n = model_names[cid]
        return n if n is not None else f"class_{cid}"

    coco_categories = [
        {"id": cid, "name": _name_for_cat(cid)}
        for cid in sorted(gt_cat_ids_present)
    ]
    print(f"[COCO] Registering {len(coco_categories)} GT categories: "
          f"{[(c['id'], c['name']) for c in coco_categories]}")

    # Filter DT: image-id missing → drop; cat-id not in GT (post-shift) → drop.
    # The latter is what discards the noise predictions on dormant
    # head slots (classes 2..79 in an 80-output head trained 2-class).
    new_dt_list = []
    n_drop_imgmiss = 0
    n_drop_cat_oor = 0
    for d in dt_list:
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
        print(f"[COCO] Dropped {n_drop_cat_oor}/{len(dt_list)} DT predictions "
              f"on cats not in GT (dormant head slots in an "
              f"{len(model_names) if hasattr(model_names,'__len__') else '?'}-class head "
              f"trained on {len(gt_cat_ids_present)}-class data).")
    if n_drop_imgmiss > 0:
        print(f"[COCO] Dropped {n_drop_imgmiss} DT predictions whose image_id "
              f"is not present in the val image set.")

    coco_gt_dict = {
        "images": images,
        "annotations": annotations,
        "categories": coco_categories,
    }

    gt_path = data_dir / "_coco_gt_tmp.json"
    gt_path.write_text(json.dumps(coco_gt_dict))

    coco_gt = COCO(str(gt_path))
    coco_dt = coco_gt.loadRes(new_dt_list)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

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

    gt_path.unlink(missing_ok=True)


def evaluate(weight_path, data_dir, dataset_name, base_weights=None):
    print("=" * 60)
    print(f"Đánh giá {dataset_name} trên '{data_dir}'".center(60))
    print("=" * 60)

    model = load_model_custom(weight_path, base_weights)
    model_names = model.names
    yaml_path = create_yaml(data_dir, model_names)
    print(f"Config: {yaml_path}")

    try:
        metrics = model.val(
            data=yaml_path, split="val", batch=4,
            verbose=True,
            save_json=True,
        )
        print(f"\n{'─' * 40}")
        print(f" Ultralytics – {dataset_name}")
        print("─" * 40)
        print(f" mAP@50    : {metrics.box.map50:.4f}")
        print(f" mAP@50-95 : {metrics.box.map:.4f}")
        print("─" * 40)
    except Exception as e:
        print(f"[!] Ultralytics val thất bại: {e}")
        return

    print("\n[COCO Size Metrics] Tính AP_small / AP_medium / AP_large ...")
    run_coco_size_eval(Path(metrics.save_dir), Path(data_dir), model_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",      default="best.pt",
                        help="Path to FusionDA checkpoint (state_dict format).")
    parser.add_argument("--base-weights", default=None,
                        help="Pretrained base used to instantiate the architecture. "
                             "If omitted, auto-selected from the checkpoint's head "
                             "layer index (23 → yolo26s.pt, 24 → yolov5mu.pt).")
    parser.add_argument("--source-dir",   default=None)
    parser.add_argument("--target-dir",   default=None)
    parser.add_argument(
        "--real-fog-dirs", nargs="+", default=[], metavar="DIR",
        help="Một hoặc nhiều folder real-fog, mỗi folder chứa images/ và labels/ "
             "(chuẩn YOLO). Ví dụ: --real-fog-dirs RTTS_yolo DAWN FoggyDriving",
    )
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        print(f"[LỖI] Không tìm thấy: '{args.weights}'")
        exit(1)

    eval_targets = []
    if args.source_dir:
        eval_targets.append((args.source_dir, "Source Test"))
    if args.target_dir:
        eval_targets.append((args.target_dir, "Target Test"))
    for fog_dir in args.real_fog_dirs:
        eval_targets.append((fog_dir, f"Real Fog – {Path(fog_dir).name}"))

    for dir_path, name in eval_targets:
        if os.path.exists(dir_path):
            evaluate(args.weights, dir_path, name, args.base_weights)
        else:
            print(f"[!] Không tìm thấy thư mục '{dir_path}'")
