import os
import json
import yaml
import torch
import argparse
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image as PILImage


def create_yaml(data_dir, model_names):
    yaml_path = Path(data_dir) / "eval_data.yaml"
    data_path = Path(data_dir).absolute()
    val_dir = "val/images"
    for p in Path(data_dir).rglob("val/images"):
        if p.is_dir():
            val_dir = str(p.relative_to(data_dir)).replace("\\", "/")
            break
    dataset_yaml = {
        "path": str(data_path),
        "train": val_dir, "val": val_dir, "test": val_dir,
        "nc": len(model_names), "names": model_names,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, sort_keys=False)
    return str(yaml_path)


def _detect_nc_from_state_dict(sd):
    """Detect number of classes from a state_dict by inspecting the detection head."""
    # Detection head classification layers: model.23.cv3.X.2.bias has shape [nc]
    for key in ["model.23.cv3.0.2.bias", "model.23.one2one_cv3.0.2.bias"]:
        if key in sd:
            return sd[key].shape[0]
    return None


def load_model_custom(weight_path, base_weights):
    print(f"Loading weights từ {weight_path}...")
    try:
        ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)

        # ── Case 1: checkpoint chứa state_dict (từ train.py) ──────────
        if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
            sd = ckpt["model"]
            ckpt_nc = _detect_nc_from_state_dict(sd)
            print(f"[*] Checkpoint state_dict: nc={ckpt_nc}")

            # Phát hiện nc từ base-weights
            base_model = None
            base_nc = None
            try:
                base_model = YOLO(base_weights)
                base_nc = base_model.model.model[-1].nc  # detection head nc
            except Exception:
                pass

            if ckpt_nc is not None and base_nc is not None and ckpt_nc != base_nc:
                # nc khác nhau → rebuild model với đúng nc
                print(f"[!] nc mismatch: checkpoint={ckpt_nc}, base={base_nc}")
                print(f"[*] Rebuild model với nc={ckpt_nc} từ base architecture...")

                # Lấy config YAML từ base model, override nc
                from ultralytics.nn.tasks import DetectionModel
                import copy

                # Build model with correct nc using base model's yaml
                base_cfg = base_model.model.yaml
                custom_cfg = copy.deepcopy(base_cfg)
                custom_cfg["nc"] = ckpt_nc

                new_model = DetectionModel(custom_cfg, nc=ckpt_nc)
                new_model.load_state_dict(sd)
                print(f"[✓] Load state_dict thành công! (nc={ckpt_nc})")

                # Wrap into YOLO object
                base_model.model = new_model
                base_model.model.args = base_model.model.args if hasattr(base_model.model, 'args') else {}
                return base_model
            else:
                # nc match hoặc không phát hiện được → load bình thường
                print(f"[*] Mở base-weights: {base_weights}")
                model = YOLO(base_weights) if base_model is None else base_model
                try:
                    model.model.load_state_dict(sd)
                    print("[✓] Load state_dict thành công!")
                except Exception as e:
                    print(f"[!] strict=True thất bại: {e}")
                    model.model.load_state_dict(sd, strict=False)
                    print("[✓] Load state_dict (strict=False) thành công!")
                return model

        # ── Case 2: checkpoint là full model object (YOLO save format) ──
        else:
            return YOLO(weight_path)
    except Exception as e:
        print(f"Lỗi khi mở checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return YOLO(weight_path)


def run_coco_size_eval(save_dir: Path, data_dir: Path, model_names: dict):
    """
    Dùng predictions.json từ model.val(save_json=True) — không inference lại.
    Mọi image_id được quy về số nguyên để pycocotools không bị lỗi mismatch.
    """
    dt_path = save_dir / "predictions.json"
    if not dt_path.exists():
        print(f"[!] Không tìm thấy {dt_path} — bỏ qua COCO size eval.")
        return

    # Load detections ultralytics đã lưu
    dt_list = json.loads(dt_path.read_text())
    if not dt_list:
        print("[!] predictions.json rỗng — bỏ qua.")
        return
    print(f"[COCO] Đọc {len(dt_list)} detections từ {dt_path}")

    # Tìm val/images và val/labels
    img_dir = next((p for p in data_dir.rglob("val/images") if p.is_dir()), None)
    if img_dir is None:
        print("[!] Không tìm thấy val/images — bỏ qua.")
        return
    label_dir = img_dir.parent / "labels"

    # 1. Quét file thực tế
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    img_files = [p for p in img_dir.iterdir() if p.suffix.lower() in img_exts]

    # ultralytics map: image_id = int(stem) nếu numeric, ngược lại là stem
    ul_id_to_filepath = {
        (int(p.stem) if p.stem.isnumeric() else p.stem): p
        for p in img_files
    }

    used_dt_ids = {d["image_id"] for d in dt_list}

    # ⚠ CRITICAL: we MUST register every val image in the COCO GT — including
    # images where the model produced zero detections. If we only register
    # images that appear in predictions.json, every GT object on a
    # zero-detection image is silently dropped from pycocotools. Those lost
    # FNs bias AP_small the hardest (small objects are the ones models miss
    # most often), which is exactly the symptom where an `ultralytics.train`
    # model looks "correct" (few zero-detection images) and a custom DA model
    # looks "wrong" (many zero-detection images on the target domain).
    images, annotations = [], []
    ann_id = 1
    ul_id_to_safe_id = {}
    next_safe_id = 1

    # Iterate over ALL val images, not just ones with predictions.
    # Sort for deterministic safe_id assignment across runs.
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
                if len(parts) < 5: continue
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

    # Diagnostic: how many images had at least one prediction?
    n_total = len(all_val_ids)
    n_with_preds = len(used_dt_ids & set(all_val_ids))
    n_zero_pred = n_total - n_with_preds
    print(f"[COCO] Val images: {n_total} total, {n_with_preds} with predictions, "
          f"{n_zero_pred} zero-detection")
    if n_zero_pred > 0:
        print(f"[COCO] {n_zero_pred} zero-detection image(s) contribute only FNs — "
              f"their GT objects (often small) are now correctly counted.")

    new_dt_list = []
    # Xác định xem predictions có bị +1 hay không
    # Ultralytics mặc định gán category_id = class_idx + 1 với tập dữ liệu custom
    # GT labels dùng 0-indexed (0=person, 1=car), predictions có thể 1-indexed (1=person, 2=car)
    gt_cat_ids = set(int(k) for k in model_names.keys())   # {0, 1}
    dt_cat_ids = {d.get("category_id", -1) for d in dt_list}
    
    # Nếu predictions KHÔNG overlap với GT categories nhưng shift -1 lại match → cần shift
    shift_cat = False
    if dt_cat_ids and not dt_cat_ids.issubset(gt_cat_ids):
        shifted_dt_cats = {c - 1 for c in dt_cat_ids}
        if shifted_dt_cats.issubset(gt_cat_ids):
            shift_cat = True
            print(f"[COCO] Auto-detected category_id +1 shift: dt={sorted(dt_cat_ids)} → gt={sorted(gt_cat_ids)}")

    for d in dt_list:
        ul_id = d["image_id"]
        if ul_id in ul_id_to_safe_id:
            d["image_id"] = ul_id_to_safe_id[ul_id]
            if shift_cat:
                d["category_id"] -= 1
            new_dt_list.append(d)

    coco_gt_dict = {
        "images": images, 
        "annotations": annotations, 
        "categories": [{"id": int(k), "name": v} for k, v in model_names.items()]
    }

    gt_path = data_dir / "_coco_gt_tmp.json"
    gt_path.write_text(json.dumps(coco_gt_dict))

    # COCOeval
    coco_gt   = COCO(str(gt_path))
    coco_dt   = coco_gt.loadRes(new_dt_list)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats   # 12 metrics chuẩn COCO
    rows = [
        ("AP",        "0.50:0.95", stats[0]),
        ("AP@50",     "0.50",      stats[1]),
        ("AP@75",     "0.75",      stats[2]),
        ("AP_small",  "0.50:0.95", stats[3]),   # area < 32²
        ("AP_medium", "0.50:0.95", stats[4]),   # 32² – 96²
        ("AP_large",  "0.50:0.95", stats[5]),   # area > 96²
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


def evaluate(weight_path, data_dir, dataset_name, base_weights):
    print("=" * 60)
    print(f"Đánh giá {dataset_name} trên '{data_dir}'".center(60))
    print("=" * 60)

    model = load_model_custom(weight_path, base_weights)
    model_names = model.names
    yaml_path = create_yaml(data_dir, model_names)
    print(f"Config: {yaml_path}")

    # ── Một lần val(), vừa lấy mAP vừa dump predictions.json ──
    try:
        metrics = model.val(
            data=yaml_path, split="val", batch=4,
            verbose=True,
            save_json=True,   # ← dump COCO predictions, tái dùng cho pycocotools
        )
        print(f"\n{'─'*40}")
        print(f" Ultralytics – {dataset_name}")
        print("─" * 40)
        print(f" mAP@50    : {metrics.box.map50:.4f}")
        print(f" mAP@50-95 : {metrics.box.map:.4f}")
        print("─" * 40)
    except Exception as e:
        print(f"[!] Ultralytics val thất bại: {e}")
        return

    # ── pycocotools đọc lại file JSON, không inference lại ────
    print("\n[COCO Size Metrics] Tính AP_small / AP_medium / AP_large ...")
    run_coco_size_eval(Path(metrics.save_dir), Path(data_dir), model_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",      default="best.pt")
    parser.add_argument("--base-weights", default="yolo26s.pt")
    parser.add_argument("--source-dir",   default="source_test")
    parser.add_argument("--target-dir",   default="target_test")
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        print(f"[LỖI] Không tìm thấy: '{args.weights}'")
        exit(1)

    for dir_path, name in [(args.source_dir, "Source Test"),
                           (args.target_dir, "Target Test")]:
        if os.path.exists(dir_path):
            evaluate(args.weights, dir_path, name, args.base_weights)
        else:
            print(f"[!] Không tìm thấy thư mục '{dir_path}'")