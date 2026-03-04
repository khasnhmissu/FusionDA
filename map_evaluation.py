"""
Object Detection Evaluation - mAP@50, mAP@75, mAP@[.5:.95], Precision, Recall, F1
====================================================================================
Input format (YOLO-style, normalized [0, 1]):

  labels/<image_name>.txt   → mỗi dòng: class_id  x_center  y_center  width  height
  predicts/<image_name>.txt → mỗi dòng: class_id  confidence  x_center  y_center  width  height

Usage:
  evaluator = DetectionEvaluator(labels_dir="labels", predicts_dir="predicts")
  results   = evaluator.evaluate()
  evaluator.print_report(results)
"""

import os
import glob
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# 1. DATA LOADING
# ──────────────────────────────────────────────

def load_labels(labels_dir: str) -> Dict[str, List[List[float]]]:
    """
    Load ground-truth từ folder labels.

    Returns:
        { "img_001": [[class_id, x_c, y_c, w, h], ...], ... }
    """
    data = {}
    txt_files = glob.glob(os.path.join(labels_dir, "*.txt"))

    if not txt_files:
        raise FileNotFoundError(f"Không tìm thấy file .txt nào trong: {labels_dir}")

    for fpath in txt_files:
        name = os.path.splitext(os.path.basename(fpath))[0]
        boxes = []
        with open(fpath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError(
                        f"[labels] {fpath}: dòng '{line}' phải có 5 giá trị "
                        f"(class_id x_c y_c w h), nhưng có {len(parts)}"
                    )
                boxes.append([float(p) for p in parts])
        data[name] = boxes  # rỗng nếu file không có object
    return data


def load_predicts(predicts_dir: str) -> Dict[str, List[List[float]]]:
    """
    Load predictions từ folder predicts.

    Returns:
        { "img_001": [[class_id, conf, x_c, y_c, w, h], ...], ... }
    """
    data = {}
    txt_files = glob.glob(os.path.join(predicts_dir, "*.txt"))

    if not txt_files:
        raise FileNotFoundError(f"Không tìm thấy file .txt nào trong: {predicts_dir}")

    for fpath in txt_files:
        name = os.path.splitext(os.path.basename(fpath))[0]
        boxes = []
        with open(fpath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 6:
                    raise ValueError(
                        f"[predicts] {fpath}: dòng '{line}' phải có 6 giá trị "
                        f"(class_id conf x_c y_c w h), nhưng có {len(parts)}"
                    )
                boxes.append([float(p) for p in parts])
        data[name] = boxes
    return data


# ──────────────────────────────────────────────
# 2. IoU
# ──────────────────────────────────────────────

def xywh_to_xyxy(box: List[float]) -> Tuple[float, float, float, float]:
    """[x_center, y_center, w, h] → [x1, y1, x2, y2]"""
    x_c, y_c, w, h = box
    return x_c - w / 2, y_c - h / 2, x_c + w / 2, y_c + h / 2


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Tính IoU giữa 2 box ở dạng [x_center, y_center, w, h].
    """
    x1_min, y1_min, x1_max, y1_max = xywh_to_xyxy(box1)
    x2_min, y2_min, x2_max, y2_max = xywh_to_xyxy(box2)

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0.0, inter_x_max - inter_x_min) * max(0.0, inter_y_max - inter_y_min)
    area1      = max(0.0, x1_max - x1_min) * max(0.0, y1_max - y1_min)
    area2      = max(0.0, x2_max - x2_min) * max(0.0, y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


# ──────────────────────────────────────────────
# 3. MATCHING PREDICTIONS → GROUND TRUTHS
# ──────────────────────────────────────────────

def match_predictions(
    all_labels:   Dict[str, List[List[float]]],
    all_predicts: Dict[str, List[List[float]]],
    iou_threshold: float,
) -> Dict[int, Dict]:
    """
    Với mỗi class, duyệt tất cả predictions (đã sort theo confidence giảm dần),
    đánh dấu TP / FP và đếm tổng GT (để tính FN sau).

    Returns:
        {
          class_id: {
            "scores":   [conf, ...],   # sorted descending
            "tp":       [1/0, ...],    # 1 = TP, 0 = FP
            "n_gt":     int            # tổng số GT box của class này
          }
        }
    """
    # Gom tất cả prediction theo class
    # entry: (image_name, confidence, box_xywh)
    class_preds: Dict[int, List[Tuple]] = defaultdict(list)

    for img_name, preds in all_predicts.items():
        for pred in preds:
            cls  = int(pred[0])
            conf = pred[1]
            box  = pred[2:]  # [x_c, y_c, w, h]
            class_preds[cls].append((img_name, conf, box))

    # Đếm GT theo class
    class_n_gt: Dict[int, int] = defaultdict(int)
    for img_name, gts in all_labels.items():
        for gt in gts:
            class_n_gt[int(gt[0])] += 1

    # Tập tất cả class_id (union của label + predict)
    all_classes = set(class_n_gt.keys()) | set(class_preds.keys())

    results = {}
    for cls in all_classes:
        preds_cls = class_preds.get(cls, [])

        # Sort theo confidence giảm dần
        preds_cls.sort(key=lambda x: x[1], reverse=True)

        scores = []
        tp_arr = []

        # Track GT đã match (image_name → set of matched gt indices)
        matched_gt: Dict[str, set] = defaultdict(set)

        for img_name, conf, pred_box in preds_cls:
            scores.append(conf)

            gt_boxes_all = all_labels.get(img_name, [])
            gt_boxes_cls = [
                (i, gt[1:])  # (original_index, [x_c, y_c, w, h])
                for i, gt in enumerate(gt_boxes_all)
                if int(gt[0]) == cls
            ]

            best_iou   = 0.0
            best_gt_idx = -1

            for orig_idx, gt_box in gt_boxes_cls:
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou    = iou
                    best_gt_idx = orig_idx

            if best_iou >= iou_threshold and best_gt_idx not in matched_gt[img_name]:
                tp_arr.append(1)
                matched_gt[img_name].add(best_gt_idx)
            else:
                tp_arr.append(0)

        results[cls] = {
            "scores": np.array(scores, dtype=np.float32),
            "tp":     np.array(tp_arr, dtype=np.int32),
            "n_gt":   class_n_gt.get(cls, 0),
        }

    return results


# ──────────────────────────────────────────────
# 4. PRECISION-RECALL CURVE & AP
# ──────────────────────────────────────────────

def compute_pr_curve(
    tp_arr: np.ndarray,
    n_gt:   int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Từ mảng TP (đã sort theo confidence), tính Precision và Recall tại mỗi threshold.

    Returns:
        precisions, recalls  (np.ndarray, shape [N])
    """
    if len(tp_arr) == 0 or n_gt == 0:
        return np.array([1.0]), np.array([0.0])

    tp_cum = np.cumsum(tp_arr)
    fp_cum = np.cumsum(1 - tp_arr)

    recalls    = tp_cum / n_gt
    precisions = tp_cum / (tp_cum + fp_cum)

    # Thêm điểm đầu (0,1) để curve bắt đầu từ recall = 0
    recalls    = np.concatenate([[0.0], recalls])
    precisions = np.concatenate([[1.0], precisions])

    return precisions, recalls


def compute_ap(precisions: np.ndarray, recalls: np.ndarray) -> float:
    """
    Tính AP (Area Under PR Curve) theo phương pháp 101-point interpolation
    chuẩn COCO.
    """
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        # Precision tại recall >= t
        mask = recalls >= t
        if mask.any():
            ap += np.max(precisions[mask])
    return ap / 101.0


# ──────────────────────────────────────────────
# 5. PRECISION / RECALL / F1 TẠI NGƯỠNG CỐ ĐỊNH
# ──────────────────────────────────────────────

def compute_prf_at_threshold(
    class_results:  Dict[int, Dict],
    conf_threshold: float = 0.5,
    iou_threshold:  float = 0.5,
    all_labels:     Optional[Dict] = None,
    all_predicts:   Optional[Dict] = None,
) -> Dict[int, Dict]:
    """
    Tính Precision, Recall, F1 cho từng class tại conf_threshold cụ thể.
    (Dùng lại class_results đã match ở iou_threshold tương ứng)
    """
    per_class_prf = {}

    for cls, data in class_results.items():
        scores = data["scores"]
        tp     = data["tp"]
        n_gt   = data["n_gt"]

        # Lọc theo confidence threshold
        mask   = scores >= conf_threshold
        tp_fil = tp[mask]

        tp_count = int(tp_fil.sum())
        fp_count = int((1 - tp_fil).sum())
        fn_count = n_gt - tp_count

        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
        recall    = tp_count / n_gt                   if n_gt > 0                  else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        per_class_prf[cls] = {
            "TP": tp_count, "FP": fp_count, "FN": fn_count,
            "Precision": round(precision, 4),
            "Recall":    round(recall,    4),
            "F1":        round(f1,        4),
        }

    return per_class_prf


# ──────────────────────────────────────────────
# 6. MAIN EVALUATOR CLASS
# ──────────────────────────────────────────────

class DetectionEvaluator:
    """
    Đánh giá object detection với các chỉ số:
      - mAP@50
      - mAP@75
      - mAP@[.5:.95] (COCO-style)
      - Precision, Recall, F1 (tại conf_threshold & iou=0.5)
    """

    def __init__(
        self,
        labels_dir:     str,
        predicts_dir:   str,
        conf_threshold: float = 0.5,
        class_names:    Optional[Dict[int, str]] = None,
    ):
        self.labels_dir     = labels_dir
        self.predicts_dir   = predicts_dir
        self.conf_threshold = conf_threshold
        self.class_names    = class_names or {}

        print("📂 Đang load dữ liệu...")
        self.all_labels   = load_labels(labels_dir)
        self.all_predicts = load_predicts(predicts_dir)

        # Thống kê
        all_label_names   = set(self.all_labels.keys())
        all_predict_names = set(self.all_predicts.keys())
        only_in_labels    = all_label_names - all_predict_names
        only_in_predicts  = all_predict_names - all_label_names

        print(f"   Labels  : {len(all_label_names)} file(s)")
        print(f"   Predicts: {len(all_predict_names)} file(s)")

        if only_in_labels:
            print(f"   ⚠️  {len(only_in_labels)} ảnh có label nhưng không có predict → toàn bộ GT thành FN")
        if only_in_predicts:
            print(f"   ⚠️  {len(only_in_predicts)} ảnh có predict nhưng không có label → toàn bộ prediction thành FP")

        # Đảm bảo mọi ảnh trong predicts đều có key trong labels (và ngược lại)
        for name in only_in_predicts:
            self.all_labels[name] = []
        for name in only_in_labels:
            self.all_predicts[name] = []

    def _class_name(self, cls_id: int) -> str:
        return self.class_names.get(cls_id, f"class_{cls_id}")

    def evaluate(self) -> Dict:
        """
        Chạy toàn bộ evaluation. Trả về dict chứa tất cả chỉ số.
        """
        iou_thresholds = np.round(np.arange(0.5, 1.0, 0.05), 2)  # [0.50, 0.55, ..., 0.95]

        # ── AP theo từng IoU threshold ──────────────────
        ap_table: Dict[float, Dict[int, float]] = {}  # iou_thr → { class_id → AP }

        for iou_thr in iou_thresholds:
            class_results = match_predictions(self.all_labels, self.all_predicts, iou_thr)
            ap_per_class  = {}

            for cls, data in class_results.items():
                if data["n_gt"] == 0 and len(data["scores"]) == 0:
                    continue
                prec, rec = compute_pr_curve(data["tp"], data["n_gt"])
                ap_per_class[cls] = compute_ap(prec, rec)

            ap_table[iou_thr] = ap_per_class

        # ── Tổng hợp mAP ───────────────────────────────
        all_classes = sorted(
            set(cls for ap_dict in ap_table.values() for cls in ap_dict)
        )

        # mAP@50
        ap50_per_class = ap_table[0.50]
        map50 = float(np.mean(list(ap50_per_class.values()))) if ap50_per_class else 0.0

        # mAP@75
        ap75_per_class = ap_table[0.75]
        map75 = float(np.mean(list(ap75_per_class.values()))) if ap75_per_class else 0.0

        # mAP@[.5:.95]
        map5095_per_class = {}
        for cls in all_classes:
            aps = [ap_table[thr][cls] for thr in iou_thresholds if cls in ap_table[thr]]
            map5095_per_class[cls] = float(np.mean(aps)) if aps else 0.0
        map5095 = float(np.mean(list(map5095_per_class.values()))) if map5095_per_class else 0.0

        # ── Precision / Recall / F1 tại IoU=0.5 ───────
        class_results_50 = match_predictions(self.all_labels, self.all_predicts, 0.5)
        prf_per_class    = compute_prf_at_threshold(
            class_results_50,
            conf_threshold=self.conf_threshold,
        )

        # ── PR Curve data (cho IoU=0.5, để vẽ nếu cần) ─
        pr_curves = {}
        for cls, data in class_results_50.items():
            prec, rec = compute_pr_curve(data["tp"], data["n_gt"])
            pr_curves[cls] = {"precision": prec, "recall": rec}

        return {
            "mAP@50":         round(map50,   4),
            "mAP@75":         round(map75,   4),
            "mAP@[.5:.95]":   round(map5095, 4),
            "AP@50_per_class":      ap50_per_class,
            "AP@75_per_class":      ap75_per_class,
            "AP@[.5:.95]_per_class": map5095_per_class,
            "PRF_per_class":        prf_per_class,
            "PR_curves":            pr_curves,
            "all_classes":          all_classes,
        }

    def print_report(self, results: Dict, show_per_class: bool = True) -> None:
        """In báo cáo đẹp ra terminal."""

        sep  = "═" * 70
        sep2 = "─" * 70

        print(f"\n{sep}")
        print(f"{'OBJECT DETECTION EVALUATION REPORT':^70}")
        print(f"{sep}")

        # ── Overall metrics ───────────────────────────
        print(f"\n{'📊 OVERALL METRICS':}")
        print(f"  {'mAP@50':<25} {results['mAP@50']:.4f}")
        print(f"  {'mAP@75':<25} {results['mAP@75']:.4f}")
        print(f"  {'mAP@[.5:.95]':<25} {results['mAP@[.5:.95]']:.4f}")

        # Macro Precision / Recall / F1
        prf = results["PRF_per_class"]
        if prf:
            macro_p  = np.mean([v["Precision"] for v in prf.values()])
            macro_r  = np.mean([v["Recall"]    for v in prf.values()])
            macro_f1 = np.mean([v["F1"]        for v in prf.values()])
            print(f"\n  (tại conf ≥ {self.conf_threshold}, IoU ≥ 0.5)")
            print(f"  {'Precision (macro)':<25} {macro_p:.4f}")
            print(f"  {'Recall    (macro)':<25} {macro_r:.4f}")
            print(f"  {'F1        (macro)':<25} {macro_f1:.4f}")

        # ── Per-class metrics ─────────────────────────
        if show_per_class and results["all_classes"]:
            print(f"\n{'📋 PER-CLASS METRICS':}")
            print(f"{sep2}")

            header = f"  {'Class':<18} {'AP@50':>7} {'AP@75':>7} {'AP@5095':>8} {'P':>7} {'R':>7} {'F1':>7} {'TP':>5} {'FP':>5} {'FN':>5}"
            print(header)
            print(f"  {'-'*66}")

            for cls in results["all_classes"]:
                name    = self._class_name(cls)
                ap50    = results["AP@50_per_class"].get(cls,       0.0)
                ap75    = results["AP@75_per_class"].get(cls,       0.0)
                ap5095  = results["AP@[.5:.95]_per_class"].get(cls, 0.0)
                prf_cls = results["PRF_per_class"].get(cls, {})
                p       = prf_cls.get("Precision", 0.0)
                r       = prf_cls.get("Recall",    0.0)
                f1      = prf_cls.get("F1",        0.0)
                tp      = prf_cls.get("TP",        0)
                fp      = prf_cls.get("FP",        0)
                fn      = prf_cls.get("FN",        0)

                print(
                    f"  {name:<18} {ap50:>7.4f} {ap75:>7.4f} {ap5095:>8.4f} "
                    f"{p:>7.4f} {r:>7.4f} {f1:>7.4f} {tp:>5} {fp:>5} {fn:>5}"
                )

            print(f"  {'-'*66}")

            # Dòng tổng kết
            ap50_vals   = [results["AP@50_per_class"].get(c,       0.0) for c in results["all_classes"]]
            ap75_vals   = [results["AP@75_per_class"].get(c,       0.0) for c in results["all_classes"]]
            ap5095_vals = [results["AP@[.5:.95]_per_class"].get(c, 0.0) for c in results["all_classes"]]
            print(
                f"  {'MEAN':<18} {np.mean(ap50_vals):>7.4f} {np.mean(ap75_vals):>7.4f} "
                f"{np.mean(ap5095_vals):>8.4f} {macro_p:>7.4f} {macro_r:>7.4f} {macro_f1:>7.4f}"
            )

        print(f"\n{sep}\n")

    def plot_pr_curves(self, results: Dict, save_path: Optional[str] = None) -> None:
        """
        Vẽ Precision-Recall curve cho từng class (tại IoU=0.5).
        Yêu cầu: matplotlib
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("⚠️  matplotlib chưa được cài. Chạy: pip install matplotlib")
            return

        pr_curves   = results["PR_curves"]
        all_classes = results["all_classes"]
        n_classes   = len(all_classes)

        if n_classes == 0:
            print("Không có class nào để vẽ.")
            return

        cols = min(3, n_classes)
        rows = (n_classes + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
        axes = np.array(axes).flatten() if n_classes > 1 else [axes]

        for i, cls in enumerate(all_classes):
            ax   = axes[i]
            data = pr_curves.get(cls)
            name = self._class_name(cls)
            ap50 = results["AP@50_per_class"].get(cls, 0.0)

            if data is not None:
                ax.plot(data["recall"], data["precision"],
                        color="steelblue", linewidth=2)
                ax.fill_between(data["recall"], data["precision"],
                                alpha=0.1, color="steelblue")

            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("Recall",    fontsize=11)
            ax.set_ylabel("Precision", fontsize=11)
            ax.set_title(f"{name}  (AP@50 = {ap50:.4f})", fontsize=12)
            ax.grid(True, linestyle="--", alpha=0.5)

        # Ẩn axes thừa
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(
            f"Precision-Recall Curves (IoU=0.5)\n"
            f"mAP@50={results['mAP@50']:.4f}  "
            f"mAP@75={results['mAP@75']:.4f}  "
            f"mAP@[.5:.95]={results['mAP@[.5:.95]']:.4f}",
            fontsize=13, y=1.01,
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"✅ Đã lưu PR curve: {save_path}")
        else:
            plt.show()


# ──────────────────────────────────────────────
# 7. QUICK TEST VỚI DỮ LIỆU GIẢ
# ──────────────────────────────────────────────

def _generate_demo_data(
    labels_dir:   str = "demo_labels",
    predicts_dir: str = "demo_predicts",
    n_images:     int = 20,
    n_classes:    int = 3,
    seed:         int = 42,
) -> None:
    """Tạo dữ liệu demo để test nhanh."""
    import random
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(labels_dir,   exist_ok=True)
    os.makedirs(predicts_dir, exist_ok=True)

    for i in range(n_images):
        name      = f"img_{i:04d}"
        n_objects = random.randint(1, 5)

        gt_boxes  = []
        for _ in range(n_objects):
            cls = random.randint(0, n_classes - 1)
            xc  = round(random.uniform(0.1, 0.9), 4)
            yc  = round(random.uniform(0.1, 0.9), 4)
            w   = round(random.uniform(0.05, 0.3), 4)
            h   = round(random.uniform(0.05, 0.3), 4)
            gt_boxes.append((cls, xc, yc, w, h))

        # Ghi label
        with open(os.path.join(labels_dir, f"{name}.txt"), "w") as f:
            for cls, xc, yc, w, h in gt_boxes:
                f.write(f"{cls} {xc} {yc} {w} {h}\n")

        # Ghi predict: 80% TP (với noise nhỏ) + thêm 1-2 FP ngẫu nhiên
        preds = []
        for cls, xc, yc, w, h in gt_boxes:
            if random.random() < 0.8:
                conf = round(random.uniform(0.5, 0.99), 4)
                preds.append((cls, conf,
                               round(xc + random.gauss(0, 0.01), 4),
                               round(yc + random.gauss(0, 0.01), 4),
                               round(w  + random.gauss(0, 0.01), 4),
                               round(h  + random.gauss(0, 0.01), 4)))

        for _ in range(random.randint(0, 2)):  # FP
            cls  = random.randint(0, n_classes - 1)
            conf = round(random.uniform(0.1, 0.5), 4)
            preds.append((cls, conf,
                           round(random.uniform(0.1, 0.9), 4),
                           round(random.uniform(0.1, 0.9), 4),
                           round(random.uniform(0.05, 0.3), 4),
                           round(random.uniform(0.05, 0.3), 4)))

        with open(os.path.join(predicts_dir, f"{name}.txt"), "w") as f:
            for row in preds:
                f.write(" ".join(str(v) for v in row) + "\n")

    print(f"✅ Đã tạo demo data: {n_images} ảnh, {n_classes} classes")
    print(f"   Folder labels  : {labels_dir}/")
    print(f"   Folder predicts: {predicts_dir}/")


# ──────────────────────────────────────────────
# 8. ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Object Detection mAP Evaluator")
    parser.add_argument("--labels",   type=str, default=None,  help="Đường dẫn folder labels")
    parser.add_argument("--predicts", type=str, default=None,  help="Đường dẫn folder predicts")
    parser.add_argument("--conf",     type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    parser.add_argument("--plot",     action="store_true",     help="Vẽ PR curves")
    parser.add_argument("--save-plot",type=str, default=None,  help="Lưu PR curves ra file ảnh")
    parser.add_argument("--demo",     action="store_true",     help="Chạy với dữ liệu demo")
    args = parser.parse_args()

    if args.demo or (args.labels is None and args.predicts is None):
        print("🔧 Chạy chế độ DEMO...\n")
        _generate_demo_data()
        labels_dir   = "demo_labels"
        predicts_dir = "demo_predicts"
    else:
        if not args.labels or not args.predicts:
            parser.error("Cần cung cấp cả --labels và --predicts (hoặc dùng --demo)")
        labels_dir   = args.labels
        predicts_dir = args.predicts

    class_names = {0: "person", 1: "car"}  # Tuỳ chỉnh theo project

    evaluator = DetectionEvaluator(
        labels_dir     = labels_dir,
        predicts_dir   = predicts_dir,
        conf_threshold = args.conf,
        class_names    = class_names,
    )

    results = evaluator.evaluate()
    evaluator.print_report(results)

    if args.plot or args.save_plot:
        evaluator.plot_pr_curves(results, save_path=args.save_plot)