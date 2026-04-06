"""
GradCAM / EigenCAM Visualization — RQ2 Direct Evidence
======================================================
Vẽ heatmap attention trên ảnh foggy, so sánh Full GRL vs NoGRL.

Full GRL: heatmap tập trung vào objects (person, car) — domain-invariant
NoGRL:    heatmap lan ra background/texture — domain-specific

Fix list (v2):
  1. Class names đọc từ data.yaml (KHÔNG dùng pretrained COCO names)
  2. Class-Discriminative EigenCAM cho global heatmap (PCA per-class từ detection boxes)
  3. Chọn ảnh có nhiều detection nhất thay vì linspace
  4. Alpha và colormap tối ưu cho heatmap rõ hơn
  5. Hỗ trợ YOLO26s đầy đủ (C2PSA auto-detect)

Usage:
    python gradcam_explain.py \\
        --weights yolo26s.pt \\
        --checkpoints \\
            results/full/weights/best.pt \\
        --names "Full (GRL)" \\
        --images datasets/target_real/target_real/val/images \\
        --data data.yaml \\
        --output results/gradcam \\
        --n-images 10 \\
        --global-heatmap \\
        --device 0

Dependencies:
    pip install grad-cam pyyaml
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml

from ultralytics import YOLO
from domain_adaptation import find_last_backbone_layer


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


# ==============================================================================
# Utilities
# ==============================================================================

def load_class_names(data_yaml_path):
    """
    Load class names từ data.yaml của dataset (KHÔNG dùng pretrained weights names).
    Trả về dict {int: str}, ví dụ {0: 'person', 1: 'car'}.
    """
    if data_yaml_path is None or not Path(data_yaml_path).exists():
        print("[WARNING] data.yaml không tìm thấy, dùng class index làm tên.")
        return {}
    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    names = cfg.get('names', {})
    # Hỗ trợ cả list và dict
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    elif isinstance(names, dict):
        names = {int(k): v for k, v in names.items()}
    print(f"[ClassNames] Loaded {len(names)} classes from {data_yaml_path}:")
    for idx, name in sorted(names.items()):
        print(f"  {idx}: {name}")
    return names


def load_model(weights_arch, checkpoint_path, device):
    """Load YOLO26s architecture + custom checkpoint."""
    yolo = YOLO(weights_arch)
    model = yolo.model.to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        # Thử nhiều key phổ biến
        for key in ('model', 'ema', 'state_dict'):
            if key in ckpt:
                state = ckpt[key]
                # Nếu là YOLO object thì lấy .state_dict()
                if hasattr(state, 'state_dict'):
                    state = state.state_dict()
                elif hasattr(state, 'float'):
                    state = state.float().state_dict()
                model.load_state_dict(state, strict=False)
                print(f"[Checkpoint] Loaded key='{key}' from {checkpoint_path}")
                break
        else:
            model.load_state_dict(ckpt, strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    model.eval()
    return model


def letterbox(img, new_shape=640, stride=32):
    """Resize + pad. Returns (img, pad_info) where pad_info = (top, bottom, left, right)."""
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
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, (top, bottom, left, right)


def preprocess(img_path, imgsz=640, device=None):
    """Load + letterbox + normalize → tensor [1, 3, H, W]."""
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_lb, pad_info = letterbox(img_rgb, new_shape=imgsz)

    tensor = img_lb.transpose(2, 0, 1).astype(np.float32) / 255.0
    tensor = np.ascontiguousarray(tensor)
    tensor = torch.from_numpy(tensor).unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device)
    return img_rgb, img_lb, pad_info, tensor


# ==============================================================================
# Detection parsing — YOLO26s output: [1, num_anchors, 4+nc] or [1, 4+nc, anchors]
# ==============================================================================

def parse_detections(preds, conf_thresh=0.25, nc=None):
    """
    Parse YOLO raw predictions thành list of (x1,y1,x2,y2,conf,cls_id).
    Handles both xywh và xyxy, and both output shapes.

    YOLO26s ultralytics output thường là tensor [1, 4+nc, num_anchors].
    """
    if isinstance(preds, (list, tuple)):
        preds = preds[0]
    if not isinstance(preds, torch.Tensor):
        return [], {}

    if preds.dim() == 3:
        # Shape [1, 4+nc, num_anchors] → transpose → [1, num_anchors, 4+nc]
        if preds.shape[1] < preds.shape[2]:
            # shape [1, 4+nc, N] → already rows=classes
            pred = preds[0]  # [4+nc, N]
        else:
            # shape [1, N, 4+nc]
            pred = preds[0].T  # [4+nc, N]
    elif preds.dim() == 2:
        pred = preds.T if preds.shape[0] < preds.shape[1] else preds
    else:
        return [], {}

    # pred shape: [4+nc, N]
    boxes_xywh = pred[:4, :]   # [4, N]
    class_scores = pred[4:, :]  # [nc, N]

    if nc is None:
        nc = class_scores.shape[0]

    # Convert xywh → xyxy
    cx, cy, w_box, h_box = boxes_xywh[0], boxes_xywh[1], boxes_xywh[2], boxes_xywh[3]
    x1 = cx - w_box / 2
    y1 = cy - h_box / 2
    x2 = cx + w_box / 2
    y2 = cy + h_box / 2

    # Determine conf
    if class_scores.max() > 1.0 or class_scores.min() < 0.0:
        class_scores_prob = torch.sigmoid(class_scores)
    else:
        class_scores_prob = class_scores

    max_scores, max_cls = torch.max(class_scores_prob, dim=0)  # [N]

    mask = max_scores > conf_thresh
    detections = []
    class_detection_count = {}

    for i in range(mask.sum().item()):
        idx = mask.nonzero(as_tuple=False).squeeze(1)[i].item()
        cls_id = int(max_cls[idx].item())
        conf = float(max_scores[idx].item())
        box = (float(x1[idx].item()), float(y1[idx].item()),
               float(x2[idx].item()), float(y2[idx].item()))
        detections.append((*box, conf, cls_id))
        class_detection_count[cls_id] = class_detection_count.get(cls_id, 0) + 1

    return detections, class_detection_count


# ==============================================================================
# EigenCAM — Class-Aware Version
# ==============================================================================

class YOLO26EigenCAM:
    """
    EigenCAM SOTA implementation cho YOLO26s / YOLOv8.

    Global heatmap per class:
    - Class-Discriminative: chỉ dùng spatial positions tương ứng với
      detection boxes của class đó → heatmap khác nhau rõ ràng giữa các class.

    Single-image heatmap:
    - Global PCA trên toàn bộ feature map (standard EigenCAM).
    """

    def __init__(self, model, target_layer_idx=None):
        self.model = model
        self.features = None
        self.hook = None

        # Auto-detect layer nếu không chỉ định
        if target_layer_idx is None or target_layer_idx < 0:
            target_layer_idx = find_last_backbone_layer(model)
            print(f"[EigenCAM] Auto-detected backbone end layer: {target_layer_idx}")

        # Register hook
        layers = self._get_layers(model)
        if layers is None:
            raise RuntimeError("Cannot find model layers!")

        target = layers[target_layer_idx]
        self.hook = target.register_forward_hook(self._hook_fn)
        self.target_name = target.__class__.__name__
        print(f"[EigenCAM] Hook on layer idx={target_layer_idx}: {self.target_name}")

    def _get_layers(self, model):
        if hasattr(model, 'model') and isinstance(model.model, nn.Sequential):
            return model.model
        elif hasattr(model, 'model') and hasattr(model.model, 'model'):
            return model.model.model
        elif isinstance(model, nn.Sequential):
            return model
        try:
            return model.model
        except Exception:
            return None

    def _hook_fn(self, module, input, output):
        self.features = output.detach()

    def _compute_eigencam_full(self, feat):
        """
        Standard EigenCAM: SVD on full feature map.
        feat: [C, H, W] numpy
        Returns: heatmap [H, W] normalized [0,1]
        """
        C, H, W = feat.shape
        feat_2d = feat.reshape(C, H * W)

        # Center features (quan trọng: giúp PCA phản ánh đúng variance)
        feat_2d = feat_2d - feat_2d.mean(axis=1, keepdims=True)

        try:
            U, S, Vt = np.linalg.svd(feat_2d, full_matrices=False)
            cam = Vt[0].reshape(H, W)
        except np.linalg.LinAlgError:
            cam = feat_2d.mean(axis=0).reshape(H, W)

        # Clamp negative (chỉ giữ positive activations — quan trọng cho heatmap đỏ)
        cam = np.maximum(cam, 0)
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam

    def _compute_eigencam_class(self, feat, boxes_normalized, img_h, img_w):
        """
        Class-Discriminative EigenCAM:
        Chiếu feature map theo spatial locations trong detection boxes của class này.
        feat: [C, H_feat, W_feat] numpy
        boxes_normalized: list of (x1,y1,x2,y2) đã normalized [0,1] theo img size

        Returns: heatmap [H_feat, W_feat] normalized [0,1]
        """
        C, H_feat, W_feat = feat.shape

        # Build spatial weight mask từ detection boxes
        mask = np.zeros((H_feat, W_feat), dtype=np.float32)
        for (x1n, y1n, x2n, y2n) in boxes_normalized:
            # Convert to feature map coords
            fx1 = int(max(0, x1n * W_feat))
            fy1 = int(max(0, y1n * H_feat))
            fx2 = int(min(W_feat, x2n * W_feat))
            fy2 = int(min(H_feat, y2n * H_feat))
            if fx2 > fx1 and fy2 > fy1:
                mask[fy1:fy2, fx1:fx2] = 1.0

        if mask.sum() == 0:
            # Fallback về full EigenCAM nếu không có box nào trong feature map
            return self._compute_eigencam_full(feat)

        # Lấy các vectors tại vị trí trong boxes
        feat_2d = feat.reshape(C, H_feat * W_feat)
        mask_flat = mask.reshape(-1).astype(bool)
        feat_in_box = feat_2d[:, mask_flat]  # [C, K]

        if feat_in_box.shape[1] < 2:
            return self._compute_eigencam_full(feat)

        # Center
        feat_in_box = feat_in_box - feat_in_box.mean(axis=1, keepdims=True)

        try:
            U, S, Vt = np.linalg.svd(feat_in_box, full_matrices=False)
            # Project first principal component back to full spatial space
            first_pc = U[:, 0]  # [C]
            cam_flat = feat_2d.T @ first_pc  # [H_feat*W_feat]
            cam = cam_flat.reshape(H_feat, W_feat)
        except np.linalg.LinAlgError:
            cam = feat_in_box.mean(axis=0).reshape(H_feat, W_feat) if False else \
                feat_2d.mean(axis=0).reshape(H_feat, W_feat)

        # Clamp negative
        cam = np.maximum(cam, 0)
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam

    def forward(self, img_tensor, pad_info=None):
        """
        Chạy forward pass và trả về (heatmap_full, raw_preds).
        heatmap_full: standard EigenCAM [H, W] ∈ [0,1]
        """
        self.model.eval()
        with torch.no_grad():
            preds = self.model(img_tensor)

        if self.features is None:
            return np.zeros((img_tensor.shape[2], img_tensor.shape[3])), preds

        feat = self.features[0].cpu().numpy()  # [C, H_feat, W_feat]
        cam = self._compute_eigencam_full(feat)

        # Resize to input size
        img_h, img_w = img_tensor.shape[2], img_tensor.shape[3]
        cam_resized = cv2.resize(cam.astype(np.float32), (img_w, img_h))

        # Mask padding
        cam_resized = self._mask_padding(cam_resized, pad_info, img_h, img_w)
        return cam_resized, preds

    def forward_class(self, img_tensor, pad_info, detections, class_id, img_h_orig, img_w_orig):
        """
        Class-Discriminative EigenCAM cho một class cụ thể.
        detections: list of (x1,y1,x2,y2,conf,cls_id) trong coordinate của img_tensor (pixel)
        """
        self.model.eval()
        with torch.no_grad():
            preds = self.model(img_tensor)

        if self.features is None:
            return np.zeros((img_tensor.shape[2], img_tensor.shape[3]))

        feat = self.features[0].cpu().numpy()  # [C, H_feat, W_feat]
        C, H_feat, W_feat = feat.shape

        img_h, img_w = img_tensor.shape[2], img_tensor.shape[3]

        # Filter boxes của class này + normalize [0,1]
        boxes_norm = []
        for det in detections:
            if len(det) >= 6 and int(det[5]) == class_id:
                x1, y1, x2, y2 = det[:4]
                # Normalize bằng input tensor size
                boxes_norm.append((
                    max(0.0, x1 / img_w),
                    max(0.0, y1 / img_h),
                    min(1.0, x2 / img_w),
                    min(1.0, y2 / img_h),
                ))

        cam = self._compute_eigencam_class(feat, boxes_norm, img_h, img_w)
        cam_resized = cv2.resize(cam.astype(np.float32), (img_w, img_h))
        cam_resized = self._mask_padding(cam_resized, pad_info, img_h, img_w)
        return cam_resized

    def _mask_padding(self, cam, pad_info, img_h, img_w):
        if pad_info is None:
            return cam
        top, bottom, left, right = pad_info
        if top > 0:
            cam[:top, :] = 0
        if bottom > 0:
            cam[-bottom:, :] = 0
        if left > 0:
            cam[:, :left] = 0
        if right > 0:
            cam[:, -right:] = 0
        # Re-normalize content area
        y1 = top
        y2 = img_h - bottom if bottom > 0 else img_h
        x1 = left
        x2 = img_w - right if right > 0 else img_w
        content = cam[y1:y2, x1:x2]
        if content.size > 0 and content.max() > content.min():
            cam[y1:y2, x1:x2] = (content - content.min()) / (content.max() - content.min())
        return cam

    def remove(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None

    def __del__(self):
        self.remove()


# ==============================================================================
# Image selection: ưu tiên ảnh có nhiều detection nhất
# ==============================================================================

def select_images_by_richness(images_dir, model, device, imgsz=640,
                               n_images=10, conf_thresh=0.25, nc=None, seed=42):
    """
    Chọn ảnh có số lượng detection nhiều nhất (phong phú nhất).
    Điều này đảm bảo heatmap rõ ràng và có màu đỏ tập trung vào objects.
    """
    image_files = sorted([
        f for f in Path(images_dir).iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if len(image_files) == 0:
        raise FileNotFoundError(f"No images found in {images_dir}")

    if len(image_files) <= n_images:
        print(f"[Select] Only {len(image_files)} images available, using all.")
        return image_files

    print(f"[Select] Scoring {len(image_files)} images by detection richness...")

    scores = []
    model.eval()
    with torch.no_grad():
        for img_path in image_files:
            try:
                _, _, pad_info, tensor = preprocess(img_path, imgsz, device)
                preds = model(tensor)
                _, cls_counts = parse_detections(preds, conf_thresh=conf_thresh, nc=nc)
                total_dets = sum(cls_counts.values())
                n_classes = len(cls_counts)
                # Score: tổng detection + bonus cho nhiều class khác nhau
                score = total_dets + n_classes * 2
            except Exception as e:
                score = 0
            scores.append(score)

    # Sort theo score giảm dần, lấy n_images ảnh đầu
    ranked = sorted(zip(scores, image_files), key=lambda x: x[0], reverse=True)
    selected = [f for _, f in ranked[:n_images]]

    print(f"[Select] Selected top {n_images} images by detection richness:")
    for score, f in ranked[:n_images]:
        print(f"  {f.name}: score={score:.0f}")

    return selected


# ==============================================================================
# Heatmap overlay
# ==============================================================================

def apply_heatmap(img_rgb, heatmap, alpha=0.55, colormap=cv2.COLORMAP_JET):
    """
    Overlay heatmap on image.
    alpha=0.55 cho màu đỏ đậm hơn, dễ thấy hơn so với 0.45.
    """
    # Boost contrast của heatmap với power transform
    heatmap_boosted = np.power(heatmap, 0.7)  # gamma < 1 → kéo giá trị thấp lên
    heatmap_uint8 = (heatmap_boosted * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = (img_rgb.astype(np.float32) * (1 - alpha) +
               heatmap_colored.astype(np.float32) * alpha)
    return overlay.clip(0, 255).astype(np.uint8)


def draw_detections_on_image(img_rgb, detections, class_names, line_thickness=2):
    """Vẽ bounding boxes lên ảnh để dễ so sánh với heatmap."""
    img_out = img_rgb.copy()
    colors = {0: (255, 80, 80), 1: (80, 200, 255)}  # person=đỏ, car=xanh
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        color = colors.get(int(cls_id), (200, 200, 200))
        cv2.rectangle(img_out,
                      (int(x1), int(y1)), (int(x2), int(y2)),
                      color, line_thickness)
        label = f"{class_names.get(int(cls_id), str(int(cls_id)))} {conf:.2f}"
        cv2.putText(img_out, label,
                    (int(x1), max(0, int(y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img_out


# ==============================================================================
# Output generators
# ==============================================================================

def create_comparison_grid(image_files, all_heatmaps, model_names, output_dir,
                            imgsz=640, all_detections=None, class_names=None):
    """Create side-by-side comparison grid với detection boxes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_images = len(image_files)
    n_models = len(model_names)

    # n_cols: Original + [det] + n_models
    has_det = all_detections is not None
    n_cols = 1 + (1 if has_det else 0) + n_models

    fig, axes = plt.subplots(n_images, n_cols,
                              figsize=(4 * n_cols, 4 * n_images))
    if n_images == 1:
        axes = axes.reshape(1, -1)

    col_titles = ['Original'] + (['Detections'] if has_det else []) + model_names

    for i, img_path in enumerate(image_files):
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_lb, _ = letterbox(img_rgb, new_shape=imgsz)

        col = 0
        # Col 0: Original
        axes[i, col].imshow(img_lb)
        if i == 0:
            axes[i, col].set_title(col_titles[col], fontsize=11, fontweight='bold')
        axes[i, col].axis('off')
        axes[i, col].set_ylabel(img_path.stem[:18], fontsize=8, rotation=0,
                                 labelpad=55, va='center')
        col += 1

        # Col 1 (optional): Detections
        if has_det:
            dets_name = model_names[0]
            dets = all_detections.get(dets_name, {}).get(img_path.name, [])
            det_img = draw_detections_on_image(img_lb, dets, class_names or {})
            axes[i, col].imshow(det_img)
            if i == 0:
                axes[i, col].set_title(col_titles[col], fontsize=11, fontweight='bold')
            axes[i, col].axis('off')
            col += 1

        # Heatmap columns
        for j, name in enumerate(model_names):
            heatmap = all_heatmaps[name][i]
            overlay = apply_heatmap(img_lb, heatmap, alpha=0.55)
            axes[i, col + j].imshow(overlay)
            if i == 0:
                axes[i, col + j].set_title(name, fontsize=11, fontweight='bold')
            axes[i, col + j].axis('off')

    fig.suptitle('EigenCAM Activation Maps — Target Domain (Foggy)',
                 fontsize=15, fontweight='bold', y=1.005)
    plt.tight_layout()

    save_path = output_dir / 'eigencam_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def create_individual_heatmaps(image_files, all_heatmaps, model_names, output_dir,
                                all_detections=None, class_names=None, imgsz=640):
    """Save individual heatmaps: heatmap, original, overlay side by side."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    indiv_dir = output_dir / 'individual'
    indiv_dir.mkdir(exist_ok=True)

    for i, img_path in enumerate(image_files):
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_lb, _ = letterbox(img_rgb, new_shape=imgsz)

        for name in model_names:
            heatmap = all_heatmaps[name][i]
            overlay = apply_heatmap(img_lb, heatmap, alpha=0.60)

            # Save pure heatmap (colormap only)
            heatmap_boosted = np.power(heatmap, 0.7)
            heatmap_uint8 = (heatmap_boosted * 255).astype(np.uint8)
            heatmap_vis = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

            # 3-panel: original | heatmap | overlay
            h, w = img_lb.shape[:2]
            panel = np.zeros((h, w * 3, 3), dtype=np.uint8)
            panel[:, :w] = cv2.cvtColor(img_lb, cv2.COLOR_RGB2BGR)
            panel[:, w:2*w] = heatmap_vis
            panel[:, 2*w:] = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

            # Add labels
            for col_i, label in enumerate(['Original', 'EigenCAM', 'Overlay']):
                cv2.putText(panel, label, (col_i * w + 8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

            save_name = f'{img_path.stem}_{name.replace(" ", "_").lower()}.png'
            cv2.imwrite(str(indiv_dir / save_name), panel)

    print(f"✅ Saved {len(image_files) * len(model_names)} individual heatmaps to {indiv_dir}")


def create_global_class_heatmaps(global_dir, model_name, heatmap_sums_cls, heatmap_counts_cls,
                                   class_names, imgsz=640):
    """
    Tạo global heatmap trung bình cho mỗi class.
    Dùng class-discriminative accumulation → các class khác nhau rõ ràng.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not heatmap_sums_cls:
        print(f"[GlobalHeatmap] No class heatmaps to save for {model_name}")
        return

    n_classes = len(heatmap_sums_cls)
    fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 5))
    if n_classes == 1:
        axes = [axes]

    for idx, (c, h_sum) in enumerate(sorted(heatmap_sums_cls.items())):
        count = heatmap_counts_cls[c]
        h_avg = h_sum / max(count, 1)
        # Normalize
        if h_avg.max() > h_avg.min():
            h_avg = (h_avg - h_avg.min()) / (h_avg.max() - h_avg.min())

        cls_name = class_names.get(c, f"cls_{c}")

        # Save raw PNG
        h_uint8 = (h_avg * 255).astype(np.uint8)
        h_colored = cv2.applyColorMap(h_uint8, cv2.COLORMAP_JET)
        save_name = f'{model_name.replace(" ", "_").lower()}_{cls_name}_global_n{count}.png'
        cv2.imwrite(str(global_dir / save_name), h_colored)

        # Plot in figure
        axes[idx].imshow(h_avg, cmap='jet', vmin=0, vmax=1)
        axes[idx].set_title(f'{cls_name}\n(n={count} images)', fontsize=13, fontweight='bold')
        axes[idx].axis('off')

        # Colorbar
        im = axes[idx].imshow(h_avg, cmap='jet', vmin=0, vmax=1)
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

    fig.suptitle(f'Class-Discriminative Global EigenCAM — {model_name}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = global_dir / f'{model_name.replace(" ", "_").lower()}_global_classes.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved global class heatmaps comparison: {fig_path}")


# ==============================================================================
# Main run
# ==============================================================================

def run(opt):
    device = torch.device(f'cuda:{opt.device}' if opt.device.isdigit() else opt.device)
    output_dir = Path(opt.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load class names từ data.yaml (KHÔNG dùng pretrained COCO weights names)
    class_names = load_class_names(opt.data)
    nc = len(class_names) if class_names else None

    # --- Load first model tạm để score images ---
    print(f"\n[SelectImages] Loading first model to score images by detection richness...")
    first_ckpt = opt.checkpoints[0]
    temp_model = load_model(opt.weights, first_ckpt, device)

    image_files = select_images_by_richness(
        images_dir=opt.images,
        model=temp_model,
        device=device,
        imgsz=opt.imgsz,
        n_images=opt.n_images,
        conf_thresh=opt.conf_thresh,
        nc=nc,
    )
    del temp_model
    torch.cuda.empty_cache()

    print(f"\n📸 Selected {len(image_files)} images for visualization")

    # --- Process each model ---
    all_heatmaps = {}
    all_detections = {}

    for ckpt_path, name in zip(opt.checkpoints, opt.names):
        print(f"\n{'='*55}")
        print(f"  Processing: {name}")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"{'='*55}")

        model = load_model(opt.weights, ckpt_path, device)

        layer_idx = opt.layer if opt.layer >= 0 else None
        eigencam = YOLO26EigenCAM(model, target_layer_idx=layer_idx)

        heatmaps = []
        det_per_img = {}  # img_name → list of detections

        # Per-class accumulation (class-discriminative)
        heatmap_sums_cls = {}   # {cls_id: np.ndarray summed}
        heatmap_counts_cls = {} # {cls_id: int}

        for img_path in image_files:
            img_rgb, img_lb, pad_info, tensor = preprocess(img_path, opt.imgsz, device)
            img_h_tensor, img_w_tensor = tensor.shape[2], tensor.shape[3]

            # Forward (standard EigenCAM for single-image visualization)
            heatmap_full, preds = eigencam.forward(tensor, pad_info=pad_info)
            heatmaps.append(heatmap_full)

            # Parse detections
            detections, cls_counts = parse_detections(preds, conf_thresh=opt.conf_thresh, nc=nc)
            det_per_img[img_path.name] = detections

            if detections:
                dets_str = ', '.join([f"{class_names.get(c,'?')}:{n}" for c, n in cls_counts.items()])
                print(f"  {img_path.name}: [{dets_str}]")
            else:
                print(f"  {img_path.name}: [no detections > {opt.conf_thresh}]")

            # Global class-discriminative heatmap accumulation
            if opt.global_heatmap and detections:
                present_classes = set(int(d[5]) for d in detections)
                for cls_id in present_classes:
                    cam_cls = eigencam.forward_class(
                        tensor, pad_info, detections, cls_id,
                        img_h_tensor, img_w_tensor
                    )
                    if cls_id not in heatmap_sums_cls:
                        heatmap_sums_cls[cls_id] = np.zeros_like(cam_cls)
                        heatmap_counts_cls[cls_id] = 0
                    heatmap_sums_cls[cls_id] += cam_cls
                    heatmap_counts_cls[cls_id] += 1

        all_heatmaps[name] = heatmaps
        all_detections[name] = det_per_img

        # Save global class heatmaps
        if opt.global_heatmap:
            global_dir = output_dir / 'global_class_heatmaps'
            global_dir.mkdir(exist_ok=True)
            create_global_class_heatmaps(
                global_dir, name,
                heatmap_sums_cls, heatmap_counts_cls,
                class_names, imgsz=opt.imgsz
            )

        eigencam.remove()
        del model
        torch.cuda.empty_cache()

    # Create comparison grid (with detection column)
    create_comparison_grid(
        image_files, all_heatmaps, opt.names, output_dir, opt.imgsz,
        all_detections=all_detections,
        class_names=class_names
    )

    # Individual heatmaps (3-panel)
    create_individual_heatmaps(
        image_files, all_heatmaps, opt.names, output_dir,
        all_detections=all_detections,
        class_names=class_names,
        imgsz=opt.imgsz
    )

    print(f"\n{'='*60}")
    print(f"{'EIGENCAM COMPLETE':^60}")
    print(f"{'='*60}")
    print(f"  Output: {output_dir}")
    print(f"  Classes: { {k: v for k, v in class_names.items()} }")
    print(f"{'='*60}")


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Class-Discriminative EigenCAM Visualization for YOLO26s'
    )

    parser.add_argument('--weights', type=str, default='yolo26s.pt',
                        help='YOLO26s architecture weights (pretrained base)')
    parser.add_argument('--checkpoints', nargs='+', required=True,
                        help='Checkpoint paths for each model to compare')
    parser.add_argument('--names', nargs='+', required=True,
                        help='Model names for legend (must match --checkpoints count)')
    parser.add_argument('--images', type=str, required=True,
                        help='Target images directory')
    parser.add_argument('--data', type=str, default='data.yaml',
                        help='Path to data.yaml (for correct class names)')
    parser.add_argument('--output', type=str,
                        default='results_explainable/gradcam',
                        help='Output directory')
    parser.add_argument('--n-images', type=int, default=10,
                        help='Number of images to visualize (selected by detection richness)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--layer', type=int, default=-1,
                        help='Target layer index (-1=auto-detect backbone end, '
                             '9=C2PSA for YOLO26s, 9=SPPF for YOLOv8)')
    parser.add_argument('--conf-thresh', type=float, default=0.25,
                        help='Confidence threshold for detection parsing')
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA device id or cpu')
    parser.add_argument('--global-heatmap', action='store_true',
                        help='Generate class-discriminative global average heatmaps')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)
