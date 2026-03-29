"""
GradCAM / EigenCAM Visualization — RQ2 Direct Evidence
======================================================
Vẽ heatmap attention trên ảnh foggy, so sánh Full GRL vs NoGRL.

Full GRL: heatmap tập trung vào objects (person, car) — domain-invariant
NoGRL: heatmap lan ra background/texture — domain-specific

Usage:
    python gradcam_explain.py \
        --weights yolov8l.pt \
        --checkpoints \
            results/full-yolov8/weights/best.pt \
            results/nogrl-yolov8/weights/best.pt \
            results/freeze_teacher-yolov8/weights/best.pt \
            results/freeze_teacher-nogrl-yolov8/weights/best.pt \
        --names "Full (GRL)" "NoGRL" "Freeze Full" "Freeze NoGRL" \
        --images target_test/target_test/val/images \
        --output results_explainable/gradcam \
        --n-images 8 \
        --device 0

Dependencies:
    pip install grad-cam
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from ultralytics import YOLO


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def load_model(weights_arch, checkpoint_path, device):
    """Load YOLOv8 architecture + checkpoint."""
    yolo = YOLO(weights_arch)
    model = yolo.model.to(device)
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt
    
    model.load_state_dict(state_dict, strict=False)
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


class YOLOv8EigenCAM:
    """
    EigenCAM implementation for YOLOv8.
    
    EigenCAM: Uses first principal component of feature maps.
    More stable than GradCAM for detection models (no gradient issues).
    """
    
    def __init__(self, model, target_layer_idx=9):
        """
        Args:
            model: YOLOv8 model
            target_layer_idx: Layer index in model.model (9 = SPPF backbone output)
        """
        self.model = model
        self.features = None
        self.hook = None
        
        # Register hook
        if hasattr(model, 'model') and hasattr(model.model, 'model'):
            target = model.model.model[target_layer_idx]
        elif hasattr(model, 'model'):
            target = model.model[target_layer_idx]
        else:
            target = model[target_layer_idx]
        
        self.hook = target.register_forward_hook(self._hook_fn)
        self.target_name = target.__class__.__name__
        print(f"[EigenCAM] Hook on layer {target_layer_idx}: {self.target_name}")
    
    def _hook_fn(self, module, input, output):
        self.features = output
    
    def __call__(self, img_tensor, pad_info=None):
        """
        Generate EigenCAM heatmap.
        
        Args:
            img_tensor: [1, 3, H, W] normalized tensor
            pad_info: (top, bottom, left, right) padding from letterbox
            
        Returns:
            heatmap: [H, W] numpy array (0-1), padding regions masked to 0
        """
        self.model.eval()
        with torch.no_grad():
            _ = self.model(img_tensor)
        
        if self.features is None:
            return np.zeros((img_tensor.shape[2], img_tensor.shape[3]))
        
        # features shape: [1, C, H_feat, W_feat]
        feat = self.features[0]  # [C, H_feat, W_feat]
        C, H, W = feat.shape
        
        # Reshape to [C, H*W]
        feat_2d = feat.reshape(C, H * W).cpu().numpy()
        
        # SVD → first principal component
        # feat_2d: [C, H*W], U: [C, k], S: [k], Vt: [k, H*W]
        try:
            U, S, Vt = np.linalg.svd(feat_2d, full_matrices=False)
            # First component
            cam = Vt[0].reshape(H, W)
        except np.linalg.LinAlgError:
            cam = feat_2d.mean(axis=0).reshape(H, W)
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Resize to input image size
        img_h, img_w = img_tensor.shape[2], img_tensor.shape[3]
        cam_resized = cv2.resize(cam.astype(np.float32), (img_w, img_h))
        
        # Mask padding regions and re-normalize within content area
        if pad_info is not None:
            top, bottom, left, right = pad_info
            if top > 0:
                cam_resized[:top, :] = 0
            if bottom > 0:
                cam_resized[-bottom:, :] = 0
            if left > 0:
                cam_resized[:, :left] = 0
            if right > 0:
                cam_resized[:, -right:] = 0
            
            # Re-normalize content area to [0, 1]
            y1, y2 = top, img_h - bottom if bottom > 0 else img_h
            x1, x2 = left, img_w - right if right > 0 else img_w
            content = cam_resized[y1:y2, x1:x2]
            if content.size > 0 and content.max() > content.min():
                cam_resized[y1:y2, x1:x2] = \
                    (content - content.min()) / (content.max() - content.min())
        
        return cam_resized
    
    def remove(self):
        if self.hook is not None:
            self.hook.remove()
    
    def __del__(self):
        self.remove()


def apply_heatmap(img_rgb, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """Overlay heatmap on image."""
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    overlay = (img_rgb.astype(np.float32) * (1 - alpha) + 
               heatmap_colored.astype(np.float32) * alpha)
    return overlay.clip(0, 255).astype(np.uint8)


def select_images(images_dir, n_images=8, seed=42):
    """Select images for visualization."""
    image_files = sorted([
        f for f in Path(images_dir).iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ])
    
    if len(image_files) <= n_images:
        return image_files
    
    # Sample evenly across dataset
    np.random.seed(seed)
    indices = np.linspace(0, len(image_files) - 1, n_images, dtype=int)
    return [image_files[i] for i in indices]


def create_comparison_grid(image_files, all_heatmaps, model_names, output_dir, imgsz=640):
    """Create side-by-side comparison grid."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    
    n_images = len(image_files)
    n_models = len(model_names)
    
    fig, axes = plt.subplots(n_images, n_models + 1, 
                              figsize=(4 * (n_models + 1), 4 * n_images))
    
    if n_images == 1:
        axes = axes.reshape(1, -1)
    
    for i, img_path in enumerate(image_files):
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_lb, _ = letterbox(img_rgb, new_shape=imgsz)
        
        # Column 0: Original image
        axes[i, 0].imshow(img_lb)
        if i == 0:
            axes[i, 0].set_title('Original', fontsize=12, fontweight='bold')
        axes[i, 0].axis('off')
        axes[i, 0].set_ylabel(img_path.stem[:20], fontsize=9, rotation=0,
                               labelpad=60, va='center')
        
        # Columns 1..n: Heatmaps
        for j, name in enumerate(model_names):
            heatmap = all_heatmaps[name][i]
            overlay = apply_heatmap(img_lb, heatmap, alpha=0.45)
            
            axes[i, j + 1].imshow(overlay)
            if i == 0:
                axes[i, j + 1].set_title(name, fontsize=12, fontweight='bold')
            axes[i, j + 1].axis('off')
    
    fig.suptitle('EigenCAM Activation Maps — Target Domain (Foggy)',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    save_path = output_dir / 'eigencam_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def create_individual_heatmaps(image_files, all_heatmaps, model_names, output_dir, imgsz=640):
    """Save individual heatmaps for each image × model."""
    indiv_dir = output_dir / 'individual'
    indiv_dir.mkdir(exist_ok=True)
    
    for i, img_path in enumerate(image_files):
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_lb, _ = letterbox(img_rgb, new_shape=imgsz)
        
        for name in model_names:
            heatmap = all_heatmaps[name][i]
            overlay = apply_heatmap(img_lb, heatmap, alpha=0.45)
            
            save_name = f'{img_path.stem}_{name.replace(" ", "_").lower()}.png'
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(indiv_dir / save_name), overlay_bgr)
    
    print(f"✅ Saved {len(image_files) * len(model_names)} individual heatmaps to {indiv_dir}")


def run(opt):
    device = torch.device(f'cuda:{opt.device}' if opt.device.isdigit() else opt.device)
    output_dir = Path(opt.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Select images
    image_files = select_images(opt.images, opt.n_images)
    print(f"📸 Selected {len(image_files)} images for visualization")
    
    # Process each model
    all_heatmaps = {}
    
    for ckpt_path, name in zip(opt.checkpoints, opt.names):
        print(f"\n{'='*50}")
        print(f"Processing: {name}")
        print(f"Checkpoint: {ckpt_path}")
        print(f"{'='*50}")
        
        model = load_model(opt.weights, ckpt_path, device)
        eigencam = YOLOv8EigenCAM(model, target_layer_idx=opt.layer)
        
        heatmaps = []
        for img_path in image_files:
            img_bgr = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_lb, pad_info = letterbox(img_rgb, new_shape=opt.imgsz)
            
            img_tensor = img_lb.transpose(2, 0, 1)
            img_tensor = np.ascontiguousarray(img_tensor)
            img_tensor = torch.from_numpy(img_tensor).to(device).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0)
            
            heatmap = eigencam(img_tensor, pad_info=pad_info)
            heatmaps.append(heatmap)
        
        all_heatmaps[name] = heatmaps
        eigencam.remove()
        del model
        torch.cuda.empty_cache()
    
    # Create outputs
    create_comparison_grid(image_files, all_heatmaps, opt.names, output_dir, opt.imgsz)
    create_individual_heatmaps(image_files, all_heatmaps, opt.names, output_dir, opt.imgsz)
    
    print(f"\n{'='*60}")
    print(f"{'EIGENCAM COMPLETE':^60}")
    print(f"{'='*60}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")


def parse_args():
    parser = argparse.ArgumentParser(description='EigenCAM Visualization for YOLOv8')
    
    parser.add_argument('--weights', type=str, default='yolov8l.pt',
                        help='YOLOv8 architecture weights')
    parser.add_argument('--checkpoints', nargs='+', required=True,
                        help='Checkpoint paths for each model')
    parser.add_argument('--names', nargs='+', required=True,
                        help='Model names for legend')
    parser.add_argument('--images', type=str, required=True,
                        help='Target images directory')
    parser.add_argument('--output', type=str,
                        default='results_explainable/gradcam')
    parser.add_argument('--n-images', type=int, default=8,
                        help='Number of images to visualize')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--layer', type=int, default=9,
                        help='Target layer index (9=SPPF backbone output)')
    parser.add_argument('--device', type=str, default='0')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)
