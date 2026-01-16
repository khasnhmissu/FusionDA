"""
Image Quality Assessment Tool
=============================
Đánh giá chất lượng ảnh với các metrics phổ biến trong papers.

FULL-REFERENCE (cần ảnh ground truth):
    - PSNR: Peak Signal-to-Noise Ratio (higher is better)
    - SSIM: Structural Similarity Index (higher is better, max=1)
    - LPIPS: Learned Perceptual Image Patch Similarity (lower is better)

NO-REFERENCE (không cần ground truth):
    - NIQE: Natural Image Quality Evaluator (lower is better)
    - BRISQUE: Blind/Referenceless Image Spatial Quality Evaluator (lower is better)

DISTRIBUTION-BASED:
    - FID: Fréchet Inception Distance (lower is better)

Usage:
------
# Full-reference (so sánh với ground truth)
python evaluate_image_quality.py --input ./results/fake_B --reference ./datasets/testB --mode full

# No-reference (không cần ground truth)
python evaluate_image_quality.py --input ./results/fake_B --mode no-ref

# FID only (so sánh 2 distributions)
python evaluate_image_quality.py --input ./results/fake_B --reference ./datasets/testB --mode fid

# All metrics
python evaluate_image_quality.py --input ./results/fake_B --reference ./datasets/testB --mode all
"""

import os
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================
# UTILITY FUNCTIONS
# ============================================

def get_image_paths(folder, extensions=('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
    """Lấy tất cả đường dẫn ảnh trong folder."""
    folder = Path(folder)
    paths = []
    for ext in extensions:
        paths.extend(folder.glob(f'*{ext}'))
        paths.extend(folder.glob(f'*{ext.upper()}'))
    return sorted(paths)


def load_image_numpy(path, size=None):
    """Load ảnh thành numpy array (0-255, uint8)."""
    img = Image.open(path).convert('RGB')
    if size:
        img = img.resize(size, Image.BICUBIC)
    return np.array(img)


def load_image_tensor(path, size=None):
    """Load ảnh thành torch tensor (0-1, float32)."""
    import torch
    img = load_image_numpy(path, size)
    tensor = torch.from_numpy(img).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    return tensor


# ============================================
# FULL-REFERENCE METRICS
# ============================================

def calculate_psnr(img1, img2):
    """
    PSNR - Peak Signal-to-Noise Ratio
    Higher is better. Typical range: 20-40 dB
    """
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_ssim(img1, img2):
    """
    SSIM - Structural Similarity Index
    Higher is better. Range: -1 to 1 (typically 0.5-1.0 for good images)
    """
    from skimage.metrics import structural_similarity as ssim
    return ssim(img1, img2, channel_axis=2, data_range=255)


def calculate_lpips(img1_tensor, img2_tensor, lpips_model):
    """
    LPIPS - Learned Perceptual Image Patch Similarity
    Lower is better. Range: 0 to 1
    """
    import torch
    with torch.no_grad():
        # LPIPS expects input in range [-1, 1]
        img1_normalized = img1_tensor * 2 - 1
        img2_normalized = img2_tensor * 2 - 1
        distance = lpips_model(img1_normalized, img2_normalized)
    return distance.item()


# ============================================
# NO-REFERENCE METRICS
# ============================================

def calculate_niqe(img):
    """
    NIQE - Natural Image Quality Evaluator
    Lower is better. Typical range: 2-10
    """
    try:
        import pyiqa
        import torch
        
        # Create metric (singleton pattern for efficiency)
        if not hasattr(calculate_niqe, 'metric'):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            calculate_niqe.metric = pyiqa.create_metric('niqe', device=device)
        
        # Convert to tensor
        tensor = torch.from_numpy(img).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        
        with torch.no_grad():
            score = calculate_niqe.metric(tensor)
        return score.item()
    except ImportError:
        print("Warning: pyiqa not installed. Install with: pip install pyiqa")
        return None


def calculate_brisque(img):
    """
    BRISQUE - Blind/Referenceless Image Spatial Quality Evaluator
    Lower is better. Typical range: 0-100
    """
    try:
        import pyiqa
        import torch
        
        if not hasattr(calculate_brisque, 'metric'):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            calculate_brisque.metric = pyiqa.create_metric('brisque', device=device)
        
        tensor = torch.from_numpy(img).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        
        with torch.no_grad():
            score = calculate_brisque.metric(tensor)
        return score.item()
    except ImportError:
        print("Warning: pyiqa not installed. Install with: pip install pyiqa")
        return None


# ============================================
# DISTRIBUTION-BASED METRICS
# ============================================

def calculate_fid(folder1, folder2):
    """
    FID - Fréchet Inception Distance
    Lower is better. Typical range: 0-300 (good: <50)
    
    Đo khoảng cách giữa 2 distributions của ảnh.
    """
    try:
        from pytorch_fid import fid_score
        import torch
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        fid_value = fid_score.calculate_fid_given_paths(
            [str(folder1), str(folder2)],
            batch_size=50,
            device=device,
            dims=2048
        )
        return fid_value
    except ImportError:
        print("Warning: pytorch-fid not installed. Install with: pip install pytorch-fid")
        return None


# ============================================
# MAIN EVALUATION FUNCTIONS
# ============================================

def evaluate_full_reference(input_folder, reference_folder, use_lpips=True):
    """
    Đánh giá Full-Reference: PSNR, SSIM, LPIPS
    Yêu cầu: Ảnh trong input và reference phải có tên giống nhau.
    """
    import torch
    
    input_paths = get_image_paths(input_folder)
    ref_paths = get_image_paths(reference_folder)
    
    # Match files by name
    input_names = {p.name: p for p in input_paths}
    ref_names = {p.name: p for p in ref_paths}
    common_names = set(input_names.keys()) & set(ref_names.keys())
    
    if len(common_names) == 0:
        print("Warning: No matching filenames found between input and reference folders.")
        print(f"Input folder has {len(input_paths)} images, reference has {len(ref_paths)} images.")
        print("Falling back to order-based matching...")
        common_names = None
    
    # Initialize LPIPS model
    lpips_model = None
    if use_lpips:
        try:
            import lpips
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            lpips_model = lpips.LPIPS(net='alex').to(device)
            lpips_model.eval()
        except ImportError:
            print("Warning: lpips not installed. Install with: pip install lpips")
            use_lpips = False
    
    # Calculate metrics
    psnr_scores = []
    ssim_scores = []
    lpips_scores = []
    
    if common_names:
        pairs = [(input_names[name], ref_names[name]) for name in sorted(common_names)]
    else:
        pairs = list(zip(sorted(input_paths), sorted(ref_paths)))
    
    print(f"\nEvaluating {len(pairs)} image pairs...")
    
    for input_path, ref_path in tqdm(pairs, desc="Full-Reference Metrics"):
        # Load images
        img_input = load_image_numpy(input_path)
        img_ref = load_image_numpy(ref_path)
        
        # Resize if needed
        if img_input.shape != img_ref.shape:
            h, w = img_ref.shape[:2]
            img_input = load_image_numpy(input_path, size=(w, h))
        
        # PSNR
        psnr_scores.append(calculate_psnr(img_input, img_ref))
        
        # SSIM
        ssim_scores.append(calculate_ssim(img_input, img_ref))
        
        # LPIPS
        if use_lpips and lpips_model:
            device = next(lpips_model.parameters()).device
            tensor_input = load_image_tensor(input_path, size=(w, h) if img_input.shape != img_ref.shape else None).to(device)
            tensor_ref = load_image_tensor(ref_path).to(device)
            lpips_scores.append(calculate_lpips(tensor_input, tensor_ref, lpips_model))
    
    results = {
        'PSNR': {
            'mean': np.mean(psnr_scores),
            'std': np.std(psnr_scores),
            'interpretation': 'higher is better'
        },
        'SSIM': {
            'mean': np.mean(ssim_scores),
            'std': np.std(ssim_scores),
            'interpretation': 'higher is better (max=1)'
        }
    }
    
    if lpips_scores:
        results['LPIPS'] = {
            'mean': np.mean(lpips_scores),
            'std': np.std(lpips_scores),
            'interpretation': 'lower is better'
        }
    
    return results


def evaluate_no_reference(input_folder):
    """
    Đánh giá No-Reference: NIQE, BRISQUE
    Không cần ảnh ground truth.
    """
    input_paths = get_image_paths(input_folder)
    
    niqe_scores = []
    brisque_scores = []
    
    print(f"\nEvaluating {len(input_paths)} images...")
    
    for path in tqdm(input_paths, desc="No-Reference Metrics"):
        img = load_image_numpy(path)
        
        # NIQE
        niqe = calculate_niqe(img)
        if niqe is not None:
            niqe_scores.append(niqe)
        
        # BRISQUE
        brisque = calculate_brisque(img)
        if brisque is not None:
            brisque_scores.append(brisque)
    
    results = {}
    
    if niqe_scores:
        results['NIQE'] = {
            'mean': np.mean(niqe_scores),
            'std': np.std(niqe_scores),
            'interpretation': 'lower is better'
        }
    
    if brisque_scores:
        results['BRISQUE'] = {
            'mean': np.mean(brisque_scores),
            'std': np.std(brisque_scores),
            'interpretation': 'lower is better'
        }
    
    return results


def print_results(results, title=""):
    """In kết quả đánh giá."""
    print("\n" + "=" * 60)
    print(f" IMAGE QUALITY ASSESSMENT RESULTS {title}")
    print("=" * 60)
    
    for metric, values in results.items():
        print(f"\n{metric}:")
        print(f"  Mean: {values['mean']:.4f} (± {values['std']:.4f})")
        print(f"  Note: {values['interpretation']}")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Image Quality Assessment Tool')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to input/generated images folder')
    parser.add_argument('--reference', '-r', type=str, default=None,
                        help='Path to reference/ground-truth images folder')
    parser.add_argument('--mode', '-m', type=str, default='all',
                        choices=['full', 'no-ref', 'fid', 'all'],
                        help='Evaluation mode: full (PSNR/SSIM/LPIPS), no-ref (NIQE/BRISQUE), fid, all')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path to save results as JSON')
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input):
        raise ValueError(f"Input folder not found: {args.input}")
    
    if args.mode in ['full', 'fid', 'all'] and args.reference is None:
        raise ValueError(f"Reference folder required for mode: {args.mode}")
    
    if args.reference and not os.path.exists(args.reference):
        raise ValueError(f"Reference folder not found: {args.reference}")
    
    all_results = {}
    
    # Full-Reference Metrics
    if args.mode in ['full', 'all']:
        print("\n[1/3] Calculating Full-Reference Metrics (PSNR, SSIM, LPIPS)...")
        fr_results = evaluate_full_reference(args.input, args.reference)
        all_results.update(fr_results)
        print_results(fr_results, "- Full Reference")
    
    # No-Reference Metrics
    if args.mode in ['no-ref', 'all']:
        print("\n[2/3] Calculating No-Reference Metrics (NIQE, BRISQUE)...")
        nr_results = evaluate_no_reference(args.input)
        all_results.update(nr_results)
        print_results(nr_results, "- No Reference")
    
    # FID
    if args.mode in ['fid', 'all']:
        print("\n[3/3] Calculating FID (Fréchet Inception Distance)...")
        fid_value = calculate_fid(args.input, args.reference)
        if fid_value is not None:
            all_results['FID'] = {
                'mean': fid_value,
                'std': 0.0,
                'interpretation': 'lower is better'
            }
            print(f"\nFID: {fid_value:.4f} (lower is better)")
    
    # Print all results
    if args.mode == 'all':
        print_results(all_results, "- ALL METRICS")
    
    # Save results
    if args.output:
        import json
        # Convert numpy types to Python types for JSON
        results_json = {}
        for k, v in all_results.items():
            results_json[k] = {
                'mean': float(v['mean']),
                'std': float(v['std']),
                'interpretation': v['interpretation']
            }
        with open(args.output, 'w') as f:
            json.dump(results_json, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    return all_results


if __name__ == '__main__':
    main()
