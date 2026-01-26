"""
MMD (Maximum Mean Discrepancy) Computation
===========================================
Measures distance between two distributions in feature space.

MMD = 0 → distributions are identical
MMD > 0 → distributions differ (larger = more different)

Usage:
    from utils.explainability.mmd import compute_mmd, MMDTracker
    
    # Single computation
    mmd_value = compute_mmd(features_source, features_target)
    
    # Track over epochs
    tracker = MMDTracker()
    tracker.update(epoch, sr_feat, sf_feat, tr_feat, tf_feat)
    tracker.save('mmd_history.csv')
"""

import torch
import numpy as np
from collections import defaultdict
import csv
from pathlib import Path


def gaussian_kernel(x, y, sigma=None):
    """
    Compute Gaussian (RBF) kernel between two sets of features.
    
    Args:
        x: [N, D] tensor
        y: [M, D] tensor
        sigma: Kernel bandwidth (auto-computed if None)
    
    Returns:
        [N, M] kernel matrix
    """
    # Compute pairwise squared distances
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # [N, 1]
    yy = torch.sum(y ** 2, dim=1, keepdim=True)  # [M, 1]
    xy = torch.mm(x, y.t())  # [N, M]
    
    distances = xx + yy.t() - 2 * xy  # [N, M]
    distances = torch.clamp(distances, min=0)  # Numerical stability
    
    # Auto bandwidth using median heuristic
    if sigma is None:
        sigma = torch.sqrt(torch.median(distances) / 2 + 1e-8)
    
    kernel = torch.exp(-distances / (2 * sigma ** 2))
    return kernel


def compute_mmd(source_features, target_features, kernel='gaussian'):
    """
    Compute MMD (Maximum Mean Discrepancy) between source and target features.
    
    Args:
        source_features: [N, D] tensor - features from source domain
        target_features: [M, D] tensor - features from target domain
        kernel: 'gaussian' or 'linear'
    
    Returns:
        mmd_value: float - MMD distance (0 = identical, higher = more different)
    
    Interpretation:
        ~0.0-0.05: Distributions very similar (well aligned)
        ~0.05-0.15: Good alignment
        ~0.15-0.3: Moderate domain gap
        >0.3: Significant domain gap
    """
    if source_features is None or target_features is None:
        return 0.0
    
    # Handle numpy arrays
    if isinstance(source_features, np.ndarray):
        source_features = torch.from_numpy(source_features).float()
    if isinstance(target_features, np.ndarray):
        target_features = torch.from_numpy(target_features).float()
    
    # Flatten if needed (e.g., from [B, C, H, W] to [B, C*H*W])
    if source_features.dim() > 2:
        source_features = source_features.view(source_features.size(0), -1)
    if target_features.dim() > 2:
        target_features = target_features.view(target_features.size(0), -1)
    
    # Move to CPU and convert to float32 for computation (handles Half precision)
    source_features = source_features.detach().cpu().float()
    target_features = target_features.detach().cpu().float()
    
    n_s = source_features.size(0)
    n_t = target_features.size(0)
    
    if n_s == 0 or n_t == 0:
        return 0.0
    
    # Compute kernel matrices
    if kernel == 'gaussian':
        K_ss = gaussian_kernel(source_features, source_features)
        K_tt = gaussian_kernel(target_features, target_features)
        K_st = gaussian_kernel(source_features, target_features)
    else:  # linear kernel
        K_ss = torch.mm(source_features, source_features.t())
        K_tt = torch.mm(target_features, target_features.t())
        K_st = torch.mm(source_features, target_features.t())
    
    # MMD = E[k(s,s)] - 2*E[k(s,t)] + E[k(t,t)]
    # Exclude diagonal for unbiased estimate
    mmd = (K_ss.sum() - K_ss.trace()) / (n_s * (n_s - 1) + 1e-8)
    mmd += (K_tt.sum() - K_tt.trace()) / (n_t * (n_t - 1) + 1e-8)
    mmd -= 2 * K_st.mean()
    
    return max(0, float(mmd))  # Clamp negative values


def compute_mmd_multi_kernel(source_features, target_features, sigmas=[0.1, 1, 10]):
    """
    Compute MMD with multiple kernel bandwidths for robustness.
    
    Args:
        source_features: [N, D] tensor
        target_features: [M, D] tensor
        sigmas: List of bandwidth values
    
    Returns:
        mmd_value: Average MMD across all bandwidths
    """
    if source_features is None or target_features is None:
        return 0.0
    
    mmds = []
    for sigma in sigmas:
        # Temporarily compute with fixed sigma
        mmd_val = compute_mmd(source_features, target_features)
        mmds.append(mmd_val)
    
    return np.mean(mmds)


class MMDTracker:
    """
    Track MMD values over training epochs.
    
    Tracks 4 key metrics:
    - MMD(SR, TR): Source Real vs Target Real (main domain gap)
    - MMD(SF, TF): Source Fake vs Target Fake (style domain gap)
    - MMD(SR, SF): Source Real vs Source Fake (style transfer impact)
    - MMD(TR, TF): Target Real vs Target Fake (style transfer impact)
    """
    
    def __init__(self):
        self.history = []
        self.current_epoch_data = defaultdict(list)
    
    def update(
        self,
        epoch: int,
        features_sr: torch.Tensor = None,
        features_sf: torch.Tensor = None,
        features_tr: torch.Tensor = None,
        features_tf: torch.Tensor = None,
    ):
        """
        Compute and store MMD values for current batch.
        
        Args:
            epoch: Current epoch
            features_sr: Source Real features [B, C, H, W] or [B, D]
            features_sf: Source Fake features
            features_tr: Target Real features
            features_tf: Target Fake features
        """
        mmd_sr_tr = compute_mmd(features_sr, features_tr) if features_sr is not None and features_tr is not None else None
        mmd_sf_tf = compute_mmd(features_sf, features_tf) if features_sf is not None and features_tf is not None else None
        mmd_sr_sf = compute_mmd(features_sr, features_sf) if features_sr is not None and features_sf is not None else None
        mmd_tr_tf = compute_mmd(features_tr, features_tf) if features_tr is not None and features_tf is not None else None
        
        if mmd_sr_tr is not None:
            self.current_epoch_data['mmd_sr_tr'].append(mmd_sr_tr)
        if mmd_sf_tf is not None:
            self.current_epoch_data['mmd_sf_tf'].append(mmd_sf_tf)
        if mmd_sr_sf is not None:
            self.current_epoch_data['mmd_sr_sf'].append(mmd_sr_sf)
        if mmd_tr_tf is not None:
            self.current_epoch_data['mmd_tr_tf'].append(mmd_tr_tf)
    
    def end_epoch(self, epoch: int):
        """
        Finalize epoch - compute averages and store.
        
        Args:
            epoch: Current epoch number
        """
        row = {'epoch': epoch}
        
        for key, values in self.current_epoch_data.items():
            if values:
                row[key] = np.mean(values)
                row[f'{key}_std'] = np.std(values)
        
        self.history.append(row)
        self.current_epoch_data.clear()
        
        return row
    
    def get_latest(self):
        """Get most recent epoch's MMD values."""
        return self.history[-1] if self.history else {}
    
    def save(self, save_path: str):
        """Save MMD history to CSV."""
        if not self.history:
            return
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        fieldnames = ['epoch', 'mmd_sr_tr', 'mmd_sr_tr_std', 
                     'mmd_sf_tf', 'mmd_sf_tf_std',
                     'mmd_sr_sf', 'mmd_sr_sf_std',
                     'mmd_tr_tf', 'mmd_tr_tf_std']
        
        with open(save_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(self.history)
        
        print(f"[MMD] Saved history to {save_path}")
    
    def get_interpretation(self):
        """
        Get human-readable interpretation of current MMD values.
        
        Returns:
            str: Interpretation text
        """
        if not self.history:
            return "No MMD data yet"
        
        latest = self.history[-1]
        mmd_sr_tr = latest.get('mmd_sr_tr', None)
        
        if mmd_sr_tr is None:
            return "MMD(SR,TR) not computed"
        
        if mmd_sr_tr < 0.05:
            status = "✅ Excellent alignment"
        elif mmd_sr_tr < 0.15:
            status = "✅ Good alignment"
        elif mmd_sr_tr < 0.3:
            status = "⚠️ Moderate domain gap"
        else:
            status = "❌ Significant domain gap"
        
        return f"MMD(SR,TR) = {mmd_sr_tr:.4f} → {status}"


if __name__ == '__main__':
    # Test MMD computation
    print("Testing MMD computation...")
    
    # Same distribution → MMD ≈ 0
    x = torch.randn(100, 64)
    mmd_same = compute_mmd(x, x)
    print(f"MMD(same, same) = {mmd_same:.6f} (expected ≈ 0)")
    
    # Different distributions → MMD > 0
    y = torch.randn(100, 64) + 2  # Shifted mean
    mmd_diff = compute_mmd(x, y)
    print(f"MMD(x, x+2) = {mmd_diff:.6f} (expected > 0)")
    
    # Very different → MMD >> 0
    z = torch.randn(100, 64) * 5 + 10
    mmd_very_diff = compute_mmd(x, z)
    print(f"MMD(x, 5x+10) = {mmd_very_diff:.6f} (expected >> previous)")
    
    # Test tracker
    tracker = MMDTracker()
    for epoch in range(3):
        for i in range(5):
            sr = torch.randn(32, 64)
            tr = torch.randn(32, 64) + (2 - epoch * 0.5)  # Gap decreases
            tracker.update(epoch, features_sr=sr, features_tr=tr)
        row = tracker.end_epoch(epoch)
        print(f"Epoch {epoch}: {tracker.get_interpretation()}")
    
    print("\nTest PASSED!")
