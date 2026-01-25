"""
Feature Space Visualization (UMAP + t-SNE)
==========================================
Visualize domain alignment in 2D feature space.

UMAP: Fast, preserves global + local structure
t-SNE: Standard in papers, validates UMAP findings

Usage:
    from utils.explainability.feature_viz import FeatureVisualizer
    
    viz = FeatureVisualizer(save_dir='runs/exp/explainability')
    viz.plot_umap(features_dict, labels_dict, epoch=50)
    viz.plot_tsne(features_dict, labels_dict, epoch=50)
"""

import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

# Import visualization libraries with fallbacks
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[Warning] matplotlib not installed. Run: pip install matplotlib")

try:
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[Warning] scikit-learn not installed. Run: pip install scikit-learn")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("[Warning] umap-learn not installed. Run: pip install umap-learn")


# Color schemes
DOMAIN_COLORS = {
    'SR': '#2E86AB',   # Blue - Source Real
    'SF': '#7EBDC2',   # Light Blue - Source Fake
    'TR': '#F18F01',   # Orange - Target Real
    'TF': '#FFBA49',   # Light Orange - Target Fake
}

CLASS_COLORS = plt.cm.tab10.colors if MATPLOTLIB_AVAILABLE else None


def prepare_features(features, max_samples=1000):
    """
    Prepare features for visualization.
    
    Args:
        features: Tensor [B, C, H, W] or [B, D] or numpy array
        max_samples: Maximum samples to visualize (for speed)
    
    Returns:
        numpy array [N, D]
    """
    if features is None:
        return None
    
    # Convert to numpy
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    
    # Flatten if needed
    if features.ndim > 2:
        features = features.reshape(features.shape[0], -1)
    
    # Subsample if too large
    if len(features) > max_samples:
        indices = np.random.choice(len(features), max_samples, replace=False)
        features = features[indices]
    
    return features.astype(np.float32)


class FeatureVisualizer:
    """
    Visualize feature space using UMAP and t-SNE.
    
    Produces plots showing:
    - Domain distribution (SR, SF, TR, TF)
    - Class distribution (if labels provided)
    - Alignment status
    """
    
    def __init__(self, save_dir: str = 'explainability'):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # UMAP reducer (reuse for consistency)
        self._umap_reducer = None
    
    def _combine_features(
        self,
        features_sr=None,
        features_sf=None,
        features_tr=None,
        features_tf=None,
        labels_sr=None,
        labels_tr=None,
        max_per_domain=500,
    ):
        """Combine features from all domains with labels."""
        all_features = []
        all_domains = []
        all_classes = []
        
        domains = [
            ('SR', features_sr, labels_sr),
            ('SF', features_sf, labels_sr),  # SF uses SR labels
            ('TR', features_tr, labels_tr),
            ('TF', features_tf, labels_tr),  # TF uses TR labels
        ]
        
        for domain_name, features, labels in domains:
            features = prepare_features(features, max_per_domain)
            if features is None or len(features) == 0:
                continue
            
            n = len(features)
            all_features.append(features)
            all_domains.extend([domain_name] * n)
            
            if labels is not None:
                if isinstance(labels, torch.Tensor):
                    labels = labels.detach().cpu().numpy()
                labels = np.asarray(labels).flatten()
                if len(labels) == n:
                    all_classes.extend(labels.tolist())
                elif len(labels) > n:
                    all_classes.extend(labels[:n].tolist())
                else:
                    all_classes.extend(labels.tolist() + [-1] * (n - len(labels)))
            else:
                all_classes.extend([-1] * n)
        
        if not all_features:
            return None, None, None
        
        combined = np.vstack(all_features)
        return combined, np.array(all_domains), np.array(all_classes)
    
    def plot_umap(
        self,
        features_sr=None,
        features_sf=None,
        features_tr=None,
        features_tf=None,
        labels_sr=None,
        labels_tr=None,
        epoch: int = 0,
        title: str = None,
    ):
        """
        Generate UMAP visualization.
        
        Args:
            features_*: Feature tensors for each domain
            labels_*: Class labels (optional)
            epoch: Current epoch (for filename)
            title: Custom title
        
        Returns:
            save_path: Path to saved figure
        """
        if not UMAP_AVAILABLE or not MATPLOTLIB_AVAILABLE:
            print("[UMAP] Missing dependencies. Install: pip install umap-learn matplotlib")
            return None
        
        # Combine features
        combined, domains, classes = self._combine_features(
            features_sr, features_sf, features_tr, features_tf,
            labels_sr, labels_tr
        )
        
        if combined is None:
            print("[UMAP] No features to visualize")
            return None
        
        print(f"[UMAP] Projecting {len(combined)} samples...")
        
        # UMAP projection
        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            metric='euclidean',
            random_state=42,
        )
        
        try:
            embedding = reducer.fit_transform(combined)
        except Exception as e:
            print(f"[UMAP] Error: {e}")
            return None
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Domain coloring
        for domain in np.unique(domains):
            mask = domains == domain
            ax1.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=DOMAIN_COLORS.get(domain, 'gray'),
                label=domain,
                alpha=0.6,
                s=20,
            )
        
        ax1.set_title('Domain Distribution')
        ax1.legend(loc='upper right')
        ax1.set_xlabel('UMAP-1')
        ax1.set_ylabel('UMAP-2')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Class coloring (if available)
        unique_classes = np.unique(classes[classes >= 0])
        if len(unique_classes) > 0:
            for i, cls in enumerate(unique_classes):
                mask = classes == cls
                color = CLASS_COLORS[i % len(CLASS_COLORS)] if CLASS_COLORS else 'gray'
                ax2.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    c=[color],
                    label=f'Class {int(cls)}',
                    alpha=0.6,
                    s=20,
                )
            ax2.set_title('Class Distribution')
            ax2.legend(loc='upper right', fontsize=8, ncol=2)
        else:
            ax2.scatter(embedding[:, 0], embedding[:, 1], c='gray', alpha=0.5, s=20)
            ax2.set_title('Class Distribution (no labels)')
        
        ax2.set_xlabel('UMAP-1')
        ax2.set_ylabel('UMAP-2')
        ax2.grid(True, alpha=0.3)
        
        # Title
        epoch_title = title or f'UMAP Feature Space - Epoch {epoch}'
        fig.suptitle(epoch_title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        save_path = self.save_dir / f'umap_epoch_{epoch:03d}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[UMAP] Saved: {save_path}")
        return str(save_path)
    
    def plot_tsne(
        self,
        features_sr=None,
        features_sf=None,
        features_tr=None,
        features_tf=None,
        labels_sr=None,
        labels_tr=None,
        epoch: int = 0,
        perplexity: int = 30,
    ):
        """
        Generate t-SNE visualization (for validation).
        
        Args:
            features_*: Feature tensors
            labels_*: Class labels (optional)
            epoch: Current epoch
            perplexity: t-SNE perplexity parameter
        
        Returns:
            save_path: Path to saved figure
        """
        if not SKLEARN_AVAILABLE or not MATPLOTLIB_AVAILABLE:
            print("[t-SNE] Missing dependencies")
            return None
        
        # Combine features
        combined, domains, classes = self._combine_features(
            features_sr, features_sf, features_tr, features_tf,
            labels_sr, labels_tr,
            max_per_domain=300,  # Smaller for t-SNE (slower)
        )
        
        if combined is None:
            print("[t-SNE] No features to visualize")
            return None
        
        print(f"[t-SNE] Projecting {len(combined)} samples...")
        
        # t-SNE projection
        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, len(combined) - 1),
            n_iter=1000,
            random_state=42,
        )
        
        try:
            embedding = tsne.fit_transform(combined)
        except Exception as e:
            print(f"[t-SNE] Error: {e}")
            return None
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Domain coloring
        for domain in np.unique(domains):
            mask = domains == domain
            ax1.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=DOMAIN_COLORS.get(domain, 'gray'),
                label=domain,
                alpha=0.6,
                s=25,
            )
        
        ax1.set_title('Domain Distribution')
        ax1.legend(loc='upper right')
        ax1.set_xlabel('t-SNE-1')
        ax1.set_ylabel('t-SNE-2')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Class coloring
        unique_classes = np.unique(classes[classes >= 0])
        if len(unique_classes) > 0:
            for i, cls in enumerate(unique_classes):
                mask = classes == cls
                color = CLASS_COLORS[i % len(CLASS_COLORS)] if CLASS_COLORS else 'gray'
                ax2.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    c=[color],
                    label=f'Class {int(cls)}',
                    alpha=0.6,
                    s=25,
                )
            ax2.legend(loc='upper right', fontsize=8, ncol=2)
        else:
            ax2.scatter(embedding[:, 0], embedding[:, 1], c='gray', alpha=0.5, s=25)
        
        ax2.set_title('Class Distribution')
        ax2.set_xlabel('t-SNE-1')
        ax2.set_ylabel('t-SNE-2')
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(f't-SNE Feature Space - Epoch {epoch}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        save_path = self.save_dir / f'tsne_epoch_{epoch:03d}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[t-SNE] Saved: {save_path}")
        return str(save_path)


def demo():
    """Generate demo visualizations with synthetic data."""
    if not MATPLOTLIB_AVAILABLE:
        print("Cannot run demo without matplotlib")
        return
    
    print("Generating demo feature visualizations...")
    
    # Create synthetic features
    np.random.seed(42)
    n = 200
    
    # Source features (centered at origin)
    features_sr = np.random.randn(n, 64) * 0.5
    features_sf = features_sr + np.random.randn(n, 64) * 0.2  # Similar to SR
    
    # Target features (shifted)
    features_tr = np.random.randn(n, 64) * 0.5 + 2  # Shifted
    features_tf = features_tr + np.random.randn(n, 64) * 0.2
    
    # Class labels (3 classes)
    labels_sr = np.random.randint(0, 3, n)
    labels_tr = np.random.randint(0, 3, n)
    
    # Visualize
    viz = FeatureVisualizer(save_dir='demo_explainability')
    
    if UMAP_AVAILABLE:
        viz.plot_umap(
            features_sr, features_sf, features_tr, features_tf,
            labels_sr, labels_tr,
            epoch=0, title='Demo: Before Alignment'
        )
        
        # "After alignment" - make TR closer to SR
        features_tr_aligned = features_sr + np.random.randn(n, 64) * 0.3
        features_tf_aligned = features_tr_aligned + np.random.randn(n, 64) * 0.2
        
        viz.plot_umap(
            features_sr, features_sf, features_tr_aligned, features_tf_aligned,
            labels_sr, labels_tr,
            epoch=100, title='Demo: After Alignment'
        )
    else:
        print("Skipping UMAP demo (not installed)")
    
    if SKLEARN_AVAILABLE:
        viz.plot_tsne(
            features_sr, features_sf, features_tr, features_tf,
            labels_sr, labels_tr,
            epoch=0
        )
    else:
        print("Skipping t-SNE demo (not installed)")
    
    print("\nDemo complete!")


if __name__ == '__main__':
    demo()
