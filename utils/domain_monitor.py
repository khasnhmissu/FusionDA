"""
Unified Domain Adaptation Monitor
==================================
Single class integrating all 4 explainability methods.

Components:
- UMAP/t-SNE: Visual feature space analysis
- MMD: Quantitative domain gap measurement
- Domain Accuracy: GRL effectiveness tracking

Usage:
    from utils.domain_monitor import DomainMonitor
    
    monitor = DomainMonitor(save_dir='runs/exp')
    
    # Per iteration
    monitor.update_domain_accuracy(domain_pred_source, domain_pred_target, epoch, i)
    
    # At specified epochs
    monitor.analyze_features(features_dict, labels_dict, epoch)
    
    # End of training
    monitor.finalize()
"""

import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
import csv

from .explainability.mmd import MMDTracker, compute_mmd
from .explainability.feature_viz import FeatureVisualizer
from .explainability.domain_metrics import DomainAccuracyTracker


class DomainMonitor:
    """
    Unified monitor for domain adaptation training.
    
    Integrates 4 explainability methods:
    1. UMAP - Visual feature space (at specified epochs)
    2. t-SNE - Validation visualization (at key epochs)
    3. MMD - Quantitative domain gap (every epoch)
    4. Domain Accuracy - GRL effectiveness (every iteration)
    
    Outputs:
    - explainability/umap_epoch_XXX.png
    - explainability/tsne_epoch_XXX.png
    - explainability/mmd_history.csv
    - explainability/domain_accuracy.csv
    - explainability/summary_dashboard.png
    """
    
    def __init__(
        self,
        save_dir: str,
        umap_epochs: list = None,
        tsne_epochs: list = None,
        mmd_every_n_epochs: int = 1,
        verbose: bool = True,
    ):
        """
        Initialize domain monitor.
        
        Args:
            save_dir: Base directory for saving outputs
            umap_epochs: Epochs to generate UMAP (e.g., [0, 50, 100, 199])
            tsne_epochs: Epochs to generate t-SNE (e.g., [0, 100, 199])
            mmd_every_n_epochs: Compute MMD every N epochs
            verbose: Print status messages
        """
        self.save_dir = Path(save_dir)
        self.explainability_dir = self.save_dir / 'explainability'
        self.explainability_dir.mkdir(parents=True, exist_ok=True)
        
        # Default schedules
        self.umap_epochs = set(umap_epochs or [0, 25, 50, 100, 150, 199])
        self.tsne_epochs = set(tsne_epochs or [0, 100, 199])
        self.mmd_every_n_epochs = mmd_every_n_epochs
        self.verbose = verbose
        
        # Initialize trackers
        self.mmd_tracker = MMDTracker()
        self.domain_acc_tracker = DomainAccuracyTracker()
        self.feature_viz = FeatureVisualizer(save_dir=str(self.explainability_dir))
        
        # Feature storage for end-of-epoch analysis
        self._epoch_features = defaultdict(list)
        self._epoch_labels = defaultdict(list)
        
        # Status tracking
        self.current_epoch = 0
        self.total_epochs = 200
        
        if verbose:
            print(f"[DomainMonitor] Initialized")
            print(f"  - Save dir: {self.explainability_dir}")
            print(f"  - UMAP epochs: {sorted(self.umap_epochs)}")
            print(f"  - t-SNE epochs: {sorted(self.tsne_epochs)}")
    
    def set_total_epochs(self, total_epochs: int):
        """Set total epochs for proper scheduling."""
        self.total_epochs = total_epochs
        # Auto-adjust schedules if needed
        last_epoch = total_epochs - 1
        if last_epoch not in self.umap_epochs:
            self.umap_epochs.add(last_epoch)
        if last_epoch not in self.tsne_epochs:
            self.tsne_epochs.add(last_epoch)
    
    def update_domain_accuracy(
        self,
        domain_pred_source: torch.Tensor,
        domain_pred_target: torch.Tensor,
        epoch: int = None,
        iteration: int = None,
    ):
        """
        Update domain accuracy tracker with batch predictions.
        
        Call this every iteration when GRL is active.
        
        Args:
            domain_pred_source: [B, 1] logits for source domain
            domain_pred_target: [B, 1] logits for target domain
            epoch: Current epoch (for logging)
            iteration: Current iteration (for logging)
        """
        if domain_pred_source is None or domain_pred_target is None:
            return
        
        self.domain_acc_tracker.update(domain_pred_source, domain_pred_target)
        
        if epoch is not None:
            self.current_epoch = epoch
    
    def collect_features(
        self,
        features_sr: torch.Tensor = None,
        features_sf: torch.Tensor = None,
        features_tr: torch.Tensor = None,
        features_tf: torch.Tensor = None,
        labels_sr: torch.Tensor = None,
        labels_tr: torch.Tensor = None,
    ):
        """
        Collect features for end-of-epoch analysis.
        
        Call this for a few batches per epoch to accumulate features.
        
        Args:
            features_*: Feature tensors [B, C, H, W] or [B, D]
            labels_*: Class labels [B]
        """
        if features_sr is not None:
            self._epoch_features['sr'].append(features_sr.detach().cpu())
        if features_sf is not None:
            self._epoch_features['sf'].append(features_sf.detach().cpu())
        if features_tr is not None:
            self._epoch_features['tr'].append(features_tr.detach().cpu())
        if features_tf is not None:
            self._epoch_features['tf'].append(features_tf.detach().cpu())
        
        if labels_sr is not None:
            self._epoch_labels['sr'].append(labels_sr.detach().cpu() if isinstance(labels_sr, torch.Tensor) else torch.tensor(labels_sr))
        if labels_tr is not None:
            self._epoch_labels['tr'].append(labels_tr.detach().cpu() if isinstance(labels_tr, torch.Tensor) else torch.tensor(labels_tr))
    
    def _get_collected_features(self):
        """Concatenate collected features."""
        result = {}
        for key, tensors in self._epoch_features.items():
            if tensors:
                result[key] = torch.cat(tensors, dim=0)
        return result
    
    def _get_collected_labels(self):
        """Concatenate collected labels."""
        result = {}
        for key, tensors in self._epoch_labels.items():
            if tensors:
                result[key] = torch.cat(tensors, dim=0)
        return result
    
    def end_epoch(self, epoch: int):
        """
        End of epoch processing.
        
        - Computes MMD
        - Generates UMAP/t-SNE if scheduled
        - Saves domain accuracy
        
        Args:
            epoch: Current epoch number
        """
        self.current_epoch = epoch
        
        # Get collected features
        features = self._get_collected_features()
        labels = self._get_collected_labels()
        
        # 1. Compute MMD
        if features and epoch % self.mmd_every_n_epochs == 0:
            self.mmd_tracker.update(
                epoch,
                features.get('sr'),
                features.get('sf'),
                features.get('tr'),
                features.get('tf'),
            )
            mmd_result = self.mmd_tracker.end_epoch(epoch)
            
            if self.verbose:
                print(f"[MMD] {self.mmd_tracker.get_interpretation()}")
        
        # 2. Domain accuracy
        domain_result = self.domain_acc_tracker.end_epoch(epoch)
        if self.verbose:
            print(f"[DomainAcc] {self.domain_acc_tracker.get_interpretation()}")
        
        # 3. UMAP visualization
        if epoch in self.umap_epochs and features:
            if self.verbose:
                print(f"[UMAP] Generating epoch {epoch} visualization...")
            self.feature_viz.plot_umap(
                features.get('sr'),
                features.get('sf'),
                features.get('tr'),
                features.get('tf'),
                labels.get('sr'),
                labels.get('tr'),
                epoch=epoch,
            )
        
        # 4. t-SNE visualization
        if epoch in self.tsne_epochs and features:
            if self.verbose:
                print(f"[t-SNE] Generating epoch {epoch} visualization...")
            self.feature_viz.plot_tsne(
                features.get('sr'),
                features.get('sf'),
                features.get('tr'),
                features.get('tf'),
                labels.get('sr'),
                labels.get('tr'),
                epoch=epoch,
            )
        
        # Clear collected features
        self._epoch_features.clear()
        self._epoch_labels.clear()
        
        return {
            'mmd': self.mmd_tracker.get_latest(),
            'domain_acc': domain_result,
        }
    
    def get_status_string(self) -> str:
        """
        Get current status as formatted string.
        
        Returns:
            str: Multi-line status summary
        """
        lines = [
            f"=== Domain Adaptation Status (Epoch {self.current_epoch}) ===",
            self.mmd_tracker.get_interpretation(),
            self.domain_acc_tracker.get_interpretation(),
        ]
        return "\n".join(lines)
    
    def generate_summary_dashboard(self):
        """
        Generate summary dashboard with all metrics.
        
        Creates a combined plot showing:
        - MMD timeline
        - Domain accuracy timeline
        - Status interpretation
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[DomainMonitor] matplotlib not available for dashboard")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: MMD(SR, TR) over time
        ax1 = axes[0, 0]
        if self.mmd_tracker.history:
            epochs = [h['epoch'] for h in self.mmd_tracker.history]
            mmd_values = [h.get('mmd_sr_tr', 0) for h in self.mmd_tracker.history]
            ax1.plot(epochs, mmd_values, 'b-o', linewidth=2, markersize=4)
            ax1.axhline(y=0.15, color='g', linestyle='--', label='Good alignment threshold')
            ax1.axhline(y=0.3, color='r', linestyle='--', label='Domain gap threshold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MMD(SR, TR)')
        ax1.set_title('Domain Gap (MMD)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: All MMD values
        ax2 = axes[0, 1]
        if self.mmd_tracker.history:
            epochs = [h['epoch'] for h in self.mmd_tracker.history]
            for key, label, color in [
                ('mmd_sr_tr', 'SR-TR (main)', 'blue'),
                ('mmd_sf_tf', 'SF-TF', 'cyan'),
                ('mmd_sr_sf', 'SR-SF (style)', 'green'),
                ('mmd_tr_tf', 'TR-TF (style)', 'orange'),
            ]:
                values = [h.get(key, 0) for h in self.mmd_tracker.history]
                if any(v > 0 for v in values):
                    ax2.plot(epochs, values, '-o', label=label, color=color, 
                            linewidth=1.5, markersize=3)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MMD')
        ax2.set_title('All MMD Values')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Domain accuracy
        ax3 = axes[1, 0]
        if self.domain_acc_tracker.history:
            epochs = [h['epoch'] for h in self.domain_acc_tracker.history]
            acc_values = [h.get('accuracy', 0.5) * 100 for h in self.domain_acc_tracker.history]
            ax3.plot(epochs, acc_values, 'r-o', linewidth=2, markersize=4)
            ax3.axhline(y=50, color='g', linestyle='--', label='Target (confused)')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy (%)')
        ax3.set_title('Domain Discriminator Accuracy')
        ax3.set_ylim(0, 100)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Status text
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        status_text = self.get_status_string()
        
        # Add interpretation guide
        guide = """
Decision Guide:
✅ Good: MMD(SR,TR) < 0.15 AND DomainAcc ~ 50%
⚠️ Check: MMD > 0.3 OR DomainAcc > 70%
❌ Issue: MMD not decreasing OR DomainAcc > 85%

Expected Timeline:
• Epoch 0-20: MMD ~0.8, Acc ~90%
• Epoch 20-100: MMD decreasing, Acc decreasing
• Epoch 100+: MMD ~0.05-0.15, Acc ~50-55%
"""
        ax4.text(0.1, 0.9, status_text, transform=ax4.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace')
        ax4.text(0.1, 0.4, guide, transform=ax4.transAxes,
                fontsize=9, verticalalignment='top')
        
        fig.suptitle('Domain Adaptation Monitoring Dashboard', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.explainability_dir / 'summary_dashboard.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[DomainMonitor] Dashboard saved: {save_path}")
    
    def finalize(self):
        """
        Finalize monitoring - save all data and generate summary.
        
        Call this at the end of training.
        """
        # Save CSV files
        self.mmd_tracker.save(self.explainability_dir / 'mmd_history.csv')
        self.domain_acc_tracker.save(self.explainability_dir / 'domain_accuracy.csv')
        
        # Generate dashboard
        self.generate_summary_dashboard()
        
        # Save summary JSON
        summary = {
            'total_epochs': self.current_epoch + 1,
            'final_mmd': self.mmd_tracker.get_latest(),
            'final_domain_acc': self.domain_acc_tracker.get_latest(),
            'interpretation': {
                'mmd': self.mmd_tracker.get_interpretation(),
                'domain_acc': self.domain_acc_tracker.get_interpretation(),
            },
            'umap_epochs': sorted(list(self.umap_epochs)),
            'tsne_epochs': sorted(list(self.tsne_epochs)),
        }
        
        with open(self.explainability_dir / 'monitoring_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n[DomainMonitor] Training complete!")
        print(f"  - MMD final: {self.mmd_tracker.get_interpretation()}")
        print(f"  - Domain Acc: {self.domain_acc_tracker.get_interpretation()}")
        print(f"  - All outputs: {self.explainability_dir}")


def create_monitor_from_args(opt, save_dir: str) -> DomainMonitor:
    """
    Create DomainMonitor from training arguments.
    
    Args:
        opt: Training arguments namespace
        save_dir: Save directory
    
    Returns:
        DomainMonitor instance
    """
    # Determine UMAP epochs based on total epochs
    total_epochs = getattr(opt, 'epochs', 200)
    
    if total_epochs <= 50:
        umap_epochs = [0, total_epochs // 2, total_epochs - 1]
        tsne_epochs = [0, total_epochs - 1]
    else:
        step = total_epochs // 4
        umap_epochs = [0, step, step * 2, step * 3, total_epochs - 1]
        tsne_epochs = [0, total_epochs // 2, total_epochs - 1]
    
    monitor = DomainMonitor(
        save_dir=save_dir,
        umap_epochs=umap_epochs,
        tsne_epochs=tsne_epochs,
        verbose=True,
    )
    monitor.set_total_epochs(total_epochs)
    
    return monitor


if __name__ == '__main__':
    # Test unified monitor
    print("Testing DomainMonitor...")
    
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        monitor = DomainMonitor(save_dir=tmpdir, verbose=True)
        monitor.set_total_epochs(5)
        
        # Simulate training
        for epoch in range(5):
            # Simulate iterations
            for i in range(3):
                # Collect features
                sr = torch.randn(16, 64)
                tr = torch.randn(16, 64) + (2 - epoch * 0.3)  # Gap decreases
                sf = sr + torch.randn(16, 64) * 0.1
                tf = tr + torch.randn(16, 64) * 0.1
                
                monitor.collect_features(sr, sf, tr, tf)
                
                # Domain accuracy
                if epoch > 0:  # GRL active after warmup
                    src_pred = torch.randn(16, 1) + (2 - epoch * 0.4)
                    tgt_pred = torch.randn(16, 1) - (2 - epoch * 0.4)
                    monitor.update_domain_accuracy(src_pred, tgt_pred, epoch, i)
            
            # End of epoch
            result = monitor.end_epoch(epoch)
            print(f"\nEpoch {epoch} complete")
        
        # Finalize
        monitor.finalize()
        
        # Check outputs
        from pathlib import Path
        outputs = list(Path(tmpdir).rglob('*'))
        print(f"\nGenerated {len(outputs)} files")
        for f in outputs:
            print(f"  - {f.name}")
    
    print("\nTest PASSED!")
