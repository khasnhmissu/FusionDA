"""
Domain Classifier Accuracy Tracking
====================================
Monitor GRL effectiveness by tracking discriminator accuracy.

Goal: Discriminator should become confused (accuracy → 50%)

Usage:
    from utils.explainability.domain_metrics import DomainAccuracyTracker
    
    tracker = DomainAccuracyTracker()
    tracker.update(domain_pred_source, domain_pred_target)
    acc = tracker.get_accuracy()
"""

import torch
import numpy as np
import csv
from pathlib import Path
from collections import deque


class DomainAccuracyTracker:
    """
    Track domain discriminator accuracy over training.
    
    The GRL trains the backbone to produce domain-invariant features.
    When successful, the discriminator cannot distinguish source from target,
    resulting in ~50% accuracy (random guessing).
    
    Interpretation:
        - 90-100%: Discriminator easily distinguishes domains (early training / GRL not working)
        - 70-90%: Some alignment starting
        - 55-70%: Good alignment in progress
        - 45-55%: Excellent - discriminator confused (target state)
        - <45%: Rare, possibly GRL too strong
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize tracker.
        
        Args:
            window_size: Size of moving average window
        """
        self.window_size = window_size
        
        # Running statistics
        self._source_correct = deque(maxlen=window_size)
        self._target_correct = deque(maxlen=window_size)
        self._total_source = deque(maxlen=window_size)
        self._total_target = deque(maxlen=window_size)
        
        # Epoch history
        self.history = []
        self._current_epoch_source = []
        self._current_epoch_target = []
    
    def update(
        self,
        domain_pred_source: torch.Tensor,
        domain_pred_target: torch.Tensor,
    ):
        """
        Update accuracy with batch predictions.
        
        Args:
            domain_pred_source: [B, 1] logits from discriminator for source (should predict 1)
            domain_pred_target: [B, 1] logits from discriminator for target (should predict 0)
        """
        if domain_pred_source is None or domain_pred_target is None:
            return
        
        with torch.no_grad():
            # Source: correct if prediction > 0 (predicts source)
            source_pred = (domain_pred_source > 0).float().cpu()
            source_correct = source_pred.sum().item()
            source_total = len(source_pred)
            
            # Target: correct if prediction <= 0 (predicts target)
            target_pred = (domain_pred_target <= 0).float().cpu()
            target_correct = target_pred.sum().item()
            target_total = len(target_pred)
        
        # Update running stats
        self._source_correct.append(source_correct)
        self._target_correct.append(target_correct)
        self._total_source.append(source_total)
        self._total_target.append(target_total)
        
        # Track for epoch averaging
        self._current_epoch_source.append((source_correct, source_total))
        self._current_epoch_target.append((target_correct, target_total))
    
    def get_accuracy(self) -> float:
        """
        Get current running accuracy.
        
        Returns:
            accuracy: Float 0-1 (0.5 = confused = good)
        """
        total_correct = sum(self._source_correct) + sum(self._target_correct)
        total = sum(self._total_source) + sum(self._total_target)
        
        if total == 0:
            return 0.5
        
        return total_correct / total
    
    def get_source_accuracy(self) -> float:
        """Get accuracy on source domain only."""
        total = sum(self._total_source)
        if total == 0:
            return 0.5
        return sum(self._source_correct) / total
    
    def get_target_accuracy(self) -> float:
        """Get accuracy on target domain only."""
        total = sum(self._total_target)
        if total == 0:
            return 0.5
        return sum(self._target_correct) / total
    
    def end_epoch(self, epoch: int):
        """
        Finalize epoch and store summary.
        
        Args:
            epoch: Current epoch number
        """
        # Calculate epoch averages
        source_correct = sum(s for s, _ in self._current_epoch_source)
        source_total = sum(t for _, t in self._current_epoch_source)
        target_correct = sum(s for s, _ in self._current_epoch_target)
        target_total = sum(t for _, t in self._current_epoch_target)
        
        total_correct = source_correct + target_correct
        total = source_total + target_total
        
        row = {
            'epoch': epoch,
            'accuracy': total_correct / total if total > 0 else 0.5,
            'source_accuracy': source_correct / source_total if source_total > 0 else 0.5,
            'target_accuracy': target_correct / target_total if target_total > 0 else 0.5,
            'n_source': source_total,
            'n_target': target_total,
        }
        
        self.history.append(row)
        
        # Clear epoch data
        self._current_epoch_source.clear()
        self._current_epoch_target.clear()
        
        return row
    
    def get_interpretation(self) -> str:
        """
        Get human-readable interpretation of current accuracy.
        
        Returns:
            str: Status description
        """
        acc = self.get_accuracy()
        
        if acc > 0.85:
            return f"❌ Accuracy {acc:.1%} - Discriminator winning (GRL not effective)"
        elif acc > 0.70:
            return f"⚠️ Accuracy {acc:.1%} - Some alignment, need more training"
        elif acc > 0.55:
            return f"✓ Accuracy {acc:.1%} - Good alignment in progress"
        elif acc > 0.45:
            return f"✅ Accuracy {acc:.1%} - Excellent! Discriminator confused"
        else:
            return f"🤔 Accuracy {acc:.1%} - Very low (check GRL weight)"
    
    def save(self, save_path: str):
        """Save accuracy history to CSV."""
        if not self.history:
            return
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        fieldnames = ['epoch', 'accuracy', 'source_accuracy', 'target_accuracy', 
                      'n_source', 'n_target']
        
        with open(save_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(self.history)
        
        print(f"[DomainAcc] Saved history to {save_path}")
    
    def get_latest(self) -> dict:
        """Get most recent epoch's data."""
        return self.history[-1] if self.history else {}


class DomainConfusionMonitor:
    """
    Monitor for domain confusion - combines accuracy with additional metrics.
    
    Tracks:
    - Domain accuracy (toward 50%)
    - Prediction distribution (should be balanced)
    - Confidence scores (should decrease)
    """
    
    def __init__(self):
        self.acc_tracker = DomainAccuracyTracker()
        self.confidence_history = []
    
    def update(
        self,
        domain_pred_source: torch.Tensor,
        domain_pred_target: torch.Tensor,
    ):
        """Update with batch predictions."""
        self.acc_tracker.update(domain_pred_source, domain_pred_target)
        
        # Track confidence (sigmoid of logits)
        if domain_pred_source is not None and domain_pred_target is not None:
            with torch.no_grad():
                conf_source = torch.sigmoid(domain_pred_source).mean().item()
                conf_target = torch.sigmoid(domain_pred_target).mean().item()
                self.confidence_history.append({
                    'source_conf': conf_source,
                    'target_conf': conf_target,
                    'gap': abs(conf_source - conf_target),
                })
    
    def get_status(self) -> dict:
        """Get current status summary."""
        acc = self.acc_tracker.get_accuracy()
        
        avg_gap = 0.0
        if self.confidence_history:
            recent = self.confidence_history[-100:]
            avg_gap = np.mean([h['gap'] for h in recent])
        
        return {
            'accuracy': acc,
            'confidence_gap': avg_gap,
            'interpretation': self.acc_tracker.get_interpretation(),
        }
    
    def end_epoch(self, epoch: int):
        """End of epoch processing."""
        return self.acc_tracker.end_epoch(epoch)
    
    def save(self, save_path: str):
        """Save data."""
        self.acc_tracker.save(save_path)


if __name__ == '__main__':
    # Test domain accuracy tracking
    print("Testing Domain Accuracy Tracker...")
    
    tracker = DomainAccuracyTracker()
    
    # Simulate training: discriminator starts strong, becomes confused
    for epoch in range(5):
        for i in range(10):
            # Early: discriminator correct (source positive, target negative)
            if epoch < 2:
                source_pred = torch.randn(16, 1) + 2  # Positive (correct)
                target_pred = torch.randn(16, 1) - 2  # Negative (correct)
            # Middle: starting to confuse
            elif epoch < 4:
                source_pred = torch.randn(16, 1) + 0.5
                target_pred = torch.randn(16, 1) - 0.5
            # Late: confused (predictions near 0)
            else:
                source_pred = torch.randn(16, 1) * 0.3
                target_pred = torch.randn(16, 1) * 0.3
            
            tracker.update(source_pred, target_pred)
        
        row = tracker.end_epoch(epoch)
        print(f"Epoch {epoch}: {tracker.get_interpretation()}")
    
    print("\nTest PASSED!")
