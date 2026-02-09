"""Helper functions for FDA training."""
import math


def get_adaptive_conf_thres(epoch, total_epochs, base_conf=0.5, min_conf=None, max_conf=0.7):
    """
    Adaptive confidence threshold with curriculum learning.
    Starts at base_conf and increases to max_conf over training.
    """
    start_conf = base_conf if min_conf is None else min_conf
    progress = epoch / max(total_epochs, 1)
    # Cosine schedule: smooth increase from start to max
    return start_conf + (max_conf - start_conf) * (1 - math.cos(math.pi * progress)) / 2


def filter_pseudo_labels_by_uncertainty(predictions, min_confidence=0.3):
    """Filter pseudo-labels keeping only predictions with confidence >= min_confidence."""
    filtered = []
    for pred in predictions:
        if pred is not None and len(pred) > 0:
            mask = pred[:, 4] >= min_confidence
            filtered.append(pred[mask])
        else:
            filtered.append(pred)
    return filtered


def get_progressive_lambda(epoch, total_epochs, warmup_epochs=20, min_lambda=0.0, max_lambda=1.0):
    """Progressive lambda weight: quadratic increase during warmup, then constant."""
    if epoch < warmup_epochs:
        progress = epoch / warmup_epochs
        return min_lambda + (max_lambda - min_lambda) * (progress ** 2)
    return max_lambda