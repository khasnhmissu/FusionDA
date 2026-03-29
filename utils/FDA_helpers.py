"""Helper functions for FDA training."""
import math


def get_adaptive_conf_thres(epoch, total_epochs, base_conf=0.5, min_conf=None, max_conf=0.7,
                             burn_in_epochs=20):
    """
    Adaptive confidence threshold with curriculum learning.
    Starts at base_conf and increases to max_conf over training.
    After burn-in, the schedule runs over the remaining epochs.
    """
    # During burn-in, return max_conf (high threshold for safety)
    if epoch < burn_in_epochs:
        return max_conf
    
    start_conf = base_conf if min_conf is None else min_conf
    effective_epoch = epoch - burn_in_epochs
    effective_total = max(total_epochs - burn_in_epochs, 1)
    progress = min(effective_epoch / effective_total, 1.0)
    
    # Cosine schedule: smooth increase from start_conf to max_conf
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