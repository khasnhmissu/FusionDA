import math

def get_adaptive_conf_thres(epoch, total_epochs, base_conf=0.5, min_conf=0.25, max_conf=0.7):
    """Adaptive confidence threshold"""
    progress = epoch / total_epochs
    conf_thres = min_conf + (max_conf - min_conf) * 0.5 * (1 + math.cos(math.pi * progress))
    return conf_thres

def filter_pseudo_labels_by_uncertainty(predictions, uncertainty_threshold=0.25):
    """Uncertainty-based filtering"""
    filtered_preds = []
    for pred in predictions:
        if pred is not None and len(pred) > 0:
            uncertainty = 1 - pred[:, 4]
            mask = uncertainty < uncertainty_threshold
            filtered_preds.append(pred[mask])
        else:
            filtered_preds.append(pred)
    return filtered_preds

def get_progressive_lambda(epoch, total_epochs, warmup_epochs=20, min_lambda=0.0, max_lambda=1.0):
    """Progressive lambda weight"""
    if epoch < warmup_epochs:
        progress = epoch / warmup_epochs
        return min_lambda + (max_lambda - min_lambda) * (progress ** 2)
    return max_lambda