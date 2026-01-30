import math

def get_adaptive_conf_thres(epoch, total_epochs, base_conf=0.5, min_conf=0.15, max_conf=0.5):
    """
    Adaptive confidence threshold với curriculum learning.
    Bắt đầu thấp (lấy nhiều pseudo-labels) rồi tăng dần (lọc chất lượng cao hơn).
    
    Args:
        epoch: Current epoch
        total_epochs: Total epochs
        base_conf: Base confidence (unused, kept for compatibility)
        min_conf: Minimum confidence threshold (start of training)
        max_conf: Maximum confidence threshold (end of training)
    
    Returns:
        conf_thres: Confidence threshold for current epoch
    """
    progress = epoch / max(total_epochs, 1)
    # Curriculum learning: start low, increase gradually
    # cos(0) = 1, cos(pi) = -1
    # (1 - cos(pi * progress)) / 2: 0 -> 1
    conf_thres = min_conf + (max_conf - min_conf) * (1 - math.cos(math.pi * progress)) / 2
    return conf_thres

def filter_pseudo_labels_by_uncertainty(predictions, min_confidence=0.3):
    """
    Filter pseudo-labels by confidence score.
    Giữ lại predictions có confidence >= min_confidence.
    
    Args:
        predictions: List of predictions from NMS [x1,y1,x2,y2,conf,cls]
        min_confidence: Minimum confidence to keep (default 0.3)
    
    Returns:
        filtered_preds: List of filtered predictions
    """
    filtered_preds = []
    for pred in predictions:
        if pred is not None and len(pred) > 0:
            # pred[:, 4] is confidence score
            confidence = pred[:, 4]
            mask = confidence >= min_confidence
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