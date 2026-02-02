import math

def get_adaptive_conf_thres(epoch, total_epochs, base_conf=0.35, min_conf=None, max_conf=0.7):
    """
    Adaptive confidence threshold với curriculum learning.
    Bắt đầu từ base_conf rồi tăng dần đến max_conf để lọc chất lượng cao hơn.
    
    Args:
        epoch: Current epoch
        total_epochs: Total epochs
        base_conf: Base confidence threshold (start of training) - from CLI --conf-thres
        min_conf: Deprecated, use base_conf instead
        max_conf: Maximum confidence threshold (end of training)
    
    Returns:
        conf_thres: Confidence threshold for current epoch
    """
    # Use base_conf as the starting point
    start_conf = base_conf if min_conf is None else min_conf
    
    progress = epoch / max(total_epochs, 1)
    # Curriculum learning: start from base_conf, increase to max_conf
    # cos(0) = 1, cos(pi) = -1
    # (1 - cos(pi * progress)) / 2: 0 -> 1
    conf_thres = start_conf + (max_conf - start_conf) * (1 - math.cos(math.pi * progress)) / 2
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