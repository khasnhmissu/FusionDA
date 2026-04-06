"""Domain Adaptation components: GRL and Domain Discriminator."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer.
    Forward: identity
    Backward: gradient * (-alpha)
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class GradientReversalLayer(nn.Module):
    """Wrapper for GradientReversalFunction."""
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


class DomainDiscriminator(nn.Module):
    """
    Domain Discriminator for YOLOv8.
    Classifies backbone features as source (1) or target (0).
    GRL is integrated - gradients automatically reversed to backbone.
    """
    def __init__(self, in_channels=512, hidden_dim=256, dropout=0.3):
        super().__init__()
        
        self.grl = GradientReversalLayer(alpha=1.0)
        
        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1, stride=1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.discriminator.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, alpha=None):
        # Safety check for NaN/Inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            return torch.zeros(x.size(0), 1, x.size(2), x.size(3), device=x.device, requires_grad=True)
        
        if alpha is not None:
            self.grl.alpha = alpha
        
        return self.discriminator(self.grl(x))


class YOLOv8FeatureHook:
    """
    Hook để extract features từ backbone cuối (YOLO26s hoặc YOLOv8).

    Dùng find_last_backbone_layer() để auto-detect đúng layer.
    YOLO26s approximate:
    - 8: SPPF
    - 9: C2PSA  <- Recommended (attention-aware features)
    YOLOv8 classic:
    - 4: C2f (P3)
    - 6: C2f (P4)
    - 9: SPPF (backbone output) <- Default
    """
    
    def __init__(self, model, layer_idx=9):
        self.features = None
        self._hook_handle = None
        self.layer_idx = layer_idx
        self._register_hook(model)
    
    def _register_hook(self, model):
        try:
            # Handle different model wrapper patterns
            if hasattr(model, 'model') and isinstance(model.model, nn.Sequential):
                target = model.model[self.layer_idx]
            elif hasattr(model, 'model') and hasattr(model.model, 'model'):
                target = model.model.model[self.layer_idx]
            elif isinstance(model, nn.Sequential):
                target = model[self.layer_idx]
            else:
                target = model.model[self.layer_idx]
            
            self._hook_handle = target.register_forward_hook(self._hook_fn)
            print(f"[GRL] Hook at layer {self.layer_idx}: {target.__class__.__name__}")
            
        except Exception as e:
            print(f"[GRL] Cannot register hook: {e}")
            self._hook_handle = None
    
    def _hook_fn(self, module, input, output):
        self.features = output  # Keep gradient flow
    
    def get_features(self):
        features = self.features
        self.features = None
        return features
    
    def remove(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
    
    def __del__(self):
        try:
            self.remove()
        except:
            pass


def get_grl_alpha(epoch, total_epochs, warmup_epochs=10, max_alpha=1.0):
    """
    Progressive GRL alpha using DANN schedule.
    Returns 0 during warmup, then increases smoothly to max_alpha.
    """
    if epoch < warmup_epochs:
        return 0.0
    
    p = min((epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1), 1.0)
    alpha = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
    return max_alpha * alpha


def compute_domain_loss(domain_pred_source, domain_pred_target):
    """Domain adversarial loss: BCE for classifying source=1, target=0."""
    # Safety check
    if (torch.isnan(domain_pred_source).any() or torch.isnan(domain_pred_target).any() or
        torch.isinf(domain_pred_source).any() or torch.isinf(domain_pred_target).any()):
        return torch.tensor(0.0, device=domain_pred_source.device, requires_grad=True)
    
    source_labels = torch.ones_like(domain_pred_source)
    target_labels = torch.zeros_like(domain_pred_target)
    
    loss_source = F.binary_cross_entropy_with_logits(domain_pred_source, source_labels)
    loss_target = F.binary_cross_entropy_with_logits(domain_pred_target, target_labels)
    
    return (loss_source + loss_target) / 2


def get_domain_accuracy(domain_pred_source, domain_pred_target):
    """Calculate domain discriminator accuracy for monitoring."""
    with torch.no_grad():
        source_correct = (domain_pred_source > 0).float().sum()
        target_correct = (domain_pred_target <= 0).float().sum()
        total = domain_pred_source.numel() + domain_pred_target.numel()
        return ((source_correct + target_correct) / total).item()


def find_last_backbone_layer(model):
    """
    Auto-detect index của layer cuối backbone (SPPF hoặc C2PSA).
    Ưu tiên C2PSA nếu có (YOLO26), fallback về SPPF, fallback về 9.
    Tìm kiếm từ cuối model để lấy layer backbone cuối cùng trước head.
    """
    layers = None
    if hasattr(model, 'model') and isinstance(model.model, nn.Sequential):
        layers = model.model
    elif hasattr(model, 'model') and hasattr(model.model, 'model'):
        layers = model.model.model
    elif isinstance(model, nn.Sequential):
        layers = model
    else:
        try:
            layers = model.model
        except Exception:
            return 9

    # Tìm C2PSA cuối cùng (YOLO26) hoặc SPPF (YOLOv8)
    last_c2psa = last_sppf = -1
    for idx, layer in enumerate(layers):
        name = layer.__class__.__name__
        if name == 'C2PSA':
            last_c2psa = idx
        elif name == 'SPPF':
            last_sppf = idx

    if last_c2psa >= 0:
        return last_c2psa  # YOLO26: C2PSA là layer cuối backbone
    if last_sppf >= 0:
        return last_sppf   # YOLOv8: SPPF là layer cuối backbone
    return 9               # Fallback cứng


def create_feature_hook(model, layer_name='backbone'):
    """Create feature hook for YOLO26/YOLOv8. Auto-detects backbone end layer."""
    layer_idx = find_last_backbone_layer(model)
    print(f"[FeatureHook] Auto-detected backbone end layer: {layer_idx}")
    return YOLOv8FeatureHook(model, layer_idx=layer_idx)