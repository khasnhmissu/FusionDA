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
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.discriminator.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, alpha=None):
        # Safety check for NaN/Inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            return torch.zeros(x.size(0), 1, device=x.device, requires_grad=True)
        
        if alpha is not None:
            self.grl.alpha = alpha
        
        return self.discriminator(self.grl(x))


class YOLOv8FeatureHook:
    """
    Hook to extract features from YOLOv8 backbone.
    
    Layer indices:
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
    Progressive GRL alpha using quadratic schedule.
    Returns 0 during warmup, then increases to max_alpha.
    """
    if epoch < warmup_epochs:
        return 0.0
    
    progress = min((epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1), 1.0)
    return max_alpha * (progress ** 2)


def compute_domain_loss(domain_pred_source, domain_pred_target):
    """Domain adversarial loss: BCE for classifying source=1, target=0."""
    # Safety check
    if (torch.isnan(domain_pred_source).any() or torch.isnan(domain_pred_target).any() or
        torch.isinf(domain_pred_source).any() or torch.isinf(domain_pred_target).any()):
        return torch.tensor(0.0, device=domain_pred_source.device, requires_grad=True)
    
    # Clamp to prevent extreme gradients
    domain_pred_source = torch.clamp(domain_pred_source, -10, 10)
    domain_pred_target = torch.clamp(domain_pred_target, -10, 10)
    
    source_labels = torch.ones(domain_pred_source.size(0), 1, device=domain_pred_source.device)
    target_labels = torch.zeros(domain_pred_target.size(0), 1, device=domain_pred_target.device)
    
    loss_source = F.binary_cross_entropy_with_logits(domain_pred_source, source_labels)
    loss_target = F.binary_cross_entropy_with_logits(domain_pred_target, target_labels)
    
    return (loss_source + loss_target) / 2


def get_domain_accuracy(domain_pred_source, domain_pred_target):
    """Calculate domain discriminator accuracy for monitoring."""
    with torch.no_grad():
        source_correct = (domain_pred_source > 0).float().sum()
        target_correct = (domain_pred_target <= 0).float().sum()
        total = domain_pred_source.size(0) + domain_pred_target.size(0)
        return ((source_correct + target_correct) / total).item()


def create_feature_hook(model, layer_name='backbone'):
    """Create feature hook for YOLOv8."""
    layer_idx = 9 if layer_name == 'backbone' else 12
    return YOLOv8FeatureHook(model, layer_idx=layer_idx)