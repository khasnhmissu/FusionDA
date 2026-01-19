import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# GRADIENT REVERSAL LAYER
# ============================================================================
class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer - Core của Domain Adversarial Training.
    Forward: identity
    Backward: gradient * (-alpha)
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
    
class GradientReversalLayer(nn.Module):
    """Wrapper cho Gradient Reversal Function"""
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)
    
# ============================================================================
# DOMAIN DISCRIMINATOR
# ============================================================================
class DomainDiscriminator(nn.Module):
    """
    Domain Discriminator cho YOLOv8.
    Nhận features từ backbone và phân loại source/target domain.
    
    GRL được tích hợp bên trong - gradient tự động đảo ngược về backbone.
    """
    def __init__(self, in_channels=512, hidden_dim=256, dropout=0.3):
        super().__init__()
        
        # GRL tích hợp
        self.grl = GradientReversalLayer(alpha=1.0)
        
        # Lightweight discriminator
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
        """Initialize weights with small values for stable training"""
        for m in self.discriminator.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, alpha=None):
        """
        Args:
            x: Feature maps từ backbone [B, C, H, W]
            alpha: GRL alpha value (optional)
        
        Returns:
            domain_pred: [B, 1] logits (source=1, target=0)
        """
        if alpha is not None:
            self.grl.alpha = alpha
        
        x = self.grl(x)
        return self.discriminator(x)
# ============================================================================
# YOLOV8 FEATURE HOOK
# ============================================================================
class YOLOv8FeatureHook:
    """
    Hook để extract features từ YOLOv8 backbone.
    
    QUAN TRỌNG: Mỗi instance chỉ lưu 1 feature map tại 1 thời điểm.
    Để lấy source và target features, cần forward 2 lần riêng biệt.
    
    Usage:
        # For DetectionModel directly
        hook = YOLOv8FeatureHook(model, layer_idx=9)
        
        # Forward source
        _ = model(source_imgs)
        source_features = hook.get_features()
        
        # Forward target  
        _ = model(target_imgs)
        target_features = hook.get_features()
    
    Layer indices for YOLOv8:
        - Layer 4: C2f (P3 features) 
        - Layer 6: C2f (P4 features)
        - Layer 9: SPPF (backbone output) <- Default
    """
    
    def __init__(self, model, layer_idx=9):
        """
        Args:
            model: YOLOv8 DetectionModel OR YOLO wrapper
            layer_idx: Index của layer cần hook (9 = backbone output)
        """
        self.features = None
        self._hook_handle = None
        self.layer_idx = layer_idx
        
        self._register_hook(model)
    
    def _register_hook(self, model):
        """Đăng ký hook vào layer"""
        try:
            # Handle different model wrapper patterns
            # Pattern 1: Direct DetectionModel
            if hasattr(model, 'model') and isinstance(model.model, nn.Sequential):
                target_layer = model.model[self.layer_idx]
            # Pattern 2: DetectionModel without Sequential wrapper
            elif hasattr(model, 'model') and hasattr(model.model, 'model'):
                target_layer = model.model.model[self.layer_idx]
            # Pattern 3: nn.Sequential directly  
            elif isinstance(model, nn.Sequential):
                target_layer = model[self.layer_idx]
            else:
                # Fallback: try direct access
                target_layer = model.model[self.layer_idx]
            
            self._hook_handle = target_layer.register_forward_hook(self._hook_fn)
            print(f"[GRL] Registered hook at layer {self.layer_idx}: {target_layer.__class__.__name__}")
            
        except Exception as e:
            print(f"[GRL Warning] Cannot register hook at layer {self.layer_idx}: {e}")
            print(f"[GRL] Model type: {type(model)}")
            if hasattr(model, 'model'):
                print(f"[GRL] model.model type: {type(model.model)}")
            self._hook_handle = None
    
    def _hook_fn(self, module, input, output):
        """Hook function - LƯU FEATURES VỚI GRADIENT"""
        # QUAN TRỌNG: Không .detach() để giữ gradient flow
        self.features = output
    
    def get_features(self):
        """
        Lấy features và clear buffer.
        
        Returns:
            features: [B, C, H, W] tensor với gradient
        """
        features = self.features
        self.features = None  # Clear để tránh memory leak
        return features
    
    def remove(self):
        """Xóa hook khi không cần nữa"""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
    
    def __del__(self):
        self.remove()


def get_grl_alpha(epoch, total_epochs, warmup_epochs=10, max_alpha=1.0):
    """
    Tính alpha cho GRL theo epoch (progressive training).
    Matches original implementation.
    
    Args:
        epoch: Current epoch
        total_epochs: Total training epochs  
        warmup_epochs: Warmup period (alpha = 0)
        max_alpha: Maximum alpha value
    
    Returns:
        alpha: GRL multiplier (float)
    """
    if epoch < warmup_epochs:
        return 0.0
    
    # Progressive increase from 0 to max_alpha
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    progress = min(progress, 1.0)
    
    # Sigmoid-like curve for smooth transition
    alpha = max_alpha * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)
    
    return alpha
def compute_domain_loss(domain_pred_source, domain_pred_target):
    """
    Tính Domain Adversarial Loss.
    
    Args:
        domain_pred_source: Predictions cho source domain [B, 1]
        domain_pred_target: Predictions cho target domain [B, 1]
    
    Returns:
        domain_loss: Binary cross entropy loss (scalar)
    """
    batch_size_source = domain_pred_source.size(0)
    batch_size_target = domain_pred_target.size(0)
    
    # Labels: 1 for source, 0 for target
    source_labels = torch.ones(batch_size_source, 1, device=domain_pred_source.device)
    target_labels = torch.zeros(batch_size_target, 1, device=domain_pred_target.device)
    
    # Binary cross entropy with logits
    loss_source = F.binary_cross_entropy_with_logits(
        domain_pred_source, source_labels, reduction='mean'
    )
    loss_target = F.binary_cross_entropy_with_logits(
        domain_pred_target, target_labels, reduction='mean'
    )
    
    domain_loss = (loss_source + loss_target) / 2
    
    return domain_loss
def get_domain_accuracy(domain_pred_source, domain_pred_target):
    """
    Tính accuracy của domain discriminator (để monitoring).
    
    Args:
        domain_pred_source: Predictions cho source domain [B, 1]
        domain_pred_target: Predictions cho target domain [B, 1]
    
    Returns:
        accuracy: float [0, 1]
    """
    with torch.no_grad():
        # Source: predict > 0 là đúng (label = 1)
        source_correct = (domain_pred_source > 0).float().sum()
        # Target: predict <= 0 là đúng (label = 0)
        target_correct = (domain_pred_target <= 0).float().sum()
        
        total = domain_pred_source.size(0) + domain_pred_target.size(0)
        accuracy = (source_correct + target_correct) / total
        
    return accuracy.item()

def create_feature_hook(model, layer_name='backbone'):
    """
    Tạo feature hook cho YOLOv8 model.
    
    Args:
        model: YOLOv8 model
        layer_name: 'backbone' (layer 9) hoặc 'neck' (layer 12)
    
    Returns:
        YOLOv8FeatureHook instance
    """
    layer_idx = 9 if layer_name == 'backbone' else 12
    return YOLOv8FeatureHook(model, layer_idx=layer_idx)