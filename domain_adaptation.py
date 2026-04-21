"""Domain Adaptation components: GRL and Domain Discriminator (SOTA).

Changes vs previous version:
- DomainDiscriminator: Conv-spatial → GAP + MLP (stable gradients, correct accuracy)
- compute_domain_loss: switched back to BCE with Label Smoothing, focal loss limits GRL effectiveness
- get_domain_accuracy: Fixed for 2D [B,1] output (no more 16740% bug)
- MultiScaleDomainDiscriminator: P3/P4/P5 multi-scale alignment (SOTA)
- find_multiscale_layers: Auto-detect P3/P4/P5 layer indices
- MultiScaleFeatureHook: Manages multiple hooks simultaneously
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------

class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer (Ganin et al., 2016).
    Forward : identity
    Backward : gradient * (-alpha)
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class GradientReversalLayer(nn.Module):
    """Thin wrapper around GradientReversalFunction."""
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


# ---------------------------------------------------------------------------
# Domain Discriminator  (GAP + MLP  —  SOTA, stable gradients)
# ---------------------------------------------------------------------------

class DomainDiscriminator(nn.Module):
    """
    Domain discriminator for a single feature scale.

    Architecture (SOTA DANN-style):
        Input [B, C, H, W]
          ↓  GRL  (alpha reversal)
          ↓  AdaptiveAvgPool2d(1)   →  [B, C]
          ↓  LayerNorm(C)
          ↓  FC(C → hidden)  + LeakyReLU(0.2)  + Dropout(p)
          ↓  FC(hidden → hidden//4)  + LeakyReLU(0.2)  + Dropout(p)
          ↓  FC(hidden//4 → 1)       →  logits [B, 1]

    Using GAP (instead of spatial conv output) gives:
      • One scalar per image → accuracy calculation is trivially correct
      • Gradient reversal normalised over the whole feature map
      • Smaller discriminator → less prone to overfitting / vanishing GRL grad
    """

    def __init__(self, in_channels: int = 512, hidden_dim: int = 1024,
                 dropout: float = 0.3):
        super().__init__()
        self.grl = GradientReversalLayer(alpha=1.0)

        self.gap = nn.AdaptiveAvgPool2d(1)   # [B, C, H, W] → [B, C, 1, 1]

        mid = max(hidden_dim // 4, 64)
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, mid),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(mid, 1),   # raw logit
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor, alpha: float = None) -> torch.Tensor:
        """
        Args:
            x     : [B, C, H, W]  backbone feature map
            alpha : GRL scale (overrides self.grl.alpha if provided)
        Returns:
            logits: [B, 1]
        """
        if torch.isnan(x).any() or torch.isinf(x).any():
            return torch.zeros(x.size(0), 1, device=x.device)

        if alpha is not None:
            self.grl.alpha = alpha

        x = self.grl(x)                  # gradient reversal
        x = self.gap(x).flatten(1)       # [B, C]
        return self.classifier(x)        # [B, 1]


# ---------------------------------------------------------------------------
# Multi-Scale Domain Discriminator  — two implementations
#
#   MultiScaleDomainDiscriminator  (DEPRECATED)
#     N independent discriminator heads → backbone must fool all N → hard
#     keeps domain_acc high; use only for ablation.
#
#   MultiScaleFusedDiscriminator   (SOTA / RECOMMENDED)
#     MF-DANN style: GAP each scale → concat → ONE shared MLP.
#     Backbone only needs to fool a single discriminator → domain confusion
#     is achievable and domain_acc decreases to ~0.5 as expected.
# ---------------------------------------------------------------------------

class MultiScaleDomainDiscriminator(nn.Module):
    """
    [DEPRECATED — use MultiScaleFusedDiscriminator instead]

    N independent heads, one per scale.  Backbone must fool ALL N simultaneously
    which is structurally very hard → domain_acc stays high permanently.
    Kept for ablation / backward-compatibility only.
    """

    def __init__(self, in_channels_list: list, hidden_dim: int = 512,
                 dropout: float = 0.3, scale_weights: list = None):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DomainDiscriminator(ch, hidden_dim=hidden_dim, dropout=dropout)
            for ch in in_channels_list
        ])
        n = len(in_channels_list)
        if scale_weights is not None:
            self.scale_weights = scale_weights
        elif n == 3:
            self.scale_weights = [0.5, 0.3, 0.2]
        else:
            self.scale_weights = [1.0 / n] * n

    def set_alpha(self, alpha: float):
        for d in self.discriminators:
            d.grl.alpha = alpha

    def forward(self, features_list: list, alpha: float = None):
        return [d(f, alpha) for d, f in zip(self.discriminators, features_list)]


class MultiScaleFusedDiscriminator(nn.Module):
    """
    MF-DANN style multi-scale domain discriminator — P3-biased for small objects.

    Architecture (P3-biased):
        P3 [B,C3] →GRL(α×1.5)→ GAP → Bottleneck(256) ─┐
        P4 [B,C4] →GRL(α×1.0)→ GAP → Bottleneck(128) ─┼─ cat[B,448] → MLP → [B,1]
        P5 [B,C5] →GRL(α×0.5)→ GAP → Bottleneck(64)  ─┘

    Two P3-biasing mechanisms (targeting mAP-small improvement):

      1. Asymmetric Bottleneck Dims [256, 128, 64]:
         P3 occupies 256/448 ≈ 57% of the MLP input. Discriminator must pay
         attention to P3 features → stronger gradient signal flows back to the
         early backbone layers responsible for small-object features.

      2. Per-scale GRL Alpha Multipliers [1.5, 1.0, 0.5]:
         P3 GRL reversal is 3× stronger than P5. Early backbone layers (0→P3)
         receive a stronger adversarial signal → forced to align small-object
         features across source/target domains.

    Why fused (not N independent heads):
      • Backbone only needs to fool ONE discriminator → domain_acc → ~0.5.
      • Single unified loss scalar → cleaner gradient graph.

    Args:
        in_channels_list        : [C_P3, C_P4, C_P5] backbone channel counts
        hidden_dim              : MLP hidden width
        dropout                 : dropout probability
        bottleneck_dims         : per-scale bottleneck sizes (default [256,128,64])
        scale_alpha_multipliers : per-scale GRL alpha multiplier (default [1.5,1.0,0.5])
    """

    def __init__(self, in_channels_list: list, hidden_dim: int = 512,
                 dropout: float = 0.3,
                 bottleneck_dims: list = None,
                 scale_alpha_multipliers: list = None):
        """
        Args:
            in_channels_list      : [C_P3, C_P4, C_P5] channel counts
            hidden_dim            : MLP hidden width
            dropout               : dropout probability
            bottleneck_dims       : per-scale bottleneck dims after GAP.
                                    Default [256, 128, 64] → P3 occupies 57% of
                                    discriminator input, biasing domain signal
                                    toward small-object (P3) features.
            scale_alpha_multipliers: per-scale GRL alpha multipliers.
                                    Default [1.5, 1.0, 0.5] → P3 gradient
                                    reversal is 3× stronger than P5, directly
                                    pressing early backbone layers to align
                                    small-object features across domains.
        """
        super().__init__()
        from torch.nn.utils import spectral_norm

        n = len(in_channels_list)

        # ── Per-scale GRL alpha multipliers ─────────────────────────────────
        # P3 (small objects) gets stronger reversal, P5 (semantics) lighter.
        # Ratio 1.5 : 1.0 : 0.5 → P3 gradient is 3× P5, pushing early backbone
        # layers to produce domain-invariant small-object representations.
        if scale_alpha_multipliers is not None:
            assert len(scale_alpha_multipliers) == n, \
                f"scale_alpha_multipliers must have {n} elements"
            self.scale_alpha_multipliers = scale_alpha_multipliers
        else:
            self.scale_alpha_multipliers = [1.5, 1.0, 0.5] if n == 3 else [1.0] * n

        # Per-scale GRL + GAP
        self.grls = nn.ModuleList([
            GradientReversalLayer(alpha=1.0) for _ in in_channels_list
        ])
        self.gaps = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1) for _ in in_channels_list
        ])

        # ── Asymmetric Bottleneck dims ────────────────────────────────────────
        # P3=256, P4=128, P5=64  →  P3 contributes 256/448 ≈ 57% of MLP input
        # Previously uniform 128 (33% each) gave P5 undue influence because
        # its semantic features are easier to classify → discriminator leaned
        # on P5, leaving P3 under-aligned.
        if bottleneck_dims is not None:
            assert len(bottleneck_dims) == n, \
                f"bottleneck_dims must have {n} elements"
            self.bottleneck_dims = bottleneck_dims
        else:
            self.bottleneck_dims = [256, 128, 64] if n == 3 else [128] * n

        self.compressors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(ch, bdim),
                nn.LayerNorm(bdim),
                nn.ReLU(inplace=True)
            ) for ch, bdim in zip(in_channels_list, self.bottleneck_dims)
        ])

        total_ch = sum(self.bottleneck_dims)   # 256+128+64 = 448
        mid = max(hidden_dim // 2, 64)

        # SOTA Discriminator using Spectral Normalization
        # Limits Lipschitz constant to smooth the decision boundary
        self.classifier = nn.Sequential(
            spectral_norm(nn.Linear(total_ch, hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            spectral_norm(nn.Linear(hidden_dim, mid)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(mid, 1),
        )
        self._init_weights()

    def _init_weights(self):
        # Only initialize standard layers. Spectral Norm wrappers are handled smoothly.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def set_alpha(self, alpha: float):
        """Broadcast GRL alpha to all scales with per-scale multipliers.

        Each scale receives  alpha * multiplier:
          P3: alpha * 1.5  (stronger reversal → early backbone layers align)
          P4: alpha * 1.0
          P5: alpha * 0.5  (lighter reversal → semantic layers disturbed less)
        """
        for grl, mult in zip(self.grls, self.scale_alpha_multipliers):
            grl.alpha = alpha * mult

    def forward(self, features_list: list, alpha: float = None) -> torch.Tensor:
        """
        Args:
            features_list : list of [B, Ci, Hi, Wi] tensors (one per scale)
            alpha         : GRL alpha; updates all scales with per-scale multipliers if provided
        Returns:
            logits : [B, 1]  — single scalar logit per image
        """
        if alpha is not None:
            self.set_alpha(alpha)  # applies scale_alpha_multipliers internally

        pooled = []
        for grl, gap, comp, feat, bdim in zip(
            self.grls, self.gaps, self.compressors, features_list, self.bottleneck_dims
        ):
            if feat is None:
                continue
            if torch.isnan(feat).any() or torch.isinf(feat).any():
                # Safe fallback with correct per-scale dim
                pooled.append(torch.zeros(feat.size(0), bdim, device=feat.device))
                continue
            x = grl(feat)            # gradient reversal (per-scale alpha applied)
            x = gap(x).flatten(1)   # [B, Ci]
            x = comp(x)             # [B, bdim]  (256 / 128 / 64)
            pooled.append(x)

        if not pooled:
            # All features invalid — return zero logit (no gradient)
            b = features_list[0].size(0) if features_list else 1
            return torch.zeros(b, 1, device=features_list[0].device)

        x = torch.cat(pooled, dim=1)  # [B, sum(bottleneck_dims)] e.g. [B, 448]
        return self.classifier(x)     # [B, 1]


# ---------------------------------------------------------------------------
# Feature Hooks
# ---------------------------------------------------------------------------

class YOLOv8FeatureHook:
    """
    Forward hook to capture backbone features from a single layer.

    Supports YOLO26s / YOLOv8 model wrapper patterns.
    """

    def __init__(self, model, layer_idx: int = 9):
        self.features = None
        self._hook_handle = None
        self.layer_idx = layer_idx
        self._register_hook(model)

    def _register_hook(self, model):
        try:
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
        self.features = output

    def get_features(self):
        f = self.features
        self.features = None
        return f

    def remove(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def __del__(self):
        try:
            self.remove()
        except Exception:
            pass


class MultiScaleFeatureHook:
    """
    Manages multiple YOLOv8FeatureHook instances for multi-scale GRL.

    Usage:
        hook = MultiScaleFeatureHook(model, layer_indices=[4, 6, 9])
        _ = model(img)
        feats = hook.get_features()   # list of tensors, one per scale
    """

    def __init__(self, model, layer_indices: list):
        self.hooks = [
            YOLOv8FeatureHook(model, idx) for idx in layer_indices
        ]

    def get_features(self) -> list:
        """Return list of captured feature tensors (one per scale)."""
        return [h.get_features() for h in self.hooks]

    def all_available(self, features: list) -> bool:
        """True only when every scale captured a valid tensor."""
        return all(f is not None for f in features)

    def remove(self):
        for h in self.hooks:
            h.remove()

    def __del__(self):
        try:
            self.remove()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Alpha schedule
# ---------------------------------------------------------------------------

def get_grl_alpha(epoch: int, total_epochs: int,
                  warmup_epochs: int = 10, max_alpha: float = 1.0) -> float:
    """
    DANN progressive GRL alpha schedule.
    Returns 0 during warmup then smoothly approaches max_alpha.
    """
    if epoch < warmup_epochs:
        return 0.0
    p = min((epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1), 1.0)
    alpha = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
    return max_alpha * alpha


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def focal_bce_loss(logits: torch.Tensor, labels: torch.Tensor,
                   gamma: float = 2.0, alpha: float = 0.25) -> torch.Tensor:
    """
    Focal Binary Cross-Entropy loss.

    Focal weight  (1 - p_t)^gamma  down-weights easy samples so the
    discriminator focuses on hard boundary examples — more stable training.

    Args:
        logits : raw discriminator output [B, 1]
        labels : ground-truth domain labels [B, 1]
        gamma  : focusing parameter (2 recommended)
        alpha  : class balance weight (0.25 recommended)
    Returns:
        scalar loss
    """
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    probs = torch.sigmoid(logits)
    p_t = probs * labels + (1 - probs) * (1 - labels)
    focal_weight = (1.0 - p_t) ** gamma
    # alpha weighting
    alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
    loss = alpha_t * focal_weight * bce
    return loss.mean()


def compute_domain_loss(domain_pred_source: torch.Tensor,
                        domain_pred_target: torch.Tensor,
                        use_focal: bool = False,
                        gamma: float = 2.0,
                        label_smoothing: float = 0.1) -> torch.Tensor:
    """
    Adversarial domain loss: source → 1, target → 0.
    Uses Label Smoothing over pure BCE to prevent discriminator overconfidence.

    Args:
        domain_pred_source : [B, 1] logits for source images
        domain_pred_target : [B, 1] logits for target images
        use_focal          : use Focal-BCE vs plain BCE
        gamma              : focal parameter
        label_smoothing    : label smoothing for BCE to prevent overconfidence
    Returns:
        scalar loss
    """
    if (torch.isnan(domain_pred_source).any() or
            torch.isnan(domain_pred_target).any() or
            torch.isinf(domain_pred_source).any() or
            torch.isinf(domain_pred_target).any()):
        return torch.tensor(0.0, device=domain_pred_source.device,
                            requires_grad=True)

    # Label smoothing to soften target distributions and regularize discriminator
    src_labels = torch.ones_like(domain_pred_source) * (1.0 - label_smoothing)
    tgt_labels = torch.zeros_like(domain_pred_target) + label_smoothing

    if use_focal:
        loss_src = focal_bce_loss(domain_pred_source, src_labels, gamma=gamma)
        loss_tgt = focal_bce_loss(domain_pred_target, tgt_labels, gamma=gamma)
    else:
        loss_src = F.binary_cross_entropy_with_logits(domain_pred_source, src_labels)
        loss_tgt = F.binary_cross_entropy_with_logits(domain_pred_target, tgt_labels)

    return (loss_src + loss_tgt) * 0.5


def compute_multiscale_domain_loss(src_logits_list: list,
                                   tgt_logits_list: list,
                                   scale_weights: list = None,
                                   use_focal: bool = False) -> torch.Tensor:
    """
    Domain loss across multiple scales.

    Note: kept for backward compatibility with code paths that still use the
    independent-head MultiScaleDomainDiscriminator.  For the new fused
    MultiScaleFusedDiscriminator use compute_domain_loss() directly.

    Args:
        src_logits_list : list of [B, 1] source logits (one per scale)
        tgt_logits_list : list of [B, 1] target logits (one per scale)
        scale_weights   : optional per-scale weights (default equal)
        use_focal       : use Focal-BCE
    Returns:
        scalar weighted sum of per-scale domain losses
    """
    n = len(src_logits_list)
    if scale_weights is None:
        scale_weights = [1.0 / n] * n

    # Use list comprehension + sum() — avoids the leaf-tensor requires_grad=True
    # anti-pattern that creates disconnected graph nodes.
    losses = [
        w * compute_domain_loss(sl, tl, use_focal=use_focal)
        for w, sl, tl in zip(scale_weights, src_logits_list, tgt_logits_list)
        if sl is not None and tl is not None
    ]
    if not losses:
        device = src_logits_list[0].device if src_logits_list else 'cpu'
        return torch.zeros(1, device=device).squeeze()
    return sum(losses)


# ---------------------------------------------------------------------------
# Accuracy metric  (FIXED — works for [B,1] output from GAP+MLP)
# ---------------------------------------------------------------------------

def get_domain_accuracy(domain_pred_source: torch.Tensor,
                        domain_pred_target: torch.Tensor) -> float:
    """
    Discriminator accuracy — how well it separates source vs target.

    NOTE: Designed for [B, 1] logit tensors produced by DomainDiscriminator
    (after GAP). With spatial [B,1,H,W] outputs the old code was dividing
    (H*W*B correct pixels) by (B) → 16740% bug.  Fixed.

    Args:
        domain_pred_source : [B, 1] logits — should predict > 0 (source)
        domain_pred_target : [B, 1] logits — should predict ≤ 0 (target)
    Returns:
        accuracy in [0, 1];  0.5 ≈ discriminator confused (GRL working)
    """
    with torch.no_grad():
        # Pool to [B, 1] if spatial output accidentally passed
        if domain_pred_source.dim() > 2:
            domain_pred_source = domain_pred_source.flatten(2).mean(-1, keepdim=True)
        if domain_pred_target.dim() > 2:
            domain_pred_target = domain_pred_target.flatten(2).mean(-1, keepdim=True)

        src_correct = (domain_pred_source > 0).float().sum().item()
        tgt_correct = (domain_pred_target <= 0).float().sum().item()
        total = domain_pred_source.numel() + domain_pred_target.numel()
        return (src_correct + tgt_correct) / max(total, 1)


def get_multiscale_domain_accuracy(src_logits_list: list,
                                   tgt_logits_list: list) -> float:
    """Mean domain accuracy across all scales."""
    accs = [
        get_domain_accuracy(sl, tl)
        for sl, tl in zip(src_logits_list, tgt_logits_list)
        if sl is not None and tl is not None
    ]
    return float(sum(accs) / len(accs)) if accs else 0.5


# ---------------------------------------------------------------------------
# Auto-detect layer indices
# ---------------------------------------------------------------------------

def find_last_backbone_layer(model) -> int:
    """
    Auto-detect index of the final backbone layer (SPPF or C2PSA).
    Priority: C2PSA (YOLO26) > SPPF (YOLOv8) > fallback 9.
    """
    layers = _get_layers(model)
    if layers is None:
        return 9

    last_c2psa = last_sppf = -1
    for idx, layer in enumerate(layers):
        name = layer.__class__.__name__
        if name == 'C2PSA':
            last_c2psa = idx
        elif name == 'SPPF':
            last_sppf = idx

    if last_c2psa >= 0:
        return last_c2psa
    if last_sppf >= 0:
        return last_sppf
    return 9


def find_multiscale_layers(model) -> list:
    """
    Auto-detect P3/P4/P5 **backbone** layer indices for multi-scale GRL.

    WHY BACKBONE (not neck):
      Hooking neck PAN outputs means GRL gradient must travel through extra
      neck layers before reaching backbone weights — longer path, more
      gradient vanishing, less effective alignment.  Hooking directly at
      backbone mid/end gives a shorter, cleaner gradient path.

    Strategy (YOLOv8 / YOLO26 backbone):
      Layer 0-1 : input stem Conv (stride 2, 4)
      Layer 2   : C2f  (stride 4  — too shallow, skip)
      Layer 3   : Conv (stride 2)
      Layer 4   : C2f  (stride 8)  ← P3  small-object features
      Layer 5   : Conv (stride 2)
      Layer 6   : C2f  (stride 16) ← P4  medium-object features
      Layer 7   : Conv (stride 2)
      Layer 8   : C2f  (stride 32) ← deep backbone
      Layer 9   : SPPF/C2PSA       ← P5  backbone end, semantic

    We collect ALL C2f/C2/C2PSA within the backbone (idx <= backbone_end),
    then pick the last-3 as [P3_idx, P4_idx, P5_idx=backbone_end].

    Returns list of 3 indices.  Falls back to [4, 6, backbone_end].
    """
    layers = _get_layers(model)
    backbone_end = find_last_backbone_layer(model)

    if layers is None:
        print(f"[GRL] Cannot resolve layers — falling back to [4, 6, {backbone_end}]")
        return [4, 6, backbone_end]

    # All C2f/C2/C2PSA within backbone (index strictly <= backbone_end)
    c2f_backbone = [
        idx for idx, layer in enumerate(layers)
        if idx <= backbone_end
        and layer.__class__.__name__ in ('C2f', 'C2', 'C2PSA')
    ]

    print(f"[GRL] Backbone end={backbone_end}, backbone C2f/C2PSA indices={c2f_backbone}")

    if len(c2f_backbone) >= 3:
        # Last 3 backbone C2f = stride 8 / 16 / deepest (P3/P4/P5 equivalent)
        # backbone_end (SPPF/C2PSA) is always used as P5 anchor
        p3_idx = c2f_backbone[-3]   # third-to-last C2f in backbone
        p4_idx = c2f_backbone[-2]   # second-to-last C2f in backbone
        p5_idx = backbone_end        # backbone end (SPPF/C2PSA) = P5
        chosen = [p3_idx, p4_idx, p5_idx]
    elif len(c2f_backbone) == 2:
        chosen = [c2f_backbone[0], c2f_backbone[-1], backbone_end]
    elif len(c2f_backbone) == 1:
        chosen = [c2f_backbone[0], backbone_end, backbone_end]
    else:
        # Unexpected architecture — safe fallback
        print("[GRL] WARNING: no backbone C2f found, using positional fallback.")
        chosen = [max(1, backbone_end - 5),
                  max(1, backbone_end - 3),
                  backbone_end]

    print(
        f"[GRL] Multi-scale backbone hooks "
        f"→ P3=layer{chosen[0]}, P4=layer{chosen[1]}, P5=layer{chosen[2]} "
        f"(backbone stride-8/16/32 features — direct gradient path)"
    )
    return chosen


def _get_layers(model):
    """Helper: resolve the nn.Sequential layer list from a YOLO model."""
    if hasattr(model, 'model') and isinstance(model.model, nn.Sequential):
        return model.model
    elif hasattr(model, 'model') and hasattr(model.model, 'model'):
        return model.model.model
    elif isinstance(model, nn.Sequential):
        return model
    else:
        try:
            return model.model
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Legacy compat alias
# ---------------------------------------------------------------------------

def create_feature_hook(model, layer_name: str = 'backbone'):
    """Create single-scale feature hook (auto-detects backbone end layer)."""
    layer_idx = find_last_backbone_layer(model)
    print(f"[FeatureHook] Auto-detected backbone end layer: {layer_idx}")
    return YOLOv8FeatureHook(model, layer_idx=layer_idx)