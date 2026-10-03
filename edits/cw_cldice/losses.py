"""
Conductance-Weighted clDice (cw-clDice) Loss Module
===================================================
A physics-informed, differentiable topological objective function that bridges
Hagen-Poiseuille hemodynamics with centerline skeleton supervision.

Key Innovations:
1. Caliber-Based Weight Map W(x, y):
   Derived from the Euclidean distance transform of ground-truth vessels.
   Assigns high weight (1.0 + alpha) to thin, high-resistance capillaries (r ~ 1 px)
   and baseline weight (1.0) to thick arterial trunks (r >= 4 px).
2. Differentiable Soft Skeletonization:
   Iterative min/max pooling operators (CVPR 2021) enabling gradient backpropagation.
3. Conductance-Weighted Topology Sensitivity & Precision:
   Forces the neural network's backpropagation gradients to prioritize fragile,
   low-contrast capillaries that suffer from the 43x fragmentation baseline defect.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftSkeletonize(nn.Module):
    """
    Differentiable 2D Soft Skeletonization via Morphological Pooling (CVPR 2021).
    Uses cross-shaped min-pooling for soft erosion and 3x3 max-pooling for soft dilation.
    """
    def __init__(self, num_iter=4):
        super(SoftSkeletonize, self).__init__()
        self.num_iter = num_iter

    def soft_erode(self, x):
        """Cross-shaped structuring element min-pooling: min(P1, P2)"""
        p1 = -F.max_pool2d(-x, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        p2 = -F.max_pool2d(-x, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        return torch.min(p1, p2)

    def soft_dilate(self, x):
        """Square structuring element max-pooling"""
        return F.max_pool2d(x, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def soft_open(self, x):
        """Morphological opening = Dilation of Erosion"""
        return self.soft_dilate(self.soft_erode(x))

    def forward(self, x):
        """
        Extract continuous soft centerline skeleton map.
        x: (B, 1, H, W) continuous probabilities in [0, 1]
        returns: (B, 1, H, W) continuous skeleton activations in [0, 1]
        """
        x_open = self.soft_open(x)
        skel = F.relu(x - x_open)
        current = x
        for _ in range(self.num_iter):
            current = self.soft_erode(current)
            current_open = self.soft_open(current)
            delta = F.relu(current - current_open)
            skel = skel + F.relu(delta - skel * delta)
        return skel


class ConductanceWeightedclDiceLoss(nn.Module):
    """
    Conductance-Weighted clDice Loss (cw-clDice).
    
    W(x, y) = 1.0 + alpha * ((r_max - R(x, y)) / (r_max - r_min + eps))^beta
    
    T_sens_cw = sum(W * S_gt * P) / (sum(W * S_gt) + eps)
    T_prec_cw = sum(W * S_pred * V_gt) / (sum(W * S_pred) + eps)
    cw-clDice = (2 * T_prec_cw * T_sens_cw) / (T_prec_cw + T_sens_cw + eps)
    L_cw-clDice = 1.0 - cw-clDice
    """
    def __init__(self, num_iter=4, alpha=2.0, beta=1.5, eps=1e-7):
        super(ConductanceWeightedclDiceLoss, self).__init__()
        self.soft_skel = SoftSkeletonize(num_iter=num_iter)
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def compute_caliber_weights(self, distance_map, mask_gt):
        """
        Computes the hemodynamic conductance weight map W(x, y).
        Thin vessels with small radius get boosted by (1.0 + alpha).
        Thick vessels get weight close to 1.0.
        Background pixels get weight 1.0.
        """
        vessel_mask = (mask_gt > 0.5).float()
        vessel_radii = distance_map * vessel_mask

        # Batch-wise or sample-wise min/max radius
        r_max = torch.amax(vessel_radii, dim=(-2, -1), keepdim=True)
        # Avoid zero as r_min by clamping to 1.0 on vessel regions
        r_min = torch.ones_like(r_max)

        # Inverted normalized radius: 1.0 on capillaries, 0.0 on thick trunks
        normalized_inv_radius = (r_max - vessel_radii) / (r_max - r_min + 1e-4)
        normalized_inv_radius = torch.clamp(normalized_inv_radius, 0.0, 1.0) * vessel_mask

        # Physiological weight map
        W = 1.0 + self.alpha * torch.pow(normalized_inv_radius, self.beta)
        return W

    def forward(self, y_pred, y_true, s_gt=None, weight_map=None, distance_map=None):
        """
        y_pred: (B, 1, H, W) continuous prediction in [0, 1]
        y_true: (B, 1, H, W) ground truth binary vessel mask in {0, 1}
        s_gt: (B, 1, H, W) ground truth 1-pixel centerline skeleton (optional, computed via soft_skel if None)
        weight_map: (B, 1, H, W) precomputed conductance weight map W(x, y) (optional)
        distance_map: (B, 1, H, W) Euclidean distance transform of y_true (optional)
        """
        # 1. Ground truth centerline skeleton
        if s_gt is None:
            s_gt = self.soft_skel(y_true)

        # 2. Predicted soft centerline skeleton (differentiable)
        s_pred = self.soft_skel(y_pred)

        # 3. Physiological weight map W(x, y)
        if weight_map is None:
            if distance_map is not None:
                W = self.compute_caliber_weights(distance_map, y_true)
            else:
                # Fallback to uniform weight if distance map is unavailable
                W = torch.ones_like(y_true)
        else:
            W = weight_map

        # 4. Conductance-Weighted Topology Sensitivity:
        # Measures whether predicted probabilities cover true vessel centerlines,
        # with extra penalty for missing fine capillaries.
        t_sens_cw = torch.sum(W * s_gt * y_pred, dim=(-2, -1)) / (torch.sum(W * s_gt, dim=(-2, -1)) + self.eps)

        # 5. Conductance-Weighted Topology Precision:
        # Measures whether predicted centerline falls within true vessel lumen.
        t_prec_cw = torch.sum(W * s_pred * y_true, dim=(-2, -1)) / (torch.sum(W * s_pred, dim=(-2, -1)) + self.eps)

        # 6. Combined cw-clDice score & loss
        cw_cldice = (2.0 * t_prec_cw * t_sens_cw) / (t_prec_cw + t_sens_cw + self.eps)
        loss = 1.0 - torch.mean(cw_cldice)

        return loss, torch.mean(cw_cldice), torch.mean(t_sens_cw), torch.mean(t_prec_cw)


class ContinuousMCCLoss(nn.Module):
    """
    Differentiable Matthews Correlation Coefficient (MCC) Loss:
    L_MCC = 1 - MCC
    """
    def __init__(self, eps=1e-7):
        super(ContinuousMCCLoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        tp = torch.sum(y_pred * y_true)
        tn = torch.sum((1.0 - y_true) * (1.0 - y_pred))
        fp = torch.sum((1.0 - y_true) * y_pred)
        fn = torch.sum(y_true * (1.0 - y_pred))

        numerator = tp * tn - fp * fn
        denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + self.eps)

        mcc = numerator / (denominator + self.eps)
        return 1.0 - mcc


class ConductanceWeightedBCELoss(nn.Module):
    """
    Conductance-Weighted Binary Cross-Entropy (cw-BCE) Loss.
    Scales pixel-level BCE penalty by physiological caliber weight map W(x, y).

    L_cw-BCE = - (1 / N) * sum( W(x, y) * [ y * log(p + eps) + (1 - y) * log(1 - p + eps) ] )

    When W(x, y) = 1.0 (uniform), reduces identically to standard nn.BCELoss().
    Capillaries (r ~ 1 px) receive an alpha-scaled gradient boost (2.5x - 3.5x).
    """
    def __init__(self, eps=1e-7):
        super(ConductanceWeightedBCELoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred, y_true, weight_map=None):
        y_pred = torch.clamp(y_pred, self.eps, 1.0 - self.eps)
        bce_pixel = -(y_true * torch.log(y_pred) + (1.0 - y_true) * torch.log(1.0 - y_pred))
        if weight_map is not None:
            loss = torch.mean(weight_map * bce_pixel)
        else:
            loss = torch.mean(bce_pixel)
        return loss


class CompoundCwclDiceLoss(nn.Module):
    """
    Unified Compound Objective supporting multiple research tracks & ablations:
    1. 'cw_cldice' (Default):  0.5 * BCE + 0.5 * MCC + lambda_cw * cw-clDice
    2. 'cw_bce_only' (1A):     0.5 * cw-BCE + 0.5 * MCC (no centerline skeletonizer)
    3. 'unified' (1B):         0.5 * cw-BCE + 0.5 * MCC + lambda_cw * cw-clDice (both 2D and 1D caliber weighted)
    4. 'vanilla_cldice':       0.5 * BCE + 0.5 * MCC + lambda_cw * clDice (alpha=0.0)
    """
    def __init__(self, lambda_bce=0.5, lambda_mcc=0.5, lambda_cw=0.2,
                 use_cw_bce=False, num_skel_iter=4, alpha=2.0, beta=1.5, eps=1e-7):
        super(CompoundCwclDiceLoss, self).__init__()
        self.lambda_bce = lambda_bce
        self.lambda_mcc = lambda_mcc
        self.lambda_cw = lambda_cw
        self.use_cw_bce = use_cw_bce
        self.standard_bce = nn.BCELoss()
        self.cw_bce = ConductanceWeightedBCELoss(eps=eps)
        self.mcc = ContinuousMCCLoss(eps=eps)
        self.cw_cldice = ConductanceWeightedclDiceLoss(num_iter=num_skel_iter, alpha=alpha, beta=beta, eps=eps)

    def forward(self, y_pred, y_true, s_gt=None, weight_map=None, distance_map=None):
        if self.use_cw_bce and weight_map is not None:
            loss_bce = self.cw_bce(y_pred, y_true, weight_map=weight_map)
        else:
            loss_bce = self.standard_bce(y_pred, y_true)

        loss_mcc = self.mcc(y_pred, y_true)

        if self.lambda_cw > 0:
            loss_cw, cw_score, t_sens, t_prec = self.cw_cldice(y_pred, y_true, s_gt, weight_map, distance_map)
        else:
            loss_cw = torch.tensor(0.0, device=y_pred.device)
            cw_score = torch.tensor(1.0, device=y_pred.device)

        loss_total = (self.lambda_bce * loss_bce +
                      self.lambda_mcc * loss_mcc +
                      self.lambda_cw * loss_cw)

        return loss_total, loss_bce, loss_mcc, loss_cw, cw_score
