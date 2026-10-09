"""
CAD-Topo-CSA: Caliber-Adaptive Directional Cross-Scale Spatial Attention
========================================================================
Architectural Innovation on SA-UNetv2 Skip Connections (Track 2)

Key Geometric Innovations:
1. Wide Trunk Branch: 7x7 isotropic square convolution. Preserves crisp boundary margins
   of main arterial trunks without cross-sectional distortion.
2. Longitudinal Capillary Branches: Parallel 1x21 horizontal and 21x1 vertical strip convolutions.
   Physically bridges optical dropouts along capillary trajectories without perpendicular noise.
3. Self-Routing Channel Gate (SE-MLP): Automatically learns to dynamically weight trunk vs.
   directional strip context based on intrinsic spatial features.

Strict Invariants:
- Zero external inputs (no GT masks, no distance transforms, no conductance values at test time).
- Zero auxiliary losses.
- Lightweight parameter footprint: < 250 parameters per module (~700 params total, < 0.3% overhead).
- Fully autonomous feed-forward inference on unseen retinal images.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CADTopoCSAModule(nn.Module):
    """
    Caliber-Adaptive Directional Cross-Scale Spatial Attention (CAD-Topo-CSA).
    Drop-in replacement for CrossScaleSpatialAttention in SA-UNetv2 skip connections.
    Supports configurable strip height h for transverse contrast checking (h x 21 and 21 x h).
    """
    def __init__(self, trunk_kernel=7, strip_length=21, strip_height=1, reduction=2):
        super(CADTopoCSAModule, self).__init__()
        assert strip_height % 2 == 1, f"strip_height must be odd, got {strip_height}"
        assert strip_length % 2 == 1, f"strip_length must be odd, got {strip_length}"
        self.trunk_kernel = trunk_kernel
        self.strip_length = strip_length
        self.strip_height = strip_height

        # Branch 1: Wide Trunk Path (Isotropic Square Conv 7x7)
        self.conv_trunk = nn.Conv2d(
            2, 1, kernel_size=trunk_kernel, padding=trunk_kernel // 2, bias=False
        )

        # Branch 2: Horizontal Capillary Strip Path (h x strip_length)
        self.conv_strip_h = nn.Conv2d(
            2, 1,
            kernel_size=(strip_height, strip_length),
            padding=(strip_height // 2, strip_length // 2),
            bias=False
        )

        # Branch 3: Vertical Capillary Strip Path (strip_length x h)
        self.conv_strip_v = nn.Conv2d(
            2, 1,
            kernel_size=(strip_length, strip_height),
            padding=(strip_length // 2, strip_height // 2),
            bias=False
        )

        # Self-Routing Channel Gate: Squeeze-and-Excitation across the 3 scale paths
        num_branches = 3
        hidden_dim = max(4, num_branches * reduction)
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_branches, hidden_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_branches, bias=True),
            nn.Softmax(dim=-1)
        )

        # Fusion 1x1 Convolution & Sigmoid Gating
        self.fusion = nn.Conv2d(num_branches, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        # Weight Initialization
        nn.init.kaiming_normal_(self.conv_trunk.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv_strip_h.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv_strip_v.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fusion.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, fe, fd):
        """
        fe: (B, C_enc, H, W) encoder skip feature
        fd: (B, C_dec, H, W) decoder context feature
        """
        # 1. Spatial channel descriptors
        avg_fe = torch.mean(fe, dim=1, keepdim=True)
        avg_fd = torch.mean(fd, dim=1, keepdim=True)
        concat = torch.cat([avg_fe, avg_fd], dim=1)  # (B, 2, H, W)

        # 2. Multi-Scale Geometric Contexts
        f_trunk = self.conv_trunk(concat)      # (B, 1, H, W) - circular trunk walls
        f_strip_h = self.conv_strip_h(concat)  # (B, 1, H, W) - horizontal capillaries
        f_strip_v = self.conv_strip_v(concat)  # (B, 1, H, W) - vertical capillaries

        f_multi = torch.cat([f_trunk, f_strip_h, f_strip_v], dim=1)  # (B, 3, H, W)

        # 3. Dynamic Channel Routing
        routing_weights = self.router(f_multi)  # (B, 3)
        routing_weights = routing_weights.view(routing_weights.size(0), 3, 1, 1)

        f_routed = f_multi * routing_weights

        # 4. Final Spatial Attention Gate
        attn_map = self.sigmoid(self.fusion(f_routed))

        return fe * attn_map
