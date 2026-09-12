import torch
import torch.nn as nn
import torch.nn.functional as F


class DropBlock2D(nn.Module):
    """
    DropBlock: A regularization method for convolutional networks (Ghiasi et al., NeurIPS 2018).
    Drops contiguous square regions from feature maps.
    """
    def __init__(self, drop_prob=0.15, block_size=7):
        super(DropBlock2D, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        if not self.training or self.drop_prob <= 0.0:
            return x

        with torch.no_grad():
            batch_size, channels, height, width = x.shape
            # Effective region where drop centers can fall
            gamma = (self.drop_prob * height * width) / (
                self.block_size ** 2 * (height - self.block_size + 1) * (width - self.block_size + 1)
            )
            gamma = min(max(gamma, 0.0), 1.0)

            # Sample drop centers from Bernoulli distribution
            mask = (torch.rand((batch_size, 1, height - self.block_size + 1, width - self.block_size + 1), device=x.device) < gamma).float()
            
            # Pad mask to original dimensions
            pad_h = (self.block_size - 1) // 2
            pad_w = (self.block_size - 1) // 2
            pad_h_extra = (self.block_size - 1) % 2
            pad_w_extra = (self.block_size - 1) % 2
            mask = F.pad(mask, (pad_w, pad_w + pad_w_extra, pad_h, pad_h + pad_h_extra), value=0.0)

            # Apply max pooling to expand each center into a full square block
            mask = F.max_pool2d(mask, kernel_size=self.block_size, stride=1, padding=self.block_size // 2)
            if mask.shape[-2:] != x.shape[-2:]:
                mask = F.interpolate(mask, size=x.shape[-2:], mode='nearest')

            block_mask = 1.0 - mask
            normalize_scale = block_mask.numel() / (block_mask.sum() + 1e-7)

        return x * block_mask * normalize_scale


class ConvBlock(nn.Module):
    """
    Optimized ConvBlock in SA-UNetv2:
    Conv3x3 -> DropBlock2D -> GroupNorm(8) -> SiLU
    """
    def __init__(self, in_channels, out_channels, drop_prob=0.15, block_size=7, num_groups=8):
        super(ConvBlock, self).__init__()
        # Ensure num_groups divides out_channels
        if out_channels % num_groups != 0:
            num_groups = min(g for g in [8, 4, 2, 1] if out_channels % g == 0)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.dropblock = DropBlock2D(drop_prob=drop_prob, block_size=block_size)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.act = nn.SiLU(inplace=True)

        # He normal initialization
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        return self.act(self.gn(self.dropblock(self.conv(x))))


class SpatialAttention(nn.Module):
    """
    Spatial Attention (SA) module used in the bottleneck:
    Averages across channels -> 7x7 Conv -> Sigmoid -> Elementwise multiply
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        scale = self.sigmoid(self.conv(avg_out))
        return x * scale


class CrossScaleSpatialAttention(nn.Module):
    """
    Cross-scale Spatial Attention (CSA) module in SA-UNetv2 skip connections:
    F_out = F_e * sigmoid( Conv7x7( [ AvgPool(F_e) ; AvgPool(F_d) ] ) )
    Bridges the semantic gap by conditioning encoder features on decoder context.
    """
    def __init__(self, kernel_size=7):
        super(CrossScaleSpatialAttention, self).__init__()
        # Input has 2 channels (1 from AvgPool(Fe), 1 from AvgPool(Fd))
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, fe, fd):
        avg_fe = torch.mean(fe, dim=1, keepdim=True)
        avg_fd = torch.mean(fd, dim=1, keepdim=True)
        concat = torch.cat([avg_fe, avg_fd], dim=1)
        attn_map = self.sigmoid(self.conv(concat))
        return fe * attn_map


class SA_UNetv2(nn.Module):
    """
    SA-UNetv2: Rethinking Spatial Attention U-Net for Retinal Vessel Segmentation (ISBI 2026).
    Parameters: ~0.26M
    Channels: 16 -> 32 -> 48 -> 64
    """
    def __init__(self, in_channels=3, out_channels=1, start_neurons=16, drop_prob=0.15, block_size=7):
        super(SA_UNetv2, self).__init__()
        c1 = start_neurons * 1  # 16
        c2 = start_neurons * 2  # 32
        c3 = start_neurons * 3  # 48
        c4 = start_neurons * 4  # 64

        # --- Encoder ---
        self.enc1_1 = ConvBlock(in_channels, c1, drop_prob, block_size)
        self.enc1_2 = ConvBlock(c1, c1, drop_prob, block_size)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2_1 = ConvBlock(c1, c2, drop_prob, block_size)
        self.enc2_2 = ConvBlock(c2, c2, drop_prob, block_size)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc3_1 = ConvBlock(c2, c3, drop_prob, block_size)
        self.enc3_2 = ConvBlock(c3, c3, drop_prob, block_size)
        self.pool3 = nn.MaxPool2d(2, 2)

        # --- Bottleneck ---
        self.bottleneck_1 = ConvBlock(c3, c4, drop_prob, block_size)
        self.bottleneck_sa = SpatialAttention(kernel_size=7)
        self.bottleneck_2 = ConvBlock(c4, c4, drop_prob, block_size)

        # --- Decoder Stage 3 ---
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.csa3 = CrossScaleSpatialAttention(kernel_size=7)
        self.dec3_1 = ConvBlock(c3 * 2, c3, drop_prob, block_size)
        self.dec3_2 = ConvBlock(c3, c3, drop_prob, block_size)

        # --- Decoder Stage 2 ---
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.csa2 = CrossScaleSpatialAttention(kernel_size=7)
        self.dec2_1 = ConvBlock(c2 * 2, c2, drop_prob, block_size)
        self.dec2_2 = ConvBlock(c2, c2, drop_prob, block_size)

        # --- Decoder Stage 1 ---
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.csa1 = CrossScaleSpatialAttention(kernel_size=7)
        self.dec1_1 = ConvBlock(c1 * 2, c1, drop_prob, block_size)
        self.dec1_2 = ConvBlock(c1, c1, drop_prob, block_size)

        # --- Output Head ---
        self.final_conv = nn.Conv2d(c1, out_channels, kernel_size=1)
        nn.init.kaiming_normal_(self.final_conv.weight, mode='fan_out', nonlinearity='linear')

    def forward(self, x):
        # Encoder
        x1 = self.enc1_2(self.enc1_1(x))
        p1 = self.pool1(x1)

        x2 = self.enc2_2(self.enc2_1(p1))
        p2 = self.pool2(x2)

        x3 = self.enc3_2(self.enc3_1(p2))
        p3 = self.pool3(x3)

        # Bottleneck
        bm = self.bottleneck_1(p3)
        bm = self.bottleneck_sa(bm)
        bm = self.bottleneck_2(bm)

        # Decoder 3
        d3 = self.up3(bm)
        gated_x3 = self.csa3(x3, d3)
        u3 = torch.cat([d3, gated_x3], dim=1)
        u3 = self.dec3_2(self.dec3_1(u3))

        # Decoder 2
        d2 = self.up2(u3)
        gated_x2 = self.csa2(x2, d2)
        u2 = torch.cat([d2, gated_x2], dim=1)
        u2 = self.dec2_2(self.dec2_1(u2))

        # Decoder 1
        d1 = self.up1(u2)
        gated_x1 = self.csa1(x1, d1)
        u1 = torch.cat([d1, gated_x1], dim=1)
        u1 = self.dec1_2(self.dec1_1(u1))

        # Sigmoid output
        out = torch.sigmoid(self.final_conv(u1))
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = SA_UNetv2(in_channels=3, out_channels=1, start_neurons=16)
    print("SA-UNetv2 Trainable Parameters:", f"{count_parameters(model) / 1e6:.4f}M ({count_parameters(model):,} parameters)")
    dummy = torch.randn(2, 3, 592, 592)
    out = model(dummy)
    print("Forward pass successful! Output shape:", out.shape)
