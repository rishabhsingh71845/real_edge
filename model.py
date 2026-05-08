import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# U-Net Building Blocks
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """Two sequential (Conv2d → BatchNorm → ReLU) blocks."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """MaxPool downscaling followed by DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    """Bilinear upsampling (or transposed conv) followed by DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Pad to handle odd spatial dimensions
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                         diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    """1×1 projection to the final class logits."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    Standard U-Net with skip connections.
    Bilinear upsampling avoids transposed-convolution checkerboard artifacts,
    which is especially beneficial when combined with TV loss regularization.
    """

    def __init__(self, n_channels: int, n_classes: int, bilinear: bool = True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


# ---------------------------------------------------------------------------
# Total Variation (TV) Denoising Loss
# ---------------------------------------------------------------------------

class TotalVariationLoss(nn.Module):
    """
    Anisotropic Total Variation Loss applied to the probability maps (softmax logits).

    TV loss penalizes large gradients between neighboring pixels in the prediction map,
    forcing the model to produce spatially smooth class boundaries. This is the primary
    mechanism for eliminating checkerboard / synthetic rendering artifacts on edges.

    Formula (anisotropic TV):
        TV(x) = Σ |x_{i+1,j} - x_{i,j}| + |x_{i,j+1} - x_{i,j}|

    Trade-off note:
        Higher TV_LOSS_WEIGHT → smoother boundaries, potentially slightly lower mIoU.
        Start at 0.001 and tune upward if artifacts remain.
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # Convert raw logits to class probabilities before computing gradients
        probs = torch.softmax(logits, dim=1)  # (B, C, H, W)

        # Horizontal and vertical finite differences
        diff_h = torch.abs(probs[:, :, 1:, :] - probs[:, :, :-1, :])
        diff_w = torch.abs(probs[:, :, :, 1:] - probs[:, :, :, :-1])

        tv_loss = diff_h.mean() + diff_w.mean()
        return tv_loss
