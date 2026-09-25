import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with LeakyReLU activation and optional residual connection."""
    def __init__(self, in_channels: int, out_channels: int, use_residual: bool = True):
        super().__init__()
        self.use_residual = use_residual and (in_channels == out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.act1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.act2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.use_residual else None
        out = self.act1(self.conv1(x))
        out = self.conv2(out)
        if residual is not None:
            out = out + residual
        return self.act2(out)


class SpatialUNet(nn.Module):
    """
    Lightweight 2D Spatial U-Net for Real-Time 1 SPP Monte Carlo Denoising.
    
    Default input channels (7):
      - [0:3] Noisy HDR Radiance (RGB, log-transformed)
      - [3:5] View-space Normal (2D projected vector or [3:6] for 3D)
      - [5:6] Linearized Depth (Z)
      - [6:7] Material Roughness
      
    Outputs (3):
      - [0:3] Denoised HDR Radiance (RGB)
    """
    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 3,
        base_channels: int = 32,
        num_levels: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_levels = num_levels

        # Encoder levels
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        current_in = in_channels
        ch = base_channels

        for i in range(num_levels):
            self.encoders.append(ConvBlock(current_in, ch))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            current_in = ch
            ch = ch * 2

        # Bottleneck
        self.bottleneck = ConvBlock(current_in, ch)

        # Decoder levels
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for i in range(num_levels):
            up_in = ch
            up_out = ch // 2
            self.upconvs.append(
                nn.ConvTranspose2d(up_in, up_out, kernel_size=2, stride=2)
            )
            # Skip connection concatenates up_out + up_out = 2 * up_out
            self.decoders.append(ConvBlock(up_out * 2, up_out))
            ch = up_out

        # Final reconstruction layer
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        skips = []
        out = x
        for enc, pool in zip(self.encoders, self.pools):
            out = enc(out)
            skips.append(out)
            out = pool(out)

        # Bottleneck
        out = self.bottleneck(out)

        # Decoder
        for upconv, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            out = upconv(out)
            # Ensure spatial dimension match if odd resolution
            if out.shape != skip.shape:
                out = F.interpolate(out, size=skip.shape[2:], mode='bilinear', align_corners=False)
            out = torch.cat([out, skip], dim=1)
            out = dec(out)

        return self.final_conv(out)
