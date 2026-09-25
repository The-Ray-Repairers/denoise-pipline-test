import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .unet import ConvBlock


class ConvGRUCell(nn.Module):
    """Convolutional GRU cell for spatial-temporal latent memory propagation."""
    def __init__(self, hidden_dim: int, input_dim: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2

        self.conv_gates = nn.Conv2d(
            input_dim + hidden_dim, 2 * hidden_dim, kernel_size=kernel_size, padding=padding
        )
        self.conv_candidate = nn.Conv2d(
            input_dim + hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding
        )

    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        if h_prev is None:
            h_prev = torch.zeros(
                x.size(0), self.hidden_dim, x.size(2), x.size(3), device=x.device, dtype=x.dtype
            )

        combined = torch.cat([x, h_prev], dim=1)
        gates = torch.sigmoid(self.conv_gates(combined))
        r_gate, z_gate = torch.chunk(gates, 2, dim=1)

        candidate_input = torch.cat([x, r_gate * h_prev], dim=1)
        candidate = torch.tanh(self.conv_candidate(candidate_input))

        h_next = (1 - z_gate) * h_prev + z_gate * candidate
        return h_next


class RecurrentUNet(nn.Module):
    """
    Recurrent Denoising Autoencoder (R-DAE) for Real-Time Path-Traced Sequences.
    
    Integrates a ConvGRU cell at the latent bottleneck.
    Temporal hidden states are maintained across consecutive frames to suppress boiling artifacts.
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

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        current_in = in_channels
        ch = base_channels

        for _ in range(num_levels):
            self.encoders.append(ConvBlock(current_in, ch))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            current_in = ch
            ch = ch * 2

        self.latent_dim = ch
        # Bottleneck with ConvGRU
        self.bottleneck_pre = ConvBlock(current_in, ch)
        self.recurrent_cell = ConvGRUCell(hidden_dim=ch, input_dim=ch)
        self.bottleneck_post = ConvBlock(ch, ch)

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for _ in range(num_levels):
            up_in = ch
            up_out = ch // 2
            self.upconvs.append(
                nn.ConvTranspose2d(up_in, up_out, kernel_size=2, stride=2)
            )
            self.decoders.append(ConvBlock(up_out * 2, up_out))
            ch = up_out

        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward_frame(
        self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Denoises a single frame and returns the reconstructed frame and the new hidden state.
        
        Args:
            x: Input frame [B, C, H, W]
            h_prev: Previous latent hidden state [B, latent_dim, H // 2^levels, W // 2^levels]
        """
        # Encoder
        skips = []
        out = x
        for enc, pool in zip(self.encoders, self.pools):
            out = enc(out)
            skips.append(out)
            out = pool(out)

        # Bottleneck Recurrent Processing
        out = self.bottleneck_pre(out)
        h_next = self.recurrent_cell(out, h_prev)
        out = self.bottleneck_post(h_next)

        # Decoder
        for upconv, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            out = upconv(out)
            if out.shape != skip.shape:
                out = F.interpolate(out, size=skip.shape[2:], mode='bilinear', align_corners=False)
            out = torch.cat([out, skip], dim=1)
            out = dec(out)

        denoised = self.final_conv(out)
        return denoised, h_next

    def forward_sequence(
        self, seq: torch.Tensor, h_init: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes a sequence of frames [B, T, C, H, W].
        
        Returns:
            denoised_seq: [B, T, out_channels, H, W]
            h_final: Final hidden state
        """
        b, t, c, h, w = seq.shape
        outputs = []
        h = h_init

        for step in range(t):
            frame_in = seq[:, step, ...]
            denoised_frame, h = self.forward_frame(frame_in, h)
            outputs.append(denoised_frame)

        denoised_seq = torch.stack(outputs, dim=1)
        return denoised_seq, h

    def forward(
        self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None
    ):
        if x.dim() == 5:
            # Sequence input [B, T, C, H, W]
            return self.forward_sequence(x, h_prev)
        else:
            # Single frame input [B, C, H, W]
            return self.forward_frame(x, h_prev)
