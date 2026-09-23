"""
Attention U-Net for Pixel-Level SAR Oil Spill Detection and Segmentation.

Architecture:
  SAR Image
      ↓
  Preprocessing (Min-Max / Standard Scale)
      ↓
  Contracting Path (4-level Encoder with Double Convolutions)
      ↓
  Bottleneck (1024-dim Latent Representation)
      ↓
  Expanding Path (4-level Decoder with Attention Gates on Skip Connections)
      ↓
  Pixel Logit Map (1x1 Conv)
      ↓
  Sigmoid -> Pixel Probability Map
      ↓
  Thresholding (e.g. 0.5) -> Binary Mask (0=Non-Oil, 1=Oil Spill)
"""

import math
from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class DoubleConv(nn.Module):
        """(Convolution => [BN] => ReLU) * 2"""

        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.double_conv(x)


    class AttentionGate(nn.Module):
        """
        Attention Gate mechanism that filters the skip connection features
        using the gating signal from the coarser (deeper) decoder layer.
        Suppresses irrelevant ocean clutter and highlights dark-slick radar backscatter drops.
        """

        def __init__(self, f_g: int, f_l: int, f_int: int):
            """
            f_g: number of channels in gating signal
            f_l: number of channels in skip connection (encoder)
            f_int: number of intermediate channels
            """
            super().__init__()
            self.w_g = nn.Sequential(
                nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(f_int)
            )

            self.w_x = nn.Sequential(
                nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(f_int)
            )

            self.psi = nn.Sequential(
                nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )

            self.relu = nn.ReLU(inplace=True)

        def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            """
            g: gating signal from decoder
            x: skip connection from encoder
            """
            g1 = self.w_g(g)
            x1 = self.w_x(x)

            # Ensure spatial alignment
            if g1.shape[2:] != x1.shape[2:]:
                g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=True)

            psi = self.relu(g1 + x1)
            psi = self.psi(psi)

            return x * psi


    class AttentionUNet(nn.Module):
        """
        Attention U-Net for SAR Oil Spill Pixel-Level Segmentation.
        Outputs single-channel raw logits (pass through Sigmoid for pixel probability).
        """

        def __init__(self, in_channels: int = 3, out_channels: int = 1):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels

            # Encoder (Contracting Path)
            self.inc = DoubleConv(in_channels, 64)
            self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
            self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
            self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
            self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

            # Decoder (Expanding Path with Attention Gates)
            self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
            self.att1 = AttentionGate(f_g=512, f_l=512, f_int=256)
            self.conv1 = DoubleConv(1024, 512)

            self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
            self.att2 = AttentionGate(f_g=256, f_l=256, f_int=128)
            self.conv2 = DoubleConv(512, 256)

            self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
            self.att3 = AttentionGate(f_g=128, f_l=128, f_int=64)
            self.conv3 = DoubleConv(256, 128)

            self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            self.att4 = AttentionGate(f_g=64, f_l=64, f_int=32)
            self.conv4 = DoubleConv(128, 64)

            # Final 1x1 classification layer
            self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Encoder
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)

            # Decoder with Attention Gates
            d1 = self.up1(x5)
            x4_att = self.att1(g=d1, x=x4)
            d1 = torch.cat([x4_att, d1], dim=1)
            d1 = self.conv1(d1)

            d2 = self.up2(d1)
            x3_att = self.att2(g=d2, x=x3)
            d2 = torch.cat([x3_att, d2], dim=1)
            d2 = self.conv2(d2)

            d3 = self.up3(d2)
            x2_att = self.att3(g=d3, x=x2)
            d3 = torch.cat([x2_att, d3], dim=1)
            d3 = self.conv3(d3)

            d4 = self.up4(d3)
            x1_att = self.att4(g=d4, x=x1)
            d4 = torch.cat([x1_att, d4], dim=1)
            d4 = self.conv4(d4)

            logits = self.outc(d4)
            return logits

        def predict_probability(self, x: torch.Tensor) -> torch.Tensor:
            """Convenience forward pass returning sigmoid probabilities in [0, 1]."""
            logits = self.forward(x)
            return torch.sigmoid(logits)

else:
    class AttentionUNet:
        """Fallback class when PyTorch is not available."""
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required to instantiate AttentionUNet.")


def get_attention_unet(in_channels: int = 3, out_channels: int = 1) -> AttentionUNet:
    """Factory function for Attention U-Net."""
    return AttentionUNet(in_channels=in_channels, out_channels=out_channels)


if __name__ == "__main__":
    if TORCH_AVAILABLE:
        model = get_attention_unet(in_channels=3, out_channels=1)
        dummy_input = torch.randn(1, 3, 256, 256)
        output = model(dummy_input)
        print("Attention U-Net initialized successfully.")
        print(f"Input shape:  {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params:,}")
    else:
        print("PyTorch is not available.")
