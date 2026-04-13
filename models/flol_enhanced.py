"""
models/flol_enhanced.py
========================
FLOL-FS: Fast Low-Light Enhancement with Frequency Supervision

Architecture (unchanged from original FLOL):
  Input (3ch)
    └─► FIE Step  (Fourier Image Enhancement)  → x_lol  (intermediate)
    └─► Denoiser  (U-Net style, spatial)        → x_hat  (final output)

We deliberately keep the ORIGINAL FLOL architecture unchanged.
Our contribution is entirely in the LOSS FUNCTION (losses.py),
not in model complexity. This is our key research argument:
"Better supervision, not bigger model."

Original FLOL paper: https://arxiv.org/abs/2501.09718
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# BUILDING BLOCKS
# ─────────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation channel attention.
    Reweights feature channels by global context.
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(self.pool(x))          # (B, C)
        return x * w[:, :, None, None]


class ConvBnRelu(nn.Module):
    """Conv → BN → ReLU block."""
    def __init__(self, in_ch: int, out_ch: int,
                 kernel: int = 3, stride: int = 1):
        super().__init__()
        pad = kernel // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                      padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SpatialBlock(nn.Module):
    """
    Residual spatial block with optional Channel Attention.
    Used in encoder, bottleneck, and decoder of DenoiserStep.
    """
    def __init__(self, channels: int, use_ca: bool = True):
        super().__init__()
        self.conv1 = ConvBnRelu(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.ca    = ChannelAttention(channels) if use_ca else nn.Identity()
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.ca(out)
        return self.relu(out + residual)


# ─────────────────────────────────────────────────────────────
# FIE STEP  (Fourier Image Enhancement)
# ─────────────────────────────────────────────────────────────

class FIEStep(nn.Module):
    """
    Fourier Image Enhancement Step.

    Processes image in frequency domain:
      1. Convert Y channel (luminance) to frequency domain via FFT
      2. Learn frequency-domain corrections
      3. Reconstruct enhanced image

    This is the core of FLOL's approach — fast global
    illumination correction via frequency manipulation.
    """

    def __init__(self, channels: int = 16):
        super().__init__()

        # Frequency branch: processes FFT magnitude
        self.freq_branch = nn.Sequential(
            nn.Conv2d(1, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 1),
        )

        # Spatial branch: processes spatial features
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 3, 3, padding=1),
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(4, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Extract luminance (Y channel from RGB)
        y = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)

        # Frequency domain processing
        fft_y   = torch.fft.rfft2(y, norm='ortho')
        mag     = torch.abs(fft_y)
        phase   = torch.angle(fft_y)

        # Learn correction in log-magnitude space (more stable)
        log_mag  = torch.log1p(mag)
        # Resize to spatial dims for CNN processing
        log_mag_spatial = F.interpolate(log_mag, size=(H, W),
                                        mode='bilinear', align_corners=False)
        freq_feat = self.freq_branch(log_mag_spatial)

        # Spatial processing
        spatial_feat = self.spatial_branch(x)

        # Fuse frequency and spatial features
        fused = self.fusion(torch.cat([freq_feat, spatial_feat], dim=1))

        # Apply enhancement as multiplicative correction
        out = x * fused + x
        return torch.clamp(out, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────
# DENOISER STEP  (U-Net style spatial refinement)
# ─────────────────────────────────────────────────────────────

class DenoiserStep(nn.Module):
    """
    U-Net style denoiser for spatial refinement.

    After FIE fixes global illumination, Denoiser:
    - Removes noise amplified by brightness boost
    - Recovers fine spatial details
    - Corrects local color inconsistencies

    Architecture: 3-level encoder → bottleneck → 3-level decoder
    """

    def __init__(self, channels: int = 16, use_ca: bool = True):
        super().__init__()
        c = channels

        # Encoder
        self.enc1 = nn.Sequential(
            ConvBnRelu(3, c),
            SpatialBlock(c, use_ca=use_ca),
        )
        self.down1 = nn.Conv2d(c, c * 2, 2, stride=2)  # /2

        self.enc2 = nn.Sequential(
            ConvBnRelu(c * 2, c * 2),
            SpatialBlock(c * 2, use_ca=use_ca),
        )
        self.down2 = nn.Conv2d(c * 2, c * 4, 2, stride=2)  # /4

        # Bottleneck
        self.bottleneck = nn.Sequential(
            SpatialBlock(c * 4, use_ca=use_ca),
            SpatialBlock(c * 4, use_ca=use_ca),
        )

        # Decoder
        self.up2   = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2  = nn.Sequential(
            ConvBnRelu(c * 4, c * 2),
            SpatialBlock(c * 2, use_ca=use_ca),
        )

        self.up1   = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1  = nn.Sequential(
            ConvBnRelu(c * 2, c),
            SpatialBlock(c, use_ca=use_ca),
        )

        # Output head
        self.head = nn.Sequential(
            nn.Conv2d(c, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))

        # Bottleneck
        b  = self.bottleneck(self.down2(e2))

        # Decoder with skip connections
        d2 = self.dec2(torch.cat([self.up2(b),  e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)


# ─────────────────────────────────────────────────────────────
# FULL MODEL
# ─────────────────────────────────────────────────────────────

class FLOL_CA(nn.Module):
    """
    FLOL-FS: Fast Low-Light Enhancement with Frequency Supervision.

    Two-stage pipeline:
      Stage 1 → FIEStep:      Fast global illumination via FFT
      Stage 2 → DenoiserStep: Spatial refinement + denoising

    Returns both outputs for dual supervision during training:
      x_lol = FIE output  (supervised by LOL auxiliary loss)
      x_hat = final output (supervised by full loss)

    At inference, only x_hat is used.

    Parameters:
      channels : int  base feature channels (default 16 = lightweight)
      use_ca   : bool enable Channel Attention in Denoiser (default True)
    """

    def __init__(self, channels: int = 16, use_ca: bool = True):
        super().__init__()
        self.fie      = FIEStep(channels=channels)
        self.denoiser = DenoiserStep(channels=channels, use_ca=use_ca)

    def forward(self, x: torch.Tensor):
        x_lol = self.fie(x)           # Stage 1 output
        x_hat = self.denoiser(x_lol)  # Stage 2 output
        return x_hat, x_lol

    def enhance(self, x: torch.Tensor) -> torch.Tensor:
        """Inference-only forward pass. Returns final output only."""
        x_hat, _ = self.forward(x)
        return x_hat


# ─────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    model  = FLOL_CA(channels=16, use_ca=True)
    x      = torch.rand(2, 3, 128, 128)
    x_hat, x_lol = model(x)

    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params     : {n/1e3:.1f}K")
    print(f"Input      : {x.shape}")
    print(f"x_lol      : {x_lol.shape}")
    print(f"x_hat      : {x_hat.shape}")
    print(f"Output range: [{x_hat.min():.3f}, {x_hat.max():.3f}]")
