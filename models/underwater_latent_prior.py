import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, groups=8):
        super().__init__()
        norm_groups = groups if out_ch % groups == 0 else 1
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(norm_groups, out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels, groups=8):
        super().__init__()
        norm_groups = groups if channels % groups == 0 else 1
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(norm_groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(norm_groups, channels)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        x = self.act(x + residual)
        return x


class FFTStatHead(nn.Module):
    """Build a compact global underwater degradation descriptor from FFT statistics."""

    def __init__(self, out_dim=128, fft_size=64, hidden=64):
        super().__init__()
        self.fft_size = int(fft_size)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False
        )
        low_mask, mid_mask, high_mask = self._make_band_masks(self.fft_size)
        self.register_buffer("low_mask", low_mask, persistent=False)
        self.register_buffer("mid_mask", mid_mask, persistent=False)
        self.register_buffer("high_mask", high_mask, persistent=False)
        self.map_net = nn.Sequential(
            nn.Conv2d(6, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8 if hidden % 8 == 0 else 1, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8 if hidden % 8 == 0 else 1, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.stat_net = nn.Sequential(
            nn.Linear(6, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 2, out_dim),
            nn.LayerNorm(out_dim),
        )

    @staticmethod
    def _make_band_masks(size, low_cutoff=0.18, mid_cutoff=0.42):
        coord = torch.linspace(-1.0, 1.0, int(size))
        yy, xx = torch.meshgrid(coord, coord, indexing="ij")
        radius = torch.sqrt(xx * xx + yy * yy).view(1, 1, int(size), int(size))
        low = (radius <= float(low_cutoff)).float()
        mid = ((radius > float(low_cutoff)) & (radius <= float(mid_cutoff))).float()
        high = (radius > float(mid_cutoff)).float()
        return low, mid, high

    @staticmethod
    def _masked_mean(x, mask, eps=1e-6):
        return (x * mask).sum(dim=(-2, -1)) / mask.sum(dim=(-2, -1)).clamp_min(eps)

    @staticmethod
    def _standardize_map(x, eps=1e-6):
        center = x.mean(dim=(-2, -1), keepdim=True)
        scale = x.std(dim=(-2, -1), keepdim=True).clamp_min(eps)
        return (x - center) / scale

    def forward(self, image):
        device_type = image.device.type if image.device.type in ("cuda", "cpu") else "cuda"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            x = image.float() * self.std.float() + self.mean.float()
            x = x.clamp(0.0, 1.0)
            if x.shape[-2:] != (self.fft_size, self.fft_size):
                x = F.interpolate(x, size=(self.fft_size, self.fft_size), mode="bilinear", align_corners=False)

            amp = torch.fft.fft2(x, norm="ortho").abs()
            amp = torch.fft.fftshift(amp, dim=(-2, -1))
            log_amp = torch.log1p(amp)
            log_amp_norm = self._standardize_map(log_amp)

            luma_log_amp = 0.299 * log_amp[:, 0:1] + 0.587 * log_amp[:, 1:2] + 0.114 * log_amp[:, 2:3]
            low_mask = self.low_mask.to(device=log_amp.device, dtype=log_amp.dtype)
            mid_mask = self.mid_mask.to(device=log_amp.device, dtype=log_amp.dtype)
            high_mask = self.high_mask.to(device=log_amp.device, dtype=log_amp.dtype)
            low_map = self._standardize_map(luma_log_amp * low_mask)
            mid_map = self._standardize_map(luma_log_amp * mid_mask)
            high_map = self._standardize_map(luma_log_amp * high_mask)
            map_feat = self.map_net(torch.cat([log_amp_norm, low_map, mid_map, high_map], dim=1))

            luma_amp = 0.299 * amp[:, 0:1] + 0.587 * amp[:, 1:2] + 0.114 * amp[:, 2:3]
            e_low = self._masked_mean(luma_amp, low_mask)
            e_mid = self._masked_mean(luma_amp, mid_mask)
            e_high = self._masked_mean(luma_amp, high_mask)
            e_total = (e_low + e_mid + e_high).clamp_min(1e-6)
            band_ratios = torch.cat([e_low / e_total, e_mid / e_total, e_high / e_total], dim=1)

            low_rgb = self._masked_mean(amp, low_mask)
            r, g, b = low_rgb[:, 0:1], low_rgb[:, 1:2], low_rgb[:, 2:3]
            channel_ratios = torch.cat(
                [
                    torch.log((r + 1e-6) / (g + b + 1e-6)),
                    torch.log((g + 1e-6) / (r + b + 1e-6)),
                    torch.log((b + 1e-6) / (r + g + 1e-6)),
                ],
                dim=1,
            )
            stat_feat = self.stat_net(torch.cat([band_ratios, channel_ratios], dim=1))
            return self.fuse(torch.cat([map_feat, stat_feat], dim=1))


class UnderwaterLatentPriorEncoder(nn.Module):
    """Global-local latent prior encoder for underwater degradation reasoning.

    Outputs:
      - z_deg: global degradation descriptor, shape [B, global_dim]
      - pyramid: list of 4 spatial degradation priors from high to low resolution
    """

    def __init__(
        self,
        in_ch=3,
        base_ch=32,
        pyramid_channels=(32, 64, 128, 256),
        global_dim=128,
        fft_size=64,
        stat_hidden=64,
        use_fft_prior=True,
    ):
        super().__init__()
        if len(pyramid_channels) != 4:
            raise ValueError("pyramid_channels must contain exactly 4 stages")

        c1, c2, c3, c4 = [int(c) for c in pyramid_channels]
        self.use_fft_prior = bool(use_fft_prior)
        self.stem = nn.Sequential(
            ConvNormAct(in_ch, base_ch, stride=1),
            ConvNormAct(base_ch, c1, stride=1),
            ResidualConvBlock(c1),
        )
        self.stage2 = nn.Sequential(
            ConvNormAct(c1, c2, stride=2),
            ResidualConvBlock(c2),
        )
        self.stage3 = nn.Sequential(
            ConvNormAct(c2, c3, stride=2),
            ResidualConvBlock(c3),
        )
        self.stage4 = nn.Sequential(
            ConvNormAct(c3, c4, stride=2),
            ResidualConvBlock(c4),
        )
        self.pyramid_proj = nn.ModuleList(
            [
                nn.Conv2d(c1, c1, kernel_size=1),
                nn.Conv2d(c2, c2, kernel_size=1),
                nn.Conv2d(c3, c3, kernel_size=1),
                nn.Conv2d(c4, c4, kernel_size=1),
            ]
        )
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c4, global_dim),
            nn.LayerNorm(global_dim),
        )
        self.global_fft = FFTStatHead(out_dim=global_dim, fft_size=fft_size, hidden=stat_hidden)
        self.global_fuse = nn.Sequential(
            nn.Linear(global_dim * 2, global_dim),
            nn.GELU(),
            nn.Linear(global_dim, global_dim),
            nn.LayerNorm(global_dim),
        )

    def forward(self, image):
        p1 = self.stem(image)
        p2 = self.stage2(p1)
        p3 = self.stage3(p2)
        p4 = self.stage4(p3)
        pyramid = [
            self.pyramid_proj[0](p1),
            self.pyramid_proj[1](p2),
            self.pyramid_proj[2](p3),
            self.pyramid_proj[3](p4),
        ]
        z_spatial = self.global_pool(p4)
        z_fft = self.global_fft(image) if self.use_fft_prior else torch.zeros_like(z_spatial)
        z_deg = self.global_fuse(torch.cat([z_spatial, z_fft], dim=1))
        return z_deg, pyramid


class DegMapGenerator(nn.Module):
    """Generate explicit spatial degradation maps from decoder features and latent priors."""

    def __init__(self, feat_ch, prior_ch, hidden_ch=64, out_ch=1):
        super().__init__()
        self.prior_proj = nn.Conv2d(prior_ch, hidden_ch, kernel_size=1, bias=False)
        self.feat_proj = nn.Conv2d(feat_ch, hidden_ch, kernel_size=1, bias=False)
        self.fuse = nn.Sequential(
            ConvNormAct(hidden_ch * 2, hidden_ch, stride=1, groups=8),
            ResidualConvBlock(hidden_ch),
            nn.Conv2d(hidden_ch, out_ch, kernel_size=1),
        )

    def forward(self, feat, prior):
        if feat.shape[-2:] != prior.shape[-2:]:
            prior = F.interpolate(prior, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        fused = torch.cat([self.feat_proj(feat), self.prior_proj(prior)], dim=1)
        return torch.sigmoid(self.fuse(fused))


class MultiScaleDegMapGenerator(nn.Module):
    """Convenience wrapper for 4-scale degradation-map generation."""

    def __init__(self, feat_channels, prior_channels, hidden_channels=None, out_ch=1):
        super().__init__()
        if len(feat_channels) != len(prior_channels):
            raise ValueError("feat_channels and prior_channels must have the same length")
        if hidden_channels is None:
            hidden_channels = [max(32, min(int(f), 128)) for f in feat_channels]
        self.heads = nn.ModuleList(
            [
                DegMapGenerator(f_ch, p_ch, hidden_ch=h_ch, out_ch=out_ch)
                for f_ch, p_ch, h_ch in zip(feat_channels, prior_channels, hidden_channels)
            ]
        )

    def forward(self, feats, priors):
        if len(feats) != len(self.heads) or len(priors) != len(self.heads):
            raise ValueError("feats/priors length must match the number of heads")
        return [head(feat, prior) for head, feat, prior in zip(self.heads, feats, priors)]
