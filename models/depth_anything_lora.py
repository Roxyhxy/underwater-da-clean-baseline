import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from depth_anything_v2.dpt import DepthAnythingV2


class LoRALinear(nn.Module):
    def __init__(self, linear, rank=8, alpha=16.0, dropout=0.0):
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError("LoRALinear expects nn.Linear")
        self.linear = linear
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(self.rank, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_a = nn.Linear(linear.in_features, self.rank, bias=False)
        self.lora_b = nn.Linear(self.rank, linear.out_features, bias=False)
        self.enabled = True
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        self.to(device=linear.weight.device, dtype=linear.weight.dtype)
        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.linear(x)
        if not self.enabled:
            return out
        return out + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling


class ConvLoRA2d(nn.Module):
    def __init__(self, conv, rank=2, alpha=4.0, dropout=0.0):
        super().__init__()
        if not isinstance(conv, nn.Conv2d):
            raise TypeError("ConvLoRA2d expects nn.Conv2d")
        if conv.groups != 1:
            raise ValueError("ConvLoRA2d only supports groups=1")
        self.conv = conv
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(self.rank, 1)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.lora_down = nn.Conv2d(conv.in_channels, self.rank, kernel_size=1, bias=False)
        self.lora_up = nn.Conv2d(self.rank, conv.out_channels, kernel_size=1, bias=False)
        self.enabled = True
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        self.to(device=conv.weight.device, dtype=conv.weight.dtype)
        for param in self.conv.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.conv(x)
        if not self.enabled:
            return out
        delta = self.lora_up(self.lora_down(self.dropout(x)))
        return out + delta * self.scaling


class AquaDegradeEncoder(nn.Module):
    def __init__(self, style_dim=128, hidden=64, fft_size=64):
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
            nn.Linear(hidden * 2, style_dim),
            nn.LayerNorm(style_dim),
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
    def _standardize_map(x, eps=1e-6):
        center = x.mean(dim=(-2, -1), keepdim=True)
        scale = x.std(dim=(-2, -1), keepdim=True).clamp_min(eps)
        return (x - center) / scale

    @staticmethod
    def _masked_mean(x, mask, eps=1e-6):
        return (x * mask).sum(dim=(-2, -1)) / mask.sum(dim=(-2, -1)).clamp_min(eps)

    def forward(self, image):
        device_type = image.device.type if image.device.type in ("cuda", "cpu") else "cuda"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            x = image.float() * self.std.float() + self.mean.float()
            x = x.clamp(0.0, 1.0)
            if x.shape[-2:] != (self.fft_size, self.fft_size):
                x = F.interpolate(
                    x, size=(self.fft_size, self.fft_size), mode="bilinear", align_corners=False
                )

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


class AquaStyleLoRALinear(nn.Module):
    def __init__(self, linear, rank=8, alpha=16.0, dropout=0.0, style_dim=128):
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError("AquaStyleLoRALinear expects nn.Linear")
        self.linear = linear
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(self.rank, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_a = nn.Linear(linear.in_features, self.rank, bias=False)
        self.lora_b = nn.Linear(self.rank, linear.out_features, bias=False)
        self.style_gate = nn.Linear(style_dim, self.rank)
        self.enabled = True
        self._style = None
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        nn.init.zeros_(self.style_gate.weight)
        nn.init.zeros_(self.style_gate.bias)
        self.to(device=linear.weight.device, dtype=linear.weight.dtype)
        for param in self.linear.parameters():
            param.requires_grad = False

    def set_style(self, style):
        self._style = style

    def forward(self, x):
        out = self.linear(x)
        if not self.enabled:
            return out
        z = self.lora_a(self.dropout(x))
        if self._style is not None:
            gate = 2.0 * torch.sigmoid(self.style_gate(self._style.to(device=x.device, dtype=x.dtype)))
            z = z * gate[:, None, :]
        return out + self.lora_b(z) * self.scaling


def _get_parent_module(root, name):
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _set_child_module(parent, child_name, module):
    setattr(parent, child_name, module)


def _block_index_from_name(name):
    match = re.search(r"blocks\.([0-9]+)\.", name)
    if match is None:
        return None
    return int(match.group(1))


def inject_lora(model, rank=8, alpha=16.0, dropout=0.0, target="qkv", last_n_blocks=12, mode="aquadegrade", style_dim=128):
    pretrained = model.base.pretrained
    n_blocks = len(pretrained.blocks)
    first_block = max(0, n_blocks - int(last_n_blocks))
    allow = {"qkv"}
    if target == "qkv_proj":
        allow = {"qkv", "proj"}
    elif target == "qkv_mlp":
        allow = {"qkv", "fc1", "fc2"}
    elif target != "qkv":
        raise ValueError(f"Unsupported LoRA target: {target}")

    injected = []
    for name, module in list(pretrained.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        block_idx = _block_index_from_name(name)
        if block_idx is None or block_idx < first_block:
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in allow:
            continue
        parent, child_name = _get_parent_module(pretrained, name)
        if mode == "plain":
            wrapped = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        elif mode == "aquadegrade":
            wrapped = AquaStyleLoRALinear(
                module, rank=rank, alpha=alpha, dropout=dropout, style_dim=style_dim
            )
        else:
            raise ValueError(f"Unsupported clean baseline LoRA mode: {mode}")
        _set_child_module(parent, child_name, wrapped)
        injected.append(name)
    return injected


def _decoder_convlora_match(name, target):
    if target == "post_vfe":
        return (
            name.startswith("scratch.refinenet")
            or name == "scratch.output_conv1"
            or name == "scratch.output_conv2.0"
        )
    if target == "refinenet":
        return name.startswith("scratch.refinenet")
    if target == "tail":
        return name in {"scratch.output_conv1", "scratch.output_conv2.0"}
    if target == "all":
        return True
    raise ValueError(f"Unsupported decoder ConvLoRA target: {target}")


def inject_decoder_convlora(model, rank=2, alpha=4.0, dropout=0.0, target="tail"):
    depth_head = model.base.depth_head
    injected = []
    for name, module in list(depth_head.named_modules()):
        if isinstance(module, ConvLoRA2d):
            continue
        if not isinstance(module, nn.Conv2d):
            continue
        if module.groups != 1:
            continue
        if not _decoder_convlora_match(name, target):
            continue
        parent, child_name = _get_parent_module(depth_head, name)
        _set_child_module(parent, child_name, ConvLoRA2d(module, rank=rank, alpha=alpha, dropout=dropout))
        injected.append(f"depth_head.{name}")
    return injected


class DepthAnythingLoRA(nn.Module):
    def __init__(
        self,
        encoder="vits",
        features=64,
        out_channels=(48, 96, 192, 384),
        max_depth=40.0,
        lora_rank=8,
        lora_alpha=16.0,
        lora_dropout=0.0,
        lora_target="qkv",
        lora_last_n_blocks=12,
        lora_mode="aquadegrade",
        style_dim=128,
        style_hidden=64,
        style_fft_size=64,
        use_decoder_lora=False,
        decoder_lora_rank=2,
        decoder_lora_alpha=4.0,
        decoder_lora_dropout=0.0,
        decoder_lora_target="tail",
    ):
        super().__init__()
        self.base = DepthAnythingV2(
            encoder=encoder,
            features=features,
            out_channels=list(out_channels),
            use_bn=False,
            use_clstoken=False,
            max_depth=max_depth,
        )
        self.lora_config = {
            "rank": lora_rank,
            "alpha": lora_alpha,
            "dropout": lora_dropout,
            "target": lora_target,
            "last_n_blocks": lora_last_n_blocks,
            "mode": lora_mode,
            "style_dim": style_dim,
        }
        self.decoder_lora_config = {
            "enabled": bool(use_decoder_lora),
            "rank": decoder_lora_rank,
            "alpha": decoder_lora_alpha,
            "dropout": decoder_lora_dropout,
            "target": decoder_lora_target,
        }
        self.style_encoder = None
        if lora_mode == "aquadegrade":
            self.style_encoder = AquaDegradeEncoder(
                style_dim=style_dim, hidden=style_hidden, fft_size=style_fft_size
            )
        self.injected_lora = []
        self.injected_decoder_lora = []

    def load_base_weights(self, state_dict, strict=False):
        return self.base.load_state_dict(state_dict, strict=strict)

    def freeze_base_and_inject_lora(self):
        for param in self.base.parameters():
            param.requires_grad = False
        self.injected_lora = inject_lora(
            self,
            rank=self.lora_config["rank"],
            alpha=self.lora_config["alpha"],
            dropout=self.lora_config["dropout"],
            target=self.lora_config["target"],
            last_n_blocks=self.lora_config["last_n_blocks"],
            mode=self.lora_config["mode"],
            style_dim=self.lora_config["style_dim"],
        )
        if self.style_encoder is not None:
            for param in self.style_encoder.parameters():
                param.requires_grad = True
        if self.decoder_lora_config["enabled"]:
            self.injected_decoder_lora = inject_decoder_convlora(
                self,
                rank=self.decoder_lora_config["rank"],
                alpha=self.decoder_lora_config["alpha"],
                dropout=self.decoder_lora_config["dropout"],
                target=self.decoder_lora_config["target"],
            )

    def _set_style_context(self, style):
        for module in self.modules():
            if isinstance(module, AquaStyleLoRALinear):
                module.set_style(style)

    def _lora_modules(self):
        for module in self.modules():
            if isinstance(module, (LoRALinear, AquaStyleLoRALinear, ConvLoRA2d)):
                yield module

    def set_lora_enabled(self, enabled):
        for module in self._lora_modules():
            module.enabled = bool(enabled)

    def _baseline_disp(self, image):
        states = [(module, module.enabled) for module in self._lora_modules()]
        self.set_lora_enabled(False)
        try:
            return self.base(image).detach()
        finally:
            for module, enabled in states:
                module.enabled = enabled

    def forward(self, image):
        style = self.style_encoder(image) if self.style_encoder is not None else None
        self._set_style_context(style)
        try:
            return self.base(image)
        finally:
            self._set_style_context(None)

    @torch.no_grad()
    def infer_image(self, raw_image, input_size=518):
        image, (h, w) = self.base.image2tensor(raw_image, input_size)
        depth = self.forward(image)
        depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
        return depth.cpu().numpy()


def extract_lora_state_dict(model):
    prefixes = (
        "style_encoder.",
        "base.pretrained.",
        "base.depth_head.",
    )
    state = model.state_dict()
    keep = {}
    for key, value in state.items():
        if key.startswith("style_encoder."):
            keep[key] = value
        elif ".lora_" in key:
            keep[key] = value
    return keep
