import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose

from depth_anything_v2.dinov2 import DINOv2
from depth_anything_v2.util.blocks import FeatureFusionBlock, _make_scratch
from depth_anything_v2.util.transform import NormalizeImage, PrepareForNet, Resize

from .underwater_latent_prior import MultiScaleDegMapGenerator, UnderwaterLatentPriorEncoder


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class GlobalDegradationModulation(nn.Module):
    """FiLM-style global modulation from latent degradation descriptor."""

    def __init__(self, latent_dim, feat_dim):
        super().__init__()
        self.to_gamma = nn.Sequential(
            nn.Linear(latent_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, feat_dim),
        )
        self.to_beta = nn.Sequential(
            nn.Linear(latent_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, feat_dim),
        )
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.reset_parameters()

    def reset_parameters(self):
        # Keep the residual branch at zero initially, but retain a live path
        # from z_deg to the zero-initialized output layer.
        for branch in (self.to_gamma, self.to_beta):
            nn.init.xavier_uniform_(branch[0].weight)
            nn.init.zeros_(branch[0].bias)
            nn.init.zeros_(branch[2].weight)
            nn.init.zeros_(branch[2].bias)

    def forward(self, feat, z_deg):
        gamma = self.to_gamma(z_deg).unsqueeze(-1).unsqueeze(-1)
        beta = self.to_beta(z_deg).unsqueeze(-1).unsqueeze(-1)
        scale = torch.sigmoid(self.scale)
        return feat * (1.0 + scale * gamma) + scale * beta


class PlainResidualConvAdapter(nn.Module):
    """Parameter-matched content adapter without degradation conditioning."""

    def __init__(self, channels, hidden_channels=256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, feat):
        return feat + self.body(feat)


class LatentPriorDPTHead(nn.Module):
    """Depth head with latent-prior bottleneck modulation and explicit deg-map fusion."""

    def __init__(
        self,
        in_channels,
        latent_dim=128,
        prior_channels=(32, 64, 128, 256),
        features=256,
        use_bn=False,
        out_channels=(256, 512, 1024, 1024),
        use_clstoken=False,
        deg_map_scale=0.2,
        use_global_prior=True,
        use_local_prior=True,
        use_deg_map=True,
        deg_map_spatial_mean=False,
        use_plain_adapter=False,
        adapter_hidden=256,
    ):
        super().__init__()
        self.use_clstoken = use_clstoken
        self.deg_map_scale = float(deg_map_scale)
        self.use_global_prior = bool(use_global_prior)
        self.use_local_prior = bool(use_local_prior)
        self.use_deg_map = bool(use_deg_map)
        self.deg_map_spatial_mean = bool(deg_map_spatial_mean)
        self.use_plain_adapter = bool(use_plain_adapter)
        if self.deg_map_spatial_mean and not self.use_deg_map:
            raise ValueError("deg_map_spatial_mean requires use_deg_map=True")

        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for out_channel in out_channels
            ]
        )

        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0],
                    out_channels=out_channels[0],
                    kernel_size=4,
                    stride=4,
                    padding=0,
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1],
                    out_channels=out_channels[1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3],
                    out_channels=out_channels[3],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU(),
                    )
                )

        self.scratch = _make_scratch(out_channels, features, groups=1, expand=False)
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32
        self.scratch.output_conv1 = nn.Conv2d(
            head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1
        )
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
        )

        self.global_mod = GlobalDegradationModulation(latent_dim, features)
        self.deg_map_generator = MultiScaleDegMapGenerator(
            feat_channels=[features, features, features, features],
            prior_channels=list(prior_channels),
            hidden_channels=[max(32, features // 2)] * 4,
            out_ch=1,
        )
        self.prior_to_feat = nn.ModuleList(
            [nn.Conv2d(ch, features, kernel_size=1, bias=False) for ch in prior_channels]
        )
        if self.use_local_prior and not self.use_deg_map:
            self.scalar_gate_logits = nn.Parameter(torch.zeros(4))
        else:
            self.register_parameter("scalar_gate_logits", None)
        self.plain_adapters = nn.ModuleList(
            [PlainResidualConvAdapter(features, adapter_hidden) for _ in range(4)]
        ) if self.use_plain_adapter else nn.ModuleList()
        self.reset_prior_injection()

    def reset_prior_injection(self):
        # Start exactly from the pretrained depth head. The prior branch learns
        # residual corrections instead of perturbing baseline features at step 0.
        for projection in self.prior_to_feat:
            nn.init.zeros_(projection.weight)

    def _inject_deg_prior(self, feat, prior, deg_map):
        if prior.shape[-2:] != feat.shape[-2:]:
            prior = F.interpolate(prior, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        correction = self.prior_to_feat[self._deg_index](prior)
        return feat + self.deg_map_scale * correction * deg_map

    def forward(self, out_features, patch_h, patch_w, z_deg, prior_pyramid, return_aux=False):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        layer_feats = [layer_1_rn, layer_2_rn, layer_3_rn, layer_4_rn]
        if self.use_plain_adapter:
            layer_feats = [adapter(feat) for adapter, feat in zip(self.plain_adapters, layer_feats)]
            layer_1_rn, layer_2_rn, layer_3_rn, layer_4_rn = layer_feats

        if self.use_local_prior:
            if self.use_deg_map:
                deg_maps = self.deg_map_generator(layer_feats, prior_pyramid)
                if self.deg_map_spatial_mean:
                    deg_maps = [
                        deg_map.mean(dim=(-2, -1), keepdim=True).expand_as(deg_map)
                        for deg_map in deg_maps
                    ]
            else:
                deg_maps = [
                    torch.sigmoid(self.scalar_gate_logits[index]).to(dtype=feat.dtype)
                    .view(1, 1, 1, 1)
                    .expand(feat.shape[0], 1, *feat.shape[-2:])
                    for index, feat in enumerate(layer_feats)
                ]
            injected = []
            for index, (feat, prior, deg_map) in enumerate(zip(layer_feats, prior_pyramid, deg_maps)):
                self._deg_index = index
                injected.append(self._inject_deg_prior(feat, prior, deg_map))
            layer_1_rn, layer_2_rn, layer_3_rn, layer_4_rn = injected
        else:
            deg_maps = [feat.new_zeros(feat.shape[0], 1, *feat.shape[-2:]) for feat in layer_feats]

        if self.use_global_prior:
            layer_4_rn = self.global_mod(layer_4_rn, z_deg)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(
            out,
            (int(patch_h * 14), int(patch_w * 14)),
            mode="bilinear",
            align_corners=True,
        )
        out = self.scratch.output_conv2(out)

        if not return_aux:
            return out

        aux = {
            "z_deg": z_deg,
            "prior_pyramid": prior_pyramid,
            "deg_maps": deg_maps,
            "decoder_feats": {
                "layer_1_rn": layer_1_rn,
                "layer_2_rn": layer_2_rn,
                "layer_3_rn": layer_3_rn,
                "layer_4_rn": layer_4_rn,
                "path_4": path_4,
                "path_3": path_3,
                "path_2": path_2,
                "path_1": path_1,
            },
        }
        return out, aux


class DepthAnythingLatentPrior(nn.Module):
    """Depth Anything with UnderwaterLatentPriorEncoder, separate from LoRA baseline."""

    def __init__(
        self,
        encoder="vits",
        features=64,
        out_channels=(48, 96, 192, 384),
        max_depth=40.0,
        prior_base_ch=32,
        prior_channels=(32, 64, 128, 256),
        latent_dim=128,
        prior_fft_size=64,
        prior_stat_hidden=64,
        deg_map_scale=0.2,
        use_global_prior=True,
        use_local_prior=True,
        use_fft_prior=True,
        use_deg_map=True,
        deg_map_spatial_mean=False,
        use_plain_adapter=False,
        adapter_hidden=256,
    ):
        super().__init__()
        self.intermediate_layer_idx = {
            "vits": [2, 5, 8, 11],
            "vitb": [2, 5, 8, 11],
            "vitl": [4, 11, 17, 23],
            "vitg": [9, 19, 29, 39],
        }
        self.max_depth = max_depth
        self.encoder = encoder
        self.use_global_prior = bool(use_global_prior)
        self.use_local_prior = bool(use_local_prior)
        self.use_fft_prior = bool(use_fft_prior)
        self.use_deg_map = bool(use_deg_map)
        self.deg_map_spatial_mean = bool(deg_map_spatial_mean)
        self.use_plain_adapter = bool(use_plain_adapter)
        self.latent_dim = int(latent_dim)
        self.pretrained = DINOv2(model_name=encoder)
        self.latent_prior_encoder = UnderwaterLatentPriorEncoder(
            in_ch=3,
            base_ch=prior_base_ch,
            pyramid_channels=prior_channels,
            global_dim=latent_dim,
            fft_size=prior_fft_size,
            stat_hidden=prior_stat_hidden,
            use_fft_prior=self.use_fft_prior,
        )
        self.depth_head = LatentPriorDPTHead(
            in_channels=self.pretrained.embed_dim,
            latent_dim=latent_dim,
            prior_channels=prior_channels,
            features=features,
            use_bn=False,
            out_channels=list(out_channels),
            use_clstoken=False,
            deg_map_scale=deg_map_scale,
            use_global_prior=self.use_global_prior,
            use_local_prior=self.use_local_prior,
            use_deg_map=self.use_deg_map,
            deg_map_spatial_mean=self.deg_map_spatial_mean,
            use_plain_adapter=self.use_plain_adapter,
            adapter_hidden=adapter_hidden,
        )

    def load_base_weights(self, state_dict, strict=False):
        own_state = self.state_dict()
        filtered_state = {}
        new_branch_prefixes = (
            "latent_prior_encoder.",
            "depth_head.global_mod.",
            "depth_head.deg_map_generator.",
            "depth_head.prior_to_feat.",
            "depth_head.scalar_gate_logits",
            "depth_head.plain_adapters.",
        )
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[len("module.") :]
            if key not in own_state:
                continue
            if own_state[key].shape != value.shape:
                continue
            filtered_state[key] = value

        expected_base_keys = {
            key for key in own_state if not key.startswith(new_branch_prefixes)
        }
        loaded_base_keys = expected_base_keys.intersection(filtered_state)
        missing_base_keys = sorted(expected_base_keys - loaded_base_keys)
        self.base_load_stats = {
            "loaded": len(loaded_base_keys),
            "expected": len(expected_base_keys),
            "missing": missing_base_keys,
        }
        return self.load_state_dict(filtered_state, strict=strict)

    def freeze_pretrained_backbone(self):
        for param in self.pretrained.parameters():
            param.requires_grad = False

    def unfreeze_pretrained_backbone(self):
        for param in self.pretrained.parameters():
            param.requires_grad = True

    def freeze_base_depth_head(self):
        protected_prefixes = (
            "global_mod.",
            "deg_map_generator.",
            "prior_to_feat.",
            "scalar_gate_logits",
            "plain_adapters.",
        )
        for name, param in self.depth_head.named_parameters():
            param.requires_grad = any(name.startswith(prefix) for prefix in protected_prefixes)

    def unfreeze_base_depth_head(self):
        for param in self.depth_head.parameters():
            param.requires_grad = True

    def freeze_latent_prior_encoder(self):
        for param in self.latent_prior_encoder.parameters():
            param.requires_grad = False

    def unfreeze_latent_prior_encoder(self):
        for param in self.latent_prior_encoder.parameters():
            param.requires_grad = True

    @staticmethod
    def _set_module_trainable(module, trainable):
        for param in module.parameters():
            param.requires_grad = bool(trainable)

    def freeze_disabled_prior_components(self):
        if not self.use_deg_map:
            self._set_module_trainable(self.depth_head.deg_map_generator, False)

        if not self.use_local_prior:
            self._set_module_trainable(self.depth_head.deg_map_generator, False)
            self._set_module_trainable(self.depth_head.prior_to_feat, False)
            self._set_module_trainable(self.latent_prior_encoder.pyramid_proj, False)

        if not self.use_global_prior:
            self._set_module_trainable(self.depth_head.global_mod, False)
            self._set_module_trainable(self.latent_prior_encoder.global_pool, False)
            self._set_module_trainable(self.latent_prior_encoder.global_fft, False)
            self._set_module_trainable(self.latent_prior_encoder.global_fuse, False)
        elif not self.use_fft_prior:
            self._set_module_trainable(self.latent_prior_encoder.global_fft, False)

        if not self.use_global_prior and not self.use_local_prior:
            self._set_module_trainable(self.latent_prior_encoder, False)

        if not self.use_plain_adapter:
            self._set_module_trainable(self.depth_head.plain_adapters, False)

    def configure_trainable(
        self,
        freeze_backbone=True,
        freeze_base_head=False,
        train_latent_prior=True,
    ):
        if freeze_backbone:
            self.freeze_pretrained_backbone()
        else:
            self.unfreeze_pretrained_backbone()

        if freeze_base_head:
            self.freeze_base_depth_head()
        else:
            self.unfreeze_base_depth_head()

        if train_latent_prior:
            self.unfreeze_latent_prior_encoder()
        else:
            self.freeze_latent_prior_encoder()
        self.freeze_disabled_prior_components()

    def forward(self, image, return_aux=False):
        patch_h, patch_w = image.shape[-2] // 14, image.shape[-1] // 14
        out_features = self.pretrained.get_intermediate_layers(
            image, self.intermediate_layer_idx[self.encoder], return_class_token=True
        )
        if self.use_global_prior or self.use_local_prior:
            z_deg, prior_pyramid = self.latent_prior_encoder(image)
        else:
            z_deg = image.new_zeros(image.shape[0], self.latent_dim)
            prior_pyramid = [None] * 4
        depth = self.depth_head(
            out_features,
            patch_h,
            patch_w,
            z_deg,
            prior_pyramid,
            return_aux=return_aux,
        )
        if return_aux:
            pred, aux = depth
            return pred.squeeze(1), aux
        return depth.squeeze(1)

    @torch.no_grad()
    def infer_image(self, raw_image, input_size=518, return_aux=False):
        image, (h, w) = self.image2tensor(raw_image, input_size)
        output = self.forward(image, return_aux=return_aux)
        if return_aux:
            depth, aux = output
        else:
            depth, aux = output, None

        depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
        if not return_aux:
            return depth.cpu().numpy()

        aux["deg_maps_fullres"] = [
            F.interpolate(m, (h, w), mode="bilinear", align_corners=False)[0, 0].cpu()
            for m in aux["deg_maps"]
        ]
        return depth.cpu().numpy(), aux

    def image2tensor(self, raw_image, input_size=518):
        transform = Compose(
            [
                Resize(
                    width=input_size,
                    height=input_size,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )
        h, w = raw_image.shape[:2]
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        image = transform({"image": image})["image"]
        image = torch.from_numpy(image).unsqueeze(0)
        image = image.to(next(self.parameters()).device)
        return image, (h, w)
