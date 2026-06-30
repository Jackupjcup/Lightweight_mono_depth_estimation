"""Student DualDPT head adapted for MobileNetV2 multi-channel spatial features.

Building blocks (FeatureFusionBlock, ResidualConvUnit, _make_scratch, _make_fusion_block)
are copied from DA3's dpt.py and head_utils.py with minimal changes:
  - FloatFunctional replaced with plain addition (no quantisation-aware training)
  - LayerNorm / token reshape removed (student input is already [B, C, H, W])
  - resize_layers removed (MobileNetV2 naturally has multi-scale features)
  - projects accept per-stage dim_in instead of a single dim_in
"""

from typing import List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from addict import Dict


# ---------------------------------------------------------------------------
# Positional embedding utilities (from DA3 head_utils.py)
# ---------------------------------------------------------------------------

class Permute(nn.Module):
    def __init__(self, dims: Tuple[int, ...]):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self.dims)


def create_uv_grid(
    width: int,
    height: int,
    aspect_ratio: float = None,
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> torch.Tensor:
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)
    diag_factor = (aspect_ratio ** 2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag_factor
    span_y = 1.0 / diag_factor
    left_x = -span_x * (width - 1) / width
    right_x = span_x * (width - 1) / width
    top_y = -span_y * (height - 1) / height
    bottom_y = span_y * (height - 1) / height
    x_coords = torch.linspace(left_x, right_x, steps=width, dtype=dtype, device=device)
    y_coords = torch.linspace(top_y, bottom_y, steps=height, dtype=dtype, device=device)
    uu, vv = torch.meshgrid(x_coords, y_coords, indexing="xy")
    return torch.stack((uu, vv), dim=-1)


def _make_sincos_pos_embed(embed_dim: int, pos: torch.Tensor, omega_0: float = 100) -> torch.Tensor:
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / omega_0 ** omega
    pos = pos.reshape(-1)
    out = torch.einsum("m,d->md", pos, omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1).float()


def position_grid_to_embed(pos_grid: torch.Tensor, embed_dim: int, omega_0: float = 100) -> torch.Tensor:
    H, W, _ = pos_grid.shape
    pos_flat = pos_grid.reshape(-1, 2)
    emb_x = _make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 0], omega_0)
    emb_y = _make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 1], omega_0)
    return torch.cat([emb_x, emb_y], dim=-1).view(H, W, embed_dim)


def custom_interpolate(
    x: torch.Tensor,
    size: Tuple[int, int] = None,
    scale_factor: float = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    if size is None:
        assert scale_factor is not None
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))
    INT_MAX = 1610612736
    total = size[0] * size[1] * x.shape[0] * x.shape[1]
    if total > INT_MAX:
        chunks = torch.chunk(x, chunks=(total // INT_MAX) + 1, dim=0)
        outs = [F.interpolate(c, size=size, mode=mode, align_corners=align_corners) for c in chunks]
        return torch.cat(outs, dim=0).contiguous()
    return F.interpolate(x, size=size, mode=mode, align_corners=align_corners)


# ---------------------------------------------------------------------------
# FPN building blocks (from DA3 dpt.py, FloatFunctional → plain add)
# ---------------------------------------------------------------------------

class ResidualConvUnit(nn.Module):
    def __init__(self, features: int, activation: nn.Module, bn: bool, groups: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, 1, 1, bias=True, groups=groups)
        self.conv2 = nn.Conv2d(features, features, 3, 1, 1, bias=True, groups=groups)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    def __init__(
        self,
        features: int,
        activation: nn.Module,
        deconv: bool = False,
        bn: bool = False,
        expand: bool = False,
        align_corners: bool = True,
        size: Tuple[int, int] = None,
        has_residual: bool = True,
        groups: int = 1,
    ):
        super().__init__()
        self.align_corners = align_corners
        self.size = size
        self.has_residual = has_residual
        self.resConfUnit1 = (
            ResidualConvUnit(features, activation, bn, groups=groups) if has_residual else None
        )
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn, groups=groups)
        out_features = (features // 2) if expand else features
        self.out_conv = nn.Conv2d(features, out_features, 1, 1, 0, bias=True, groups=groups)

    def forward(self, *xs: torch.Tensor, size: Tuple[int, int] = None) -> torch.Tensor:
        y = xs[0]
        if self.has_residual and len(xs) > 1 and self.resConfUnit1 is not None:
            y = y + self.resConfUnit1(xs[1])
        y = self.resConfUnit2(y)
        if (size is None) and (self.size is None):
            up_kwargs = {"scale_factor": 2}
        elif size is None:
            up_kwargs = {"size": self.size}
        else:
            up_kwargs = {"size": size}
        y = custom_interpolate(y, **up_kwargs, mode="bilinear", align_corners=self.align_corners)
        y = self.out_conv(y)
        return y


def _make_fusion_block(features: int, size=None, has_residual=True, groups=1, inplace=False):
    return FeatureFusionBlock(
        features=features,
        activation=nn.ReLU(inplace=inplace),
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=size,
        has_residual=has_residual,
        groups=groups,
    )


def _make_scratch(in_shape: List[int], out_shape: int, groups: int = 1, expand: bool = False):
    scratch = nn.Module()
    c1 = out_shape
    c2 = out_shape * (2 if expand else 1)
    c3 = out_shape * (4 if expand else 1)
    c4 = out_shape * (8 if expand else 1)
    scratch.layer1_rn = nn.Conv2d(in_shape[0], c1, 3, 1, 1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], c2, 3, 1, 1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], c3, 3, 1, 1, bias=False, groups=groups)
    scratch.layer4_rn = nn.Conv2d(in_shape[3], c4, 3, 1, 1, bias=False, groups=groups)
    return scratch


# ---------------------------------------------------------------------------
# Student DualDPT
# ---------------------------------------------------------------------------

class StudentDualDPT(nn.Module):
    """DualDPT adapted for MobileNetV2 spatial feature inputs.

    Differences from the DA3 DualDPT:
      1. projects: per-stage input channels (dim_ins) instead of uniform dim_in
      2. No LayerNorm (student uses BN in backbone)
      3. No resize_layers (student features are naturally multi-scale)
      4. No token→spatial reshape (inputs already [B, C, H, W])
      5. Exposes projected features for distillation via return_projected_feats
    """

    def __init__(
        self,
        dim_ins: Sequence[int] = (24, 32, 96, 320),
        *,
        output_dim: int = 2,
        activation: str = "exp",
        conf_activation: str = "expp1",
        features: int = 256,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
        pos_embed: bool = True,
        aux_pyramid_levels: int = 4,
        aux_out1_conv_num: int = 5,
        head_names: Tuple[str, str] = ("depth", "ray"),
    ):
        super().__init__()
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.aux_levels = aux_pyramid_levels
        self.aux_out1_conv_num = aux_out1_conv_num
        self.head_main, self.head_aux = head_names

        # Per-stage 1x1 projection (different input channels per stage)
        self.projects = nn.ModuleList(
            [nn.Conv2d(dim_ins[i], out_channels[i], kernel_size=1) for i in range(4)]
        )

        # Scratch: stage adapters + fusion
        self.scratch = _make_scratch(list(out_channels), features, expand=False)

        # Main fusion chain
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        # Main output head
        head_features_1 = features
        head_features_2 = 32
        self.scratch.output_conv1 = nn.Conv2d(
            head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1
        )
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
        )

        # Aux fusion chain (completely separate)
        self.scratch.refinenet1_aux = _make_fusion_block(features)
        self.scratch.refinenet2_aux = _make_fusion_block(features)
        self.scratch.refinenet3_aux = _make_fusion_block(features)
        self.scratch.refinenet4_aux = _make_fusion_block(features, has_residual=False)

        self.scratch.output_conv1_aux = nn.ModuleList(
            [self._make_aux_out1_block(head_features_1) for _ in range(self.aux_levels)]
        )

        self.scratch.output_conv2_aux = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                    Permute((0, 2, 3, 1)),
                    nn.LayerNorm(head_features_2),
                    Permute((0, 3, 1, 2)),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_features_2, 7, kernel_size=1, stride=1, padding=0),
                )
                for _ in range(self.aux_levels)
            ]
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        feats: List[torch.Tensor],
        H: int = None,
        W: int = None,
        return_projected_feats: bool = False,
    ) -> Union[Dict, Tuple[Dict, List[torch.Tensor]]]:
        """
        Args:
            feats: list of 4 spatial feature maps [B, C_i, H_i, W_i] from backbone.
            H, W: original image dimensions (for pos_embed aspect ratio). If None,
                   inferred from the largest feature map.
            return_projected_feats: if True also return the 4 projected features
                                    for distillation loss.
        """
        if H is None or W is None:
            H = feats[0].shape[2] * 4
            W = feats[0].shape[3] * 4

        projected = []
        for i, feat in enumerate(feats):
            x = self.projects[i](feat)
            if self.pos_embed:
                x = self._add_pos_embed(x, W, H)
            projected.append(x)

        fused_main, fused_aux_pyr = self._fuse(projected)

        # Upsample main to original resolution
        h_out, w_out = H, W
        fused_main = custom_interpolate(fused_main, (h_out, w_out), mode="bilinear", align_corners=True)
        if self.pos_embed:
            fused_main = self._add_pos_embed(fused_main, W, H)

        main_logits = self.scratch.output_conv2(fused_main)
        fmap = main_logits.permute(0, 2, 3, 1)
        main_pred = self._apply_activation(fmap[..., :-1], self.activation)
        main_conf = self._apply_activation(fmap[..., -1], self.conf_activation)

        # Aux head (last level only)
        last_aux = fused_aux_pyr[-1]
        if self.pos_embed:
            last_aux = self._add_pos_embed(last_aux, W, H)
        last_aux_logits = self.scratch.output_conv2_aux[-1](last_aux)
        fmap_last = last_aux_logits.permute(0, 2, 3, 1)
        aux_pred = self._apply_activation(fmap_last[..., :-1], "linear")
        aux_conf = self._apply_activation(fmap_last[..., -1], self.conf_activation)

        out = Dict({
            self.head_main: main_pred.squeeze(-1),
            f"{self.head_main}_conf": main_conf,
            self.head_aux: aux_pred,
            f"{self.head_aux}_conf": aux_conf,
        })
        if return_projected_feats:
            return out, projected
        return out

    # ------------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------------

    def _fuse(self, feats: List[torch.Tensor]):
        l1, l2, l3, l4 = feats
        l1_rn = self.scratch.layer1_rn(l1)
        l2_rn = self.scratch.layer2_rn(l2)
        l3_rn = self.scratch.layer3_rn(l3)
        l4_rn = self.scratch.layer4_rn(l4)

        out = self.scratch.refinenet4(l4_rn, size=l3_rn.shape[2:])
        aux_out = self.scratch.refinenet4_aux(l4_rn, size=l3_rn.shape[2:])
        aux_list: List[torch.Tensor] = []
        if self.aux_levels >= 4:
            aux_list.append(aux_out)

        out = self.scratch.refinenet3(out, l3_rn, size=l2_rn.shape[2:])
        aux_out = self.scratch.refinenet3_aux(aux_out, l3_rn, size=l2_rn.shape[2:])
        if self.aux_levels >= 3:
            aux_list.append(aux_out)

        out = self.scratch.refinenet2(out, l2_rn, size=l1_rn.shape[2:])
        aux_out = self.scratch.refinenet2_aux(aux_out, l2_rn, size=l1_rn.shape[2:])
        if self.aux_levels >= 2:
            aux_list.append(aux_out)

        out = self.scratch.refinenet1(out, l1_rn)
        aux_out = self.scratch.refinenet1_aux(aux_out, l1_rn)
        aux_list.append(aux_out)

        out = self.scratch.output_conv1(out)
        aux_list = [self.scratch.output_conv1_aux[i](a) for i, a in enumerate(aux_list)]

        return out, aux_list

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _add_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        pw, ph = x.shape[-1], x.shape[-2]
        pe = create_uv_grid(pw, ph, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pe = position_grid_to_embed(pe, x.shape[1]) * ratio
        pe = pe.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pe

    def _make_aux_out1_block(self, in_ch: int) -> nn.Sequential:
        if self.aux_out1_conv_num == 5:
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch // 2, 3, 1, 1),
                nn.Conv2d(in_ch // 2, in_ch, 3, 1, 1),
                nn.Conv2d(in_ch, in_ch // 2, 3, 1, 1),
                nn.Conv2d(in_ch // 2, in_ch, 3, 1, 1),
                nn.Conv2d(in_ch, in_ch // 2, 3, 1, 1),
            )
        if self.aux_out1_conv_num == 3:
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch // 2, 3, 1, 1),
                nn.Conv2d(in_ch // 2, in_ch, 3, 1, 1),
                nn.Conv2d(in_ch, in_ch // 2, 3, 1, 1),
            )
        if self.aux_out1_conv_num == 1:
            return nn.Sequential(nn.Conv2d(in_ch, in_ch // 2, 3, 1, 1))
        raise ValueError(f"aux_out1_conv_num {self.aux_out1_conv_num} not supported")

    @staticmethod
    def _apply_activation(x: torch.Tensor, activation: str) -> torch.Tensor:
        act = activation.lower()
        if act == "exp":
            return torch.exp(x)
        if act == "expp1":
            return torch.exp(x) + 1
        if act == "expm1":
            return torch.expm1(x)
        if act == "relu":
            return torch.relu(x)
        if act == "sigmoid":
            return torch.sigmoid(x)
        if act == "softplus":
            return F.softplus(x)
        if act == "softplusp1":
            return F.softplus(x) + 1
        if act == "tanh":
            return torch.tanh(x)
        return x
