"""Frozen DA3 teacher for full distillation (features + depth + ray + cam).

Extends the feature-extraction-only teacher with predict_all() for
single-forward-pass extraction of all teacher outputs. Does NOT call
the full DepthAnything3Net.forward() because _process_camera_estimation
deletes ray from the output dict.
"""

import sys
from typing import Dict, List

import torch
import torch.nn as nn


class TeacherDA3Distill(nn.Module):
    """Wrap frozen DA3 anyview branch for full prediction extraction.

    Usage::

        teacher = TeacherDA3Distill(model_dir=..., da3_src=..., device="cuda")
        out = teacher.predict_all(images)
        # out["features"]: list of 4 tensors
        # out["depth"]:    (B, H, W)
        # out["ray"]:      (B, Ht, Wt, 6)
        # out["cam_params"]: (B, 9)
    """

    def __init__(
        self,
        model_dir: str,
        da3_src: str,
        device: str = "cuda",
    ):
        super().__init__()

        if da3_src not in sys.path:
            sys.path.insert(0, da3_src)

        from depth_anything_3.api import DepthAnything3

        model = DepthAnything3.from_pretrained(model_dir)
        model = model.to(device)
        model.eval()

        self.anyview: nn.Module = model.model.da3
        self.anyview.eval()
        self.anyview.requires_grad_(False)

        self._hooked_feats: Dict[int, torch.Tensor] = {}
        dpt_head = self.anyview.head
        for i, layer in enumerate(dpt_head.resize_layers):
            layer.register_forward_hook(self._make_hook(i))

        self._device = device

    def _make_hook(self, idx: int):
        def hook_fn(module, inp, out):
            self._hooked_feats[idx] = out.detach()
        return hook_fn

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> List[torch.Tensor]:
        """Run teacher forward and return 4 DPT-projected feature maps.

        Args:
            images: [B, 3, H, W] on the correct device, ImageNet-normalised.
        Returns:
            list of 4 tensors, channels [256, 512, 1024, 1024].
        """
        self._hooked_feats.clear()

        x = images.unsqueeze(1)  # [B, 1, 3, H, W]

        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            self.anyview(x)

        feats = [self._hooked_feats[i].float() for i in range(4)]
        return feats

    @torch.no_grad()
    def predict_all(self, images: torch.Tensor) -> dict:
        """Run teacher forward: features + depth + ray + cam_params.

        Calls backbone → _process_depth_head → cam_dec individually
        (not the full forward) to preserve the ray output.

        Args:
            images: [B, 3, H, W] ImageNet-normalised, on device.
        Returns:
            dict with:
                features:   list of 4 float32 tensors
                depth:      (B, H, W) float32  — relative depth (exp activated)
                ray:        (B, Ht, Wt, 6) float32
                cam_params: (B, 9) float32 — [t(3), qvec(4), fov(2)]
        """
        self._hooked_feats.clear()
        x = images.unsqueeze(1)  # [B, 1, 3, H, W]
        H, W = images.shape[-2], images.shape[-1]

        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            feats, _ = self.anyview.backbone(
                x, cam_token=None, export_feat_layers=[],
                ref_view_strategy="saddle_balanced",
            )
            with torch.autocast(device_type="cuda", enabled=False):
                output = self.anyview._process_depth_head(feats, H, W)
                pose_enc = self.anyview.cam_dec(feats[-1][1])

        features = [self._hooked_feats[i].float() for i in range(4)]

        return {
            "features": features,
            "depth": output.depth.squeeze(1).float(),
            "ray": output.ray.squeeze(1).float(),
            "cam_params": pose_enc.squeeze(1).float(),
        }

    def to(self, *args, **kwargs):
        self.anyview = self.anyview.to(*args, **kwargs)
        return super().to(*args, **kwargs)
