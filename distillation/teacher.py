"""Frozen DA3 teacher wrapper with hook-based DPT feature extraction.

DA3 source code is NOT modified. We add its src dir to sys.path at runtime
and load the model via from_pretrained. Forward hooks on the DualDPT's
resize_layers capture the 4-level features for distillation.
"""

import sys
from typing import Dict, List

import torch
import torch.nn as nn


class TeacherDA3(nn.Module):
    """Wrap the frozen DA3 anyview branch for feature extraction.

    Usage::

        teacher = TeacherDA3(
            model_dir="/path/to/DA3NESTED-GIANT-LARGE-1.1",
            da3_src="/path/to/Depth-Anything-3-main/src",
        )
        teacher_feats = teacher.extract_features(images)  # list of 4 tensors
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

        # Keep only the anyview branch (skip metric branch to save time)
        self.anyview: nn.Module = model.model.da3
        self.anyview.eval()
        self.anyview.requires_grad_(False)

        # Register hooks on DualDPT resize_layers to capture projected features
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

        # DA3 expects [B, N, 3, H, W] where N = number of views
        x = images.unsqueeze(1)  # [B, 1, 3, H, W]

        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            self.anyview(x)

        feats = [self._hooked_feats[i].float() for i in range(4)]
        return feats

    def to(self, *args, **kwargs):
        # Override to also move the anyview branch
        self.anyview = self.anyview.to(*args, **kwargs)
        return super().to(*args, **kwargs)
