from typing import List, Tuple, Union

import torch
import torch.nn as nn
from addict import Dict

from .backbone import MobileNetV2Backbone
from .student_dpt import StudentDualDPT
from .cam_head import StudentCameraHead


class FastDepthModel(nn.Module):
    """Student model: MobileNetV2 backbone + StudentDualDPT + CameraHead.

    Outputs depth, depth_conf, ray, ray_conf, and camera parameters.
    Optionally returns projected DPT features for distillation.
    """

    def __init__(
        self,
        backbone_pretrained: bool = True,
        dpt_features: int = 256,
        dpt_out_channels: tuple = (256, 512, 1024, 1024),
        cam_dim: int = 3072,
        depth_activation: str = "softplus",
        conf_activation: str = "softplusp1",
    ):
        super().__init__()
        self.backbone = MobileNetV2Backbone(pretrained=backbone_pretrained)
        self.head = StudentDualDPT(
            dim_ins=self.backbone.feature_channels,
            features=dpt_features,
            out_channels=dpt_out_channels,
            activation=depth_activation,
            conf_activation=conf_activation,
        )
        self.cam_head = StudentCameraHead(
            in_channels=self.backbone.feature_channels[-1],
            cam_dim=cam_dim,
        )

    def forward(
        self,
        images: torch.Tensor,
        return_distill_feats: bool = False,
    ) -> Union[Dict, Tuple[Dict, List[torch.Tensor]]]:
        """
        Args:
            images: [B, 3, H, W] ImageNet-normalised.
            return_distill_feats: if True, also return 4 projected feature maps.
        Returns:
            output: Dict with depth, depth_conf, ray, ray_conf.
            (optional) proj_feats: list of 4 tensors for distillation.
        """
        feats = self.backbone(images)
        H, W = images.shape[2], images.shape[3]

        out, proj_feats = self.head(
            feats, H=H, W=W, return_projected_feats=True
        )

        cam_out = self.cam_head(feats[-1])
        out.cam = cam_out

        if return_distill_feats:
            return out, proj_feats
        return out
