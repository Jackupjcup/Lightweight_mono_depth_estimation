import sys
import torch
import torch.nn as nn


class StudentCameraHead(nn.Module):
    """Predict camera pose from MobileNetV2 L4 feature via global average pooling.

    Since MobileNetV2 has no CLS token, we pool the final feature map and project
    to the dimension expected by DA3's CameraDec.
    """

    def __init__(self, in_channels: int = 320, cam_dim: int = 3072):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(in_channels, cam_dim)

        self.cam_dec = self._build_cam_dec(cam_dim)

    @staticmethod
    def _build_cam_dec(cam_dim: int) -> nn.Module:
        """Lightweight camera decoder (same architecture as DA3 CameraDec)."""
        return nn.ModuleDict({
            "backbone": nn.Sequential(
                nn.Linear(cam_dim, cam_dim),
                nn.ReLU(),
                nn.Linear(cam_dim, cam_dim),
                nn.ReLU(),
            ),
            "fc_t": nn.Linear(cam_dim, 3),
            "fc_qvec": nn.Linear(cam_dim, 4),
            "fc_fov": nn.Sequential(nn.Linear(cam_dim, 2), nn.ReLU()),
        })

    def forward(self, feat: torch.Tensor) -> dict:
        """
        Args:
            feat: [B, C, H, W] — L4 feature from backbone.
        Returns:
            dict with 't' [B,3], 'qvec' [B,4], 'fov' [B,2], 'pose_enc' [B,9].
        """
        B = feat.shape[0]
        x = self.pool(feat).flatten(1)      # [B, 320]
        x = self.proj(x)                     # [B, cam_dim]
        x = self.cam_dec["backbone"](x)      # [B, cam_dim]

        t = self.cam_dec["fc_t"](x.float())          # [B, 3]
        qvec = self.cam_dec["fc_qvec"](x.float())    # [B, 4]
        fov = self.cam_dec["fc_fov"](x.float())       # [B, 2]
        pose_enc = torch.cat([t, qvec, fov], dim=-1)  # [B, 9]

        return {"t": t, "qvec": qvec, "fov": fov, "pose_enc": pose_enc}
