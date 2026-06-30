import torch
import torch.nn as nn
import torchvision.models as models


class MobileNetV2Backbone(nn.Module):
    """Extract 4-scale feature maps from torchvision MobileNetV2.

    Tap-off points (for 644x476 input):
        L1 (S2): features[0:4]  -> [B, 24,  119, 161]  stride=4
        L2 (S3): features[4:7]  -> [B, 32,   60,  81]  stride=8
        L3 (S5): features[7:14] -> [B, 96,   30,  41]  stride=16
        L4 (S7): features[14:18]-> [B, 320,  15,  21]  stride=32
    """

    feature_channels = [24, 32, 96, 320]

    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        )
        feats = backbone.features
        self.stage1 = nn.Sequential(*feats[0:4])
        self.stage2 = nn.Sequential(*feats[4:7])
        self.stage3 = nn.Sequential(*feats[7:14])
        self.stage4 = nn.Sequential(*feats[14:18])

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, 3, H, W] input images (ImageNet-normalised).
        Returns:
            list of 4 tensors [L1, L2, L3, L4].
        """
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return [f1, f2, f3, f4]
