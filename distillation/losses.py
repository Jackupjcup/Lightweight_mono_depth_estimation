from typing import List

import torch
import torch.nn.functional as F


def feature_distillation_loss(
    student_feats: List[torch.Tensor],
    teacher_feats: List[torch.Tensor],
    weights: List[float] = None,
) -> torch.Tensor:
    """Multi-scale feature MSE distillation loss.

    Interpolates student features to match teacher spatial dimensions,
    then computes per-level MSE. Teacher features must be detached.

    Args:
        student_feats: 4 tensors from student DPT projects, [B, C_i, Hs_i, Ws_i].
        teacher_feats: 4 tensors from teacher DPT projects+resize, [B, C_i, Ht_i, Wt_i].
        weights: per-level loss weights (default: [1, 1, 1, 1]).
    Returns:
        Scalar loss.
    """
    if weights is None:
        weights = [1.0] * len(student_feats)

    total = torch.tensor(0.0, device=student_feats[0].device)
    per_level = []
    for w, s_feat, t_feat in zip(weights, student_feats, teacher_feats):
        s_aligned = F.interpolate(
            s_feat, size=t_feat.shape[2:], mode="bilinear", align_corners=False
        )
        level_loss = F.mse_loss(s_aligned, t_feat.detach())
        per_level.append(level_loss.item())
        total = total + w * level_loss

    return total, per_level
