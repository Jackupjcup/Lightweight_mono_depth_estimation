"""Distillation losses — all targets from DA3 teacher, no GT supervision.

No confidence weighting (teacher output is oracle).
Log-space depth loss prevents exp-activation gradient vanishing.
"""

import torch
import torch.nn.functional as F


def log_depth_distill_loss(
    pred_depth: torch.Tensor,
    teacher_depth: torch.Tensor,
) -> torch.Tensor:
    """L1 in log-space: |log(pred) - log(teacher)|.

    Since pred = exp(logit), log(pred) = logit, so the gradient
    w.r.t. logit is sign(logit - log(teacher)) — never vanishes
    regardless of how small the depth is.

    Args:
        pred_depth:    (B, H, W) student depth (exp activated, > 0).
        teacher_depth: (B, H, W) teacher depth (exp activated, > 0).
    """
    eps = 1e-8
    log_pred = torch.log(pred_depth + eps)
    log_teacher = torch.log(teacher_depth + eps)
    if log_pred.shape != log_teacher.shape:
        log_teacher = F.interpolate(
            log_teacher.unsqueeze(1),
            size=log_pred.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
    return (log_pred - log_teacher).abs().mean()


def ray_distill_loss(
    pred_ray: torch.Tensor,
    teacher_ray: torch.Tensor,
) -> torch.Tensor:
    """Per-pixel L1 ray distillation. Interpolates teacher to student size.

    Args:
        pred_ray:    (B, Hs, Ws, 6) student ray field.
        teacher_ray: (B, Ht, Wt, 6) teacher ray field.
    """
    if pred_ray.shape[1:3] != teacher_ray.shape[1:3]:
        t = teacher_ray.permute(0, 3, 1, 2)  # (B, 6, Ht, Wt)
        t = F.interpolate(
            t,
            size=(pred_ray.shape[1], pred_ray.shape[2]),
            mode="bilinear",
            align_corners=False,
        )
        teacher_ray = t.permute(0, 2, 3, 1)  # (B, Hs, Ws, 6)
    return (pred_ray - teacher_ray).abs().mean()


def cam_distill_loss(
    pred_cam: torch.Tensor,
    teacher_cam: torch.Tensor,
    w_t: float = 1.0,
    w_q: float = 1.0,
    w_fov: float = 0.5,
) -> torch.Tensor:
    """Per-component L1 camera parameter loss.

    Args:
        pred_cam:    (B, 9) = [t(3), qvec(4), fov(2)].
        teacher_cam: (B, 9) = [t(3), qvec(4), fov(2)].
    """
    loss_t = (pred_cam[:, :3] - teacher_cam[:, :3]).abs().mean()
    loss_q = (pred_cam[:, 3:7] - teacher_cam[:, 3:7]).abs().mean()
    loss_fov = (pred_cam[:, 7:] - teacher_cam[:, 7:]).abs().mean()
    return w_t * loss_t + w_q * loss_q + w_fov * loss_fov


def grad_distill_loss(
    pred_depth: torch.Tensor,
    teacher_depth: torch.Tensor,
) -> torch.Tensor:
    """Finite-difference depth gradient L1 loss (teacher as target).

    Args:
        pred_depth:    (B, H, W) student depth.
        teacher_depth: (B, H, W) teacher depth.
    """
    if pred_depth.shape != teacher_depth.shape:
        teacher_depth = F.interpolate(
            teacher_depth.unsqueeze(1),
            size=pred_depth.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
    diff = pred_depth - teacher_depth
    grad_x = (diff[:, :, 1:] - diff[:, :, :-1]).abs().clamp(max=100)
    grad_y = (diff[:, 1:, :] - diff[:, :-1, :]).abs().clamp(max=100)
    return (grad_x.mean() + grad_y.mean()) / 2
