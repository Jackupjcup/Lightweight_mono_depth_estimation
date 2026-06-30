"""Phase 2 task-level losses — v1: no valid_mask, all pixels participate.

Based on objective_losses.py. All per-pixel losses compute over every pixel
instead of masking by valid_mask.
"""

import torch
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def qvec_to_rotmat(qvec: torch.Tensor) -> torch.Tensor:
    i, j, k, r = torch.unbind(qvec, -1)
    two_s = 2.0 / (qvec * qvec).sum(-1)

    R = torch.stack([
        1 - two_s * (j*j + k*k),  two_s * (i*j - k*r),      two_s * (i*k + j*r),
        two_s * (i*j + k*r),      1 - two_s * (i*i + k*k),   two_s * (j*k - i*r),
        two_s * (i*k - j*r),      two_s * (j*k + i*r),        1 - two_s * (i*i + j*j),
    ], dim=-1).reshape(qvec.shape[:-1] + (3, 3))

    return R


def build_pixel_dirs(K: np.ndarray, H: int, W: int) -> torch.Tensor:
    fx_norm = 2.0 * K[0, 0] / W
    fy_norm = 2.0 * K[1, 1] / H
    cx_norm = 2.0 * (K[0, 2] + 0.5) / W
    cy_norm = 2.0 * (K[1, 2] + 0.5) / H

    u = np.linspace(1.0 / W, 2.0 - 1.0 / W, W, dtype=np.float64)
    v = np.linspace(1.0 / H, 2.0 - 1.0 / H, H, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)

    dirs = np.stack([
        (uu - cx_norm) / fx_norm,
        (vv - cy_norm) / fy_norm,
        np.ones_like(uu),
    ], axis=-1).astype(np.float32)

    return torch.from_numpy(dirs)


# ---------------------------------------------------------------------------
# Individual losses
# ---------------------------------------------------------------------------

def depth_loss(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    depth_conf: torch.Tensor,
    lambda_c: float = 0.2,
) -> torch.Tensor:
    depth_conf = depth_conf.clamp(min=1e-6, max=1e4)
    error = (pred_depth - gt_depth).abs()
    loss = depth_conf * error - lambda_c * torch.log(depth_conf)
    return loss.mean()


def ray_loss(
    pred_ray: torch.Tensor,
    gt_ray: torch.Tensor,
    ray_conf: torch.Tensor,
    lambda_c: float = 0.2,
) -> torch.Tensor:
    ray_conf = ray_conf.clamp(min=1e-6, max=1e4)
    error = (pred_ray - gt_ray).abs().mean(dim=-1)
    loss = ray_conf * error - lambda_c * torch.log(ray_conf)
    return loss.mean()


def point_cloud_loss(
    pred_depth: torch.Tensor,
    pred_qvec: torch.Tensor,
    pred_t: torch.Tensor,
    gt_depth: torch.Tensor,
    gt_cam_params: torch.Tensor,
    pixel_dirs: torch.Tensor,
) -> torch.Tensor:
    B, H, W = pred_depth.shape

    gt_t = gt_cam_params[:, :3]
    gt_qvec = gt_cam_params[:, 3:7]

    R_pred = qvec_to_rotmat(pred_qvec)
    R_gt = qvec_to_rotmat(gt_qvec)

    dirs_flat = pixel_dirs.reshape(-1, 3).T

    rotated_pred = torch.matmul(R_pred, dirs_flat).permute(0, 2, 1).reshape(B, H, W, 3)
    rotated_gt = torch.matmul(R_gt, dirs_flat).permute(0, 2, 1).reshape(B, H, W, 3)

    pred_point = pred_depth.unsqueeze(-1) * rotated_pred + pred_t[:, None, None, :]
    gt_point = gt_depth.unsqueeze(-1) * rotated_gt + gt_t[:, None, None, :]

    error = (pred_point - gt_point).abs().mean(dim=-1)
    return error.mean()


def camera_loss(
    pred_cam_enc: torch.Tensor,
    gt_cam_params: torch.Tensor,
    w_t: float = 1.0,
    w_q: float = 1.0,
    w_fov: float = 0.5,
) -> torch.Tensor:
    loss_t = (pred_cam_enc[:, :3] - gt_cam_params[:, :3]).abs().mean()
    loss_q = (pred_cam_enc[:, 3:7] - gt_cam_params[:, 3:7]).abs().mean()
    loss_fov = (pred_cam_enc[:, 7:] - gt_cam_params[:, 7:]).abs().mean()
    return w_t * loss_t + w_q * loss_q + w_fov * loss_fov


def gradient_loss(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
) -> torch.Tensor:
    diff = pred_depth - gt_depth

    grad_x = (diff[:, :, 1:] - diff[:, :, :-1]).abs()
    grad_y = (diff[:, 1:, :] - diff[:, :-1, :]).abs()

    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)

    return (grad_x.mean() + grad_y.mean()) * 0.5


# ---------------------------------------------------------------------------
# Combined Phase 2 loss
# ---------------------------------------------------------------------------

def phase2_loss(student_out, batch, pixel_dirs, cfg):
    pred_depth = student_out.depth.float()
    pred_depth_conf = student_out.depth_conf.float()
    pred_ray = student_out.ray.float()
    pred_ray_conf = student_out.ray_conf.float()
    pred_t = student_out.cam["t"].float()
    pred_qvec = student_out.cam["qvec"].float()
    pred_cam_enc = student_out.cam["pose_enc"].float()

    gt_depth = batch["depth"]
    gt_ray = batch["ray"]
    gt_cam = batch["cam_params"]

    lambda_c = getattr(cfg, "lambda_c", 0.2)

    ld = depth_loss(pred_depth, gt_depth, pred_depth_conf, lambda_c)
    lm = ray_loss(pred_ray, gt_ray, pred_ray_conf, lambda_c)
    lp = point_cloud_loss(
        pred_depth, pred_qvec, pred_t,
        gt_depth, gt_cam, pixel_dirs,
    )
    lc = camera_loss(
        pred_cam_enc, gt_cam,
        w_t=getattr(cfg, "cam_trans_weight", 1.0),
        w_q=getattr(cfg, "cam_rot_weight", 1.0),
        w_fov=getattr(cfg, "cam_fov_weight", 0.5),
    )
    lg = gradient_loss(pred_depth, gt_depth)

    w_depth = getattr(cfg, "depth_weight", 1.0)
    w_ray = getattr(cfg, "ray_weight", 1.0)
    w_point = getattr(cfg, "point_weight", 1.0)
    w_cam = getattr(cfg, "cam_weight", 1.0)
    w_grad = getattr(cfg, "grad_weight", 1.0)

    total = (
        w_depth * ld
        + w_ray * lm
        + w_point * lp
        + w_cam * lc
        + w_grad * lg
    )

    loss_dict = {
        "depth": (w_depth * ld).item(),
        "ray": (w_ray * lm).item(),
        "point": (w_point * lp).item(),
        "cam": (w_cam * lc).item(),
        "grad": (w_grad * lg).item(),
    }

    return total, loss_dict
