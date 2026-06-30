"""Phase 2 task-level losses following the DA3 paper.

L = LD(depth) + LM(ray) + LP(point cloud) + LC(camera) + Lgrad(depth gradient)

All losses use L1 norm. Depth and ray losses include confidence weighting:
    loss = conf * |pred - gt| - lambda_c * log(conf)
"""

import torch
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def qvec_to_rotmat(qvec: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (xyzw) to rotation matrix. Differentiable.

    Handles non-unit quaternions via implicit normalization (matches DA3's
    quat_to_mat: two_s = 2 / ||q||^2).

    Args:
        qvec: [..., 4] quaternion in xyzw convention (scipy default).
    Returns:
        R: [..., 3, 3] rotation matrix.
    """
    i, j, k, r = torch.unbind(qvec, -1)
    two_s = 2.0 / (qvec * qvec).sum(-1)

    R = torch.stack([
        1 - two_s * (j*j + k*k),  two_s * (i*j - k*r),      two_s * (i*k + j*r),
        two_s * (i*j + k*r),      1 - two_s * (i*i + k*k),   two_s * (j*k - i*r),
        two_s * (i*k - j*r),      two_s * (j*k + i*r),        1 - two_s * (i*i + j*j),
    ], dim=-1).reshape(qvec.shape[:-1] + (3, 3))

    return R


def build_pixel_dirs(K: np.ndarray, H: int, W: int) -> torch.Tensor:
    """Precompute K^{-1} @ pixel_coords for all pixels.

    Uses DA3 normalized [0,2] space, consistent with dataloader bearing_grid.

    Args:
        K: (3,3) pixel-space intrinsics.
        H, W: image resolution.
    Returns:
        pixel_dirs: (H, W, 3) float32 tensor.
    """
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


def _downsample_mask(mask: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Downsample boolean mask conservatively (any-invalid → invalid).

    Args:
        mask: [B, H, W] bool.
    Returns:
        [B, target_h, target_w] bool.
    """
    inv = (~mask).float().unsqueeze(1)
    inv_down = F.max_pool2d(
        inv, kernel_size=(mask.shape[1] // target_h, mask.shape[2] // target_w),
    )
    return inv_down.squeeze(1) < 0.5


# ---------------------------------------------------------------------------
# Individual losses
# ---------------------------------------------------------------------------

def depth_loss(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    depth_conf: torch.Tensor,
    valid_mask: torch.Tensor,
    lambda_c: float = 0.2,
) -> torch.Tensor:
    """LD: confidence-weighted L1 depth loss.

    LD = mean_over_valid[ conf * |pred - gt| - lambda_c * log(conf) ]
    """
    depth_conf = depth_conf.clamp(min=1e-6, max=1e4)
    error = (pred_depth - gt_depth).abs()
    loss = depth_conf * error - lambda_c * torch.log(depth_conf)
    valid = valid_mask.bool()
    if valid.sum() < 1:
        return loss.mean() * 0
    return loss[valid].mean()


def ray_loss(
    pred_ray: torch.Tensor,
    gt_ray: torch.Tensor,
    ray_conf: torch.Tensor,
    valid_mask: torch.Tensor,
    lambda_c: float = 0.2,
) -> torch.Tensor:
    """LM: confidence-weighted L1 ray loss.

    Args:
        pred_ray: [B, H_ray, W_ray, 6]
        gt_ray: [B, H_ray, W_ray, 6]
        ray_conf: [B, H_ray, W_ray]
        valid_mask: [B, H, W] full-resolution mask (will be downsampled).
    """
    ray_conf = ray_conf.clamp(min=1e-6, max=1e4)
    ray_h, ray_w = pred_ray.shape[1], pred_ray.shape[2]
    mask_down = _downsample_mask(valid_mask, ray_h, ray_w)

    error = (pred_ray - gt_ray).abs().mean(dim=-1)
    loss = ray_conf * error - lambda_c * torch.log(ray_conf)
    valid = mask_down.bool()
    if valid.sum() < 1:
        return loss.mean() * 0
    return loss[valid].mean()


def point_cloud_loss(
    pred_depth: torch.Tensor,
    pred_qvec: torch.Tensor,
    pred_t: torch.Tensor,
    gt_depth: torch.Tensor,
    gt_cam_params: torch.Tensor,
    pixel_dirs: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """LP: point cloud L1 loss from depth + camera params + K.

    pred_point = pred_depth * (R_pred @ pixel_dirs) + T_pred
    gt_point   = gt_depth   * (R_gt   @ pixel_dirs) + T_gt

    Args:
        pred_depth: [B, H, W]
        pred_qvec: [B, 4] predicted quaternion (xyzw).
        pred_t: [B, 3] predicted translation.
        gt_depth: [B, H, W]
        gt_cam_params: [B, 9] = [t(3), qvec(4), fov(2)].
        pixel_dirs: [H, W, 3] precomputed K^{-1} @ meshgrid (constant buffer).
        valid_mask: [B, H, W]
    """
    B, H, W = pred_depth.shape

    gt_t = gt_cam_params[:, :3]
    gt_qvec = gt_cam_params[:, 3:7]

    R_pred = qvec_to_rotmat(pred_qvec)
    R_gt = qvec_to_rotmat(gt_qvec)

    # pixel_dirs: (H, W, 3) → (H*W, 3) → transpose → (3, N)
    dirs_flat = pixel_dirs.reshape(-1, 3).T  # (3, H*W)

    # R @ dirs_flat → (B, 3, N) → (B, H, W, 3)
    rotated_pred = torch.matmul(R_pred, dirs_flat).permute(0, 2, 1).reshape(B, H, W, 3)
    rotated_gt = torch.matmul(R_gt, dirs_flat).permute(0, 2, 1).reshape(B, H, W, 3)

    pred_point = pred_depth.unsqueeze(-1) * rotated_pred + pred_t[:, None, None, :]
    gt_point = gt_depth.unsqueeze(-1) * rotated_gt + gt_t[:, None, None, :]

    error = (pred_point - gt_point).abs().mean(dim=-1)
    valid = valid_mask.bool()
    if valid.sum() < 1:
        return error.mean() * 0
    return error[valid].mean()


def camera_loss(
    pred_cam_enc: torch.Tensor,
    gt_cam_params: torch.Tensor,
    w_t: float = 1.0,
    w_q: float = 1.0,
    w_fov: float = 0.5,
) -> torch.Tensor:
    """LC: per-component L1 camera parameter loss.

    Args:
        pred_cam_enc: [B, 9] = [t(3), qvec(4), fov(2)].
        gt_cam_params: [B, 9] = [t(3), qvec(4), fov(2)].
    """
    loss_t = (pred_cam_enc[:, :3] - gt_cam_params[:, :3]).abs().mean()
    loss_q = (pred_cam_enc[:, 3:7] - gt_cam_params[:, 3:7]).abs().mean()
    loss_fov = (pred_cam_enc[:, 7:] - gt_cam_params[:, 7:]).abs().mean()
    return w_t * loss_t + w_q * loss_q + w_fov * loss_fov


def gradient_loss(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Lgrad: finite-difference depth gradient L1 loss.

    Lgrad = ||nabla_x(pred) - nabla_x(gt)||_1 + ||nabla_y(pred) - nabla_y(gt)||_1
    """
    diff = pred_depth - gt_depth

    grad_x = (diff[:, :, 1:] - diff[:, :, :-1]).abs()
    mask_x = valid_mask[:, :, 1:] & valid_mask[:, :, :-1]

    grad_y = (diff[:, 1:, :] - diff[:, :-1, :]).abs()
    mask_y = valid_mask[:, 1:, :] & valid_mask[:, :-1, :]

    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)

    n_valid = mask_x.sum() + mask_y.sum()
    if n_valid < 1:
        return grad_x.mean() * 0

    loss = grad_x[mask_x].sum() + grad_y[mask_y].sum()
    return loss / n_valid


# ---------------------------------------------------------------------------
# Combined Phase 2 loss
# ---------------------------------------------------------------------------

def phase2_loss(student_out, batch, pixel_dirs, cfg):
    """Compute all Phase 2 task losses.

    Args:
        student_out: model output Dict with depth, depth_conf, ray, ray_conf, cam.
        batch: dict with gt depth, ray, cam_params, valid_mask (already on device).
        pixel_dirs: (H, W, 3) precomputed constant buffer (on device).
        cfg: loss config (OmegaConf node).

    Returns:
        total: scalar loss.
        loss_dict: dict of individual loss values for logging.
    """
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
    valid_mask = batch["valid_mask"]

    lambda_c = getattr(cfg, "lambda_c", 0.2)

    ld = depth_loss(pred_depth, gt_depth, pred_depth_conf, valid_mask, lambda_c)
    lm = ray_loss(pred_ray, gt_ray, pred_ray_conf, valid_mask, lambda_c)
    lp = point_cloud_loss(
        pred_depth, pred_qvec, pred_t,
        gt_depth, gt_cam, pixel_dirs, valid_mask,
    )
    lc = camera_loss(
        pred_cam_enc, gt_cam,
        w_t=getattr(cfg, "cam_trans_weight", 1.0),
        w_q=getattr(cfg, "cam_rot_weight", 1.0),
        w_fov=getattr(cfg, "cam_fov_weight", 0.5),
    )
    lg = gradient_loss(pred_depth, gt_depth, valid_mask)

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
