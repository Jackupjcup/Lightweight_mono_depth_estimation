"""Preprocessing utilities for TartanGround → DA3 training pipeline.

Handles:
- 640×640 → 644×476 crop+reflect-pad (RGB & depth)
- Intrinsics adjustment after spatial transform
- Planar depth → point cloud → per-frame scale factor (DA3 paper normalization)
"""

import cv2
import numpy as np

# ---- Spatial transform constants ----
SRC_H, SRC_W = 640, 640
TGT_H, TGT_W = 476, 644

CROP_TOP = (SRC_H - TGT_H) // 2   # 82
CROP_BOT = SRC_H - CROP_TOP        # 558
PAD_L = (TGT_W - SRC_W) // 2      # 2
PAD_R = TGT_W - SRC_W - PAD_L     # 2

# ---- Original TartanGround intrinsics (all scenes, 640×640, FOV=90°) ----
K_ORIG = np.array([
    [320.0,   0.0, 319.5],
    [  0.0, 320.0, 319.5],
    [  0.0,   0.0,   1.0],
], dtype=np.float64)

# ---- Adjusted intrinsics after crop+pad to 644×476 ----
# Vertical center crop removes 82px top → cy shifts by -82
# Horizontal reflect pad adds 2px left → cx shifts by +2
# fx, fy unchanged (no resize)
K_CROPPED = np.array([
    [320.0,   0.0, 321.5],
    [  0.0, 320.0, 237.5],
    [  0.0,   0.0,   1.0],
], dtype=np.float64)


def decode_depth(path: str) -> np.ndarray:
    """Decode TartanGround RGBA-encoded float32 depth (planar, meters).

    Must use cv2 (BGRA byte order) — PIL's RGBA order puts the zero B channel
    in byte 2, destroying the exponent LSB and creating 4× quantization bands.
    """
    depth_rgba = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # (640, 640, 4) uint8 BGRA
    return depth_rgba.view("<f4").squeeze(axis=-1)  # (640, 640) float32


def crop_and_pad(img: np.ndarray) -> np.ndarray:
    """Center-crop vertically 640→476, reflect-pad horizontally 640→644."""
    cropped = img[CROP_TOP:CROP_BOT]
    if img.ndim == 3:
        return np.pad(cropped, ((0, 0), (PAD_L, PAD_R), (0, 0)), mode="reflect")
    return np.pad(cropped, ((0, 0), (PAD_L, PAD_R)), mode="reflect")


def depth_to_pointcloud(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Convert planar depth map to 3D point cloud in camera coordinates.

    Args:
        depth: (H, W) planar depth in meters (along optical axis).
        K: (3, 3) camera intrinsic matrix.

    Returns:
        points: (H, W, 3) point cloud [X, Y, Z] in camera frame.
    """
    h, w = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = np.arange(w, dtype=np.float64) + 0.5  # pixel centers
    v = np.arange(h, dtype=np.float64) + 0.5
    u, v = np.meshgrid(u, v)

    Z = depth.astype(np.float64)
    X = (u - cx) / fx * Z
    Y = (v - cy) / fy * Z

    return np.stack([X, Y, Z], axis=-1)  # (H, W, 3)


def compute_scale_factor(depth: np.ndarray, K: np.ndarray,
                         min_depth: float = 0.5,
                         max_depth: float = 1000.0) -> float:
    """Compute DA3-style per-frame scale factor.

    Scale = mean ℓ2 norm of valid reprojected points P.
    Valid = depth within [min_depth, max_depth] to exclude sky/clipping artifacts.

    Args:
        depth: (H, W) planar depth in meters.
        K: (3, 3) intrinsic matrix.
        min_depth: minimum valid depth (TartanGround near clip = 0.5m).
        max_depth: maximum valid depth (exclude far-clip / sky).

    Returns:
        scale: mean ‖P‖₂ of valid points.
    """
    points = depth_to_pointcloud(depth, K)  # (H, W, 3)
    norms = np.linalg.norm(points, axis=-1)  # (H, W)

    valid = (depth >= min_depth) & (depth <= max_depth)
    valid_norms = norms[valid]

    if valid_norms.size == 0:
        return 1.0

    return float(valid_norms.mean())


def normalize_depth(depth: np.ndarray, K: np.ndarray, **kwargs) -> tuple:
    """Full pipeline: depth → point cloud → scale → normalized depth.

    Returns:
        (normalized_depth, scale_factor)
    """
    scale = compute_scale_factor(depth, K, **kwargs)
    return depth / scale, scale
