"""Visualize crop+reflect-pad, DA3-style normalization, and DA3 prediction on 20 TartanGround samples.

Layout per sample (4 columns × 2 rows):
  Row 0: Original RGB | Cropped RGB | Scale Factor Info | DA3 Input (cropped RGB)
  Row 1: Original Depth | Cropped Depth | Normalized Depth | DA3 Predicted Depth
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

from preprocess import (
    SRC_H, SRC_W, TGT_H, TGT_W,
    CROP_TOP, CROP_BOT, PAD_L,
    K_CROPPED,
    decode_depth, crop_and_pad, normalize_depth,
)

DATA_ROOT = "/data-tos-daily/TartanGround"
OUT_DIR = "/data-vepfs/lrd_projs/mono_depth_estimation/fast_mono_depth/dataset_process/vis_crop_clamp"
DA3_MODEL_DIR = "/data-vepfs/lrd_projs/mono_depth_estimation/da3/models/DA3NESTED-GIANT-LARGE-1.1"
DEPTH_CLAMP = 100.0

DEPTH_CMAP = "turbo_r"  # red=near, blue=far


def adaptive_clamp(depth: np.ndarray, min_log_gap: float = 1.0) -> tuple:
    """Detect sky/far-plane boundary via the largest multiplicative gap in depth.

    Works in log-space so a jump from 50m→9000m is detected even if
    the sky cluster itself is dense.  min_log_gap is the minimum
    ln-ratio to count as a sky boundary (default 1.0 ≈ 2.7× jump).

    Returns (clamped_depth, clamp_value).
    """
    valid = depth[depth > 0.5].ravel()
    if valid.size == 0:
        return depth, float(depth.max())

    sorted_d = np.sort(valid)
    log_d = np.log(sorted_d)
    log_diffs = np.diff(log_d)

    gap_idx = np.argmax(log_diffs)
    if log_diffs[gap_idx] >= min_log_gap:
        clamp_val = float(sorted_d[gap_idx])
    else:
        clamp_val = float(depth.max())

    return np.clip(depth, 0, clamp_val), clamp_val


def load_da3_model():
    """Load DA3 model via the official Python API."""
    sys.path.insert(0, "/data-vepfs/lrd_projs/mono_depth_estimation/da3/src")
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(DA3_MODEL_DIR).to("cuda")
    return model


def da3_predict(model, rgb_crop):
    """Run DA3 on a cropped RGB (H, W, 3) uint8 numpy array."""
    prediction = model.inference(image=[rgb_crop])
    return prediction.depth[0]


def collect_samples(n=20):
    samples = []
    for scene in sorted(os.listdir(DATA_ROOT)):
        diff_dir = os.path.join(DATA_ROOT, scene, "Data_diff")
        if not os.path.isdir(diff_dir):
            continue
        trajs = sorted([d for d in os.listdir(diff_dir) if d.startswith("P")])
        if not trajs:
            continue
        traj = trajs[0]
        rgb_dir = os.path.join(diff_dir, traj, "image_lcam_front")
        dep_dir = os.path.join(diff_dir, traj, "depth_lcam_front")
        if not os.path.isdir(rgb_dir) or not os.path.isdir(dep_dir):
            continue
        rgb_path = os.path.join(rgb_dir, "000000_lcam_front.png")
        dep_path = os.path.join(dep_dir, "000000_lcam_front_depth.png")
        if os.path.isfile(rgb_path) and os.path.isfile(dep_path):
            samples.append((rgb_path, dep_path, scene))
        if len(samples) >= n:
            break
    return samples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adap-depth", action="store_true",
                        help="use adaptive per-frame depth clamping (log-space gap detection)")
    parser.add_argument("--depth-clamp", type=float, default=DEPTH_CLAMP,
                        help=f"fixed depth clamp in meters (default: {DEPTH_CLAMP}), ignored when --adap-depth is set")
    da3_group = parser.add_mutually_exclusive_group()
    da3_group.add_argument("--da3-anyview", action="store_true",
                           help="use only the anyview branch (ViT-G + DualDPT, relative depth)")
    da3_group.add_argument("--da3-metric", action="store_true",
                           help="use only the metric branch (ViT-L + DPT, metric depth)")
    da3_group.add_argument("--da3-nested", action="store_true",
                           help="use full nested model with metric alignment (default)")
    return parser.parse_args()


def main():
    args = parse_args()
    da3_mode = "anyview" if args.da3_anyview else "metric" if args.da3_metric else "nested"
    da3_suffix = f"_{da3_mode}" if da3_mode != "nested" else ""
    adap_suffix = "_adap" if args.adap_depth else ""
    out_dir = OUT_DIR + da3_suffix + adap_suffix
    os.makedirs(out_dir, exist_ok=True)
    samples = collect_samples(20)
    print(f"Found {len(samples)} samples")
    if args.adap_depth:
        print("Mode: adaptive per-frame depth clamping")
    else:
        print(f"Mode: fixed depth clamp = {args.depth_clamp}m")
    print(f"Output: {out_dir}")

    print(f"Loading DA3 model ({da3_mode})...")
    da3_model = load_da3_model()
    if args.da3_anyview:
        da3_model.model = da3_model.model.da3
    elif args.da3_metric:
        da3_model.model = da3_model.model.da3_metric
    print("DA3 model loaded.")

    for i, (rgb_path, dep_path, scene) in enumerate(samples):
        rgb_orig = np.array(Image.open(rgb_path))
        dep_raw = decode_depth(dep_path)
        if args.adap_depth:
            dep_orig, clamp_val = adaptive_clamp(dep_raw)
        else:
            clamp_val = args.depth_clamp
            dep_orig = np.clip(dep_raw, 0, clamp_val)

        rgb_crop = crop_and_pad(rgb_orig)
        dep_crop = crop_and_pad(dep_orig)

        dep_norm, scale = normalize_depth(dep_crop, K_CROPPED)

        # DA3 prediction on cropped image
        da3_depth = da3_predict(da3_model, rgb_crop)

        # Absolute depth colorbar range (clamp outliers)
        vmin_abs = 0
        vmax_abs = np.percentile(dep_orig, 98)

        # Normalized depth colorbar range
        vmax_norm = float(dep_norm.max())

        # DA3 depth colorbar range
        vmax_da3 = np.percentile(da3_depth, 98)

        fig, axes = plt.subplots(2, 4, figsize=(28, 12),
                                 gridspec_kw={"hspace": 0.25, "wspace": 0.18})
        fig.suptitle(
            f"[{i}] {scene}  —  640×640 → 644×476    scale={scale:.2f}m    clamp={clamp_val:.1f}m    DA3={da3_mode}",
            fontsize=14,
        )

        # ---- Row 0: RGB ----
        axes[0, 0].imshow(rgb_orig)
        rect = patches.Rectangle(
            (-0.5, CROP_TOP - 0.5), SRC_W, TGT_H,
            linewidth=2, edgecolor="lime", facecolor="none", linestyle="--",
        )
        axes[0, 0].add_patch(rect)
        axes[0, 0].axhline(CROP_TOP, color="red", lw=1, ls=":")
        axes[0, 0].axhline(CROP_BOT, color="red", lw=1, ls=":")
        axes[0, 0].set_title(f"Original RGB  {SRC_W}×{SRC_H}")

        axes[0, 1].imshow(rgb_crop)
        axes[0, 1].axvline(PAD_L - 0.5, color="cyan", lw=1, ls=":")
        axes[0, 1].axvline(PAD_L + SRC_W - 0.5, color="cyan", lw=1, ls=":")
        axes[0, 1].set_title(f"Cropped+Padded RGB  {TGT_W}×{TGT_H}")

        axes[0, 2].imshow(rgb_crop, alpha=0.3)
        axes[0, 2].text(
            0.5, 0.5,
            f"scale = mean ‖P‖₂\n= {scale:.2f} m",
            transform=axes[0, 2].transAxes, fontsize=16,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85),
        )
        axes[0, 2].set_title("Scale Factor Info")

        axes[0, 3].imshow(rgb_crop)
        axes[0, 3].set_title(f"DA3 Input ({da3_mode})")

        # ---- Row 1: Depth ----
        im0 = axes[1, 0].imshow(dep_orig, cmap=DEPTH_CMAP, vmin=vmin_abs, vmax=vmax_abs)
        rect2 = patches.Rectangle(
            (-0.5, CROP_TOP - 0.5), SRC_W, TGT_H,
            linewidth=2, edgecolor="lime", facecolor="none", linestyle="--",
        )
        axes[1, 0].add_patch(rect2)
        axes[1, 0].axhline(CROP_TOP, color="red", lw=1, ls=":")
        axes[1, 0].axhline(CROP_BOT, color="red", lw=1, ls=":")
        axes[1, 0].set_title(f"Original Depth  [{dep_orig.min():.1f}, {dep_orig.max():.1f}]m")
        fig.colorbar(im0, ax=axes[1, 0], fraction=0.046, pad=0.02, label="depth (m)")

        im1 = axes[1, 1].imshow(dep_crop, cmap=DEPTH_CMAP, vmin=vmin_abs, vmax=vmax_abs)
        axes[1, 1].axvline(PAD_L - 0.5, color="cyan", lw=1, ls=":")
        axes[1, 1].axvline(PAD_L + SRC_W - 0.5, color="cyan", lw=1, ls=":")
        axes[1, 1].set_title(f"Cropped Depth  [{dep_crop.min():.1f}, {dep_crop.max():.1f}]m")
        fig.colorbar(im1, ax=axes[1, 1], fraction=0.046, pad=0.02, label="depth (m)")

        im2 = axes[1, 2].imshow(dep_norm, cmap=DEPTH_CMAP, vmin=0, vmax=vmax_norm)
        axes[1, 2].set_title(
            f"Normalized Depth  [{dep_norm.min():.3f}, {dep_norm.max():.3f}]"
        )
        fig.colorbar(im2, ax=axes[1, 2], fraction=0.046, pad=0.02, label="depth / scale")

        da3_label = f"DA3 {da3_mode}"
        im3 = axes[1, 3].imshow(da3_depth, cmap=DEPTH_CMAP, vmin=0, vmax=vmax_da3)
        axes[1, 3].set_title(
            f"{da3_label}  [{da3_depth.min():.2f}, {da3_depth.max():.2f}]"
        )
        fig.colorbar(im3, ax=axes[1, 3], fraction=0.046, pad=0.02, label=da3_label)

        for ax in axes.flat:
            ax.axis("off")

        out_path = os.path.join(out_dir, f"{i:02d}_{scene}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(
            f"  [{i:02d}] {scene}  "
            f"clamp={clamp_val:.1f}m  scale={scale:.2f}m  "
            f"norm_p98={np.percentile(dep_norm, 98):.2f}  da3_p98={np.percentile(da3_depth, 98):.2f}  "
            f"→ {out_path}"
        )

    print(f"\nDone. {len(samples)} visualizations saved to {out_dir}/")


if __name__ == "__main__":
    main()
