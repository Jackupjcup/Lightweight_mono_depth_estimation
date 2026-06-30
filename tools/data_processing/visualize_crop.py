"""Visualize sky-aware depth normalization on TartanGround samples.

Layout per sample (5 columns × 2 rows):
  Row 0: Cropped RGB | Sky Mask on RGB | Scale Info     | (anyview RGB)   | (nested RGB)
  Row 1: Cropped Depth | Binary Sky Mask | Rescaled Depth | Anyview Depth  | Nested Depth
"""

import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch

from preprocess import (
    TGT_H, TGT_W,
    K_CROPPED,
    decode_depth, crop_and_pad, depth_to_pointcloud, normalize_depth,
)

DATA_ROOT = "/data-tos-daily/TartanGround"
OUT_DIR = "/data-vepfs/lrd_projs/mono_depth_estimation/fast_mono_depth/dataset_process/vis_crop"
DA3_MODEL_DIR = "/data-vepfs/lrd_projs/mono_depth_estimation/da3/models/DA3NESTED-GIANT-LARGE-1.1"

DEPTH_CMAP = "turbo_r"
DEPTH_CLAMP = 100.0


def load_da3_model():
    sys.path.insert(0, "/data-vepfs/lrd_projs/mono_depth_estimation/da3/src")
    from depth_anything_3.api import DepthAnything3
    return DepthAnything3.from_pretrained(DA3_MODEL_DIR).to("cuda")


def da3_predict(model, rgb_crop):
    """Run DA3 and return (nested_depth, anyview_depth, sky_mask).

    nested_depth: (H, W) final output from the full nested model.
    anyview_depth: (H, W) depth from the anyview branch only.
    sky_mask: (H, W) bool array from the metric branch (or None).
    """
    captured = {}

    def _sky_hook(module, inp, out):
        if "sky" in out:
            captured["sky"] = out["sky"].detach().cpu()

    def _anyview_hook(module, inp, out):
        if "depth" in out:
            captured["anyview_depth"] = out["depth"].detach().cpu()

    h_sky = model.model.da3_metric.register_forward_hook(_sky_hook)
    h_any = model.model.da3.register_forward_hook(_anyview_hook)
    try:
        prediction = model.inference(image=[rgb_crop])
    finally:
        h_sky.remove()
        h_any.remove()

    nested_depth = prediction.depth[0]

    anyview_depth = None
    if "anyview_depth" in captured:
        anyview_depth = captured["anyview_depth"].squeeze().numpy()

    sky_mask = None
    if "sky" in captured:
        sky_raw = captured["sky"].squeeze().numpy()
        sky_small = sky_raw >= 0.5
        h_crop, w_crop = rgb_crop.shape[:2]
        if sky_small.shape != (h_crop, w_crop):
            sky_mask = np.array(Image.fromarray(sky_small).resize((w_crop, h_crop), Image.NEAREST))
        else:
            sky_mask = sky_small
    elif prediction.sky is not None:
        sky_mask = prediction.sky[0]

    return nested_depth, anyview_depth, sky_mask


def compute_sky_aware_scale(dep_crop, sky_mask):
    """Compute scale = mean ‖P‖₂ of non-sky points."""
    points = depth_to_pointcloud(dep_crop, K_CROPPED)
    norms = np.linalg.norm(points, axis=-1)
    non_sky = norms[~sky_mask]
    if non_sky.size == 0:
        return 1.0
    return float(non_sky.mean())


def collect_samples(n=20):
    samples = []
    print("Scanning scenes...", end="", flush=True)
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
            print(f" {scene}", end="", flush=True)
        if len(samples) >= n:
            break
    print()
    return samples


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    samples = collect_samples(50)
    print(f"Found {len(samples)} samples")

    print("Loading DA3 model...")
    da3_model = load_da3_model()
    print("DA3 model loaded.")

    for i, (rgb_path, dep_path, scene) in enumerate(samples):
        rgb_crop = crop_and_pad(np.array(Image.open(rgb_path)))
        dep_crop_raw = crop_and_pad(decode_depth(dep_path))

        # Col 0: clamp >100m then normalize_depth (mean ‖P‖₂)
        dep_clamped = np.clip(dep_crop_raw, 0, DEPTH_CLAMP)
        dep_norm, scale_clamp = normalize_depth(dep_clamped, K_CROPPED)

        nested_depth, anyview_depth, sky_mask = da3_predict(da3_model, rgb_crop)
        if sky_mask is None:
            print(f"  [{i:02d}] {scene}  sky=N/A, skipping")
            continue

        # Col 2: sky-aware scale on clamped depth
        scale_sky = compute_sky_aware_scale(dep_clamped, sky_mask)
        dep_rescaled_sky = dep_clamped / scale_sky

        sky_ratio = sky_mask.sum() / sky_mask.size * 100
        vmax_norm = np.percentile(dep_norm, 98)
        vmax_rescaled_sky = np.percentile(dep_rescaled_sky, 98)
        vmax_anyview = np.percentile(anyview_depth, 98) if anyview_depth is not None else 1
        vmax_nested = np.percentile(nested_depth, 98)

        fig, axes = plt.subplots(2, 5, figsize=(35, 12),
                                 gridspec_kw={"hspace": 0.25, "wspace": 0.18})
        fig.suptitle(f"[{i}] {scene}  —  {TGT_W}×{TGT_H}    "
                     f"sky={sky_ratio:.1f}%    clamp_scale={scale_clamp:.2f}m    sky_scale={scale_sky:.2f}m",
                     fontsize=14)

        # ---- Col 0: Clamped + Rescaled ----
        axes[0, 0].imshow(rgb_crop)
        axes[0, 0].text(
            0.5, 0.05,
            f"clamp={DEPTH_CLAMP:.0f}m  scale={scale_clamp:.2f}m",
            transform=axes[0, 0].transAxes, fontsize=12,
            ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85),
        )
        axes[0, 0].set_title(f"Cropped RGB  {TGT_W}×{TGT_H}")

        im0 = axes[1, 0].imshow(dep_norm, cmap=DEPTH_CMAP, vmin=0, vmax=vmax_norm)
        axes[1, 0].set_title(
            f"Clamp+Rescale  [{dep_norm.min():.3f}, {dep_norm.max():.3f}]"
        )
        fig.colorbar(im0, ax=axes[1, 0], fraction=0.046, pad=0.02, label="depth / scale")

        # ---- Col 1: Sky Mask ----
        axes[0, 1].imshow(rgb_crop)
        sky_overlay = np.zeros((*sky_mask.shape, 4))
        sky_overlay[sky_mask] = [1, 0, 0, 0.45]
        axes[0, 1].imshow(sky_overlay)
        axes[0, 1].set_title(f"Sky Mask on RGB  ({sky_ratio:.1f}%)")

        axes[1, 1].imshow(sky_mask.astype(np.uint8), cmap="gray", vmin=0, vmax=1)
        axes[1, 1].set_title("Sky Mask (binary)")

        # ---- Col 2: Sky-aware Rescaled Depth ----
        axes[0, 2].imshow(rgb_crop, alpha=0.3)
        axes[0, 2].text(
            0.5, 0.5,
            f"sky-aware scale\n= mean ‖P‖₂ (non-sky)\n= {scale_sky:.2f} m",
            transform=axes[0, 2].transAxes, fontsize=16,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85),
        )
        axes[0, 2].set_title("Sky-aware Scale Info")

        im2 = axes[1, 2].imshow(dep_rescaled_sky, cmap=DEPTH_CMAP, vmin=0, vmax=vmax_rescaled_sky)
        axes[1, 2].set_title(
            f"Sky Rescaled  [{dep_rescaled_sky.min():.3f}, {dep_rescaled_sky.max():.3f}]"
        )
        fig.colorbar(im2, ax=axes[1, 2], fraction=0.046, pad=0.02, label="depth / sky_scale")

        # ---- Col 3: Anyview Branch Depth ----
        axes[0, 3].imshow(rgb_crop)
        axes[0, 3].set_title("Anyview Input")

        if anyview_depth is not None:
            im3 = axes[1, 3].imshow(anyview_depth, cmap=DEPTH_CMAP, vmin=0, vmax=vmax_anyview)
            axes[1, 3].set_title(
                f"Anyview Depth  [{anyview_depth.min():.2f}, {anyview_depth.max():.2f}]"
            )
            fig.colorbar(im3, ax=axes[1, 3], fraction=0.046, pad=0.02, label="anyview depth")
        else:
            axes[1, 3].set_title("Anyview Depth (N/A)")

        # ---- Col 4: Nested (full) Depth ----
        axes[0, 4].imshow(rgb_crop)
        axes[0, 4].set_title("Nested Input")

        im4 = axes[1, 4].imshow(nested_depth, cmap=DEPTH_CMAP, vmin=0, vmax=vmax_nested)
        axes[1, 4].set_title(
            f"Nested Depth  [{nested_depth.min():.2f}, {nested_depth.max():.2f}]"
        )
        fig.colorbar(im4, ax=axes[1, 4], fraction=0.046, pad=0.02, label="nested depth")

        for ax in axes.flat:
            ax.axis("off")

        out_path = os.path.join(OUT_DIR, f"{i:02d}_{scene}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [{i:02d}] {scene}  sky={sky_ratio:.1f}%  "
              f"clamp_scale={scale_clamp:.2f}m  sky_scale={scale_sky:.2f}m  → {out_path}")

    print(f"\nDone. {len(samples)} visualizations saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
