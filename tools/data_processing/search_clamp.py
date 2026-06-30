"""Search for optimal DEPTH_CLAMP that minimizes the gap between
normalized-depth range and DA3-predicted-depth range across 20 samples."""

import os, sys
import numpy as np
from PIL import Image

sys.path.insert(0, "/data-vepfs/lrd_projs/mono_depth_estimation/da3/src")

from preprocess import (
    K_CROPPED,
    decode_depth, crop_and_pad, normalize_depth,
)

DATA_ROOT = "/data-tos-daily/TartanGround"
DA3_MODEL_DIR = "/data-vepfs/lrd_projs/mono_depth_estimation/da3/models/DA3NESTED-GIANT-LARGE-1.1"


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


def main():
    samples = collect_samples(20)
    print(f"Found {len(samples)} samples")

    # --- Step 1: pre-load raw depths and DA3 predictions (both independent of clamp) ---
    raw_depths = []
    da3_depths = []

    from depth_anything_3.api import DepthAnything3
    print("Loading DA3 model...")
    model = DepthAnything3.from_pretrained(DA3_MODEL_DIR).to("cuda")
    print("DA3 model loaded.")

    for i, (rgb_path, dep_path, scene) in enumerate(samples):
        raw_depths.append(decode_depth(dep_path))
        rgb_crop = crop_and_pad(np.array(Image.open(rgb_path)))
        pred = model.inference(image=[rgb_crop])
        da3_depths.append(pred.depth[0])
        print(f"  [{i:02d}] {scene}  da3 p98={np.percentile(pred.depth[0], 98):.2f}")

    # --- Step 2: search DEPTH_CLAMP ---
    candidates = list(range(5, 51, 1)) + list(range(55, 105, 5)) + list(range(110, 510, 10)) + list(range(550, 2050, 50)) + list(range(2100, 5100, 100))

    print(f"\nSearching {len(candidates)} DEPTH_CLAMP candidates...")
    print(f"{'CLAMP':>8}  {'mean|log_ratio|':>16}  {'median|log_ratio|':>18}  {'worst|log_ratio|':>18}")

    best_clamp = None
    best_score = float("inf")
    all_results = []

    for clamp in candidates:
        log_ratios = []
        for j in range(len(samples)):
            dep_clamped = np.clip(raw_depths[j], 0, clamp)
            dep_crop = crop_and_pad(dep_clamped)
            dep_norm, scale = normalize_depth(dep_crop, K_CROPPED)

            p98_norm = np.percentile(dep_norm, 98)
            p98_da3 = np.percentile(da3_depths[j], 98)

            if p98_da3 > 0 and p98_norm > 0:
                log_ratios.append(abs(np.log(p98_norm / p98_da3)))

        mean_lr = np.mean(log_ratios)
        median_lr = np.median(log_ratios)
        worst_lr = np.max(log_ratios)
        all_results.append((clamp, mean_lr, median_lr, worst_lr))

        if mean_lr < best_score:
            best_score = mean_lr
            best_clamp = clamp

    # print top 10
    all_results.sort(key=lambda x: x[1])
    print(f"\n--- Top 10 DEPTH_CLAMP candidates (by mean |log ratio|) ---")
    for clamp, mean_lr, median_lr, worst_lr in all_results[:10]:
        print(f"  CLAMP={clamp:>5}  mean={mean_lr:.4f}  median={median_lr:.4f}  worst={worst_lr:.4f}")

    print(f"\nBest DEPTH_CLAMP = {best_clamp}  (mean |log(p98_norm/p98_da3)| = {best_score:.4f})")

    # --- Step 3: show per-sample detail for the best clamp ---
    print(f"\n--- Per-sample detail for DEPTH_CLAMP={best_clamp} ---")
    for j, (_, _, scene) in enumerate(samples):
        dep_clamped = np.clip(raw_depths[j], 0, best_clamp)
        dep_crop = crop_and_pad(dep_clamped)
        dep_norm, scale = normalize_depth(dep_crop, K_CROPPED)

        p98_norm = np.percentile(dep_norm, 98)
        p98_da3 = np.percentile(da3_depths[j], 98)
        ratio = p98_norm / p98_da3 if p98_da3 > 0 else float("inf")

        print(f"  [{j:02d}] {scene:<30s}  p98_norm={p98_norm:>7.2f}  p98_da3={p98_da3:>7.2f}  ratio={ratio:.3f}  scale={scale:.2f}")


if __name__ == "__main__":
    main()
