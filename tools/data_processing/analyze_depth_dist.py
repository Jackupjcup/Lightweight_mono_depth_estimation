"""Analyze depth distribution of 20 TartanGround samples to understand layering."""

import os
import numpy as np
from preprocess import decode_depth

DATA_ROOT = "/data-tos-daily/TartanGround"


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
        dep_dir = os.path.join(diff_dir, traj, "depth_lcam_front")
        if not os.path.isdir(dep_dir):
            continue
        dep_path = os.path.join(dep_dir, "000000_lcam_front_depth.png")
        if os.path.isfile(dep_path):
            samples.append((dep_path, scene))
        if len(samples) >= n:
            break
    return samples


def find_gap(depth_flat, percentiles=(90, 95, 99)):
    """Find the depth gap between scene objects and sky."""
    sorted_d = np.sort(depth_flat)
    n = len(sorted_d)

    results = {}
    for p in percentiles:
        idx = int(n * p / 100)
        results[f"p{p}"] = sorted_d[min(idx, n - 1)]

    # Find largest gap in the top 10% of depth values
    top_start = int(n * 0.90)
    top_vals = sorted_d[top_start:]
    if len(top_vals) > 1:
        diffs = np.diff(top_vals)
        gap_idx = np.argmax(diffs)
        gap_below = top_vals[gap_idx]
        gap_above = top_vals[gap_idx + 1]
        gap_size = gap_above - gap_below
        results["gap_below"] = gap_below
        results["gap_above"] = gap_above
        results["gap_size"] = gap_size
        results["gap_position"] = (top_start + gap_idx) / n * 100
    return results


def main():
    samples = collect_samples(20)
    print(f"Found {len(samples)} samples\n")

    print(f"{'Scene':<30s}  {'min':>6}  {'p50':>6}  {'p90':>7}  {'p95':>7}  {'p99':>8}  {'max':>8}  |  {'gap_below':>10}  {'gap_above':>10}  {'gap_size':>9}  {'gap_%':>6}  {'sky%':>5}")
    print("-" * 160)

    for dep_path, scene in samples:
        dep = decode_depth(dep_path)
        flat = dep.flatten()
        valid = flat[flat > 0.1]

        info = find_gap(valid)
        gap_below = info.get("gap_below", 0)
        gap_above = info.get("gap_above", 0)
        gap_size = info.get("gap_size", 0)
        gap_pos = info.get("gap_position", 0)

        sky_pct = np.sum(valid > gap_below) / len(valid) * 100 if gap_below > 0 else 0

        print(
            f"{scene:<30s}  "
            f"{valid.min():>6.1f}  {np.median(valid):>6.1f}  "
            f"{info['p90']:>7.1f}  {info['p95']:>7.1f}  {info['p99']:>8.1f}  {valid.max():>8.1f}  |  "
            f"{gap_below:>10.1f}  {gap_above:>10.1f}  {gap_size:>9.1f}  {gap_pos:>5.1f}%  {sky_pct:>4.1f}%"
        )


if __name__ == "__main__":
    main()
