"""Compare scale computation: with valid_mask (depth<100m) vs without mask (all pixels).

Samples from val_fortieth across diverse indoor/outdoor scenes.
Outputs:
  1. Per-sample figure: RGB | depth (masked vs unmasked regions) | scale comparison bar
  2. Summary scatter plot: scale_masked vs scale_all, colored by scene type
  3. Console statistics
"""

import json
import os
import sys
import zlib

import cv2
import lmdb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

LMDB_PATH = "/data-vepfs/lrd_projs/mono_depth_estimation/fast_mono_depth/data/tartanground.lmdb"
INDEX_JSON = "/data-vepfs/lrd_projs/mono_depth_estimation/fast_mono_depth/data/annotations/tartanground_val_fortieth.json"
OUT_DIR = "/data-vepfs/lrd_projs/mono_depth_estimation/fast_mono_depth/dataset_process/vis_scale_mask"
DEPTH_CAP = 100.0
INPUT_H, INPUT_W = 476, 644
DEPTH_CMAP = "turbo_r"


def _compute_scale(depth, valid_mask, K):
    h, w = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = np.arange(w, dtype=np.float64) + 0.5
    v = np.arange(h, dtype=np.float64) + 0.5
    u, v = np.meshgrid(u, v)
    Z = depth.astype(np.float64)
    X = (u - cx) / fx * Z
    Y = (v - cy) / fy * Z
    pts = np.stack([X, Y, Z], axis=-1)
    norms = np.linalg.norm(pts[valid_mask], axis=-1)
    if norms.size == 0:
        return 1.0
    return float(norms.mean())


def pick_diverse_samples(samples, n_per_scene=1, max_total=30):
    """Pick samples from diverse scenes, preferring variety."""
    by_scene = {}
    for i, s in enumerate(samples):
        scene = s["key_prefix"].split("/")[0]
        by_scene.setdefault(scene, []).append(i)

    picked = []
    for scene in sorted(by_scene.keys()):
        idxs = by_scene[scene]
        step = max(1, len(idxs) // n_per_scene)
        for j in range(0, len(idxs), step):
            if len(picked) >= max_total:
                break
            picked.append(idxs[j])
        if len(picked) >= max_total:
            break
    return picked


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(INDEX_JSON) as f:
        index = json.load(f)
    K = np.array(index["meta"]["K_cropped"], dtype=np.float64)
    samples = index["samples"]

    env = lmdb.open(LMDB_PATH, readonly=True, lock=False, readahead=False, map_size=1 << 40)

    # --- Load DA3 model for anyview depth ---
    DA3_SRC = "/data-vepfs/lrd_projs/mono_depth_estimation/da3/src"
    DA3_MODEL_DIR = "/data-vepfs/lrd_projs/mono_depth_estimation/da3/models/DA3NESTED-GIANT-LARGE-1.1"
    print("Loading DA3 model...")
    if DA3_SRC not in sys.path:
        sys.path.insert(0, DA3_SRC)
    from depth_anything_3.api import DepthAnything3
    da3_model = DepthAnything3.from_pretrained(DA3_MODEL_DIR).to("cuda")
    da3_model.eval()
    print("DA3 model loaded.")

    picked_idxs = pick_diverse_samples(samples, n_per_scene=1, max_total=30)

    results = []

    for rank, idx in enumerate(picked_idxs):
        s = samples[idx]
        scene = s["key_prefix"].split("/")[0]
        frame_key = f"{s['key_prefix']}/{s['frame']:06d}"

        with env.begin() as txn:
            rgb_buf = txn.get(f"{frame_key}/rgb".encode())
            depth_buf = txn.get(f"{frame_key}/depth".encode())

        image = cv2.imdecode(np.frombuffer(rgb_buf, dtype=np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # --- DA3 pure anyview branch inference (same as teacher in training) ---
        imgs_cpu, _, _ = da3_model._preprocess_inputs([image], None, None, 504, "upper_bound_resize")
        imgs_dev, _, _ = da3_model._prepare_model_inputs(imgs_cpu, None, None)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            da3_out = da3_model.model.da3(imgs_dev)
        da3_depth_t = da3_out.depth[0, 0]  # [H_proc, W_proc]
        da3_depth_t = torch.nn.functional.interpolate(
            da3_depth_t[None, None].float(), size=(INPUT_H, INPUT_W), mode="bilinear", align_corners=False
        )[0, 0]
        da3_depth = da3_depth_t.cpu().numpy()

        depth = np.frombuffer(zlib.decompress(depth_buf), dtype=np.float32).reshape(INPUT_H, INPUT_W).copy()

        valid_mask = depth < DEPTH_CAP
        n_valid = valid_mask.sum()
        n_total = valid_mask.size
        pct_valid = 100.0 * n_valid / n_total

        depth_clipped = np.clip(depth, 0, DEPTH_CAP)

        scale_masked = _compute_scale(depth_clipped, valid_mask, K)

        all_mask = np.ones_like(valid_mask)
        scale_all = _compute_scale(depth_clipped, all_mask, K)

        ratio = scale_all / scale_masked if scale_masked > 0 else float("inf")
        diff_abs = scale_all - scale_masked
        diff_pct = 100.0 * diff_abs / scale_masked if scale_masked > 0 else float("inf")

        results.append({
            "scene": scene, "idx": idx,
            "scale_masked": scale_masked,
            "scale_clipped_all": scale_all,
            "pct_valid": pct_valid,
            "ratio": ratio,
            "diff_abs": diff_abs,
            "depth_max": float(depth.max()),
            "depth_median_valid": float(np.median(depth[valid_mask])) if n_valid > 0 else 0,
        })

        # --- DA3 anyview AbsRel (median-scaled to GT, same as eval.py) ---
        gt_norm = depth_clipped / scale_masked
        metric_mask = valid_mask & (gt_norm > 1e-3)
        n_metric = metric_mask.sum()
        if n_metric > 0:
            da3_v = da3_depth[metric_mask]
            gt_v = gt_norm[metric_mask]
            median_scale = np.median(gt_v) / np.clip(np.median(da3_v), 1e-8, None)
            da3_scaled = da3_v * median_scale
            da3_absrel = float(np.mean(np.abs(da3_scaled - gt_v) / gt_v))
        else:
            da3_absrel = float("nan")
            median_scale = float("nan")

        # --- Rescaled depths ---
        depth_norm_valid = depth_clipped / scale_masked
        depth_norm_all = depth_clipped / scale_all

        # --- Filled version: invalid pixels → max valid depth ---
        valid_max_norm = depth_norm_valid[valid_mask].max() if n_valid > 0 else depth_norm_valid.max()
        depth_norm_filled = depth_norm_valid.copy()
        depth_norm_filled[~valid_mask] = valid_max_norm

        # --- Per-sample figure: 3 rows x 2 cols + bar on right ---
        fig = plt.figure(figsize=(28, 20))
        gs = gridspec.GridSpec(3, 3, width_ratios=[1, 1, 0.45], wspace=0.20, hspace=0.28)

        vmax_depth = np.percentile(depth_clipped, 99)
        vmax_valid = np.percentile(depth_norm_valid, 99)
        vmax_all = np.percentile(depth_norm_all, 99)

        # (0,0) RGB
        ax00 = fig.add_subplot(gs[0, 0])
        ax00.imshow(image)
        ax00.set_title(f"RGB — {scene}", fontsize=12)
        ax00.axis("off")

        # (0,1) Depth clipped
        ax01 = fig.add_subplot(gs[0, 1])
        im01 = ax01.imshow(depth_clipped, cmap=DEPTH_CMAP, vmin=0, vmax=vmax_depth)
        ax01.set_title(
            f"Depth (clipped to {DEPTH_CAP:.0f}m)\n"
            f"valid={pct_valid:.1f}%  range=[{depth_clipped.min():.1f}, {depth_clipped.max():.1f}]m",
            fontsize=11,
        )
        ax01.axis("off")
        fig.colorbar(im01, ax=ax01, fraction=0.046, pad=0.02, label="depth (m)")

        # (1,0) Rescaled by valid-only scale (white = masked)
        ax10 = fig.add_subplot(gs[1, 0])
        depth_norm_valid_vis = depth_norm_valid.copy()
        depth_norm_valid_vis[~valid_mask] = np.nan
        cmap_masked = plt.get_cmap(DEPTH_CMAP).copy()
        cmap_masked.set_bad(color="white")
        im10 = ax10.imshow(depth_norm_valid_vis, cmap=cmap_masked, vmin=0, vmax=vmax_valid)
        ax10.set_title(
            f"depth / scale_valid (white=masked)\n"
            f"scale={scale_masked:.2f}m  "
            f"range=[{depth_norm_valid.min():.3f}, {depth_norm_valid.max():.3f}]",
            fontsize=11, color="#2ecc71", fontweight="bold",
        )
        ax10.axis("off")
        fig.colorbar(im10, ax=ax10, fraction=0.046, pad=0.02, label="depth / scale")

        # (1,1) Rescaled by all-clipped scale
        ax11 = fig.add_subplot(gs[1, 1])
        im11 = ax11.imshow(depth_norm_all, cmap=DEPTH_CMAP, vmin=0, vmax=vmax_all)
        ax11.set_title(
            f"depth / scale_all (no mask)\n"
            f"scale={scale_all:.2f}m  "
            f"range=[{depth_norm_all.min():.3f}, {depth_norm_all.max():.3f}]",
            fontsize=11, color="#e67e22", fontweight="bold",
        )
        ax11.axis("off")
        fig.colorbar(im11, ax=ax11, fraction=0.046, pad=0.02, label="depth / scale")

        # (2,0) Filled: invalid pixels → valid max value
        ax20 = fig.add_subplot(gs[2, 0])
        im20 = ax20.imshow(depth_norm_filled, cmap=DEPTH_CMAP, vmin=0, vmax=vmax_valid)
        ax20.set_title(
            f"depth / scale_valid (invalid→valid_max)\n"
            f"scale={scale_masked:.2f}m  valid_max={valid_max_norm:.3f}  "
            f"range=[{depth_norm_filled.min():.3f}, {depth_norm_filled.max():.3f}]",
            fontsize=11, color="#3498db", fontweight="bold",
        )
        ax20.axis("off")
        fig.colorbar(im20, ax=ax20, fraction=0.046, pad=0.02, label="depth / scale")

        # (2,1) DA3 anyview depth
        ax21 = fig.add_subplot(gs[2, 1])
        vmax_da3 = np.percentile(da3_depth, 99)
        im21 = ax21.imshow(da3_depth, cmap=DEPTH_CMAP, vmin=0, vmax=vmax_da3)
        ax21.set_title(
            f"DA3 Anyview Depth (pure da3 branch)\n"
            f"range=[{da3_depth.min():.3f}, {da3_depth.max():.3f}]  "
            f"AbsRel={da3_absrel:.4f}  median_scale={median_scale:.3f}",
            fontsize=11, color="#9b59b6", fontweight="bold",
        )
        ax21.axis("off")
        fig.colorbar(im21, ax=ax21, fraction=0.046, pad=0.02, label="DA3 depth")

        # Right column: scale comparison bar (spans all rows)
        ax_bar = fig.add_subplot(gs[:, 2])
        labels = ["valid only\n(depth<100m)", "all pixels\n(clipped to 100m)"]
        values = [scale_masked, scale_all]
        colors = ["#2ecc71", "#e67e22"]
        bars = ax_bar.barh(labels, values, color=colors, height=0.45)
        for bar, val in zip(bars, values):
            ax_bar.text(bar.get_width() + max(values) * 0.02, bar.get_y() + bar.get_height() / 2,
                        f"{val:.2f}m", va="center", fontsize=12, fontweight="bold")
        ax_bar.set_xlabel("Scale (mean ||P||₂)", fontsize=12)
        ax_bar.set_title(f"Scale Comparison\nratio={ratio:.2f}x\nΔ={diff_abs:+.2f}m ({diff_pct:+.1f}%)", fontsize=12)
        ax_bar.set_xlim(0, max(values) * 1.4)

        fig.suptitle(
            f"[{rank}] {scene}  —  valid: {pct_valid:.1f}%  |  "
            f"scale_valid={scale_masked:.2f}m  scale_all={scale_all:.2f}m  "
            f"ratio={ratio:.2f}x  Δ={diff_abs:+.2f}m ({diff_pct:+.1f}%)",
            fontsize=13, fontweight="bold",
        )

        out_path = os.path.join(OUT_DIR, f"{rank:02d}_{scene}.png")
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  [{rank:02d}] {scene:<30s}  valid={pct_valid:5.1f}%  "
              f"scale_v={scale_masked:8.2f}  scale_a={scale_all:8.2f}  "
              f"ratio={ratio:.4f}  Δ={diff_abs:+.2f}m ({diff_pct:+.1f}%)")

    env.close()

    # --- Summary scatter plot ---
    fig2, axes2 = plt.subplots(1, 3, figsize=(24, 7))

    pcts = [r["pct_valid"] for r in results]
    s_masked = [r["scale_masked"] for r in results]
    s_clipped = [r["scale_clipped_all"] for r in results]
    ratios = [r["ratio"] for r in results]
    diffs = [r["diff_abs"] for r in results]
    scenes = [r["scene"] for r in results]

    # Plot 1: scale_masked vs scale_clipped_all scatter
    ax = axes2[0]
    sc = ax.scatter(s_masked, s_clipped, c=pcts, cmap="RdYlGn", s=80, edgecolors="k", linewidths=0.5)
    max_val = max(max(s_masked), max(s_clipped)) * 1.1
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="y=x (no diff)")
    ax.set_xlabel("Scale (valid pixels only, depth<100m)", fontsize=12)
    ax.set_ylabel("Scale (all pixels, clipped to 100m)", fontsize=12)
    ax.set_title("Scale: Valid-Only vs All-Clipped", fontsize=13)
    ax.legend()
    fig2.colorbar(sc, ax=ax, label="% valid pixels")
    for i, r in enumerate(results):
        if r["ratio"] > 1.3 or r["pct_valid"] < 70:
            ax.annotate(r["scene"], (r["scale_masked"], r["scale_clipped_all"]),
                        fontsize=7, rotation=15, alpha=0.8)

    # Plot 2: ratio vs % valid
    ax = axes2[1]
    sc2 = ax.scatter(pcts, ratios, c=s_masked, cmap="viridis", s=80, edgecolors="k", linewidths=0.5)
    ax.axhline(1.0, color="k", ls="--", alpha=0.5)
    ax.set_xlabel("% Valid Pixels (depth < 100m)", fontsize=12)
    ax.set_ylabel("Ratio (all_clipped / valid_only)", fontsize=12)
    ax.set_title("Scale Ratio vs Valid Pixel %", fontsize=13)
    fig2.colorbar(sc2, ax=ax, label="scale_valid (m)")
    for i, r in enumerate(results):
        if r["ratio"] > 1.15 or r["pct_valid"] < 70:
            ax.annotate(r["scene"], (r["pct_valid"], r["ratio"]),
                        fontsize=7, rotation=15, alpha=0.8)

    # Plot 3: histogram of diff %
    ax = axes2[2]
    diff_pcts = [100.0 * r["diff_abs"] / r["scale_masked"] if r["scale_masked"] > 0 else 0 for r in results]
    ax.hist(diff_pcts, bins=20, color="#3498db", edgecolor="k", alpha=0.8)
    ax.axvline(0, color="r", ls="--", lw=2, label="0% (no diff)")
    ax.axvline(np.median(diff_pcts), color="orange", ls="-", lw=2,
               label=f"median={np.median(diff_pcts):.1f}%")
    ax.set_xlabel("Scale Difference (%)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Scale Differences", fontsize=13)
    ax.legend()

    fig2.suptitle(
        f"Scale: valid_mask vs clipped_all — {len(results)} samples, {len(set(scenes))} scenes\n"
        f"median ratio={np.median(ratios):.4f}  "
        f"max ratio={max(ratios):.4f}  "
        f"median Δ%={np.median(diff_pcts):.1f}%  "
        f"mean valid%={np.mean(pcts):.1f}%",
        fontsize=14, fontweight="bold",
    )
    fig2.tight_layout()
    summary_path = os.path.join(OUT_DIR, "summary_scatter.png")
    fig2.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"\nSummary plot saved to {summary_path}")

    # --- Console summary ---
    print("\n" + "=" * 80)
    print("SUMMARY: scale(valid_only) vs scale(all_clipped_to_100m)")
    print("=" * 80)
    ratios_arr = np.array(ratios)
    pcts_arr = np.array(pcts)
    diffs_arr = np.array(diffs)
    dpcts_arr = np.array(diff_pcts)
    print(f"Samples: {len(results)}  |  Scenes: {len(set(scenes))}")
    print(f"Valid pixel %: min={pcts_arr.min():.1f}  median={np.median(pcts_arr):.1f}  max={pcts_arr.max():.1f}")
    print(f"Scale ratio (all_clipped/valid): min={ratios_arr.min():.4f}  median={np.median(ratios_arr):.4f}  max={ratios_arr.max():.4f}")
    print(f"Abs diff (m): min={diffs_arr.min():.2f}  median={np.median(diffs_arr):.2f}  max={diffs_arr.max():.2f}")
    print(f"Rel diff (%): min={dpcts_arr.min():.1f}  median={np.median(dpcts_arr):.1f}  max={dpcts_arr.max():.1f}")
    print(f"  ratio > 1.05: {(ratios_arr > 1.05).sum()} / {len(ratios_arr)}")
    print(f"  ratio > 1.10: {(ratios_arr > 1.10).sum()} / {len(ratios_arr)}")
    print(f"  ratio > 1.20: {(ratios_arr > 1.20).sum()} / {len(ratios_arr)}")
    print(f"  ratio > 1.50: {(ratios_arr > 1.50).sum()} / {len(ratios_arr)}")

    print("\nTop 10 most affected (sorted by ratio):")
    sorted_results = sorted(results, key=lambda r: r["ratio"], reverse=True)
    for r in sorted_results[:10]:
        dp = 100.0 * r["diff_abs"] / r["scale_masked"] if r["scale_masked"] > 0 else 0
        print(f"  {r['scene']:<30s}  valid={r['pct_valid']:5.1f}%  "
              f"scale_v={r['scale_masked']:7.2f}  scale_a={r['scale_clipped_all']:7.2f}  "
              f"ratio={r['ratio']:.4f}  Δ={r['diff_abs']:+.2f}m ({dp:+.1f}%)")

    print("\nTop 5 least affected:")
    for r in sorted_results[-5:]:
        dp = 100.0 * r["diff_abs"] / r["scale_masked"] if r["scale_masked"] > 0 else 0
        print(f"  {r['scene']:<30s}  valid={r['pct_valid']:5.1f}%  "
              f"scale_v={r['scale_masked']:7.2f}  scale_a={r['scale_clipped_all']:7.2f}  "
              f"ratio={r['ratio']:.4f}  Δ={r['diff_abs']:+.2f}m ({dp:+.1f}%)")


if __name__ == "__main__":
    main()
