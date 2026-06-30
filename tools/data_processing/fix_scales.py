#!/usr/bin/env python3
"""Recompute all scales with depth clamp=100m, update JSON index files.

Reads depth from LMDB, applies np.clip(depth, 0, 100), computes
scale = mean ||P||_2 of valid points, then regenerates index/train/val JSON.
"""

import json
import os
import sys
import time
import zlib

import lmdb
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import K_CROPPED, TGT_H, TGT_W, depth_to_pointcloud

OUT_DIR = "/root/vepfs/lrd_projs/mono_depth_estimation/fast_mono_depth/data"
LMDB_DIR = os.path.join(OUT_DIR, "tartanground.lmdb")
ANNO_DIR = os.path.join(OUT_DIR, "annotations")
INDEX_PATH = os.path.join(ANNO_DIR, "tartanground_index.json")
TRAIN_PATH = os.path.join(ANNO_DIR, "tartanground_train.json")
VAL_PATH = os.path.join(ANNO_DIR, "tartanground_val.json")

DEPTH_CLAMP = 100.0


def compute_scale_clamped(depth_f32):
    depth_clamped = np.clip(depth_f32, 0, DEPTH_CLAMP)
    points = depth_to_pointcloud(depth_clamped, K_CROPPED)
    norms = np.linalg.norm(points, axis=-1)
    valid = (depth_clamped >= 0.5) & (depth_clamped <= DEPTH_CLAMP)
    valid_norms = norms[valid]
    if valid_norms.size == 0:
        return 1.0
    return float(valid_norms.mean())


def main():
    with open(INDEX_PATH) as f:
        index = json.load(f)

    samples = index["samples"]
    total = len(samples)
    print(f"Recomputing scales for {total} samples (clamp={DEPTH_CLAMP}m)...\n")

    env = lmdb.open(LMDB_DIR, readonly=True, lock=False)
    t0 = time.time()

    with env.begin() as txn:
        for i, s in enumerate(samples):
            frame_key = f"{s['key_prefix']}/{s['frame']:06d}"
            depth_buf = txn.get(f"{frame_key}/depth".encode())
            if depth_buf is None:
                s["scale"] = 1.0
                continue
            depth = np.frombuffer(
                zlib.decompress(depth_buf), dtype=np.float32
            ).reshape(TGT_H, TGT_W)
            s["scale"] = round(compute_scale_clamped(depth), 6)

            if (i + 1) % 50000 == 0 or (i + 1) == total:
                elapsed = time.time() - t0
                fps = (i + 1) / elapsed
                eta = (total - i - 1) / fps if fps > 0 else 0
                print(f"\r  {i+1}/{total}  ({fps:.0f}/s, ETA {eta:.0f}s)   ",
                      end="", flush=True)

    env.close()
    elapsed = time.time() - t0
    print(f"\n\nDone: {total} scales in {elapsed:.1f}s\n")

    # Save index
    with open(INDEX_PATH, "w") as f:
        json.dump(index, f)
    print(f"Saved {INDEX_PATH}")

    # Update train/val with new scales
    sample_scale = {(s["key_prefix"], s["frame"]): s["scale"] for s in samples}

    for path, name in [(TRAIN_PATH, "train"), (VAL_PATH, "val")]:
        with open(path) as f:
            split = json.load(f)
        for s in split["samples"]:
            s["scale"] = sample_scale[(s["key_prefix"], s["frame"])]
        with open(path, "w") as f:
            json.dump(split, f)
        print(f"Saved {path} ({len(split['samples'])} samples)")

    # Verify
    scales = [s["scale"] for s in samples]
    print(f"\nScale stats: min={min(scales):.3f}, max={max(scales):.3f}, "
          f"mean={np.mean(scales):.3f}, median={np.median(scales):.3f}")


if __name__ == "__main__":
    main()
