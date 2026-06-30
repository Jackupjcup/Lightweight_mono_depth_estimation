#!/usr/bin/env python3
"""One-time fix: reprocess 17 incomplete trajectories + backfill all scales into JSON.

1. Find old trajectories with missing frames (first frame exists but last doesn't)
2. Reprocess them from source → write rgb/depth/pose to LMDB + collect scales
3. Read scale from LMDB for remaining old complete trajectories
4. Merge with .scales_cache.json (new trajectories)
5. Regenerate index + train/val JSON with full scale coverage
"""

import json
import os
import sys
import time

import lmdb
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from build_tartanground_lmdb import (
    LMDB_DIR,
    INDEX_PATH,
    TRAIN_PATH,
    VAL_PATH,
    OUT_DIR,
    SCAN_CACHE,
    build_index,
    fmt_eta,
    process_trajectory,
)
from multiprocessing import Pool


def find_incomplete_trajs(trajs, lmdb_dir):
    env = lmdb.open(lmdb_dir, readonly=True, lock=False)
    incomplete = []
    with env.begin() as txn:
        for t in trajs:
            kp = t["key_prefix"]
            if txn.get(f"{kp}/000000/rgb".encode()) is None:
                continue
            last = t["num_frames"] - 1
            if txn.get(f"{kp}/{last:06d}/rgb".encode()) is None:
                incomplete.append(t)
    env.close()
    return incomplete


def read_scales_from_lmdb(trajs, lmdb_dir):
    """Read scale values from LMDB for old trajectories that have scale keys."""
    env = lmdb.open(lmdb_dir, readonly=True, lock=False)
    all_scales = {}
    with env.begin() as txn:
        for t in trajs:
            kp = t["key_prefix"]
            traj_scales = {}
            for i in range(t["num_frames"]):
                val = txn.get(f"{kp}/{i:06d}/scale".encode())
                if val is None:
                    break
                traj_scales[i] = float(np.frombuffer(val, dtype=np.float64)[0])
            if traj_scales:
                all_scales[kp] = traj_scales
    env.close()
    return all_scales


def main():
    # Load scan cache
    print("Loading scan cache...")
    with open(SCAN_CACHE) as f:
        trajs = json.load(f)
    print(f"  {len(trajs)} trajectories\n")

    # Step 1: find incomplete
    print("Finding incomplete trajectories...")
    incomplete = find_incomplete_trajs(trajs, LMDB_DIR)
    print(f"  {len(incomplete)} incomplete trajectories\n")

    # Step 2: reprocess incomplete
    if incomplete:
        print("--- Reprocessing incomplete trajectories ---")
        mdb_path = os.path.join(LMDB_DIR, "data.mdb")
        current_size = os.path.getsize(mdb_path)
        map_size = int(current_size * 1.2)
        env = lmdb.open(LMDB_DIR, map_size=map_size)

        new_scales = {}
        t0 = time.time()

        with Pool(16) as pool:
            for i, result in enumerate(pool.imap_unordered(process_trajectory, incomplete)):
                kv_pairs = result["kv_pairs"]
                if kv_pairs:
                    with env.begin(write=True) as txn:
                        for k, v in kv_pairs:
                            txn.put(k.encode(), v)

                new_scales[result["key_prefix"]] = {
                    frame_idx: sc for frame_idx, sc in result["scales"]
                }

                elapsed = time.time() - t0
                print(
                    f"\r  [{i+1}/{len(incomplete)}] {result['key_prefix']:<50s} "
                    f"frames={result['num_processed']}/{result['num_frames']}  "
                    f"{fmt_eta(elapsed)}",
                    end="", flush=True,
                )

        env.close()
        print(f"\n  Done: {len(incomplete)} trajectories in {fmt_eta(time.time() - t0)}\n")
    else:
        new_scales = {}

    # Step 3: read scales from LMDB for old complete trajectories
    print("Reading scales from LMDB for old trajectories...")
    incomplete_set = {t["key_prefix"] for t in incomplete}
    old_complete = [t for t in trajs if t["key_prefix"] not in incomplete_set]
    lmdb_scales = read_scales_from_lmdb(old_complete, LMDB_DIR)
    old_with_scale = {kp for kp, sc in lmdb_scales.items() if sc}
    print(f"  Read scales for {len(old_with_scale)} old trajectories\n")

    # Step 4: merge all scales
    print("Merging scales...")
    cache_path = os.path.join(OUT_DIR, ".scales_cache.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            raw = json.load(f)
        cache_scales = {kp: {int(k): v for k, v in frames.items()} for kp, frames in raw.items()}
    else:
        cache_scales = {}

    all_scales = {}
    all_scales.update(lmdb_scales)
    all_scales.update(cache_scales)
    all_scales.update(new_scales)

    has_scale = sum(1 for kp in trajs if all_scales.get(kp["key_prefix"]))
    print(f"  {has_scale}/{len(trajs)} trajectories have scales\n")

    # Step 5: regenerate index
    print("--- Regenerating index + train/val split ---")
    total_frames = sum(t["num_frames"] for t in trajs)
    build_index(trajs, total_frames, all_scales)

    # Verify scale coverage
    with open(INDEX_PATH) as f:
        idx = json.load(f)
    samples = idx["samples"]
    n_scale = sum(1 for s in samples if "scale" in s)
    print(f"\nScale coverage: {n_scale}/{len(samples)} ({n_scale/len(samples)*100:.1f}%)")

    if n_scale < len(samples):
        missing = [s for s in samples if "scale" not in s]
        print(f"Missing scale samples (first 5):")
        for s in missing[:5]:
            print(f"  {s['key_prefix']}/{s['frame']:06d}")


if __name__ == "__main__":
    main()
