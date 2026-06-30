#!/usr/bin/env python3
"""Build LMDB from TartanGround for fast training IO.

Reads from TOS, applies crop+pad (640×640 → 644×476), stores:
  - RGB as JPEG q95 bytes
  - Depth as zlib-compressed float32 bytes
  - Pose as float64[7] bytes (tx,ty,tz,qx,qy,qz,qw)
  - Scale factor as float64 bytes

Optional append modes (run after base build):
  --add-ray : compute ray from pose + bearing grid, store as zlib float16
  --add-sky : run DA3 metric branch inference, store sky as zlib float16

Usage:
    python tools/build_tartanground_lmdb.py [--workers 32]
    python tools/build_tartanground_lmdb.py --add-ray
    python tools/build_tartanground_lmdb.py --add-sky --sky-model-dir <path> [--gpu 0]
    python tools/build_tartanground_lmdb.py --verify
"""

import argparse
import glob
import json
import os
import random
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool

import cv2
import lmdb
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import (
    K_CROPPED,
    K_ORIG,
    TGT_H,
    TGT_W,
    compute_scale_factor,
    crop_and_pad,
    decode_depth,
)

SRC_ROOT = "/data-tos-daily/TartanGround"
OUT_DIR = "/root/vepfs/lrd_projs/mono_depth_estimation/fast_mono_depth/data"
LMDB_DIR = os.path.join(OUT_DIR, "tartanground.lmdb")
ANNO_DIR = os.path.join(OUT_DIR, "annotations")
INDEX_PATH = os.path.join(ANNO_DIR, "tartanground_index.json")
TRAIN_PATH = os.path.join(ANNO_DIR, "tartanground_train.json")
VAL_PATH = os.path.join(ANNO_DIR, "tartanground_val.json")
SCAN_CACHE = os.path.join(OUT_DIR, ".scan_cache.json")

JPEG_QUALITY = 95
ZLIB_LEVEL = 1
VAL_RATIO = 0.1

# --- Ray constants ---
H_RAY, W_RAY = TGT_H // 2, TGT_W // 2  # 238, 322

R_NED_CV = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
], dtype=np.float64)


def compute_bearing_grid():
    """Precompute bearing directions at half resolution from K_CROPPED."""
    fx = K_CROPPED[0, 0] / 2
    fy = K_CROPPED[1, 1] / 2
    cx = K_CROPPED[0, 2] / 2
    cy = K_CROPPED[1, 2] / 2

    u = np.arange(W_RAY, dtype=np.float64) + 0.5
    v = np.arange(H_RAY, dtype=np.float64) + 0.5
    uu, vv = np.meshgrid(u, v)

    bx = (uu - cx) / fx
    by = (vv - cy) / fy
    bz = np.ones_like(uu)

    return np.stack([bx, by, bz], axis=-1)  # (H_RAY, W_RAY, 3)


# =========================================================================
# Scan
# =========================================================================

def _scan_one_traj(traj_dir):
    """Scan a single trajectory directory (IO-bound, for threading)."""
    parts = traj_dir.split("/")
    scene, robot, traj = parts[-3], parts[-2], parts[-1]

    img_dir = os.path.join(traj_dir, "image_lcam_front")
    dep_dir = os.path.join(traj_dir, "depth_lcam_front")
    meta_path = os.path.join(traj_dir, f"{traj}_metadata.json")

    if not os.path.isdir(img_dir) or not os.path.isdir(dep_dir):
        return None

    frames = [f for f in os.listdir(img_dir) if f.endswith(".png")]
    if not frames:
        return None

    metadata = {}
    if os.path.isfile(meta_path):
        try:
            with open(meta_path) as f:
                metadata = json.load(f)
        except Exception:
            pass

    return {
        "scene": scene,
        "robot": robot,
        "traj": traj,
        "traj_dir": traj_dir,
        "num_frames": len(frames),
        "key_prefix": f"{scene}/{robot}/{traj}",
        "metadata": {
            "robot_height": metadata.get("robot_height"),
            "path_length": metadata.get("path_length"),
            "num_poses": metadata.get("num_poses"),
        },
    }


def scan_trajectories(src_root):
    """Find all trajectories via glob, scan in parallel threads."""
    traj_dirs = sorted(glob.glob(os.path.join(src_root, "*/Data_*/P*")))
    print(f"  Found {len(traj_dirs)} candidate directories, scanning frames...",
          flush=True)

    trajs = []
    with ThreadPoolExecutor(max_workers=64) as pool:
        futures = {pool.submit(_scan_one_traj, d): d for d in traj_dirs}
        for i, fut in enumerate(as_completed(futures)):
            result = fut.result()
            if result is not None:
                trajs.append(result)
            if (i + 1) % 100 == 0 or (i + 1) == len(traj_dirs):
                print(f"\r  Scanned {i+1}/{len(traj_dirs)} dirs, "
                      f"{len(trajs)} valid trajectories",
                      end="", flush=True)
    print(flush=True)
    trajs.sort(key=lambda t: t["key_prefix"])
    return trajs


# =========================================================================
# Base build
# =========================================================================

def process_trajectory(traj_info):
    """Process all frames in one trajectory, return list of (key, value) pairs."""
    traj_dir = traj_info["traj_dir"]
    key_prefix = traj_info["key_prefix"]
    num_frames = traj_info["num_frames"]

    img_dir = os.path.join(traj_dir, "image_lcam_front")
    dep_dir = os.path.join(traj_dir, "depth_lcam_front")
    pose_path = os.path.join(traj_dir, "pose_lcam_front.txt")

    poses = None
    if os.path.isfile(pose_path):
        poses = np.loadtxt(pose_path, dtype=np.float64)

    kv_pairs = []
    scales = []
    errors = []

    img_files = sorted(f for f in os.listdir(img_dir) if f.endswith(".png"))

    for i, img_file in enumerate(img_files):
        frame_id = f"{i:06d}"
        frame_key = f"{key_prefix}/{frame_id}"

        try:
            img_path = os.path.join(img_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                errors.append(f"unreadable image: {img_path}")
                continue
            img_cropped = crop_and_pad(img)
            _, jpg_buf = cv2.imencode(
                ".jpg", img_cropped, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )
            rgb_bytes = jpg_buf.tobytes()
        except Exception as e:
            errors.append(f"image error {frame_key}: {e}")
            continue

        dep_file = f"{frame_id}_lcam_front_depth.png"
        dep_path = os.path.join(dep_dir, dep_file)
        try:
            depth_f32 = decode_depth(dep_path)
            if depth_f32 is None:
                errors.append(f"unreadable depth: {dep_path}")
                continue
            depth_cropped = crop_and_pad(depth_f32)
            scale = compute_scale_factor(depth_cropped, K_CROPPED)
            depth_bytes = zlib.compress(depth_cropped.tobytes(), ZLIB_LEVEL)
        except Exception as e:
            errors.append(f"depth error {frame_key}: {e}")
            continue

        if poses is not None and i < len(poses):
            pose_bytes = poses[i].tobytes()
        else:
            pose_bytes = np.zeros(7, dtype=np.float64).tobytes()

        kv_pairs.append((f"{frame_key}/rgb", rgb_bytes))
        kv_pairs.append((f"{frame_key}/depth", depth_bytes))
        kv_pairs.append((f"{frame_key}/pose", pose_bytes))
        scales.append((i, scale))

    return {
        "key_prefix": key_prefix,
        "num_processed": len(kv_pairs) // 3,
        "num_frames": num_frames,
        "errors": errors,
        "kv_pairs": kv_pairs,
        "scales": scales,
    }


def fmt_eta(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}min"
    else:
        return f"{seconds / 3600:.1f}h"


def _load_scales_cache():
    """Load incremental scales cache from previous run."""
    path = os.path.join(OUT_DIR, ".scales_cache.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        raw = json.load(f)
    return {kp: {int(k): v for k, v in frames.items()} for kp, frames in raw.items()}


def _save_scales_cache(all_scales):
    """Save scales cache (overwrites)."""
    path = os.path.join(OUT_DIR, ".scales_cache.json")
    with open(path, "w") as f:
        json.dump(all_scales, f)


def _find_completed_trajs(trajs, output_dir):
    """Check LMDB for trajectories whose first frame already exists."""
    mdb_path = os.path.join(output_dir, "data.mdb")
    if not os.path.exists(mdb_path) or os.path.getsize(mdb_path) == 0:
        return set()
    completed = set()
    env = lmdb.open(output_dir, readonly=True, lock=False)
    with env.begin() as txn:
        for t in trajs:
            if txn.get(f"{t['key_prefix']}/000000/rgb".encode()) is not None:
                completed.add(t["key_prefix"])
    env.close()
    return completed


def build_lmdb(trajs, output_dir, workers):
    """Build LMDB with automatic resume — skips already-completed trajectories."""
    os.makedirs(output_dir, exist_ok=True)

    # --- Resume detection ---
    all_scales = _load_scales_cache()
    completed = _find_completed_trajs(trajs, output_dir)
    remaining = [t for t in trajs if t["key_prefix"] not in completed]
    done_frames = sum(t["num_frames"] for t in trajs if t["key_prefix"] in completed)
    total_all_frames = sum(t["num_frames"] for t in trajs)

    if completed:
        print(f"Resume: {len(completed)} trajectories ({done_frames} frames) already done, "
              f"{len(remaining)} remaining", flush=True)

    if not remaining:
        print("All trajectories complete.", flush=True)
        return total_all_frames, all_scales

    est_remaining = sum(t["num_frames"] for t in remaining)
    est_bytes = total_all_frames * 320 * 1024
    map_size = max(est_bytes * 2, 1 << 30)
    print(f"LMDB map_size: {map_size / (1 << 40):.1f} TB")

    env = lmdb.open(output_dir, map_size=map_size)

    num_remaining = len(remaining)
    processed_frames = 0
    total_errors = []
    t0 = time.time()

    with Pool(workers) as pool:
        for i, result in enumerate(pool.imap_unordered(process_trajectory, remaining)):
            kv_pairs = result["kv_pairs"]
            if kv_pairs:
                with env.begin(write=True) as txn:
                    for k, v in kv_pairs:
                        txn.put(k.encode(), v)

            all_scales[result["key_prefix"]] = {
                frame_idx: sc for frame_idx, sc in result["scales"]
            }
            _save_scales_cache(all_scales)

            processed_frames += result["num_processed"]
            total_errors.extend(result["errors"])

            total_done = done_frames + processed_frames
            elapsed = time.time() - t0
            fps = processed_frames / elapsed if elapsed > 0 else 0
            eta = (est_remaining - processed_frames) / fps if fps > 0 else 0
            lmdb_size_gb = 0
            mdb_path = os.path.join(output_dir, "data.mdb")
            if os.path.exists(mdb_path):
                lmdb_size_gb = os.path.getsize(mdb_path) / (1 << 30)
            print(
                f"\r  [{i+1}/{num_remaining}] {result['key_prefix']:<50s} "
                f"frames={total_done}/{total_all_frames} "
                f"{fps:.0f}fps  "
                f"size={lmdb_size_gb:.1f}GB  "
                f"ETA {fmt_eta(eta)}   ",
                end="",
                flush=True,
            )

    env.close()
    elapsed = time.time() - t0
    total_done = done_frames + processed_frames
    print(f"\n\nDone: {total_done} frames in {fmt_eta(elapsed)}")

    if total_errors:
        print(f"\n{len(total_errors)} errors:")
        for e in total_errors[:20]:
            print(f"  {e}")
        if len(total_errors) > 20:
            print(f"  ... and {len(total_errors) - 20} more")

    return total_done, all_scales


# =========================================================================
# Index + split
# =========================================================================

def build_index(trajs, total_frames_written, all_scales=None):
    """Build JSON index and train/val splits (stratified by scene).

    Args:
        all_scales: dict {key_prefix: {frame_idx: scale_float}} from build_lmdb.
                    Scale is stored per-sample in JSON (not in LMDB) so it can
                    be recomputed without rebuilding the database.
    """
    if all_scales is None:
        all_scales = {}

    samples = []
    for traj_idx, t in enumerate(trajs):
        traj_scales = all_scales.get(t["key_prefix"], {})
        for frame in range(t["num_frames"]):
            sample = {
                "key_prefix": t["key_prefix"],
                "frame": frame,
                "traj_idx": traj_idx,
            }
            if frame in traj_scales:
                sample["scale"] = round(traj_scales[frame], 6)
            samples.append(sample)

    traj_entries = []
    for t in trajs:
        entry = dict(t)
        entry.pop("traj_dir", None)
        traj_entries.append(entry)

    index = {
        "meta": {
            "total_frames": len(samples),
            "total_trajectories": len(trajs),
            "K_cropped": K_CROPPED.tolist(),
            "K_orig": K_ORIG.tolist(),
            "spatial_transform": "center_crop_640to476 + reflect_pad_640to644",
            "target_size": [TGT_H, TGT_W],
            "rgb_encoding": f"jpeg_q{JPEG_QUALITY}",
            "depth_encoding": f"zlib_level{ZLIB_LEVEL}_float32",
            "pose_format": "tx,ty,tz,qx,qy,qz,qw (float64)",
        },
        "trajectories": traj_entries,
        "samples": samples,
    }

    with open(INDEX_PATH, "w") as f:
        json.dump(index, f)
    print(f"Index saved: {INDEX_PATH} ({len(samples)} samples)")

    # Stratified train/val split by scene
    scene_trajs = {}
    for traj_idx, t in enumerate(trajs):
        scene_trajs.setdefault(t["scene"], []).append(traj_idx)

    val_traj_set = set()
    rng = random.Random(42)
    for scene, traj_indices in sorted(scene_trajs.items()):
        n_val = max(1, round(len(traj_indices) * VAL_RATIO))
        rng.shuffle(traj_indices)
        for idx in traj_indices[:n_val]:
            val_traj_set.add(idx)

    train_samples = [s for s in samples if s["traj_idx"] not in val_traj_set]
    val_samples = [s for s in samples if s["traj_idx"] in val_traj_set]

    train_trajs = [i for i in range(len(trajs)) if i not in val_traj_set]
    val_trajs = sorted(val_traj_set)

    for path, split_samples, split_trajs, name in [
        (TRAIN_PATH, train_samples, train_trajs, "train"),
        (VAL_PATH, val_samples, val_trajs, "val"),
    ]:
        split_data = {
            "meta": index["meta"],
            "trajectory_indices": split_trajs,
            "samples": split_samples,
        }
        with open(path, "w") as f:
            json.dump(split_data, f)

        n_scenes = len(set(trajs[i]["scene"] for i in split_trajs))
        print(
            f"{name}: {len(split_samples)} samples, "
            f"{len(split_trajs)} trajectories, "
            f"{n_scenes} scenes → {path}"
        )


# =========================================================================
# Verify
# =========================================================================

def verify_lmdb(lmdb_dir, index_path, num_samples=100):
    """Verify LMDB contents against source data."""
    with open(index_path) as f:
        index = json.load(f)

    samples = index["samples"]
    rng = random.Random(42)
    check_indices = rng.sample(range(len(samples)), min(num_samples, len(samples)))

    env = lmdb.open(lmdb_dir, readonly=True, lock=False)
    ok = 0
    has_ray = False
    has_sky = False

    with env.begin() as txn:
        for idx in check_indices:
            s = samples[idx]
            prefix = s["key_prefix"]
            frame = s["frame"]
            frame_key = f"{prefix}/{frame:06d}"

            rgb_buf = txn.get(f"{frame_key}/rgb".encode())
            depth_buf = txn.get(f"{frame_key}/depth".encode())
            pose_buf = txn.get(f"{frame_key}/pose".encode())

            if any(b is None for b in [rgb_buf, depth_buf, pose_buf]):
                print(f"  FAIL: missing key(s) for {frame_key}")
                continue

            img = cv2.imdecode(
                np.frombuffer(rgb_buf, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if img is None or img.shape != (TGT_H, TGT_W, 3):
                print(f"  FAIL: bad image shape for {frame_key}: {img.shape if img is not None else 'None'}")
                continue

            depth = np.frombuffer(
                zlib.decompress(depth_buf), dtype=np.float32
            ).reshape(TGT_H, TGT_W)
            if depth.shape != (TGT_H, TGT_W):
                print(f"  FAIL: bad depth shape for {frame_key}")
                continue

            pose = np.frombuffer(pose_buf, dtype=np.float64)
            if pose.shape != (7,):
                print(f"  FAIL: bad pose shape for {frame_key}: {pose.shape}")
                continue

            if "scale" not in s:
                print(f"  WARN: no scale in JSON for {frame_key}")


            # Check optional fields
            ray_buf = txn.get(f"{frame_key}/ray".encode())
            if ray_buf is not None:
                has_ray = True
                ray = np.frombuffer(
                    zlib.decompress(ray_buf), dtype=np.float16
                ).reshape(H_RAY, W_RAY, 6)
                if ray.shape != (H_RAY, W_RAY, 6):
                    print(f"  FAIL: bad ray shape for {frame_key}")
                    continue

            sky_buf = txn.get(f"{frame_key}/sky".encode())
            if sky_buf is not None:
                has_sky = True
                sky = np.frombuffer(
                    zlib.decompress(sky_buf), dtype=np.float16
                ).reshape(TGT_H, TGT_W)
                if sky.shape != (TGT_H, TGT_W):
                    print(f"  FAIL: bad sky shape for {frame_key}")
                    continue

            ok += 1

    env.close()

    mdb_path = os.path.join(lmdb_dir, "data.mdb")
    size_gb = os.path.getsize(mdb_path) / (1 << 30) if os.path.exists(mdb_path) else 0
    print(f"\nVerification: {ok}/{len(check_indices)} passed")
    print(f"LMDB size: {size_gb:.1f} GB")
    print(f"Total samples in index: {len(samples)}")
    print(f"Optional fields: ray={'yes' if has_ray else 'no'}, sky={'yes' if has_sky else 'no'}")


# =========================================================================
# --add-ray: append ray field from pose
# =========================================================================

def _lmdb_map_size_for_append(lmdb_dir, extra_bytes_per_frame, num_frames):
    """Compute map_size large enough for existing data + appended fields."""
    mdb_path = os.path.join(lmdb_dir, "data.mdb")
    current = os.path.getsize(mdb_path) if os.path.exists(mdb_path) else 0
    extra = extra_bytes_per_frame * num_frames
    return int((current + extra) * 1.5)


def add_ray_to_lmdb(lmdb_dir, index_path):
    """Append ray field to existing LMDB, computed from stored poses.

    Ray = concat(R_w2c_cv @ bearing_grid, T_w2c_cv) → (238, 322, 6) float16.
    Coordinate conversion: pose is c2w in NED, converted to w2c in OpenCV via R_NED_CV.
    """
    from scipy.spatial.transform import Rotation

    with open(index_path) as f:
        index = json.load(f)
    all_samples = index["samples"]
    total_all = len(all_samples)

    # --- Resume: skip samples that already have ray ---
    samples = all_samples
    done_offset = 0
    mdb_path = os.path.join(lmdb_dir, "data.mdb")
    if os.path.exists(mdb_path) and os.path.getsize(mdb_path) > 0:
        env_ro = lmdb.open(lmdb_dir, readonly=True, lock=False)
        with env_ro.begin() as txn:
            first_missing = None
            for i, s in enumerate(all_samples):
                key = f"{s['key_prefix']}/{s['frame']:06d}/ray".encode()
                if txn.get(key) is None:
                    first_missing = i
                    break
            if first_missing is None:
                env_ro.close()
                print(f"All {total_all} ray keys already exist.", flush=True)
                return
            done_offset = first_missing
            samples = all_samples[first_missing:]
        env_ro.close()
        if done_offset > 0:
            print(f"Resume: {done_offset}/{total_all} ray keys exist, "
                  f"continuing from sample {done_offset}", flush=True)

    total = len(samples)
    print(f"Adding ray to {total} samples...", flush=True)

    bearing = compute_bearing_grid()
    bearing_flat = bearing.reshape(-1, 3).T  # (3, H_RAY*W_RAY)

    map_size = _lmdb_map_size_for_append(lmdb_dir, 280 * 1024, total_all)
    env = lmdb.open(lmdb_dir, map_size=map_size)

    BATCH = 5000
    t0 = time.time()

    for batch_start in range(0, total, BATCH):
        batch = samples[batch_start:batch_start + BATCH]
        kv_pairs = []

        with env.begin() as txn:
            for s in batch:
                frame_key = f"{s['key_prefix']}/{s['frame']:06d}"
                pose_buf = txn.get(f"{frame_key}/pose".encode())
                if pose_buf is None:
                    continue
                pose = np.frombuffer(pose_buf, dtype=np.float64)

                t_c2w_ned = pose[:3]
                quat = pose[3:]  # qx, qy, qz, qw — matches scipy convention
                R_c2w_ned = Rotation.from_quat(quat).as_matrix()

                R_c2w_cv = R_c2w_ned @ R_NED_CV
                R_w2c_cv = R_c2w_cv.T
                T_w2c_cv = -R_w2c_cv @ t_c2w_ned

                ray_dir = (R_w2c_cv @ bearing_flat).T.reshape(H_RAY, W_RAY, 3)
                t_tile = np.broadcast_to(
                    T_w2c_cv.reshape(1, 1, 3), (H_RAY, W_RAY, 3)
                ).copy()
                ray = np.concatenate([ray_dir, t_tile], axis=-1).astype(np.float16)

                ray_bytes = zlib.compress(ray.tobytes(), ZLIB_LEVEL)
                kv_pairs.append((f"{frame_key}/ray".encode(), ray_bytes))

        with env.begin(write=True) as txn:
            for k, v in kv_pairs:
                txn.put(k, v)

        done = batch_start + len(batch)
        done_total = done_offset + done
        elapsed = time.time() - t0
        fps = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / fps if fps > 0 else 0
        mdb_size = os.path.getsize(os.path.join(lmdb_dir, "data.mdb")) / (1 << 30)
        print(
            f"\r  ray: {done_total}/{total_all} ({fps:.0f}/s, "
            f"size={mdb_size:.1f}GB, ETA {fmt_eta(eta)})   ",
            end="", flush=True,
        )

    env.close()
    print(f"\nRay done: {total_all} frames in {fmt_eta(time.time() - t0)}")


# =========================================================================
# --add-sky: append sky field via DA3 metric branch
# =========================================================================

def add_sky_to_lmdb(lmdb_dir, index_path, da3_src, model_dir, batch_size, gpu):
    """Append sky field to existing LMDB using DA3 nested model's metric branch.

    Loads the nested model and captures sky output from the da3_metric
    sub-module via a forward hook (same approach as visualize_crop.py).
    Stores raw sky confidence (relu-activated, before thresholding) as
    zlib-compressed float16 (476, 644).
    """
    import torch

    sys.path.insert(0, da3_src)
    from depth_anything_3.api import DepthAnything3

    device = f"cuda:{gpu}"
    print(f"Loading DA3 nested model from {model_dir}...", flush=True)
    model = DepthAnything3.from_pretrained(model_dir).to(device)
    model.eval()
    print("DA3 model loaded.", flush=True)

    with open(index_path) as f:
        index = json.load(f)
    all_samples = index["samples"]
    total_all = len(all_samples)

    # --- Resume: skip samples that already have sky ---
    samples = all_samples
    done_offset = 0
    mdb_path = os.path.join(lmdb_dir, "data.mdb")
    if os.path.exists(mdb_path) and os.path.getsize(mdb_path) > 0:
        env_ro = lmdb.open(lmdb_dir, readonly=True, lock=False)
        with env_ro.begin() as txn:
            first_missing = None
            for i, s in enumerate(all_samples):
                key = f"{s['key_prefix']}/{s['frame']:06d}/sky".encode()
                if txn.get(key) is None:
                    first_missing = i
                    break
            if first_missing is None:
                env_ro.close()
                print(f"All {total_all} sky keys already exist.", flush=True)
                return
            done_offset = first_missing
            samples = all_samples[first_missing:]
        env_ro.close()
        if done_offset > 0:
            print(f"Resume: {done_offset}/{total_all} sky keys exist, "
                  f"continuing from sample {done_offset}", flush=True)

    total = len(samples)
    print(f"Adding sky to {total} samples (batch={batch_size}, gpu={gpu})...",
          flush=True)

    map_size = _lmdb_map_size_for_append(lmdb_dir, 110 * 1024, total_all)
    env = lmdb.open(lmdb_dir, map_size=map_size)

    t0 = time.time()
    captured = {}

    def _sky_hook(module, inp, out):
        if "sky" in out:
            captured["sky"] = out["sky"].detach().cpu()

    hook_handle = model.model.da3_metric.register_forward_hook(_sky_hook)

    try:
        for batch_start in range(0, total, batch_size):
            batch = samples[batch_start:batch_start + batch_size]
            captured.clear()

            # Read RGB from LMDB and decode
            images = []
            frame_keys = []
            with env.begin() as txn:
                for s in batch:
                    frame_key = f"{s['key_prefix']}/{s['frame']:06d}"
                    rgb_buf = txn.get(f"{frame_key}/rgb".encode())
                    if rgb_buf is None:
                        continue
                    img = cv2.imdecode(
                        np.frombuffer(rgb_buf, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    frame_keys.append(frame_key)

            if not images:
                continue

            # Run inference — hook captures sky from metric branch
            with torch.no_grad():
                model.inference(image=images)

            if "sky" not in captured:
                continue

            sky_np = captured["sky"].float().numpy()
            while sky_np.ndim > 3:
                sky_np = sky_np.squeeze(0)
            if sky_np.ndim == 2:
                sky_np = sky_np[np.newaxis]

            # Write to LMDB
            kv_pairs = []
            for i, frame_key in enumerate(frame_keys):
                if i >= sky_np.shape[0]:
                    break
                sky_frame = cv2.resize(
                    sky_np[i], (TGT_W, TGT_H), interpolation=cv2.INTER_LINEAR
                )
                sky_bytes = zlib.compress(
                    sky_frame.astype(np.float16).tobytes(), ZLIB_LEVEL
                )
                kv_pairs.append((f"{frame_key}/sky".encode(), sky_bytes))

            with env.begin(write=True) as txn:
                for k, v in kv_pairs:
                    txn.put(k, v)

            done = batch_start + len(batch)
            done_total = done_offset + done
            elapsed = time.time() - t0
            fps = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / fps if fps > 0 else 0
            mdb_size = os.path.getsize(os.path.join(lmdb_dir, "data.mdb")) / (1 << 30)
            print(
                f"\r  sky: {done_total}/{total_all} ({fps:.1f}/s, "
                f"size={mdb_size:.1f}GB, ETA {fmt_eta(eta)})   ",
                end="", flush=True,
            )
    finally:
        hook_handle.remove()

    env.close()
    print(f"\nSky done: {total_all} frames in {fmt_eta(time.time() - t0)}")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build TartanGround LMDB for training"
    )
    parser.add_argument("--src", default=SRC_ROOT)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--verify", action="store_true")

    # Append modes
    parser.add_argument(
        "--add-ray", action="store_true",
        help="Append ray field (from pose + bearing grid) to existing LMDB",
    )
    parser.add_argument(
        "--add-sky", action="store_true",
        help="Append sky field (DA3 metric branch inference) to existing LMDB",
    )
    parser.add_argument(
        "--sky-model-dir", type=str,
        default="/data-vepfs/lrd_projs/mono_depth_estimation/da3/models/DA3NESTED-GIANT-LARGE-1.1",
        help="Path to DA3 nested model directory (for --add-sky)",
    )
    parser.add_argument(
        "--da3-src", type=str,
        default="/data-vepfs/lrd_projs/mono_depth_estimation/da3/src",
        help="Path to DA3 source directory (for --add-sky)",
    )
    parser.add_argument("--sky-batch-size", type=int, default=16)
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()

    if args.verify:
        verify_lmdb(LMDB_DIR, INDEX_PATH)
        return

    if args.add_ray:
        add_ray_to_lmdb(LMDB_DIR, INDEX_PATH)
        return

    if args.add_sky:
        add_sky_to_lmdb(
            LMDB_DIR, INDEX_PATH,
            da3_src=args.da3_src,
            model_dir=args.sky_model_dir,
            batch_size=args.sky_batch_size,
            gpu=args.gpu,
        )
        return

    # --- Full base build ---
    print(f"Source: {args.src}")
    print(f"Output: {LMDB_DIR}")
    print(f"Workers: {args.workers}")

    if os.path.isfile(SCAN_CACHE):
        print(f"Loading scan cache from {SCAN_CACHE}...")
        with open(SCAN_CACHE) as f:
            trajs = json.load(f)
        print(f"Loaded {len(trajs)} trajectories from cache\n")
    else:
        print("Scanning trajectories...")
        sys.stdout.flush()
        trajs = scan_trajectories(args.src)
        with open(SCAN_CACHE, "w") as f:
            json.dump(trajs, f)
        print(f"Scan cached to {SCAN_CACHE}\n")

    total_frames = sum(t["num_frames"] for t in trajs)
    print(f"Found {len(trajs)} trajectories, ~{total_frames} frames\n")
    sys.stdout.flush()

    print("--- Building LMDB ---")
    sys.stdout.flush()
    written, all_scales = build_lmdb(trajs, LMDB_DIR, args.workers)

    print("\n--- Building index + train/val split ---")
    sys.stdout.flush()
    build_index(trajs, written, all_scales)

    print("\n--- Verifying ---")
    sys.stdout.flush()
    verify_lmdb(LMDB_DIR, INDEX_PATH, num_samples=50)


if __name__ == "__main__":
    main()
