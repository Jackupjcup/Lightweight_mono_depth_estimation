#!/usr/bin/env python3
"""Build pseudo-label LMDB using DA3 anyview branch.

Reads RGB + GT depth + pose from an existing TartanGround LMDB.
Runs DA3 anyview branch to produce relative depth and 6D ray field.
Sky mask is derived from GT depth >= depth_cap (default 100m).
Scale is removed.

Output LMDB keys per frame:
  - rgb   : JPEG q95 bytes (copied from source)
  - depth : zlib float32 (476, 644) — DA3 anyview relative depth
  - pose  : float64[7] bytes (copied from source)
  - sky   : zlib float16 (476, 644) — binary mask from GT depth >= cap
  - ray   : zlib float16 (238, 322, 6) — DA3 anyview 6D ray field

Usage:
    python tools/build_da3_pseudo_lmdb.py --gpu 0
    python tools/build_da3_pseudo_lmdb.py --verify --out-lmdb <path>
"""

import argparse
import json
import os
import sys
import time
import zlib

import cv2
import lmdb
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TGT_H, TGT_W = 476, 644
RAY_H, RAY_W = TGT_H // 2, TGT_W // 2  # 238, 322
ZLIB_LEVEL = 1

SRC_LMDB = "/root/vepfs/lrd_projs/mono_depth_estimation/fast_mono_depth/data/tartanground.lmdb"
SRC_INDEX = "/root/vepfs/lrd_projs/mono_depth_estimation/fast_mono_depth/data/annotations/tartanground_index.json"
DEFAULT_OUT_LMDB = "/root/vepfs/lrd_projs/mono_depth_estimation/fast_mono_depth/data/tartanground_da3pseudo.lmdb"
DEFAULT_OUT_ANNO = "/root/vepfs/lrd_projs/mono_depth_estimation/fast_mono_depth/data/annotations"
DEFAULT_MODEL_DIR = "/data-vepfs/lrd_projs/mono_body_dataset/Depth-Anything-3-main/models/DA3NESTED-GIANT-LARGE-1.1/"
DEFAULT_DA3_SRC = "/data-vepfs/lrd_projs/mono_body_dataset/Depth-Anything-3-main/src"

def fmt_eta(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}min"
    else:
        return f"{seconds / 3600:.1f}h"


# ---------------------------------------------------------------------------
# DA3 model loading + inference helpers
# ---------------------------------------------------------------------------

def load_da3_anyview(model_dir, da3_src, device):
    """Load DA3 nested model and return the anyview branch + input processor."""
    sys.path.insert(0, da3_src)
    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.utils.io.input_processor import InputProcessor

    print(f"Loading DA3 model from {model_dir}...", flush=True)
    model = DepthAnything3.from_pretrained(model_dir).to(device)
    model.eval()
    print("DA3 model loaded.", flush=True)

    anyview = model.model.da3
    input_processor = InputProcessor()

    return model, anyview, input_processor


def run_anyview_inference(anyview, input_processor, images_rgb, process_res, device):
    """Run anyview branch forward, capturing ray before it gets deleted.

    Args:
        anyview: DepthAnything3Net (the anyview sub-model)
        input_processor: DA3 InputProcessor instance
        images_rgb: list of (H, W, 3) uint8 numpy arrays (RGB order)
        process_res: DA3 processing resolution
        device: torch device

    Returns:
        depth_np: (N, H_out, W_out) float32 numpy
        ray_np:   (N, H_out, W_out, 6) float32 numpy
    """
    captured = {}

    def _head_hook(module, inp, out):
        if "ray" in out:
            captured["ray"] = out["ray"].detach().clone()

    hook = anyview.head.register_forward_hook(_head_hook)

    try:
        imgs_cpu, _, _ = input_processor(
            images_rgb, process_res=process_res,
            process_res_method="upper_bound_resize",
            sequential=True, print_progress=False,
        )
        imgs = imgs_cpu.to(device, non_blocking=True)[None].float()  # (1, N, 3, H, W)
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=autocast_dtype):
            output = anyview(imgs)
    finally:
        hook.remove()

    depth = output["depth"].squeeze(0).cpu().float().numpy()  # (N, H, W)

    # ray from hook — shape (1, N, H, W, 6) from DualDPT forward
    ray = captured["ray"].squeeze(0).cpu().float().numpy()
    if ray.ndim == 3:
        ray = ray[np.newaxis]

    return depth, ray


# ---------------------------------------------------------------------------
# LMDB read / write helpers
# ---------------------------------------------------------------------------

def read_rgb_from_lmdb(txn, frame_key):
    """Read and decode RGB from LMDB."""
    rgb_buf = txn.get(f"{frame_key}/rgb".encode())
    if rgb_buf is None:
        return None, None
    img = cv2.imdecode(np.frombuffer(rgb_buf, dtype=np.uint8), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb, rgb_buf


def read_gt_depth_from_lmdb(txn, frame_key):
    """Read GT depth from LMDB (zlib float32)."""
    depth_buf = txn.get(f"{frame_key}/depth".encode())
    if depth_buf is None:
        return None
    depth = np.frombuffer(
        zlib.decompress(depth_buf), dtype=np.float32,
    ).reshape(TGT_H, TGT_W).copy()
    return depth


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build_pseudo_lmdb(args):
    device = torch.device(f"cuda:{args.gpu}")

    # Load model
    model, anyview, input_processor = load_da3_anyview(
        args.model_dir, args.da3_src, device,
    )

    # Load source index
    with open(args.src_index) as f:
        index = json.load(f)
    all_samples = index["samples"]
    total_all = len(all_samples)

    # Open source LMDB
    src_env = lmdb.open(args.src_lmdb, readonly=True, lock=False, readahead=False, map_size=1 << 40)

    # Resume detection
    samples = all_samples
    done_offset = 0
    out_mdb = os.path.join(args.out_lmdb, "data.mdb")
    if os.path.exists(out_mdb) and os.path.getsize(out_mdb) > 0:
        env_ro = lmdb.open(args.out_lmdb, readonly=True, lock=False)
        with env_ro.begin() as txn:
            first_missing = None
            for i, s in enumerate(all_samples):
                key = f"{s['key_prefix']}/{s['frame']:06d}/ray".encode()
                if txn.get(key) is None:
                    first_missing = i
                    break
            if first_missing is None:
                env_ro.close()
                print(f"All {total_all} samples already exist.", flush=True)
                return
            done_offset = first_missing
            samples = all_samples[first_missing:]
        env_ro.close()
        if done_offset > 0:
            print(f"Resume: {done_offset}/{total_all} done, "
                  f"continuing from sample {done_offset}", flush=True)

    total = len(samples)
    print(f"Processing {total} samples (batch={args.batch_size}, gpu={args.gpu})...",
          flush=True)

    # Output LMDB
    os.makedirs(args.out_lmdb, exist_ok=True)
    est_bytes = total_all * 500 * 1024  # ~500KB per frame estimate
    existing_size = os.path.getsize(out_mdb) if os.path.exists(out_mdb) else 0
    map_size = max(int((existing_size + est_bytes) * 1.5), 1 << 30)
    out_env = lmdb.open(args.out_lmdb, map_size=map_size)

    t0 = time.time()
    batch_size = args.batch_size
    depth_cap = args.depth_cap

    for batch_start in range(0, total, batch_size):
        batch = samples[batch_start:batch_start + batch_size]

        # 1) Read from source LMDB
        images_rgb = []
        frame_keys = []
        rgb_bufs = []
        pose_bufs = []
        gt_depths = []

        with src_env.begin() as txn:
            for s in batch:
                frame_key = f"{s['key_prefix']}/{s['frame']:06d}"
                img_rgb, rgb_buf = read_rgb_from_lmdb(txn, frame_key)
                if img_rgb is None:
                    continue
                gt_depth = read_gt_depth_from_lmdb(txn, frame_key)
                if gt_depth is None:
                    continue
                pose_buf = txn.get(f"{frame_key}/pose".encode())
                if pose_buf is None:
                    continue

                images_rgb.append(img_rgb)
                frame_keys.append(frame_key)
                rgb_bufs.append(rgb_buf)
                pose_bufs.append(pose_buf)
                gt_depths.append(gt_depth)

        if not images_rgb:
            continue

        # 2) DA3 anyview inference
        depth_pred, ray_pred = run_anyview_inference(
            anyview, input_processor, images_rgb, args.process_res, device,
        )
        # depth_pred: (N, H_proc, W_proc) float32
        # ray_pred:   (N, H_proc, W_proc, 6) float32

        # 3) Build KV pairs for output LMDB
        kv_pairs = []
        for i, frame_key in enumerate(frame_keys):
            if i >= depth_pred.shape[0]:
                break

            # RGB — copy as-is
            kv_pairs.append((f"{frame_key}/rgb".encode(), rgb_bufs[i]))

            # Depth — resize DA3 prediction to target resolution
            d = depth_pred[i]  # (H_proc, W_proc)
            d_resized = cv2.resize(d, (TGT_W, TGT_H), interpolation=cv2.INTER_LINEAR)
            depth_bytes = zlib.compress(d_resized.astype(np.float32).tobytes(), ZLIB_LEVEL)
            kv_pairs.append((f"{frame_key}/depth".encode(), depth_bytes))

            # Pose — copy as-is
            kv_pairs.append((f"{frame_key}/pose".encode(), pose_bufs[i]))

            # Sky mask — from GT depth >= depth_cap
            sky_mask = (gt_depths[i] >= depth_cap).astype(np.float16)
            sky_bytes = zlib.compress(sky_mask.tobytes(), ZLIB_LEVEL)
            kv_pairs.append((f"{frame_key}/sky".encode(), sky_bytes))

            # Ray — resize DA3 ray field to half resolution
            r = ray_pred[i]  # (H_proc, W_proc, 6)
            # Resize each of the 6 channels separately
            r_resized = np.stack([
                cv2.resize(r[:, :, c], (RAY_W, RAY_H), interpolation=cv2.INTER_LINEAR)
                for c in range(6)
            ], axis=-1)  # (RAY_H, RAY_W, 6)
            ray_bytes = zlib.compress(r_resized.astype(np.float16).tobytes(), ZLIB_LEVEL)
            kv_pairs.append((f"{frame_key}/ray".encode(), ray_bytes))

        # 4) Write to output LMDB
        with out_env.begin(write=True) as txn:
            for k, v in kv_pairs:
                txn.put(k, v)

        # Progress
        done = batch_start + len(batch)
        done_total = done_offset + done
        elapsed = time.time() - t0
        fps = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / fps if fps > 0 else 0
        mdb_size = os.path.getsize(os.path.join(args.out_lmdb, "data.mdb")) / (1 << 30)
        print(
            f"\r  [{done_total}/{total_all}] {fps:.1f}/s  "
            f"size={mdb_size:.1f}GB  ETA {fmt_eta(eta)}   ",
            end="", flush=True,
        )

    out_env.close()
    src_env.close()
    elapsed = time.time() - t0
    print(f"\n\nDone: {total_all} frames in {fmt_eta(elapsed)}")


# ---------------------------------------------------------------------------
# JSON generation
# ---------------------------------------------------------------------------

def build_json(args):
    """Generate new JSON annotations (no scale field)."""
    with open(args.src_index) as f:
        index = json.load(f)

    # Update meta
    meta = dict(index["meta"])
    meta["depth_source"] = "da3_anyview_relative"
    if "scale" in meta:
        del meta["scale"]

    # Remove scale from samples
    new_samples = []
    for s in index["samples"]:
        ns = {k: v for k, v in s.items() if k != "scale"}
        new_samples.append(ns)

    # Write full index
    new_index = {
        "meta": meta,
        "samples": new_samples,
    }
    if "trajectories" in index:
        new_index["trajectories"] = index["trajectories"]

    out_index_path = os.path.join(args.out_anno_dir, "tartanground_da3pseudo_index.json")
    with open(out_index_path, "w") as f:
        json.dump(new_index, f)
    print(f"Index saved: {out_index_path} ({len(new_samples)} samples)")

    # Build train/val splits from existing split files
    for split_name in ["train", "val", "train_tenth", "val_tenth",
                       "train_twentieth", "val_twentieth",
                       "train_quarter", "val_quarter"]:
        src_split = os.path.join(
            os.path.dirname(args.src_index),
            f"tartanground_{split_name}.json",
        )
        if not os.path.exists(src_split):
            continue

        with open(src_split) as f:
            split_data = json.load(f)

        split_out = {
            "meta": meta,
        }
        if "trajectory_indices" in split_data:
            split_out["trajectory_indices"] = split_data["trajectory_indices"]

        split_samples = []
        for s in split_data["samples"]:
            ns = {k: v for k, v in s.items() if k != "scale"}
            split_samples.append(ns)
        split_out["samples"] = split_samples

        out_path = os.path.join(args.out_anno_dir, f"tartanground_da3pseudo_{split_name}.json")
        with open(out_path, "w") as f:
            json.dump(split_out, f)
        print(f"{split_name}: {len(split_samples)} samples → {out_path}")


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

def verify_lmdb(args):
    """Verify output LMDB contents."""
    import random

    # Load JSON
    index_path = os.path.join(args.out_anno_dir, "tartanground_da3pseudo_index.json")
    if not os.path.exists(index_path):
        index_path = args.src_index
    with open(index_path) as f:
        index = json.load(f)

    samples = index["samples"]
    rng = random.Random(42)
    num_check = min(args.verify_num, len(samples))
    check_indices = rng.sample(range(len(samples)), num_check)

    env = lmdb.open(args.out_lmdb, readonly=True, lock=False)
    ok = 0

    with env.begin() as txn:
        for idx in check_indices:
            s = samples[idx]
            frame_key = f"{s['key_prefix']}/{s['frame']:06d}"

            # Check all required keys
            rgb_buf = txn.get(f"{frame_key}/rgb".encode())
            depth_buf = txn.get(f"{frame_key}/depth".encode())
            pose_buf = txn.get(f"{frame_key}/pose".encode())
            sky_buf = txn.get(f"{frame_key}/sky".encode())
            ray_buf = txn.get(f"{frame_key}/ray".encode())

            if any(b is None for b in [rgb_buf, depth_buf, pose_buf, sky_buf, ray_buf]):
                missing = []
                for name, buf in [("rgb", rgb_buf), ("depth", depth_buf),
                                  ("pose", pose_buf), ("sky", sky_buf), ("ray", ray_buf)]:
                    if buf is None:
                        missing.append(name)
                print(f"  FAIL: missing {missing} for {frame_key}")
                continue

            # Decode and verify shapes
            img = cv2.imdecode(np.frombuffer(rgb_buf, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None or img.shape != (TGT_H, TGT_W, 3):
                print(f"  FAIL: bad rgb shape for {frame_key}")
                continue

            depth = np.frombuffer(
                zlib.decompress(depth_buf), dtype=np.float32,
            ).reshape(TGT_H, TGT_W)

            pose = np.frombuffer(pose_buf, dtype=np.float64)
            if pose.shape != (7,):
                print(f"  FAIL: bad pose shape {pose.shape} for {frame_key}")
                continue

            sky = np.frombuffer(
                zlib.decompress(sky_buf), dtype=np.float16,
            ).reshape(TGT_H, TGT_W)

            ray = np.frombuffer(
                zlib.decompress(ray_buf), dtype=np.float16,
            ).reshape(RAY_H, RAY_W, 6)

            # No scale key should exist
            scale_buf = txn.get(f"{frame_key}/scale".encode())
            if scale_buf is not None:
                print(f"  WARN: unexpected scale key for {frame_key}")

            ok += 1

    env.close()

    mdb_path = os.path.join(args.out_lmdb, "data.mdb")
    size_gb = os.path.getsize(mdb_path) / (1 << 30) if os.path.exists(mdb_path) else 0
    print(f"\nVerification: {ok}/{num_check} passed")
    print(f"LMDB size: {size_gb:.1f} GB")
    print(f"Total samples in index: {len(samples)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build DA3 pseudo-label LMDB from existing TartanGround",
    )
    parser.add_argument("--src-lmdb", default=SRC_LMDB)
    parser.add_argument("--src-index", default=SRC_INDEX)
    parser.add_argument("--out-lmdb", default=DEFAULT_OUT_LMDB)
    parser.add_argument("--out-anno-dir", default=DEFAULT_OUT_ANNO)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--da3-src", default=DEFAULT_DA3_SRC)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--depth-cap", type=float, default=100.0,
                        help="GT depth >= cap → sky mask")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--verify-num", type=int, default=100)
    parser.add_argument("--json-only", action="store_true",
                        help="Only generate JSON annotations (skip LMDB build)")
    args = parser.parse_args()

    if args.verify:
        verify_lmdb(args)
        return

    if args.json_only:
        build_json(args)
        return

    build_pseudo_lmdb(args)
    build_json(args)


if __name__ == "__main__":
    main()
