"""Batch convert TartanGround image_lcam_front & depth_lcam_front to 644×476.

Uses crop_and_pad from preprocess.py:
  - Vertical center crop: 640 → 476
  - Horizontal reflect pad: 640 → 644

Output is saved alongside the source folder with a _644x476 suffix:
  .../Pxxxx/image_lcam_front_644x476/
  .../Pxxxx/depth_lcam_front_644x476/
"""

import argparse
import glob
import os
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import crop_and_pad, decode_depth


def process_image(args):
    src_path, dst_path = args
    try:
        img = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if img is None:
            return f"SKIP (unreadable): {src_path}"
        out = crop_and_pad(img)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        cv2.imwrite(dst_path, out)
        return None
    except Exception as e:
        return f"ERR {src_path}: {e}"


def process_depth(args):
    src_path, dst_path = args
    try:
        depth_f32 = decode_depth(src_path)
        if depth_f32 is None:
            return f"SKIP (unreadable): {src_path}"
        out = crop_and_pad(depth_f32)
        # re-encode float32 → BGRA uint8 PNG (same format as source)
        bgra = out.view(np.uint8).reshape(out.shape[0], out.shape[1], 4)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        cv2.imwrite(dst_path, bgra)
        return None
    except Exception as e:
        return f"ERR {src_path}: {e}"


def collect_tasks(src_root):
    """Use glob to find image/depth dirs directly — much faster than os.walk."""
    image_tasks = []
    depth_tasks = []

    # Structure: TartanGround/Scene/Data_*/P*/image_lcam_front/*.png
    for img_dir in sorted(glob.glob(os.path.join(src_root, "*/*/P*/image_lcam_front"))):
        dst_dir = img_dir + "_644x476"
        for f in sorted(os.listdir(img_dir)):
            if f.endswith(".png"):
                image_tasks.append((os.path.join(img_dir, f), os.path.join(dst_dir, f)))

    for dep_dir in sorted(glob.glob(os.path.join(src_root, "*/*/P*/depth_lcam_front"))):
        dst_dir = dep_dir + "_644x476"
        for f in sorted(os.listdir(dep_dir)):
            if f.endswith(".png"):
                depth_tasks.append((os.path.join(dep_dir, f), os.path.join(dst_dir, f)))

    return image_tasks, depth_tasks


def main():
    parser = argparse.ArgumentParser(description="Batch crop+pad TartanGround to 644×476")
    parser.add_argument("--src", default="/data-tos-daily/TartanGround",
                        help="Source TartanGround root")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: cpu_count)")
    args = parser.parse_args()

    src_root = args.src
    workers = args.workers or min(cpu_count(), 32)

    print(f"Source:  {src_root}")
    print("Output:  <parent_dir>/{image,depth}_lcam_front_644x476/")
    print(f"Workers: {workers}")
    print("Scanning files...")

    image_tasks, depth_tasks = collect_tasks(src_root)
    print(f"Found {len(image_tasks)} images, {len(depth_tasks)} depth maps")

    if not image_tasks and not depth_tasks:
        print("Nothing to do.")
        return

    with Pool(workers) as pool:
        print(f"\n--- Processing {len(image_tasks)} images ---")
        errors = []
        for i, err in enumerate(pool.imap_unordered(process_image, image_tasks, chunksize=64)):
            if err:
                errors.append(err)
            if (i + 1) % 5000 == 0 or (i + 1) == len(image_tasks):
                print(f"  images: {i + 1}/{len(image_tasks)}")

        print(f"\n--- Processing {len(depth_tasks)} depth maps ---")
        for i, err in enumerate(pool.imap_unordered(process_depth, depth_tasks, chunksize=64)):
            if err:
                errors.append(err)
            if (i + 1) % 5000 == 0 or (i + 1) == len(depth_tasks):
                print(f"  depth:  {i + 1}/{len(depth_tasks)}")

    if errors:
        print(f"\n{len(errors)} errors:")
        for e in errors[:20]:
            print(f"  {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
    else:
        print("\nDone — no errors.")


if __name__ == "__main__":
    main()
