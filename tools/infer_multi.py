"""Fast Mono Depth — Multi-student Inference: compare multiple student checkpoints side-by-side.

Usage example:
cd /data-vepfs/lrd_projs/mono_depth_estimation/fast_mono_depth && \
conda activate fast_mono_depth_lrd && \
PYTHONPATH="$(pwd):$PYTHONPATH" python tools/infer_multi.py \
    --checkpoints work_dirs/v0_v1/20260620_035604/best_delta.pt \
                  work_dirs/v0_tenth_v1/20260618_154428/best_delta.pt \
                  work_dirs/v1_tenth/20260622_062707/best_delta.pt \
    --labels "v0_v1 best" "v0_tenth_v1 best" "v1 best" \
    --index_json data/annotations/tartanground_val_tenth.json \
    --lmdb_path data/tartanground.lmdb \
    --num_samples 10 \
    --output work_dirs/infer_res/infer_multi.png
"""

import argparse
import json
import random
import sys
import os
import zlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import lmdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

from models.fast_depth_model import FastDepthModel

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def parse_args():
    p = argparse.ArgumentParser(description="Fast Mono Depth — Multi-student Inference")
    p.add_argument("--checkpoints", type=str, nargs="+", required=True,
                   help="One or more student checkpoint paths")
    p.add_argument("--labels", type=str, nargs="*", default=None,
                   help="Display labels for each checkpoint (defaults to basename)")
    p.add_argument("--index_json", type=str, required=True)
    p.add_argument("--lmdb_path",  type=str, required=True)
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--output",      type=str, default="infer_multi.png")
    p.add_argument("--input_height", type=int, default=476)
    p.add_argument("--input_width",  type=int, default=644)
    p.add_argument("--depth_cap",   type=float, default=100.0)
    p.add_argument("--device",      type=str, default="cuda:0")
    p.add_argument("--valid_mask",  action="store_true",
                   help="Mask pixels with GT depth >= depth_cap as white in student columns")
    return p.parse_args()


def load_student(checkpoint, device):
    ckpt  = torch.load(checkpoint, map_location=device, weights_only=False)
    model = FastDepthModel(
        backbone_pretrained=False,
        dpt_features=256,
        dpt_out_channels=(256, 512, 1024, 1024),
    )
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    epoch = ckpt.get("epoch", "?")
    step  = ckpt.get("global_step", "?")
    print(f"  Loaded {os.path.basename(checkpoint)}: epoch={epoch}, step={step}")
    return model


def read_sample(env, sample, input_h, input_w, depth_cap):
    key = f"{sample['key_prefix']}/{sample['frame']:06d}"
    with env.begin() as txn:
        rgb_buf   = txn.get(f"{key}/rgb".encode())
        depth_buf = txn.get(f"{key}/depth".encode())

    image = cv2.imdecode(np.frombuffer(rgb_buf, np.uint8), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    depth = np.frombuffer(zlib.decompress(depth_buf), np.float32).reshape(input_h, input_w).copy()
    valid = depth < depth_cap
    np.clip(depth, None, depth_cap, out=depth)
    scale = sample.get("scale", 1.0)

    return image, depth, valid, scale


def main():
    args = parse_args()
    random.seed(args.seed)

    # Resolve labels
    labels = args.labels
    if labels is None:
        labels = [os.path.basename(os.path.dirname(c)) + "/" + os.path.basename(c)
                  for c in args.checkpoints]
    if len(labels) != len(args.checkpoints):
        raise ValueError(f"--labels count ({len(labels)}) must match --checkpoints count ({len(args.checkpoints)})")

    with open(args.index_json) as f:
        index = json.load(f)
    samples = index["samples"]
    n       = min(args.num_samples, len(samples))
    chosen  = random.sample(samples, n)
    print(f"Loaded {len(samples)} samples, randomly picked {n}")

    # Load all student models
    print("Loading student models...")
    students = []
    for ckpt_path in args.checkpoints:
        students.append(load_student(ckpt_path, args.device))
    print(f"All {len(students)} student(s) loaded.")

    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    env = lmdb.open(args.lmdb_path, readonly=True, lock=False, readahead=False, map_size=1 << 40)

    # Layout: RGB | GT | student_0 | student_1 | ...
    num_cols = 2 + len(students)
    fig, axes = plt.subplots(n, num_cols, figsize=(6 * num_cols, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    ckpt_names = ", ".join(os.path.basename(c) for c in args.checkpoints)
    fig.suptitle(f"Multi-student Inference — {ckpt_names}", fontsize=12, y=1.0)

    for row, sample in enumerate(chosen):
        image_np, gt_depth, valid, scale = read_sample(
            env, sample, args.input_height, args.input_width, args.depth_cap,
        )
        gt_depth_norm = gt_depth / scale

        img_t = torch.from_numpy(image_np.transpose(2, 0, 1).copy()).float() / 255.0
        img_t = normalize(img_t).unsqueeze(0).to(args.device)

        # Run all students
        pred_depths = []
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for model in students:
                out = model(img_t)
                pred_depths.append(out.depth[0].float().cpu().numpy())

        label = f"{sample['key_prefix']}/f{sample['frame']:06d}"

        # Col 0: RGB
        axes[row, 0].imshow(image_np)
        axes[row, 0].set_title(f"RGB: {label}", fontsize=8)
        axes[row, 0].axis("off")

        # Col 1: GT depth
        gt_masked = np.ma.masked_where(~valid, gt_depth_norm)
        vmax_gt   = np.percentile(gt_depth_norm[valid], 95) if valid.sum() > 0 else 1.0
        im_gt     = axes[row, 1].imshow(gt_masked, cmap="turbo_r", vmin=0, vmax=vmax_gt)
        axes[row, 1].set_title(f"GT depth (scale={scale:.2f})", fontsize=8)
        axes[row, 1].axis("off")
        plt.colorbar(im_gt, ax=axes[row, 1], fraction=0.03, pad=0.02)

        # Col 2+: each student
        for col_idx, (pred_depth, lbl) in enumerate(zip(pred_depths, labels)):
            col = 2 + col_idx
            if args.valid_mask:
                pred_plot = np.ma.masked_where(~valid, pred_depth)
                vmax_pred = np.percentile(pred_depth[valid], 95) if valid.sum() > 0 else np.percentile(pred_depth, 95)
            else:
                pred_plot = pred_depth
                vmax_pred = np.percentile(pred_depth, 95)

            im_pred = axes[row, col].imshow(pred_plot, cmap="turbo_r", vmin=0, vmax=vmax_pred)
            axes[row, col].set_title(lbl, fontsize=8)
            axes[row, col].axis("off")
            plt.colorbar(im_pred, ax=axes[row, col], fraction=0.03, pad=0.02)

    env.close()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=100, bbox_inches="tight")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
