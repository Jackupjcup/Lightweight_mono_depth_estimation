"""Fast Mono Depth — Inference: sample images from LMDB, compare GT / DA3 teacher / student.

cd /data-vepfs/lrd_projs/mono_depth_estimation/fast_mono_depth && conda activate fast_mono_depth_lrd && PYTHONPATH="$(pwd):$PYTHONPATH" python tools/infer.py \
    --checkpoint work_dirs/v0_tenth/20260617_113745/ckpt_epoch0001.pt \
    --index_json data/annotations/tartanground_val_tenth.json \
    --lmdb_path data/tartanground.lmdb \
    --num_samples 10 \
    --output work_dirs/infer_res/infer_epoch1_4col.png

cd /data-vepfs/lrd_projs/mono_depth_estimation/fast_mono_depth && conda activate fast_mono_depth_lrd && PYTHONPATH="$(pwd):$PYTHONPATH" python tools/infer.py \
    --checkpoint work_dirs/v0_tenth_v1/20260618_154428/best_delta.pt \
    --index_json data/annotations/tartanground_val_tenth.json \
    --lmdb_path data/tartanground.lmdb \
    --num_samples 10 \
    --valid_mask \
    --output work_dirs/infer_res/infer_epoch0_softplus_best_mask.png

cd /data-vepfs/lrd_projs/mono_depth_estimation/fast_mono_depth && conda activate fast_mono_depth_lrd && PYTHONPATH="$(pwd):$PYTHONPATH" python tools/infer.py \
    --checkpoint work_dirs/v0_distill/20260618_092606/ckpt_epoch0000.pt \
    --index_json data/annotations/tartanground_val_tenth.json \
    --lmdb_path data/tartanground.lmdb \
    --num_samples 10 \
    --output work_dirs/infer_res/infer_distill_4col.png

cd /data-vepfs/lrd_projs/mono_depth_estimation/fast_mono_depth && conda activate fast_mono_depth_lrd && PYTHONPATH="$(pwd):$PYTHONPATH" python tools/infer.py \
    --checkpoint /data-vepfs/lrd_projs/mono_depth_estimation/fast_mono_depth/work_dirs/v1_tenth/20260622_062707/best_delta.pt \
    --index_json data/annotations/tartanground_val_tenth.json \
    --lmdb_path data/tartanground.lmdb \
    --num_samples 10 \
    --output work_dirs/infer_res/infer_v1_tenth_seed351.png

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
from distillation.teacher import TeacherDA3

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args():
    p = argparse.ArgumentParser(description="Fast Mono Depth — Inference")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--index_json", type=str, required=True)
    p.add_argument("--lmdb_path", type=str, required=True)
    p.add_argument("--teacher_model_dir", type=str,
                    default="/data-vepfs/lrd_projs/mono_body_dataset/Depth-Anything-3-main/models/DA3NESTED-GIANT-LARGE-1.1/")
    p.add_argument("--da3_src", type=str,
                    default="/data-vepfs/lrd_projs/mono_body_dataset/Depth-Anything-3-main/src")
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--seed", type=int, default=351)
    p.add_argument("--output", type=str, default="infer_results.png")
    p.add_argument("--input_height", type=int, default=476)
    p.add_argument("--input_width", type=int, default=644)
    p.add_argument("--depth_cap", type=float, default=100.0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--valid_mask", action="store_true",
                   help="Mask pixels with GT depth >= depth_cap as white in DA3 and Student columns")
    return p.parse_args()


def load_student(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = FastDepthModel(
        backbone_pretrained=False,
        dpt_features=256,
        dpt_out_channels=(256, 512, 1024, 1024),
    )
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    epoch = ckpt.get("epoch", "?")
    step = ckpt.get("global_step", "?")
    print(f"Student loaded: epoch={epoch}, step={step}")
    return model


def read_sample(env, sample, input_h, input_w, depth_cap):
    key = f"{sample['key_prefix']}/{sample['frame']:06d}"
    with env.begin() as txn:
        rgb_buf = txn.get(f"{key}/rgb".encode())
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

    with open(args.index_json) as f:
        index = json.load(f)
    samples = index["samples"]
    n = min(args.num_samples, len(samples))
    chosen = random.sample(samples, n)
    print(f"Loaded {len(samples)} samples, randomly picked {n}")

    # Load models
    student = load_student(args.checkpoint, args.device)

    print("Loading DA3 teacher...")
    teacher = TeacherDA3(
        model_dir=args.teacher_model_dir,
        da3_src=args.da3_src,
        device=args.device,
    )
    print("Teacher loaded.")

    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    env = lmdb.open(args.lmdb_path, readonly=True, lock=False, readahead=False, map_size=1 << 40)

    fig, axes = plt.subplots(n, 4, figsize=(24, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(f"Inference — {os.path.basename(args.checkpoint)}", fontsize=14, y=1.0)

    for row, sample in enumerate(chosen):
        image_np, gt_depth, valid, scale = read_sample(
            env, sample, args.input_height, args.input_width, args.depth_cap,
        )
        gt_depth_norm = gt_depth / scale

        img_t = torch.from_numpy(image_np.transpose(2, 0, 1).copy()).float() / 255.0
        img_t = normalize(img_t).unsqueeze(0).to(args.device)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            # DA3 teacher: anyview relative depth
            x_teacher = img_t.unsqueeze(1)  # [1, 1, 3, H, W]
            teacher_out = teacher.anyview(x_teacher)
            da3_depth = teacher_out.depth[0, 0].float().cpu().numpy()  # [H, W]

            # Student
            student_out = student(img_t)
            pred_depth = student_out.depth[0].float().cpu().numpy()

        label = f"{sample['key_prefix']}/f{sample['frame']:06d}"

        # Col 0: RGB
        axes[row, 0].imshow(image_np)
        axes[row, 0].set_title(f"RGB: {label}", fontsize=8)
        axes[row, 0].axis("off")

        # Col 1: GT depth (normalized)
        gt_masked = np.ma.masked_where(~valid, gt_depth_norm)
        vmax_gt = np.percentile(gt_depth_norm[valid], 95) if valid.sum() > 0 else 1.0
        im_gt = axes[row, 1].imshow(gt_masked, cmap="turbo_r", vmin=0, vmax=vmax_gt)
        axes[row, 1].set_title(f"GT depth_norm (scale={scale:.2f})", fontsize=8)
        axes[row, 1].axis("off")
        plt.colorbar(im_gt, ax=axes[row, 1], fraction=0.03, pad=0.02)

        # Col 2: DA3 teacher relative depth
        if args.valid_mask:
            da3_plot = np.ma.masked_where(~valid, da3_depth)
        else:
            da3_plot = da3_depth
        vmax_da3 = np.percentile(da3_depth[valid], 95) if valid.sum() > 0 else np.percentile(da3_depth, 95)
        im_da3 = axes[row, 2].imshow(da3_plot, cmap="turbo_r", vmin=0, vmax=vmax_da3)
        axes[row, 2].set_title("DA3 anyview depth (relative)", fontsize=8)
        axes[row, 2].axis("off")
        plt.colorbar(im_da3, ax=axes[row, 2], fraction=0.03, pad=0.02)

        # Col 3: Student pred depth
        if args.valid_mask:
            pred_plot = np.ma.masked_where(~valid, pred_depth)
        else:
            pred_plot = pred_depth
        vmax_pred = np.percentile(pred_depth[valid], 95) if (args.valid_mask and valid.sum() > 0) else np.percentile(pred_depth, 95)
        im_pred = axes[row, 3].imshow(pred_plot, cmap="turbo_r", vmin=0, vmax=vmax_pred)
        axes[row, 3].set_title("Student pred depth", fontsize=8)
        axes[row, 3].axis("off")
        plt.colorbar(im_pred, ax=axes[row, 3], fraction=0.03, pad=0.02)

    env.close()
    plt.tight_layout()
    plt.savefig(args.output, dpi=100, bbox_inches="tight")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
