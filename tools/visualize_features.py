"""Visualize 4-level features from teacher (T1~T4), student projected (P1~P4),
and student backbone raw (f1~f4).

Image is loaded from a TartanGround LMDB via a JSON annotation index.

Usage:
    # Random sample from val split
    python tools/visualize_features.py \
        --json /root/vepfs/lrd_projs/mono_depth_estimation/fast_mono_depth/data/annotations/tartanground_val.json

    # Specific sample index
    python tools/visualize_features.py \
        --json /root/vepfs/lrd_projs/mono_depth_estimation/fast_mono_depth/data/annotations/tartanground_val.json \
        --idx 42

    # Skip teacher (no DA3 needed)
    python tools/visualize_features.py --json ... --no-teacher

    # Load a trained student checkpoint
    python tools/visualize_features.py --json ... --student-ckpt work_dirs/xxx/best.pt

    # 随机取一张图
    python tools/visualize_features.py \
        --json data/annotations/tartanground_val.json

    # 指定第42个样本
    python tools/visualize_features.py \
        --json data/annotations/tartanground_val.json \
        --idx 42 --out work_dirs/vis_feat/vis_42.png \
        --student-ckpt /root/data-vepfs/lrd_projs/mono_depth_estimation/fast_mono_depth/work_dirs/v0_v1/20260620_035604/best_delta.pt

    # 不加载老师（快速看学生特征）
    python tools/visualize_features.py \
        --json data/annotations/tartanground_val.json \
        --no-teacher
"""

import argparse
import json
import os
import random
import sys
import zlib

import cv2
import lmdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.fast_depth_model import FastDepthModel

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
LMDB_PATH     = "/data-vepfs/lrd_projs/mono_depth_estimation/fast_mono_depth/data/tartanground.lmdb"

_normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)


# ── LMDB helpers ──────────────────────────────────────────────────────────────

def load_rgb_from_lmdb(env: lmdb.Environment, key_prefix: str, frame: int) -> np.ndarray:
    """Return (H, W, 3) uint8 RGB."""
    key = f"{key_prefix}/{frame:06d}/rgb".encode()
    with env.begin() as txn:
        buf = txn.get(key)
    assert buf is not None, f"Key not found in LMDB: {key.decode()}"
    img = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb_to_tensor(img: np.ndarray) -> torch.Tensor:
    """(H, W, 3) uint8 → (1, 3, H, W) float32, ImageNet-normalised."""
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # [3, H, W]
    return _normalize(t).unsqueeze(0)                             # [1, 3, H, W]


# ── Feature → heatmap ─────────────────────────────────────────────────────────

def feat_to_heatmap(feat: torch.Tensor) -> np.ndarray:
    """[B, C, H, W] → [H, W] float32 in [0, 1] via channel mean."""
    hmap = feat.float().mean(dim=1)[0].cpu().numpy()   # [H, W]
    lo, hi = hmap.min(), hmap.max()
    if hi > lo:
        hmap = (hmap - lo) / (hi - lo)
    else:
        hmap = np.zeros_like(hmap)
    return hmap


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _show_feat(ax, feat: torch.Tensor, title_top: str, cmap: str = "inferno"):
    """Plot one feature map.  Burn H×W into the image, not just the title."""
    hmap = feat_to_heatmap(feat)
    h, w = hmap.shape
    ax.imshow(hmap, cmap=cmap)
    # White text label burned into the bottom-left of the heatmap
    ax.text(
        0.02, 0.04, f"{w}×{h}",
        transform=ax.transAxes,
        fontsize=7, color="white",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.45, lw=0),
    )
    ax.set_title(title_top, fontsize=8)
    ax.axis("off")


def _show_img(ax, img_rgb: np.ndarray, label: str):
    ax.imshow(img_rgb)
    ax.set_title(label, fontsize=8)
    ax.axis("off")


def _row_label(ax, text: str):
    ax.set_ylabel(text, fontsize=7.5, rotation=0, labelpad=65, va="center")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json", required=True,
        help="Annotation JSON, e.g. .../tartanground_val.json",
    )
    parser.add_argument(
        "--idx", type=int, default=None,
        help="Sample index in JSON (random if omitted)",
    )
    parser.add_argument("--lmdb", default=LMDB_PATH)
    parser.add_argument("--no-teacher", action="store_true")
    parser.add_argument(
        "--teacher-dir",
        default="/data-vepfs/lrd_projs/mono_depth_estimation/fast_mono_depth/mono_body_dataset/"
                "Depth-Anything-3-main/models/DA3NESTED-GIANT-LARGE-1.1/",
    )
    parser.add_argument(
        "--da3-src",
        default="/data-vepfs/lrd_projs/mono_depth_estimation/fast_mono_depth/mono_body_dataset/"
                "Depth-Anything-3-main/src",
    )
    parser.add_argument("--student-ckpt", default=None)
    parser.add_argument("--out", default="feature_vis.png")
    args = parser.parse_args()

    if args.no_teacher and args.student_ckpt is None:
        parser.error(
            "--no-teacher requires --student-ckpt.\n"
            "  P1~P4 (projects[i] output) are randomly initialized without a checkpoint"
            " — nothing meaningful to visualize.\n"
            "  f1~f4 (backbone) use ImageNet pretrained weights and are valid,"
            " but in that case just load the teacher too."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Pick sample from JSON ─────────────────────────────────────────────────
    with open(args.json) as f:
        index = json.load(f)
    samples = index["samples"]
    idx = args.idx if args.idx is not None else random.randint(0, len(samples) - 1)
    sample = samples[idx]
    print(f"Sample [{idx}/{len(samples)-1}]  key={sample['key_prefix']}  frame={sample['frame']}")

    # ── Load RGB from LMDB ────────────────────────────────────────────────────
    env = lmdb.open(args.lmdb, readonly=True, lock=False, readahead=False, map_size=1 << 40)
    img_rgb = load_rgb_from_lmdb(env, sample["key_prefix"], sample["frame"])
    env.close()
    print(f"Image shape: {img_rgb.shape}  dtype={img_rgb.dtype}")

    image_tensor = rgb_to_tensor(img_rgb).to(device)  # [1, 3, 476, 644]

    # ── Student ───────────────────────────────────────────────────────────────
    student = FastDepthModel(backbone_pretrained=True).to(device)
    if args.student_ckpt:
        ckpt = torch.load(args.student_ckpt, map_location=device)
        state = ckpt.get("model", ckpt)
        student.load_state_dict(state, strict=False)
        print(f"Loaded student ckpt: {args.student_ckpt}")
    student.eval()

    with torch.no_grad():
        raw_feats = student.backbone(image_tensor)                        # f1~f4
        _, proj_feats = student(image_tensor, return_distill_feats=True)  # P1~P4

    print("Student raw backbone features (before projects[i]):")
    for i, f in enumerate(raw_feats):
        print(f"  f{i+1}: {tuple(f.shape)}  channels={f.shape[1]}")
    print("Student projected features (after projects[i], used for loss):")
    for i, p in enumerate(proj_feats):
        print(f"  P{i+1}: {tuple(p.shape)}  channels={p.shape[1]}")

    # ── Teacher ───────────────────────────────────────────────────────────────
    teacher_feats = None
    if not args.no_teacher:
        from distillation.teacher import TeacherDA3
        teacher = TeacherDA3(
            model_dir=args.teacher_dir,
            da3_src=args.da3_src,
            device=str(device),
        )
        teacher_feats = teacher.extract_features(image_tensor)
        print("Teacher features (hook after projects + resize_layers):")
        for i, t in enumerate(teacher_feats):
            print(f"  T{i+1}: {tuple(t.shape)}  channels={t.shape[1]}")

    # ── Build figure ──────────────────────────────────────────────────────────
    n_rows = 2 + (1 if teacher_feats is not None else 0)
    # 5 cols: input image + L1 L2 L3 L4
    fig, axes = plt.subplots(n_rows, 5, figsize=(19, 3.8 * n_rows),
                             gridspec_kw={"wspace": 0.04, "hspace": 0.35})
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    STRIDES = ["stride 4", "stride 8", "stride 16", "stride 32"]
    row = 0

    if teacher_feats is not None:
        _show_img(axes[row, 0], img_rgb, "Input")
        _row_label(axes[row, 0], "Teacher\n(after projects\n+ resize_layers)")
        for col, (feat, s) in enumerate(zip(teacher_feats, STRIDES), start=1):
            c = feat.shape[1]
            _show_feat(axes[row, col], feat,
                       f"T{col}  {c}ch  [{s}]")
        row += 1

    _show_img(axes[row, 0], img_rgb, "Input")
    _row_label(axes[row, 0], "Student projected\n(after projects[i]\nfor distill loss)")
    for col, (feat, s) in enumerate(zip(proj_feats, STRIDES), start=1):
        c = feat.shape[1]
        _show_feat(axes[row, col], feat,
                   f"P{col}  {c}ch  [{s}]")
    row += 1

    _show_img(axes[row, 0], img_rgb, "Input")
    _row_label(axes[row, 0], "Student raw\n(backbone output\nbefore projects[i])")
    for col, (feat, s) in enumerate(zip(raw_feats, STRIDES), start=1):
        c = feat.shape[1]
        _show_feat(axes[row, col], feat,
                   f"f{col}  {c}ch  [{s}]")

    src_name = os.path.basename(args.json)
    title = (
        f"Feature maps  |  {src_name}  idx={idx}\n"
        f"{sample['key_prefix']} / frame {sample['frame']}\n"
        f"Channel compression: mean over C → [H,W] → normalize [0,1]"
    )
    fig.suptitle(title, fontsize=9, y=1.01)

    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
