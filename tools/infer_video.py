"""Fast Mono Depth — Video inference: side-by-side RGB + depth prediction.

cd /data-vepfs/lrd_projs/mono_depth_estimation/fast_mono_depth && conda activate fast_mono_depth_lrd && \
PYTHONPATH="$(pwd):$PYTHONPATH" python tools/infer_video.py \
    --checkpoint work_dirs/v1_tenth/20260622_062707/best_delta.pt \
    --input /data-vepfs/lrd_projs/mono_depth_estimation/fast_mono_depth/data/val_videos/camera_0_20260615_060355_886.h264 \
    --output work_dirs/infer_res/infer_videos/camera_0_20260615_060355_886.mp4
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
import torch
import yaml
from torchvision import transforms

from models.fast_depth_model import FastDepthModel

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args():
    p = argparse.ArgumentParser(description="Fast Mono Depth — Video Inference")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--input", type=str, required=True, help="Input video path")
    p.add_argument("--output", type=str, required=True, help="Output video path")
    p.add_argument("--input_height", type=int, default=476)
    p.add_argument("--input_width", type=int, default=644)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--colormap", type=str, default="turbo_r",
                   help="Matplotlib colormap for depth (default: turbo_r, red=near blue=far)")
    p.add_argument("--fps", type=float, default=None,
                   help="Output FPS (default: same as input)")
    return p.parse_args()


def load_student(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)

    ckpt_dir = os.path.dirname(checkpoint)
    cfg_path = os.path.join(ckpt_dir, "config.yaml")
    mcfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            mcfg = yaml.safe_load(f).get("model", {})
        print(f"Model config loaded from {cfg_path}")

    model = FastDepthModel(
        backbone_pretrained=False,
        dpt_features=mcfg.get("dpt_features", 256),
        dpt_out_channels=tuple(mcfg.get("dpt_out_channels", [256, 512, 1024, 1024])),
        depth_activation=mcfg.get("depth_activation", "softplus"),
        conf_activation=mcfg.get("conf_activation", "softplusp1"),
    )
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    epoch = ckpt.get("epoch", "?")
    step = ckpt.get("global_step", "?")
    act = mcfg.get("depth_activation", "softplus")
    print(f"Student loaded: epoch={epoch}, step={step}, depth_activation={act}")
    return model


import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


def depth_to_frame(depth_np, colormap, target_w, target_h):
    """Render depth map with colorbar and range text using matplotlib, return BGR uint8."""
    vmin = 0
    vmax = float(np.percentile(depth_np, 95))
    vmax = max(vmax, vmin + 1e-6)
    dmin = float(depth_np.min())
    dmax = float(depth_np.max())

    dpi = 100
    fig = Figure(figsize=(target_w / dpi, target_h / dpi), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0.0, 0.05, 0.88, 0.90])
    cax = fig.add_axes([0.89, 0.05, 0.03, 0.90])

    im = ax.imshow(depth_np, cmap=colormap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_axis_off()
    ax.set_title(f"range: [{dmin:.2f}, {dmax:.2f}]", fontsize=10, pad=2)
    fig.colorbar(im, cax=cax)

    canvas.draw()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(int(fig.get_figheight() * dpi), int(fig.get_figwidth() * dpi), 4)
    frame_rgb = buf[:, :, :3]
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    if frame_bgr.shape[0] != target_h or frame_bgr.shape[1] != target_w:
        frame_bgr = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    plt.close(fig)
    return frame_bgr


def main():
    args = parse_args()
    device = args.device

    student = load_student(args.checkpoint, device)
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: cannot open {args.input}")
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = None
    fps = args.fps or src_fps
    print(f"Input: {src_w}x{src_h} @ {src_fps}fps, {total_frames} frames")

    out_h = src_h
    out_w = src_w * 2
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (out_w, out_h))

    frame_idx = 0
    with torch.no_grad():
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(frame_rgb, (args.input_width, args.input_height),
                                 interpolation=cv2.INTER_LINEAR)

            img_t = torch.from_numpy(resized.transpose(2, 0, 1).copy()).float() / 255.0
            img_t = normalize(img_t).unsqueeze(0).to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = student(img_t)
                depth = out.depth[0].float().cpu().numpy()

            depth_full = cv2.resize(depth, (src_w, src_h), interpolation=cv2.INTER_LINEAR)
            depth_bgr = depth_to_frame(depth_full, args.colormap, src_w, src_h)

            combined = np.hstack([frame_bgr, depth_bgr])
            writer.write(combined)

            frame_idx += 1
            if frame_idx % 50 == 0 or (total_frames and frame_idx == total_frames):
                tf = total_frames or "?"
                print(f"  [{frame_idx}/{tf}]")

    cap.release()
    writer.release()
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
