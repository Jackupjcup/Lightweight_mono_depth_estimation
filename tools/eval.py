"""Fast Mono Depth — Standalone evaluation on a validation JSON index.

Usage:
cd /data-vepfs/lrd_projs/mono_depth_estimation/fast_mono_depth && conda activate fast_mono_depth_lrd && \
PYTHONPATH="$(pwd):$PYTHONPATH" python tools/eval.py \
    --checkpoint work_dirs/v0_tenth/20260618_104616/ckpt_epoch0000.pt \
    --config configs/v0_tenth_v1.yaml \
    --index_json data/annotations/tartanground_val_tenth.json


cd /data-vepfs/lrd_projs/mono_depth_estimation/fast_mono_depth && \
PYTHONPATH="$(pwd):$PYTHONPATH" python tools/eval.py \
    --index_json data/annotations/tartanground_val_tenth.json \
    --checkpoint work_dirs/v0_tenth_v1/20260618_154428/best_delta.pt\
    --config configs/v0_tenth_v1.yaml \
    --eval_da3 \
    --no-valid-mask

cd /data-vepfs/lrd_projs/mono_depth_estimation/fast_mono_depth && \
PYTHONPATH="$(pwd):$PYTHONPATH" python tools/eval.py \
    --index_json data/annotations/tartanground_val_tenth.json \
    --checkpoint work_dirs/v1_tenth/20260622_062707/best_delta.pt\
    --config configs/v1_tenth.yaml \
    --eval_da3 
    
"""

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from data.tartanground_lmdb_dataset import TartanGroundLMDBDataset
from distillation.losses import feature_distillation_loss
from distillation.objective_losses import phase2_loss, build_pixel_dirs
from distillation.teacher import TeacherDA3
from models.fast_depth_model import FastDepthModel


def parse_args():
    p = argparse.ArgumentParser(description="Fast Mono Depth — Evaluation")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--config", type=str, required=True, help="Training config YAML")
    p.add_argument("--index_json", type=str, default=None,
                   help="Override val index JSON (default: from config)")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--eval_da3", action="store_true",
                   help="Also evaluate DA3 anyview depth (median-scaled to GT)")
    p.add_argument("--no_valid_mask", action="store_true",
                   help="Ignore valid_mask: compute metrics on all pixels (gt>1e-3 only)")
    return p.parse_args()


def _gaussian_kernel_1d(size, sigma, device):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    return g / g.sum()


def compute_ssim(x, y, window_size=7, sigma=1.5):
    B, C, H, W = x.shape
    data_range = max((x.max() - x.min()).item(), (y.max() - y.min()).item(), 1e-8)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    k1d = _gaussian_kernel_1d(window_size, sigma, x.device)
    kernel = k1d[:, None] * k1d[None, :]
    kernel = kernel.expand(C, 1, window_size, window_size)
    pad = window_size // 2

    mu_x = nn.functional.conv2d(x, kernel, padding=pad, groups=C)
    mu_y = nn.functional.conv2d(y, kernel, padding=pad, groups=C)

    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = nn.functional.conv2d(x * x, kernel, padding=pad, groups=C) - mu_x_sq
    sigma_y_sq = nn.functional.conv2d(y * y, kernel, padding=pad, groups=C) - mu_y_sq
    sigma_xy = nn.functional.conv2d(x * y, kernel, padding=pad, groups=C) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
    return ssim_map.mean().item()


def main():
    args = parse_args()
    device = args.device

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)
    dc = cfg.data
    mc = cfg.model
    lc = cfg.loss

    # --- Load student from checkpoint ---
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    ckpt_dir = os.path.dirname(args.checkpoint)
    ckpt_cfg_path = os.path.join(ckpt_dir, "config.yaml")
    ckpt_mcfg = {}
    if os.path.exists(ckpt_cfg_path):
        with open(ckpt_cfg_path) as f:
            ckpt_mcfg = yaml.safe_load(f).get("model", {})

    student = FastDepthModel(
        backbone_pretrained=False,
        dpt_features=ckpt_mcfg.get("dpt_features", mc.dpt_features),
        dpt_out_channels=tuple(ckpt_mcfg.get("dpt_out_channels", list(mc.dpt_out_channels))),
        depth_activation=ckpt_mcfg.get("depth_activation", getattr(mc, "depth_activation", "softplus")),
        conf_activation=ckpt_mcfg.get("conf_activation", getattr(mc, "conf_activation", "softplusp1")),
    )
    student.load_state_dict(ckpt["model"])
    student = student.to(device).eval()

    epoch = ckpt.get("epoch", "?")
    step = ckpt.get("global_step", "?")
    act = ckpt_mcfg.get("depth_activation", getattr(mc, "depth_activation", "softplus"))
    print(f"Student loaded: epoch={epoch}, step={step}, depth_activation={act}")
    print(f"  Params: {sum(p.numel() for p in student.parameters()) / 1e6:.2f}M")

    # --- Load teacher ---
    print("Loading DA3 teacher...")
    teacher = TeacherDA3(
        model_dir=cfg.teacher.model_dir,
        da3_src=cfg.teacher.da3_src,
        device=device,
    )
    print("Teacher loaded.")

    # --- Dataset ---
    val_index = args.index_json or dc.val_index
    val_dataset = TartanGroundLMDBDataset(
        lmdb_path=dc.lmdb_path,
        index_json=val_index,
        input_height=dc.input_height,
        input_width=dc.input_width,
        ray_height=dc.ray_height,
        ray_width=dc.ray_width,
        depth_cap=dc.depth_cap,
        augment=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Val set: {len(val_dataset)} samples, {len(val_loader)} batches")

    # --- Pixel dirs for point cloud loss ---
    phase = getattr(lc, "phase", 1)
    pixel_dirs = None
    if phase >= 2:
        with open(val_index) as f:
            K_np = np.array(json.load(f)["meta"]["K_cropped"], dtype=np.float64)
        pixel_dirs = build_pixel_dirs(K_np, dc.input_height, dc.input_width).to(device)

    feat_weights = list(lc.feat_weights)
    delta_thr = getattr(cfg.logging, "delta_threshold", 1.25)

    # --- Eval loop ---
    val_loss_sum = 0.0
    val_feat_sum = [0.0] * 4
    val_ssim_sum = [0.0] * 4
    val_task_sum = {}
    val_steps = 0
    absrel_sum = 0.0
    delta_correct = 0
    n_metric_pixels = 0
    sq_rel_sum = 0.0
    rmse_sum = 0.0
    log_rmse_sum = 0.0
    delta2_correct = 0
    delta3_correct = 0
    silog_g_sum = 0.0
    silog_g2_sum = 0.0

    # DA3 depth metric accumulators
    da3_absrel_sum = 0.0
    da3_sq_rel_sum = 0.0
    da3_rmse_sum = 0.0
    da3_log_rmse_sum = 0.0
    da3_delta_correct = 0
    da3_delta2_correct = 0
    da3_delta3_correct = 0
    da3_silog_g_sum = 0.0
    da3_silog_g2_sum = 0.0
    da3_n_pixels = 0

    t0 = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            images = batch["image"].to(device, non_blocking=True)
            t_feats = teacher.extract_features(images)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                v_out, s_feats = student(images, return_distill_feats=True)
                v_feat_loss, v_per_level = feature_distillation_loss(
                    s_feats, t_feats, weights=feat_weights,
                )
                v_loss = v_feat_loss
                v_task = {}
                if phase >= 2:
                    v_batch_dev = {
                        "depth": batch["depth_normalized"].to(device, non_blocking=True),
                        "ray": batch["ray"].to(device, non_blocking=True),
                        "cam_params": batch["cam_params"].to(device, non_blocking=True),
                        "valid_mask": batch["valid_mask"].to(device, non_blocking=True),
                    }
                    v_task_loss, v_task = phase2_loss(v_out, v_batch_dev, pixel_dirs, lc)
                    v_loss = v_loss + v_task_loss

            val_loss_sum += v_loss.item()
            for i, lv in enumerate(v_per_level):
                val_feat_sum[i] += lv
            for i, (sf, tf) in enumerate(zip(s_feats, t_feats)):
                sf_a = nn.functional.interpolate(sf.float(), size=tf.shape[2:], mode="bilinear", align_corners=False)
                val_ssim_sum[i] += compute_ssim(sf_a, tf.detach().float())
            for k, v in v_task.items():
                val_task_sum[k] = val_task_sum.get(k, 0.0) + v
            val_steps += 1

            gt_d = batch["depth_normalized"].to(device, non_blocking=True)
            vmask = batch["valid_mask"].to(device, non_blocking=True)
            if args.no_valid_mask:
                metric_mask = gt_d > 1e-3
            else:
                metric_mask = vmask & (gt_d > 1e-3)

            # --- Student depth metrics ---
            if phase >= 2:
                pred_d = v_out.depth.float()
                n = metric_mask.sum().item()
                if n > 0:
                    pred_v = pred_d[metric_mask]
                    gt_v = gt_d[metric_mask]

                    absrel_sum += ((pred_v - gt_v).abs() / gt_v).sum().item()
                    sq_rel_sum += (((pred_v - gt_v) ** 2) / gt_v).sum().item()
                    rmse_sum += ((pred_v - gt_v) ** 2).sum().item()
                    log_rmse_sum += ((torch.log(pred_v.clamp(min=1e-6)) - torch.log(gt_v)) ** 2).sum().item()

                    ratio = torch.max(pred_v / gt_v, gt_v / pred_v)
                    delta_correct += (ratio < delta_thr).sum().item()
                    delta2_correct += (ratio < delta_thr ** 2).sum().item()
                    delta3_correct += (ratio < delta_thr ** 3).sum().item()

                    g = torch.log(pred_v.clamp(min=1e-6)) - torch.log(gt_v)
                    silog_g_sum += g.sum().item()
                    silog_g2_sum += (g ** 2).sum().item()
                    n_metric_pixels += n

            # --- DA3 anyview depth metrics (per-image median scaling) ---
            if args.eval_da3:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    x_teacher = images.unsqueeze(1)
                    da3_out = teacher.anyview(x_teacher)
                    da3_depth = da3_out.depth[:, 0].float()  # [B, H, W]

                B = da3_depth.shape[0]
                for b in range(B):
                    m = metric_mask[b]
                    n_b = m.sum().item()
                    if n_b < 10:
                        continue
                    da3_v = da3_depth[b][m]
                    gt_v = gt_d[b][m]
                    scale_factor = gt_v.median() / da3_v.median().clamp(min=1e-8)
                    da3_scaled = da3_v * scale_factor

                    da3_absrel_sum += ((da3_scaled - gt_v).abs() / gt_v).sum().item()
                    da3_sq_rel_sum += (((da3_scaled - gt_v) ** 2) / gt_v).sum().item()
                    da3_rmse_sum += ((da3_scaled - gt_v) ** 2).sum().item()
                    da3_log_rmse_sum += ((torch.log(da3_scaled.clamp(min=1e-6)) - torch.log(gt_v)) ** 2).sum().item()

                    ratio = torch.max(da3_scaled / gt_v, gt_v / da3_scaled)
                    da3_delta_correct += (ratio < delta_thr).sum().item()
                    da3_delta2_correct += (ratio < delta_thr ** 2).sum().item()
                    da3_delta3_correct += (ratio < delta_thr ** 3).sum().item()

                    da3_g = torch.log(da3_scaled.clamp(min=1e-6)) - torch.log(gt_v)
                    da3_silog_g_sum += da3_g.sum().item()
                    da3_silog_g2_sum += (da3_g ** 2).sum().item()
                    da3_n_pixels += n_b

            if (batch_idx + 1) % 10 == 0 or batch_idx == len(val_loader) - 1:
                print(f"  [{batch_idx+1}/{len(val_loader)}] ...")

    elapsed = time.time() - t0

    # --- Aggregate ---
    val_avg = val_loss_sum / max(val_steps, 1)
    val_feat_avg = sum(val_feat_sum) / max(val_steps, 1)
    feat_per = [val_feat_sum[i] / max(val_steps, 1) for i in range(4)]
    ssim_avg = [val_ssim_sum[i] / max(val_steps, 1) for i in range(4)]

    print()
    print("=" * 70)
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Config     : {args.config}")
    print(f"Val index  : {val_index}")
    print(f"Samples    : {len(val_dataset)}  |  Batches: {val_steps}  |  Time: {elapsed:.1f}s")
    print(f"valid_mask : {'OFF (all pixels)' if args.no_valid_mask else 'ON (depth < cap)'}")
    print("=" * 70)

    print(f"\nTotal Loss : {val_avg:.6f}")
    print(f"Feat  Loss : {val_feat_avg:.6f}")
    feat_str = "  ".join(f"L{i+1}={feat_per[i]:.6f}" for i in range(4))
    print(f"  Per-level: {feat_str}")
    ssim_str = "  ".join(f"S{i+1}={ssim_avg[i]:.4f}" for i in range(4))
    print(f"  SSIM     : {ssim_str}")

    if val_task_sum:
        print("\nTask Losses:")
        for k in ["depth", "ray", "point", "cam", "grad"]:
            if k in val_task_sum:
                print(f"  {k:>8s} : {val_task_sum[k] / max(val_steps, 1):.6f}")

    if n_metric_pixels > 0:
        absrel = absrel_sum / n_metric_pixels
        sq_rel = sq_rel_sum / n_metric_pixels
        rmse = math.sqrt(rmse_sum / n_metric_pixels)
        log_rmse = math.sqrt(log_rmse_sum / n_metric_pixels)
        d1 = delta_correct / n_metric_pixels
        d2 = delta2_correct / n_metric_pixels
        d3 = delta3_correct / n_metric_pixels

        silog = math.sqrt(max(silog_g2_sum / n_metric_pixels - (silog_g_sum / n_metric_pixels) ** 2, 0.0))

        print(f"\nDepth Metrics ({n_metric_pixels:,} valid pixels):")
        print(f"  AbsRel   : {absrel:.4f}")
        print(f"  SqRel    : {sq_rel:.4f}")
        print(f"  RMSE     : {rmse:.4f}")
        print(f"  log RMSE : {log_rmse:.4f}")
        print(f"  SILog    : {silog:.4f}")
        print(f"  delta1   : {d1:.4f}  (< {delta_thr})")
        print(f"  delta2   : {d2:.4f}  (< {delta_thr}^2)")
        print(f"  delta3   : {d3:.4f}  (< {delta_thr}^3)")

    if args.eval_da3 and da3_n_pixels > 0:
        da3_absrel = da3_absrel_sum / da3_n_pixels
        da3_sq_rel = da3_sq_rel_sum / da3_n_pixels
        da3_rmse = math.sqrt(da3_rmse_sum / da3_n_pixels)
        da3_log_rmse = math.sqrt(da3_log_rmse_sum / da3_n_pixels)
        da3_d1 = da3_delta_correct / da3_n_pixels
        da3_d2 = da3_delta2_correct / da3_n_pixels
        da3_d3 = da3_delta3_correct / da3_n_pixels

        da3_silog = math.sqrt(max(da3_silog_g2_sum / da3_n_pixels - (da3_silog_g_sum / da3_n_pixels) ** 2, 0.0))

        print(f"\nDA3 Anyview Depth Metrics (median-scaled, {da3_n_pixels:,} valid pixels):")
        print(f"  AbsRel   : {da3_absrel:.4f}")
        print(f"  SqRel    : {da3_sq_rel:.4f}")
        print(f"  RMSE     : {da3_rmse:.4f}")
        print(f"  log RMSE : {da3_log_rmse:.4f}")
        print(f"  SILog    : {da3_silog:.4f}")
        print(f"  delta1   : {da3_d1:.4f}  (< {delta_thr})")
        print(f"  delta2   : {da3_d2:.4f}  (< {delta_thr}^2)")
        print(f"  delta3   : {da3_d3:.4f}  (< {delta_thr}^3)")

    print("=" * 70)


if __name__ == "__main__":
    main()
