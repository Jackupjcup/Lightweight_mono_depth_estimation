"""Fast Mono Depth — Knowledge distillation training (supports single-GPU and DDP)."""

import argparse
import math
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from data.tartanground_lmdb_dataset import TartanGroundLMDBDataset
from distillation.losses import feature_distillation_loss
from distillation.objective_losses import phase2_loss, build_pixel_dirs
from distillation.teacher import TeacherDA3
from models.fast_depth_model import FastDepthModel


def parse_args():
    parser = argparse.ArgumentParser(description="Fast Mono Depth — Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def build_scheduler(optimizer, cfg, steps_per_epoch: int):
    tc = cfg.training
    total_steps = tc.total_epochs * steps_per_epoch
    warmup_steps = tc.warmup_epochs * steps_per_epoch
    hold_steps = getattr(tc, "hold_epochs", 0) * steps_per_epoch
    decay_steps = total_steps - warmup_steps - hold_steps
    sched = getattr(tc, "scheduler", "cosine")

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        if step < warmup_steps + hold_steps:
            return 1.0
        progress = (step - warmup_steps - hold_steps) / max(decay_steps, 1)
        if sched == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        if sched == "linear":
            return 1.0 - progress
        if sched == "constant":
            return 1.0
        raise ValueError(f"Unknown scheduler: {sched}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def reduce_scalar(val, device):
    """All-reduce a Python scalar across DDP processes."""
    t = torch.tensor(val, device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()


def _gaussian_kernel_1d(size: int, sigma: float, device):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    return g / g.sum()


def compute_ssim(x, y, window_size=7, sigma=1.5):
    """Compute mean SSIM between feature maps (dynamic data_range).

    Args:
        x, y: [B, C, H, W] already spatially aligned.
    Returns:
        Scalar mean SSIM value in [-1, 1].
    """
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
    cfg = OmegaConf.load(args.config)

    # ---- DDP setup ----
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1

    if is_distributed:
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
        torch.cuda.set_device(local_rank)

    device = f"cuda:{local_rank}"
    is_main = local_rank == 0

    tc = cfg.training
    dc = cfg.data
    mc = cfg.model

    # ---- Teacher (frozen, no DDP needed) ----
    if is_main:
        print("Loading DA3 teacher...")
    teacher = TeacherDA3(
        model_dir=cfg.teacher.model_dir,
        da3_src=cfg.teacher.da3_src,
        device=device,
    )
    if is_main:
        print("Teacher loaded and frozen.")

    # ---- Student ----
    student = FastDepthModel(
        backbone_pretrained=mc.backbone_pretrained,
        dpt_features=mc.dpt_features,
        dpt_out_channels=tuple(mc.dpt_out_channels),
        depth_activation=getattr(mc, "depth_activation", "softplus"),
        conf_activation=getattr(mc, "conf_activation", "softplusp1"),
    ).to(device)
    if is_main:
        print(f"Student params: {sum(p.numel() for p in student.parameters()) / 1e6:.2f}M")
    if is_distributed:
        student = DDP(student, device_ids=[local_rank], find_unused_parameters=True, static_graph=True)

    # ---- Data ----
    cj = OmegaConf.to_container(dc.color_jitter, resolve=True) if dc.get("color_jitter") else None
    train_dataset = TartanGroundLMDBDataset(
        lmdb_path=dc.lmdb_path,
        index_json=dc.train_index,
        input_height=dc.input_height,
        input_width=dc.input_width,
        ray_height=dc.ray_height,
        ray_width=dc.ray_width,
        depth_cap=dc.depth_cap,
        augment=dc.augment,
        flip_prob=dc.flip_prob,
        color_jitter=cj,
    )
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
    dataloader = DataLoader(
        train_dataset,
        batch_size=tc.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=tc.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_dataset = TartanGroundLMDBDataset(
        lmdb_path=dc.lmdb_path,
        index_json=dc.val_index,
        input_height=dc.input_height,
        input_width=dc.input_width,
        ray_height=dc.ray_height,
        ray_width=dc.ray_width,
        depth_cap=dc.depth_cap,
        augment=False,
    )
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=getattr(tc, "val_batch_size", tc.batch_size * 2),
        shuffle=False,
        sampler=val_sampler,
        num_workers=getattr(tc, "val_num_workers", tc.num_workers),
        pin_memory=True,
        drop_last=False,
    )
    if is_main:
        print(
            f"Train: {len(train_dataset)} samples, {len(dataloader)} batches/epoch | "
            f"Val: {len(val_dataset)} samples, {len(val_dataloader)} batches"
        )

    # ---- Optimizer & Scheduler (lr scaled by world_size) ----
    effective_lr = tc.learning_rate * world_size
    raw_student = student.module if is_distributed else student
    optimizer = torch.optim.AdamW(
        raw_student.parameters(), lr=effective_lr, weight_decay=tc.weight_decay,
    )
    scheduler = build_scheduler(optimizer, cfg, len(dataloader))
    lc = cfg.loss

    # ---- Early stopping config ----
    es_metric = getattr(cfg.logging, "early_stop_metric", "none")
    es_patience = getattr(cfg.logging, "early_stop_patience", 10)
    es_counter = 0

    # ---- Resume ----
    start_epoch = 0
    global_step = 0
    best_delta = 0.0
    best_feat_ssim = -float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw_student.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best_delta = ckpt.get("best_delta", 0.0)
        best_feat_ssim = ckpt.get("best_feat_ssim", -float("inf"))
        es_counter = ckpt.get("es_counter", 0)
        if is_main:
            print(f"Resumed from epoch {start_epoch}, step {global_step}, best_delta={best_delta:.4f}, best_feat_ssim={best_feat_ssim:.4f}, es_counter={es_counter}")

    # ---- Logging (rank 0 only) ----
    project_root = Path(__file__).resolve().parent.parent
    config_stem = Path(args.config).stem
    run_name = os.environ.get("RUN_NAME", time.strftime("%Y%m%d_%H%M%S"))
    log_dir = project_root / "work_dirs" / config_stem / run_name
    writer = None
    if is_main:
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir)
        OmegaConf.save(cfg, log_dir / "config.yaml")

    # ---- Precompute pixel_dirs for point cloud loss (Phase 2) ----
    pixel_dirs = None
    phase = getattr(lc, "phase", 1)
    if phase >= 2:
        import json
        with open(dc.train_index) as f:
            K_np = np.array(json.load(f)["meta"]["K_cropped"], dtype=np.float64)
        pixel_dirs = build_pixel_dirs(K_np, dc.input_height, dc.input_width).to(device)

    # ---- Training loop ----
    feat_weights = list(lc.feat_weights)

    for epoch in range(start_epoch, tc.total_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        student.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, batch in enumerate(dataloader):
            images = batch["image"].to(device, non_blocking=True)
            images_clean = batch["image_clean"].to(device, non_blocking=True)

            # Teacher forward (frozen, no grad) on clean images (no color jitter)
            teacher_feats = teacher.extract_features(images_clean)

            # Student forward (mixed precision)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output, student_feats = student(images, return_distill_feats=True)
                feat_loss, per_level = feature_distillation_loss(
                    student_feats, teacher_feats, weights=feat_weights
                )
                loss = feat_loss

                # Phase 2: add task-level losses
                task_dict = {}
                if phase >= 2:
                    batch_dev = {
                        "depth": batch["depth_normalized"].to(device, non_blocking=True),
                        "ray": batch["ray"].to(device, non_blocking=True),
                        "cam_params": batch["cam_params"].to(device, non_blocking=True),
                        "valid_mask": batch["valid_mask"].to(device, non_blocking=True),
                    }
                    task_loss, task_dict = phase2_loss(
                        output, batch_dev, pixel_dirs, lc,
                    )
                    loss = loss + task_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(raw_student.parameters(), max_norm=float("inf"))
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if is_main and global_step % cfg.logging.log_interval == 0:
                with torch.no_grad():
                    ssim_vals = []
                    for sf, tf in zip(student_feats, teacher_feats):
                        sf_a = nn.functional.interpolate(sf.float(), size=tf.shape[2:], mode="bilinear", align_corners=False)
                        ssim_vals.append(compute_ssim(sf_a, tf.float()))
                lr = optimizer.param_groups[0]["lr"]
                msg = (
                    f"[Epoch {epoch}/{tc.total_epochs}] "
                    f"Step {batch_idx+1}/{len(dataloader)} | Loss {loss.item():.6f} | "
                    f"L1={per_level[0]:.4f} L2={per_level[1]:.4f} "
                    f"L3={per_level[2]:.4f} L4={per_level[3]:.4f} | "
                    f"S1={ssim_vals[0]:.4f} S2={ssim_vals[1]:.4f} "
                    f"S3={ssim_vals[2]:.4f} S4={ssim_vals[3]:.4f}"
                )
                if task_dict:
                    msg += (
                        f" | D={task_dict['depth']:.4f} R={task_dict['ray']:.4f}"
                        f" P={task_dict['point']:.4f} C={task_dict['cam']:.4f}"
                        f" G={task_dict['grad']:.4f}"
                    )
                msg += f" | GN {grad_norm:.2f} | LR {lr:.2e}"
                print(msg)
                writer.add_scalar("loss/total", loss.item(), global_step)
                writer.add_scalar("loss/feat", feat_loss.item(), global_step)
                for i, lv in enumerate(per_level):
                    writer.add_scalar(f"loss/level_{i+1}", lv, global_step)
                for i, sv in enumerate(ssim_vals):
                    writer.add_scalar(f"loss/ssim_{i+1}", sv, global_step)
                for k, v in task_dict.items():
                    writer.add_scalar(f"loss/{k}", v, global_step)
                writer.add_scalar("grad_norm", grad_norm.item(), global_step)
                writer.add_scalar("lr", lr, global_step)

        avg_loss = epoch_loss / len(dataloader)
        elapsed = time.time() - t0
        if is_main:
            print(f"Epoch {epoch} done — avg loss {avg_loss:.6f} — {elapsed:.1f}s")

        # ---- Validation ----
        eval_interval = getattr(cfg.logging, "eval_interval", 5)
        do_eval = ((epoch + 1) % eval_interval == 0) or (epoch == tc.total_epochs - 1)

        if do_eval:
            student.eval()
            val_loss_sum = 0.0
            val_feat_sum = [0.0] * 4
            val_ssim_sum = [0.0] * 4
            val_task_sum = {}
            val_steps = 0
            absrel_sum = 0.0
            delta_correct = 0
            n_metric_pixels = 0
            delta_thr = getattr(cfg.logging, "delta_threshold", 1.25)

            with torch.no_grad():
                for val_batch in val_dataloader:
                    val_images = val_batch["image"].to(device, non_blocking=True)
                    val_images_clean = val_batch["image_clean"].to(device, non_blocking=True)
                    t_feats = teacher.extract_features(val_images_clean)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        v_out, s_feats = student(val_images, return_distill_feats=True)
                        v_feat_loss, v_per_level = feature_distillation_loss(
                            s_feats, t_feats, weights=feat_weights,
                        )
                        v_loss = v_feat_loss
                        v_task = {}
                        if phase >= 2:
                            v_batch_dev = {
                                "depth": val_batch["depth_normalized"].to(device, non_blocking=True),
                                "ray": val_batch["ray"].to(device, non_blocking=True),
                                "cam_params": val_batch["cam_params"].to(device, non_blocking=True),
                                "valid_mask": val_batch["valid_mask"].to(device, non_blocking=True),
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

                    if phase >= 2:
                        pred_d = v_out.depth.float()
                        gt_d = val_batch["depth_normalized"].to(device, non_blocking=True)
                        vmask = val_batch["valid_mask"].to(device, non_blocking=True)
                        metric_mask = vmask & (gt_d > 1e-3)
                        n = metric_mask.sum().item()
                        if n > 0:
                            pred_v = pred_d[metric_mask]
                            gt_v = gt_d[metric_mask]
                            absrel_sum += ((pred_v - gt_v).abs() / gt_v).sum().item()
                            ratio = torch.max(pred_v / gt_v, gt_v / pred_v)
                            delta_correct += (ratio < delta_thr).sum().item()
                            n_metric_pixels += n

            # ---- All-reduce val metrics across GPUs ----
            if is_distributed:
                val_loss_sum = reduce_scalar(val_loss_sum, device)
                val_steps = int(reduce_scalar(val_steps, device))
                for i in range(4):
                    val_feat_sum[i] = reduce_scalar(val_feat_sum[i], device)
                    val_ssim_sum[i] = reduce_scalar(val_ssim_sum[i], device)
                for k in list(val_task_sum.keys()):
                    val_task_sum[k] = reduce_scalar(val_task_sum[k], device)
                absrel_sum = reduce_scalar(absrel_sum, device)
                delta_correct = reduce_scalar(delta_correct, device)
                n_metric_pixels = int(reduce_scalar(n_metric_pixels, device))

            val_avg = val_loss_sum / max(val_steps, 1)
            val_feat_avg = sum(val_feat_sum) / max(val_steps, 1)
            val_ssim_mean = sum(val_ssim_sum) / (4 * max(val_steps, 1))
            val_absrel = 0.0
            val_delta = 0.0
            if n_metric_pixels > 0:
                val_absrel = absrel_sum / n_metric_pixels
                val_delta = delta_correct / n_metric_pixels

            if is_main:
                writer.add_scalar("val/loss", val_avg, global_step)
                writer.add_scalar("val/feat", val_feat_avg, global_step)
                for i in range(4):
                    writer.add_scalar(f"val/level_{i+1}", val_feat_sum[i] / max(val_steps, 1), global_step)
                for k, v in val_task_sum.items():
                    writer.add_scalar(f"val/{k}", v / max(val_steps, 1), global_step)

                ssim_avg = [val_ssim_sum[i] / max(val_steps, 1) for i in range(4)]
                for i in range(4):
                    writer.add_scalar(f"val/ssim_{i+1}", ssim_avg[i], global_step)
                ssim_str = " ".join(f"S{i+1}={ssim_avg[i]:.4f}" for i in range(4))
                print(f"  SSIM: {ssim_str}")
                feat_per = [val_feat_sum[i] / max(val_steps, 1) for i in range(4)]
                feat_str = " ".join(f"L{i+1}={feat_per[i]:.4f}" for i in range(4))
                msg = f"  Val loss: {val_avg:.6f} | feat: {val_feat_avg:.6f} | {feat_str}"
                if n_metric_pixels > 0:
                    msg += f" | AbsRel={val_absrel:.4f} δ1={val_delta:.4f}"
                    writer.add_scalar("val/absrel", val_absrel, global_step)
                    writer.add_scalar(f"val/delta_{delta_thr}", val_delta, global_step)
                print(msg)

            # --- Check improvements (all ranks identical after all-reduce) ---
            # "feat" early-stop / best metric tracks the 4-level average SSIM (higher = better)
            feat_improved = val_ssim_mean > best_feat_ssim
            delta_improved = n_metric_pixels > 0 and val_delta > best_delta

            if feat_improved:
                best_feat_ssim = val_ssim_mean
            if delta_improved:
                best_delta = val_delta

            # --- Early stopping counter (all ranks compute identically) ---
            if es_metric != "none":
                es_improved = (es_metric == "delta" and delta_improved) or \
                              (es_metric == "feat" and feat_improved)
                es_counter = 0 if es_improved else es_counter + 1

            # --- Save best checkpoints (rank 0 only) ---
            if is_main and feat_improved:
                best_feat_path = log_dir / "best_feat.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_delta": best_delta,
                        "best_feat_ssim": best_feat_ssim,
                        "es_counter": es_counter,
                        "model": raw_student.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    },
                    best_feat_path,
                )
                print(f"  New best avg-SSIM={best_feat_ssim:.4f} → {best_feat_path}")

            if is_main and delta_improved:
                best_delta_path = log_dir / "best_delta.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_delta": best_delta,
                        "best_feat_ssim": best_feat_ssim,
                        "es_counter": es_counter,
                        "model": raw_student.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    },
                    best_delta_path,
                )
                print(f"  New best δ1={best_delta:.4f} → {best_delta_path}")

            # --- Early stop break ---
            if es_metric != "none":
                if is_main:
                    print(f"  Early stop: metric={es_metric}, patience={es_counter}/{es_patience}")
                if es_counter >= es_patience:
                    if is_main:
                        print(f"Early stopping triggered at epoch {epoch} (no improvement for {es_patience} evals)")
                    break

        # Save periodic checkpoint (rank 0 only)
        if is_main and ((epoch + 1) % cfg.logging.save_interval == 0 or epoch == tc.total_epochs - 1):
            ckpt_path = log_dir / f"ckpt_epoch{epoch:04d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_delta": best_delta,
                    "best_feat_ssim": best_feat_ssim,
                    "es_counter": es_counter,
                    "model": raw_student.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                ckpt_path,
            )
            print(f"Saved checkpoint: {ckpt_path}")

        # Barrier: wait for checkpoint I/O before next epoch
        if is_distributed:
            dist.barrier()

    if is_main:
        writer.close()
        print("Training complete.")
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
