"""Fast Mono Depth — Full distillation training (all targets from DA3 teacher).

All pseudo-labels (depth, ray, cam) come from the frozen DA3 anyview branch.
Uses log-space depth loss to prevent exp-activation gradient vanishing.
Checkpoint is compatible with the original FastDepthModel for Phase 2 GT fine-tuning.
"""

import argparse
import math
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from data.tartanground_lmdb_dataset import TartanGroundLMDBDataset
from distillation.losses import feature_distillation_loss
from distillation.losses_distill import (
    cam_distill_loss,
    grad_distill_loss,
    log_depth_distill_loss,
    ray_distill_loss,
)
from distillation.teacher_distill import TeacherDA3Distill
from models.fast_depth_model import FastDepthModel


def parse_args():
    parser = argparse.ArgumentParser(description="Fast Mono Depth — Distillation Training")
    parser.add_argument("--config", type=str, default="configs/v0_tenth_distill.yaml")
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


def _compute_feature_ssim(x, y, window_size=7):
    """Compute mean SSIM between two (B, C, H, W) feature tensors."""
    C = x.shape[1]
    sigma = 1.5
    coords = torch.arange(window_size, device=x.device, dtype=x.dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1).contiguous()
    pad = window_size // 2
    mu_x = F.conv2d(x, window, padding=pad, groups=C)
    mu_y = F.conv2d(y, window, padding=pad, groups=C)
    sigma_x_sq = F.conv2d(x * x, window, padding=pad, groups=C) - mu_x * mu_x
    sigma_y_sq = F.conv2d(y * y, window, padding=pad, groups=C) - mu_y * mu_y
    sigma_xy = F.conv2d(x * y, window, padding=pad, groups=C) - mu_x * mu_y
    data_range = max(
        (x.max() - x.min()).item(),
        (y.max() - y.min()).item(),
        1e-8,
    )
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x_sq + sigma_y_sq + c2)
    )
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
    teacher = TeacherDA3Distill(
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
    ).to(device)
    if is_main:
        print(f"Student params: {sum(p.numel() for p in student.parameters()) / 1e6:.2f}M")
    if is_distributed:
        student = DDP(student, device_ids=[local_rank], find_unused_parameters=True)

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

    # ---- Early stopping ----
    es_patience = getattr(cfg.logging, "early_stop_patience", 10)
    es_counter = 0
    best_loss = float("inf")

    # ---- Resume ----
    start_epoch = 0
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw_student.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best_loss = ckpt.get("best_loss", float("inf"))
        es_counter = ckpt.get("es_counter", 0)
        if is_main:
            print(
                f"Resumed from epoch {start_epoch}, step {global_step}, "
                f"best_loss={best_loss:.6f}, es={es_counter}"
            )

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

    # ---- Loss weights ----
    feat_weights = list(lc.feat_weights)
    w_depth = getattr(lc, "depth_weight", 10.0)
    w_ray = getattr(lc, "ray_weight", 1.0)
    w_cam = getattr(lc, "cam_weight", 1.0)
    w_grad = getattr(lc, "grad_weight", 1.0)
    w_cam_t = getattr(lc, "cam_trans_weight", 1.0)
    w_cam_q = getattr(lc, "cam_rot_weight", 1.0)
    w_cam_fov = getattr(lc, "cam_fov_weight", 0.5)

    # ---- Training loop ----
    for epoch in range(start_epoch, tc.total_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        student.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, batch in enumerate(dataloader):
            images = batch["image"].to(device, non_blocking=True)
            images_clean = batch["image_clean"].to(device, non_blocking=True)

            # Teacher: single forward on clean images (no color jitter)
            teacher_out = teacher.predict_all(images_clean)

            # Student forward (mixed precision)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output, student_feats = student(images, return_distill_feats=True)

                # Feature distillation loss (MSE, same as original)
                feat_loss, per_level = feature_distillation_loss(
                    student_feats, teacher_out["features"], weights=feat_weights,
                )

                # Task-level distillation losses
                ld = log_depth_distill_loss(
                    output.depth.float(), teacher_out["depth"],
                )
                lr_val = ray_distill_loss(
                    output.ray.float(), teacher_out["ray"],
                )
                lc_val = cam_distill_loss(
                    output.cam["pose_enc"].float(), teacher_out["cam_params"],
                    w_t=w_cam_t, w_q=w_cam_q, w_fov=w_cam_fov,
                )
                lg = grad_distill_loss(
                    output.depth.float(), teacher_out["depth"],
                )

                loss = (
                    feat_loss
                    + w_depth * ld
                    + w_ray * lr_val
                    + w_cam * lc_val
                    + w_grad * lg
                )

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
                    for sf, tf in zip(student_feats, teacher_out["features"]):
                        sf_a = F.interpolate(
                            sf.float(), size=tf.shape[2:],
                            mode="bilinear", align_corners=False,
                        )
                        ssim_vals.append(_compute_feature_ssim(sf_a, tf.float()))
                cur_lr = optimizer.param_groups[0]["lr"]
                msg = (
                    f"[Epoch {epoch}/{tc.total_epochs}] "
                    f"Step {batch_idx+1}/{len(dataloader)} | Loss {loss.item():.6f} | "
                    f"L1={per_level[0]:.4f} L2={per_level[1]:.4f} "
                    f"L3={per_level[2]:.4f} L4={per_level[3]:.4f} | "
                    f"S1={ssim_vals[0]:.4f} S2={ssim_vals[1]:.4f} "
                    f"S3={ssim_vals[2]:.4f} S4={ssim_vals[3]:.4f} | "
                    f"D={w_depth * ld.item():.4f} R={w_ray * lr_val.item():.4f} "
                    f"C={w_cam * lc_val.item():.4f} G={w_grad * lg.item():.4f} | "
                    f"GN {grad_norm:.2f} | LR {cur_lr:.2e}"
                )
                print(msg)
                writer.add_scalar("loss/total", loss.item(), global_step)
                writer.add_scalar("loss/feat", feat_loss.item(), global_step)
                for i, lv in enumerate(per_level):
                    writer.add_scalar(f"loss/level_{i+1}", lv, global_step)
                for i, sv in enumerate(ssim_vals):
                    writer.add_scalar(f"loss/ssim_{i+1}", sv, global_step)
                writer.add_scalar("loss/depth", (w_depth * ld).item(), global_step)
                writer.add_scalar("loss/ray", (w_ray * lr_val).item(), global_step)
                writer.add_scalar("loss/cam", (w_cam * lc_val).item(), global_step)
                writer.add_scalar("loss/grad", (w_grad * lg).item(), global_step)
                writer.add_scalar("grad_norm", grad_norm.item(), global_step)
                writer.add_scalar("lr", cur_lr, global_step)

        avg_loss = epoch_loss / len(dataloader)
        elapsed = time.time() - t0
        if is_main:
            print(f"Epoch {epoch} done — avg loss {avg_loss:.6f} — {elapsed:.1f}s")

        # ---- Validation ----
        eval_interval = getattr(cfg.logging, "eval_interval", 1)
        do_eval = ((epoch + 1) % eval_interval == 0) or (epoch == tc.total_epochs - 1)

        if do_eval:
            student.eval()
            val_loss_sum = 0.0
            val_feat_sum = [0.0] * 4
            val_ssim_sum = [0.0] * 4
            val_depth_sum = 0.0
            val_ray_sum = 0.0
            val_cam_sum = 0.0
            val_grad_sum = 0.0
            val_steps = 0
            absrel_sum = 0.0
            delta_correct = 0
            n_metric_pixels = 0
            delta_thr = getattr(cfg.logging, "delta_threshold", 1.25)

            with torch.no_grad():
                for val_batch in val_dataloader:
                    val_images = val_batch["image"].to(device, non_blocking=True)
                    val_images_clean = val_batch["image_clean"].to(device, non_blocking=True)
                    t_out = teacher.predict_all(val_images_clean)

                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        v_out, s_feats = student(val_images, return_distill_feats=True)
                        v_feat_loss, v_per_level = feature_distillation_loss(
                            s_feats, t_out["features"], weights=feat_weights,
                        )
                        v_ld = log_depth_distill_loss(
                            v_out.depth.float(), t_out["depth"],
                        )
                        v_lr = ray_distill_loss(
                            v_out.ray.float(), t_out["ray"],
                        )
                        v_lc = cam_distill_loss(
                            v_out.cam["pose_enc"].float(), t_out["cam_params"],
                            w_t=w_cam_t, w_q=w_cam_q, w_fov=w_cam_fov,
                        )
                        v_lg = grad_distill_loss(
                            v_out.depth.float(), t_out["depth"],
                        )
                        v_loss = (
                            v_feat_loss
                            + w_depth * v_ld
                            + w_ray * v_lr
                            + w_cam * v_lc
                            + w_grad * v_lg
                        )

                    val_loss_sum += v_loss.item()
                    for i, lv in enumerate(v_per_level):
                        val_feat_sum[i] += lv
                    for i, (sf, tf) in enumerate(zip(s_feats, t_out["features"])):
                        sf_a = F.interpolate(
                            sf.float(), size=tf.shape[2:],
                            mode="bilinear", align_corners=False,
                        )
                        val_ssim_sum[i] += _compute_feature_ssim(sf_a, tf.detach().float())
                    val_depth_sum += (w_depth * v_ld).item()
                    val_ray_sum += (w_ray * v_lr).item()
                    val_cam_sum += (w_cam * v_lc).item()
                    val_grad_sum += (w_grad * v_lg).item()
                    val_steps += 1

                    # GT metrics (median-scaled alignment)
                    pred_d = v_out.depth.float()
                    gt_d = val_batch["depth_normalized"].to(device, non_blocking=True)
                    vmask = val_batch["valid_mask"].to(device, non_blocking=True)
                    metric_mask = vmask & (gt_d > 1e-3)
                    n = metric_mask.sum().item()
                    if n > 0:
                        pred_v = pred_d[metric_mask]
                        gt_v = gt_d[metric_mask]
                        s = gt_v.median() / pred_v.median().clamp(min=1e-8)
                        pred_aligned = pred_v * s
                        absrel_sum += ((pred_aligned - gt_v).abs() / gt_v).sum().item()
                        ratio = torch.max(pred_aligned / gt_v, gt_v / pred_aligned)
                        delta_correct += (ratio < delta_thr).sum().item()
                        n_metric_pixels += n

            # ---- All-reduce val metrics across GPUs ----
            if is_distributed:
                val_loss_sum = reduce_scalar(val_loss_sum, device)
                val_steps = int(reduce_scalar(val_steps, device))
                for i in range(4):
                    val_feat_sum[i] = reduce_scalar(val_feat_sum[i], device)
                    val_ssim_sum[i] = reduce_scalar(val_ssim_sum[i], device)
                val_depth_sum = reduce_scalar(val_depth_sum, device)
                val_ray_sum = reduce_scalar(val_ray_sum, device)
                val_cam_sum = reduce_scalar(val_cam_sum, device)
                val_grad_sum = reduce_scalar(val_grad_sum, device)
                absrel_sum = reduce_scalar(absrel_sum, device)
                delta_correct = reduce_scalar(delta_correct, device)
                n_metric_pixels = int(reduce_scalar(n_metric_pixels, device))

            val_avg = val_loss_sum / max(val_steps, 1)
            val_feat_avg = sum(val_feat_sum) / max(val_steps, 1)
            val_absrel = absrel_sum / max(n_metric_pixels, 1)
            val_delta = delta_correct / max(n_metric_pixels, 1)

            if is_main:
                ssim_avg = [val_ssim_sum[i] / max(val_steps, 1) for i in range(4)]
                feat_per = [val_feat_sum[i] / max(val_steps, 1) for i in range(4)]
                ssim_str = " ".join(f"S{i+1}={ssim_avg[i]:.4f}" for i in range(4))
                feat_str = " ".join(f"L{i+1}={feat_per[i]:.4f}" for i in range(4))
                print(f"  SSIM: {ssim_str}")
                print(
                    f"  Val loss: {val_avg:.6f} | feat: {val_feat_avg:.6f} | {feat_str} | "
                    f"D={val_depth_sum / max(val_steps, 1):.4f} "
                    f"R={val_ray_sum / max(val_steps, 1):.4f} "
                    f"C={val_cam_sum / max(val_steps, 1):.4f} "
                    f"G={val_grad_sum / max(val_steps, 1):.4f} | "
                    f"AbsRel={val_absrel:.4f} δ1={val_delta:.4f}"
                )
                writer.add_scalar("val/loss", val_avg, global_step)
                writer.add_scalar("val/feat", val_feat_avg, global_step)
                for i in range(4):
                    writer.add_scalar(f"val/level_{i+1}", feat_per[i], global_step)
                    writer.add_scalar(f"val/ssim_{i+1}", ssim_avg[i], global_step)
                writer.add_scalar("val/depth", val_depth_sum / max(val_steps, 1), global_step)
                writer.add_scalar("val/ray", val_ray_sum / max(val_steps, 1), global_step)
                writer.add_scalar("val/cam", val_cam_sum / max(val_steps, 1), global_step)
                writer.add_scalar("val/grad", val_grad_sum / max(val_steps, 1), global_step)
                writer.add_scalar("val/absrel", val_absrel, global_step)
                writer.add_scalar(f"val/delta_{delta_thr}", val_delta, global_step)

            # --- Early stopping (all ranks compute identically after all-reduce) ---
            loss_improved = val_avg < best_loss
            if loss_improved:
                best_loss = val_avg
            es_counter = 0 if loss_improved else es_counter + 1

            # --- Save best checkpoint (rank 0 only) ---
            if is_main and loss_improved:
                best_path = log_dir / "best_loss.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_loss": best_loss,
                        "es_counter": es_counter,
                        "model": raw_student.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    },
                    best_path,
                )
                print(f"  New best loss={best_loss:.6f} → {best_path}")

            # --- Early stop break ---
            if is_main:
                print(f"  Early stop: patience={es_counter}/{es_patience}")
            if es_counter >= es_patience:
                if is_main:
                    print(
                        f"Early stopping at epoch {epoch} "
                        f"(no improvement for {es_patience} evals)"
                    )
                break

        # Save periodic checkpoint (rank 0 only)
        if is_main and (
            (epoch + 1) % cfg.logging.save_interval == 0
            or epoch == tc.total_epochs - 1
        ):
            ckpt_path = log_dir / f"ckpt_epoch{epoch:04d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_loss": best_loss,
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
