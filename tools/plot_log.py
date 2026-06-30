"""
Parse a train.log and plot training loss curves + eval metric curves.

python tools/plot_log.py work_dirs/v0_v1/20260619_133112/train.log

python tools/plot_log.py --phase-1 work_dirs/v1_tenth_phase1/20260623_075545/train.log
"""

import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_log(log_path):
    train_steps = []  # (global_step, epoch, step_in_epoch, total_steps, loss, L1-4, S1-4, D, R, P, C, G, LR)
    val_epochs = []   # (epoch, val_loss, feat, L1-4, absrel, delta1)
    val_ssim = []     # (epoch, S1-4)
    epoch_summary = []  # (epoch, avg_loss, time_s)

    step_pattern = re.compile(
        r'\[Epoch (\d+)/\d+\] Step (\d+)/(\d+) \| Loss ([\d.]+) \|'
        r' L1=([\d.]+) L2=([\d.]+) L3=([\d.]+) L4=([\d.]+) \|'
        r' S1=([\d.]+) S2=([\d.]+) S3=([\d.]+) S4=([\d.]+)'
        r'(?: \| D=([\d.]+) R=([\d.]+) P=([\d.]+) C=([\d.]+) G=([\d.]+))?'
        r'(?: \| GN ([\d.]+))?'
        r' \| LR ([\d.eE+-]+)'
    )

    val_pattern = re.compile(
        r'Val loss: ([\d.]+) \| feat: ([\d.]+) \|'
        r' L1=([\d.]+) L2=([\d.]+) L3=([\d.]+) L4=([\d.]+)'
        r'(?: \| AbsRel=([\d.]+) δ1=([\d.]+))?'
    )

    ssim_pattern = re.compile(
        r'SSIM: S1=([\d.]+) S2=([\d.]+) S3=([\d.]+) S4=([\d.]+)'
    )

    epoch_done_pattern = re.compile(
        r'Epoch (\d+) done — avg loss ([\d.]+) — ([\d.]+)s'
    )

    global_step = 0
    current_epoch = -1

    with open(log_path) as f:
        for line in f:
            m = step_pattern.search(line)
            if m:
                epoch = int(m.group(1))
                step = int(m.group(2))
                total = int(m.group(3))
                if epoch != current_epoch:
                    current_epoch = epoch
                global_step += 1
                loss = float(m.group(4))
                L = [float(m.group(i)) for i in range(5, 9)]
                S = [float(m.group(i)) for i in range(9, 13)]
                D = float(m.group(13)) if m.group(13) else None
                R = float(m.group(14)) if m.group(14) else None
                P = float(m.group(15)) if m.group(15) else None
                C = float(m.group(16)) if m.group(16) else None
                G = float(m.group(17)) if m.group(17) else None
                GN = float(m.group(18)) if m.group(18) else None
                lr = float(m.group(19))
                train_steps.append({
                    "global_step": global_step,
                    "epoch": epoch, "step": step, "total": total,
                    "loss": loss,
                    "L": L, "S": S,
                    "D": D, "R": R, "P": P, "C": C, "G": G,
                    "GN": GN,
                    "lr": lr,
                })
                continue

            m = epoch_done_pattern.search(line)
            if m:
                epoch_summary.append({
                    "epoch": int(m.group(1)),
                    "avg_loss": float(m.group(2)),
                    "time": float(m.group(3)),
                })
                continue

            m = ssim_pattern.search(line)
            if m:
                val_ssim.append([float(m.group(i)) for i in range(1, 5)])
                continue

            m = val_pattern.search(line)
            if m:
                ep = len(val_epochs)
                if epoch_summary:
                    ep = epoch_summary[-1]["epoch"]
                entry = {
                    "epoch": ep,
                    "val_loss": float(m.group(1)),
                    "feat": float(m.group(2)),
                    "L": [float(m.group(i)) for i in range(3, 7)],
                }
                if m.group(7):
                    entry["absrel"] = float(m.group(7))
                    entry["delta1"] = float(m.group(8))
                val_epochs.append(entry)

    # attach ssim to val_epochs
    for i, ve in enumerate(val_epochs):
        if i < len(val_ssim):
            ve["S"] = val_ssim[i]

    return train_steps, val_epochs, epoch_summary


def plot_all(train_steps, val_epochs, epoch_summary, out_dir):
    if not train_steps:
        print("No training steps found in log.")
        return

    steps = [s["global_step"] for s in train_steps]
    has_task = train_steps[0]["D"] is not None
    has_gn = train_steps[0]["GN"] is not None
    has_val = len(val_epochs) > 0
    has_depth_metrics = has_val and "absrel" in val_epochs[0]
    has_ssim = has_val and "S" in val_epochs[0]

    # Count rows: 3 train rows (if has_task or has_gn) or 2, plus val rows
    train_rows = 3 if (has_task or has_gn) else 2
    val_rows = 0
    if has_val:
        val_rows = 1  # Feature Losses L1-L4
        if has_ssim:
            val_rows += 1  # SSIM
        if has_depth_metrics:
            val_rows += 1  # AbsRel + δ1
    n_rows = train_rows + val_rows

    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 5 * n_rows))
    fig.suptitle("Training & Validation Curves", fontsize=14)

    # =====================================================================
    # Training section
    # =====================================================================

    # (0,0) Total loss
    ax = axes[0, 0]
    ax.plot(steps, [s["loss"] for s in train_steps], alpha=0.3, linewidth=0.5, color="C0")
    window = min(20, len(steps) // 5 + 1)
    if window > 1:
        kernel = np.ones(window) / window
        smoothed = np.convolve([s["loss"] for s in train_steps], kernel, mode="valid")
        ax.plot(steps[window-1:], smoothed, color="C0", linewidth=1.5, label="smoothed")
    ax.set_ylabel("Total Loss")
    ax.set_xlabel("Step")
    ax.set_title("Total Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (0,1) Feature losses L1-L4
    ax = axes[0, 1]
    for i in range(4):
        vals = [s["L"][i] for s in train_steps]
        ax.plot(steps, vals, alpha=0.5, linewidth=0.8, label=f"L{i+1}")
    ax.set_ylabel("Feature Loss")
    ax.set_xlabel("Step")
    ax.set_title("Feature Distillation Losses (L1–L4)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,0) SSIM S1-S4
    ax = axes[1, 0]
    for i in range(4):
        vals = [s["S"][i] for s in train_steps]
        ax.plot(steps, vals, alpha=0.5, linewidth=0.8, label=f"S{i+1}")
    ax.set_ylabel("SSIM")
    ax.set_xlabel("Step")
    ax.set_title("Feature SSIM (S1–S4)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,1) Learning rate
    ax = axes[1, 1]
    ax.plot(steps, [s["lr"] for s in train_steps], color="C3", linewidth=1)
    ax.set_ylabel("LR")
    ax.set_xlabel("Step")
    ax.set_title("Learning Rate Schedule")
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    ax.grid(True, alpha=0.3)

    # (2,0) Task losses D/R/P/C/G  or  Grad Norm
    if has_task:
        ax = axes[2, 0]
        for key, label, color in [("D", "Depth", "C0"), ("R", "Ray", "C1"),
                                   ("P", "Point", "C2"), ("C", "Cam", "C3"),
                                   ("G", "Grad", "C4")]:
            vals = [s[key] for s in train_steps if s[key] is not None]
            if vals:
                ax.plot(steps[:len(vals)], vals, alpha=0.5, linewidth=0.8, label=label, color=color)
        ax.set_ylabel("Task Loss")
        ax.set_xlabel("Step")
        ax.set_title("Task Losses (D/R/P/C/G)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    elif has_gn:
        ax = axes[2, 0]
        gn_vals = [s["GN"] for s in train_steps]
        ax.plot(steps, gn_vals, alpha=0.3, linewidth=0.5, color="C4")
        window = min(20, len(steps) // 5 + 1)
        if window > 1:
            kernel = np.ones(window) / window
            smoothed = np.convolve(gn_vals, kernel, mode="valid")
            ax.plot(steps[window-1:], smoothed, color="C4", linewidth=1.5, label="smoothed")
        ax.set_ylabel("Grad Norm")
        ax.set_xlabel("Step")
        ax.set_title("Gradient Norm")
        ax.legend()
        ax.grid(True, alpha=0.3)

    if has_task or has_gn:
        # (2,1) Epoch avg loss
        ax = axes[2, 1]
        if epoch_summary:
            ep = [e["epoch"] for e in epoch_summary]
            ax.plot(ep, [e["avg_loss"] for e in epoch_summary], "o-", color="C0", markersize=3)
            ax.set_ylabel("Avg Loss")
            ax.set_xlabel("Epoch")
            ax.set_title("Epoch Average Loss")
            ax.grid(True, alpha=0.3)

    # =====================================================================
    # Validation section
    # =====================================================================
    if not has_val:
        plt.tight_layout()
        p = os.path.join(out_dir, "curves.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {p}")
        return

    vep = [v["epoch"] for v in val_epochs]
    row = train_rows

    # Val Feature Losses L1-L4
    ax = axes[row, 0]
    for i in range(4):
        vals = [v["L"][i] for v in val_epochs]
        best = min(vals)
        best_ep = vep[vals.index(best)]
        ax.plot(vep, vals, "o-", markersize=3, label=f"L{i+1} (best={best:.4f} @ep{best_ep})")
    ax.set_ylabel("Feature Loss")
    ax.set_xlabel("Epoch")
    ax.set_title("Val Feature Losses (L1–L4)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Val Feature Loss total
    ax = axes[row, 1]
    feat_vals = [v["feat"] for v in val_epochs]
    best_feat = min(feat_vals)
    best_feat_ep = vep[feat_vals.index(best_feat)]
    ax.plot(vep, feat_vals, "o-", color="C1", markersize=4,
            label=f"feat (best={best_feat:.4f} @ep{best_feat_ep})")
    ax.set_ylabel("Feat Loss")
    ax.set_xlabel("Epoch")
    ax.set_title("Val Feature Loss")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    row += 1

    # Val SSIM
    if has_ssim:
        ax = axes[row, 0]
        for i in range(4):
            vals = [v["S"][i] for v in val_epochs]
            best = max(vals)
            best_ep = vep[vals.index(best)]
            ax.plot(vep, vals, "o-", markersize=3, label=f"S{i+1} (best={best:.4f} @ep{best_ep})")
        ax.set_ylabel("SSIM")
        ax.set_xlabel("Epoch")
        ax.set_title("Val Feature SSIM (S1–S4)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[row, 1]
        avg_ssim = [np.mean(v["S"]) for v in val_epochs]
        best_ms = max(avg_ssim)
        best_ms_ep = vep[avg_ssim.index(best_ms)]
        ax.plot(vep, avg_ssim, "o-", color="C2", markersize=4,
                label=f"mean SSIM (best={best_ms:.4f} @ep{best_ms_ep})")
        ax.set_ylabel("Mean SSIM")
        ax.set_xlabel("Epoch")
        ax.set_title("Val Mean SSIM")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        row += 1

    # Depth metrics: AbsRel + delta1
    if has_depth_metrics:
        ax = axes[row, 0]
        absrel_vals = [v["absrel"] for v in val_epochs]
        best_ar = min(absrel_vals)
        best_ar_ep = vep[absrel_vals.index(best_ar)]
        ax.plot(vep, absrel_vals, "o-", color="C3", markersize=4,
                label=f"AbsRel (best={best_ar:.4f} @ep{best_ar_ep})")
        ax.set_ylabel("AbsRel")
        ax.set_xlabel("Epoch")
        ax.set_title("Val AbsRel (lower is better)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[row, 1]
        delta_vals = [v["delta1"] for v in val_epochs]
        best_d1 = max(delta_vals)
        best_d1_ep = vep[delta_vals.index(best_d1)]
        ax.plot(vep, delta_vals, "o-", color="C2", markersize=4,
                label=f"δ1 (best={best_d1:.4f} @ep{best_d1_ep})")
        ax.set_ylabel("δ1")
        ax.set_xlabel("Epoch")
        ax.set_title("Val δ1 (higher is better)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p = os.path.join(out_dir, "curves.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {p}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log", type=str, help="Path to train.log")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Output directory (default: same as log)")
    p.add_argument("--phase-1", action="store_true",
                   help="Phase-1 log format (GN instead of task losses)")
    args = p.parse_args()

    out_dir = args.out_dir or os.path.dirname(args.log)
    train_steps, val_epochs, epoch_summary = parse_log(args.log)
    print(f"Parsed: {len(train_steps)} train steps, {len(val_epochs)} val evals, {len(epoch_summary)} epoch summaries")
    plot_all(train_steps, val_epochs, epoch_summary, out_dir)


if __name__ == "__main__":
    main()
