<div align="center">

# MobileNetV2 Monocular Depth Estimation

**A Lightweight Monocular Depth Estimation Framework via Knowledge Distillation**

Distills a frozen Depth Anything 3 (DA3, ViT-G) teacher into a ~52M-parameter MobileNetV2 student,
preserving depth estimation capability while drastically reducing inference cost.

<sub>Python 3.11 · PyTorch (cu128) · DDP · TartanGround / LMDB</sub>

[中文](README_CHN.md) · [Design Doc PLAN_v2](PLAN_v2.md)

</div>

---

## 1. Overview

This project is a **knowledge distillation training framework** that transfers the capabilities of
the large monocular depth model **Depth Anything 3 (NESTED-GIANT-LARGE-1.1)** into a lightweight
**MobileNetV2** student network.

- **Teacher**: DA3 anyview branch (ViT-G, fully frozen, zero source-code modifications, features extracted via forward hooks).
- **Student**: `FastDepthModel` (~52M params), directly outputs metric depth and jointly predicts ray fields and camera parameters.
- **Data**: TartanGround (UE synthetic, ~1.11M frames), stored in LMDB; ray/camera GT computed online.
- **Input resolution**: **644 × 476** (multiple of 14, satisfying DA3 `patch_size=14`).

### Key Features

- **Two-phase training**: Phase 1 — feature alignment only; Phase 2 — full task loss on top.
- **Zero-transform teacher distillation**: All alignment happens on the student side (1×1 channel projection + interpolation to teacher resolution).
- **Switchable sky strategies**: `v0` (ignore sky) and `v1` (predict sky), each with its own dataset + loss implementation.
- **Multi-GPU DDP training** with bfloat16 mixed precision; supports checkpoint resumption and warm-start.

---

## 2. System Architecture

```
Image [B, 3, 476, 644]
    │
    ├─▶ Teacher (DA3, frozen, no_grad) ─▶ T1–T4  (detached, target features)
    │                                        │
    └─▶ Student (FastDepthModel)             │
           ├─ MobileNetV2 Backbone ─▶ f1–f4  │
           ├─ StudentDualDPT                 │
           │    ├─ projects (1×1) ─▶ P1–P4 ──┴─ Feature distillation MSE   (Phase 1 & 2)
           │    └─ FPN + heads    ─▶ depth, ray                            (Phase 2)
           └─ CameraHead          ─▶ t, qvec, fov                          (Phase 2)
```

| Phase | Objective | Loss |
|-------|-----------|------|
| Phase 1 | Feature-space alignment | Feature distillation MSE(P1–P4, T1–T4) |
| Phase 2 | Full-task distillation | Feature MSE + Depth + Ray + Point Cloud + Camera + Gradient |

---

## 3. Model Architecture

### 3.1 Student Model `FastDepthModel`

```
FastDepthModel
├── MobileNetV2Backbone (ImageNet pretrained)
│   ├── Stage1 features[0:4]   → [B,  24, 119, 161]  stride 4
│   ├── Stage2 features[4:7]   → [B,  32,  60,  81]  stride 8
│   ├── Stage3 features[7:14]  → [B,  96,  30,  41]  stride 16
│   └── Stage4 features[14:18] → [B, 320,  15,  21]  stride 32
│
├── StudentDualDPT (adapted from DA3 DualDPT)
│   ├── projects : 4× Conv2d(1×1) → channels [256, 512, 1024, 1024]  (align to teacher)
│   ├── sincos positional encoding
│   ├── FPN scratch + dual fusion chain (main / aux independent)
│   ├── main head → depth + depth_conf        (softplus / softplus+1)
│   └── aux  head → ray [·,·,6] + ray_conf
│
└── StudentCameraHead
    └── AdaptiveAvgPool → Linear(320→3072) → CameraDec → { t[3], qvec[4], fov[2] }
```

### 3.2 Parameter Distribution (total 52.08M)

| Module | Params | Share |
|--------|-------:|------:|
| MobileNetV2 backbone | **1.81M** | **3.5%** |
| StudentDualDPT head | **30.37M** | **58.3%** |
| &nbsp;&nbsp;└─ projects (1×1 convs) | 0.45M | 0.9% |
| &nbsp;&nbsp;└─ scratch (FPN fusion + output heads) | 29.92M | 57.5% |
| StudentCameraHead | **19.89M** | **38.2%** |
| **Total** | **52.08M** | 100% |

> The backbone accounts for only **3.5%**; the vast majority of parameters reside in the heads
> (DPT fusion + camera head, together ~**96.5%**). The camera head alone takes ~**38%** due to
> `Linear(320→3072)` + MLP(3072²×2) — if you need to compress the model, the heads (especially
> the camera head) offer far greater returns than the backbone.

### 3.3 Teacher Model DA3 NESTED-GIANT-LARGE-1.1

- Loads only the anyview branch (skips the metric branch, saving ~30% inference time); fully frozen.
- Forward hooks attached to DualDPT's 4 `resize_layers`, capturing features at channels `[256, 512, 1024, 1024]`.
- **No alignment transform is applied to teacher features** — only `detach + float32` before serving as distillation targets.

---

## 4. Data & Preprocessing

**TartanGround** (739 trajectories, ~1.11M frames), LMDB storage.

| Item | Value |
|------|-------|
| Full training set | 1,079,955 samples |
| Full validation set | 127,762 samples |
| Original / target resolution | 640×640 / 476×644 |

- **Spatial transform**: 640×640 → vertical center crop (82px top & bottom) → 640×476 → horizontal reflect pad (2px each side) → **644×476**.
- **Online GT**: From pose (c2w NED) + K, converted to OpenCV/DA3 convention via `R_NED_CV = [[0,0,1],[1,0,0],[0,1,0]]`, computing ray fields `(238,322,6)` and camera parameters `[t(3), qvec(4), fov(2)]`.
- **Normalization / clipping**: ImageNet mean-std; `depth_cap = 100m`.
- **Augmentation**: Random horizontal flip (geometric quantities flipped accordingly) + ColorJitter (image only).

---

## 5. Distillation & Losses

**Feature distillation** (per-level MSE, student interpolated to teacher resolution before computing):

| Level | Channels | Weight |
|-------|:--------:|:------:|
| L1 / L2 / L3 / L4 | 256 / 512 / 1024 / 1024 | 0.5 / 0.5 / 1.0 / 2.0 |

**Task losses** (Phase 2, following the DA3 paper, `λc = 0.2`):

| Loss | Formula | Description |
|------|---------|-------------|
| `L_depth` | `conf·|pred−gt| − λc·log(conf)` | Confidence-weighted L1 depth |
| `L_ray` | `conf·|pred−gt| − λc·log(conf)` | Confidence-weighted L1 ray (half-resolution) |
| `L_point` | `|pred_point − gt_point|` | Depth + camera → world point cloud L1 |
| `L_cam` | `w_t|Δt| + w_q|Δq| + w_fov|Δfov|` | Camera parameter L1 |
| `L_grad` | `|∇x diff| + |∇y diff|` | Depth gradient L1 |

---

## 6. Sky Handling Strategies: v0 & v1

TartanGround sky/distant-background depth ≥ 100m. This project provides two switchable implementations:

| Strategy | Modules | Mechanism | Meaning |
|----------|---------|-----------|---------|
| **v0 — Ignore sky** | `objective_losses.py` + `tartanground_lmdb_dataset.py` | `valid_mask = depth < 100m`; per-pixel loss computed only on non-sky pixels | Sky excluded from supervision and evaluation |
| **v1 — Predict sky** | `objective_losses_v1.py` + `tartanground_lmdb_dataset_v1.py` | No mask; sky pixels filled with the image's `max_valid` normalized depth; full-image participation | Sky supervised as "farthest point", producing a continuous usable output |

---

## 7. Experimental Results

> The following are exploratory experiments on a **1/10 data subset** (107,995 train / 12,776 val),
> evaluated on the full validation set with the best_delta checkpoint (~3.35 billion valid pixels, depth<100m).
> The student directly outputs metric depth; DA3 teacher outputs relative depth aligned to GT via
> **per-image median scaling**.

| Metric | Student **v0** (ignore sky) | Student **v1** (predict sky) | DA3 Teacher (median-scaled) | Direction |
|--------|:---:|:---:|:---:|:---:|
| AbsRel | **0.3190** | 0.4028 | 0.2462 | ↓ |
| SqRel | **0.2056** | 0.8844 | 0.1924 | ↓ |
| RMSE | **0.5953** | 0.8035 | 0.8835 | ↓ |
| log RMSE | **0.4034** | 0.4429 | 0.4672 | ↓ |
| δ₁ (<1.25) | **0.6001** | 0.5806 | 0.6840 | ↑ |
| δ₂ (<1.25²) | **0.8339** | 0.8195 | 0.8311 | ↑ |
| δ₃ (<1.25³) | **0.9219** | 0.9103 | 0.9007 | ↑ |

- The student outperforms or matches median-scaled DA3 on **RMSE / log RMSE / δ₂ / δ₃** (due to directly learning metric depth), but still lags on AbsRel / δ₁.
- Subset experiments are not pure "sky handling" ablations — different runs also involve hyperparameter differences such as learning rate. The table above is best treated as an **order-of-magnitude reference** rather than a strict A/B conclusion.

Full reports: `work_dirs/v0_tenth_v1/report.md`, `work_dirs/v1_tenth/20260622_062707/report.md`.

---

## 8. Installation

```bash
bash setup_env.sh
# Or manually:
conda create -n fast_mono_depth_lrd python=3.11 -y
conda activate fast_mono_depth_lrd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install timm einops omegaconf addict safetensors huggingface_hub
pip install opencv-python tensorboard tqdm pillow numpy scipy matplotlib lmdb xformers
```

The DA3 teacher source code **requires no modification** — simply point `teacher.da3_src` / `teacher.model_dir` in the config to your local paths.

---

## 9. Usage

### Training (multi-GPU DDP)

```bash
bash train_v1.sh                 # Fresh training with configs/v1.yaml
RESUME=auto bash train_v1.sh     # Resume the latest run

# Or directly with torchrun
export PYTHONPATH="$(pwd):$PYTHONPATH"
torchrun --nproc_per_node=4 --master_port=29501 tools/train_v1.py --config configs/v1.yaml
```

Phase switching: change `loss.phase` in the config (`1` = feature distillation only, `2` = full task).

### Evaluation

```bash
export PYTHONPATH="$(pwd):$PYTHONPATH"
python tools/eval.py \
    --checkpoint work_dirs/v0_tenth_v1/20260618_154428/best_delta.pt \
    --config     configs/v0_tenth_v1.yaml \
    --index_json data/annotations/tartanground_val_tenth.json \
    --eval_da3            # Also evaluate DA3 teacher (median-scaled alignment)
# Append --no-valid-mask to switch to v1 full-pixel evaluation mode
```

### Inference

```bash
python tools/infer.py         # Single image / directory
python tools/infer_video.py   # Video
```

---

## 10. Directory Structure

```
fast_mono_depth/
├── configs/                              # Nested YAML training configs (v0*/v1* series)
├── models/
│   ├── backbone.py                       # MobileNetV2 four-scale features
│   ├── student_dpt.py                    # Student DualDPT (main depth + aux ray)
│   ├── cam_head.py                       # Camera parameter head
│   └── fast_depth_model.py               # Full student model assembly
├── distillation/
│   ├── teacher.py                        # DA3 teacher wrapper (frozen + hooks)
│   ├── losses.py                         # Feature distillation MSE
│   ├── objective_losses.py               # Phase 2 task loss — v0 (valid_mask, ignore sky)
│   └── objective_losses_v1.py            # Phase 2 task loss — v1 (no mask, predict sky)
├── data/                                 # Datasets & indices (LMDB, not version-controlled)
│   ├── tartanground_lmdb_dataset.py      # v0 dataset (returns valid_mask)
│   └── tartanground_lmdb_dataset_v1.py   # v1 dataset (sky → max, no valid_mask)
├── tools/
│   ├── train.py / train_v1.py            # Training scripts (single-GPU / DDP)
│   ├── eval.py                           # Standalone evaluation (optional DA3 comparison)
│   ├── infer.py / infer_video.py         # Inference & visualization
│   └── build_lmdb_dataset/               # LMDB construction tools
├── train_v1.sh / train_v1_phase1.sh      # Launch scripts (resume / warm-start)
├── setup_env.sh
└── PLAN_v2.md                            # Detailed design document
```

> Note: `data/`, `work_dirs/`, `dataset_process/` are large (LMDB ~594GB, weights, visualizations) and excluded via `.gitignore`.

---

## 11. Limitations & Future Work

- **Data volume**: Current results use only a 1/10 subset (107K frames); the full 1.08M-frame set remains to be trained.
- **Under-convergence**: Most runs were early-stopped/interrupted around epoch 30 (100 planned).
- **Weak shallow feature alignment**: L1 SSIM ≈ 0.18, stemming from the architectural gap between MobileNetV2 and DINOv2/Giant; phased Phase 1 → Phase 2 training may help.
- **Synthetic data only**: Real-world generalization (NYU/KITTI/ETH3D) is yet to be evaluated.
- **High learning rate instability**: LR=2e-4 triggers depth-confidence collapse and SqRel degradation; LR ≤ 1e-4 with gradient clipping is recommended.

---

## Acknowledgements

- [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) — Teacher model and DualDPT/loss design reference.
- [TartanGround / TartanAir](https://github.com/castacks/tartanair_tools) — Training data source.
- [torchvision MobileNetV2](https://pytorch.org/vision/stable/models/mobilenetv2.html) — Student backbone network.
