<div align="center">

# MobileNetV2 Monocular Depth Estimation

**基于知识蒸馏的轻量级单目深度估计框架**

以冻结的 Depth Anything 3 (DA3, ViT-G) 为教师，蒸馏出约 52M 参数的 MobileNetV2 学生模型，
在保留深度估计能力的同时大幅降低推理成本。

<sub>Python 3.11 · PyTorch (cu128) · DDP · TartanGround / LMDB</sub>

[English](README.md) · [设计文档 PLAN_v2](PLAN_v2.md)

</div>

---

## 1. 概述

本项目是一个 **知识蒸馏 (Knowledge Distillation) 训练框架**，把大型单目深度模型
**Depth Anything 3 (NESTED-GIANT-LARGE-1.1)** 的能力迁移到轻量的 **MobileNetV2** 学生网络。

- 教师：DA3 anyview 分支（ViT-G，全程冻结，源码零修改，forward hook 取特征）。
- 学生：`FastDepthModel`（≈ 52M 参数），直接输出 metric 深度，并联合预测射线场与相机参数。
- 数据：TartanGround（UE 合成，约 111 万帧），LMDB 存储，射线/相机 GT 在线计算。
- 输入分辨率：**644 × 476**（14 的倍数，满足 DA3 patch_size=14）。

### 主要特性

- **两阶段训练**：Phase 1 仅特征对齐，Phase 2 叠加全任务损失。
- **教师特征零变换蒸馏**：对齐动作全部在学生侧完成（1×1 升通道 + 插值到教师尺寸）。
- **可切换的天空策略**：`v0`（忽略天空）与 `v1`（预测天空）两套 dataset + loss 实现。
- **多卡 DDP 训练**，bfloat16 混合精度，支持断点续训与暖启动。

---

## 2. 系统架构

```
Image [B, 3, 476, 644]
    │
    ├─▶ Teacher (DA3, frozen, no_grad) ─▶ T1–T4  (detached, 目标特征)
    │                                        │
    └─▶ Student (FastDepthModel)             │
           ├─ MobileNetV2 Backbone ─▶ f1–f4  │
           ├─ StudentDualDPT                 │
           │    ├─ projects (1×1) ─▶ P1–P4 ──┴─ 特征蒸馏 MSE      (Phase 1 & 2)
           │    └─ FPN + heads    ─▶ depth, ray                   (Phase 2)
           └─ CameraHead          ─▶ t, qvec, fov                 (Phase 2)
```

| 阶段 | 目标 | 损失 |
|------|------|------|
| Phase 1 | 特征空间对齐 | 特征蒸馏 MSE(P1–P4, T1–T4) |
| Phase 2 | 全任务蒸馏 | 特征 MSE + 深度 + 射线 + 点云 + 相机 + 梯度 |

---

## 3. 模型结构

### 3.1 学生模型 `FastDepthModel`

```
FastDepthModel
├── MobileNetV2Backbone (ImageNet 预训练)
│   ├── Stage1 features[0:4]   → [B,  24, 119, 161]  stride 4
│   ├── Stage2 features[4:7]   → [B,  32,  60,  81]  stride 8
│   ├── Stage3 features[7:14]  → [B,  96,  30,  41]  stride 16
│   └── Stage4 features[14:18] → [B, 320,  15,  21]  stride 32
│
├── StudentDualDPT (改编自 DA3 DualDPT)
│   ├── projects : 4× Conv2d(1×1) → 通道 [256, 512, 1024, 1024]  （对齐教师）
│   ├── sincos 位置编码
│   ├── FPN scratch + 双 fusion chain（main / aux 独立）
│   ├── main head → depth + depth_conf        （softplus / softplus+1）
│   └── aux  head → ray [·,·,6] + ray_conf
│
└── StudentCameraHead
    └── AdaptiveAvgPool → Linear(320→3072) → CameraDec → { t[3], qvec[4], fov[2] }
```

### 3.2 参数量分布（总计 52.08M）

| 模块 | 参数量 | 占比 |
|------|-------:|-----:|
| MobileNetV2 backbone | **1.81M** | **3.5%** |
| StudentDualDPT head | **30.37M** | **58.3%** |
| &nbsp;&nbsp;└─ projects（1×1 投影） | 0.45M | 0.9% |
| &nbsp;&nbsp;└─ scratch（FPN 融合 + 输出头） | 29.92M | 57.5% |
| StudentCameraHead | **19.89M** | **38.2%** |
| **合计** | **52.08M** | 100% |

> Backbone 仅占 **3.5%**，绝大部分参数集中在 head（DPT 融合 + 相机头，合计约 **96.5%**）。
> 相机头因 `Linear(320→3072)` + MLP(3072²×2) 单独就占约 **38%**——如需压缩模型，
> head（尤其相机头）比 backbone 收益大得多。

### 3.3 教师模型 DA3 NESTED-GIANT-LARGE-1.1

- 仅加载 anyview 分支（跳过 metric 分支，省约 30% 推理时间），全程冻结。
- 在 DualDPT 的 4 个 `resize_layers` 上挂 forward hook，取通道 `[256, 512, 1024, 1024]` 的特征。
- **教师特征不做任何对齐变换**，仅 `detach + float32` 后作为蒸馏目标。

---

## 4. 数据与预处理

**TartanGround**（739 条轨迹，约 111 万帧），LMDB 存储。

| 项目 | 数值 |
|------|------|
| 完整训练集 | 1,079,955 samples |
| 完整验证集 | 127,762 samples |
| 原始 / 目标分辨率 | 640×640 / 476×644 |

- **空间变换**：640×640 → 垂直中心裁剪（上下各 82px）→ 640×476 → 水平 reflect pad（各 2px）→ **644×476**。
- **在线 GT**：由 pose(c2w NED) + K 经 `R_NED_CV = [[0,0,1],[1,0,0],[0,1,0]]` 转 OpenCV/DA3 约定，
  计算射线场 `(238,322,6)` 与相机参数 `[t(3), qvec(4), fov(2)]`。
- **归一化 / 截断**：ImageNet mean-std；`depth_cap = 100m`。
- **增强**：随机水平翻转（几何量同步翻转）+ ColorJitter（仅图像）。

---

## 5. 蒸馏与损失

**特征蒸馏**（逐级 MSE，学生插值到教师尺寸后计算）：

| 级别 | 通道 | 权重 |
|------|:----:|:----:|
| L1 / L2 / L3 / L4 | 256 / 512 / 1024 / 1024 | 0.5 / 0.5 / 1.0 / 2.0 |

**任务损失**（Phase 2，遵循 DA3 论文，`λc = 0.2`）：

| 损失 | 公式 | 说明 |
|------|------|------|
| `L_depth` | `conf·|pred−gt| − λc·log(conf)` | 置信度加权 L1 深度 |
| `L_ray` | `conf·|pred−gt| − λc·log(conf)` | 置信度加权 L1 射线（半分辨率） |
| `L_point` | `|pred_point − gt_point|` | depth + 相机 → 世界点云 L1 |
| `L_cam` | `w_t|Δt| + w_q|Δq| + w_fov|Δfov|` | 相机参数 L1 |
| `L_grad` | `|∇x diff| + |∇y diff|` | 深度梯度 L1 |

---

## 6. 天空处理策略：v0 与 v1

TartanGround 的天空/极远背景深度 ≥ 100m。本项目提供两套可切换实现：

| 策略 | 模块 | 机制 | 含义 |
|------|------|------|------|
| **v0 — 忽略天空** | `objective_losses.py` + `tartanground_lmdb_dataset.py` | `valid_mask = depth < 100m`；逐像素损失仅在非天空像素计算 | 天空不参与监督与评估 |
| **v1 — 预测天空** | `objective_losses_v1.py` + `tartanground_lmdb_dataset_v1.py` | 去掉 mask；天空归一化深度填为该图 `max_valid`；全图参与 | 天空被监督为“最远处”，输出连续可用 |

---

## 7. 实验结果

> 以下为 **1/10 数据子集**（107,995 训练 / 12,776 验证）上的探索性实验，best_delta checkpoint
> 全验证集评估（约 33.5 亿有效像素，depth<100m）。学生直接输出 metric 深度；DA3 教师输出相对深度，
> 经**逐图像中值缩放**对齐 GT。

| 指标 | Student **v0**（忽略天空） | Student **v1**（预测天空） | DA3 教师（median-scaled） | 方向 |
|------|:---:|:---:|:---:|:---:|
| AbsRel | **0.3190** | 0.4028 | 0.2462 | ↓ |
| SqRel | **0.2056** | 0.8844 | 0.1924 | ↓ |
| RMSE | **0.5953** | 0.8035 | 0.8835 | ↓ |
| log RMSE | **0.4034** | 0.4429 | 0.4672 | ↓ |
| δ₁ (<1.25) | **0.6001** | 0.5806 | 0.6840 | ↑ |
| δ₂ (<1.25²) | **0.8339** | 0.8195 | 0.8311 | ↑ |
| δ₃ (<1.25³) | **0.9219** | 0.9103 | 0.9007 | ↑ |

- 学生在 **RMSE / log RMSE / δ₂ / δ₃** 上优于或持平中值缩放后的 DA3（直接学习 metric 深度所致），
  在 AbsRel / δ₁ 上仍落后。
- 子集实验并非纯粹的“天空处理”消融，不同 run 还伴随学习率等超参差异，上表宜作**量级参考**而非严格 A/B 结论。

完整报告：`work_dirs/v0_tenth_v1/report.md`、`work_dirs/v1_tenth/20260622_062707/report.md`。

---

## 8. 安装

```bash
bash setup_env.sh
# 或手动：
conda create -n fast_mono_depth_lrd python=3.11 -y
conda activate fast_mono_depth_lrd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install timm einops omegaconf addict safetensors huggingface_hub
pip install opencv-python tensorboard tqdm pillow numpy scipy matplotlib lmdb xformers
```

教师 DA3 源码**无需修改**，在 config 的 `teacher.da3_src` / `teacher.model_dir` 指向本地路径即可。

---

## 9. 使用

### 训练（多卡 DDP）

```bash
bash train_v1.sh                 # 全新训练 configs/v1.yaml
RESUME=auto bash train_v1.sh     # 续训最近一次 run

# 或直接使用 torchrun
export PYTHONPATH="$(pwd):$PYTHONPATH"
torchrun --nproc_per_node=4 --master_port=29501 tools/train_v1.py --config configs/v1.yaml
```

阶段切换：修改 config 中 `loss.phase`（`1` = 仅特征蒸馏，`2` = 全任务）。

### 评估

```bash
export PYTHONPATH="$(pwd):$PYTHONPATH"
python tools/eval.py \
    --checkpoint work_dirs/v0_tenth_v1/20260618_154428/best_delta.pt \
    --config     configs/v0_tenth_v1.yaml \
    --index_json data/annotations/tartanground_val_tenth.json \
    --eval_da3            # 同时评估 DA3 教师（中值缩放对齐）
# 追加 --no-valid-mask 切换到 v1 全像素评估口径
```

### 推理

```bash
python tools/infer.py         # 单图 / 目录
python tools/infer_video.py   # 视频
```

---

## 10. 目录结构

```
fast_mono_depth/
├── configs/                              # 嵌套 YAML 训练配置（v0*/v1* 系列）
├── models/
│   ├── backbone.py                       # MobileNetV2 四尺度特征
│   ├── student_dpt.py                    # Student 版 DualDPT（main depth + aux ray）
│   ├── cam_head.py                       # 相机参数头
│   └── fast_depth_model.py               # 组合完整学生模型
├── distillation/
│   ├── teacher.py                        # DA3 教师封装（冻结 + hook）
│   ├── losses.py                         # 特征蒸馏 MSE
│   ├── objective_losses.py               # Phase2 任务损失 — v0（含 valid_mask，忽略天空）
│   └── objective_losses_v1.py            # Phase2 任务损失 — v1（无 mask，预测天空）
├── data/                                 # 数据集与索引（LMDB，未纳入版本管理）
│   ├── tartanground_lmdb_dataset.py      # v0 数据集（返回 valid_mask）
│   └── tartanground_lmdb_dataset_v1.py   # v1 数据集（天空填 max，无 valid_mask）
├── tools/
│   ├── train.py / train_v1.py            # 训练主脚本（单卡 / DDP）
│   ├── eval.py                           # 独立评估（可选对比 DA3）
│   ├── infer.py / infer_video.py         # 推理可视化
│   └── build_lmdb_dataset/               # LMDB 构建工具
├── train_v1.sh / train_v1_phase1.sh      # 启动脚本（含续训 / 暖启）
├── setup_env.sh
└── PLAN_v2.md                            # 详细设计文档
```

> 注：`data/`、`work_dirs/`、`dataset_process/` 体积较大（LMDB ~594GB、权重、可视化），已在 `.gitignore` 中排除。

---

## 11. 局限与后续

- **数据量**：现有结果仅用 1/10 子集（107K 帧），完整集 1.08M 帧待训。
- **未充分收敛**：多数 run 在约 30 epoch 早停/中断（计划 100）。
- **浅层特征对齐弱**：L1 SSIM ≈ 0.18，源于 MobileNetV2 与 DINOv2/Giant 架构差异；可尝试 Phase 1 → Phase 2 分阶段训练。
- **仅合成数据**：真实场景（NYU/KITTI/ETH3D）泛化性待评估。
- **高学习率不稳定**：LR=2e-4 会引发深度置信度崩溃与 SqRel 恶化；建议 LR ≤ 1e-4 并加梯度裁剪。

---

## 引用

- [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) — 教师模型与 DualDPT/损失设计参考。
- [TartanGround / TartanAir](https://github.com/castacks/tartanair_tools) — 训练数据来源。
- [torchvision MobileNetV2](https://pytorch.org/vision/stable/models/mobilenetv2.html) — 学生骨干网络。
