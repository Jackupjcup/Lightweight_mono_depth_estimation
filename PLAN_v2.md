# Fast Mono Depth — 知识蒸馏训练框架

用 MobileNetV2 student 替换 DA3 的 ViT-G backbone，通过冻结的 DA3 teacher 进行知识蒸馏，实现快速单目深度估计。

**输入分辨率**: 644×476（14 的倍数，满足 DA3 patch_size=14 要求）

---

## 目录

| 章节 | 内容 | 状态 |
|------|------|------|
| [1. 框架总览](#1-框架总览) | 整体架构、训练阶段、数据流 | — |
| [2. 项目结构 & 环境](#2-项目结构--环境) | 文件组织、Conda 环境 | ✅ |
| [3. Student 模型](#3-student-模型) | MobileNetV2 + DualDPT + CamHead | ✅ |
| [4. Teacher & 蒸馏](#4-teacher--蒸馏) | DA3 封装、Hook、MSE Loss | ✅ |
| [5. 数据预处理](#5-数据预处理--tos--lmdb) | TartanGround → LMDB 构建 | ✅ |
| [6. DataLoader & 在线 GT 计算](#6-dataloader--在线-gt-计算) | ray / cam_params / valid_mask / flip 同步 | ✅ |
| [7. 训练流程 & Config](#7-训练流程--config) | train.py + 嵌套 YAML 配置 | ✅ |
| [8. Phase 2 Loss](#8-phase-2-loss--da3-论文任务级损失函数) | LD / LM / LP / LC / Lgrad 五项任务级 loss | ✅ |
| [附录 A. DA3 参考](#附录-a-da3-参考) | Nested 双分支架构 / 推理 vs 蒸馏差异 | 参考 |

---

## 1. 框架总览

### 1.1 训练阶段

| Phase | 目标 | Loss | DataLoader | 状态 |
|-------|------|------|------------|------|
| **Phase 1** | 特征空间对齐 | MSE(P1~P4, T1~T4) | ImageFolderDataset（只需图片） | ✅ |
| **Phase 2** | 全任务蒸馏 | + depth_loss + ray_loss + cam_loss | TartanGroundLMDBDataset（全字段） | 待实现 |

### 1.2 端到端数据流

```
Image [B, 3, 476, 644]
    │
    ├──▶ Teacher (frozen, no_grad) ──▶ T1~T4 (detached)
    │                                       │
    └──▶ Student                            │
            ├── MobileNetV2  ──▶ f1~f4      │
            ├── StudentDualDPT              │
            │     ├── projects(f1~f4) ──▶ P1~P4 ─── MSE Loss ◄───┘  (Phase 1)
            │     └── FPN + heads ──▶ depth, ray                      (Phase 2)
            └── CameraHead ──▶ t, qvec, fov                          (Phase 2)
```

### 1.3 Student 输出一览

```
depth       [B, 476, 644]       绝对深度图（米制）
depth_conf  [B, 476, 644]       深度置信度
ray         [B, 238, 322, 6]    每像素 6D 方向场
ray_conf    [B, 238, 322]       ray 置信度
cam.t       [B, 3]              平移向量
cam.qvec    [B, 4]              旋转四元数
cam.fov     [B, 2]              视场角 [fov_h, fov_w]（与 DA3 一致）
proj_feats  [P1, P2, P3, P4]    蒸馏特征（训练时额外返回）
```

---

## 2. 项目结构 & 环境

### 2.1 文件组织

```
fast_mono_depth/
├── configs/
│   └── default.yaml                  # 嵌套结构训练配置
├── models/
│   ├── backbone.py                   # MobileNetV2 四层特征提取
│   ├── student_dpt.py                # Student 版 DualDPT（从 DA3 复制构建块 + 修改）
│   ├── cam_head.py                   # 相机参数预测头
│   └── fast_depth_model.py           # Student 完整模型（组合以上三者）
├── distillation/
│   ├── teacher.py                    # DA3 teacher 封装（冻结 + hook 提取特征）
│   └── losses.py                     # 特征蒸馏 MSE loss
├── data/
│   ├── dataset.py                    # Phase 1 简单图片数据集
│   ├── tartanground_lmdb_dataset.py  # Phase 2 LMDB Dataset（在线 ray 计算）
│   ├── tartanground.lmdb/            # LMDB 数据库 (~334 GB)
│   └── annotations/
│       ├── tartanground_index.json   # 完整索引
│       ├── tartanground_train.json   # train split 索引
│       └── tartanground_val.json     # val split 索引
├── tools/
│   ├── train.py                      # 主训练脚本
│   ├── preprocess.py                 # crop/pad、depth 解码、归一化工具
│   └── build_tartanground_lmdb.py    # LMDB 构建 + ray/sky 追加 + 验证
└── setup_env.sh                      # Conda 环境安装脚本
```

### 2.2 Conda 环境

`fast_mono_depth_lrd` conda 环境：
- Python 3.11, torch>=2.0+cu129, torchvision, timm, einops, omegaconf, addict, safetensors, opencv-python, tensorboard, xformers, scipy
- DA3 源码**零修改**，运行时通过 `sys.path.insert` 导入

---

## 3. Student 模型

### 3.1 整体 Flow Chart

```
                              Input Image
                           [B, 3, 476, 644]
                                  │
          ┌───────────────────────┼──────────────────────────────────────────────┐
          │                       │                                              │
          ▼                       ▼                                              │
  ┌═══ MobileNetV2 Backbone (torchvision, 19 blocks) ═══════════════════┐       │
  │                                                                      │       │
  │  features[0]: Conv2d 3→32, stride=2, BN, ReLU6                      │       │
  │               → [B, 32, 238, 322]                                    │       │
  │       │                                                              │       │
  │  features[1]: IRB (t=1, 32→16, s=1) ×1         ── S1                │       │
  │               → [B, 16, 238, 322]                                    │       │
  │       │                                                              │       │
  │  features[2]: IRB (t=6, 16→24, s=2)                                 │       │
  │  features[3]: IRB (t=6, 24→24, s=1)            ── S2  ★ 抽取 f1     │       │
  │               → [B, 24, 119, 161]   ─────────────────────────────┐   │       │
  │       │                                                          │   │       │
  │  features[4]: IRB (t=6, 24→32, s=2)                             │   │       │
  │  features[5]: IRB (t=6, 32→32, s=1)                             │   │       │
  │  features[6]: IRB (t=6, 32→32, s=1)            ── S3  ★ 抽取 f2 │   │       │
  │               → [B, 32, 60, 81]     ──────────────────────────┐  │   │       │
  │       │                                                       │  │   │       │
  │  features[7]:  IRB (t=6, 32→64, s=2)                         │  │   │       │
  │  features[8]:  IRB (t=6, 64→64, s=1)                         │  │   │       │
  │  features[9]:  IRB (t=6, 64→64, s=1)                         │  │   │       │
  │  features[10]: IRB (t=6, 64→64, s=1)           ── S4         │  │   │       │
  │               → [B, 64, 30, 41]     (跳过，不抽取)            │  │   │       │
  │       │                                                       │  │   │       │
  │  features[11]: IRB (t=6, 64→96, s=1)                         │  │   │       │
  │  features[12]: IRB (t=6, 96→96, s=1)                         │  │   │       │
  │  features[13]: IRB (t=6, 96→96, s=1)           ── S5  ★ 抽取 f3    │       │
  │               → [B, 96, 30, 41]     ───────────────────────┐  │  │   │       │
  │       │                                                    │  │  │   │       │
  │  features[14]: IRB (t=6, 96→160, s=2)                     │  │  │   │       │
  │  features[15]: IRB (t=6, 160→160, s=1)                    │  │  │   │       │
  │  features[16]: IRB (t=6, 160→160, s=1)         ── S6      │  │  │   │       │
  │               → [B, 160, 15, 21]    (跳过，不抽取)         │  │  │   │       │
  │       │                                                    │  │  │   │       │
  │  features[17]: IRB (t=6, 160→320, s=1)         ── S7  ★ 抽取 f4    │       │
  │               → [B, 320, 15, 21]    ────────────────────┐  │  │  │   │       │
  │       │                                                 │  │  │  │   │       │
  │  features[18]: Conv2d 320→1280, BN, ReLU6      ── S8   │  │  │  │   │       │
  │               → [B, 1280, 15, 21]   (分类头，不使用)     │  │  │  │   │       │
  │                                                         │  │  │  │   │       │
  └═════════════════════════════════════════════════════════════════════════┘     │
          │                                                 │  │  │  │           │
          │  4 个抽取的特征图:                               │  │  │  │           │
          │  f1 = S2 output [B, 24,  119, 161]  stride=4  ◀─┤──┤──┘  │           │
          │  f2 = S3 output [B, 32,   60,  81]  stride=8  ◀─┤──┘     │           │
          │  f3 = S5 output [B, 96,   30,  41]  stride=16 ◀─┤        │           │
          │  f4 = S7 output [B, 320,  15,  21]  stride=32 ◀─┘        │           │
          │                                                           │           │
          │                                                           │           │
          │              ┌─ DA3 Teacher (frozen) ─────────────────────┤           │
          │              │  ViT-G Backbone                            │           │
          │              │  unsqueeze(1) → [B,1,3,476,644]            │           │
          │              │  Patch Embed 14×14 → [B,1,1564,3072]       │           │
          │              │  40层 Transformer                           │           │
          │              │  抽取层 [19, 27, 33, 39]                    │           │
          │              │       │                                     │           │
          │              │       ▼                                     │           │
          │              │  DualDPT._forward_impl                     │           │
          │              │  ┌──────────────────────────────────────┐   │           │
          │              │  │ 4× tokens [B,1564,3072]              │   │           │
          │              │  │      │                               │   │           │
          │              │  │ LayerNorm(3072)                      │   │           │
          │              │  │ reshape → [B,3072,34,46]             │   │           │
          │              │  │      │                               │   │           │
          │              │  │ projects[i]: Conv2d(3072→oc)         │   │           │
          │              │  │      │                               │   │           │
          │              │  │ resize_layers[i]  ← forward hooks    │   │           │
          │              │  │      │                               │   │           │
          │              │  │  T1 [B, 256, 136,184] (×4 deconv)    │   │           │
          │              │  │  T2 [B, 512,  68, 92] (×2 deconv)    │   │           │
          │              │  │  T3 [B,1024,  34, 46] (identity)     │   │           │
          │              │  │  T4 [B,1024,  17, 23] (÷2 conv)      │   │           │
          │              │  └──────────────────────────────────────┘   │           │
          │              └────────────────────────────────────────────┘           │
          │                                  │                                    │
          │                                  │ teacher_feats (detached)           │
          │                                  │                                    │
        │                               │                               │
          ├─────────────────────────────────┐   │                                    │
          │                                 │   │                                    │
          ▼                                 │   │                                    │
  ┌═══ StudentDualDPT ══════════════════════│═══│════════════════════════════┐       │
  │                                         │   │                            │       │
  │  ┌─ projects (1×1 Conv 通道投影) ──────┐│   │                            │       │
  │  │                                      ││   │                            │       │
  │  │ f1 [B, 24,119,161]→Conv2d(24→256)   ││   │                            │       │
  │  │   = P1 [B, 256, 119,161]            ││───┤                            │       │
  │  │                                      ││   │                            │       │
  │  │ f2 [B, 32, 60, 81]→Conv2d(32→512)   ││   │   ┌──────────────────────┐ │       │
  │  │   = P2 [B, 512,  60, 81]            ││───┤   │  Feature Distill     │ │       │
  │  │                                      ││   │   │  Loss (MSE)          │ │       │
  │  │ f3 [B, 96, 30, 41]→Conv2d(96→1024)  ││   │   │                      │ │       │
  │  │   = P3 [B,1024,  30, 41]            ││───┼──▶│  Interpolate Pi→Ti   │ │       │
  │  │                                      ││   │   │  size then MSE       │ │       │
  │  │ f4 [B,320, 15, 21]→Conv2d(320→1024) ││   │   │                      │ │       │
  │  │   = P4 [B,1024,  15, 21]            ││───┘   │  L = Σ wi·MSE(Pi,Ti) │ │       │
  │  └──────────────────────────────────────┘│       └──────────────────────┘ │       │
  │              + pos_embed                 │                                │       │
  │                  │                       │                                │       │
  │                  ▼                       │                                │       │
  │  ┌─ _fuse (FPN 自顶向下融合) ──────────┐│                                │       │
  │  │                                      ││                                │       │
  │  │  scratch.layer{1-4}_rn               ││                                │       │
  │  │  (3×3 Conv 通道适配器)               ││                                │       │
  │  │           │                          ││                                │       │
  │  │  refinenet4(L4, →L3 size=30×41)      ││                                │       │
  │  │      ↓                               ││                                │       │
  │  │  refinenet3(+L3, →L2 size=60×81)     ││                                │       │
  │  │      ↓                               ││                                │       │
  │  │  refinenet2(+L2, →L1 size=119×161)   ││                                │       │
  │  │      ↓                               ││                                │       │
  │  │  refinenet1(+L1, scale_factor=×2)    ││                                │       │
  │  │      ↓                               ││                                │       │
  │  │   [B, 256, 238, 322]                 ││                                │       │
  │  └──────────────────────────────────────┘│                                │       │
  │         │                  │              │                                │       │
  │    MAIN FPN            AUX FPN           │                                │       │
  │    (独立 refinenet)    (独立 refinenet)   │                                │       │
  │         │                  │              │                                │       │
  │         ▼                  ▼              │                                │       │
  │    output_conv1       output_conv1_aux   │                                │       │
  │    [B,128,238,322]    [B,128,238,322]    │                                │       │
  │         │                  │              │                                │       │
  │    interpolate        (last level)       │                                │       │
  │    → [B,128,476,644]  output_conv2_aux   │                                │       │
  │         │              [B,7,238,322]     │                                │       │
  │    output_conv2            │              │                                │       │
  │    [B,2,476,644]      split + activate   │                                │       │
  │         │                  │              │                                │       │
  │    split + activate        │              │                                │       │
  │         │                  │              │                                │       │
  │    ┌────┴────┐       ┌─────┴─────┐       │                                │       │
  │    │ depth   │       │ ray       │       │                                │       │
  │    │ [B,476, │       │ [B,238,   │       │                                │       │
  │    │  644]   │       │  322,6]   │       │                                │       │
  │    │         │       │           │       │                                │       │
  │    │ depth_  │       │ ray_conf  │       │                                │       │
  │    │ conf    │       │ [B,238,   │       │                                │       │
  │    │ [B,476, │       │  322]     │       │                                │       │
  │    │  644]   │       └───────────┘       │                                │       │
  │    └─────────┘                           │                                │       │
  └══════════════════════════════════════════┘                                │       │
                                                                              │       │
                                                                              │       │
  ┌═══ StudentCameraHead ═══════════════════════════════════════════┐         │       │
  │                                                                  │         │       │
  │  f4 [B, 320, 15, 21]  ◀─────────────────────────────────────────┤─────────┘       │
  │         │                                                        │                 │
  │  AdaptiveAvgPool2d(1) → [B, 320]                                │                 │
  │         │                                                        │                 │
  │  Linear(320 → 3072)  → [B, 3072]                                │                 │
  │         │                                                        │                 │
  │  CameraDec backbone: Linear→ReLU→Linear→ReLU                    │                 │
  │         │                                                        │                 │
  │     ┌───┼──────────┐                                             │                 │
  │     ▼   ▼          ▼                                             │                 │
  │     t   qvec       fov                                           │                 │
  │   [B,3] [B,4]     [B,2]                                         │                 │
  │         │                                                        │                 │
  │     cat → pose_enc [B, 9]                                        │                 │
  └══════════════════════════════════════════════════════════════════┘                 │
                                                                                       │
  ◀────────────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────┐
  │                         最终输出                                  │
  │                                                                  │
  │  depth       [B, 476, 644]        绝对深度图（米制）              │
  │  depth_conf  [B, 476, 644]        深度置信度                      │
  │  ray         [B, 238, 322, 6]     每像素 6D 方向场                │
  │  ray_conf    [B, 238, 322]        ray 置信度                      │
  │  cam.t       [B, 3]               平移向量                        │
  │  cam.qvec    [B, 4]               旋转四元数                      │
  │  cam.fov     [B, 2]               视场角 [fov_h, fov_w]            │
  │                                                                  │
  │  (训练时额外返回)                                                 │
  │  proj_feats  [P1, P2, P3, P4]    4 个蒸馏特征图                  │
  └──────────────────────────────────────────────────────────────────┘
```

### 3.2 `models/backbone.py` — MobileNetV2 特征提取

使用 `torchvision.models.mobilenet_v2(pretrained=True)`，切分 `model.features` 为 4 段：

| DPT Level | features 切片 | 输出通道 | 输出尺寸 (644×476) |
|-----------|-------------|---------|-------------------|
| L1 (S2) | `[0:4]` | 24 | 161×119 |
| L2 (S3) | `[4:7]` | 32 | 81×60 |
| L3 (S5) | `[7:14]` | 96 | 41×30 |
| L4 (S7) | `[14:18]` | 320 | 21×15 |

### 3.3 `models/student_dpt.py` — Student 版 DualDPT

从 DA3 **复制**构建块（`_make_scratch`, `FeatureFusionBlock`, `ResidualConvUnit` 等），DA3 源码**零修改**。

相对原版 DualDPT 的 4 处修改：
1. **projects 改为多通道输入**: `Conv2d(dim_ins[i], out_channels[i], 1)`，`dim_ins=[24,32,96,320]`
2. **删除 LayerNorm**: MobileNetV2 已有 BN
3. **删除 resize_layers**: MobileNetV2 特征天然多尺度，不需从 patch grid 生成
4. **删除 token→spatial reshape**: 输入已是 `[B, C, H, W]`

新增 `return_projected_feats` 标志，训练时同时返回 P1~P4 用于蒸馏。

FPN 融合链（refinenet1-4）、输出头（depth+conf, ray+conf）结构不变。

### 3.4 `models/cam_head.py` — 相机参数头

```
L4 [B,320,15,21] → AdaptiveAvgPool2d(1) → [B,320] → Linear(320,3072) → CameraDec → [t(3), qvec(4), fov(2)]
```

CameraDec 架构同 DA3（Linear→ReLU→Linear→ReLU + 3 个输出分支）。

### 3.5 `models/fast_depth_model.py` — Student 完整模型

```python
def forward(self, images, return_distill_feats=False):
    feats = self.backbone(images)                               # f1~f4
    out, proj_feats = self.head(feats, return_projected_feats=True)  # DualDPT
    out.cam = self.cam_head(feats[-1])                          # CameraHead
    if return_distill_feats:
        return out, proj_feats
    return out
```

---

## 4. Teacher & 蒸馏

### 4.1 `distillation/teacher.py` — DA3 Teacher 封装

加载 DA3 anyview 分支（跳过 metric 分支节省 ~30% 推理时间），冻结全部参数。

在 DualDPT 的 `resize_layers[0..3]` 上注册 forward hook 捕获特征：

| Level | Teacher 特征形状 | 通道 | Student 特征形状 | 通道 |
|-------|-----------------|------|-----------------|------|
| L1 | [B, 256, 136, 184] | 256 | [B, 256, 119, 161] | 256 ✅ |
| L2 | [B, 512, 68, 92] | 512 | [B, 512, 60, 81] | 512 ✅ |
| L3 | [B, 1024, 34, 46] | 1024 | [B, 1024, 30, 41] | 1024 ✅ |
| L4 | [B, 1024, 17, 23] | 1024 | [B, 1024, 15, 21] | 1024 ✅ |

通道自动对齐。空间维度通过 bilinear interpolation 在 loss 中对齐（student → teacher size）。

### 4.2 `distillation/losses.py` — 特征蒸馏 Loss

```python
def feature_distillation_loss(student_feats, teacher_feats, weights=[1,1,1,1]):
    for w, s_feat, t_feat in zip(weights, student_feats, teacher_feats):
        s_aligned = F.interpolate(s_feat, size=t_feat.shape[2:], mode='bilinear')
        loss += w * F.mse_loss(s_aligned, t_feat.detach())
```

### 4.3 蒸馏对应关系

| Level | Teacher (ViT-G→DualDPT) | Student (MobileNetV2→StudentDPT) | 对齐 |
|-------|------------------------|--------------------------------|------|
| L1 | 层19 → project(3072→256) → resize(×4) | S2(24) → project(24→256) | 256=256 ✅ |
| L2 | 层27 → project(3072→512) → resize(×2) | S3(32) → project(32→512) | 512=512 ✅ |
| L3 | 层33 → project(3072→1024) → identity | S5(96) → project(96→1024) | 1024=1024 ✅ |
| L4 | 层39 → project(3072→1024) → resize(÷2) | S7(320) → project(320→1024) | 1024=1024 ✅ |

---

## 5. 数据预处理 — TOS → LMDB

### 5.1 数据集概况

| 项目 | 数值 |
|------|------|
| 数据源 | TartanGround（52 scenes, 739 trajectories, ~111 万帧） |
| 原始分辨率 | 640×640 |
| 目标分辨率 | 644×476（垂直中心裁剪 + 水平 reflect pad） |
| 坐标系 | NED (X=forward, Y=right, Z=down)，pose = c2w |
| Pose 格式 | tx,ty,tz,qx,qy,qz,qw (float64, Hamilton convention) |

### 5.2 空间变换 & 内参

```
640×640 → 垂直裁剪 82px top/bot → 640×476 → 水平 pad 2px L/R → 644×476

K_ORIG    = [320, 0, 319.5]  →  K_CROPPED = [320, 0, 321.5]
            [0, 320, 319.5]                  [0, 320, 237.5]
            [0,   0,   1.0]                  [0,   0,   1.0]

cx: 319.5 + 2 (pad_left) = 321.5
cy: 319.5 - 82 (crop_top) = 237.5
```

### 5.3 LMDB 存储设计

Key: `{Scene}/{Robot}/{Traj}/{Frame:06d}/{field}`

**LMDB 字段**（大 payload，逐帧二进制数据）：

| 字段 | 格式 | 大小/帧 | 构建阶段 |
|------|------|---------|---------|
| RGB | JPEG q95 bytes | ~139 KB | 基础构建 |
| Depth | zlib(float32[476×644].tobytes()) | ~177 KB | 基础构建 |
| Pose | float64[7].tobytes() (tx,ty,tz,qx,qy,qz,qw, c2w NED) | 56 B | 基础构建 |
| Ray | zlib(float16[238×322×6].tobytes()) | ~269 KB | `--add-ray`（可选） |

**JSON 字段**（标量 metadata，方便修改不用重建 LMDB）：

| 字段 | 位置 | 说明 |
|------|------|------|
| Scale | 每个 sample 的 `"scale"` | float，mean ‖P‖₂ of valid 3D points |

基础构建 ~334 GB，各阶段独立可分别运行。

读取方式：
- rgb: `cv2.imdecode(np.frombuffer(val, uint8), IMREAD_COLOR)` → `(476, 644, 3)` uint8 BGR
- depth: `np.frombuffer(zlib.decompress(val), float32).reshape(476, 644)`
- pose: `np.frombuffer(val, float64)` → `(7,)`
- scale: `sample["scale"]`（从 JSON，不从 LMDB）
- ray: `np.frombuffer(zlib.decompress(val), float16).reshape(238, 322, 6)`

### 5.4 JSON 索引 & Train/Val 划分

**`tartanground_index.json`**（完整索引）：

```json
{
  "meta": {
    "total_frames": 1111752,
    "total_trajectories": 739,
    "K_cropped": [[320.0, 0.0, 321.5], [0.0, 320.0, 237.5], [0.0, 0.0, 1.0]],
    "K_orig": [[320.0, 0.0, 319.5], [0.0, 320.0, 319.5], [0.0, 0.0, 1.0]],
    "spatial_transform": "center_crop_640to476 + reflect_pad_640to644",
    "target_size": [476, 644],
    "rgb_encoding": "jpeg_q95",
    "depth_encoding": "zlib_level1_float32",
    "pose_format": "tx,ty,tz,qx,qy,qz,qw (float64)"
  },
  "trajectories": [
    {
      "scene": "AbandonedSchool",
      "robot": "Data_diff",
      "traj": "P1000",
      "num_frames": 2425,
      "key_prefix": "AbandonedSchool/Data_diff/P1000",
      "metadata": {"robot_height": 0.5, "path_length": 120.3, "num_poses": 2425}
    },
    ...
  ],
  "samples": [
    {"key_prefix": "AbandonedSchool/Data_diff/P1000", "frame": 0, "traj_idx": 0, "scale": 12.3456},
    {"key_prefix": "AbandonedSchool/Data_diff/P1000", "frame": 1, "traj_idx": 0, "scale": 12.3501},
    ...
  ]
}
```

**`tartanground_train.json` / `tartanground_val.json`**（split 索引）：

```json
{
  "meta": { ... },
  "trajectory_indices": [0, 1, 3, 5, ...],
  "samples": [
    {"key_prefix": "...", "frame": 0, "traj_idx": 0, "scale": 12.3456},
    ...
  ]
}
```

划分方式：每个场景取 ~10% 轨迹做 val（seed=42），同一轨迹所有帧不跨 split 避免时序泄露。

### 5.5 构建命令

```bash
# 基础构建（~5h, 32 workers 并行读 TOS）
python tools/build_tartanground_lmdb.py --workers 32

# 追加 ray（从 LMDB 读 pose 计算，CPU，~30min）
python tools/build_tartanground_lmdb.py --add-ray

# 追加 sky（从 LMDB 读 RGB 推理 DA3 metric，GPU，~2h）
python tools/build_tartanground_lmdb.py --add-sky

# 验证
python tools/build_tartanground_lmdb.py --verify
```

---

## 6. DataLoader & 在线 GT 计算

### 6.1 设计选择

Ray 不从 LMDB 读取（离线存储 ~280GB），改为在线从 pose + K 计算（~3ms/sample，numpy）。bearing_grid 是常量（只依赖 K），`__init__` 一次性预计算。

### 6.2 bearing_grid 预计算

使用 DA3 normalized [0,2] 空间，K 用 **W_input=644** 归一化（与 DA3 `camray_to_caminfo` 一致）：

```python
fx_norm = 2 * K[0,0] / W_INPUT           # = 2 * 320 / 644 = 0.99379
fy_norm = 2 * K[1,1] / H_INPUT           # = 2 * 320 / 476 = 1.34454
cx_norm = 2 * (K[0,2] + 0.5) / W_INPUT   # = 2 * 322 / 644 = 1.0
cy_norm = 2 * (K[1,2] + 0.5) / H_INPUT   # = 2 * 238 / 476 = 1.0

u = np.linspace(1/W_RAY, 2 - 1/W_RAY, W_RAY)   # W_RAY=322
v = np.linspace(1/H_RAY, 2 - 1/H_RAY, H_RAY)   # H_RAY=238
bearing_grid = [(uu - cx_norm) / fx_norm, (vv - cy_norm) / fy_norm, 1]  # (238, 322, 3)
```

### 6.3 在线 ray + cam_params（每帧）

```
pose (c2w NED) → R_c2w_ned
R_c2w_cv = R_c2w_ned @ R_NED_CV       # R_NED_CV = [[0,0,1],[1,0,0],[0,1,0]]
R_w2c_cv = R_c2w_cv.T
T_w2c_cv = -R_w2c_cv @ t_c2w

ray_dir    = R_w2c_cv @ bearing_flat   → (238, 322, 3)
ray        = concat(ray_dir, T_w2c_cv) → (238, 322, 6) float32

cam_params = [T_w2c_cv(3), qvec_w2c(4, xyzw), fov_h, fov_w] → (9,) float32
```

### 6.4 valid_mask — 基于深度阈值

直接用原始深度值判定远景/天空像素，无需 DA3 sky 推理和 LMDB 存储：

```python
# depth >= depth_cap → 远景/天空 → mask 掉
# depth <  depth_cap → valid
valid_mask = depth < depth_cap   # depth_cap 默认 100.0，config 可调
np.clip(depth, None, depth_cap, out=depth)
```

mask 在 clip 之前生成，clip 之后被 mask 掉的像素值恒为 depth_cap。

### 6.5 增强一致性（flip 同步）

拆掉 `transforms.Compose` 中的 `RandomHorizontalFlip`，手动控制同步翻转：

| 模态 | flip 操作 |
|------|----------|
| image | 列翻转 |
| depth | 列翻转 |
| valid_mask | 列翻转 |
| ray | 列翻转 + `ray[:,:,0] *= -1`（bearing_x）+ `ray[:,:,3] *= -1`（translation_x） |
| cam_params | `t_x *= -1`, `qy *= -1`, `qz *= -1`（等效 pre-multiply `diag(-1,1,1)` on R_w2c） |

ColorJitter 仅对 image tensor 做（不影响几何量）。

### 6.6 返回字段

```python
{
    "image":            (3, 476, 644) float32,    # ImageNet 归一化
    "depth":            (476, 644) float32,        # metric planar depth (m), clamp(max=100)
    "depth_normalized": (476, 644) float32,        # clamp(depth, max=100) / scale
    "ray":              (238, 322, 6) float32,     # 在线计算 GT ray
    "cam_params":       (9,) float32,              # GT [t(3), qvec(4), fov_h, fov_w]
    "valid_mask":       (476, 644) bool,           # depth < depth_cap → True
    "pose":             (7,) float64,              # 原始 c2w NED pose（调试用）
    "scale":            float64 scalar,            # 原始 scale（调试用）
}
```

---

## 7. 训练流程 & Config

### 7.1 训练循环

```python
for epoch in range(total_epochs):
    # ---- Train ----
    student.train()
    for batch in train_dataloader:
        images = batch["image"].cuda()
        teacher_feats = teacher.extract_features(images)    # T1~T4
        with autocast(dtype=bfloat16):
            output, student_feats = student(images, return_distill_feats=True)
            loss = feature_distillation_loss(student_feats, teacher_feats)
            # Phase 2: + depth_loss + ray_loss + cam_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)

    # ---- Validation (every eval_interval epochs, no grad) ----
    if (epoch + 1) % eval_interval == 0 or epoch == total_epochs - 1:
        student.eval()
        with torch.no_grad():
            for batch in val_dataloader:       # augment=False, shuffle=False
                # 同样的 loss 计算，累加求均值
                # Phase 2: 计算 AbsRel 和 δ 指标
                metric_mask = valid_mask & (gt_depth > 1e-3)
                AbsRel = mean(|pred - gt| / gt)
                δ      = fraction(max(pred/gt, gt/pred) < delta_threshold)
        # log val/loss, val/feat, val/level_*, val/absrel, val/delta_<thr> to TensorBoard
        # 若 δ 创新高 → 保存 best.pt（含 best_delta 字段）
```

超参：AdamW lr=1e-4, weight_decay=0.05, cosine decay, warmup 5 epochs, 100 epochs。

### 7.2 Config 结构 (`configs/default.yaml`)

嵌套结构，所有参数显式命名方便调试：

```yaml
model:
  input_height: 476
  input_width: 644
  backbone_pretrained: true
  dpt_features: 256
  dpt_out_channels: [256, 512, 1024, 1024]

teacher:
  model_dir: .../DA3NESTED-GIANT-LARGE-1.1/
  da3_src: .../Depth-Anything-3-main/src

data:
  lmdb_path: .../tartanground.lmdb
  train_index / val_index
  input_height: 476 / input_width: 644
  ray_height: 238 / ray_width: 322
  coord_system: ned
  ned_to_cv: [[0,0,1],[1,0,0],[0,1,0]]
  depth_cap: 100.0
  augment: true / flip_prob: 0.5
  color_jitter: {brightness: 0.2, contrast: 0.2, saturation: 0.2, hue: 0.05}

training:
  batch_size: 4 / val_batch_size: 8
  num_workers: 4 / val_num_workers: 4
  learning_rate: 1e-4 / weight_decay: 0.05
  optimizer: adamw / scheduler: cosine
  warmup_epochs: 5 / total_epochs: 100

loss:
  phase: 1              # 1 = feature distill only, 2 = full task losses
  feat_weights: [1.0, 1.0, 1.0, 1.0]
  depth_weight: 1.0 / ray_weight: 1.0 / point_weight: 1.0
  cam_weight: 1.0 / grad_weight: 1.0
  lambda_c: 0.2         # confidence regularization coefficient
  cam_trans_weight: 1.0 / cam_rot_weight: 1.0 / cam_fov_weight: 0.5

logging:
  log_interval: 50          # 每 N step 打印+写 TensorBoard
  save_interval: 10         # 每 N epoch 存 checkpoint
  eval_interval: 5          # 每 N epoch 做 val 评估（含 AbsRel / δ）
  delta_threshold: 1.25     # δ = % pixels where max(ŷ/y, y/ŷ) < threshold
```

---

## 8. Phase 2 Loss — DA3 论文任务级损失函数

### 8.1 总体公式

DA3 论文训练目标为 5 项加权和：

```
L = LD(D̂, D) + LM(R̂, M) + LP(D̂⊙d+t, P) + βLC(ĉ, v) + αLgrad(D̂, D)
```

其中 α=1, β=1。所有 loss 基于 L1 norm。

实现文件：`distillation/objective_losses.py`（新建）

### 8.2 LD — 深度 Loss（置信度加权 L1）

```
LD = (1/|Ω|) Σ_p∈Ω [ Dc,p · |D̂p - Dp| - λc · log(Dc,p) ]
```

- **输入**: pred_depth `[B,476,644]`, gt_depth `[B,476,644]`, depth_conf `[B,476,644]`, valid_mask `[B,476,644]`
- depth_conf 激活为 `exp(x)+1 ≥ 1`，因此 `log(conf) ≥ 0`
- `conf * error` 鼓励模型在困难像素上降低置信度
- `-λc * log(conf)` 防止置信度坍缩到 0
- λc 默认 0.2

### 8.3 LM — Ray Loss（置信度加权 L1）

- **输入**: pred_ray `[B,238,322,6]`, gt_ray `[B,238,322,6]`, ray_conf `[B,238,322]`, valid_mask `[B,476,644]`
- valid_mask 从全分辨率下采样到半分辨率（max_pool 保守策略：任一子像素无效 → 整像素无效）
- `error = |pred_ray - gt_ray|` 对 6 通道取均值
- 置信度加权公式同 LD：`ray_conf * error - λc * log(ray_conf)`
- ray_conf 激活同 depth_conf 为 `exp(x)+1 ≥ 1`

### 8.4 LP — 点云 Loss（depth + cam_params + K，不使用 ray）

```
pixel_dirs = K⁻¹ @ meshgrid(476,644)   -- 预计算常量
LP = L1( D̂ · (R̂_w2c @ pixel_dirs) + T̂,  D · (R_w2c @ pixel_dirs) + T )
```

- **不使用 ray**，只用 depth + 相机参数 + 固定内参 K
- `pixel_dirs [476,644,3]` 从 K 预计算一次，注册为常量 buffer
- 可微分 quaternion → rotation matrix：`qvec [B,4] → R_w2c [B,3,3]`
- GT 点云从 GT depth + GT cam_params 计算，预测点云从 pred depth + pred cam_params 计算
- 全分辨率，和 depth 天然对齐
- **意义**：强制 depth 预测和 camera 预测的 3D 一致性，梯度同时回传到 depth head 和 cam_head

### 8.5 LC — 相机参数 Loss（per-component L1）

```
LC = w_t · L1(pred_t, gt_t) + w_q · L1(pred_qvec, gt_qvec) + w_fov · L1(pred_fov, gt_fov)
```

- **输入**: pred_cam.pose_enc `[B,9]`, gt_cam_params `[B,9]`，格式均为 `[t(3), qvec(4), fov(2)]`
- 默认权重：w_t=1.0, w_q=1.0, w_fov=0.5

### 8.6 Lgrad — 深度梯度 Loss

```
Lgrad = ||∇x(D̂ - D)||₁ + ||∇y(D̂ - D)||₁
```

- 有限差分：`grad_x = |diff[:,:,1:] - diff[:,:,:-1]|`
- mask 要求相邻两像素都 valid
- clamp(max=100) 防止 outlier
- **意义**：保持深度边缘锐利，同时保证平坦区域平滑

### 8.7 Phase 控制

通过 `loss.phase` 配置控制训练阶段：

| Phase | Loss | 训练循环行为 |
|-------|------|-------------|
| 1 | feature_distillation_loss only | student output 丢弃（现有行为） |
| 2 | feature_distillation_loss + LD + LM + LP + LC + Lgrad | 完整 student output + GT 字段参与 loss |

Phase 1 → Phase 2 切换只需修改 `loss.phase: 2`，可从 Phase 1 checkpoint 续训。

### 8.8 实现文件

```
distillation/
├── losses.py              # Phase 1 特征蒸馏 MSE loss（不变）
└── objective_losses.py    # Phase 2 任务级 loss（新建）
    ├── qvec_to_rotmat()        — 可微分四元数→旋转矩阵
    ├── build_pixel_dirs()      — K⁻¹ @ meshgrid 预计算
    ├── depth_loss()            — LD
    ├── ray_loss()              — LM
    ├── point_cloud_loss()      — LP
    ├── camera_loss()           — LC
    ├── gradient_loss()         — Lgrad
    └── phase2_loss()           — 组合函数，返回 total + 各分量 dict
```

---

## 附录 A. DA3 参考

### A.1 Nested 模型双分支架构

DA3 Nested (`NestedDepthAnything3Net`) = Anyview 分支 + Metric 分支：

```
┌──────────────────────────────────────────────────────────────────┐
│  Anyview (ViT-G + DualDPT)      │  Metric (ViT-L + DPT)        │
├─────────────────────────────────┼───────────────────────────────┤
│  DinoV2-Giant (dim=3072)        │  DinoV2-Large (dim=1024)      │
│  DualDPT: main + aux 双路       │  DPT: main + sky head         │
│  depth + conf (exp激活)          │  depth (exp激活) + sky (relu) │
│  Camera: 预测 extrinsics+K      │  Sky: threshold=0.3 判定天空   │
│  输出: 相对深度 (无量纲)         │  输出: 归一化 metric 深度      │
└─────────────────────────────────┴───────────────────────────────┘
                    │                        │
                    ▼                        ▼
            _apply_metric_scaling: metric_depth *= focal/300
            _apply_depth_alignment: anyview_depth *= scale_factor
            _handle_sky_regions: sky_depth = non_sky_p99
                    │
                    ▼
            最终: metric-aligned 米制深度
```

### A.2 Anyview 分支处理流程

```
RGB (H,W,3) → InputProcessor → resize 到 504 → 对齐 14 倍数 → ImageNet normalize
  → ViT-G (3072d, 40层) → 提取层 [19,27,33,39]
  → DualDPT → main head: exp(logits) = 相对深度 (无量纲)
             → aux head:  ray (238,322,7) + camera estimation
```

### A.3 Metric 分支处理流程

```
RGB → ViT-L (1024d, 24层) → 提取层 [4,11,17,23]
  → DPT → main head: exp(logits) = 归一化 metric 深度
         → sky head:  relu(logits) = sky confidence
  → sky >= 0.3 的像素 depth 置为 non_sky p99
```

### A.4 DA3 正常推理 vs 蒸馏的关键差异

| 项目 | DA3 正常推理 | 蒸馏 |
|------|------------|------|
| 分辨率 | 最长边缩到 504 (如 504×378) | 原始 644×476 |
| Extrinsics | 可选提供 → cam_enc 编码 | 不传 → 用可学习 camera_token |
| 取值 | DualDPT 完整 forward → depth/ray | 只 hook resize_layers → 4 层特征 |

### A.5 Ray 字段含义

DualDPT aux head 输出 (238,322,7)，split 为 ray(6) + ray_conf(1)：
- **channels 0-2**: bearing = R_w2c @ K_norm^{-1} @ pixel_grid — 编码旋转 R 和内参 K
- **channels 3-5**: w2c 平移 T — 所有像素相同

DA3 通过 `camray_to_caminfo` 从 ray 恢复 R, T, focal, principal_point（RANSAC homography + QL decomposition）。

### A.6 坐标系约定

```
TartanGround NED:  X=forward, Y=right, Z=down     (pose = c2w)
DA3 OpenCV:        X=right,   Y=down,  Z=forward   (ray = w2c)

NED → OpenCV:  R_NED_CV = [[0,0,1],[1,0,0],[0,1,0]]
R_c2w_cv = R_c2w_ned @ R_NED_CV
R_w2c_cv = R_c2w_cv.T
T_w2c_cv = -R_w2c_cv @ t_c2w_ned
```

### A.7 DA3 [0,2] 归一化空间

DA3 内部将像素坐标映射到 [0,2]×[0,2]，主点在 (1,1)：
- 网格: `linspace(1/W, 2-1/W, W)` — pixel j 映射到 `(2j+1)/W`
- K 归一化: `fx_norm = 2*fx_pixel/W_input`, `cx_norm = 2*(cx_pixel+0.5)/W_input`
- 注意: K 用 **W_input=644** 归一化（不是 W_aux=322）
