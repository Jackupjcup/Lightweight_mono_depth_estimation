# Fast Mono Depth — Phase 1 训练架构构建计划

## Context

用 MobileNetV2 替换 DA3 的 ViT-G backbone，通过冻结的 DA3 teacher 进行知识蒸馏。Phase 1 只做特征级 MSE 蒸馏（不涉及 depth/ray/cam loss）。目标是构建完整的模型框架和训练脚本。

**输入分辨率**: 644×476（最接近 640×480 的 14 的倍数，DA3 patch_size=14 要求）

---

## 1. 项目结构

```
fast_mono_depth/
├── configs/
│   └── default.yaml              # 训练超参
├── models/
│   ├── __init__.py
│   ├── backbone.py               # MobileNetV2 四层特征提取
│   ├── student_dpt.py            # 修改版 DualDPT（接受不同通道数输入）
│   ├── cam_head.py               # 相机参数预测头
│   └── fast_depth_model.py       # Student 完整模型
├── distillation/
│   ├── __init__.py
│   ├── teacher.py                # DA3 teacher 封装（冻结 + hook 提取特征）
│   └── losses.py                 # 特征蒸馏 MSE loss
├── data/
│   ├── __init__.py
│   └── dataset.py                # 图片数据集（Phase 1 不需要 GT depth）
├── tools/                        # 训练 & 分析脚本
│   ├── train.py                  # 主训练脚本（sys.path 指向项目根以导入 data/distillation/models）
│   ├── preprocess.py             # 预处理工具（crop/pad、depth 解码、归一化）
│   ├── analyze_depth_dist.py     # 深度分布分析
│   ├── search_clamp.py           # 搜索最优 DEPTH_CLAMP
│   ├── visualize_crop.py         # 可视化 crop+pad（输出到 dataset_process/vis_crop）
│   ├── visualize_crop_clamp.py   # 可视化 crop+clamp（输出到 dataset_process/vis_crop_clamp）
│   └── visualize_crop_rescale.py # 可视化 crop+rescale（输出到 dataset_process/vis_crop_rescale）
├── dataset_process/              # 数据集预处理 & 可视化输出
│   ├── vis_crop/                 # visualize_crop.py 输出
│   ├── vis_crop_clamp/           # visualize_crop_clamp.py 输出
│   └── vis_crop_rescale/         # visualize_crop_rescale.py 输出
└── setup_env.sh                  # Conda 环境安装脚本
```

---

## 2. 环境搭建 — `setup_env.sh`

新建 `fast_mono_depth_lrd` conda 环境：
- Python 3.11, torch>=2.0+cu129, torchvision, timm, einops, omegaconf, addict, safetensors, huggingface_hub, opencv-python, tensorboard, tqdm, xformers
- DA3 源码**零修改**，不需 pip install，运行时通过 `sys.path.insert` 导入

---

## 2.5 学生模型完整 Flow Chart

以 644×476 输入为例，标注每一步的 tensor 形状。

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
  │  cam.fov     [B, 2]               视场角                          │
  │                                                                  │
  │  (训练时额外返回)                                                 │
  │  proj_feats  [P1, P2, P3, P4]    4 个蒸馏特征图                  │
  └──────────────────────────────────────────────────────────────────┘
```

### 训练时数据流总结

```
Image [B,3,476,644]
     │
     ├──▶ Teacher (frozen, no_grad) ──▶ T1~T4 (detach)
     │                                      │
     └──▶ Student ──▶ P1~P4                 │
              │           │                 │
              │      Interpolate(Pi→Ti size) │
              │           │                 │
              │      MSE Loss ◀─────────────┘
              │           │
              │      loss.backward()  ← 仅更新 Student 参数
              │
              └──▶ depth, ray, cam (Phase 1 不计算这些 loss)
```

---

## 3. 各模块实现细节

### 3.1 `models/backbone.py` — MobileNetV2 特征提取

使用 `torchvision.models.mobilenet_v2(pretrained=True)`，切分 `model.features` 为 4 段：

| DPT Level | features 切片 | 输出通道 | 644×476 输出尺寸 |
|-----------|-------------|---------|----------------|
| L1 (S2) | `[0:4]` | 24 | 161×119 |
| L2 (S3) | `[4:7]` | 32 | 81×60 |
| L3 (S5) | `[7:14]` | 96 | 41×30 |
| L4 (S7) | `[14:18]` | 320 | 21×15 |

暴露 `feature_channels = [24, 32, 96, 320]` 供下游模块使用。

### 3.2 `models/student_dpt.py` — Student 版 DualDPT（全新文件，DA3 零修改）

> **重要**：这是从 DA3 代码**复制构建块**到我们自己的新文件中，DA3 源码完全不改动。Teacher 模型原样加载使用。

**从 DA3 复制到 student_dpt.py 的构建块**（来自 `da3/src/depth_anything_3/model/dpt.py`）：
- `_make_scratch`, `_make_fusion_block`, `FeatureFusionBlock`, `ResidualConvUnit`
- 复制后将 `nn.quantized.FloatFunctional()` 的 `skip_add.add(out, x)` 替换为普通 `out + x`，避免混合精度训练问题

**从 DA3 复制到 student_dpt.py 的工具**（来自 `head_utils.py`）：
- `Permute`, `create_uv_grid`, `position_grid_to_embed`, `custom_interpolate`

**Student DualDPT 相对原版的 4 处修改**（仅在复制的代码上改）：

1. **projects 改为多通道输入**: `Conv2d(dim_ins[i], out_channels[i], 1)` 其中 `dim_ins=[24,32,96,320]`
2. **删除 LayerNorm**: MobileNetV2 已有 BN，不需要 token 级 LayerNorm
3. **删除 resize_layers**: MobileNetV2 特征天然具有多尺度空间层级（56×/28×/14×/7×），不需要从统一 patch grid 生成多尺度
4. **删除 token→spatial reshape**: 输入已经是 `[B, C, H, W]` 空间特征图

**新增**: `return_projected_feats` 标志，当 True 时同时返回 projects 后的 4 个特征图用于蒸馏

FPN 融合链（refinenet1-4）、输出头（depth+conf, ray+conf）结构不变。

### 3.3 `models/cam_head.py` — 相机参数头

```
L4 [B, 320, H, W] → AdaptiveAvgPool2d(1) → [B, 320] → Linear(320, 3072) → CameraDec
```

CameraDec 直接复用 DA3 的 `cam_dec.py`（通过 sys.path 导入），输出 `[B, 1, 9]`（t:3 + qvec:4 + fov:2）。Phase 1 不训练此头。

### 3.4 `models/fast_depth_model.py` — Student 完整模型

组合 backbone + StudentDualDPT + StudentCameraHead，前向接口：
```python
def forward(self, images, return_distill_feats=False):
    feats = self.backbone(images)          # 4 个 [B, C_i, H_i, W_i]
    output, proj_feats = self.head(feats)  # DualDPT 输出 + 蒸馏特征
    if return_distill_feats:
        return output, proj_feats
    return output
```

### 3.5 `distillation/teacher.py` — DA3 Teacher 封装

**加载方式**: 直接调用 anyview 分支（跳过 metric 分支节省 ~30% 推理时间）
```python
sys.path.insert(0, da3_src_path)
from depth_anything_3.api import DepthAnything3
model = DepthAnything3.from_pretrained(teacher_model_dir)
teacher_da3 = model.model.da3  # DepthAnything3Net (anyview branch only)
```

**Hook 提取特征**: 在 DualDPT 的 `resize_layers[0..3]` 上注册 forward hook，捕获 projects + resize 后的特征：

| Level | Teacher 特征形状 (644×476) | 通道 |
|-------|--------------------------|------|
| L1 | [B, 256, 136, 184] | 256 |
| L2 | [B, 512, 68, 92] | 512 |
| L3 | [B, 1024, 34, 46] | 1024 |
| L4 | [B, 1024, 17, 23] | 1024 |

**关键**: DA3 输入需要 `[B, N, 3, H, W]`（N=views），单图时 `images.unsqueeze(1)`。

### 3.6 `distillation/losses.py` — 特征蒸馏 Loss

```python
def feature_distillation_loss(student_feats, teacher_feats, weights=[1,1,1,1]):
    loss = 0.0
    for w, s_feat, t_feat in zip(weights, student_feats, teacher_feats):
        s_aligned = F.interpolate(s_feat, size=t_feat.shape[2:],
                                   mode='bilinear', align_corners=False)
        loss += w * F.mse_loss(s_aligned, t_feat.detach())
    return loss
```

空间对齐插值（student → teacher 上采样 ~14%，格式 H×W）：
- L1: 119×161 → 136×184
- L2: 60×81 → 68×92
- L3: 30×41 → 34×46
- L4: 15×21 → 17×23

### 3.7 `data/dataset.py` — 图片数据集

Phase 1 不需要 GT depth，任意图片集合即可。功能：
- 扫描目录读取 jpg/png
- Resize 到目标分辨率（默认 644×476）
- ImageNet 标准化（mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]）
- 基础数据增强（随机水平翻转、颜色抖动）

### 3.8 `tools/train.py` — 训练脚本

**训练循环核心**：
```python
for images in dataloader:
    images = images.cuda()
    
    # Teacher (frozen, no_grad)
    with torch.no_grad():
        teacher_feats = teacher.extract_features(images)
    
    # Student
    with torch.autocast('cuda', dtype=torch.bfloat16):
        output, student_feats = student(images, return_distill_feats=True)
        loss = feature_distillation_loss(student_feats, teacher_feats)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
```

**超参**：AdamW lr=1e-4, cosine decay, warmup 5 epochs, batch_size=8, 100 epochs

---

## 4. 蒸馏对应关系一览

| Level | Teacher (ViT-G→DualDPT) | Student (MobileNetV2→StudentDPT) | 通道对齐 |
|-------|------------------------|--------------------------------|---------|
| L1 | 层19 → project(3072→256) → resize(×4) | S2(24) → project(24→256) | 256=256 ✅ |
| L2 | 层27 → project(3072→512) → resize(×2) | S3(32) → project(32→512) | 512=512 ✅ |
| L3 | 层33 → project(3072→1024) → identity | S5(96) → project(96→1024) | 1024=1024 ✅ |
| L4 | 层39 → project(3072→1024) → resize(÷2) | S7(320) → project(320→1024) | 1024=1024 ✅ |

通道维度 DPT projects 自动对齐，无需额外投影层。空间维度通过 bilinear interpolation 对齐。

---

## 5. 验证策略

构建完成后运行 sanity check：
1. 加载 teacher，单张图推理，打印 4 个 hook 特征形状
2. 加载 student（随机初始化），同图推理，打印 4 个 projected 特征形状
3. 计算 feature distillation loss（应为有限正数）
4. 验证 backward 只更新 student 参数（teacher grad 全部为 None）
5. 跑 10 步训练，验证 loss 下降

---

## 6. 实现顺序

1. `setup_env.sh` → 创建环境
2. `models/backbone.py` → 无依赖
3. `models/student_dpt.py` → 复制 DA3 DPT 构建块 + 修改
4. `models/cam_head.py` → 依赖 DA3 CameraDec
5. `models/fast_depth_model.py` → 组合 2-4
6. `distillation/losses.py` → 无依赖
7. `distillation/teacher.py` → 依赖 DA3 源码
8. `data/dataset.py` → 无依赖
9. `configs/default.yaml` → 无依赖
10. `tools/train.py` → 集成所有模块（通过 sys.path 导入项目根下的 data/distillation/models）
11. 运行 sanity check 验证流程

---

## 7. DA3 Nested 模型双分支深度输出 Flowchart

DA3 Nested 模型 (`NestedDepthAnything3Net`) 由两个独立分支组成，在 `forward()` 中分别推理后融合。

### 7.1 Anyview 分支 (ViT-G + DualDPT)

```
输入 RGB (H,W,3) uint8
  │
  ▼
InputProcessor
  ├─ resize to process_res=504 (upper_bound_resize, 最长边→504, 保持宽高比)
  ├─ _make_divisible_by_resize: 宽高各四舍五入到 14 的倍数
  │   例: 644×476 → 504×372 → 504×378 (nearest_multiple(372,14)=378)
  ├─ ImageNet normalize: (x/255 - mean) / std
  └─ → tensor (1, 1, 3, H', W')
  │
  ▼
ViT-G Backbone (DinoV2-Giant, dim=1536×2=3072 with cat_token)
  ├─ cam_token = CameraEnc(extrinsics, intrinsics)  ← 无外参时为 None
  ├─ 提取 4 层特征: layers [19, 27, 33, 39]
  ├─ alt_start=13, qknorm_start=13, rope_start=13
  └─ → feats: 4×[B*S, N, 3072]
  │
  ▼
DualDPT Head (dim_in=3072, output_dim=2, features=256)
  ├─ 4 层 project + resize → 金字塔特征
  ├─ _fuse(): 4→3→2→1 top-down refinement (main 和 aux 独立)
  ├─ Main head: output_conv2 → logits (H',W',2)
  │   ├─ channel[0]: activation="exp" → depth = exp(logits)  ← 相对深度
  │   └─ channel[1]: conf_activation="expp1" → conf = exp(logits)+1
  └─ Aux head: 最后一层辅助预测 (activation="linear")
  │
  ▼
_process_camera_estimation()
  ├─ CameraDec(feats[-1]) → pose encoding
  └─ → 预测 extrinsics (w2c) + intrinsics (focal, principal point)
  │
  ▼
_process_mono_sky_estimation()
  ├─ DualDPT 无 sky head → 跳过 (DualDPT 没有 sky 输出)
  │
  ▼
OutputProcessor
  ├─ squeeze batch dim
  └─ tensor → numpy float32
  │
  ▼
输出 prediction.depth (N,H',W') — 相对深度 (exp 激活, 无量纲, 无 metric 对齐)
```

**关键特点**: depth = exp(raw)，是无量纲的相对深度。数值范围取决于网络学习到的 logits 分布，没有物理单位。

### 7.2 Metric 分支 (ViT-L + DPT)

```
输入 RGB (H,W,3) uint8
  │
  ▼
InputProcessor (同上)
  └─ → tensor (1, 1, 3, H', W')
  │
  ▼
ViT-L Backbone (DinoV2-Large, dim=1024)
  ├─ 无 cam_token (metric 分支不接收 extrinsics)
  ├─ 提取 4 层特征: layers [4, 11, 17, 23]
  ├─ alt_start=-1, qknorm_start=-1, rope_start=-1 (均关闭)
  └─ → feats: 4×[B*S, N, 1024]
  │
  ▼
DPT Head (dim_in=1024, output_dim=1, features=256)
  ├─ 4 层 project + resize → 金字塔特征
  ├─ _fuse(): 4→3→2→1 top-down refinement
  ├─ Main head: output_conv2 → logits (H',W',1)
  │   ├─ output_dim=1 → 无 confidence 输出
  │   └─ activation="exp" → depth = exp(logits)  ← 归一化 metric 深度
  └─ Sky head (use_sky_head=True):
      ├─ sky_output_conv2 → sky logits (H',W',1)
      └─ sky_activation="relu" → sky = relu(logits)  ← sky 概率图
  │
  ▼
_process_mono_sky_estimation()
  ├─ compute_sky_mask(sky, threshold=0.3) → non_sky_mask
  ├─ non_sky_max = quantile(depth[non_sky], 0.99)
  └─ depth[sky_pixels] = non_sky_max   ← 天空设为非天空最大深度
  │
  ▼
OutputProcessor
  ├─ squeeze batch dim
  └─ tensor → numpy float32
  │
  ▼
输出 prediction.depth (N,H',W') — 归一化 metric 深度
  注意: 此深度尚未经过 focal_length 缩放
  (在 nested 流程中会被 apply_metric_scaling 做 depth *= focal/300)
```

**关键特点**: depth = exp(raw)，是归一化的 metric 深度。单独使用 metric 分支时，**缺少 `apply_metric_scaling`**（那是 nested forward 中才做的），所以输出不是真正的米制深度，而是未经焦距缩放的中间值。

### 7.3 Nested 融合流程

当使用完整 `NestedDepthAnything3Net` 时，`forward()` 将两个分支融合：

```
anyview_output = self.da3(x, ...)          # anyview 分支推理
metric_output  = self.da3_metric(x)        # metric 分支推理
  │                    │
  ▼                    ▼
_apply_metric_scaling(output, metric_output)
  └─ metric_output.depth *= focal_length / 300   ← 焦距缩放到米制
  │
  ▼
_apply_depth_alignment(output, metric_output)
  ├─ non_sky_mask = compute_sky_mask(metric_output.sky, threshold=0.3)
  ├─ median_conf = quantile(anyview_conf[non_sky], 0.5)
  ├─ align_mask = 高置信度 & 非天空 & 深度比值合理的像素
  ├─ scale_factor = least_squares(metric_depth[mask], anyview_depth[mask])
  └─ anyview_depth *= scale_factor    ← anyview 深度对齐到 metric 尺度
  │
  ▼
_handle_sky_regions(output, metric_output)
  ├─ non_sky_max = min(quantile(depth[non_sky], 0.99), 200.0)
  └─ depth[sky_pixels] = non_sky_max  ← 天空区域深度置为上限
  │
  ▼
最终输出: metric-aligned 深度 (米制, anyview 结构 + metric 尺度)
```

### 7.4 对比总结

```
┌────────────┬──────────────────────────────┬──────────────────────────────────────┐
│            │  Anyview (ViT-G + DualDPT)   │         Metric (ViT-L + DPT)         │
├────────────┼──────────────────────────────┼──────────────────────────────────────┤
│ Backbone   │ DinoV2-Giant (1536×2=3072)   │ DinoV2-Large (1024)                  │
├────────────┼──────────────────────────────┼──────────────────────────────────────┤
│ Head       │ DualDPT (main + aux 双路)    │ DPT (main + sky 双 head)             │
├────────────┼──────────────────────────────┼──────────────────────────────────────┤
│ output_dim │ 2 (depth + conf)             │ 1 (depth only)                       │
├────────────┼──────────────────────────────┼──────────────────────────────────────┤
│ Sky 处理   │ 无 sky head                  │ sky head → 天空区域 depth 置为 p99   │
├────────────┼──────────────────────────────┼──────────────────────────────────────┤
│ 深度激活   │ exp(logits)                  │ exp(logits)                          │
├────────────┼──────────────────────────────┼──────────────────────────────────────┤
│ 输出性质   │ 相对深度 (无量纲)            │ 归一化 metric 深度 (未做 focal 缩放) │
├────────────┼──────────────────────────────┼──────────────────────────────────────┤
│ Camera     │ 预测 extrinsics + intrinsics │ 不预测 camera                        │
└────────────┴──────────────────────────────┴──────────────────────────────────────┘
```

### 7.5 InputProcessor 分辨率处理细节

DA3 推理时的 resize 逻辑（`upper_bound_resize` + `_make_divisible_by_resize`）：

1. **`_resize_longest_side(img, 504)`**: 将最长边缩放到 504，短边按比例缩放，保持宽高比
2. **`_make_divisible_by_resize(img, 14)`**: 宽高各四舍五入到最近的 14 的倍数 (ViT patch 对齐)

示例:
- 640×640 → 504×504 → 504×504 (已是 14 倍数)
- 644×476 → 504×373 → 504×378 (nearest_multiple(373,14)=378)

**OutputProcessor 不做 resize 回原始分辨率**，模型输出的深度图保持在 DA3 内部处理分辨率。

---

## 8. DA3 正常推理 vs TeacherDA3 蒸馏路径差异

### 8.1 模型本身

**完全相同。** `TeacherDA3` 通过 `model.model.da3` 直接取出了 `DepthAnything3Net`（anyview 分支），权重和结构与 DA3 正常推理用的是同一个网络实例。backbone 是 DinoV2 vitg（3072维），head 是 DualDPT。

### 8.2 处理流程逐步对比

| 步骤 | DA3 正常推理 (`api.inference`) | TeacherDA3 蒸馏 (`train.py`) |
|------|------|------|
| **1. 图片加载** | `InputProcessor._load_image` → PIL RGB | `ImageFolderDataset` → cv2 BGR→RGB |
| **2. Resize** | 最长边缩到 504，再对齐到 14 倍数（如 644×476→504×364） | 直接 resize 到固定 644×476（config 指定），**无 504 限制** |
| **3. Normalize** | ImageNet `mean=[.485,.456,.406], std=[.229,.224,.225]` | **相同**，ImageNet 归一化 |
| **4. 输入形状** | `[1, N, 3, H, W]`（N=视图数） | `[B, 1, 3, 476, 644]`（unsqueeze 加 N=1 维度） |
| **5. Extrinsics/Intrinsics** | 可选提供，会做 normalize（归一化 translation） | **不传**（None），cam_token=None |
| **6. cam_token** | 如果有 extrinsics → `cam_enc` 编码 cam_token；如果没有 → 用 `self.camera_token`（可学习参数）注入到 cls token | **没有 extrinsics** → cam_token=None → 用 `self.camera_token` |
| **7. DINOv2 pos_embed** | 输入如 504×364 → patch grid 36×26 → pos_embed 从 37×37 interpolate 到 36×26 | 输入 644×476 → patch grid 46×34 → pos_embed 从 37×37 interpolate 到 46×34 |
| **8. Hook 提取点** | 不用 hook，正常 forward 流过 DualDPT 的 `resize_layers` | hook 挂在 `dpt_head.resize_layers` 上，捕获 resize 后的 4 层特征 |
| **9. autocast** | `bfloat16` 或 `float16` | **相同** |
| **10. 输出处理** | `OutputProcessor` → depth/extrinsics/intrinsics | 只取 hook 到的 4 个特征 `.float()`，**忽略所有 head 输出** |

### 8.3 关键差异总结

**差异 1：分辨率不同（最大差异）**
- DA3 推理：504×364（以 644×476 原图为例）
- 蒸馏：644×476（原始分辨率）
- DINOv2 的 patch 数量不同（936 vs 1564 tokens），`interpolate_pos_encoding` 的插值比例不同，`resize_layers`（ConvTranspose2d/Conv2d）输出的空间尺寸也不同

**差异 2：无 extrinsics 输入**
- DA3 推理时可提供 extrinsics，通过 `cam_enc` 编码成 cam_token
- 蒸馏时不传 extrinsics，走 fallback 分支用可学习的 `self.camera_token`
- 两条路径最终都是在 `alt_start` 层替换 cls token，但 token 来源不同

**差异 3：只取中间特征，不取最终输出**
- DA3 推理要走完 DualDPT 的 fusion + output_conv 拿到 depth/ray
- 蒸馏只 hook `resize_layers` 的输出（projects → resize 后的 4 层特征），不关心后续的 fusion 和 head

---

## 9. TartanGround 数据预处理 — TOS → LMDB

### 9.1 背景

TartanGround 数据集存储在对象存储 `/data-tos-daily/TartanGround`，单帧读取延迟 ~500–1000ms，直接训练不可行。需要一次性预处理：crop+pad 到 644×476，连同相机内外参打包成 LMDB，存到本地 vepfs 加速训练 IO。

### 9.2 数据集规模

| 项目 | 数值 |
|------|------|
| 场景 | 52 个 |
| 机器人类型 | 3 种: Data_omni (350 traj), Data_diff (149), Data_anymal (240) |
| 总轨迹 | **739** 条 |
| 平均帧数/轨迹 | ~1504 |
| 估算总帧数 | **~111 万帧** |
| 原始分辨率 | 640×640 |
| 目标分辨率 | 644×476 (center crop + reflect pad) |

### 9.3 空间变换 & 内参推导

使用 `tools/preprocess.py` 中的 `crop_and_pad()`，操作为：

1. **垂直中心裁剪** 640→476：去掉顶部/底部各 82 行
2. **水平 reflect pad** 640→644：左右各加 2 像素

**焦距不变**（无 resize），主点平移：

```
K_ORIG    = [320,   0, 319.5]    →    K_CROPPED = [320,   0, 321.5]
            [  0, 320, 319.5]                      [  0, 320, 237.5]
            [  0,   0,   1.0]                      [  0,   0,   1.0]

cx: 319.5 + 2 (pad_left) = 321.5
cy: 319.5 - 82 (crop_top) = 237.5
```

已验证：同一物理像素通过 K_ORIG（原图坐标）和 K_CROPPED（裁剪图坐标）反投影到的 3D 点完全一致。

### 9.4 Pose 文件格式

`pose_lcam_front.txt`：每行 7 个 float64，对应一帧：

```
tx ty tz qx qy qz qw
```

行数 = 帧数（已验证：P1000 有 2425 行 pose = 2425 帧图像）。

### 9.5 LMDB 存储设计

采用模块化构建，基础字段 + 可选字段可分阶段追加（LMDB 支持追加 key 无需重建）：

**LMDB 字段**（大 payload，逐帧二进制数据）：

| 字段 | 来源 | Value 格式 | 大小/帧 | 构建阶段 |
|------|------|------------|---------|---------|
| RGB | `image_lcam_front/*.png` → `crop_and_pad` | JPEG q95 bytes | ~139 KB | 基础构建 |
| Depth | `depth_lcam_front/*.png` → `decode_depth` → `crop_and_pad` | zlib(level=1) float32 bytes | ~177 KB | 基础构建 |
| Pose | `pose_lcam_front.txt` 第 N 行 | float64[7] raw bytes (tx,ty,tz,qx,qy,qz,qw, c2w NED) | 56 B | 基础构建 |
| Ray | 从 LMDB 中的 pose 计算 (见 9.5.1) | zlib(level=1) float16 (238,322,6) | ~269 KB | `--add-ray` |
| Sky | DA3 metric 分支推理 (见 9.5.2) | zlib(level=1) float16 (476,644) | ~100 KB | `--add-sky` |

**JSON 字段**（标量 metadata，存在 JSON sample 中方便修改不用重建 LMDB）：

| 字段 | 来源 | 说明 |
|------|------|------|
| Scale | `compute_scale_factor(depth_cropped, K_CROPPED)` | float，mean ‖P‖₂ of valid 3D points |

**Key 命名**: `{Scene}/{Robot}/{Traj}/{Frame:06d}/{field}`

```
AbandonedSchool/Data_diff/P1000/000000/rgb
AbandonedSchool/Data_diff/P1000/000000/depth
AbandonedSchool/Data_diff/P1000/000000/pose
AbandonedSchool/Data_diff/P1000/000000/ray    # --add-ray 后追加
AbandonedSchool/Data_diff/P1000/000000/sky    # --add-sky 后追加
```

#### 9.5.1 Ray 计算方法

Ray 编码 w2c 旋转和平移到每像素 6D 向量，分辨率为 aux 分辨率 (238×322)。

**常量（初始化一次）：**
- `R_ned_cv = [[0,0,1],[1,0,0],[0,1,0]]`：NED → OpenCV 坐标系转换
- `bearing_grid (238,322,3)`：从 K_CROPPED 归一化内参推导的像素方向网格

**每帧计算：**
```
pose → R_c2w_ned, t_c2w_ned
R_c2w_cv = R_c2w_ned @ R_ned_cv
R_w2c_cv = R_c2w_cv.T
T_w2c_cv = -R_w2c_cv @ t_c2w_ned
ray_dir  = R_w2c_cv @ bearing_grid     → (238, 322, 3)
ray      = concat(ray_dir, T_w2c_cv)   → (238, 322, 6) float16
```

float16 精度损失 max=0.0005，对 ray 方向/平移完全可接受。

#### 9.5.2 Sky 计算方法

使用 DA3 metric 分支 (ViT-L + DPT + sky_head) 对 cropped RGB 推理：
- 输入：cropped RGB (476×644)，DA3 内部 resize 到 504×378
- 输出：`sky` map，relu 激活后的 sky confidence
- 存储：resize 到 (476×644)，float16，zlib 压缩

### 9.6 估算总大小

| 配置 | 大小/帧 | LMDB 总量 |
|------|---------|----------|
| 基础 (RGB+Depth+Pose+Scale) | 316 KB | **334 GB** |
| + Ray | 585 KB | **618 GB** |
| + Sky | 416 KB | **440 GB** |
| + Ray + Sky | 685 KB | **0.71 TB** |

vepfs 可用 4.4TB，所有配置均充裕。

### 9.7 JSON 索引

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

### 9.8 Train/Val 划分

按场景分层采样：每个场景取 ~10% 轨迹做 val（seed=42），确保所有 52 个场景在 val 中都有代表。同一轨迹的所有帧不跨 split，避免时序泄露。

### 9.9 文件列表

```
fast_mono_depth/
├── tools/
│   ├── preprocess.py                    # crop_and_pad, decode_depth, K_CROPPED 等
│   ├── build_tartanground_lmdb.py       # LMDB 构建 + 索引生成 + 验证
│   └── batch_preprocess.py              # (可选) 直接输出 PNG 文件的批处理
├── data/
│   ├── dataset.py                       # Phase 1 简单图片数据集
│   ├── tartanground_lmdb_dataset.py     # LMDB Dataset 类
│   ├── tartanground.lmdb/               # LMDB 数据库 (~334 GB)
│   ├── tartanground_index.json          # 完整索引
│   ├── tartanground_train.json          # train split
│   └── tartanground_val.json            # val split
```

### 9.10 构建命令

```bash
# Phase 1: 基础构建（预计 ~5h，32 workers 并行读 TOS）
python tools/build_tartanground_lmdb.py --workers 32

# Phase 2（可选）: 追加 ray（从 LMDB 读 pose 计算，CPU，~30min）
python tools/build_tartanground_lmdb.py --add-ray

# Phase 3（可选）: 追加 sky（从 LMDB 读 RGB 推理 DA3 metric，GPU，~2h）
python tools/build_tartanground_lmdb.py --add-sky

# 验证
python tools/build_tartanground_lmdb.py --verify
```

各阶段独立，可在不同时间按需运行。

### 9.11 Dataset 读取接口

> **注**：此为旧版接口，已被 Section 10 的 `TartanGroundLMDBDataset` 替代。Scale 从 JSON sample 读取（不从 LMDB），详见 Section 10.7。

---

## 10. Phase 2 DataLoader — 在线 ray 计算 + GT cam_params + valid_mask

### 10.1 背景

Phase 2 训练需要 GT ray、cam_params、valid_mask 字段用于 depth/ray/cam loss。ray 不再从 LMDB 读取（离线存储需要 ~270KB/帧 × 111万帧 ≈ 280GB），改为在线从 pose + K 计算（~3ms/sample，numpy）。

同时修复了原 DataLoader 的增强一致性 bug：`RandomHorizontalFlip` 只作用于 image，depth/ray 不同步。

### 10.2 修改文件

| 文件 | 改动 |
|------|------|
| `data/tartanground_lmdb_dataset.py` | 重写：在线 ray + cam_params 计算，sky→valid_mask，flip 同步增强 |
| `configs/default.yaml` | 重构为嵌套结构（model/teacher/data/training/loss/logging），暴露 ray 参数、sky_threshold |
| `data/__init__.py` | 导出 `TartanGroundLMDBDataset` |
| `tools/train.py` | 适配新 config 嵌套结构 + 从 dict batch 取 image |

### 10.3 bearing_grid 预计算（`__init__` 常量）

使用 DA3 normalized [0,2] 空间，K 用 W_input=644 归一化（与 DA3 `camray_to_caminfo` 一致）：

```python
fx_norm = 2 * K[0,0] / W_INPUT           # = 2 * 320 / 644 = 0.99379
fy_norm = 2 * K[1,1] / H_INPUT           # = 2 * 320 / 476 = 1.34454
cx_norm = 2 * (K[0,2] + 0.5) / W_INPUT   # = 2 * 322 / 644 = 1.0
cy_norm = 2 * (K[1,2] + 0.5) / H_INPUT   # = 2 * 238 / 476 = 1.0

u = np.linspace(1/W_RAY, 2 - 1/W_RAY, W_RAY)   # W_RAY=322
v = np.linspace(1/H_RAY, 2 - 1/H_RAY, H_RAY)   # H_RAY=238
bearing_grid = [(uu - cx_norm) / fx_norm, (vv - cy_norm) / fy_norm, 1]  # (238, 322, 3)
```

bearing_grid 只依赖 K，所有帧完全相同，`__init__` 一次性计算。

### 10.4 在线 ray + cam_params 计算（每帧 `__getitem__`）

```
pose (c2w NED) → R_c2w_ned → R_c2w_cv = R_c2w_ned @ R_NED_CV → R_w2c_cv = R_c2w_cv.T
                                                                 T_w2c_cv = -R_w2c_cv @ t_c2w

ray_dir   = R_w2c_cv @ bearing_flat    → (238, 322, 3)
ray       = concat(ray_dir, T_w2c_cv)  → (238, 322, 6) float32

cam_params = [T_w2c_cv(3), qvec_w2c(4, xyzw), fov(2)]  → (9,) float32
  - fov = [2*arctan(W/(2*fx)), 2*arctan(H/(2*fy))]
```

### 10.5 valid_mask — 基于 LMDB 中的 sky 字段

DA3 metric 分支的 sky head 输出 relu 激活的置信度（值越大越可能是天空），存入 LMDB 为 zlib float16 (476, 644)：

```python
# DA3 alignment.py:54-65
def compute_sky_mask(sky_prediction, threshold=0.3):
    return sky_prediction < threshold   # True = 非天空 (valid)
```

DataLoader 中：
- 从 LMDB 读 sky → `valid_mask = sky < sky_threshold`（threshold 默认 0.3，config 可调）
- 若 LMDB 无 sky 字段 → `valid_mask = np.ones(..., dtype=bool)`

### 10.6 修复增强一致性

拆掉 `transforms.Compose` 中的 `RandomHorizontalFlip`，改为手动控制，同步翻转所有模态：

| 模态 | flip 操作 |
|------|----------|
| image | 列翻转 `image[:, ::-1]` |
| depth | 列翻转 `depth[:, ::-1]` |
| valid_mask | 列翻转 `valid_mask[:, ::-1]` |
| ray | 列翻转 + `ray[:,:,0] *= -1`（bearing_x）+ `ray[:,:,3] *= -1`（translation_x） |
| cam_params | `t_x *= -1`, `qy *= -1`, `qz *= -1`（等效于 pre-multiply `diag(-1,1,1)` on R_w2c） |

ColorJitter 仍用 torchvision，只对 image tensor 做（不影响几何量）。

### 10.7 返回字段

```python
{
    "image":            (3, 476, 644) float32,   # ImageNet 归一化
    "depth":            (476, 644) float32,       # metric planar depth (m)
    "depth_normalized": (476, 644) float32,       # depth / scale
    "ray":              (238, 322, 6) float32,    # 在线计算 GT ray
    "cam_params":       (9,) float32,             # GT [t(3), qvec(4), fov(2)]
    "valid_mask":       (476, 644) bool,          # sky < threshold → True
    "pose":             (7,) float64,             # 原始 c2w NED pose（调试用）
    "scale":            float64 scalar,           # 原始 scale（调试用）
}
```

### 10.8 Config 结构（`configs/default.yaml`）

重构为嵌套结构，所有参数显式命名方便调试：

```yaml
model:
  input_height: 476
  input_width: 644
  backbone_pretrained: true
  dpt_features: 256
  dpt_out_channels: [256, 512, 1024, 1024]

teacher:
  model_dir: /data-vepfs/.../DA3NESTED-GIANT-LARGE-1.1/
  da3_src: /data-vepfs/.../Depth-Anything-3-main/src

data:
  lmdb_path: .../tartanground.lmdb
  train_index: .../tartanground_train.json
  val_index: .../tartanground_val.json
  input_height: 476
  input_width: 644
  ray_height: 238                              # input_height // 2
  ray_width: 322                                # input_width // 2
  coord_system: ned                             # TartanGround pose frame
  ned_to_cv: [[0,0,1],[1,0,0],[0,1,0]]
  sky_threshold: 0.3                            # DA3 default
  augment: true
  flip_prob: 0.5
  color_jitter: {brightness: 0.2, contrast: 0.2, saturation: 0.2, hue: 0.05}

training:
  batch_size: 4
  num_workers: 4
  learning_rate: 1.0e-4
  weight_decay: 0.01
  optimizer: adamw
  scheduler: cosine
  warmup_epochs: 5
  total_epochs: 100

loss:
  phase: 1              # 1 = feature distill only, 2 = full task losses
  feat_weights: [1.0, 1.0, 1.0, 1.0]
  # Phase 2 task loss weights (DA3 paper: α=1, β=1)
  depth_weight: 1.0
  ray_weight: 1.0
  point_weight: 1.0
  cam_weight: 1.0
  grad_weight: 1.0
  lambda_c: 0.2         # confidence regularization coefficient
  cam_trans_weight: 1.0
  cam_rot_weight: 1.0
  cam_fov_weight: 0.5

logging:
  log_interval: 50
  save_interval: 10
```

---

## 11. Phase 2 Loss — DA3 论文任务级损失函数

### 11.1 总体公式

DA3 论文训练目标：

```
L = LD(D̂, D) + LM(R̂, M) + LP(D̂⊙d+t, P) + βLC(ĉ, v) + αLgrad(D̂, D)
```

α=1, β=1。所有 loss 基于 L1 norm。实现文件：`distillation/objective_losses.py`

### 11.2 五个 Loss 详解

**LD — 深度 Loss（置信度加权 L1）**

```
LD = mean_valid[ depth_conf · |pred_depth - gt_depth| - λc · log(depth_conf) ]
```
- depth_conf 激活 `exp(x)+1 ≥ 1`，λc=0.2

**LM — Ray Loss（置信度加权 L1）**
- pred/gt ray `[B,238,322,6]`，ray_conf `[B,238,322]`
- valid_mask 从 `[B,476,644]` 下采样到 `[B,238,322]`（max_pool 保守策略）
- 公式同 LD：`ray_conf * |error| - λc * log(ray_conf)`

**LP — 点云 Loss（depth + cam_params + K，不使用 ray）**

```
pixel_dirs = K⁻¹ @ meshgrid(476,644)   -- 预计算常量
pred_point = pred_depth · (R̂_w2c @ pixel_dirs) + T̂
gt_point   = gt_depth   · (R_w2c  @ pixel_dirs) + T
LP = L1(pred_point, gt_point)
```
- `pixel_dirs` 从固定内参 K 预计算一次
- 可微分 qvec→R 转换，梯度同时回传到 depth head 和 cam_head

**LC — 相机参数 Loss（per-component L1）**

```
LC = w_t · L1(t) + w_q · L1(qvec) + w_fov · L1(fov)
```
- 默认 w_t=1.0, w_q=1.0, w_fov=0.5

**Lgrad — 深度梯度 Loss**

```
Lgrad = ||∇x(D̂ - D)||₁ + ||∇y(D̂ - D)||₁
```
- 有限差分，相邻两像素都 valid 才计算，clamp(max=100)

### 11.3 Phase 控制

`loss.phase: 1` → 仅特征蒸馏（现有行为不变）
`loss.phase: 2` → 特征蒸馏 + 5 个任务级 loss，可从 Phase 1 checkpoint 续训

### 11.4 实现文件

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
    └── phase2_loss()           — 组合函数
```
