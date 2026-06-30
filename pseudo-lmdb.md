# 使用 DA3 anyview 分支生成伪标签构建新 LMDB 数据集

## Context

现有 TartanGround 数据集（LMDB + JSON）包含 GT depth/pose/scale/sky。目标是用 DA3 anyview 分支对相同 RGB 图片推理，获取**相对深度**和 **6D ray field** 作为伪标签，sky mask 由原始 GT depth >= 100m 生成，构建一个结构兼容的新 LMDB + JSON。

---

## 现有数据结构

### 原 LMDB keys（每帧 5 个 key）

| Key | 格式 | 说明 |
|---|---|---|
| `{prefix}/{frame:06d}/rgb` | JPEG q95 bytes | RGB 图片 (476×644×3) |
| `{prefix}/{frame:06d}/depth` | zlib float32 (476×644) | GT 深度图，单位：米 |
| `{prefix}/{frame:06d}/pose` | float64[7] raw bytes | 相机位姿 (tx,ty,tz,qx,qy,qz,qw)，NED c2w |
| `{prefix}/{frame:06d}/scale` | float64 scalar raw bytes | 场景尺度因子 |
| `{prefix}/{frame:06d}/sky` | zlib float16 (476×644) | 天空置信度，relu 激活后的 raw 值 |

key_prefix 示例：`AbandonedSchool/Data_diff/P1000`

### 原 JSON 结构（如 `tartanground_train_twentieth.json`）

```json
{
  "meta": {
    "total_frames": 1207717,
    "total_trajectories": 739,
    "K_cropped": [[320,0,321.5],[0,320,237.5],[0,0,1]],
    "K_orig": [[320,0,319.5],[0,320,319.5],[0,0,1]],
    "spatial_transform": "center_crop_640to476 + reflect_pad_640to644",
    "target_size": [476, 644],
    "rgb_encoding": "jpeg_q95",
    "depth_encoding": "zlib_level1_float32",
    "pose_format": "tx,ty,tz,qx,qy,qz,qw (float64)"
  },
  "trajectory_indices": [...],
  "samples": [
    {"key_prefix": "...", "frame": 0, "traj_idx": 0, "scale": 4.72021}
  ]
}
```

---

## 新 LMDB key 结构（每帧 5 个 key，删除 scale）

| Key | 来源 | 格式 |
|---|---|---|
| `{prefix}/{frame:06d}/rgb` | 复制原 LMDB | JPEG q95 bytes |
| `{prefix}/{frame:06d}/depth` | DA3 anyview 相对深度 | zlib float32 (476×644) |
| `{prefix}/{frame:06d}/pose` | 复制原 LMDB | float64[7] raw bytes |
| `{prefix}/{frame:06d}/sky` | 原 GT depth >= 100m | zlib float16 (476×644)，0/1 mask |
| `{prefix}/{frame:06d}/ray` | DA3 anyview ray field | zlib float16 (238×322×6) |

**删除**: `scale` key（LMDB 和 JSON 均不保留）

## 新 JSON 结构

```json
{
  "meta": {
    "total_frames": ...,
    "total_trajectories": ...,
    "K_cropped": [[320,0,321.5],[0,320,237.5],[0,0,1]],
    "K_orig": [[320,0,319.5],[0,320,319.5],[0,0,1]],
    "spatial_transform": "center_crop_640to476 + reflect_pad_640to644",
    "target_size": [476, 644],
    "rgb_encoding": "jpeg_q95",
    "depth_encoding": "zlib_level1_float32",
    "depth_source": "da3_anyview_relative",
    "pose_format": "tx,ty,tz,qx,qy,qz,qw (float64)"
  },
  "trajectory_indices": [...],
  "samples": [
    {"key_prefix": "...", "frame": 0, "traj_idx": 0}
  ]
}
```

---

## DA3 anyview 推理细节

- **模型**：加载 `DA3NESTED-GIANT-LARGE-1.1`，直接调用 `model.model.da3(x)` 跑 anyview 分支
- **DualDPT head** 输出 4 个 tensor：`depth`, `depth_conf`, `ray`, `ray_conf`
  - depth: 相对深度 (B, S, H, W)，`exp` 激活
  - ray: 6D ray field (B, S, H, W, 6)，linear 激活
- **分辨率**：DA3 InputProcessor 会将输入 resize 到 `process_res` 级别（默认 504），DualDPT `down_ratio=1` 全分辨率输出
  - depth 需 bilinear resize 到 476×644
  - ray 需 bilinear resize 到 238×322（半分辨率，与原 LMDB ray 一致）
- **ray 捕获**：`DepthAnything3Net.forward` 中 `_process_camera_estimation` 和 `_process_ray_pose_estimation` 都会删除 ray，因此通过在 `anyview.head`（DualDPT）上注册 forward hook 来在 ray 被删除前捕获

## Sky mask 生成

- 从原 LMDB 读取 GT depth（zlib float32）
- `sky_mask = (gt_depth >= 100.0).astype(np.float16)`
- zlib 压缩后存入新 LMDB `{prefix}/{frame:06d}/sky`

---

## 实现方案

### 脚本: `tools/build_da3_pseudo_lmdb.py`

**主要逻辑**：
1. 加载 DA3 nested 模型权重，获取 anyview 分支 `model.model.da3`
2. 加载原 JSON index 获取 sample list
3. 分 batch 处理：
   a. 从原 LMDB 读取 RGB → DA3 InputProcessor 预处理 → anyview forward
   b. 通过 head hook 捕获 ray，从 output 提取 depth → resize 到目标分辨率
   c. 从原 LMDB 读取 GT depth → 计算 sky mask (depth >= 100m)
   d. 从原 LMDB 复制 rgb、pose
   e. 写入新 LMDB
4. 生成新 JSON（去掉 scale，加 `depth_source: da3_anyview_relative` 标注）
5. 支持 `--resume`（检查已有 ray key 跳过）和 `--verify` 模式

**关键引用**：
- DA3 模型加载: `DepthAnything3.from_pretrained(model_dir)` → `model.model.da3`
- InputProcessor: `/data-vepfs/lrd_projs/mono_body_dataset/Depth-Anything-3-main/src/depth_anything_3/utils/io/input_processor.py`
- build 模式参考: `tools/build_tartanground_lmdb.py` 的整体结构和 resume 逻辑
- 原 LMDB 路径: `/root/vepfs/lrd_projs/mono_depth_estimation/fast_mono_depth/data/tartanground.lmdb`
- 原 JSON 路径: `/root/vepfs/lrd_projs/mono_depth_estimation/fast_mono_depth/data/annotations/tartanground_index.json`

**参数**：
```
--src-lmdb       原 LMDB 路径 (默认: data/tartanground.lmdb)
--src-index      原 JSON index 路径 (默认: data/annotations/tartanground_index.json)
--out-lmdb       新 LMDB 输出路径 (默认: data/tartanground_da3pseudo.lmdb)
--out-anno-dir   新 JSON 输出目录 (默认: data/annotations)
--model-dir      DA3 模型路径 (默认: DA3NESTED-GIANT-LARGE-1.1)
--da3-src        DA3 源码路径
--batch-size     推理 batch size (默认 8)
--gpu            GPU 编号 (默认 0)
--process-res    DA3 处理分辨率 (默认 504)
--depth-cap      GT depth >= cap 标记为 sky (默认 100.0)
--verify         验证模式
--verify-num     验证抽样数 (默认 100)
--json-only      只生成 JSON，不跑推理
```

---

## 运行命令

```bash
# 首次运行（构建完整 LMDB + JSON）
cd /root/vepfs/lrd_projs/mono_depth_estimation/fast_mono_depth
python tools/build_da3_pseudo_lmdb.py --gpu 0 --batch-size 8

# 显存不足时降低 batch size
python tools/build_da3_pseudo_lmdb.py --gpu 0 --batch-size 1

# 只生成 JSON（不需要 GPU）
python tools/build_da3_pseudo_lmdb.py --json-only

# 验证输出
python tools/build_da3_pseudo_lmdb.py --verify
```

支持断点续传：中断后重新运行相同命令即可自动从断点恢复。

---

## 验证方法

1. `--verify` 模式：随机抽样 100 个 sample，检查所有 key 存在且可解码，确认无残留 scale key
2. 用 `TartanGroundLMDBDataset` (v2) 加载新 LMDB + JSON，确认 `__getitem__` 正常
3. 可视化对比：原 GT depth vs DA3 相对深度、sky mask 分布

---

## 输出文件

- **LMDB**: `data/tartanground_da3pseudo.lmdb/`
- **JSON**:
  - `data/annotations/tartanground_da3pseudo_index.json`
  - `data/annotations/tartanground_da3pseudo_train.json`
  - `data/annotations/tartanground_da3pseudo_val.json`
  - `data/annotations/tartanground_da3pseudo_train_twentieth.json`
  - `data/annotations/tartanground_da3pseudo_val_twentieth.json`
  - （以及其他已有 split 的对应版本）
