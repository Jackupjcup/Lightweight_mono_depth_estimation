# 移除 valid_mask，全像素参与训练和评估

## Context

当前所有 loss 和评估只在 valid_mask（depth < 100m）像素上计算。改动：dataloader 中将 masked 像素的归一化深度填充为 valid 像素最大值，所有 loss 和评估去掉 valid_mask。新文件加 _v1 后缀。

## 前置：清理 dataset 文件

1. **删除** `data/tartanground_lmdb_dataset.py`（旧版，scale 从 JSON 读，无 image_clean，无重试）
2. **重命名** `data/tartanground_lmdb_dataset_v2.py` → `data/tartanground_lmdb_dataset.py`
3. **更新 import**：
   - `tools/train.py:22` — `from data.tartanground_lmdb_dataset_v2 import` → `from data.tartanground_lmdb_dataset import`
   - `tools/eval.py:35` — 同上
   - `tools/train_distill.py:27` — 同上
   - `data/__init__.py:2` — 已经是正确的，无需改

## 变更

### 1. 新建 `data/tartanground_lmdb_dataset_v1.py`

基于重命名后的 `tartanground_lmdb_dataset.py`，修改 `__getitem__` 返回值（~281行）：

```python
depth_norm = (depth / scale).astype(np.float32)
max_valid = depth_norm[valid_mask].max()
depth_norm[~valid_mask] = max_valid
```

去掉返回字典中的 `valid_mask`。`_compute_scale` 不改。

### 2. 新建 `distillation/objective_losses_v1.py`

所有 loss 去掉 valid_mask：
- `depth_loss` / `ray_loss` / `point_cloud_loss` → `.mean()` 覆盖全图
- `gradient_loss` → 全图差分
- `phase2_loss` → 不再从 batch 取 valid_mask

### 3. 改 `tools/train.py`

- import loss 改为 `objective_losses_v1`
- `batch_dev` 去掉 `"valid_mask"`
- 评估：`metric_mask = gt_d > 1e-3`

### 4. 改 `tools/eval.py`

- 去掉 valid_mask，`metric_mask = gt_d > 1e-3`

### 不改的

- `_compute_scale`、`camera_loss`、config

## 验证

1. 打印 batch 的 depth_normalized 确认填充
2. loss 无 NaN/Inf
3. 对比 <20m 的 absrel 和 delta1
