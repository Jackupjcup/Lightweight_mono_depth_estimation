---
name: training-issues-4gpu
description: 4xA800 DDP训练中发现的超参数和代码逻辑问题及修复记录 (2026-06-18)
metadata:
  type: project
---

# 4×A800 DDP 训练问题排查与修复记录

日期: 2026-06-18
配置: `configs/v0_fortieth.yaml` (1/40 数据子集, ~27k train / ~3.2k val)
训练脚本: `tools/train.py`, 4×A800 via `torchrun --nproc_per_node=4`

---

## 问题 1: Learning Rate 线性缩放

**现象**: config 中 `learning_rate: 1.0e-4`，代码 `train.py:162` 执行 `effective_lr = tc.learning_rate * world_size`。

**结论**: 这是正确的。config 中的 `1e-4` 是**单卡基准 LR**。4 卡时有效 batch = `16×4=64`，是单卡的 4 倍，根据 linear scaling rule，LR 放大到 `4e-4`。无需修改。

**Why:** DDP 下每步梯度是 N 张卡的平均，等价于 N 倍 batch size，需要等比放大 LR 以保持相同的有效更新步长。

---

## 问题 2: GradScaler + bfloat16 多余开销

**现象**: `torch.amp.GradScaler("cuda")` 配合 `autocast(dtype=torch.bfloat16)` 使用。

**问题**: `GradScaler` 是为 float16 设计的，处理 fp16 的 underflow/overflow。bfloat16 的指数位与 float32 相同，不需要 loss scaling。PyTorch 在 bf16 下 scaler 实际变成 no-op，不影响正确性但有多余开销。

**修复**: 移除 `GradScaler` 相关全部代码:
- 删除 `scaler = torch.amp.GradScaler("cuda")`
- `scaler.scale(loss).backward()` → `loss.backward()`
- `scaler.step(optimizer)` → `optimizer.step()`
- 删除 `scaler.update()`
- checkpoint 保存/加载中移除 `scaler` 字段

**涉及文件**: `tools/train.py`

---

## 问题 3: Early Stopping DDP 同步 Bug (严重)

**现象**: `train.py` 原代码流程:
1. rank 0 更新 `best_feat_loss` / `best_delta`，保存 checkpoint
2. broadcast `[best_feat_loss, best_delta, es_counter]` 到所有 rank
3. 所有 rank 计算 `improved`，更新 `es_counter`
4. `es_counter >= es_patience` 时 break

**问题**: broadcast 在 `es_counter` 更新**之前**执行，所以 rank 1-3 拿到的是**旧的** `es_counter`。当 rank 0 的 `es_counter >= es_patience` 触发 break 退出循环后，其他 rank 的 `es_counter` 可能还不够，不会 break，随后在 `dist.barrier()` 处**死锁 hang 住**。

**修复**: 彻底去掉 broadcast。因为 all-reduce 之后所有 rank 的验证指标已完全一致，让所有 rank 各自独立计算 `feat_improved`/`delta_improved`、更新 `best_feat_loss`/`best_delta`/`es_counter`，结果天然相同。所有 rank 同时到达 break，不会 hang。

**新代码结构**:
```
1. all-reduce val metrics → 所有 rank 值一致
2. 所有 rank 计算 feat_improved / delta_improved
3. 所有 rank 更新 best_feat_loss / best_delta
4. 所有 rank 更新 es_counter
5. rank 0 保存 checkpoint (使用已更新的值)
6. 所有 rank 检查 es_counter >= es_patience → 同时 break
```

**涉及文件**: `tools/train.py`

---

## 问题 4: Early Stop improved 判断位置不一致

**现象**: `delta_improved` 和 `feat_improved` 在 rank 0 的 checkpoint 块中计算，用的是 broadcast 前的 `best_delta`/`best_feat_loss`（可能与其他 rank 不一致）。

**修复**: 与问题 3 一并解决。现在 `feat_improved`/`delta_improved` 在 all-reduce 之后、由所有 rank 用完全一致的值计算，不存在 rank 间不一致的风险。

**涉及文件**: `tools/train.py`

---

## 问题 5: Dataset `__getitem__` 无限递归风险

**现象**: `tartanground_lmdb_dataset_v2.py` 中:
```python
def __getitem__(self, idx):
    try:
        return self._getitem_impl(idx)
    except Exception:
        return self.__getitem__(random.randint(0, len(self) - 1))
```

**问题**: 吞掉所有异常，无递归深度限制。如果 LMDB 有系统性问题（路径错误、文件损坏），会无限递归直到 stack overflow。

**修复**: 用 for 循环替代递归，最多重试 10 次，超限后抛出 `RuntimeError` 并附带原始异常链:
```python
_MAX_RETRIES = 10

def __getitem__(self, idx):
    for attempt in range(self._MAX_RETRIES):
        try:
            target = idx if attempt == 0 else random.randint(0, len(self) - 1)
            return self._getitem_impl(target)
        except Exception as e:
            if attempt == self._MAX_RETRIES - 1:
                raise RuntimeError(
                    f"Failed after {self._MAX_RETRIES} retries (orig idx={idx})"
                ) from e
```

**涉及文件**: `data/tartanground_lmdb_dataset_v2.py`

---

## 问题 6: Validation drop_last=False + DDP pad (已知微小偏差)

**现象**: val DataLoader 使用 `drop_last=False`，`DistributedSampler` 会 pad 最后一个 batch 使各 rank batch 数相同。

**影响**: pad 的样本会被重复计数，val 指标有微小偏差。

**结论**: 可接受，未修改。

---

## 审查通过项

| 项目 | 状态 |
|---|---|
| Loss 权重配置 vs 代码读取路径 | 全部匹配 |
| 数据维度一致性 (input/ray/pixel_dirs) | 正确 |
| DistributedSampler set_epoch | 正确 |
| Optimizer AdamW weight_decay | 正确 |
| Scheduler warmup→hold→cosine 逻辑 | 正确 |
| DDP find_unused_parameters | 已开启 |
| Teacher frozen, no DDP | 正确 |

**How to apply:** 后续修改训练循环或新增多卡训练时，注意 DDP 下所有 rank 的状态必须保持一致，避免单 rank 更新状态后再 broadcast 的模式——优先让所有 rank 独立计算相同结果。
