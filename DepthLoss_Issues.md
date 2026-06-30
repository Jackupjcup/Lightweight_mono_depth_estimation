# DepthLoss Issues — 深度 loss 在 LR 峰值处"清零"塌缩

> 记录日期: 2026-06-22
> 状态: 已定位根因,修复方案待实测梯度后落地(尚未改代码)

## 1. 现象

深度 loss `D` 在训练中**突然从一个小值(~0.1)跳到 ~0.83,然后死死锁在 0.82–0.84 不动**(不是缓慢漂移,是一步跳变后变平线)。跨三个 run 复现,且这三个 run **数据集大小不同、超参不同、深度激活函数不同(exp / softplus)**,塌缩值却都统一在 ~0.8。

涉及 log:
- `work_dirs/v0_v1/20260620_035604/train.log`
- `work_dirs/v0_tenth/20260617_145121/train.log`
- `work_dirs/v0_tenth/20260618_104616/train.log`

## 2. 证据

- 塌缩都发生在 **学习率到达/接近峰值(4e-4 = base 1e-4 × world_size 4)** 时;warmup 长度不同(5/1/3 epoch),所以塌缩绑定的是"LR 到峰值",不是固定 epoch。在峰值待越久越容易触发(LOG1 在峰值 2 epoch 后才塌,LOG3 几乎一到峰值就塌)。
- 突变那一刻**只有 D 跳变**,其它分量正常:
  ```
  LOG1 Epoch7 Step182 LR=4.00e-04:
  D=0.8365  R=0.1053  P=0.0334  C=0.0350  G=0.0376   ← 只有 D 跳
  S1..S4=0.15/0.78/0.74/0.93  L1..L4 正常              ← backbone/特征健康
  ```
- 塌缩后 **R(ray)/P(point)/C(cam)/G(grad)/特征蒸馏/SSIM 全部继续正常**,只有 D 永久锁死 → **深度头单点塌缩,不是全局梯度爆炸**。
- 塌缩后 **G(梯度 loss)仍很低(~0.05)** → `∇pred ≈ ∇gt`,空间结构还在,坏的只是**绝对尺度/偏置**(`pred ≈ gt + 常数offset`)。

## 3. 根因

### 3.1 为什么是深度头脆弱
深度用 **exp / softplus 指数族激活**(`models/student_dpt.py`)。线性 L1 对预激活 `z` 的梯度 ∝ `pred = act(z)`,pred 一大梯度被指数放大。ray 头用 `linear` 激活(`student_dpt.py:313`),对同样抖动鲁棒 → 所以"换 exp 还是 softplus 都复现"(都是指数族,都放大)。

### 3.2 触发 = 峰值 LR + **没有梯度裁剪**
`tools/train.py:294-297` / `tools/train_v1.py` 训练步:
```python
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()        # ← 中间没有任何 clip_grad_norm_
scheduler.step()
```
完全没有梯度裁剪。高 LR 下一次梯度尖峰,经指数激活放大,就把深度头权重踢进退化区。

### 3.3 为什么不可逆 + 为什么统一锁在 0.8
深度 loss(`distillation/objective_losses.py:89-106`,`objective_losses_v1.py` 同形):
```
LD = mean_valid[ conf·|pred − gt| − λ·log(conf) ],  λ=0.2
```
- 塌缩后深度头**与输入解耦**(退化成常数级预测),无论 exp 还是 softplus 都饱和到同一类"input-decoupled"状态。
- conf 用 `softplusp1`(≥1),clamp [1e-6,1e4]。逐像素最优 conf 是 `λ/e`;塌缩后 `e≈0.8 > λ=0.2` → 最优 conf<1 不可行 → **conf 顶到地板(≈1)** → `−λ·log(1)=0` → **D ≈ mean|pred−gt|**。
- GT 用 `depth/scale`,`scale = mean‖P‖₂`(`data/tartanground_lmdb_dataset.py:94-109`,DA3 式尺度归一化)→ 归一化深度恒为 O(1),其平均绝对偏差 ≈ 0.8。
- 因此 **0.8 = 深度 L1 loss 的"零信息基线"**:任何与输入解耦的预测器在这套固定归一化下都给这个值。它由 **GT 归一化**决定,**与激活无关** → exp / softplus / 不同数据集都殊途同归到 0.8。
- 0.1 → 0.8 这一跳的幅度 = 深度头此前学到的全部有效信息,被一次尖峰清零。

## 4. 修复方案(讨论结论)

### 4.1 梯度裁剪 —— 用**全局**,不是只裁深度
```python
grad_norm = torch.nn.utils.clip_grad_norm_(raw_student.parameters(), max_norm=<TBD>)
```
- 尖峰体现在**总梯度范数**,全局裁剪等比缩小所有梯度,深度头的更新一并被按住,还顺带保护 ray/cam。
- 只裁深度头抓不住来自共享层(backbone/DPT fusion)的尖峰,非标准,不优先。
- **阈值不要拍脑袋**:目前**没有任何 log 记录过 grad_norm**,无实测数据。`max_norm=1.0` 只是经验值,可能偏小。
  - 正确流程:先用 `max_norm=inf`(只测不裁)+ 记录 grad_norm,跑过 LR 峰值区 → 看 p50/p90/p99/max → 把阈值设在"正常步不受影响、只切尾部尖峰"处(经验 ~3–5× 中位数)。
  - 离线测法(不需 DA3 老师):load `best_delta.pt` + 几个 val batch 跑 phase2_loss backward(phase2 用 GT),直接测 depth 项梯度范数。需要空闲 GPU。

### 4.2 log 空间深度 loss —— **可选,非必须**(偏离 DA3 配方)
当前(线性):`LD = mean[ conf·|pred−gt| − λ·log(conf) ]`
log 版:
```python
eps = 1e-3
e  = (torch.log(pred.clamp(min=eps)) - torch.log(gt.clamp(min=eps))).abs()  # = |log(pred/gt)|
LD = mean_valid[ conf·e − λ·log(conf) ]
```
影响:
- **根治指数激活病态梯度**:exp 时 `log(pred)=z`,loss=`|z−log(gt)|`,梯度=`conf·sign(...)` 有界,不再被指数放大。
- 变成**相对误差**,远近深度权重均衡(与 AbsRel/δ 一致)。
- 塌缩平台值会变(不再 0.8);需要正数 clamp;`point_cloud_loss`/`gradient_loss` 仍用线性深度不动;`lambda_c` 可能要微调。

### 4.3 为什么 DA3 用线性 L1 而不是 log(以及我方 loss 的劣势)
- DA3 在**尺度归一化空间**(`scale=mean‖P‖`)训练,深度 O(1)、动态范围小 → 线性 L1 良态、简单。log/scale-invariant 是为度量深度的大动态范围(0.1–100m)准备的,DA3 归一化后不需要。
- DA3 能稳,是因为它**有梯度裁剪、调好的 LR、大 batch**。
- 我方 loss 的劣势**不在公式**,而在:抽掉了 DA3 的稳定器(无裁剪、LR 线性 ×4=4e-4)+ 用了敏感的 exp/softplus 激活 → 线性 L1 的 `∝pred` 梯度被放大;再加 conf 退化盆地。即"**DA3 世界里没问题,你的设定里脆弱**"。

## 5. 能否"杜绝"?

**不能打包票。** 梯度裁剪 + log 拆掉了两个直接致塌机制(尖峰被踢飞 + 指数放大),实践中应能止住这三例,但残留风险仍在:
- conf 退化盆地仍存在;裁剪管幅度不管方向;bf16 精度、LR 峰值偏大等。
- 更稳还需叠加:**降/缓峰值 LR**(base 调小 / √world_size 替代线性 / 更长 warmup)、**驯化 conf 项**(`lambda_c` 调小、conf clamp 收紧、depth_weight warmup)、NaN/尖峰守卫。

## 6. 下一步行动(待执行,尚未改代码)

1. **先只加 `clip_grad_norm_(max_norm=inf)` + 记录 grad_norm 到日志/TensorBoard**(`train.py` + `train_v1.py`),**不真正裁、不碰 loss**。
2. 跑一段穿过 LR 峰值区,拿到 grad_norm 真实分布(p50/p90/p99/max)。
3. 据此定 `max_norm`(全局裁剪),开启真正裁剪。
4. 若裁剪 + 收 LR 后**仍复发**,再考虑换 log 空间深度 loss(4.2)。
5. 全程盯 grad_norm 与 D 在峰值区的表现做实证确认。

## 相关代码位置
- 训练步(无裁剪): `tools/train.py:294-297`,`tools/train_v1.py`
- 有效 LR 缩放: `tools/train.py:204`(`learning_rate × world_size`)
- 深度 loss: `distillation/objective_losses.py:89-106`(`train.py` 用),`distillation/objective_losses_v1.py`(`train_v1.py` 用)
- 深度/conf 激活: `models/student_dpt.py`(depth=exp/softplus,ray aux=linear,conf=softplusp1)
- 尺度归一化: `data/tartanground_lmdb_dataset.py:94-109`(`scale=mean‖P‖₂`)
- 已有但未启用的裁剪参考: `tools/train_distill.py`(`clip_grad_norm_(..., max_norm=inf)` + 打 `GN`)
