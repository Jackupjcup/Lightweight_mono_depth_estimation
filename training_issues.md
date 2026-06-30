# 训练问题记录

## 现状 (2026-06-16)

- LMDB 路径: `data/tartanground.lmdb`, 315GB, 2,941,016 entries
- 每帧 4 个 key: `rgb`, `depth`, `pose`, `scale`
- 共 735,254 帧, 来自之前的构建进程（中途中断, 未生成 index JSON）
- `.scales_cache.json` 不存在
- `tartanground_index.json`, `tartanground_train.json`, `tartanground_val.json` 均未生成

## 问题 1: scale 存储位置不一致

**旧代码**: scale 写入 LMDB (`{frame_key}/scale`, 8 bytes float64)
**当前代码**: scale 不写入 LMDB, 通过 `process_trajectory` 计算后存入 `.scales_cache.json`, 最终写入 index JSON
**Dataset 读取端**: 从 JSON 读 scale (`s.get("scale", 1.0)`), 不读 LMDB 中的 scale

**影响**: resume 跳过的 trajectory 不会执行 `process_trajectory`, 其 scale 不会进入 `all_scales`, 导致生成的 JSON 中这些 sample 缺少 `scale` 字段, 训练时回退到 1.0（错误值）。

**修复方案**: LMDB 和 JSON 都生成完毕后, 写一次性脚本:
1. 遍历 JSON 中缺少 `scale` 的 sample, 从 LMDB 读取 `{frame_key}/scale` 补入 JSON
2. 新处理的 trajectory 的 scale 从 `.scales_cache.json` 读取（build 过程中会生成）
3. 可选: 从 LMDB 删除 `/scale` key（不释放磁盘空间, 仅为一致性）

## 问题 2: 扫描缓存

脚本每次启动都要扫描 TOS 上 739 个 trajectory 目录, TOS 延迟高（单次 ls ~12s）, 扫描耗时长。

**已修复**: 在 `main()` 中加入 `SCAN_CACHE` (`data/.scan_cache.json`) 逻辑, 首次扫描后缓存结果, 后续启动直接加载。

## Resume 安全性

已确认 resume 逻辑（检查第一帧 rgb key 是否存在）是安全的:
- `process_trajectory` 的所有 kv_pairs 在一个 `env.begin(write=True)` 事务中写入
- LMDB 事务原子性保证: commit 前所有 key 对外不可见, 崩溃则全部回滚
- 第一帧存在 = 整个 trajectory 已完整提交

## TODO (LMDB)

- [ ] build 完成后, 补全 JSON 中缺失的 scale 值
- [ ] 确认 `--add-ray` 和 `--add-sky` 步骤是否需要执行

---

## 已修复的训练问题 (2026-06-17)

### 已修复 8: confidence bf16 溢出导致 depth_loss = nan

`depth_conf` 和 `ray_conf` 使用 `expp1` 激活（`exp(x) + 1`），在 bf16 autocast 下当 raw logit `x > ~11` 时 `exp(x)` 溢出为 inf（bf16 max ≈ 65504）。cast 到 float32 后仍为 inf，loss 计算 `inf * error - 0.2 * log(inf)` = `inf - inf` = nan。

**现象**: v0_tenth 训练 epoch 0 step 1550/1600 出现 `D=nan, Loss=nan`，step 1650 自动恢复（GradScaler 跳过了 nan 步的参数更新）。

**已修复**: `objective_losses.py` 中 `depth_loss` 和 `ray_loss` 入口加 `conf.clamp(min=1e-6, max=1e4)`，防止 bf16 inf 透传和 log(0)。

### 已修复 7: valid_mask 改为 depth-based（去掉 sky 依赖）

原方案用 DA3 sky 推理结果做 valid_mask（需 `--add-sky` 步骤 ~2h GPU + ~110GB LMDB 空间）。改为直接用 `depth < depth_cap`（默认 100m）生成 mask。TartanGround 合成数据中天空深度远超 100m，效果等价。

**已修复**: dataloader 去掉 sky 读取，在 clip 前生成 mask；config `sky_threshold` → `depth_cap`。

### 已修复 6: LMDB MapResizedError (多 worker DataLoader)

`lmdb.open()` 未指定 `map_size`，315GB 数据库在 num_workers>0 时多进程并发打开触发 `MDB_MAP_RESIZED`。

**已修复**: `map_size=1 << 40` (1TB 虚拟地址空间，readonly 模式不实际分配内存)。

### 已修复 5: phase2_loss 中 bf16/float32 dtype 不匹配

Model 在 autocast(bf16) 下前向传播，部分输出为 bf16（qvec, t, ray, ray_conf），部分为 float32（depth）。`point_cloud_loss` 中 `matmul(R_pred[bf16], pixel_dirs[f32])` 崩溃。

**已修复**: 在 `phase2_loss` 入口统一将所有 student 输出转 `.float()`，确保 loss 计算在全精度下进行。

### 已修复 1: teacher.py autocast device_type 错误

`torch.autocast(device_type=self._device)` 传了 `"cuda:0"` 而不是 `"cuda"`，会导致运行时报错。

**已修复**: 改为 `torch.autocast(device_type="cuda")`。

### 已修复 2: qvec_to_rotmat 假设单位四元数

Student `qvec_to_rotmat` 硬编码系数 `2`，假设输入是单位四元数。但 `fc_qvec` 输出未归一化，训练早期四元数远离单位球时 `point_cloud_loss` 中旋转矩阵不正确。

**已修复**: 改用 DA3 的 `two_s = 2.0 / (q·q).sum(-1)` 隐式归一化，对非单位四元数鲁棒。

### 已修复 3: dataloader fov 顺序反了

Dataloader 输出 `[fov_w, fov_h]`，DA3 convention 为 `[fov_h, fov_w]`。图像非正方形 (644×476)，fov_w ≈ 90.2° 而 fov_h ≈ 73.3°，值不同会导致 camera loss 监督信号交叉。

**已修复**: 调换 `gt_fov` 顺序为 `[fov_h, fov_w]`，与 DA3 `transform.py:36` 一致。

### 已修复 4: dataloader depth 未截断

深度值可能超过 100m（远处/天空），需要 clamp(max=100) 后再除以 scale。

**已修复**: 解码后立即 `np.clip(depth, None, 100.0)`。

---

## 待观察的训练问题

### 待观察 1: StudentDualDPT aux head 旁路死参数

`output_conv1_aux[0:2]` 每次 forward 都计算但输出丢弃（无梯度），`output_conv2_aux[0:2]` 从未被调用（纯死参数）。这些参数被 AdamW weight decay 无意义地衰减。

**DA3 行为一致**: DA3 `dualdpt.py` 也是同样的设计，只用 `[-1]` 层。

**影响**: 不影响训练正确性或收敛，仅浪费少量显存和 optimizer 状态。当前 batch_size=4 下可忽略。

**后续优化选项**:
- A. 从 optimizer 排除这些参数
- B. 裁掉 `output_conv1_aux[0:2]` 和 `output_conv2_aux[0:2]`，`aux_levels` 改为 1
- C. 保持原样（当前选择）

### 待观察 2: fast_depth_model.py 始终计算 proj_feats

即使 `return_distill_feats=False`，`self.head` 仍以 `return_projected_feats=True` 调用。推理时浪费计算。训练时无影响（总是需要 proj_feats）。

### 待观察 3: Phase 1 使用 LMDB 数据集加载多余字段

Phase 1 只用 `batch["image"]`，但 `TartanGroundLMDBDataset` 每帧还解压 depth/pose、计算 ray/cam_params。可在 Phase 1 用 `ImageFolderDataset` 或加 `phase` 参数跳过。
