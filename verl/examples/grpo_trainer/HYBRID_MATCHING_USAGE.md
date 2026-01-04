# 混合匹配策略使用指南

##  您的需求

> "先用方法2匹配，如果没有则用方法1匹配。如果都没有则跳过这条数据，并且打印匹配失败。"

## 解决方案：混合匹配 COT Loader

已为您创建：`hybrid_match_cot_loader.py`

### 匹配策略（按顺序）

```
1. 归一化匹配（方法2第一步）
   ↓ 失败
2. 模糊匹配（方法2第二步）
   ↓ 失败  
3. 精确匹配（方法1）
   ↓ 失败
4. 跳过数据 + 打印日志
```

## 快速使用

### 配置方式

```bash
python3 -m verl.trainer.main_ppo \
  # 使用支持COT的数据集类
  +data.custom_cls.path=examples.grpo_trainer.gsm8k_dataset_with_cot \
  +data.custom_cls.name=GSM8KParquetDatasetWithCOT \
  data.train_files=/nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet \
  \
  # 启用混合匹配COT
  +actor_rollout_ref.rollout.cot_augmentation.enable=true \
  +actor_rollout_ref.rollout.cot_augmentation.loader_path=examples.grpo_trainer.hybrid_match_cot_loader \
  +actor_rollout_ref.rollout.cot_augmentation.cot_file_path=/nas/dhl/CVAE/Datasets/GSM8K/processed/train_k_shot_GSM8K.jsonl \
  \
  # 混合匹配配置
  +actor_rollout_ref.rollout.cot_augmentation.fuzzy_threshold=0.95 \
  +actor_rollout_ref.rollout.cot_augmentation.skip_on_mismatch=true \
  +actor_rollout_ref.rollout.cot_augmentation.verbose=true \
  \
  # GRPO配置
  actor_rollout_ref.rollout.n=8 \
  algorithm.adv_estimator=grpo
```

### 关键参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `fuzzy_threshold` | 模糊匹配阈值（0-1） | 0.95（严格）、0.90（中等）|
| `skip_on_mismatch` | 匹配失败时是否跳过 | true（推荐）|
| `verbose` | 是否打印详细日志 | true（调试时）、false（生产）|

## 工作原理

### 示例 1：归一化匹配成功

**训练数据**：
```
"Randy has 60 mango trees.  "  # 末尾有空格
```

**COT数据**：
```
"Randy has 60 mango trees."     # 末尾无空格
```

**匹配过程**：
```
1. 归一化:
   训练: "randy has 60 mango trees" 
   COT:   "randy has 60 mango trees"
   → 匹配成功！✓
```

**日志输出**：
```
✓ 归一化匹配成功
```

---

### 示例 2：模糊匹配成功

**训练数据**：
```
"Randy has 60 mango trees on his farm. How many trees?"
```

**COT数据**：
```
"Randy has 60 mango trees. How many trees in total?"
```

**匹配过程**：
```
1. 归一化匹配: 失败（文本不完全相同）
2. 模糊匹配:
   相似度: 0.87
   阈值: 0.85
   → 匹配成功！≈
```

**日志输出**：
```
≈ 模糊匹配成功 (相似度: 0.870)
   查询: Randy has 60 mango trees on his farm. How many trees?
   匹配: Randy has 60 mango trees. How many trees in total?
```

---

### 示例 3：精确匹配成功

**训练数据**：
```
"Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all on his farm?"
```

**COT数据**：
```
完全相同的问题文本
```

**匹配过程**：
```
1. 归一化匹配: 失败（因为文本本身就完全一致，归一化后也一样，但可能在字典里用的是原始key）
2. 模糊匹配: 跳过（因为要遍历所有问题，较慢）
3. 精确匹配: 成功！✓
```

**日志输出**：
```
✓ 精确匹配成功
```

---

### 示例 4：完全匹配失败 → 跳过

**训练数据**：
```
"Sarah has 100 apples..."  # 这个问题不在COT数据中
```

**匹配过程**：
```
1. 归一化匹配: 失败
2. 模糊匹配: 失败（最高相似度 < 阈值）
3. 精确匹配: 失败
→ 跳过这条数据 ⚠️
```

**日志输出**：
```
❌ 匹配失败 - 跳过数据:
   问题: Sarah has 100 apples...
```

**结果**：这个问题的所有rollout都**不会添加COT**，使用原始prompt进行训练。

## 匹配统计

训练过程中或结束后，查看匹配统计：

```python
# 在训练循环中可以定期打印
loader.print_stats()
```

**输出示例**：
```
============================================================
COT匹配统计
============================================================
归一化匹配:  5234 / 7473 ( 70.05%)
模糊匹配:    1856 / 7473 ( 24.83%)
精确匹配:     312 / 7473 (  4.17%)
匹配失败:      71 / 7473 (  0.95%)
跳过数据:      71
============================================================
总体成功率: 99.05%
============================================================
```

## 调优建议

### 1. 调整模糊匹配阈值

**症状：匹配率太低**
```
匹配失败: 2000 / 7473 (26.76%)
```

**解决方案：降低阈值**
```bash
+actor_rollout_ref.rollout.cot_augmentation.fuzzy_threshold=0.90  # 从0.95降到0.90
```

---

**症状：模糊匹配到错误的问题**
```
≈ 模糊匹配成功 (相似度: 0.82)
   查询: Randy has 60 mango trees...
   匹配: Sarah has 50 apple trees...  # 完全不同的问题！
```

**解决方案：提高阈值**
```bash
+actor_rollout_ref.rollout.cot_augmentation.fuzzy_threshold=0.98  # 从0.95提高到0.98
```

### 2. 控制日志详细程度

**训练开始时：verbose=true**
```bash
+actor_rollout_ref.rollout.cot_augmentation.verbose=true
```
- 查看每个匹配的详细信息
- 验证匹配质量
- 发现问题

**正式训练时：verbose=false**
```bash
+actor_rollout_ref.rollout.cot_augmentation.verbose=false
```
- 减少日志输出
- 提高训练速度
- 只在匹配失败时打印

### 3. 决定是否跳过失败数据

**skip_on_mismatch=true（默认，推荐）**
```bash
+actor_rollout_ref.rollout.cot_augmentation.skip_on_mismatch=true
```
- 匹配失败的问题不添加COT
- 使用原始prompt训练
- **优点**：避免错误匹配
- **缺点**：部分数据没有COT增强

**skip_on_mismatch=false（不推荐）**
```bash
+actor_rollout_ref.rollout.cot_augmentation.skip_on_mismatch=false
```
- 匹配失败时返回空COT（实际效果和true一样）
- 不推荐，因为没有实际区别

## 与其他方案对比

| 特性 | 精确匹配 | 归一化匹配 | 模糊匹配 | 混合匹配（本方案）|
|------|---------|-----------|---------|-----------------|
| 容错性 | ❌ 低 | ✓ 中 | ✓✓ 高 | ✓✓✓ 最高 |
| 准确性 | ✓✓✓ 最高 | ✓✓ 高 | ✓ 中 | ✓✓ 高 |
| 速度 | ✓✓✓ 最快 | ✓✓ 快 | ❌ 慢 | ✓ 中等 |
| 适用场景 | 文本完全一致 | 格式差异 | 内容差异 | 所有场景 |
| 成功率 | 低-中 | 中-高 | 高 | 最高 |

## 完整示例

### 单数据集（GSM8K）

```bash
python3 -m verl.trainer.main_ppo \
  data.train_files=/nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet \
  data.val_files=/nas/dhl/Datasets/my_Datasets/gsm8k/test.parquet \
  \
  +data.custom_cls.path=examples.grpo_trainer.gsm8k_dataset_with_cot \
  +data.custom_cls.name=GSM8KParquetDatasetWithCOT \
  \
  +actor_rollout_ref.rollout.cot_augmentation.enable=true \
  +actor_rollout_ref.rollout.cot_augmentation.loader_path=examples.grpo_trainer.hybrid_match_cot_loader \
  +actor_rollout_ref.rollout.cot_augmentation.cot_file_path=/nas/dhl/CVAE/Datasets/GSM8K/processed/train_k_shot_GSM8K.jsonl \
  +actor_rollout_ref.rollout.cot_augmentation.fuzzy_threshold=0.95 \
  +actor_rollout_ref.rollout.cot_augmentation.skip_on_mismatch=true \
  +actor_rollout_ref.rollout.cot_augmentation.verbose=true \
  \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
  actor_rollout_ref.rollout.n=8 \
  algorithm.adv_estimator=grpo \
  trainer.n_gpus_per_node=8
```

### 多数据集

对于多数据集，需要在 `multi_dataset_cot_loader.py` 中也实现混合匹配策略。

或者，您可以为每个数据集使用单独的混合匹配loader（在multi_dataset_cot_loader内部调用hybrid_match）。

## 验证脚本

使用验证脚本检查匹配情况：

```bash
python3 examples/grpo_trainer/verify_question_matching.py \
  --train_file /nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet \
  --cot_file /nas/dhl/CVAE/Datasets/GSM8K/processed/train_k_shot_GSM8K.jsonl \
  --fuzzy_threshold 0.95 \
  --sample_size 20
```

**输出示例**：
```
Sample 1:
  Train ID: 100
  Train Q: Randy has 60 mango trees on his farm...
  Match: ✓ Exact
  COT ID: 12
  COT Count: 4

Sample 2:
  Train ID: 101
  Train Q: Grace just started her own business...
  Match: ≈ Fuzzy (0.960)
  COT ID: 45
  COT Count: 4

...

Full dataset statistics:
Exact matches:  5234 / 7473 (70.05%)
Fuzzy matches:  2168 / 7473 (29.00%)
No matches:       71 / 7473 ( 0.95%)
Total matched:  7402 / 7473 (99.05%)

Recommendations:
✓ Excellent matching rate!
  → Use hybrid matching with fuzzy_threshold=0.95
```

## 总结

### ✅ 已实现

- ✓ 混合匹配策略（归一化 → 模糊 → 精确）
- ✓ 失败时跳过数据
- ✓ 详细的匹配日志
- ✓ 匹配统计信息
- ✓ 与现有代码无缝集成

### 🎯 使用建议

1. **开发阶段**：
   - `verbose=true` 查看详细匹配信息
   - `fuzzy_threshold=0.95` 严格匹配
   - 使用验证脚本检查匹配率

2. **调优阶段**：
   - 根据匹配率调整阈值
   - 分析失败案例
   - 优化COT数据质量

3. **生产训练**：
   - `verbose=false` 减少日志
   - 使用调优好的阈值
   - 定期查看匹配统计

### 📁 相关文件

- 实现代码：`hybrid_match_cot_loader.py`
- 验证脚本：`verify_question_matching.py`
- 详细文档：`QUESTION_MATCHING_SOLUTIONS.md`

现在您可以高效地匹配COT数据，即使ID不一致也没问题！🎉

