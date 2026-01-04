# COT功能快速验证指南

## 问题

> "我想现在就试试，试试这个 k-cot有没有初始化成功，不想初始化模型和通讯这些，只想验证我的example cot这部分是否正确？"

## 解决方案：快速验证脚本

**不需要启动训练！** 只测试COT加载和匹配功能。

## 方式 1️⃣：超快速验证（推荐）

### 使用默认配置

```bash
cd /Users/denghuilin/Desktop/verl/examples/grpo_trainer
python quick_test_cot.py
```

**只需5秒！** 测试5个样本，立即知道结果。

### 自定义配置

```bash
python quick_test_cot.py \
  --train_file /nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet \
  --cot_file /nas/dhl/CVAE/Datasets/GSM8K/processed/train_k_shot_GSM8K.jsonl \
  --num_samples 10 \
  --fuzzy_threshold 0.95 \
  --num_repeats 4
```

### 输出示例

```
================================================================================
                            COT匹配快速验证
================================================================================

📁 训练数据: /nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet
📁 COT数据:  /nas/dhl/CVAE/Datasets/GSM8K/processed/train_k_shot_GSM8K.jsonl
🔢 测试样本: 5 个
🎯 模糊阈值: 0.95
🔄 Rollout数: 4

⏳ 正在加载COT数据...
✅ COT数据加载成功! 共 7473 个问题

⏳ 正在加载训练数据...
✅ 训练数据加载成功! 共 7473 个样本

================================================================================
                              开始测试匹配
================================================================================

▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 样本 1/5 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
问题ID: 0
问题内容: Randy has 60 mango trees on his farm...

✓ 归一化匹配成功
✅ 成功获取 4 个COT例子

第1个COT例子预览:
--------------------------------------------------------------------------------
Here's a similar example:

Question: Grace just started her own business...

Step-by-step solution:
Every two weeks, Grace will get 300*2=600 dollars...
--------------------------------------------------------------------------------

▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 样本 2/5 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
...

================================================================================
                               匹配统计
================================================================================

============================================================
COT匹配统计
============================================================
归一化匹配:     4 / 5 ( 80.00%)
模糊匹配:       1 / 5 ( 20.00%)
精确匹配:       0 / 5 (  0.00%)
匹配失败:       0 / 5 (  0.00%)
跳过数据:       0
============================================================
总体成功率: 100.00%
============================================================

================================================================================
                             结论与建议
================================================================================

✨ 测试样本匹配率: 100.0%

✅ 匹配率优秀! COT配置正确，可以开始训练！

🚀 下一步: 运行完整训练命令
```

---

## 方式 2️⃣：详细验证

### 基础测试

```bash
python test_cot_matching.py
```

会测试前10个样本，然后询问是否测试完整数据集。

### 输出内容

- ✓ COT Loader初始化状态
- ✓ 每个样本的匹配详情
- ✓ 匹配统计信息
- ✓ 完整数据集匹配率（可选）

---

## 方式 3️⃣：详细分析（最全面）

### 使用验证脚本

```bash
python verify_question_matching.py \
  --train_file /nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet \
  --cot_file /nas/dhl/CVAE/Datasets/GSM8K/processed/train_k_shot_GSM8K.jsonl \
  --sample_size 10 \
  --fuzzy_threshold 0.95
```

### 功能

- 显示每个样本的匹配类型（精确/模糊/失败）
- 计算完整数据集的匹配率
- 给出配置建议

---

## 三种方式对比

| 方式 | 速度 | 详细程度 | 适用场景 |
|------|------|---------|----------|
| quick_test_cot.py | ⚡️ 5秒 | 中 | **快速验证**（推荐首选）|
| test_cot_matching.py | 🐇 1分钟 | 高 | 详细调试 |
| verify_question_matching.py | 🐢 5分钟 | 最高 | 全面分析 |

---

## 常见问题

### Q1: 文件路径错误怎么办？

**错误信息**：
```
❌ 文件不存在: [Errno 2] No such file or directory: '/nas/dhl/...'
```

**解决方案**：
```bash
# 检查文件是否存在
ls -lh /nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet
ls -lh /nas/dhl/CVAE/Datasets/GSM8K/processed/train_k_shot_GSM8K.jsonl

# 如果路径不对，使用正确的路径
python quick_test_cot.py \
  --train_file /你的/实际/路径/train.parquet \
  --cot_file /你的/实际/路径/cot.jsonl
```

### Q2: 匹配率很低怎么办？

**情况1：匹配率 50-80%**
```bash
# 降低阈值
python quick_test_cot.py --fuzzy_threshold 0.90
```

**情况2：匹配率 < 50%**
```bash
# 大幅降低阈值
python quick_test_cot.py --fuzzy_threshold 0.85

# 或者检查数据是否对应
python verify_question_matching.py --sample_size 20
```

### Q3: 想测试多个数据集怎么办？

**GSM8K**：
```bash
python quick_test_cot.py \
  --train_file /nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet \
  --cot_file /nas/dhl/CVAE/Datasets/GSM8K/processed/train_k_shot_GSM8K.jsonl
```

**MATH**：
```bash
python quick_test_cot.py \
  --train_file /path/to/math/train.parquet \
  --cot_file /nas/dhl/CVAE/Datasets/MATH/processed/train_k_shot_MATH.jsonl
```

**NuminaMath**：
```bash
python quick_test_cot.py \
  --train_file /path/to/numina/train.parquet \
  --cot_file /nas/dhl/CVAE/Datasets/NuminaMath-CoT/processed/train_k_shot_MATH.jsonl
```

---

## 验证通过后的下一步

### ✅ 如果匹配率 > 95%

**恭喜！可以开始训练了：**

```bash
python3 -m verl.trainer.main_ppo \
  +data.custom_cls.path=examples.grpo_trainer.gsm8k_dataset_with_cot \
  +data.custom_cls.name=GSM8KParquetDatasetWithCOT \
  data.train_files=/nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet \
  \
  +actor_rollout_ref.rollout.cot_augmentation.enable=true \
  +actor_rollout_ref.rollout.cot_augmentation.loader_path=examples.grpo_trainer.hybrid_match_cot_loader \
  +actor_rollout_ref.rollout.cot_augmentation.cot_file_path=/nas/dhl/CVAE/Datasets/GSM8K/processed/train_k_shot_GSM8K.jsonl \
  +actor_rollout_ref.rollout.cot_augmentation.fuzzy_threshold=0.95 \
  +actor_rollout_ref.rollout.cot_augmentation.skip_on_mismatch=true \
  +actor_rollout_ref.rollout.cot_augmentation.verbose=false \
  \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
  actor_rollout_ref.rollout.n=8 \
  algorithm.adv_estimator=grpo \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1
```

### ⚠️ 如果匹配率 80-95%

**调整阈值后再训练：**

```bash
# 使用更宽松的阈值
+actor_rollout_ref.rollout.cot_augmentation.fuzzy_threshold=0.90
```

### ❌ 如果匹配率 < 80%

**请先排查问题：**

1. 确认训练数据和COT数据对应同一个数据集
2. 查看失败样本的问题文本
3. 考虑是否需要重新处理数据

---

## 完整验证流程

```bash
# 第1步：快速验证（5秒）
python quick_test_cot.py --num_samples 5

# 如果通过 → 开始训练 ✅
# 如果不通过 → 继续第2步

# 第2步：详细验证（1分钟）
python test_cot_matching.py

# 查看详细匹配情况，调整参数

# 第3步：全面分析（5分钟）
python verify_question_matching.py --sample_size 20

# 获得配置建议

# 第4步：应用建议，重新验证
python quick_test_cot.py --fuzzy_threshold 0.90

# 通过后 → 开始训练 🚀
```

---

## 总结

| 验证方式 | 命令 | 用时 |
|---------|------|------|
| **快速验证** ⭐ | `python quick_test_cot.py` | 5秒 |
| 详细验证 | `python test_cot_matching.py` | 1分钟 |
| 全面分析 | `python verify_question_matching.py` | 5分钟 |

**推荐流程**：先用 `quick_test_cot.py` 快速验证，通过后直接开始训练！

有问题再用其他两个脚本详细排查。

祝验证顺利！🎉

