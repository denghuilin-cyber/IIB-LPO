# 多数据集GRPO COT增强 - 完整实现总结

## 您的需求回顾

### 原始需求
1. ✅ GRPO会对一个问题rollout多次（如4次）
2. ✅ 希望每次rollout使用不同的COT例子
3. ✅ COT例子从独立的JSONL文件中加载
4. ✅ **新增需求**：支持多个数据集，每个数据集有自己的COT文件

### 数据情况
- **训练数据**：
  - GSM8K: `/nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet`
  - MATH: `/path/to/math/train.parquet`
  - NuminaMath-CoT: `/path/to/numina/train.parquet`

- **COT数据**：
  - GSM8K: `/nas/dhl/CVAE/Datasets/GSM8K/processed/train_k_shot_GSM8K.jsonl`
  - MATH: `/nas/dhl/CVAE/Datasets/MATH/processed/train_k_shot_MATH.jsonl`
  - NuminaMath-CoT: `/nas/dhl/CVAE/Datasets/NuminaMath-CoT/processed/train_k_shot_MATH.jsonl`

### 核心挑战
❓ **如何确认一个问题来自哪个数据集，从而找到对应的COT？**

### 解决方案
✅ 在训练数据中添加 `dataset_name` 字段，建立 dataset_name → COT文件 的映射关系

## 完整文件清单

### 核心功能文件

| 文件 | 路径 | 说明 | 状态 |
|------|------|------|------|
| `grpo_cot_augmentation.py` | `verl/utils/` | COT增强器核心实现 | ✅ 已创建 |
| `ray_trainer.py` | `verl/trainer/ppo/` | 训练器（已集成多数据集支持） | ✅ 已修改 |

### 单数据集支持

| 文件 | 路径 | 说明 | 状态 |
|------|------|------|------|
| `gsm8k_cot_loader.py` | `examples/grpo_trainer/` | 单数据集COT加载器 | ✅ 已创建 |
| `gsm8k_dataset_with_cot.py` | `examples/grpo_trainer/` | 单数据集数据集类 | ✅ 已创建 |
| `run_gsm8k_with_cot.sh` | `examples/grpo_trainer/` | 单数据集训练脚本 | ✅ 已创建 |

### ⭐ 多数据集支持（新）

| 文件 | 路径 | 说明 | 状态 |
|------|------|------|------|
| `multi_dataset_cot_loader.py` | `examples/grpo_trainer/` | **多数据集COT加载器** | ✅ 已创建 |
| `multi_dataset_with_cot.py` | `examples/grpo_trainer/` | **多数据集数据集类** | ✅ 已创建 |
| `multi_dataset_config.yaml` | `examples/grpo_trainer/` | **多数据集配置文件** | ✅ 已创建 |
| `run_multi_dataset_grpo.sh` | `examples/grpo_trainer/` | **多数据集训练脚本** | ✅ 已创建 |
| `MULTI_DATASET_README.md` | `examples/grpo_trainer/` | **多数据集使用文档** | ✅ 已创建 |

### 文档和示例

| 文件 | 路径 | 说明 | 状态 |
|------|------|------|------|
| `QUICKSTART_CN.md` | `examples/grpo_trainer/` | 快速开始指南（单数据集） | ✅ 已创建 |
| `GSM8K_COT_USAGE.md` | `examples/grpo_trainer/` | 完整使用文档（单数据集） | ✅ 已创建 |
| `README_COT.md` | `examples/grpo_trainer/` | 项目总览 | ✅ 已创建 |
| `custom_cot_getter_example.py` | `examples/grpo_trainer/` | 自定义COT生成器示例 | ✅ 已创建 |

## 多数据集使用方法

### 方案一：命令行配置（推荐）

```bash
python3 -m verl.trainer.main_ppo \
  # 使用多数据集类
  +data.custom_cls.path=examples.grpo_trainer.multi_dataset_with_cot \
  +data.custom_cls.name=MultiDatasetWithCOT \
  \
  # 数据集配置（JSON格式）
  +data.dataset_configs='[
    {
      "name": "gsm8k",
      "files": ["/nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet"],
      "prompt_key": "question",
      "answer_key": "answer"
    },
    {
      "name": "math",
      "files": ["/path/to/math/train.parquet"],
      "prompt_key": "problem",
      "answer_key": "solution"
    },
    {
      "name": "numina",
      "files": ["/path/to/numina/train.parquet"],
      "prompt_key": "problem",
      "answer_key": "solution"
    }
  ]' \
  \
  # 启用多数据集COT增强
  +actor_rollout_ref.rollout.cot_augmentation.enable=true \
  +actor_rollout_ref.rollout.cot_augmentation.use_multi_dataset=true \
  \
  # COT文件映射（JSON格式）
  +actor_rollout_ref.rollout.cot_augmentation.cot_file_mapping='{
    "gsm8k": "/nas/dhl/CVAE/Datasets/GSM8K/processed/train_k_shot_GSM8K.jsonl",
    "math": "/nas/dhl/CVAE/Datasets/MATH/processed/train_k_shot_MATH.jsonl",
    "numina": "/nas/dhl/CVAE/Datasets/NuminaMath-CoT/processed/train_k_shot_MATH.jsonl"
  }' \
  \
  # GRPO配置
  actor_rollout_ref.rollout.n=4 \
  algorithm.adv_estimator=grpo \
  \
  # 其他参数...
  trainer.n_gpus_per_node=8
```

### 方案二：使用配置文件（简洁）

**1. 编辑配置文件** `examples/grpo_trainer/multi_dataset_config.yaml`

```yaml
datasets:
  - name: gsm8k
    files:
      - /nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet
    prompt_key: question
    
  - name: math
    files:
      - /path/to/math/train.parquet
    prompt_key: problem
    
  - name: numina
    files:
      - /path/to/numina/train.parquet
    prompt_key: problem

cot_files:
  gsm8k: /nas/dhl/CVAE/Datasets/GSM8K/processed/train_k_shot_GSM8K.jsonl
  math: /nas/dhl/CVAE/Datasets/MATH/processed/train_k_shot_MATH.jsonl
  numina: /nas/dhl/CVAE/Datasets/NuminaMath-CoT/processed/train_k_shot_MATH.jsonl
```

**2. 运行训练**

```bash
bash examples/grpo_trainer/run_multi_dataset_grpo.sh
```

## 核心概念

### 1. 数据集标识 (`dataset_name`)

每个训练样本都包含一个 `dataset_name` 字段：

```python
{
    "input_ids": [...],
    "attention_mask": [...],
    "dataset_name": "gsm8k",  # ⭐ 关键字段
    "question": "Randy has 60 mango trees...",
    "question_id": 12
}
```

### 2. COT文件映射

建立数据集名称到COT文件的映射：

```python
{
    "gsm8k": "/path/to/gsm8k_cot.jsonl",
    "math": "/path/to/math_cot.jsonl",
    "numina": "/path/to/numina_cot.jsonl"
}
```

### 3. 自动匹配流程

```
训练循环中的每个样本:
  1. 读取 dataset_name → "gsm8k"
  2. 查找映射 → "/path/to/gsm8k_cot.jsonl"
  3. 从该文件加载COT → [COT1, COT2, COT3, COT4]
  4. 为4次rollout分别使用 → COT1, COT2, COT3, COT4
```

## 工作原理示例

假设一个GSM8K问题会被rollout 4次：

### 传统GRPO（不使用COT）
```
4次rollout都使用相同的prompt:
"Randy has 60 mango trees on his farm..."
```

### 单数据集COT GRPO
```
Rollout 1: COT例子1 + "Randy has 60 mango trees..."
Rollout 2: COT例子2 + "Randy has 60 mango trees..."
Rollout 3: COT例子3 + "Randy has 60 mango trees..."
Rollout 4: COT例子4 + "Randy has 60 mango trees..."

所有COT例子来自同一个文件
```

### ⭐ 多数据集COT GRPO（新方案）
```
GSM8K问题:
  dataset_name = "gsm8k"
  → 从 gsm8k_cot.jsonl 加载COT
  → Rollout 1-4 使用不同的GSM8K COT例子

MATH问题:
  dataset_name = "math"
  → 从 math_cot.jsonl 加载COT
  → Rollout 1-4 使用不同的MATH COT例子

Numina问题:
  dataset_name = "numina"
  → 从 numina_cot.jsonl 加载COT
  → Rollout 1-4 使用不同的Numina COT例子
```

## 配置检查清单

在训练前，请确认：

- [ ] ✅ 所有数据集文件路径正确
- [ ] ✅ 所有COT文件路径正确
- [ ] ✅ 数据集配置中的 `name` 与COT映射中的key一致
- [ ] ✅ 正确设置了 `prompt_key`（不同数据集可能不同）
- [ ] ✅ 使用了 `MultiDatasetWithCOT` 数据集类
- [ ] ✅ 启用了 `use_multi_dataset=true`

## 验证训练配置

### 期望看到的日志

```
✅ Loading dataset: gsm8k
✅   Loaded 7473 samples
✅ Loading dataset: math
✅   Loaded 5000 samples
✅ Loading dataset: numina
✅   Loaded 3000 samples
✅ Loaded 15473 total samples from 3 datasets

✅ Initializing multi-dataset COT loader...
✅ Loading COT data for gsm8k from ...
✅ Loading COT data for math from ...
✅ Loading COT data for numina from ...
✅ Multi-dataset COT loader initialized successfully
✅ Loaded COT data for 3 datasets: ['gsm8k', 'math', 'numina']
  - gsm8k: 7473 questions
  - math: 5000 questions
  - numina: 3000 questions
```

## 常见问题

### Q1: 如何确认每个问题使用了正确的COT？

**A:** 可以在训练开始时打印几个样本：

```python
# 在 multi_dataset_with_cot.py 的 __getitem__ 中临时添加
if index < 5:  # 打印前5个样本
    print(f"Sample {index}:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Question: {question[:50]}...")
```

### Q2: 不同数据集的问题字段名不同怎么办？

**A:** 在配置中为每个数据集指定正确的 `prompt_key`：

```python
{
    "name": "gsm8k",
    "prompt_key": "question",  # GSM8K使用"question"
},
{
    "name": "math",
    "prompt_key": "problem",   # MATH使用"problem"
}
```

### Q3: 可以只为部分数据集启用COT吗？

**A:** 可以！只在 `cot_file_mapping` 中包含需要COT的数据集：

```python
cot_file_mapping = {
    "gsm8k": "/path/to/gsm8k_cot.jsonl",
    "math": "/path/to/math_cot.jsonl",
    # numina 不包含，该数据集不会使用COT
}
```

### Q4: 如何调整不同数据集的采样比例？

**A:** 使用 WeightedRandomSampler（参考 MULTI_DATASET_README.md 的高级用法）

## 与单数据集方案的对比

| 特性 | 单数据集方案 | 多数据集方案 |
|------|-------------|-------------|
| 数据集数量 | 1个 | 多个 |
| COT文件 | 1个文件 | 每个数据集1个文件 |
| 数据集类 | `GSM8KParquetDatasetWithCOT` | `MultiDatasetWithCOT` |
| COT Loader | `gsm8k_cot_loader` | `multi_dataset_cot_loader` |
| 配置参数 | `cot_file_path` | `cot_file_mapping` + `use_multi_dataset` |
| 适用场景 | 单一数据集训练 | 混合多数据集训练 |

## 快速参考

### 关键配置参数

```bash
# 必需参数
+data.custom_cls.name=MultiDatasetWithCOT
+data.dataset_configs='[...]'  # 数据集列表
+actor_rollout_ref.rollout.cot_augmentation.enable=true
+actor_rollout_ref.rollout.cot_augmentation.use_multi_dataset=true
+actor_rollout_ref.rollout.cot_augmentation.cot_file_mapping='{...}'  # COT文件映射

# 可选参数
+actor_rollout_ref.rollout.cot_augmentation.match_by=question  # 或 "id"
+actor_rollout_ref.rollout.cot_augmentation.sampling_strategy=sequential
+actor_rollout_ref.rollout.cot_augmentation.use_full_cot=true
```

### 关键文件

- **多数据集Loader**: `examples/grpo_trainer/multi_dataset_cot_loader.py`
- **多数据集Dataset**: `examples/grpo_trainer/multi_dataset_with_cot.py`
- **配置文件**: `examples/grpo_trainer/multi_dataset_config.yaml`
- **训练脚本**: `examples/grpo_trainer/run_multi_dataset_grpo.sh`
- **详细文档**: `examples/grpo_trainer/MULTI_DATASET_README.md`

## 总结

✅ **完整实现**：
- 单数据集 + 多数据集 双重支持
- 自动识别数据集来源
- 自动选择对应的COT文件
- 每次rollout使用不同的COT例子

✅ **易于使用**：
- 配置简单清晰
- 支持命令行或配置文件
- 详细的文档和示例

✅ **灵活扩展**：
- 可以自定义数据集处理
- 可以自定义COT选择策略
- 可以调整采样权重

现在您可以轻松地使用多个数据集进行GRPO训练，每个数据集使用自己的COT文件！🎉🎉🎉

---

**下一步建议**：
1. 根据您的实际数据路径修改配置文件
2. 运行小规模测试（几百步）验证配置
3. 检查日志确认多数据集和COT正确加载
4. 开始完整训练

祝训练成功！🚀

