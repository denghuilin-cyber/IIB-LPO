# GRPO 多数据集 COT 增强使用指南

## 问题背景

您有多个数学推理数据集（GSM8K、MATH、NuminaMath-COT），每个数据集的COT示例存储在不同的文件中：

- **GSM8K**: `/nas/dhl/CVAE/Datasets/GSM8K/processed/train_k_shot_GSM8K.jsonl`
- **MATH**: `/nas/dhl/CVAE/Datasets/MATH/processed/train_k_shot_MATH.jsonl`
- **NuminaMath-COT**: `/nas/dhl/CVAE/Datasets/NuminaMath-CoT/processed/train_k_shot_MATH.jsonl`

**核心问题**：在训练时，如何自动识别每个问题来自哪个数据集，并使用对应的COT文件？

## 解决方案

### 核心思路

1. **数据集标识**：在训练数据中添加 `dataset_name` 字段，标识每个样本的来源
2. **COT映射**：配置 `dataset_name → COT文件路径` 的映射关系
3. **自动匹配**：训练时自动根据 `dataset_name` 选择正确的COT文件

### 工作流程

```
训练样本 → 包含 dataset_name 字段 → COT Augmenter
                                          ↓
                                    识别数据来源
                                          ↓
                              从对应的COT文件中获取示例
                                          ↓
                                    添加到prompt
```

## 快速开始

### 步骤 1: 准备数据集配置

创建或使用现有的多数据集配置：

```python
# 在训练脚本中或配置文件中
dataset_configs = [
    {
        "name": "gsm8k",  # 数据集标识符
        "files": ["/nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet"],
        "prompt_key": "question",  # 问题字段名
        "answer_key": "answer"
    },
    {
        "name": "math",
        "files": ["/path/to/math/train.parquet"],
        "prompt_key": "problem",  # MATH数据集可能用"problem"
        "answer_key": "solution"
    },
    {
        "name": "numina",
        "files": ["/path/to/numina/train.parquet"],
        "prompt_key": "problem",
        "answer_key": "solution"
    }
]
```

### 步骤 2: 配置COT文件映射

```python
cot_file_mapping = {
    "gsm8k": "/nas/dhl/CVAE/Datasets/GSM8K/processed/train_k_shot_GSM8K.jsonl",
    "math": "/nas/dhl/CVAE/Datasets/MATH/processed/train_k_shot_MATH.jsonl",
    "numina": "/nas/dhl/CVAE/Datasets/NuminaMath-CoT/processed/train_k_shot_MATH.jsonl"
}
```

### 步骤 3: 修改训练命令

```bash
python3 -m verl.trainer.main_ppo \
  # 使用多数据集类
  +data.custom_cls.path=examples.grpo_trainer.multi_dataset_with_cot \
  +data.custom_cls.name=MultiDatasetWithCOT \
  \
  # 数据集配置
  +data.dataset_configs='[
    {"name": "gsm8k", "files": ["/nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet"], "prompt_key": "question"},
    {"name": "math", "files": ["/path/to/math/train.parquet"], "prompt_key": "problem"},
    {"name": "numina", "files": ["/path/to/numina/train.parquet"], "prompt_key": "problem"}
  ]' \
  \
  # 启用多数据集COT
  +actor_rollout_ref.rollout.cot_augmentation.enable=true \
  +actor_rollout_ref.rollout.cot_augmentation.use_multi_dataset=true \
  +actor_rollout_ref.rollout.cot_augmentation.cot_file_mapping='{
    "gsm8k": "/nas/dhl/CVAE/Datasets/GSM8K/processed/train_k_shot_GSM8K.jsonl",
    "math": "/nas/dhl/CVAE/Datasets/MATH/processed/train_k_shot_MATH.jsonl",
    "numina": "/nas/dhl/CVAE/Datasets/NuminaMath-CoT/processed/train_k_shot_MATH.jsonl"
  }' \
  \
  # 其他参数...
  actor_rollout_ref.rollout.n=4 \
  algorithm.adv_estimator=grpo
```

或者使用配置文件：

```bash
python3 -m verl.trainer.main_ppo \
  --config-path examples/grpo_trainer \
  --config-name multi_dataset_config
```

## 详细说明

### 1. 数据集配置 (`dataset_configs`)

每个数据集需要提供：

| 字段 | 说明 | 示例 |
|------|------|------|
| `name` | **数据集标识符**（必需） | `"gsm8k"`, `"math"` |
| `files` | 数据文件路径列表 | `["/path/to/train.parquet"]` |
| `prompt_key` | 问题字段名 | `"question"`, `"problem"` |
| `answer_key` | 答案字段名（可选） | `"answer"`, `"solution"` |

**关键点**：
- `name` 必须与COT文件映射中的key一致
- 不同数据集可能使用不同的字段名（如GSM8K用`question`，MATH用`problem`）

### 2. COT文件映射 (`cot_file_mapping`)

格式：`{dataset_name: cot_file_path}`

```python
{
    "gsm8k": "/nas/dhl/CVAE/Datasets/GSM8K/processed/train_k_shot_GSM8K.jsonl",
    "math": "/nas/dhl/CVAE/Datasets/MATH/processed/train_k_shot_MATH.jsonl",
    "numina": "/nas/dhl/CVAE/Datasets/NuminaMath-CoT/processed/train_k_shot_MATH.jsonl"
}
```

**关键点**：
- Key必须与数据集配置中的`name`完全匹配
- Value是COT文件的完整路径
- 所有COT文件格式应该一致（都包含`selected_cots`字段）

### 3. 数据集名称匹配规则

系统使用以下规则匹配数据集和COT：

1. **自动小写化**：`"GSM8K"` → `"gsm8k"`（配置时建议直接用小写）
2. **去除空格**：`" gsm8k "` → `"gsm8k"`
3. **精确匹配**：必须完全一致才能匹配

### 4. 数据流程

```
1. 从 train.parquet 加载问题
   ↓
2. 自动添加 dataset_name 字段（来自配置）
   例如：{"question": "...", "dataset_name": "gsm8k"}
   ↓
3. 重复 (repeat) n 次（GRPO）
   ↓
4. COT Augmenter 读取 dataset_name
   ↓
5. 从 cot_file_mapping["gsm8k"] 加载COT
   ↓
6. 为每次rollout添加不同的COT例子
   ↓
7. 生成响应
```

## 配置文件方式

### 创建 YAML 配置文件

```yaml
# multi_dataset_config.yaml

datasets:
  - name: gsm8k
    files:
      - /nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet
    prompt_key: question
    answer_key: answer
    
  - name: math
    files:
      - /path/to/math/train.parquet
    prompt_key: problem
    answer_key: solution
    
  - name: numina
    files:
      - /path/to/numina/train.parquet
    prompt_key: problem
    answer_key: solution

cot_files:
  gsm8k: /nas/dhl/CVAE/Datasets/GSM8K/processed/train_k_shot_GSM8K.jsonl
  math: /nas/dhl/CVAE/Datasets/MATH/processed/train_k_shot_MATH.jsonl
  numina: /nas/dhl/CVAE/Datasets/NuminaMath-CoT/processed/train_k_shot_MATH.jsonl
```

### 使用配置文件

```bash
python3 -m verl.trainer.main_ppo \
  +data.custom_cls.path=examples.grpo_trainer.multi_dataset_with_cot \
  +data.custom_cls.name=MultiDatasetWithCOT \
  +data.dataset_config_file=examples/grpo_trainer/multi_dataset_config.yaml \
  +actor_rollout_ref.rollout.cot_augmentation.enable=true \
  +actor_rollout_ref.rollout.cot_augmentation.use_multi_dataset=true \
  +actor_rollout_ref.rollout.cot_augmentation.cot_config_file=examples/grpo_trainer/multi_dataset_config.yaml
```

## 验证配置

### 训练开始时的日志

正确配置后，您应该看到：

```
✅ Loading dataset: gsm8k
✅   Files: ['/nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet']
✅   Loaded 7473 samples

✅ Loading dataset: math
✅   Files: ['/path/to/math/train.parquet']
✅   Loaded 5000 samples

✅ Loading dataset: numina
✅   Files: ['/path/to/numina/train.parquet']
✅   Loaded 3000 samples

✅ Loaded 15473 total samples from 3 datasets

✅ Initializing multi-dataset COT loader...
✅ Loading COT data for gsm8k from /nas/dhl/CVAE/Datasets/GSM8K/processed/train_k_shot_GSM8K.jsonl...
✅ Loading COT data for math from /nas/dhl/CVAE/Datasets/MATH/processed/train_k_shot_MATH.jsonl...
✅ Loading COT data for numina from /nas/dhl/CVAE/Datasets/NuminaMath-CoT/processed/train_k_shot_MATH.jsonl...

✅ Multi-dataset COT loader initialized successfully
✅ Loaded COT data for 3 datasets: ['gsm8k', 'math', 'numina']
✅   - gsm8k: 7473 questions
✅   - math: 5000 questions
✅   - numina: 3000 questions
```

## 故障排查

### 问题 1: "Unknown dataset name"

**症状：**
```
Warning: Unknown dataset name: gsm8k
Available datasets: ['gsmk8', 'math']
```

**原因：** 数据集名称不匹配

**解决方案：**
1. 检查数据集配置中的`name`字段
2. 检查COT文件映射中的key
3. 确保两者完全一致（区分大小写）

### 问题 2: "No dataset_name or source field found"

**症状：**
```
Error: No dataset_name or source field found in batch!
```

**原因：** 没有使用正确的数据集类

**解决方案：**
确保配置了：
```bash
+data.custom_cls.path=examples.grpo_trainer.multi_dataset_with_cot \
+data.custom_cls.name=MultiDatasetWithCOT
```

### 问题 3: 某个数据集找不到COT

**症状：**
```
Warning: No COT examples found for math, key: ...
```

**原因：**
- COT文件路径错误
- 问题文本不匹配
- COT文件中没有该问题

**解决方案：**
1. 验证COT文件路径是否正确
2. 检查`match_by`配置（`question` vs `id`）
3. 打开COT文件，确认格式正确

## 高级用法

### 1. 不同数据集使用不同的COT格式

```python
# 自定义多数据集COT loader
class CustomMultiDatasetCOTLoader(MultiDatasetCOTLoader):
    def __init__(self, *args, **kwargs):
        # 不同数据集可以有不同的模板
        self.dataset_templates = {
            "gsm8k": "GSM8K Example:\nQ: {question}\nA: {rationale}\nFinal: {final_answer}\n\n",
            "math": "MATH Example:\nProblem: {question}\nSolution: {rationale}\nAnswer: {final_answer}\n\n",
            "numina": "Numina Example:\n{question}\n{rationale}\nFinal Answer: {final_answer}\n\n"
        }
        super().__init__(*args, **kwargs)
    
    def _load_single_cot_file(self, cot_file_path: str, dataset_name: str) -> Dict:
        # 使用特定数据集的模板
        self.cot_format_template = self.dataset_templates.get(dataset_name, self.cot_format_template)
        return super()._load_single_cot_file(cot_file_path)
```

### 2. 数据集权重

如果您想调整不同数据集的采样权重：

```python
# 在训练脚本中
from torch.utils.data import WeightedRandomSampler

# 为每个数据集设置权重
dataset_weights = {
    "gsm8k": 1.0,
    "math": 2.0,  # MATH数据集采样概率加倍
    "numina": 0.5
}

# 为每个样本分配权重
sample_weights = [dataset_weights[name] for name in dataset.dataset_names]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

# 在dataloader中使用
dataloader = DataLoader(dataset, sampler=sampler, ...)
```

### 3. 动态COT选择

根据问题难度或其他特征动态选择COT：

```python
def get_adaptive_cot_examples(batch, prompt_idx, num_repeats, tokenizer=None):
    """根据问题特征选择COT"""
    dataset_name = batch.non_tensor_batch["dataset_name"][prompt_idx]
    question = batch.non_tensor_batch["question"][prompt_idx]
    
    # 根据问题长度判断难度
    if len(question) > 200:
        # 长问题 -> 使用详细的COT
        return get_detailed_cots(dataset_name, question, num_repeats)
    else:
        # 短问题 -> 使用简洁的COT
        return get_concise_cots(dataset_name, question, num_repeats)
```

## 性能优化

### 1. 预加载COT数据

COT数据会在训练开始时全部加载到内存中，这样可以避免重复I/O：

- **内存占用**：每个数据集约 50-200 MB
- **加载时间**：每个数据集约 5-30 秒
- **总计**：3个数据集约 150-600 MB，加载时间 15-90 秒

### 2. 缓存策略

如果COT数据很大，可以实现LRU缓存：

```python
from functools import lru_cache

@lru_cache(maxsize=10000)
def get_cached_cot_examples(dataset_name, question_hash, num_examples):
    # 缓存最近使用的COT
    return loader.get_cot_examples(dataset_name, question, num_examples)
```

## 完整示例

参考文件：
- 配置文件：`examples/grpo_trainer/multi_dataset_config.yaml`
- 训练脚本：`examples/grpo_trainer/run_multi_dataset_grpo.sh`
- COT Loader：`examples/grpo_trainer/multi_dataset_cot_loader.py`
- 数据集类：`examples/grpo_trainer/multi_dataset_with_cot.py`

## 总结

✅ **多数据集支持**：
- 同时训练多个数据集
- 每个数据集使用自己的COT文件
- 自动识别和匹配

✅ **配置简单**：
- 只需配置数据集列表和COT文件映射
- 支持YAML配置文件

✅ **灵活扩展**：
- 可以自定义数据集处理逻辑
- 可以实现不同的COT选择策略

现在您可以轻松地使用多个数据集进行GRPO训练了！🎉

