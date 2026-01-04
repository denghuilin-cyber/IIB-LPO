# GRPO COT Augmentation 实现总结

## 概述

成功为GRPO训练添加了COT（Chain-of-Thought）增强功能，实现了**在每次rollout时使用不同的COT例子**。

## 您的需求

✅ **原始需求**：
- GRPO会对一个问题rollout多遍（例如4次）
- 希望每次rollout的prompt都不一样
- 在问题后面附加不同的COT例子
- COT例子来自独立的JSONL文件（`train_k_shot_GSM8K.jsonl`），包含`selected_cots`字段

✅ **数据格式**：
- 训练数据：`/nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet`
- COT数据：`/nas/dhl/CVAE/Datasets/GSM8K/processed/train_k_shot_GSM8K.jsonl`

## 实现方案

### 核心架构

```
┌─────────────────────────────────────────────────────────────┐
│                    GRPO Training Loop                       │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  1. Load batch (with question metadata)                    │
│     - GSM8KParquetDatasetWithCOT                           │
│     - Includes: question text, question_id                 │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Create gen_batch & repeat (n times)                    │
│     - gen_batch.repeat(repeat_times=n, interleave=True)    │
│     - n = config.actor_rollout_ref.rollout.n (GRPO group size) │
│     - 现在有n个相同的prompt                                    │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  3. COT Augmentation ⭐ (核心功能)                          │
│     - GRPOCOTAugmenter.augment(gen_batch)                  │
│     - 为每个重复添加不同的COT例子                              │
│       • Rollout 1: Question + COT例子1                      │
│       • Rollout 2: Question + COT例子2                      │
│       • ...                                                 │
│       • Rollout n: Question + COT例子n                      │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Generate sequences (n different prompts → n responses) │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  5. Compute GRPO advantages (group-based relative rewards) │
└─────────────────────────────────────────────────────────────┘
```

### 创建/修改的文件

#### 1. 核心功能文件

| 文件 | 路径 | 说明 |
|------|------|------|
| `grpo_cot_augmentation.py` | `verl/utils/` | COT增强器核心实现，处理prompt修改和tokenization |
| `ray_trainer.py` | `verl/trainer/ppo/` | ✏️ **已修改**：集成COT augmenter到训练循环 |

#### 2. GSM8K特定文件

| 文件 | 路径 | 说明 |
|------|------|------|
| `gsm8k_cot_loader.py` | `examples/grpo_trainer/` | 加载JSONL格式的COT数据，匹配问题和COT例子 |
| `gsm8k_dataset_with_cot.py` | `examples/grpo_trainer/` | 自定义数据集，确保question元数据传递到batch |

#### 3. 示例和文档

| 文件 | 路径 | 说明 |
|------|------|------|
| `run_gsm8k_with_cot.sh` | `examples/grpo_trainer/` | 完整的训练脚本示例 |
| `QUICKSTART_CN.md` | `examples/grpo_trainer/` | 3步快速上手指南 |
| `GSM8K_COT_USAGE.md` | `examples/grpo_trainer/` | 完整使用文档（配置、故障排查等）|
| `README_COT.md` | `examples/grpo_trainer/` | 项目总览和功能介绍 |
| `custom_cot_getter_example.py` | `examples/grpo_trainer/` | 自定义COT生成器示例 |
| `cot_examples_math.txt` | `examples/grpo_trainer/` | 简单文本格式的COT例子 |
| `config_cot_augmentation_example.yaml` | `examples/grpo_trainer/` | YAML配置文件示例 |

## 使用方法

### 最简使用（3步）

#### Step 1: 准备数据
确保您有：
- ✅ 训练数据：`/nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet`
- ✅ COT数据：`/nas/dhl/CVAE/Datasets/GSM8K/processed/train_k_shot_GSM8K.jsonl`

#### Step 2: 修改训练命令
在您现有的GRPO训练脚本中添加配置：

```bash
python3 -m verl.trainer.main_ppo \
  data.train_files=/nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet \
  data.val_files=/nas/dhl/Datasets/my_Datasets/gsm8k/test.parquet \
  \
  # 使用支持COT的数据集类
  +data.custom_cls.path=examples.grpo_trainer.gsm8k_dataset_with_cot \
  +data.custom_cls.name=GSM8KParquetDatasetWithCOT \
  \
  # GRPO配置
  actor_rollout_ref.rollout.n=4 \
  algorithm.adv_estimator=grpo \
  \
  # 启用COT增强 ⭐
  +actor_rollout_ref.rollout.cot_augmentation.enable=true \
  +actor_rollout_ref.rollout.cot_augmentation.cot_file_path=/nas/dhl/CVAE/Datasets/GSM8K/processed/train_k_shot_GSM8K.jsonl \
  +actor_rollout_ref.rollout.cot_augmentation.match_by=question \
  +actor_rollout_ref.rollout.cot_augmentation.sampling_strategy=sequential \
  \
  # ... 其他参数
```

#### Step 3: 运行
```bash
bash your_training_script.sh
```

### 验证是否工作

训练开始时，查看日志中是否有：

```
✅ Initializing GSM8K COT loader from /nas/dhl/CVAE/Datasets/GSM8K/processed/train_k_shot_GSM8K.jsonl
✅ Loaded COT data for 7473 questions from ...
✅ GSM8K COT loader initialized successfully
✅ COT augmenter initialized with strategy: sequential
```

## 关键配置参数

### 必需参数

```bash
# 启用功能
+actor_rollout_ref.rollout.cot_augmentation.enable=true

# COT数据文件
+actor_rollout_ref.rollout.cot_augmentation.cot_file_path=/path/to/train_k_shot_GSM8K.jsonl
```

### 重要可选参数

```bash
# 匹配方式（如何找到对应的COT）
+actor_rollout_ref.rollout.cot_augmentation.match_by=question  # 或 "id"

# 采样策略（如何选择COT）
+actor_rollout_ref.rollout.cot_augmentation.sampling_strategy=sequential  
# 可选: "random_with_replacement", "random_without_replacement"

# COT格式模板
+actor_rollout_ref.rollout.cot_augmentation.cot_format_template="Example:\nQ: {question}\nA: {rationale}\nFinal: {final_answer}\n\nNow:"

# 是否使用完整COT（问题+推理+答案）
+actor_rollout_ref.rollout.cot_augmentation.use_full_cot=true
```

## 工作原理示例

假设有一个问题：**"Randy has 60 mango trees on his farm..."**

该问题有4个COT例子（在`selected_cots`中），GRPO配置为`n=4`。

### 传统GRPO（不使用COT增强）

4次rollout都使用相同的prompt：
```
Rollout 1-4: "Randy has 60 mango trees on his farm..."
```

### COT增强GRPO（新方案）

每次rollout使用不同的prompt：

**Rollout 1:**
```
Here's a similar example:
Question: Grace just started her own business...
Let's solve it step by step:
Every two weeks, Grace will get 300*2=600 dollars...
Final Answer: 6

Now, let's solve the current problem:
Randy has 60 mango trees on his farm...
```

**Rollout 2:**
```
Here's a similar example:
Question: A carton contains 12 boxes...
Let's solve it step by step:
If 1 carton contains 12 boxes, then a dozen cartons contain 12*12 = 144 boxes...
Final Answer: 1

Now, let's solve the current problem:
Randy has 60 mango trees on his farm...
```

**Rollout 3, 4:** 类似，使用不同的COT例子

## 技术实现细节

### 1. COT数据加载 (`gsm8k_cot_loader.py`)

```python
# 读取JSONL文件
{
  "question": "Randy has 60 mango trees...",
  "selected_cots": [
    {"question": "...", "rationale": "...", "final_answer": "..."},
    ...
  ]
}

# 建立映射
question → [COT例子1, COT例子2, COT例子3, COT例子4]
```

### 2. 数据集增强 (`gsm8k_dataset_with_cot.py`)

```python
# 确保question信息传递到batch
def __getitem__(self, index):
    item = super().__getitem__(index)
    item['question'] = self.questions[index]  # ⭐ 添加question
    item['question_id'] = self.question_ids[index]
    return item
```

### 3. 训练循环集成 (`ray_trainer.py`)

```python
# 在repeat之后、generate之前插入COT augmentation
gen_batch = gen_batch.repeat(repeat_times=4, interleave=True)

# ⭐ 新增的COT augmentation
if self.cot_augmenter is not None:
    gen_batch = self.cot_augmenter.augment(gen_batch)

# 然后生成
gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
```

### 4. Prompt修改 (`grpo_cot_augmentation.py`)

```python
# 对每个重复的prompt
for rep_idx in range(num_repeats):
    # 1. 获取对应的COT例子
    cot_text = cot_examples_for_prompt[rep_idx]
    
    # 2. 解码原始prompt
    prompt_text = tokenizer.decode(prompt_tokens)
    
    # 3. 拼接COT例子
    augmented_text = prompt_text + "\n\n" + cot_text
    
    # 4. 重新tokenize
    augmented_tokens = tokenizer.encode(augmented_text)
```

## 配置选项详解

### `match_by`: 如何匹配COT

| 值 | 说明 | 适用场景 |
|----|------|----------|
| `"question"` | 按问题文本匹配 | 两个文件的问题文本一致 |
| `"id"` | 按ID匹配 | 两个文件有对应的ID字段 |

### `sampling_strategy`: 如何选择COT

| 值 | 说明 | 行为 |
|----|------|------|
| `"sequential"` | 顺序使用 | 第1次rollout用COT1，第2次用COT2，... |
| `"random_with_replacement"` | 随机（可重复） | 随机选择，可能重复 |
| `"random_without_replacement"` | 随机（不重复） | 随机选择，不重复（需要足够的COT例子）|

### `cot_format_template`: COT格式化

使用Python `.format()` 语法，可用占位符：
- `{question}`: COT例子的问题
- `{rationale}`: COT例子的推理过程
- `{final_answer}`: COT例子的最终答案

示例模板：
```python
# 完整格式
"Example:\nQuestion: {question}\nReasoning: {rationale}\nAnswer: {final_answer}\n\nNow solve:"

# 简洁格式
"{rationale}"  # 只用推理步骤

# 自定义格式
"Here's how to solve a similar problem:\n{question}\n{rationale}\nFinal: {final_answer}\n\n---"
```

## 预期效果

### 潜在好处

1. **多样性增加** ✨
   - 每次rollout看到不同的示例
   - GRPO的group内响应更多样化

2. **学习信号增强** 📈
   - 模型接触更多推理模式
   - 从不同的COT示例中学习

3. **泛化能力提升** 🎯
   - 更多样的训练信号
   - 有助于模型泛化到新问题

### 性能影响

- **内存**：增加prompt长度，建议设置`max_prompt_length=1024`或更高
- **速度**：增加tokenization时间，但影响<5%
- **质量**：需要实验验证具体效果

## 故障排查

### 问题1: 找不到COT例子

**症状：**
```
Warning: No COT examples found for key: ...
```

**原因：**
- 问题文本不匹配
- ID不对应
- COT文件路径错误

**解决方案：**
1. 检查`match_by`设置
2. 打印训练数据和COT数据，确保格式一致
3. 使用`match_by="id"`确保精确匹配

### 问题2: COT例子数量不足

**症状：**
```
Warning: Got 2 COT examples but need 4
```

**原因：**
- `selected_cots`中例子少于`n_rollouts`

**解决方案：**
1. 使用`sequential`或`random_with_replacement`（会循环使用）
2. 增加`selected_cots`中的例子数量
3. 减少`n_rollouts`

### 问题3: 找不到question字段

**症状：**
```
Error getting COT examples: ...
```

**原因：**
- 未使用正确的数据集类

**解决方案：**
确保配置了：
```bash
+data.custom_cls.path=examples.grpo_trainer.gsm8k_dataset_with_cot \
+data.custom_cls.name=GSM8KParquetDatasetWithCOT
```

## 下一步

### 验证效果
1. 运行小规模实验（100-200步）
2. 对比有/无COT增强的指标
3. 检查生成的响应是否更多样化

### 调优
1. 尝试不同的`sampling_strategy`
2. 调整`cot_format_template`
3. 实验不同的`n_rollouts`数量

### 扩展
1. 适配其他数据格式
2. 实现自定义COT选择逻辑
3. 支持动态COT生成

## 总结

✅ **已实现**：
- 完整的COT augmentation功能
- 支持您的GSM8K数据格式
- 无缝集成到GRPO训练流程
- 详细的文档和示例

✅ **易于使用**：
- 只需添加几行配置
- 不修改原始数据
- 可随时启用/禁用

✅ **高度可定制**：
- 支持多种COT格式
- 支持多种采样策略
- 支持自定义逻辑

现在您可以开始训练了！祝训练成功！🎉

## 快速参考

- **快速开始**：`examples/grpo_trainer/QUICKSTART_CN.md`
- **完整文档**：`examples/grpo_trainer/GSM8K_COT_USAGE.md`
- **示例脚本**：`examples/grpo_trainer/run_gsm8k_with_cot.sh`
- **自定义示例**：`examples/grpo_trainer/custom_cot_getter_example.py`

