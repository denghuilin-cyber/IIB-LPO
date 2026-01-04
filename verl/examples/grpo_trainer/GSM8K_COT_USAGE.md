# GSM8K GRPO Training with COT Augmentation

本文档介绍如何在GRPO训练中为每次rollout使用不同的COT例子。

## 问题背景

在GRPO（Group Relative Policy Optimization）算法中，每个问题会被rollout多次（例如4次），以生成多个响应并计算相对优势。默认情况下，每次rollout使用完全相同的prompt。

您想要的改进是：**每次rollout时，在原问题后面附加不同的COT例子**，这样可以给模型提供不同的演示来引导推理。

## 数据格式

### 训练数据
```
/nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet
```
Parquet格式，包含问题和答案。

### COT例子数据
```
/nas/dhl/CVAE/Datasets/GSM8K/processed/train_k_shot_GSM8K.jsonl
```

JSONL格式，每行一个JSON对象：
```json
{
  "id": 12,
  "question": "Randy has 60 mango trees...",
  "rationale": "Half of the number...",
  "final_answer": "85",
  "selected_cots": [
    {
      "id": 4578,
      "question": "Grace just started...",
      "rationale": "Every two weeks...",
      "final_answer": "6"
    },
    {
      "id": 5018,
      "question": "A carton contains...",
      "rationale": "If 1 carton...",
      "final_answer": "1"
    }
    // ... 更多COT例子
  ]
}
```

## 解决方案架构

### 1. 核心组件

#### a) `gsm8k_cot_loader.py`
- **功能**：加载和管理COT例子
- **核心类**：`GSM8KCOTLoader` - 将问题与COT例子匹配
- **核心函数**：`get_gsm8k_cot_examples` - COT获取函数

#### b) `gsm8k_dataset_with_cot.py`
- **功能**：自定义数据集，确保问题文本被传递到batch中
- **核心类**：`GSM8KParquetDatasetWithCOT` - 扩展数据集，包含问题元数据

#### c) `grpo_cot_augmentation.py` (in verl/utils/)
- **功能**：COT增强器，在rollout前修改prompt
- **核心类**：`GRPOCOTAugmenter` - 为每次rollout添加不同的COT

#### d) 修改的 `ray_trainer.py`
- **功能**：集成COT增强器到训练循环
- **关键点**：在 `repeat()` 之后、`generate_sequences()` 之前调用augmenter

### 2. 工作流程

```
训练循环中:
1. 从dataloader获取batch (包含question元数据)
2. 创建gen_batch
3. gen_batch.repeat(n=4)  # GRPO重复4次
4. cot_augmenter.augment(gen_batch)  # 为每次重复添加不同的COT
   - 对每个原始问题:
     a. 获取该问题对应的COT例子列表
     b. 为4次rollout分别选择不同的COT例子
     c. 将COT例子附加到prompt后面
     d. 重新tokenize
5. 生成响应
6. 计算GRPO advantage (基于group内的相对奖励)
```

## 使用方法

### 步骤 1: 准备文件

将以下文件复制到您的项目中：

```bash
# 创建工作目录
mkdir -p examples/grpo_trainer

# 复制必要的文件
cp verl/examples/grpo_trainer/gsm8k_cot_loader.py ./
cp verl/examples/grpo_trainer/gsm8k_dataset_with_cot.py ./
cp verl/examples/grpo_trainer/run_gsm8k_with_cot.sh ./
```

### 步骤 2: 配置训练脚本

编辑 `run_gsm8k_with_cot.sh`，设置正确的路径：

```bash
# 数据路径
TRAIN_DATA="/nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet"
COT_DATA="/nas/dhl/CVAE/Datasets/GSM8K/processed/train_k_shot_GSM8K.jsonl"

# GRPO配置
N_ROLLOUTS=4  # 每个问题rollout 4次

# COT配置
COT_FORMAT="Here's a similar example:\n\nQuestion: {question}\n\nStep-by-step solution:\n{rationale}\n\nFinal Answer: {final_answer}\n\n---\n\nNow solve:"
COT_MATCH_BY="question"  # 按问题文本匹配（或使用"id"）
COT_SAMPLING="sequential"  # 按顺序使用COT（或"random_with_replacement"）
```

### 步骤 3: 运行训练

```bash
chmod +x run_gsm8k_with_cot.sh
./run_gsm8k_with_cot.sh
```

## 配置选项详解

### COT Augmentation 配置参数

在配置中通过 `actor_rollout_ref.rollout.cot_augmentation` 设置：

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `enable` | 是否启用COT增强 | false | true |
| `cot_file_path` | COT数据文件路径 | None | "/path/to/train_k_shot_GSM8K.jsonl" |
| `cot_format_template` | COT格式化模板 | 见下方 | "Question: {question}..." |
| `match_by` | 匹配方式 | "question" | "question" 或 "id" |
| `sampling_strategy` | 采样策略 | "sequential" | 见下方 |
| `add_separator` | 是否添加分隔符 | true | true |
| `separator` | 分隔符字符串 | "\n\n" | "\n\n---\n\n" |
| `use_full_cot` | 使用完整COT还是仅rationale | true | true |
| `seed` | 随机种子 | None | 42 |

### COT格式化模板

模板使用Python的 `.format()` 语法，可用的占位符：
- `{question}`: COT例子的问题
- `{rationale}`: COT例子的推理过程
- `{final_answer}`: COT例子的最终答案

示例模板：

```python
# 完整格式
cot_format_template = """
Here's a similar example to help you:

Question: {question}

Step-by-step solution:
{rationale}

Final Answer: {final_answer}

---

Now solve the following problem step by step:
"""

# 简洁格式
cot_format_template = "Example: {question} → {rationale} → Answer: {final_answer}\n\nNow:"

# 仅推理步骤
cot_format_template = "{rationale}"  # 同时设置 use_full_cot=False
```

### 采样策略

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| `sequential` | 按顺序使用COT例子 | 确保每次rollout使用不同例子 |
| `random_with_replacement` | 随机采样（可重复） | 增加随机性 |
| `random_without_replacement` | 随机采样（不重复） | 需要 `len(selected_cots) >= n_rollouts` |

## 工作原理示例

假设您有一个问题，GRPO配置为 `n=4`（4次rollout），该问题有4个COT例子：

### 原始prompt（每次rollout都相同）：
```
Question: Randy has 60 mango trees on his farm...
```

### 使用COT augmentation后（每次rollout不同）：

**Rollout 1:**
```
Here's a similar example:
Question: Grace just started her own business...
Step-by-step solution: Every two weeks, Grace will get 300*2=600...
Final Answer: 6

Now solve the following problem:
Question: Randy has 60 mango trees on his farm...
```

**Rollout 2:**
```
Here's a similar example:
Question: A carton contains 12 boxes...
Step-by-step solution: If 1 carton contains 12 boxes...
Final Answer: 1

Now solve the following problem:
Question: Randy has 60 mango trees on his farm...
```

**Rollout 3:**
```
Here's a similar example:
Question: Ibrahim wants to buy an MP3 player...
Step-by-step solution: The total price of the purchases...
Final Answer: 64

Now solve the following problem:
Question: Randy has 60 mango trees on his farm...
```

**Rollout 4:**
```
Here's a similar example:
Question: Whitney bought 9 books about whales...
Step-by-step solution: The total number of books is 9 books + 7 books...
Final Answer: 179

Now solve the following problem:
Question: Randy has 60 mango trees on his farm...
```

## 匹配策略

### 按问题文本匹配 (match_by="question")
- **优点**：简单直接
- **缺点**：问题文本必须完全匹配
- **适用**：问题文本格式统一的情况

### 按ID匹配 (match_by="id")
- **优点**：精确匹配
- **缺点**：需要两个文件中的ID对应
- **适用**：数据有明确ID映射的情况

## 故障排查

### 问题1：找不到COT例子

**症状**：
```
Warning: No COT examples found for key: ...
```

**原因**：
- 问题文本不匹配
- ID不对应
- COT文件路径错误

**解决**：
1. 检查 `match_by` 设置是否正确
2. 打印训练数据和COT数据的问题/ID，确保一致
3. 使用 `match_by="id"` 确保精确匹配

### 问题2：COT例子数量不足

**症状**：
```
Warning: Got 2 COT examples but need 4
```

**原因**：
- `selected_cots` 中的例子少于 `n_rollouts`
- 使用了 `random_without_replacement` 但例子不够

**解决**：
1. 使用 `sequential` 或 `random_with_replacement` 策略（会循环使用）
2. 增加 `selected_cots` 中的例子数量
3. 减少 `n_rollouts` 数量

### 问题3：找不到question字段

**症状**：
```
Error getting COT examples: ...
```

**原因**：
- 数据集没有正确传递question元数据

**解决**：
1. 确保使用了 `GSM8KParquetDatasetWithCOT` 数据集类
2. 在配置中设置：
   ```bash
   +data.custom_cls.path=examples.grpo_trainer.gsm8k_dataset_with_cot \
   +data.custom_cls.name=GSM8KParquetDatasetWithCOT \
   ```

## 高级用法

### 自定义COT格式

如果您想要更复杂的COT格式化逻辑，可以修改 `gsm8k_cot_loader.py` 中的格式化代码：

```python
# 在 GSM8KCOTLoader._load_cot_data() 中
for cot in selected_cots:
    # 自定义格式化
    formatted_cot = f"""
    Example Problem:
    {cot['question']}
    
    Solution Strategy:
    {cot['rationale']}
    
    Result: {cot['final_answer']}
    
    Apply the same strategy to solve:
    """
    formatted_cots.append(formatted_cot)
```

### 动态选择COT

如果您想根据问题难度、类型等动态选择COT：

```python
# 修改 get_gsm8k_cot_examples 函数
def get_gsm8k_cot_examples(batch, prompt_idx: int, num_repeats: int, tokenizer=None):
    # ... 获取question
    
    # 从COT loader获取所有可用的COT
    all_cots = _global_cot_loader.get_cot_examples(question=question)
    
    # 自定义选择逻辑
    # 例如：根据问题长度选择不同难度的COT
    if len(question) > 100:
        # 长问题 -> 选择详细的COT
        selected = [cot for cot in all_cots if len(cot) > 200][:num_repeats]
    else:
        # 短问题 -> 选择简洁的COT
        selected = [cot for cot in all_cots if len(cot) <= 200][:num_repeats]
    
    return selected
```

## 性能考虑

### 内存

- COT augmentation会增加prompt长度
- 建议调整 `max_prompt_length` 以容纳额外的COT文本
- 建议值：`max_prompt_length=1024` 或更高

### 速度

- Tokenization会稍微增加预处理时间
- 对整体训练时间影响较小（<5%）
- 生成阶段（inference）仍然是主要瓶颈

## 预期效果

使用COT augmentation的潜在好处：

1. **多样性增加**：每次rollout看到不同的示例，增加响应多样性
2. **学习信号更强**：模型可以从不同的演示中学习推理模式
3. **泛化能力提升**：接触到更多示例有助于泛化

## 总结

通过这个方案，您可以：
- ✅ 在GRPO的每次rollout中使用不同的COT例子
- ✅ COT例子从您预先准备的JSONL文件中加载
- ✅ 灵活配置COT格式、匹配方式和采样策略
- ✅ 与现有GRPO训练流程无缝集成

祝训练顺利！🚀

