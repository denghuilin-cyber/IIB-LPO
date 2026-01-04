# GRPO COT增强 - 快速启动指南

## 一分钟了解

在GRPO训练中，每个问题会rollout多次（如4次）。通过这个功能，您可以让每次rollout使用**不同的COT例子**，而不是完全相同的prompt。

## 快速使用（3步）

### 1. 修改训练脚本，添加配置

在您现有的GRPO训练脚本中添加以下参数：

```bash
python3 -m verl.trainer.main_ppo \
  # ... 您现有的参数 ...
  \
  # 使用支持COT的数据集
  +data.custom_cls.path=examples.grpo_trainer.gsm8k_dataset_with_cot \
  +data.custom_cls.name=GSM8KParquetDatasetWithCOT \
  \
  # 启用COT增强
  +actor_rollout_ref.rollout.cot_augmentation.enable=true \
  +actor_rollout_ref.rollout.cot_augmentation.cot_file_path=/nas/dhl/CVAE/Datasets/GSM8K/processed/train_k_shot_GSM8K.jsonl \
  +actor_rollout_ref.rollout.cot_augmentation.match_by=question \
  +actor_rollout_ref.rollout.cot_augmentation.sampling_strategy=sequential \
  +actor_rollout_ref.rollout.cot_augmentation.use_full_cot=true
```

### 2. 确保文件在正确位置

```bash
# COT loader (已创建)
verl/examples/grpo_trainer/gsm8k_cot_loader.py

# 数据集类 (已创建)
verl/examples/grpo_trainer/gsm8k_dataset_with_cot.py

# COT augmenter (已创建)
verl/utils/grpo_cot_augmentation.py

# 修改过的trainer (已修改)
verl/trainer/ppo/ray_trainer.py
```

### 3. 运行

```bash
# 使用示例脚本
bash examples/grpo_trainer/run_gsm8k_with_cot.sh

# 或直接运行您修改后的训练命令
python3 -m verl.trainer.main_ppo ...
```

## 核心配置参数

### 必需参数

```bash
# 启用COT增强
+actor_rollout_ref.rollout.cot_augmentation.enable=true

# COT数据文件（包含selected_cots字段的JSONL）
+actor_rollout_ref.rollout.cot_augmentation.cot_file_path=/path/to/train_k_shot_GSM8K.jsonl
```

### 重要可选参数

```bash
# 匹配方式: "question" (按文本) 或 "id" (按ID)
+actor_rollout_ref.rollout.cot_augmentation.match_by=question

# 采样策略: "sequential" (顺序), "random_with_replacement" (随机可重复)
+actor_rollout_ref.rollout.cot_augmentation.sampling_strategy=sequential

# 是否使用完整COT (question+rationale+answer) 或仅rationale
+actor_rollout_ref.rollout.cot_augmentation.use_full_cot=true

# COT格式模板
+actor_rollout_ref.rollout.cot_augmentation.cot_format_template="Example:\nQ: {question}\nA: {rationale}\nFinal: {final_answer}\n\nNow solve:"
```

## 数据格式要求

### 训练数据 (train.parquet)
```
普通的GSM8K parquet文件，包含question和answer字段
```

### COT数据 (train_k_shot_GSM8K.jsonl)
```json
{
  "id": 12,
  "question": "Randy has 60 mango trees...",
  "selected_cots": [
    {
      "question": "Grace just started...",
      "rationale": "Every two weeks...",
      "final_answer": "6"
    },
    ...
  ]
}
```

## 工作原理

```
原始流程:
question → repeat(4x) → [同样的prompt] × 4 → generate

新流程:
question → repeat(4x) → augment → [不同的prompt] × 4 → generate
                                  ↓
                        prompt1: question + COT例子1
                        prompt2: question + COT例子2  
                        prompt3: question + COT例子3
                        prompt4: question + COT例子4
```

## 验证是否生效

训练开始时，您应该看到类似的日志：

```
Initializing GSM8K COT loader from /path/to/train_k_shot_GSM8K.jsonl
Loaded COT data for 7473 questions from /path/to/train_k_shot_GSM8K.jsonl
GSM8K COT loader initialized successfully
COT augmenter initialized with strategy: sequential
```

## 常见问题

### Q: 我的COT文件格式稍有不同怎么办？
A: 修改 `gsm8k_cot_loader.py` 中的 `_load_cot_data()` 方法来适配您的格式。

### Q: 能否每次rollout使用完全随机的COT？
A: 可以，设置 `sampling_strategy=random_with_replacement`

### Q: 如果某个问题找不到COT会怎样？
A: 会打印警告并使用空字符串（即不添加COT），训练继续进行。

### Q: 会影响训练速度吗？
A: 影响很小（<5%），主要是增加了tokenization时间。

### Q: 可以关闭COT增强吗？
A: 可以，设置 `enable=false` 或直接不添加相关配置。

## 完整示例

查看详细文档：`examples/grpo_trainer/GSM8K_COT_USAGE.md`

查看示例脚本：`examples/grpo_trainer/run_gsm8k_with_cot.sh`

## 需要帮助？

1. 查看完整文档：`GSM8K_COT_USAGE.md`
2. 检查示例文件：`custom_cot_getter_example.py`
3. 查看日志输出中的COT相关信息

祝训练成功！🎉

