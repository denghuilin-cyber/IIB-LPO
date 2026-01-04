# GRPO with COT Augmentation

为GRPO训练添加COT（Chain-of-Thought）增强功能，让每次rollout使用不同的示例。

## 📁 文件列表

| 文件 | 说明 | 位置 |
|------|------|------|
| `grpo_cot_augmentation.py` | COT增强器核心实现 | `verl/utils/` |
| `ray_trainer.py` (修改) | 集成COT到训练循环 | `verl/trainer/ppo/` |
| `gsm8k_cot_loader.py` | GSM8K格式的COT加载器 | `examples/grpo_trainer/` |
| `gsm8k_dataset_with_cot.py` | 支持COT的数据集类 | `examples/grpo_trainer/` |
| `run_gsm8k_with_cot.sh` | 示例训练脚本 | `examples/grpo_trainer/` |
| `QUICKSTART_CN.md` | 快速启动指南（中文）| `examples/grpo_trainer/` |
| `GSM8K_COT_USAGE.md` | 完整使用文档（中文）| `examples/grpo_trainer/` |

## 🚀 快速开始

### 最简配置

在您的训练脚本中添加：

```bash
python3 -m verl.trainer.main_ppo \
  +data.custom_cls.path=examples.grpo_trainer.gsm8k_dataset_with_cot \
  +data.custom_cls.name=GSM8KParquetDatasetWithCOT \
  +actor_rollout_ref.rollout.cot_augmentation.enable=true \
  +actor_rollout_ref.rollout.cot_augmentation.cot_file_path=/path/to/train_k_shot_GSM8K.jsonl \
  # ... 其他参数
```

详见：[QUICKSTART_CN.md](./QUICKSTART_CN.md)

## 📖 文档

- **快速启动**：[QUICKSTART_CN.md](./QUICKSTART_CN.md) - 3步上手
- **完整文档**：[GSM8K_COT_USAGE.md](./GSM8K_COT_USAGE.md) - 详细说明、配置、故障排查
- **示例代码**：[custom_cot_getter_example.py](./custom_cot_getter_example.py) - 自定义COT生成器

## 🎯 核心功能

### 1. 为每次Rollout使用不同的COT

**传统GRPO**：
```
Question → [同样的prompt] × 4 → 4个响应
```

**COT增强GRPO**：
```
Question → [不同的prompt] × 4 → 4个响应
            ↓
  Prompt 1: Question + COT例子1
  Prompt 2: Question + COT例子2  
  Prompt 3: Question + COT例子3
  Prompt 4: Question + COT例子4
```

### 2. 灵活的配置选项

- **多种数据格式**：支持TXT、JSON、JSONL
- **多种匹配方式**：按问题文本或ID匹配
- **多种采样策略**：顺序、随机、自定义
- **可定制模板**：自定义COT格式化方式

### 3. 与现有流程无缝集成

- 不修改原始训练数据
- 不影响现有GRPO逻辑
- 可随时启用/禁用
- 对性能影响<5%

## 💡 使用场景

### ✅ 适合使用的情况

- 您有预先准备的COT示例数据
- 每个问题有多个不同的COT例子
- 希望增加GRPO rollout的多样性
- 想让模型接触更多推理示例

### ⚠️ 不适合的情况

- COT例子数量很少（<rollout次数）
- 没有准备好的COT数据
- 单纯想增加prompt多样性（可用prompt augmentation）

## 🔧 配置示例

### 基础配置
```yaml
actor_rollout_ref:
  rollout:
    n: 4
    cot_augmentation:
      enable: true
      cot_file_path: /path/to/cots.jsonl
      match_by: question
      sampling_strategy: sequential
```

### 高级配置
```yaml
actor_rollout_ref:
  rollout:
    cot_augmentation:
      enable: true
      cot_file_path: /path/to/cots.jsonl
      cot_format_template: "Example: {question}\nSolution: {rationale}\nAnswer: {final_answer}\n\nNow:"
      match_by: question
      sampling_strategy: random_with_replacement
      add_separator: true
      separator: "\n\n---\n\n"
      use_full_cot: true
      seed: 42
```

## 📊 预期效果

使用COT增强可能带来的好处：

1. **响应多样性↑**：不同的COT示例引导不同的推理路径
2. **学习信号↑**：模型从更多样的示例中学习
3. **泛化能力↑**：接触更多推理模式有助于泛化

## 🛠️ 开发与扩展

### 自定义COT生成器

```python
def my_custom_cot_getter(batch, prompt_idx, num_repeats):
    """自定义COT选择逻辑"""
    # 获取问题
    question = batch.non_tensor_batch["question"][prompt_idx]
    
    # 根据问题特征选择COT
    if "multiply" in question.lower():
        return get_multiplication_cots(num_repeats)
    elif "divide" in question.lower():
        return get_division_cots(num_repeats)
    else:
        return get_general_cots(num_repeats)
```

配置：
```bash
+actor_rollout_ref.rollout.cot_augmentation.examples_getter.path=my_module \
+actor_rollout_ref.rollout.cot_augmentation.examples_getter.name=my_custom_cot_getter
```

### 适配其他数据格式

修改 `gsm8k_cot_loader.py` 中的 `_load_cot_data()` 方法：

```python
def _load_cot_data(self):
    """适配您的数据格式"""
    with open(self.cot_file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            # 自定义提取逻辑
            key = data["your_question_field"]
            cots = data["your_cots_field"]
            # ...
```

## 📝 更新日志

### v1.0 (2025-01)
- ✨ 初始版本
- ✅ 支持GSM8K格式的JSONL COT文件
- ✅ 支持多种采样策略
- ✅ 支持自定义模板
- ✅ 集成到GRPO训练循环

## 🤝 贡献

欢迎提Issue和PR！

## 📄 许可

Apache License 2.0

## 🙏 致谢

基于veRL GRPO训练框架开发。

