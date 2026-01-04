# ⚠️ 重要说明：GRPO Group Size 是动态配置的

## 用户提出的关键问题

> "但是不一定是4啊，这个4不是固定死的，而是 grpo的group大小是多少，我们就会有多少个rollout"

**您说得完全正确！** 👍

## 代码实现（正确✅）

代码实现是**完全动态的**，没有硬编码任何数字：

```python
# 在 ray_trainer.py 中
self.cot_augmenter = GRPOCOTAugmenter(
    cot_examples=cot_examples,
    cot_examples_getter=cot_examples_getter,
    tokenizer=self.tokenizer,
    num_repeats=self.config.actor_rollout_ref.rollout.n,  # ✅ 动态读取配置
    ...
)

# 在训练循环中
gen_batch = gen_batch.repeat(
    repeat_times=self.config.actor_rollout_ref.rollout.n,  # ✅ 动态读取配置
    interleave=True
)
```

**关键点**：
- `num_repeats` 不是硬编码的
- 它来自配置：`config.actor_rollout_ref.rollout.n`
- 您可以设置为任意值：4, 8, 16, 32...

## 文档问题（已修正❌→✅）

文档中为了举例方便，多处使用了"4"这个具体数字，这**造成了误导**。

### 应该这样表述：

❌ **错误示例**（容易误导）：
```
GRPO会对一个问题rollout 4次
每次rollout使用不同的COT例子
```

✅ **正确表述**：
```
GRPO会对一个问题rollout n次（n由配置决定）
每次rollout使用不同的COT例子
n = config.actor_rollout_ref.rollout.n（GRPO的group size）
```

## 实际使用示例

### 配置 Group Size = 4
```bash
python3 -m verl.trainer.main_ppo \
  actor_rollout_ref.rollout.n=4 \  # Group size = 4
  algorithm.adv_estimator=grpo \
  +actor_rollout_ref.rollout.cot_augmentation.enable=true
```

**效果**：
- 每个问题 rollout 4 次
- 使用 4 个不同的 COT 例子

### 配置 Group Size = 8
```bash
python3 -m verl.trainer.main_ppo \
  actor_rollout_ref.rollout.n=8 \  # Group size = 8
  algorithm.adv_estimator=grpo \
  +actor_rollout_ref.rollout.cot_augmentation.enable=true
```

**效果**：
- 每个问题 rollout 8 次
- 使用 8 个不同的 COT 例子

### 配置 Group Size = 16
```bash
python3 -m verl.trainer.main_ppo \
  actor_rollout_ref.rollout.n=16 \  # Group size = 16
  algorithm.adv_estimator=grpo \
  +actor_rollout_ref.rollout.cot_augmentation.enable=true
```

**效果**：
- 每个问题 rollout 16 次
- 使用 16 个不同的 COT 例子

## COT 例子数量处理

### 如果 COT 例子 < Group Size

例如：`selected_cots` 只有 4 个例子，但 group size = 8

**处理方式**：循环使用（cycle through）

```python
# 伪代码
if len(cot_examples) < num_repeats:
    # 循环使用
    for i in range(num_repeats):
        cot_example = cot_examples[i % len(cot_examples)]
```

**实际效果**（group size = 8，但只有 4 个 COT）：
- Rollout 1: COT 1
- Rollout 2: COT 2
- Rollout 3: COT 3
- Rollout 4: COT 4
- Rollout 5: COT 1（循环）
- Rollout 6: COT 2（循环）
- Rollout 7: COT 3（循环）
- Rollout 8: COT 4（循环）

### 如果 COT 例子 > Group Size

例如：`selected_cots` 有 10 个例子，但 group size = 4

**处理方式**：只使用前 n 个

```python
cot_examples = all_cots[:num_repeats]  # 只取前4个
```

**实际效果**：
- Rollout 1: COT 1
- Rollout 2: COT 2
- Rollout 3: COT 3
- Rollout 4: COT 4
- （COT 5-10 不使用）

## 配置建议

### 常用的 Group Size 设置

| Group Size | 适用场景 | 计算成本 | COT 多样性需求 |
|-----------|---------|---------|---------------|
| n=4 | 快速实验 | 低 | 4 个不同 COT |
| n=8 | 标准训练 | 中等 | 8 个不同 COT |
| n=16 | 高质量训练 | 高 | 16 个不同 COT |
| n=32 | 极致质量 | 很高 | 32 个不同 COT |

### 推荐配置

**GSM8K**（相对简单）：
```bash
actor_rollout_ref.rollout.n=4  # 或 8
```

**MATH**（较难）：
```bash
actor_rollout_ref.rollout.n=8  # 或 16
```

**混合数据集**：
```bash
actor_rollout_ref.rollout.n=8  # 平衡点
```

## 关键配置参数

```yaml
actor_rollout_ref:
  rollout:
    n: 8  # ⭐ GRPO group size，决定 rollout 次数
    temperature: 0.7
    top_p: 0.9
    
  cot_augmentation:
    enable: true
    # num_repeats 会自动设置为 rollout.n
    # 无需单独配置
```

## 验证方法

在训练日志中查找：

```
✅ COT augmenter initialized with strategy: sequential
✅ num_repeats: 8  # 这里会显示实际的 group size
```

或者查看配置确认：

```python
# 在训练脚本中打印
print(f"GRPO Group Size: {config.actor_rollout_ref.rollout.n}")
```

## 总结

✅ **代码实现**：完全动态，无硬编码
- `num_repeats` 从配置读取
- 支持任意 group size

❌ **文档问题**：为了举例方便用了"4"
- 已更正为 "n" 或 "group size"
- 强调这是可配置的

🎯 **使用建议**：
- 根据任务难度选择合适的 group size
- 确保 COT 例子数量 ≥ group size（或使用循环）
- 从小 group size 开始实验，逐步增加

感谢您的细心发现！这个澄清非常重要。🙏

