# VERL 侧熵值集成说明

本文档说明如何在 VERL 中接收和使用 vLLM 回传的熵值。

## 已完成的修改

### 1. vLLM Rollout 修改 (`verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`)

#### 修改 1: 添加 `compute_entropy` 参数支持

**位置**: 第 205-221 行

```python
kwargs = dict(
    n=1,
    logprobs=0,
    max_tokens=config.response_length,
    repetition_penalty=config.get("repetition_penalty", 1.0),
    compute_entropy=config.get("compute_entropy", False),  # 新增
)
```

**说明**: 从配置中读取 `compute_entropy` 参数并传递给 vLLM 的 `SamplingParams`。

#### 修改 2: 提取熵值

**位置**: 第 344-374 行

```python
response = []
rollout_log_probs = []
rollout_entropies = []  # 新增：存储熵值

for output in outputs:
    for sample_id in range(len(output.outputs)):
        response_ids = output.outputs[sample_id].token_ids
        response.append(response_ids)
        
        if self.config.calculate_log_probs:
            curr_log_prob = []
            for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                curr_log_prob.append(logprob[response_ids[i]].logprob)
            rollout_log_probs.append(curr_log_prob)
        
        # 新增：提取熵值
        if hasattr(output.outputs[sample_id], 'entropies') and output.outputs[sample_id].entropies:
            rollout_entropies.append(output.outputs[sample_id].entropies)

# 处理熵值（padding 和类型转换）
if rollout_entropies:
    rollout_entropies = pad_2d_list_to_length(
        rollout_entropies, 0.0, max_length=self.config.response_length
    ).to(idx.device)
    rollout_entropies = rollout_entropies.to(torch.float32)
```

**说明**: 从 vLLM 的输出中提取每个 token 的熵值，进行 padding 并转换为 tensor。

#### 修改 3: 添加熵值到返回的 DataProto

**位置**: 第 395-414 行

```python
batch = TensorDict(
    {
        "prompts": idx,
        "responses": response,
        "input_ids": seq,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    },
    batch_size=batch_size,
)

if self.config.calculate_log_probs:
    batch["rollout_log_probs"] = rollout_log_probs

# 新增：如果计算了熵值，添加到 batch 中
if rollout_entropies:
    batch["rollout_entropies"] = rollout_entropies

return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
```

**说明**: 将熵值添加到 batch 字典中，键名为 `rollout_entropies`。

### 2. PPO Trainer 修改 (`verl/trainer/ppo/ray_trainer.py`)

#### 修改: 输出 rollout 熵值

**位置**: 第 1221-1240 行

```python
# repeat to align with repeated responses in rollout
batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
batch = batch.union(gen_batch_output)

# 新增：输出 rollout 阶段的熵值（如果有）
if "rollout_entropies" in batch.batch.keys():
    rollout_entropies = batch.batch["rollout_entropies"]
    response_masks = compute_response_mask(batch)
    loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
    rollout_entropy_agg = agg_loss(
        loss_mat=rollout_entropies, 
        loss_mask=response_masks, 
        loss_agg_mode=loss_agg_mode
    )
    rollout_entropy_metrics = {"rollout/entropy": rollout_entropy_agg.detach().item()}
    metrics.update(rollout_entropy_metrics)
    print(f"[Rollout Entropy] Step {self.global_steps}: {rollout_entropy_agg.detach().item():.4f}")
```

**说明**: 
- 检查 batch 中是否有 `rollout_entropies`
- 使用 `agg_loss` 函数聚合熵值（与现有的 log_prob 处理方式一致）
- 将聚合后的熵值添加到 metrics 中，键名为 `rollout/entropy`
- 打印到控制台，方便实时查看

## 使用方法

### 1. 修改 vLLM 源码

按照 `VLLM_ENTROPY_MODIFICATIONS.md` 中的说明修改 vLLM 源码并重新安装。

### 2. 启用熵计算

在训练脚本或配置中添加：

```bash
actor_rollout_ref.rollout.compute_entropy=True
```

### 3. 运行训练

使用提供的示例脚本：

```bash
bash examples/grpo_trainer/run_qwen2-7b_math_with_entropy.sh
```

### 4. 查看输出

训练过程中会看到类似输出：

```
[Rollout Entropy] Step 1: 2.3456
[Rollout Entropy] Step 2: 2.3123
[Rollout Entropy] Step 3: 2.2890
...
```

同时，熵值会被记录到 wandb/tensorboard 中，指标名称为 `rollout/entropy`。

## 熵值说明

### rollout/entropy vs actor/entropy

- **`rollout/entropy`** (新增): 
  - 在 rollout 阶段计算
  - 使用 vLLM 推理引擎
  - 在 GPU 上从完整 logits 计算
  - 反映生成时的 policy 熵
  - 用于监控和分析

- **`actor/entropy`** (已存在):
  - 在训练阶段计算
  - 使用 Actor 模型重新前向传播
  - 用于熵正则化
  - 反映训练中的 policy 熵

### 熵值范围

- 理论范围: [0, log(vocab_size)]
- 对于 vocab_size=32000: [0, ~10.4]
- 典型值: 2.0 - 4.0
- 过低 (<1.0): 可能出现熵崩塌
- 过高 (>6.0): 模型过于随机

## 故障排除

### 1. 熵值未输出

**检查**:
- vLLM 是否正确修改并重新安装
- 配置中是否设置 `compute_entropy=True`
- 查看日志中是否有 vLLM 相关错误

### 2. 熵值全为 0

**可能原因**:
- vLLM 的 `CompletionOutput` 未正确添加 `entropies` 字段
- Sampler 中的熵计算逻辑未执行
- 检查 vLLM 修改是否完整

### 3. 熵值异常（NaN 或 Inf）

**可能原因**:
- Logits 中有 NaN 或 Inf
- Softmax 计算溢出
- 确保使用数值稳定的熵计算公式（logsumexp）

## 下一步

方案 C 完成后，您可以：

1. **监控熵崩塌**: 观察 `rollout/entropy` 随训练步数的变化
2. **对比分析**: 比较 `rollout/entropy` 和 `actor/entropy` 的差异
3. **实现方案 2**: 基于熵值实现动态分叉（I²B-LPO）

## 注意事项

1. **性能影响**: 熵计算会增加约 5-10% 的推理时间（一次额外的 softmax）
2. **内存占用**: 每个 token 存储一个 float32 熵值
3. **版本兼容**: 基于 vLLM 0.8.5.post1，其他版本可能需要调整
4. **维护成本**: 需要维护 vLLM fork，升级时注意合并冲突

