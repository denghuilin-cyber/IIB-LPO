# vLLM 0.8.5.post1 熵计算修改指南

本文档说明如何修改 vLLM 源码以支持在 GPU 上计算每个 token 的精确熵值并回传给 VERL。

## 前提条件

1. Fork vLLM 仓库: `git clone https://github.com/vllm-project/vllm.git`
2. 切换到 v0.8.5 分支: `cd vllm && git checkout v0.8.5`
3. 创建新分支: `git checkout -b entropy-support`

## 修改步骤

### 步骤 1: 修改 SamplingParams (vllm/sampling_params.py)

在 `SamplingParams` 类中添加 `compute_entropy` 参数：

```python
# 在 SamplingParams 类定义中添加
class SamplingParams:
    # ... 现有参数 ...
    
    def __init__(
        self,
        n: int = 1,
        best_of: Optional[int] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        seed: Optional[int] = None,
        use_beam_search: bool = False,
        length_penalty: float = 1.0,
        early_stopping: Union[bool, str] = False,
        stop: Optional[Union[str, List[str]]] = None,
        stop_token_ids: Optional[List[int]] = None,
        include_stop_str_in_output: bool = False,
        ignore_eos: bool = False,
        max_tokens: Optional[int] = 16,
        min_tokens: int = 0,
        logprobs: Optional[int] = None,
        prompt_logprobs: Optional[int] = None,
        detokenize: bool = True,
        skip_special_tokens: bool = True,
        spaces_between_special_tokens: bool = True,
        logits_processors: Optional[List[LogitsProcessor]] = None,
        truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None,
        # 新增参数
        compute_entropy: bool = False,  # 是否计算熵
    ):
        # ... 现有初始化代码 ...
        self.compute_entropy = compute_entropy
```

### 步骤 2: 修改 CompletionOutput (vllm/outputs.py)

在 `CompletionOutput` 类中添加 `entropies` 字段：

```python
class CompletionOutput:
    """The output data of one completion output of a request."""

    def __init__(
        self,
        index: int,
        text: str,
        token_ids: List[int],
        cumulative_logprob: Optional[float],
        logprobs: Optional[SampleLogprobs],
        finish_reason: Optional[str] = None,
        stop_reason: Optional[Union[int, str]] = None,
        lora_request: Optional[LoRARequest] = None,
        # 新增参数
        entropies: Optional[List[float]] = None,  # 每个 token 的熵值
    ):
        self.index = index
        self.text = text
        self.token_ids = token_ids
        self.cumulative_logprob = cumulative_logprob
        self.logprobs = logprobs
        self.finish_reason = finish_reason
        self.stop_reason = stop_reason
        self.lora_request = lora_request
        # 新增字段
        self.entropies = entropies if entropies is not None else []
```

### 步骤 3: 修改 SequenceOutput (vllm/sequence.py)

在 `SequenceOutput` 或 `SequenceData` 类中添加熵存储：

```python
class SequenceData:
    """Data associated with a sequence."""

    def __init__(self, prompt_token_ids: List[int]):
        self.prompt_token_ids = prompt_token_ids
        self.output_token_ids: List[int] = []
        self.cumulative_logprob = 0.0
        # 新增字段
        self.output_entropies: List[float] = []  # 存储每个输出 token 的熵
    
    def append_token_id(
        self,
        token_id: int,
        logprobs: Dict[int, Logprob],
        entropy: Optional[float] = None,  # 新增参数
    ) -> None:
        self.output_token_ids.append(token_id)
        # ... 现有代码 ...
        # 新增：存储熵值
        if entropy is not None:
            self.output_entropies.append(entropy)
```

### 步骤 4: 修改 Sampler (vllm/model_executor/layers/sampler.py)

在采样器中计算熵并添加到输出：

```python
# 在 Sampler 类的 forward 方法中

def forward(
    self,
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> Optional[SamplerOutput]:
    """
    Args:
        logits: (num_tokens, vocab_size)
        sampling_metadata: Metadata for sampling
    """
    # ... 现有采样逻辑 ...
    
    # 新增：计算熵（如果需要）
    entropies = None
    if sampling_metadata.compute_entropy:  # 需要在 SamplingMetadata 中传递此标志
        # 在 GPU 上计算熵
        # logits shape: (num_tokens, vocab_size)
        probs = torch.softmax(logits, dim=-1)
        # 使用数值稳定的熵计算公式
        # H = log(sum(exp(logits))) - sum(probs * logits)
        log_sum_exp = torch.logsumexp(logits, dim=-1)
        weighted_logits = torch.sum(probs * logits, dim=-1)
        entropies = log_sum_exp - weighted_logits  # shape: (num_tokens,)
    
    # 将熵添加到 SamplerOutput
    return SamplerOutput(
        outputs=outputs,
        sampled_token_ids=sampled_token_ids,
        logprobs=logprobs,
        entropies=entropies,  # 新增字段
    )
```

### 步骤 5: 修改 LLMEngine (vllm/engine/llm_engine.py)

在引擎的 step 方法中传递熵值：

```python
def _process_model_outputs(
    self,
    output: SamplerOutput,
    scheduler_outputs: SchedulerOutputs,
    ...
) -> List[RequestOutput]:
    """Process model outputs and update sequences."""
    
    # ... 现有代码 ...
    
    # 处理每个序列组
    for seq_group in scheduler_outputs.scheduled_seq_groups:
        seq_group_id = seq_group.seq_group_id
        
        for seq in seq_group.seqs:
            # 获取该序列的输出
            seq_id = seq.seq_id
            
            # 新增：如果有熵值，提取并存储
            entropy = None
            if output.entropies is not None:
                # 找到该序列在 batch 中的索引
                seq_idx = self._get_seq_index_in_batch(seq_id, ...)
                entropy = output.entropies[seq_idx].item()
            
            # 更新序列数据
            seq.data.append_token_id(
                token_id=new_token_id,
                logprobs=logprobs_dict,
                entropy=entropy,  # 传递熵值
            )
    
    # 构造 RequestOutput 时包含熵值
    request_outputs = []
    for seq_group in finished_seq_groups:
        request_outputs.append(
            RequestOutput(
                request_id=seq_group.request_id,
                prompt=seq_group.prompt,
                outputs=[
                    CompletionOutput(
                        index=seq.seq_id,
                        text=self._decode_sequence(seq),
                        token_ids=seq.data.output_token_ids,
                        cumulative_logprob=seq.data.cumulative_logprob,
                        logprobs=seq.data.output_logprobs,
                        entropies=seq.data.output_entropies,  # 新增：传递熵值列表
                    )
                    for seq in seq_group.seqs
                ],
            )
        )
    
    return request_outputs
```

### 步骤 6: 修改 SamplingMetadata

确保 `compute_entropy` 标志能够传递到 Sampler：

```python
# 在构造 SamplingMetadata 时
class SamplingMetadata:
    def __init__(
        self,
        seq_groups: List[SequenceGroupMetadata],
        selected_token_indices: torch.Tensor,
        categorized_sample_indices: Dict[SamplingType, torch.Tensor],
        num_prompts: int,
        # 新增
        compute_entropy: bool = False,
    ):
        self.seq_groups = seq_groups
        self.selected_token_indices = selected_token_indices
        self.categorized_sample_indices = categorized_sample_indices
        self.num_prompts = num_prompts
        self.compute_entropy = compute_entropy  # 新增字段
```

## 编译和安装

修改完成后，重新安装 vLLM：

```bash
cd vllm
pip uninstall vllm -y
pip install -e .
```

## 验证修改

创建测试脚本验证熵计算：

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2-7B-Instruct")

sampling_params = SamplingParams(
    temperature=0.8,
    max_tokens=50,
    compute_entropy=True,  # 启用熵计算
)

outputs = llm.generate(["你好，请介绍一下自己"], sampling_params)

for output in outputs:
    print(f"Generated text: {output.outputs[0].text}")
    print(f"Token IDs: {output.outputs[0].token_ids}")
    if output.outputs[0].entropies:
        print(f"Entropies: {output.outputs[0].entropies}")
        print(f"Average entropy: {sum(output.outputs[0].entropies) / len(output.outputs[0].entropies):.4f}")
```

## 注意事项

1. **版本兼容性**: 此修改基于 vLLM 0.8.5，不同版本的代码结构可能有差异
2. **性能影响**: 熵计算会增加少量计算开销（一次 softmax），但在 GPU 上执行很快
3. **数值稳定性**: 使用 `logsumexp` 公式避免数值下溢
4. **维护成本**: 需要维护 vLLM fork，升级时注意合并冲突

## 下一步

修改完 vLLM 后，继续修改 VERL 代码以接收和使用熵值（见 VERL_ENTROPY_INTEGRATION.md）

