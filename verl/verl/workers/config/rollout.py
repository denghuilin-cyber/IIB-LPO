# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from omegaconf import MISSING

from verl.base_config import BaseConfig
from verl.utils.profiler import ProfilerConfig

__all__ = [
    "SamplingConfig",
    "MultiTurnConfig",
    "CustomAsyncServerConfig",
    "AgentLoopConfig",
    "TraceConfig",
    "COTAugmentationConfig",
    "EntropyOutputConfig",
    "RolloutConfig",
]


@dataclass
class SamplingConfig(BaseConfig):
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    do_sample: bool = True
    n: int = 1


@dataclass
class MultiTurnConfig(BaseConfig):
    _mutable_fields = {"max_assistant_turns", "max_user_turns"}

    enable: bool = False
    max_assistant_turns: Optional[int] = None
    tool_config_path: Optional[str] = None
    max_user_turns: Optional[int] = None
    max_parallel_calls: int = 1
    max_tool_response_length: int = 256
    tool_response_truncate_side: str = "middle"
    interaction_config_path: Optional[str] = None
    use_inference_chat_template: bool = False
    tokenization_sanity_check_mode: str = "strict"
    format: str = "hermes"


@dataclass
class CustomAsyncServerConfig(BaseConfig):
    path: Optional[str] = None
    name: Optional[str] = None


@dataclass
class AgentLoopConfig(BaseConfig):
    num_workers: int = 8
    agent_loop_config_path: Optional[str] = None
    custom_async_server: CustomAsyncServerConfig = field(default_factory=CustomAsyncServerConfig)


@dataclass
class TraceConfig(BaseConfig):
    backend: Optional[str] = None
    token2text: bool = False


@dataclass
class COTAugmentationConfig(BaseConfig):
    """
    Configuration for Chain-of-Thought (COT) augmentation in GRPO.
    
    Enables adding different COT examples to each rollout repetition.
    """
    enable: bool = False
    
    # Single dataset COT configuration
    cot_file_path: Optional[str] = None
    loader_path: Optional[str] = None
    
    # Multi-dataset COT configuration
    use_multi_dataset: bool = False
    dataset_cot_mapping: Optional[Dict[str, str]] = None
    cot_file_mapping: Optional[Dict[str, str]] = None  # Alias for dataset_cot_mapping
    
    # COT formatting and matching
    cot_format_template: str = "Here is a reference example that demonstrates the problem-solving approach:\n\n<Example>\nQuestion: {question}\n\nStep-by-step Solution:\n{rationale}\n\nFinal Answer: {final_answer}\n</Example>\n\nNow, please solve the following problem using similar reasoning:"
    match_by: str = "question"  # or "id"
    use_full_cot: bool = True
    skip_on_mismatch: bool = True
    
    # Sampling strategy for COT examples
    sampling_strategy: str = "sequential"  # or "random_with_replacement", "random_without_replacement"
    
    # Debug/logging
    verbose: bool = False
    debug_print_augmented_prompts: bool = True  # Print augmented prompts for debugging
    debug_num_samples: int = 3  # Number of samples to print for debugging
    debug_print_full_prompt: bool = False  # 🆕 Print full prompt without truncation
    
    # Additional settings
    add_separator: bool = True
    separator: str = "\n\n"


@dataclass
class EntropyOutputConfig(BaseConfig):
    """
    Configuration for entropy output to JSONL files.
    
    Enables saving token-level entropy data during rollout phase.
    """
    enabled: bool = False  # 是否启用熵输出
    output_dir: str = "./entropy_outputs"  # 输出目录
    top_k: int = 10  # 标记熵值最高的 K 个 token
    save_interval: int = 10  # 每 N 个 step 保存一次
    mark_style: str = "both"  # "markdown" | "html" | "both"
    token_entropy_to_jsonl: bool = False  # DEBUG: 是否保存 token-level entropy


@dataclass
class ServerConfig(BaseConfig):
    """
    Configuration for SGLang server when running in server mode
    """

    timeout: float = 60.0
    max_attempts: int = 3
    retry_delay: float = 2.0
    max_connections: int = 1000
    max_start_wait_time: float = 300.0


@dataclass
class RolloutConfig(BaseConfig):
    _mutable_fields = {"max_model_len"}

    name: Optional[str] = MISSING
    mode: str = "sync"

    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    do_sample: bool = True
    n: int = 1

    # Early termination threshold for multi-turn rollout in sglang.
    # Abort remaining requests when (1 - over_sample_rate) * total_requests are completed.
    over_sample_rate: float = 0.0

    prompt_length: int = 512
    response_length: int = 512

    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.5
    ignore_eos: bool = False
    enforce_eager: bool = True
    cudagraph_capture_sizes: Optional[list] = None
    free_cache_engine: bool = True
    tensor_model_parallel_size: int = 2
    max_num_batched_tokens: int = 8192

    # TODO: enable train_kwargs
    # train_sampling_config: SamplingConfig = field(default_factory=SamplingConfig)

    val_kwargs: SamplingConfig = field(default_factory=SamplingConfig)

    max_model_len: Optional[int] = None
    max_num_seqs: int = 1024

    # note that the logprob computation should belong to the actor
    log_prob_micro_batch_size: Optional[int] = None
    log_prob_micro_batch_size_per_gpu: Optional[int] = None
    log_prob_use_dynamic_bsz: bool = False
    log_prob_max_token_len_per_gpu: int = 16384

    disable_log_stats: bool = True

    multi_stage_wake_up: bool = False
    engine_kwargs: dict = field(default_factory=dict)

    calculate_log_probs: bool = False
    
    # 新增：是否计算每个 token 的熵值（需要修改 vLLM 源码）
    compute_entropy: bool = False
    
    # 🆕 CVAE 分叉配置
    enable_cvae_branching: bool = False
    cvae_num_branches_per_path: int = 1  # 每条路径分叉次数 k
    cvae_branching_mode: str = "random"  # 分叉模式: "random"(随机), "input"(INPUT fusion), "psa"(PSA fusion), "softmax"(SOFTMAX fusion)
    cvae_model_path: str = "/nas/dhl/CVAE/models/GSM8K-MATH-trained/lars_selector_GSM8K-MATH.pth"
    cvae_embedding_model_path: str = "/nas/dhl/CVAE/models/deberta-v2-xlarge"
    cvae_injection_layers: int = 4  # PSA 模式下注入的层数（默认最后4层，"all" 表示所有层）
    cvae_latent_dim: int = 128  # 潜在向量维度（CVAE 的 z 向量维度）
    cvae_embedding_dim: int = 1536  # 嵌入向量维度（DeBERTa-v2-xlarge 的输出维度）

    agent: AgentLoopConfig = field(default_factory=AgentLoopConfig)

    trace: TraceConfig = field(default_factory=TraceConfig)

    multi_turn: MultiTurnConfig = field(default_factory=MultiTurnConfig)
    
    # COT augmentation configuration for GRPO
    cot_augmentation: Optional[COTAugmentationConfig] = field(default_factory=COTAugmentationConfig)
    
    # Entropy output configuration
    entropy_output: Optional[EntropyOutputConfig] = field(default_factory=EntropyOutputConfig)

    # Server configuration for sglang server mode
    server: ServerConfig = field(default_factory=ServerConfig)

    update_weights_bucket_megabytes: int = 512

    skip_rollout: bool = False

    skip_dump_dir: str = "/tmp/rollout_dump"

    profiler: Optional[ProfilerConfig] = None

    enable_chunked_prefill: bool = True
    load_format: str = "dummy_dtensor"

    layered_summon: bool = False

    layer_name_map: dict = field(default_factory=dict)

    sglang_engine_mode: str = "local"

    limit_images: Optional[int] = None
