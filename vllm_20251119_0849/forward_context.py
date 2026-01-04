# SPDX-License-Identifier: Apache-2.0

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group,
                                          is_v1_kv_transfer_group)
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorBase_V1
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata

logger = init_logger(__name__)

track_batchsize: bool = envs.VLLM_LOG_BATCHSIZE_INTERVAL >= 0
last_logging_time: float = 0
forward_start_time: float = 0
batchsize_logging_interval: float = envs.VLLM_LOG_BATCHSIZE_INTERVAL
batchsize_forward_time: defaultdict = defaultdict(list)


@dataclass
class DPMetadata:
    cu_tokens_across_dp_cpu: torch.Tensor


@dataclass
class ZInjectionConfig:
    """
    Configuration for z injection in CVAE branching modes.
    
    用于 I²B-LPO 分叉生成的 z 注入配置。
    
    🔒 互斥性保证：
    - mode="input" 时，只有 z_proj_input 有值
    - mode="psa" 时，只有 z_proj_psa 和 psa_injection_layers 有值
    - mode="softmax" 时，只有 z_proj_vocab 有值
    - mode="random" 时，不使用 ForwardContext（所有字段为 None）
    - 任何时刻只能有一个模式生效
    """
    mode: Optional[str] = None  # "input", "psa", "softmax", None (random 不使用)
    
    # INPUT Fusion: 在第一层前注入
    z_proj_input: Optional[torch.Tensor] = None  # [1, hidden_size]
    
    # PSA Fusion: 在指定层后注入
    z_proj_psa: Optional[torch.Tensor] = None  # [1, hidden_size]
    psa_injection_layers: Optional[list] = None  # [24, 25, 26, 27]
    
    # SOFTMAX Fusion: 在 lm_head 后注入
    z_proj_vocab: Optional[torch.Tensor] = None  # [vocab_size]


def validate_z_injection_config(config: Optional['ZInjectionConfig']) -> bool:
    """
    验证 z_injection_config 的互斥性
    
    Args:
        config: z 注入配置
    
    Returns:
        True if valid
    
    Raises:
        ValueError: 如果配置违反互斥性规则
    """
    if config is None:
        return True
    
    mode = config.mode
    z_proj_input = config.z_proj_input
    z_proj_psa = config.z_proj_psa
    psa_injection_layers = config.psa_injection_layers
    z_proj_vocab = config.z_proj_vocab
    
    # 规则1: mode 必须是合法值
    if mode not in ["input", "psa", "softmax", None]:
        raise ValueError(
            f"❌ 非法的 mode: {mode}\n"
            f"   合法值: 'input', 'psa', 'softmax', None\n"
            f"   注意: 'random' 模式不使用 ForwardContext"
        )
    
    # 规则2: mode="input" 时，只能有 z_proj_input
    if mode == "input":
        if z_proj_input is None:
            raise ValueError("❌ mode='input' 但 z_proj_input 为 None")
        if z_proj_psa is not None or z_proj_vocab is not None:
            raise ValueError(
                "❌ mode='input' 时不允许设置其他投影\n"
                "   INPUT Fusion 与 PSA/SOFTMAX Fusion 是互斥的！"
            )
    
    # 规则3: mode="psa" 时，只能有 z_proj_psa 和 psa_injection_layers
    if mode == "psa":
        if z_proj_psa is None:
            raise ValueError("❌ mode='psa' 但 z_proj_psa 为 None")
        if psa_injection_layers is None or len(psa_injection_layers) == 0:
            raise ValueError("❌ mode='psa' 但 psa_injection_layers 为空")
        if z_proj_input is not None or z_proj_vocab is not None:
            raise ValueError(
                "❌ mode='psa' 时不允许设置其他投影\n"
                "   PSA Fusion 与 INPUT/SOFTMAX Fusion 是互斥的！"
            )
    
    # 规则4: mode="softmax" 时，只能有 z_proj_vocab
    if mode == "softmax":
        if z_proj_vocab is None:
            raise ValueError("❌ mode='softmax' 但 z_proj_vocab 为 None")
        if z_proj_input is not None or z_proj_psa is not None:
            raise ValueError(
                "❌ mode='softmax' 时不允许设置其他投影\n"
                "   SOFTMAX Fusion 与 INPUT/PSA Fusion 是互斥的！"
            )
    
    # 规则5: mode=None 时，所有投影都不能有值
    if mode is None:
        if z_proj_input is not None or z_proj_psa is not None or z_proj_vocab is not None:
            raise ValueError("❌ mode=None 时不应该有任何投影向量")
    
    return True


@dataclass
class ForwardContext:
    # copy from vllm_config.compilation_config.static_forward_context
    no_compile_layers: dict[str, Any]
    # TODO: extend to support per-layer dynamic forward context
    attn_metadata: "AttentionMetadata"  # set dynamically for each forward pass
    # TODO: remove after making all virtual_engines share the same kv cache
    virtual_engine: int  # set dynamically for each forward pass
    # set dynamically for each forward pass
    dp_metadata: Optional[DPMetadata] = None
    # 🆕 Z injection config for CVAE branching (I²B-LPO)
    z_injection_config: Optional[ZInjectionConfig] = None


_forward_context: Optional[ForwardContext] = None


def get_forward_context() -> ForwardContext:
    """Get the current forward context."""
    assert _forward_context is not None, (
        "Forward context is not set. "
        "Please use `set_forward_context` to set the forward context.")
    return _forward_context


@contextmanager
def set_forward_context(attn_metadata: Any,
                        vllm_config: VllmConfig,
                        virtual_engine: int = 0,
                        num_tokens: int = 0,
                        z_injection_config: Optional[ZInjectionConfig] = None):
    """A context manager that stores the current forward context,
    can be attention metadata, etc.
    Here we can inject common logic for every model forward pass.
    
    Args:
        z_injection_config: Optional configuration for z injection (I²B-LPO branching)
    """
    global forward_start_time
    need_to_track_batchsize = track_batchsize and attn_metadata is not None
    if need_to_track_batchsize:
        forward_start_time = time.perf_counter()
    dp_metadata: Optional[DPMetadata] = None
    if vllm_config.parallel_config.data_parallel_size > 1:
        dp_size = vllm_config.parallel_config.data_parallel_size
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        if attn_metadata is not None:
            if hasattr(attn_metadata, "num_prefill_tokens"):
                # for v0 attention backends
                batchsize = attn_metadata.num_prefill_tokens + \
                    attn_metadata.num_decode_tokens
            else:
                # for v1 attention backends
                batchsize = attn_metadata.num_input_tokens
        else:
            batchsize = num_tokens
        num_tokens_across_dp = [0] * dp_size
        num_tokens_across_dp[dp_rank] = batchsize
        num_tokens_tensor = torch.tensor(num_tokens_across_dp,
                                         device="cpu",
                                         dtype=torch.int32)
        from vllm.distributed.parallel_state import get_dp_group
        dist.all_reduce(num_tokens_tensor, group=get_dp_group().cpu_group)
        cu_tokens_across_dp_cpu = torch.cumsum(num_tokens_tensor, dim=0)
        dp_metadata = DPMetadata(cu_tokens_across_dp_cpu)

    global _forward_context
    prev_context = _forward_context
    _forward_context = ForwardContext(
        no_compile_layers=vllm_config.compilation_config.
        static_forward_context,
        virtual_engine=virtual_engine,
        attn_metadata=attn_metadata,
        dp_metadata=dp_metadata,
        z_injection_config=z_injection_config)

    # KVConnector: trigger (possibly async) load before forward.
    # Each attn layer will block until the reading is complete.
    trigger_kv_transfer = (attn_metadata is not None
                           and has_kv_transfer_group()
                           and is_v1_kv_transfer_group())
    if trigger_kv_transfer:
        kv_connector = get_kv_transfer_group()
        assert isinstance(kv_connector, KVConnectorBase_V1)
        kv_connector.start_load_kv(_forward_context)

    try:
        yield
    finally:
        global last_logging_time, batchsize_logging_interval
        if need_to_track_batchsize:
            if hasattr(attn_metadata, "num_prefill_tokens"):
                # for v0 attention backends
                batchsize = attn_metadata.num_prefill_tokens + \
                    attn_metadata.num_decode_tokens
            else:
                # for v1 attention backends
                batchsize = attn_metadata.num_input_tokens
            # we use synchronous scheduling right now,
            # adding a sync point here should not affect
            # scheduling of the next batch
            torch.cuda.synchronize()
            now = time.perf_counter()
            # time measurement is in milliseconds
            batchsize_forward_time[batchsize].append(
                (now - forward_start_time) * 1000)
            if now - last_logging_time > batchsize_logging_interval:
                last_logging_time = now
                forward_stats = []
                for bs, times in batchsize_forward_time.items():
                    if len(times) <= 1:
                        # can be cudagraph / profiling run
                        continue
                    medium = torch.quantile(torch.tensor(times), q=0.5).item()
                    medium = round(medium, 2)
                    forward_stats.append((bs, len(times), medium))
                forward_stats.sort(key=lambda x: x[1], reverse=True)
                if forward_stats:
                    logger.info(("Batchsize forward time stats "
                                 "(batchsize, count, median_time(ms)): %s"),
                                forward_stats)

        # KVConnector: each attn layer triggers (possibly async) save.
        # Ensure all those operations complete before forward() is done.
        if trigger_kv_transfer:
            kv_connector = get_kv_transfer_group()
            assert isinstance(kv_connector, KVConnectorBase_V1)
            kv_connector.wait_for_save()

        _forward_context = prev_context
