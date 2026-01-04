# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import asyncio
import getpass
import inspect
import logging
import os
import pickle
import socket
import time
from contextlib import contextmanager
from dataclasses import asdict
from types import MethodType
from typing import Any, Generator

import numpy as np
import ray
import torch
import torch.distributed
import zmq
import zmq.asyncio
from filelock import FileLock
from omegaconf import ListConfig
from tensordict import TensorDict
from torch.distributed.device_mesh import DeviceMesh
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig, CompilationLevel
from vllm.lora.request import LoRARequest
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.worker.worker_base import WorkerWrapperBase

from verl import DataProto
from verl.third_party.vllm import VLLM_SLEEP_LEVEL
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.ray_utils import ray_noset_visible_devices
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.utils.vllm import TensorLoRARequest, VLLMHijack, is_version_ge
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout
from verl.utils.reward_score import default_compute_score  # 🆕 用于计算 reward

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> list[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


if is_version_ge(pkg="vllm", minver="0.7.3"):
    VLLMHijack.hijack()


class vLLMRollout(BaseRollout):
    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)

        model_path = model_config.local_path
        tokenizer = model_config.tokenizer
        self.tokenizer = tokenizer  # 🆕 保存 tokenizer 用于 debug
        model_hf_config = model_config.hf_config
        trust_remote_code = model_config.trust_remote_code
        self.lora_kwargs = (
            {"enable_lora": True, "max_loras": 1, "max_lora_rank": model_config.lora_rank}
            if model_config.lora_rank > 0
            else {}
        )

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(
                model_hf_config.llm_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(
                model_hf_config.text_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")
            assert max_position_embeddings >= config.prompt_length + config.response_length, (
                "model context length should be greater than total sequence length"
            )
        else:
            # handle type where there's a length extend factor
            # see https://qwen.readthedocs.io/en/latest/deployment/vllm.html#extended-context-support
            # for using yarn as an example
            rope_scaling_factor = rope_scaling_config.get("factor", 1.0)

            assert (
                model_hf_config.max_position_embeddings * rope_scaling_factor
                >= config.prompt_length + config.response_length
            ), (
                "model context length should be greater than total sequence length, "
                + f"got rope_scaling_factor={rope_scaling_factor} and "
                + f"max_position_embeddings={model_hf_config.max_position_embeddings}"
            )

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        # copy it to avoid secretly modifying the engine config
        engine_kwargs = config.get("engine_kwargs", {}).get("vllm", {}) or {}

        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

        compilation_config = {}

        cudagraph_capture_sizes = config.get("cudagraph_capture_sizes")
        # enforce_eager must be False to use cudagraph
        if not config.enforce_eager and cudagraph_capture_sizes:
            if isinstance(cudagraph_capture_sizes, ListConfig):
                compilation_config["compilation_config"] = CompilationConfig(
                    level=CompilationLevel.PIECEWISE, cudagraph_capture_sizes=cudagraph_capture_sizes
                )
            else:
                logger.warning(f"cudagraph_capture_sizes must be a list, but got {cudagraph_capture_sizes}")

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=config.free_cache_engine,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            max_num_seqs=config.max_num_seqs,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **compilation_config,
            **self.lora_kwargs,
            **engine_kwargs,
        )

        # 新增：保存 compute_entropy 配置
        self.compute_entropy = config.get("compute_entropy", False)
        
        # 新增：初始化 EntropyOutputWriter
        entropy_output_config = config.get("entropy_output", None)
        if entropy_output_config and entropy_output_config.get("enabled", False):
            from verl.utils.entropy_output_writer import EntropyOutputWriter, EntropyOutputConfig
            
            # 将 dict 转换为 EntropyOutputConfig 对象
            if isinstance(entropy_output_config, dict):
                entropy_config_obj = EntropyOutputConfig(**entropy_output_config)
            else:
                entropy_config_obj = entropy_output_config
            
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            self.entropy_writer = EntropyOutputWriter(
                config=entropy_config_obj,
                tokenizer=tokenizer,
                rank=rank
            )
            logger.info(f"[vLLMRollout] EntropyOutputWriter initialized at rank {rank}")
        else:
            self.entropy_writer = None
        
        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
            repetition_penalty=config.get("repetition_penalty", 1.0),
        )

        kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)) and k != "seed":
                kwargs[k] = config.get(k)
        
        # 🆕 强制添加 compute_entropy（确保传递）
        if self.compute_entropy:
            kwargs["compute_entropy"] = True
            #logger.info(f"[VERL] 🎯 强制设置 compute_entropy=True")
        
        # 🔧 关键修复：明确设置 stop_token_ids（包含 EOS token）
        # 问题：vLLM V1 可能不会自动添加 EOS token 到停止条件
        # 解决：手动添加 tokenizer 的 EOS token ID
        if "stop_token_ids" not in kwargs or kwargs["stop_token_ids"] is None:
            kwargs["stop_token_ids"] = [tokenizer.eos_token_id]
            logger.info(f"[VERL] 🔧 设置 stop_token_ids=[{tokenizer.eos_token_id}] (EOS token)")
        
        kwargs["n"] = 1  # already repeat in ray_trainer
        #logger.debug(f"[VERL] kwargs: {kwargs}")
        #logger.debug(f"[VERL] compute_entropy in kwargs: {'compute_entropy' in kwargs}")
        self.sampling_params = SamplingParams(**kwargs)
        #logger.info(f"[VERL] self.sampling_params.compute_entropy = {self.sampling_params.compute_entropy}")
        
        # 🔧 验证 _all_stop_token_ids 是否正确设置
        if hasattr(self.sampling_params, '_all_stop_token_ids'):
            if not self.sampling_params._all_stop_token_ids or tokenizer.eos_token_id not in self.sampling_params._all_stop_token_ids:
                logger.warning(f"[VERL] ⚠️ _all_stop_token_ids 不包含 EOS token，手动修复")
                self.sampling_params._all_stop_token_ids = set([tokenizer.eos_token_id])
        else:
            logger.warning(f"[VERL] ⚠️ SamplingParams 没有 _all_stop_token_ids 属性（vLLM 版本问题）")

        self.pad_token_id = tokenizer.pad_token_id
        
        # 🆕 初始化 CVAE Manager（用于思维分叉）
        self.cvae_manager = None
        if config.enable_cvae_branching: # 如果设定为 允许分叉
            from verl.utils.cvae_branching import create_cvae_manager
            try:
                # 获取 CVAE 配置参数（带默认值）
                latent_dim = config.get("cvae_latent_dim", 128)
                embedding_dim = config.get("cvae_embedding_dim", 1536)
                
                logger.info(f"[CVAE Manager] 初始化参数:")
                logger.info(f"  latent_dim: {latent_dim}")
                logger.info(f"  embedding_dim: {embedding_dim}")
                logger.info(f"  injection_layers: {config.cvae_injection_layers}")
                
                self.cvae_manager = create_cvae_manager(
                    cvae_model_path=config.cvae_model_path,
                    embedding_model_path=config.cvae_embedding_model_path,
                    injection_layers=config.cvae_injection_layers,
                    latent_dim=latent_dim,
                    embedding_dim=embedding_dim,
                    device="cuda"
                )
                logger.info(f"✅ CVAE 分叉已启用:")
                logger.info(f"   - cvae_num_branches_per_path 每条路径分叉次数: {config.cvae_num_branches_per_path}")
                logger.info(f"   - rolloutn 初始路径数 n: {config.n}")
                logger.info(f"   - 最终产生路径数: {config.n} × (1 + {config.cvae_num_branches_per_path}) = {config.n * (1 + config.cvae_num_branches_per_path)}")
            except Exception as e:
                logger.error(f"❌ CVAE Manager 初始化失败: {e}")
                logger.error("   将禁用 CVAE 分叉功能")
                self.cvae_manager = None
                import traceback # 这两行代码用于打印详细的异常堆栈信息，帮助调试错误。
                traceback.print_exc()
                # 会显示出 具体报错的文件 和 行数
                '''对比示例
                只用 logger.error(f"{e}")：
                ❌ CVAE Manager 初始化失败: No module named 'cvae_branching'
                👆 只知道出错了，但不知道在哪里出错
                加上 traceback.print_exc()：
                会在控制台输出 具体报错文件和行数
                ❌ CVAE Manager 初始化失败: No module named 'cvae_branching'Traceback (most recent call last):  File "/path/to/vllm_rollout_spmd.py", line 262, in __init__    self.cvae_manager = create_cvae_manager(  File "/path/to/verl/utils/cvae_branching.py", line 45, in create_cvae_manager    from .cvae_model import CVAEModel  File "/path/to/verl/utils/cvae_model.py", line 10, in <module>    from some_package import somethingModuleNotFoundError: No module named 'cvae_branching'
                👆 清楚地看到：'''

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    
    def _get_llm_model(self):
        """
        获取 LLM 模型对象（用于注册 attention hooks）
        
        Returns:
            torch.nn.Module: LLM 模型
        """
        try:
            model = self.inference_engine.llm_engine.model_executor.driver_worker.model_runner.model
            return model
        except Exception as e:
            logger.error(f"[CVAE Branching] 无法获取 LLM 模型: {e}")
            raise
    
    def _continue_from_prefix(
        self,
        prompt_ids: list,
        prefix_ids: list,
        pure_question: str = None,  # 🆕 添加 pure_question 参数
        sampling_params: SamplingParams = None
    ):
        """
        从给定的 prefix 继续生成
        
        Args:
            prompt_ids: 原始 prompt 的 token ids (list)
            prefix_ids: 已生成的 prefix token ids (list)
            pure_question: 纯问题（不包含 example COT），用于长度超限时的降级
            sampling_params: 采样参数（可选，默认使用 self.sampling_params）
        
        Returns:
            vLLM RequestOutput
        """
        # 拼接 prompt + prefix 作为新的 prompt
        combined_ids = prompt_ids + prefix_ids
        
        # 🔧 新增：检查长度是否超出模型限制
        max_model_len = getattr(self.inference_engine.llm_engine.model_config, 'max_model_len', 2048)
        
        if len(combined_ids) > max_model_len:
            logger.warning(f"[CVAE Branching] ⚠️ 长度超限: {len(combined_ids)} > {max_model_len}")
            logger.warning(f"  prompt_len: {len(prompt_ids)}, prefix_len: {len(prefix_ids)}")
            
            # 🎯 策略1：尝试去除 example COT，只保留 pure question
            if pure_question is not None:
                logger.info(f"[CVAE Branching] 📝 尝试策略1: 去除 example COT，使用 pure question")
                try:
                    # 重新编码 pure question
                    pure_question_ids = self.tokenizer.encode(pure_question, add_special_tokens=True)
                    combined_ids_fallback = pure_question_ids + prefix_ids
                    
                    logger.info(f"  pure_question_len: {len(pure_question_ids)}")
                    logger.info(f"  new_combined_len: {len(combined_ids_fallback)}")
                    
                    if len(combined_ids_fallback) <= max_model_len:
                        logger.info(f"  ✅ 策略1成功: {len(combined_ids_fallback)} <= {max_model_len}")
                        combined_ids = combined_ids_fallback
                    else:
                        logger.warning(f"  ❌ 策略1失败: {len(combined_ids_fallback)} > {max_model_len}")
                        raise ValueError("策略1失败，进入策略2")
                except Exception as e:
                    logger.warning(f"  策略1异常: {e}，进入策略2")
                    # 继续执行策略2
            
            # 🎯 策略2：如果策略1失败，从右侧截断 prompt（保留最近的上下文）
            if len(combined_ids) > max_model_len:
                logger.info(f"[CVAE Branching] ✂️ 执行策略2: 截断 prompt 到 max_model_len")
                
                # 计算需要保留的 prompt 长度
                # 保留策略：prefix 优先，剩余空间留给 prompt
                max_prompt_len = max_model_len - len(prefix_ids) - 10  # 留10个token的安全边界
                
                if max_prompt_len <= 0:
                    # 极端情况：prefix 本身就超长，只能截断 prefix
                    logger.error(f"  ⚠️ prefix 本身超长 ({len(prefix_ids)} tokens)，强制截断 prefix")
                    max_prefix_len = max_model_len - 100  # 至少保留100个token给prompt
                    if max_prefix_len <= 0:
                        max_prefix_len = max_model_len // 2  # 各占一半
                    
                    prefix_ids = prefix_ids[-max_prefix_len:]  # 保留 prefix 的后半部分
                    max_prompt_len = max_model_len - len(prefix_ids) - 10
                
                # 从右侧截断 prompt（保留最近的上下文）
                if len(prompt_ids) > max_prompt_len:
                    logger.warning(f"  截断 prompt: {len(prompt_ids)} -> {max_prompt_len}")
                    prompt_ids = prompt_ids[-max_prompt_len:]  # 保留 prompt 的后半部分
                
                combined_ids = prompt_ids + prefix_ids
                logger.info(f"  ✅ 策略2完成: final_len = {len(combined_ids)} <= {max_model_len}")
        
        # 计算剩余需要生成的长度
        remaining_length = self.config.response_length - len(prefix_ids)
        if remaining_length <= 0:
            logger.warning(f"[CVAE Branching] prefix 已达到最大长度 ({len(prefix_ids)} >= {self.config.response_length})")
            remaining_length = 1  # 至少生成1个token
        
        # 使用提供的 sampling_params 或默认参数
        if sampling_params is None:
            # 🔧 修复：创建临时 sampling_params，继承所有原始参数（包括 stop 相关）
            temp_params = SamplingParams(
                temperature=self.sampling_params.temperature,
                top_p=self.sampling_params.top_p,
                top_k=self.sampling_params.top_k,
                max_tokens=remaining_length,
                logprobs=self.sampling_params.logprobs,
                prompt_logprobs=self.sampling_params.prompt_logprobs,
                compute_entropy=self.compute_entropy,  # 保持熵计算
                # 🔧 关键修复：继承停止条件，避免重复生成
                stop=self.sampling_params.stop,  # 停止词列表
                stop_token_ids=self.sampling_params.stop_token_ids,  # EOS token IDs
                ignore_eos=self.sampling_params.ignore_eos,  # 是否忽略 EOS
            )
            # 🔧 关键修复：安全地继承 _all_stop_token_ids（包含 EOS token ID）
            if hasattr(self.sampling_params, '_all_stop_token_ids') and self.sampling_params._all_stop_token_ids:
                temp_params._all_stop_token_ids = self.sampling_params._all_stop_token_ids.copy()
            else:
                # Fallback: 如果原始的 _all_stop_token_ids 不存在或为空，手动创建
                temp_params._all_stop_token_ids = set([self.tokenizer.eos_token_id])
                logger.warning(f"[CVAE Branching] ⚠️ 原始 _all_stop_token_ids 为空，手动设置为 {{{self.tokenizer.eos_token_id}}}")
            
            # 🔧 额外保险：确保 EOS token 一定在集合中
            if self.tokenizer.eos_token_id not in temp_params._all_stop_token_ids:
                temp_params._all_stop_token_ids.add(self.tokenizer.eos_token_id)
                logger.warning(f"[CVAE Branching] ⚠️ _all_stop_token_ids 缺少 EOS token，已添加 {self.tokenizer.eos_token_id}")
        else:
            temp_params = sampling_params
        
        logger.debug(f"[CVAE Branching] 从 prefix 继续生成:")
        logger.debug(f"  prompt_len: {len(prompt_ids)}")
        logger.debug(f"  prefix_len: {len(prefix_ids)}")
        logger.debug(f"  combined_len: {len(combined_ids)}")
        logger.debug(f"  remaining_len: {remaining_length}")
        
        # 🔍 DEBUG: 打印拼接后的 combined_ids 解码内容
        combined_text = self.tokenizer.decode(combined_ids, skip_special_tokens=False)
        # logger.info(f"\n{'='*100}")
        # logger.info(f"[CVAE Branching] 🔍 Combined Prompt 解码后的完整内容:")
        # logger.info(f"{'='*100}")
        # logger.info(combined_text)
        # logger.info(f"{'='*100}\n")
        
        # # 🔍 DEBUG: 打印停止条件
        # logger.info(f"[CVAE Branching] 🛑 停止条件:")
        # logger.info(f"  max_tokens: {temp_params.max_tokens}")
        # logger.info(f"  ignore_eos: {temp_params.ignore_eos}")
        # logger.info(f"  stop: {temp_params.stop}")
        # logger.info(f"  stop_token_ids: {temp_params.stop_token_ids}")
        # logger.info(f"  _all_stop_token_ids: {temp_params._all_stop_token_ids}")
        # logger.info(f"  tokenizer.eos_token_id: {self.tokenizer.eos_token_id}")
        
        # 调用 vLLM 生成
        try:
            outputs = self.inference_engine.generate(
                prompts=[{"prompt_token_ids": combined_ids}],
                sampling_params=temp_params,
                use_tqdm=False
            )
            # 调用 self.inference_engine.generate让模型 从prompt_ids + prefix_ids 接着往下生成
            result = outputs[0]
            
            # 🔍 DEBUG: 打印生成结果
            #logger.info(f"[CVAE Branching] ✅ 生成完成:")
            #logger.info(f"  生成 token 数: {len(result.outputs[0].token_ids)}")
            #logger.info(f"  最后 5 个 tokens: {result.outputs[0].token_ids[-5:]}")
            #logger.info(f"  最后 5 个 tokens 解码: {self.tokenizer.decode(result.outputs[0].token_ids[-5:])}")
            #logger.info(f"  finish_reason: {result.outputs[0].finish_reason if hasattr(result.outputs[0], 'finish_reason') else 'N/A'}")
            
            return result
        except Exception as e:
            logger.error(f"[CVAE Branching] 从 prefix 继续生成失败: {e}")
            raise
    
    def _sample_z_from_question_and_prefix(
        self,
        pure_question: str,
        prefix_text: str,
        num_samples: int = 1
    ) -> torch.Tensor: # 采样zi
        """
        从 question + prefix 采样 CVAE 的 latent z
        
        Args:
            pure_question: 纯问题（不包含 example COT）
            prefix_text: 已生成的 prefix 文本
            num_samples: 采样数量（默认1，单次分叉）
        
        Returns:
            z_samples: [num_samples, latent_dim] 的 tensor
        """
        if self.cvae_manager is None: # 如果初始化失败 就RuntimeError
            raise RuntimeError("[CVAE Branching] CVAE Manager 未初始化")
        
        # 拼接 question + prefix 作为 CVAE 输入
        cvae_input = str(pure_question) + " " + prefix_text
        
        logger.debug(f"[CVAE Branching] 采样 z:")
        logger.debug(f"  pure_question: {pure_question}...")
        logger.debug(f"  prefix_text: {prefix_text}...")
        logger.debug(f"  cvae_input pure_question+prefix_text: {cvae_input[:100]}...")
        logger.debug(f"  取num_samples个zi: {num_samples}")
        
        try:
            # 调用 CVAE 采样
            z_samples = self.cvae_manager.sample_z_from_text(
                text=cvae_input,
                num_samples=num_samples
            )
            
            logger.debug(f"[CVAE Branching] ✅ 成功采样 {num_samples} 个 z 向量")
            logger.debug(f"  z shape: {z_samples.shape}")
            logger.debug(f"  z range: [{z_samples.min():.4f}, {z_samples.max():.4f}]")
            
            return z_samples
            
        except Exception as e:
            logger.error(f"[CVAE Branching] 采样 z 失败: {e}")
            raise
    
    def _branch_single_path( 
        self,
        initial_output,
        prompt_ids: list,
        pure_question: str,
        sample_idx: int,
        k: int = 3,
        epoch: int = 0,  # 🆕 添加 epoch 参数
        gt: str = None,  # 🆕 添加标准答案
        data_source: str = None,  # 🆕 添加数据源
        full_prompt: str = None  # 🆕 添加完整 prompt（用于 DAPO 数据集）
    ) -> list:
        """
        对单条路径进行 k 次 迭代分叉，是迭代分叉 每次分叉一次的那种
        
        Args:
            initial_output: 初始路径（vLLM RequestOutput）
            prompt_ids: 原始 prompt 的 token ids
            pure_question: 纯问题（不包含 example COT）
            sample_idx: 样本索引（用于日志，0=a, 1=b, 2=c...） 是rolloutn级别的区分
            k: 分叉次数
            epoch: 当前 epoch（用于保存文件路径）
            gt: 标准答案（用于计算 reward）
            data_source: 数据源（用于选择 reward 计算函数）
            # 相当于 一次只传入一道题 的一次 rollout
        Returns:
            List[RequestOutput]: [原始路径, 分叉1, 分叉2, ..., 分叉k]
        """
        # 🔧 DEBUG: 打印接收到的 sample_idx
        # logger.info(f"\n{'='*80}")
        # logger.info(f"[分叉开始] 原始样本 {sample_idx}，将进行 {k} 次分叉")
        # logger.info(f"  接收到的 sample_idx: {sample_idx} (type: {type(sample_idx)})")
        # logger.info(f"  接收到的 epoch: {epoch}")
        # logger.info(f"{'='*80}")
        
        # 🆕 生成初始分支名称（a0, b0, c0...）
        base_letter = chr(ord('a') + sample_idx)
        initial_branch_name = f"{base_letter}0"
        
        # 存储所有路径
        all_paths = [initial_output]
        
        # 🆕 收集所有分支数据（用于保存到 JSONL 和可视化）
        all_branches = []  # 包含所有分支的完整信息
        branch_tree = {}  # 用于构建可视化分叉树 保存parent path：{branch_name: {"parent": ..., "children": [...], "token_idx": ...}}
        
        # 🆕 记录初始分支（a0, b0, c0...）
        initial_entropies = initial_output.outputs[0].entropies
        initial_avg_entropy = 0.0
        if initial_entropies and len(initial_entropies) > 0:
            if isinstance(initial_entropies, list):
                initial_avg_entropy = sum(initial_entropies) / len(initial_entropies)
            else:
                initial_avg_entropy = initial_entropies.mean().item()
        
        # 🔧 修复：初始分支的 content 应该是完整的生成内容（不包含 prompt）
        initial_response_ids = initial_output.outputs[0].token_ids
        initial_text = self.tokenizer.decode(initial_response_ids, skip_special_tokens=False)
        initial_tokens = len(initial_response_ids)
        
        # 当前活跃路径（用于下一次分叉）
        active_path = initial_output
        active_branch_name = initial_branch_name  # 🆕 跟踪活跃分支名称
        
        # 🔧 修复：追踪活跃路径的完整 content（累积的）
        # 必须在 initial_text 定义之后初始化
        active_path_full_content = initial_text  # 初始化为 a0 的完整内容
        
        # 🔧 新增：追踪活跃路径的完整 token_ids（避免重新编码导致的词表溢出）
        active_path_full_token_ids_list = list(initial_response_ids)  # 初始化为 a0 的 token_ids
        
        logger.debug(f"[初始分支 {initial_branch_name}] token数: {initial_tokens}, 平均熵: {initial_avg_entropy:.4f}")
        logger.debug(f"[初始分支 {initial_branch_name}] 内容: {initial_text[:100]}...")
        
        # 🆕 计算初始分支的 reward
        initial_reward = 0.0
        if gt is not None and data_source is not None:
            try:
                initial_reward = default_compute_score(
                    data_source=data_source,
                    solution_str=initial_text,
                    ground_truth=gt
                )
                # 🔧 确保 reward 是有效的数字类型
                if initial_reward is None or not isinstance(initial_reward, (int, float)):
                    logger.warning(f"[初始分支 {initial_branch_name}] reward 返回非数字类型: {type(initial_reward)}, 使用默认值 0.0")
                    initial_reward = 0.0
                else:
                    initial_reward = float(initial_reward)
                logger.debug(f"[初始分支 {initial_branch_name}] reward: {initial_reward}")
            except Exception as e:
                logger.warning(f"[初始分支 {initial_branch_name}] 计算 reward 失败: {e}")
                initial_reward = 0.0
        
        all_branches.append({
            "branch": initial_branch_name,
            "tokens": initial_tokens,
            "avg_entropy": float(initial_avg_entropy),
            "content": initial_text,  # 完整的初始生成内容
            "reward": initial_reward  # 🆕 添加 reward (已经是 float)
        })
        branch_tree[initial_branch_name] = {
            "parent": None,
            "children": [],
            "token_idx": None
        }
        
        # 迭代分叉 k 次
        for iteration in range(k):
            # logger.info(f"\n{'─'*80}")
            # logger.info(f"[第 {iteration + 1}/{k} 轮分叉] 样本 {sample_idx}")
            # logger.info(f"{'─'*80}")
            
            # 1. 准备分叉数据（找最高熵点 + 采样 z）
            branching_data = self._prepare_branching_data_for_single_path(
                output=active_path,
                prompt_ids=prompt_ids,
                pure_question=pure_question,
                sample_idx=sample_idx
            )
            
            # 2. 提取数据
            max_entropy_idx = branching_data['max_entropy_idx']
            max_entropy_value = branching_data['max_entropy_value']
            prefix_ids = branching_data['prefix_ids']
            prefix_text = branching_data['prefix_text']
            z_samples = branching_data['z_samples']
            
            # 3. 使用采样的 z（已经是 [1, 128]）
            z = z_samples  # [1, 128]
            
            # 4. 打印 zi 信息（简化输出）
            #logger.info(f"[CVAE 采样] zi 形状: {z.shape}, 均值: {z.mean().item():.4f}, 标准差: {z.std().item():.4f}, 范围: [{z.min().item():.4f}, {z.max().item():.4f}]")
            
            # 5. 获取分叉前的完整内容
            original_response_ids = active_path.outputs[0].token_ids
            original_text = self.tokenizer.decode(original_response_ids, skip_special_tokens=False)
            #logger.debug(f"  分叉前内容: \"{original_text[:100]}...\"")
            
            # 6. 从 prefix 继续生成（根据 branching_mode 决定是否使用 z）
            branching_mode = self.config.cvae_branching_mode
            logger.info(f"!!!branching_mode:{branching_mode}")
            
            if branching_mode == "random":
                # 模式1：不使用 z，完全随机生成
                logger.info(f"[分叉模式: random] 不使用 zi，完全随机生成")
                new_path = self._continue_from_prefix(
                    prompt_ids=prompt_ids,
                    prefix_ids=prefix_ids,
                    pure_question=pure_question
                )
            
            elif branching_mode == "input":
                # 模式2：INPUT Fusion - 使用 ForwardContext 在第一层前注入 z_proj_input
                logger.info(f"[分叉模式: input] 🔒 使用 INPUT Fusion (ForwardContext) 注入 zi（SOFTMAX 已禁用）")
                logger.info(f"  z 形状: {z.shape}")
                
                # 获取 hidden_dim
                try:
                    hidden_dim = self.inference_engine.llm_engine.model_config.hidden_size
                    logger.info(f"  hidden_dim: {hidden_dim}")
                except Exception as e:
                    hidden_dim = 4096  # 默认值
                    logger.warning(f"  无法获取 hidden_size，使用默认值: {hidden_dim}, 错误: {e}")
                
                # 创建 ZInjectionConfig（通过 CVAE Manager）
                z_injection_config = self.cvae_manager.set_z_injection_for_input_fusion(
                    z=z,
                    hidden_dim=hidden_dim
                )
                
                logger.info(f"  设置 z_injection_config 到 vLLM engine...")
                # 设置 z_injection_config 到 vLLM
                self.inference_engine.set_z_injection_config(z_injection_config)
                
                try:
                    logger.info(f"  开始生成（使用 ForwardContext）...")
                    new_path = self._continue_from_prefix(
                        prompt_ids=prompt_ids,
                        prefix_ids=prefix_ids,
                        pure_question=pure_question
                    )
                    logger.info(f"  ✅ 生成完成（INPUT Fusion）")
                finally:
                    # 清除 z_injection_config
                    logger.info(f"  清除 z_injection_config...")
                    self.inference_engine.set_z_injection_config(None)
                    logger.info(f"✅ [input] INPUT Fusion 已清除")
            
            elif branching_mode == "psa":
                # 模式3：PSA Fusion - 使用 ForwardContext 在指定层注入 z（I²B-LPO branching）
                logger.info(f"[分叉模式: psa] 🔒 使用 PSA Fusion (ForwardContext) 注入 zi（INPUT/SOFTMAX 已禁用）")
                logger.info(f"  z 形状: {z.shape}")
                
                # 获取模型配置
                try:
                    hidden_dim = self.inference_engine.llm_engine.model_config.hf_config.hidden_size
                    num_layers = self.inference_engine.llm_engine.model_config.hf_config.num_hidden_layers
                    logger.info(f"  hidden_dim: {hidden_dim}")
                    logger.info(f"  num_layers: {num_layers}")
                except Exception as e:
                    hidden_dim = 4096  # Qwen3-4B 默认值
                    num_layers = 28    # Qwen3-4B 默认层数
                    logger.warning(f"  无法获取模型配置，使用默认值: hidden_dim={hidden_dim}, num_layers={num_layers}, 错误: {e}")
                
                # 获取注入层配置
                injection_layers = self.config.cvae_injection_layers  # 默认 4
                logger.info(f"  injection_layers 配置: {injection_layers}")
                
                # 🔒 创建 ZInjectionConfig（通过 CVAE Manager，确保互斥性）
                z_injection_config = self.cvae_manager.set_z_injection_for_psa_fusion(
                    z=z,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    injection_layers=injection_layers
                )
                
                logger.info(f"  设置 z_injection_config 到 vLLM engine...")
                # 设置 z_injection_config 到 vLLM
                self.inference_engine.set_z_injection_config(z_injection_config)
                
                try:
                    logger.info(f"  开始生成（使用 ForwardContext）...")
                    new_path = self._continue_from_prefix(
                        prompt_ids=prompt_ids,
                        prefix_ids=prefix_ids,
                        pure_question=pure_question
                    )
                    logger.info(f"  ✅ 生成完成（PSA Fusion）")
                finally:
                    # 清除 z_injection_config
                    logger.info(f"  清除 z_injection_config...")
                    self.inference_engine.set_z_injection_config(None)
                    logger.info(f"✅ [psa] PSA Fusion 已清除")
            
            elif branching_mode == "softmax":
                # 模式4：SOFTMAX Fusion - 使用 ForwardContext 在 softmax 前修改 logits
                logger.info(f"[分叉模式: softmax] 🔒 使用 SOFTMAX Fusion (ForwardContext) 注入 zi（INPUT 已禁用）")
                logger.info(f"  z 形状: {z.shape}")
                
                # 获取 vocab_size
                try:
                    vocab_size = self.inference_engine.llm_engine.model_config.vocab_size
                    logger.info(f"  vocab_size: {vocab_size}")
                except Exception as e:
                    vocab_size = 151936  # Qwen3 默认词表大小
                    logger.warning(f"  无法获取 vocab_size，使用默认值: {vocab_size}, 错误: {e}")
                
                # 🔒 创建 ZInjectionConfig（通过 CVAE Manager，确保互斥性）
                z_injection_config = self.cvae_manager.set_z_injection_for_softmax_fusion(
                    z=z,
                    vocab_size=vocab_size
                )
                
                logger.info(f"  设置 z_injection_config 到 vLLM engine...")
                # 设置 z_injection_config 到 vLLM
                self.inference_engine.set_z_injection_config(z_injection_config)
                
                try:
                    logger.info(f"  开始生成（使用 ForwardContext）...")
                    new_path = self._continue_from_prefix(
                        prompt_ids=prompt_ids,
                        prefix_ids=prefix_ids,
                        pure_question=pure_question
                    )
                    logger.info(f"  ✅ 生成完成（SOFTMAX Fusion）")
                finally:
                    # 清除 z_injection_config
                    logger.info(f"  清除 z_injection_config...")
                    self.inference_engine.set_z_injection_config(None)
                    logger.info(f"✅ [softmax] SOFTMAX Fusion 已清除")
            
            else:
                logger.warning(f"未知的分叉模式: {branching_mode}，使用 random 模式")
                new_path = self._continue_from_prefix(
                    prompt_ids=prompt_ids,
                    prefix_ids=prefix_ids,
                    pure_question=pure_question
                )
            
            # 7. 获取分叉后的新生成内容（只包含从分叉点之后的部分）
            new_response_ids = new_path.outputs[0].token_ids
            new_text_suffix = self.tokenizer.decode(new_response_ids, skip_special_tokens=False)
            logger.debug(f"  分叉后新内容: \"{new_text_suffix[:100]}...\"")
            
            # 🔧 修复：构建完整的推理路径（使用累积的 token_ids，避免重新编码）
            # 问题：重新编码可能产生超出词表的 token_ids（特别是特殊 token）
            # 解决：直接使用原始的 token_ids，不要重新编码
            
            # 方法：使用累积的 token_ids（从 active_path_full_token_ids_list）
            # 注意：我们需要在外层维护一个累积的 token_ids 列表
            
            # 获取前缀（到分叉点）
            if max_entropy_idx + 1 <= len(active_path_full_token_ids_list):
                parent_prefix_ids = active_path_full_token_ids_list[:max_entropy_idx + 1]
                parent_prefix_text = self.tokenizer.decode(parent_prefix_ids, skip_special_tokens=False)
            else:
                # 如果分叉点超出范围，使用整个累积内容
                logger.warning(f"  分叉点 {max_entropy_idx} 超出累积内容长度 {len(active_path_full_token_ids_list)}，使用整个内容")
                parent_prefix_ids = active_path_full_token_ids_list
                parent_prefix_text = self.tokenizer.decode(parent_prefix_ids, skip_special_tokens=False)
            
            # 完整的推理路径 = 父分支前缀 + 新生成内容
            full_content = parent_prefix_text + new_text_suffix
            
            # 🔧 DEBUG: 打印详细信息
            logger.debug(f"  [DEBUG] 累积内容 token 数: {len(active_path_full_token_ids_list)}")
            logger.debug(f"  [DEBUG] 分叉点索引: {max_entropy_idx}")
            logger.debug(f"  [DEBUG] 前缀 token 数: {len(parent_prefix_ids)}")
            logger.debug(f"  [DEBUG] 新生成 token 数: {len(new_response_ids)}")
            logger.debug(f"  [DEBUG] 累积内容长度: {len(active_path_full_content)} 字符")
            logger.debug(f"  [DEBUG] 前缀文本长度: {len(parent_prefix_text)} 字符")
            logger.debug(f"  [DEBUG] 新内容文本长度: {len(new_text_suffix)} 字符")
            logger.debug(f"  [DEBUG] 完整路径文本长度: {len(full_content)} 字符")
            logger.debug(f"  完整推理路径: 前缀({len(parent_prefix_ids)} tokens) + 新内容({len(new_response_ids)} tokens) = {len(full_content)} 字符")
            
            # 8. 添加到路径列表
            all_paths.append(new_path)
            
            # 9. 计算平均熵
            original_entropies = active_path.outputs[0].entropies
            new_entropies = new_path.outputs[0].entropies
            
            if original_entropies is None or len(original_entropies) == 0:
                logger.warning(f"原路径没有熵值，无法对比")
                original_avg_entropy = 0.0
            else:
                if isinstance(original_entropies, list):
                    original_avg_entropy = sum(original_entropies) / len(original_entropies)
                else:
                    original_avg_entropy = original_entropies.mean().item()
            
            if new_entropies is None or len(new_entropies) == 0:
                logger.warning(f"新路径没有熵值，无法对比")
                new_avg_entropy = 0.0
            else:
                if isinstance(new_entropies, list):
                    new_avg_entropy = sum(new_entropies) / len(new_entropies)
                else:
                    new_avg_entropy = new_entropies.mean().item()
            
            # 10. 打印熵值对比（简化输出）
            logger.info(f"[熵值对比] 原路径: {original_avg_entropy:.4f}, 新路径: {new_avg_entropy:.4f}")
            
            # 11. 生成新分支名称（a1, a2, a3...）
            new_branch_name = f"{base_letter}{iteration + 1}"
            
            # 12. 🆕 保存分叉前的父分支名称（用于记录）
            parent_branch_name = active_branch_name
            
            # 13. 选择活跃路径（平均熵更高的）
            selected = "new" if new_avg_entropy > original_avg_entropy else "original"
            if new_avg_entropy > original_avg_entropy:
                active_path = new_path
                active_branch_name = new_branch_name  # 🆕 更新活跃分支名称
                active_path_full_content = full_content  # 🔧 更新累积的完整内容
                
                # 🔧 更新累积的 token_ids（前缀 + 新生成的 token_ids）
                active_path_full_token_ids_list = parent_prefix_ids + new_response_ids
                
                #logger.info(f"✅ 选择新路径作为活跃路径")
                logger.debug(f"  [DEBUG] 更新 active_path_full_content 长度: {len(active_path_full_content)} 字符")
                logger.debug(f"  [DEBUG] 更新 active_path_full_token_ids_list 长度: {len(active_path_full_token_ids_list)} tokens")
            else:
                # 保持原路径（active_path_full_content 和 active_path_full_token_ids_list 不变）
                #logger.info(f"⏸️  保持原路径作为活跃路径")
                logger.debug(f"  [DEBUG] 保持 active_path_full_content 长度: {len(active_path_full_content)} 字符")
                logger.debug(f"  [DEBUG] 保持 active_path_full_token_ids_list 长度: {len(active_path_full_token_ids_list)} tokens")
            
            # 14. 🆕 记录新分支数据（新格式）
            # 计算完整路径的 token 数量（前缀 + 新生成部分）
            full_tokens_count = len(parent_prefix_ids) + len(new_response_ids)
            
            # 🆕 计算新分支的 reward
            new_reward = 0.0
            if gt is not None and data_source is not None:
                try:
                    new_reward = default_compute_score(
                        data_source=data_source,
                        solution_str=full_content,
                        ground_truth=gt
                    )
                    # 🔧 确保 reward 是有效的数字类型
                    if new_reward is None or not isinstance(new_reward, (int, float)):
                        logger.warning(f"[新分支 {new_branch_name}] reward 返回非数字类型: {type(new_reward)}, 使用默认值 0.0")
                        new_reward = 0.0
                    else:
                        new_reward = float(new_reward)
                    logger.debug(f"[新分支 {new_branch_name}] reward: {new_reward}")
                except Exception as e:
                    logger.warning(f"[新分支 {new_branch_name}] 计算 reward 失败: {e}")
                    new_reward = 0.0
            
            new_branch_record = {
                "branch": new_branch_name,
                "parent_branch": parent_branch_name,  # 父分支是分叉前的活跃分支
                "branching_point_token_idx": max_entropy_idx,  # 相对于父分支的 token 索引
                "tokens": full_tokens_count,  # 🆕 完整路径的 token 数量
                "avg_entropy": float(new_avg_entropy),
                "content": full_content,  # 🆕 完整的推理路径（前缀 + 新生成内容）
                "reward": new_reward  # 🆕 添加 reward (已经是 float)
            }
            all_branches.append(new_branch_record)
            
            # 🆕 更新分叉树
            branch_tree[new_branch_name] = {
                "parent": parent_branch_name,
                "children": [],
                "token_idx": max_entropy_idx
            }
            if parent_branch_name not in branch_tree:
                branch_tree[parent_branch_name] = {"parent": None, "children": [], "token_idx": None}
            branch_tree[parent_branch_name]["children"].append(new_branch_name)
        
        # 14. 打印最终总结（简化）
        #logger.info(f"[分叉完成] 样本 {sample_idx}，总路径数: {len(all_paths)} (原始 + {k} 个分叉)")
        
        # 15. 🆕 生成分叉树可视化（Markdown）
        forking_vis_markdown = self._generate_forking_visualization(
            branch_tree=branch_tree,
            all_branches=all_branches
        )
        
        # 16. 🆕 保存分叉数据到 JSONL（新格式）
        import json
        from pathlib import Path
        
        # 获取 output_dir（与 entropy_output 使用相同的目录）
        base_output_dir = self.config.get("entropy_output", {}).get("output_dir", "./output")
        
        # 保存到 epoch_X/ 目录下（与 gsm8k.jsonl, math.jsonl 同级）
        epoch_dir = Path(base_output_dir) / f"epoch_{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        branching_file = epoch_dir / "branching.jsonl"
        
        # 🔧 修复：确保 sample_idx 正确传递
        #logger.info(f"[保存分叉数据] sample_idx={sample_idx}, 分支数={len(all_branches)}")
        
        # 构造完整记录（新格式）
        # 🆕 根据 data_source 决定使用 pure_question 还是 full_prompt
        # 对于 DAPO 数据集，使用 full_prompt（包含完整的模板）以便验证 prompt 结构
        question_to_save = pure_question
        if data_source is not None and 'dapo' in data_source.lower() and full_prompt is not None:
            question_to_save = full_prompt
            logger.debug(f"[DAPO数据集] 使用完整 prompt 而不是 pure_question（长度: {len(full_prompt)} 字符）")
        
        branching_record = {
            "sample_idx": int(sample_idx),  # 🔧 确保是 int 类型
            "pure_question": str(question_to_save),  # 🆕 对于 DAPO 使用完整 prompt
            "gt": str(gt) if gt is not None else "",  # 🆕 添加标准答案
            "data_source": str(data_source) if data_source is not None else "",  # 🆕 添加数据源
            "branches": all_branches,
            "forking_vis_markdown": forking_vis_markdown
        }
        
        # 🔧 DEBUG: 打印记录内容
        logger.debug(f"[保存分叉数据] branching_record keys: {list(branching_record.keys())}")
        logger.debug(f"[保存分叉数据] sample_idx type: {type(branching_record['sample_idx'])}, value: {branching_record['sample_idx']}")
        
        # 写入 JSONL
        try:
            with open(branching_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(branching_record, ensure_ascii=False) + "\n")
            #logger.info(f"✅ 分叉数据已保存到 {branching_file} (sample_idx={sample_idx})")
        except Exception as e:
            logger.error(f"❌ 保存分叉数据失败: {e}")
            import traceback
            traceback.print_exc()
        
        #logger.info(f"{'='*80}\n")
        
        return all_paths
    
    def _generate_forking_visualization(
        self,
        branch_tree: dict,
        all_branches: list
    ) -> str:
        """
        生成分叉树的可视化 Markdown 文本
        
        Args:
            branch_tree: 分叉树结构 {branch_name: {"parent": ..., "children": [...], "token_idx": ...}}
            all_branches: 所有分支的完整信息列表
        
        Returns:
            str: Markdown 格式的分叉树可视化
        """
        if not branch_tree or not all_branches:
            return ""
        
        # 创建分支名称到内容的映射
        branch_content_map = {b["branch"]: b["content"] for b in all_branches}
        branch_token_idx_map = {b["branch"]: b.get("branching_point_token_idx") for b in all_branches if b.get("branching_point_token_idx") is not None}
        
        # 找到根节点（没有父节点的分支）
        root_branches = [name for name, info in branch_tree.items() if info["parent"] is None]
        
        if not root_branches:
            return ""
        
        lines = []
        
        def build_tree_recursive(branch_name: str, prefix: str = "", is_last: bool = True, depth: int = 0):
            """递归构建树形结构"""
            if branch_name not in branch_tree:
                return
            
            # 获取分支信息
            branch_info = branch_tree[branch_name]
            content = branch_content_map.get(branch_name, "")
            
            # 截取内容（前50个字符）
            content_preview = content[:50] + "..." if len(content) > 50 else content
            
            # 构建当前分支的显示
            if depth == 0:
                # 根节点
                lines.append(f"{branch_name}: {content_preview}")
            else:
                # 子节点
                connector = "|_" if is_last else "|—"
                token_idx = branch_token_idx_map.get(branch_name)
                token_info = f" (token {token_idx})" if token_idx is not None else ""
                lines.append(f"{prefix}{connector} {branch_name}{token_info}: {content_preview}")
            
            # 处理子节点
            children = branch_info.get("children", [])
            if children:
                for i, child in enumerate(children):
                    is_last_child = (i == len(children) - 1)
                    # 根据是否是最后一个子节点选择不同的前缀
                    if is_last:
                        child_prefix = prefix + "    "
                    else:
                        child_prefix = prefix + "|   "
                    build_tree_recursive(child, child_prefix, is_last_child, depth + 1)
        
        # 从每个根节点开始构建树
        for i, root in enumerate(root_branches):
            if i > 0:
                lines.append("")  # 多个根节点之间空行
            build_tree_recursive(root)
        
        return "\n".join(lines)
    
    def _prepare_branching_data_for_single_path(
        self,
        output,
        prompt_ids: list,
        pure_question: str,
        sample_idx: int
    ) -> dict:
        """
        为单条路径准备分叉所需的数据
        
        Args:
            output: vLLM RequestOutput（包含 token_ids 和 entropies）
            prompt_ids: 原始 prompt 的 token ids
            pure_question: 纯问题（不包含 example COT）
            sample_idx: 样本索引（用于日志）
        
        Returns:
            dict: {
                "max_entropy_idx": int,  # 最高熵 token 的索引（在 response 中）
                "max_entropy_value": float,  # 最高熵值
                "prefix_ids": list,  # 从开始到最高熵 token 的 token ids
                "prefix_text": str,  # prefix 的文本
                "z_samples": torch.Tensor,  # 采样的 z 向量 [num_samples, latent_dim]
                "pure_question": str,  # 纯问题
                "prompt_ids": list,  # 原始 prompt ids
            }
        """
        logger.debug(f"[CVAE Branching] 为样本 {sample_idx} 准备分叉数据...")
        
        # 1. 提取 token_ids 和 entropies
        response_ids = output.outputs[0].token_ids
        entropies = output.outputs[0].entropies
        
        if entropies is None or len(entropies) == 0:
            raise RuntimeError(f"[CVAE Branching] 样本 {sample_idx} 没有熵值，无法分叉")
        
        # 2. 找到最高熵的 token（排除第一个 token）
        if len(entropies) <= 1:
            logger.warning(f"[CVAE Branching] 样本 {sample_idx} 只有 {len(entropies)} 个 token，无法排除第一个")
            max_entropy_idx = 0
        else:
            # 排除第一个 token，找最大熵
            entropies_tensor = torch.tensor(entropies[1:])  # 排除第一个
            max_entropy_idx = torch.argmax(entropies_tensor).item() + 1  # +1 因为排除了第一个
        
        max_entropy_value = entropies[max_entropy_idx]
        
        logger.debug(f"[CVAE Branching] 样本 {sample_idx} 最高熵点:")
        logger.debug(f"  位置: token {max_entropy_idx} / {len(response_ids)}")
        logger.debug(f"  熵值: {max_entropy_value:.4f}")
        
        # 3. 提取 prefix（从开始到最高熵 token，包含该 token）
        prefix_ids = response_ids[:max_entropy_idx + 1]
        prefix_text = self.tokenizer.decode(prefix_ids, skip_special_tokens=False)
        
        logger.debug(f"[CVAE Branching] Prefix 信息:")
        logger.debug(f"  长度: {len(prefix_ids)} tokens")
        logger.debug(f"  文本: {prefix_text[:100]}...")
        
        # 4. 采样 z（每次只采样 1 个，用于单次分叉）
        z_samples = self._sample_z_from_question_and_prefix(
            pure_question=pure_question,
            prefix_text=prefix_text,
            num_samples=1  # 每次只采样 1 个 z
        )
        
        # 5. 返回所有数据
        branching_data = {
            "max_entropy_idx": max_entropy_idx,
            "max_entropy_value": max_entropy_value,
            "prefix_ids": prefix_ids,
            "prefix_text": prefix_text,
            "z_samples": z_samples,
            "pure_question": pure_question,
            "prompt_ids": prompt_ids,
            "sample_idx": sample_idx,
        }
        
        logger.debug(f"[CVAE Branching] ✅ 样本 {sample_idx} 分叉数据准备完成")
        
        return branching_data
    
    '''
    # 假设：batch_size=2, rollout_n=3, k=2

    initial_outputs = [
        RequestOutput[0] (问题1)
            └─ outputs: [CompletionOutput[0], CompletionOutput[1], CompletionOutput[2]]  # rollout_n=3
        RequestOutput[1] (问题2)
            └─ outputs: [CompletionOutput[0], CompletionOutput[1], CompletionOutput[2]]  # rollout_n=3
    ]

    # 第1层循环：拆开 batch_size
    for prompt_idx, request_output in enumerate(initial_outputs):
        # prompt_idx=0: request_output = 问题1的所有回答（3个）
        # prompt_idx=1: request_output = 问题2的所有回答（3个）
        
        # 第2层循环：拆开 rollout_n
        for sample_idx_in_prompt, completion_output in enumerate(request_output.outputs):
            # 问题1:
            #   sample_idx_in_prompt=0: completion_output = 问题1的第1次回答
            #   sample_idx_in_prompt=1: completion_output = 问题1的第2次回答
            #   sample_idx_in_prompt=2: completion_output = 问题1的第3次回答
            
            # 对这1次回答进行分叉
            branched_paths = _branch_single_path(completion_output)
            # 输入：1次回答
            # 输出：1+k=3 条路径
    '''

    def _perform_branching_for_all_paths(
        self,
        initial_outputs: list,
        idx: torch.Tensor,
        non_tensor_batch: dict,
        prompts  # 🆕 添加 prompts 参数以获取 meta_info
    ) -> list:
        """
        对所有初始路径进行分叉
        
        Args:
            initial_outputs: vLLM RequestOutput列表，长度=batch_size
                每个 RequestOutput 包含 rollout_n） 个 CompletionOutput（rollout_n）
            idx: input_ids tensor [batch_size, prompt_length]
            non_tensor_batch: 包含 pure_question 等字段
            prompts: DataProto，包含 meta_info（用于获取 epoch 等信息）
        
        Returns:
            List[RequestOutput]: batch_size × rollout_n × (1 + k) 条路径
        """
        # 🔧 修复：计算实际的初始路径数（考虑 rollout_n > 1 的情况）
        num_prompts = len(initial_outputs)  # batch_size
        rollout_n = len(initial_outputs[0].outputs) if num_prompts > 0 else 1  # 每个prompt的采样数
        total_initial_paths = num_prompts * rollout_n
        
        logger.info(f"\n{'#'*80}")
        logger.info(f"[批量分叉开始]")
        logger.info(f"  batch_size (prompts数量): {num_prompts}")
        logger.info(f"  rollout_n (每个prompt的采样数): {rollout_n}")
        logger.info(f"  初始路径总数: {total_initial_paths} ({num_prompts} × {rollout_n})")
        logger.info(f"  每条路径分叉次数(k): {self.config.cvae_num_branches_per_path}")
        logger.info(f"  预期最终路径数: {total_initial_paths} × (1 + {self.config.cvae_num_branches_per_path}) = {total_initial_paths * (1 + self.config.cvae_num_branches_per_path)}")
        logger.info(f"{'#'*80}\n")
        
        all_branched_paths = []
        k = self.config.cvae_num_branches_per_path
        
        # 🆕 获取当前 epoch（从 prompts.meta_info）
        epoch = prompts.meta_info.get("epoch", 0)
        
        # 获取 pure_question
        pure_questions = non_tensor_batch.get("pure_question", None)
        if pure_questions is None:
            # Fallback: 使用 question 字段
            pure_questions = non_tensor_batch.get("question", None)
            if pure_questions is not None:
                logger.warning(f"[批量分叉] pure_question 不存在，使用 question 字段")
            else:
                raise RuntimeError("[批量分叉] 无法获取 pure_question 或 question 字段")
        
        # 🆕 获取完整的 prompt（用于 DAPO 数据集的 prompt 验证）
        full_prompts = non_tensor_batch.get("full_prompts", None)
        
        # 🆕 获取 gt（标准答案）
        gts = non_tensor_batch.get("data_answer", None)
        if gts is None:
            gts = non_tensor_batch.get("answer", None)
            if gts is not None:
                logger.warning(f"[批量分叉] data_answer 不存在，使用 answer 字段")
            else:
                logger.warning(f"[批量分叉] 无法获取 data_answer 或 answer 字段，reward 将无法计算")
                gts = [None] * num_prompts
        
        # 🆕 获取 data_source
        data_sources = non_tensor_batch.get("data_source", None)
        if data_sources is None:
            data_sources = non_tensor_batch.get("dataset_name", None)
            if data_sources is not None:
                logger.warning(f"[批量分叉] data_source 不存在，使用 dataset_name 字段")
            else:
                logger.warning(f"[批量分叉] 无法获取 data_source 或 dataset_name 字段，reward 将无法计算")
                data_sources = [None] * num_prompts
        
        # 🔧 标准化 data_source（确保与 verl 的 reward function 匹配）
        def normalize_data_source(ds):
            if ds is None:
                return None
            ds_lower = str(ds).lower()
            if 'gsm8k' in ds_lower:
                return 'openai/gsm8k'
            elif 'math' in ds_lower and 'dapo' not in ds_lower:
                return 'lighteval/MATH'
            elif 'math_dapo' in ds_lower or 'dapo' in ds_lower:
                return 'math_dapo'
            else:
                return ds
        
        # 应用标准化
        if isinstance(data_sources, list):
            data_sources = [normalize_data_source(ds) for ds in data_sources]
        else:
            # 🔧 修复：单个值标准化后，扩展成列表（避免字符串索引错误）
            normalized = normalize_data_source(data_sources)
            data_sources = [normalized] * num_prompts
        
        # 🔧 修复：需要遍历所有 prompts 和每个 prompt 的所有采样
        global_sample_idx = 0  # 全局的样本索引（用于命名：a0, b0, c0...）
        
        for prompt_idx, request_output in enumerate(initial_outputs):
            # prompt_idx 就是 batch_idx,request_output 就是一个问题的rolloutn次回答的合集
            prompt_ids = _pre_process_inputs(self.pad_token_id, idx[prompt_idx])
            
            # 获取当前 prompt 的 prompt_ids 和 pure_question
            pure_question = pure_questions[prompt_idx]
            if isinstance(pure_question, np.ndarray):
                pure_question = pure_question.item()
            pure_question = str(pure_question)
            
            # 🆕 获取当前 prompt 的完整 prompt（如果存在，用于 DAPO 数据集）
            full_prompt = None
            if full_prompts is not None:
                full_prompt = full_prompts[prompt_idx]
                if isinstance(full_prompt, np.ndarray):
                    full_prompt = full_prompt.item()
                full_prompt = str(full_prompt) if full_prompt is not None else None
            
            # 🆕 获取当前 prompt 的 gt 和 data_source
            gt = gts[prompt_idx]
            if isinstance(gt, np.ndarray):
                gt = gt.item()
            gt = str(gt) if gt is not None else None
            
            data_source = data_sources[prompt_idx]
            if isinstance(data_source, np.ndarray):
                data_source = data_source.item()
            data_source = str(data_source) if data_source is not None else None
            
            #logger.info(f"\n[处理 i/batchsize Prompt {prompt_idx+1}/{num_prompts}]")
            #logger.info(f"  问题: {pure_question[:100]}...")
            logger.info(f"  该prompt有 rolloutn： {len(request_output.outputs)} 个采样结果")
            
            # 🔧 关键修复：遍历该 prompt 的所有采样结果（n 个）
            for sample_idx_in_prompt, completion_output in enumerate(request_output.outputs):
                # sample_idx_in_prompt 这个就是 i/rolloutn ，completion_output就是 一个ques 一次rollout的结果啊
                #logger.info(f"\n  [处理采样 {sample_idx_in_prompt+1}/{len(request_output.outputs)}]")
                #logger.info(f"    全局 sample_idx: {global_sample_idx}")
                
                # 🔧 创建一个临时的 RequestOutput，只包含当前这一个 CompletionOutput
                temp_request_output = type(request_output)(
                    request_id=request_output.request_id,
                    prompt=request_output.prompt,
                    prompt_token_ids=request_output.prompt_token_ids,
                    prompt_logprobs=request_output.prompt_logprobs,
                    outputs=[completion_output],  # 只包含当前这一个 CompletionOutput
                    finished=request_output.finished
                )
                
                # 对单条路径进行分叉
                branched_paths = self._branch_single_path(
                    initial_output=temp_request_output,  # 传入临时的 RequestOutput
                    prompt_ids=prompt_ids,
                    pure_question=pure_question,
                    sample_idx=global_sample_idx,  # 使用全局索引
                    k=k,
                    epoch=epoch,
                    gt=gt,  # 🆕 传递标准答案
                    data_source=data_source,  # 🆕 传递数据源
                    full_prompt=full_prompt  # 🆕 传递完整 prompt（用于 DAPO 数据集）
                )
                
                # 添加到总列表
                all_branched_paths.extend(branched_paths)
                
                logger.info(f"    [采样 {sample_idx_in_prompt+1} 完成] 生成了 {len(branched_paths)} 条路径")
                
                # 递增全局索引
                global_sample_idx += 1
            
            logger.info(f"[Prompt {prompt_idx+1} 完成] 总共生成了 {len(request_output.outputs) * (1 + k)} 条路径")
        
        # 打印最终总结
        logger.info(f"\n{'#'*80}")
        logger.info(f"[批量分叉完成]")
        logger.info(f"  初始路径数: {total_initial_paths}")
        logger.info(f"  最终路径数: {len(all_branched_paths)}")
        logger.info(f"  每条路径分叉: {k} 次")
        logger.info(f"{'#'*80}\n")
        
        return all_branched_paths

        '''
        _branch_single_path() 只处理1条路径的分叉。
        输入：1条路径
        输出：1+k 条路径
        这样设计的好处：
        函数职责单一
        逻辑简单清晰
        每条路径独立分叉
        示例：
    batch_size=1, rollout_n=2, k=3
    问题1:  采样1 (a0) → _branch_single_path() → a0, a1, a2, a3 (4条)  
           采样2 (b0) → _branch_single_path() → b0, b1, b2, b3 (4条)
           总共：8条路径
        '''
    
    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences for a batch of prompts.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data"), strict=True
            ):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [
                {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        for input_data in vllm_inputs:
            # Ensure token IDs are lists or numpy arrays
            if not isinstance(input_data["prompt_token_ids"], list | np.ndarray):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

            input_data["prompt_token_ids"] = list(input_data["prompt_token_ids"])

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [
                    LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")
                ] * batch_size

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            # 🔍 DEBUG: 打印 a0 路径的停止条件（用于对比）
            # logger.info(f"\n[generate_sequences] 🛑 a0 原始路径的停止条件:")
            # logger.info(f"  max_tokens: {self.sampling_params.max_tokens}")
            # logger.info(f"  ignore_eos: {self.sampling_params.ignore_eos}")
            # logger.info(f"  stop: {self.sampling_params.stop}")
            # logger.info(f"  stop_token_ids: {self.sampling_params.stop_token_ids}")
            if hasattr(self.sampling_params, '_all_stop_token_ids'):
                logger.info(f"  _all_stop_token_ids: {self.sampling_params._all_stop_token_ids}")
            else:
                logger.warning(f"  _all_stop_token_ids: 不存在！")
            logger.info(f"  tokenizer.eos_token_id: {self.tokenizer.eos_token_id}\n")
            
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=False,
            )

            # 🆕 CVAE 分叉逻辑
            if self.config.enable_cvae_branching and self.cvae_manager is not None:
                logger.info(f"\n{'🌳'*40}")
                logger.info(f"[CVAE 分叉] 启用，开始对所有路径进行分叉")
                logger.info(f"{'🌳'*40}")
                
                # 对所有初始路径进行分叉
                outputs = self._perform_branching_for_all_paths(
                    initial_outputs=outputs,
                    idx=idx,
                    non_tensor_batch=non_tensor_batch,
                    prompts=prompts  # 🆕 传递 prompts 以获取 epoch
                )
                
                # 更新 batch_size（因为路径数增加了）
                original_batch_size = batch_size
                batch_size = len(outputs)
                
                # 🔧 修复：计算扩展因子（考虑 rollout_n > 1 的情况）
                k = self.config.cvae_num_branches_per_path
                rollout_n = kwargs.get('n', 1)  # 每个prompt的采样数（rollout_n）
                
                # 扩展因子 = rollout_n × (1 + k)
                # 例如：rollout_n=2, k=3 时，每个prompt生成 2×4=8 条路径
                expansion_factor = rollout_n * (1 + k)
                
                logger.info(f"\n[CVAE 分叉] 计算扩展因子:")
                logger.info(f"  rollout_n (每个prompt的采样数): {rollout_n}")
                logger.info(f"  k (每条路径的分叉次数): {k}")
                logger.info(f"  expansion_factor = rollout_n × (1 + k) = {rollout_n} × {1 + k} = {expansion_factor}")
                
                # 扩展 idx: [original_batch_size, prompt_length] -> [batch_size, prompt_length]
                # 每个原始 prompt 需要重复 expansion_factor 次
                idx = idx.repeat_interleave(expansion_factor, dim=0)
                
                # 扩展 attention_mask 和 position_ids
                attention_mask = attention_mask.repeat_interleave(expansion_factor, dim=0)
                position_ids = position_ids.repeat_interleave(expansion_factor, dim=0)
                
                # 扩展 non_tensor_batch 中的所有字段
                for key in list(non_tensor_batch.keys()):
                    if isinstance(non_tensor_batch[key], (list, np.ndarray)):
                        # 将每个元素重复 expansion_factor 次
                        original_data = non_tensor_batch[key]
                        expanded_data = []
                        for item in original_data:
                            expanded_data.extend([item] * expansion_factor)
                        non_tensor_batch[key] = np.array(expanded_data, dtype=object)
                
                logger.info(f"\n[CVAE 分叉] 完成")
                logger.info(f"  原始 batch_size: {original_batch_size}")
                logger.info(f"  分叉后 batch_size: {batch_size}")
                logger.info(f"  扩展倍数: {expansion_factor}x")
                logger.info(f"  idx shape: {idx.shape}")
                
                # 🆕 在 meta_info 中标记分叉信息（用于 ray_trainer 处理）
                prompts.meta_info["cvae_branching_enabled"] = True
                prompts.meta_info["cvae_expansion_factor"] = expansion_factor
                prompts.meta_info["cvae_original_batch_size"] = original_batch_size
            
            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            rollout_log_probs = []
            rollout_entropies = []  # 新增：存储熵值
            
            # 🆕 DEBUG: 打印 outputs 的完整结构
            if self.compute_entropy and len(outputs) > 0:
                logger.debug(f"[VERL] 检查 outputs 结构:")
                #logger.debug(f"  type(outputs): {type(outputs)}")
                #logger.debug(f"  len(outputs): {len(outputs)}")
                if hasattr(outputs[0], '__dict__'):
                    logger.debug(f"  outputs[0].__dict__.keys(): {list(outputs[0].__dict__.keys())}")
                #logger.debug(f"  type(outputs[0]): {type(outputs[0])}")
                available_attrs = [attr for attr in dir(outputs[0]) if not attr.startswith('_')]
                logger.debug(f"  outputs[0] 可用属性: {available_attrs}")
                
                if hasattr(outputs[0], 'outputs'):
                    logger.debug(f"  outputs[0].outputs:")
                    logger.debug(f"    len: {len(outputs[0].outputs)}")
                    logger.debug(f"    type(outputs[0].outputs[0]): {type(outputs[0].outputs[0])}")
                    output0_attrs = [attr for attr in dir(outputs[0].outputs[0]) if not attr.startswith('_')]
                    logger.debug(f"    outputs[0].outputs[0] 可用属性: {output0_attrs}")
                    
                    # 检查所有可能包含熵值的属性
                    logger.debug(f"  🔍 查找熵值相关属性:")
                    for attr in output0_attrs:
                        if 'entrop' in attr.lower():
                            val = getattr(outputs[0].outputs[0], attr)
                            logger.debug(f"    ✅ outputs[0].outputs[0].{attr} = {val}")
                    
                    # 检查 entropies 属性
                    if hasattr(outputs[0].outputs[0], 'entropies'):
                        entropies_val = outputs[0].outputs[0].entropies
                        #logger.debug(f"  📊 entropies 详情:")
                        #logger.debug(f"    type: {type(entropies_val)}")
                        #logger.debug(f"    value: {entropies_val}")
                        #logger.debug(f"    is None: {entropies_val is None}")
                        bool_val = bool(entropies_val) if entropies_val is not None else "N/A"
                        #logger.debug(f"    bool(value): {bool_val}")
            
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
            
            # 新增：如果启用了 EntropyOutputWriter，收集数据并保存
            if self.entropy_writer is not None and rollout_entropies:
                # 获取 epoch 和 step 信息（从 meta_info 传递）
                epoch = prompts.meta_info.get("epoch", 0)
                step = prompts.meta_info.get("step", 0)
                
                
                # 🔧 修复：只在保存间隔时收集数据
                should_collect = (step > 0 and step % self.entropy_writer.config.save_interval == 0)
                if not should_collect:
                    #logger.debug(f"[vLLMRollout] 跳过 step {step}（下一次收集在 step {((step // self.entropy_writer.config.save_interval) + 1) * self.entropy_writer.config.save_interval}）")
                    # 跳过数据收集，直接返回
                    pass  # 继续执行后续代码，但不收集数据
                # else:
                #     logger.info(f"[vLLMRollout] 📊 收集 step {step} 的熵数据")
                 # 🔍 DEBUG: 打印 non_tensor_batch 的 keys
                # logger.debug(f"[vLLMRollout] non_tensor_batch keys: {list(non_tensor_batch.keys())}")
                # logger.debug(f"[vLLMRollout] prompts.meta_info keys: {list(prompts.meta_info.keys())}")
                
                
                # 获取 data_source（优先使用 data_source，fallback 到 dataset_name）
                # 🔧 修复：兼容 data_source 和 dataset_name 两种字段名
                data_sources = non_tensor_batch.get("data_source", None)
                if data_sources is None:
                    # 尝试使用 dataset_name（你的 dataset 使用的字段名）
                    data_sources = non_tensor_batch.get("dataset_name", None)
                    if data_sources is not None:
                        logger.debug(f"[vLLMRollout] 使用 dataset_name 作为 data_source")
                
                if data_sources is None:
                    # 如果都没有，尝试从 meta_info 获取
                    data_sources = [prompts.meta_info.get("data_source", "unknown")] * batch_size
                    logger.warning(f"[vLLMRollout] data_source 和 dataset_name 都不存在，使用默认值 'unknown'")
                elif not isinstance(data_sources, (list, np.ndarray)):
                    data_sources = [data_sources] * batch_size
                    
                
                 # 🆕 获取 pure_question（CVAE 用）
                pure_questions = non_tensor_batch.get("pure_question", None)
                if pure_questions is None:
                    # Fallback: 使用 question 字段
                    pure_questions = non_tensor_batch.get("question", None)
                    if pure_questions is not None:
                        logger.info(f"[vLLMRollout] pure_question 不存在，使用 question 字段")
                
                
                # 遍历所有样本
                for i in range(len(response)):
                    try: 
                        # 解码 prompt
                        prompt_ids = _pre_process_inputs(self.pad_token_id, idx[i])
                        prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
                        # 🔧 清理 prompt 中的冲突指令（在最终 prompt 拼接完成后）
                        conflicting_instructions = [
                            "Please reason step by step, and put your final answer within \\boxed{}.",
                            "Please reason step by step, and put your final answer within $\\boxed{}$.",
                            "Please reason step-by-step, and put your final answer within \\boxed{}.",
                            "Please reason step by step and put your final answer within \\boxed{}.",
                            "Put your final answer within \\boxed{}.",
                            "put your final answer within \\boxed{}."
                        ]
                        
                        for instruction in conflicting_instructions:
                            if instruction in prompt_text:
                                prompt_text = prompt_text.replace(instruction, "").strip()
                                logger.debug(f"[vLLMRollout] 从最终 prompt 中删除冲突指令: {instruction[:50]}...")
                        
                        # 解码 response
                        response_text = self.tokenizer.decode(response[i], skip_special_tokens=False)
                        
                        # 获取 data_source
                        data_source = data_sources[i] if i < len(data_sources) else "unknown"
                        if isinstance(data_source, np.ndarray):
                            data_source = data_source.item()
                        
                        # 获取 entropy_list
                        entropy_list = rollout_entropies[i]
                        if isinstance(entropy_list, torch.Tensor):
                            entropy_list = entropy_list.cpu().tolist()
                        
                        # 构造样本数据
                        sample_data = {
                            "epoch": epoch,
                            "step": step,
                            "data_source": str(data_source),
                            "prompt": prompt_text,
                            "response": response_text,
                            "entropy_list": entropy_list,
                        }
                        
                        # 添加到 writer
                        self.entropy_writer.add_sample(sample_data)
                        
                    except Exception as e:
                        logger.error(f"[vLLMRollout] Error collecting entropy data for sample {i}: {e}", exc_info=True)
                
                # 🔧 修复：只在 save_interval 的倍数时才 flush
                # 注意：这里不需要检查，因为我们在每个 step 都收集数据
                # flush 操作应该由外部（ray_trainer）在合适的时机调用
                # 或者在这里检查是否是保存间隔
                if step > 0 and step % self.entropy_writer.config.save_interval == 0:
                    logger.info(f"[vLLMRollout] 🎯 Flushing entropy data at step {step} (interval={self.entropy_writer.config.save_interval})")
                    self.entropy_writer.flush()
                else:
                    logger.debug(f"[vLLMRollout] Buffering entropy data at step {step} (next flush at {((step // self.entropy_writer.config.save_interval) + 1) * self.entropy_writer.config.save_interval})")
            
            
            # 简化的日志输出（仅在 DEBUG 级别显示详细信息）
            if self.compute_entropy and rollout_entropies:
                logger.info(f"[VLLM spmd] Collected {len(rollout_entropies)} samples")
                
                # 只在 DEBUG 级别打印详细的 token-by-token 信息
                if logger.isEnabledFor(logging.DEBUG) and len(rollout_entropies) > 0:
                    logger.debug(f"[Sample 0] Generated {len(response[0])} tokens and {len(rollout_entropies[0])} entropys")
                    try:
                        generated_text = self.tokenizer.decode(response[0], skip_special_tokens=False)
                        #logger.debug(f"[Sample 0] Generated text: {generated_text[:200]}...")
                        
                        '''打印 Token-level-entropy'''
                        # Token-by-token breakdown (first 10 tokens)
                        #logger.debug(f"[Sample 0] First 10 tokens:")
                        for i in range(min(10, len(response[0]))):
                            token_id = response[0][i]
                            token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
                            entropy_val = rollout_entropies[0][i] if i < len(rollout_entropies[0]) else 0.0
                            #logger.debug(f"  {i}: {repr(token_text)} (entropy={entropy_val:.4f})")
                    except Exception as e:
                        logger.error(f"[Rollout Entropy] Error in debug output: {e}")
            elif self.compute_entropy:
                logger.warning(f"[Rollout Entropy] No entropies collected! vLLM may not be modified correctly.")

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(
                idx.device
            )
            if self.config.calculate_log_probs:
                rollout_log_probs = pad_2d_list_to_length(
                    rollout_log_probs, -1, max_length=self.config.response_length
                ).to(idx.device)
                rollout_log_probs = rollout_log_probs.to(torch.float32)
            
            # 新增：处理熵值
            if len(rollout_entropies) > 0:  # 🔧 修复：使用 len() 而不是直接判断 tensor
                rollout_entropies = pad_2d_list_to_length(
                    rollout_entropies, 0.0, max_length=self.config.response_length
                ).to(idx.device)
                rollout_entropies = rollout_entropies.to(torch.float32)
            else:
                rollout_entropies = None  # 🔧 如果没有熵值，设置为 None

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if self.config.calculate_log_probs:
            # we will recompute old log prob with actor
            batch["rollout_log_probs"] = rollout_log_probs
        
        # 新增：如果计算了熵值，添加到 batch 中
        if rollout_entropies is not None:  # 🔧 修复：检查是否为 None
            batch["rollout_entropies"] = rollout_entropies

        # 🆕 保留 meta_info（包含 CVAE 分叉信息）
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tags: weights or kv_cache.
        """
        if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
            self.inference_engine.wake_up(tags=tags)
        else:
            self.inference_engine.wake_up()

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        self.inference_engine.reset_prefix_cache()
        self.inference_engine.sleep(level=VLLM_SLEEP_LEVEL)

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """Update the weights of the rollout model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        peft_config, base_sync_done = kwargs.get("peft_config", None), kwargs.get("base_sync_done", False)
        if peft_config and base_sync_done:
            lora_int_id = int(time.time_ns() % 0x7FFFFFFF)
            lora_reqest = TensorLoRARequest(
                lora_name=f"{lora_int_id}",
                lora_int_id=lora_int_id,
                lora_path="simon_lora_path",
                peft_config=asdict(peft_config),
                lora_tensors=weights,
            )
            self.inference_engine.llm_engine.add_lora(lora_reqest)
            logger.info(f"vLLM load weights, loaded_params: {len(weights)}")
        else:
            from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

            model = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
            patch_vllm_moe_model_weight_loader(model)
            model.load_weights(weights)


# https://github.com/vllm-project/vllm/issues/13175
def _monkey_patch_compute_logits(model, vocab_size: int):
    original_compute_logits = model.compute_logits

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        logits = original_compute_logits(hidden_states, sampling_metadata)
        logits[..., vocab_size:] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)


class vLLMAsyncRollout(BaseRollout):
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase, which is engine in single worker process."""

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)

        self.tokenizer = model_config.tokenizer
        self.inference_engine: WorkerWrapperBase = None
        self.address = self._init_zeromq()

    def _init_zeromq(self) -> str:
        tensor_parallel_size = self.config.tensor_model_parallel_size

        # single node: ipc, multi nodes: tcp
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        socket_type = "ipc" if tensor_parallel_size <= local_world_size else "tcp"

        # File lock to prevent multiple workers listen to same port
        with FileLock(f"/tmp/verl_vllm_zmq_{getpass.getuser()}.lock"):
            if socket_type == "ipc":
                pid = os.getpid()
                address = f"ipc:///tmp/verl_vllm_zmq_{pid}_{getpass.getuser()}.ipc"
            else:
                ip, port = self._get_free_port()
                address = f"tcp://{ip}:{port}"
            context = zmq.asyncio.Context()
            self.socket = context.socket(zmq.REP)
            self.socket.bind(address)

        loop = asyncio.get_running_loop()
        self.zmq_loop_task = loop.create_task(self._loop_forever())

        return address

    def _get_free_port(self):
        ip = ray.util.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
        return ip, port

    async def _loop_forever(self):
        while True:
            try:
                message = await self.socket.recv()
                method, args, kwargs = pickle.loads(message)
                result = await self._execute_method(method, *args, **kwargs)
                await self.socket.send(pickle.dumps(result))
            except Exception as e:
                logger.exception(f"vLLMAsyncRollout _loop_forever error: {e}")
                os._exit(-1)

    def _init_worker(self, all_kwargs: list[dict[str, Any]]):
        """Initialize worker engine."""
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        all_kwargs[0]["local_rank"] = 0 if not ray_noset_visible_devices() else int(os.environ.get("RAY_LOCAL_RANK", 0))
        self.vllm_config = all_kwargs[0]["vllm_config"]
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def _load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)
        _monkey_patch_compute_logits(self.inference_engine.worker.model_runner.model, len(self.tokenizer))

    async def _execute_method(self, method: str | bytes, *args, **kwargs):
        if method == "init_worker":
            return self._init_worker(*args, **kwargs)
        elif method == "load_model":
            return self._load_model(*args, **kwargs)
        elif method == "sleep" or method == "wake_up":
            raise ValueError("wake_up and sleep should not be called through ZeroMQ")
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tags: weights or kv_cache.
        """
        self.inference_engine.wake_up(tags=tags)

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        self.inference_engine.sleep(level=VLLM_SLEEP_LEVEL)

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """Update the weights of the rollout model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

        model = self.inference_engine.worker.model_runner.model
        patch_vllm_moe_model_weight_loader(model)
        model.load_weights(weights)

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Batch generate sequences in sync mode."""
        raise NotImplementedError

    # ==================== server mode public methods ====================

    def get_zeromq_address(self):
        return self.address
