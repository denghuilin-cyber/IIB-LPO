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
            #logger.info(f"[VERL DEBUG] 🎯 强制设置 compute_entropy=True")
        
        kwargs["n"] = 1  # already repeat in ray_trainer
        #logger.debug(f"[VERL DEBUG] kwargs: {kwargs}")
        #logger.debug(f"[VERL DEBUG] compute_entropy in kwargs: {'compute_entropy' in kwargs}")
        self.sampling_params = SamplingParams(**kwargs)
        #logger.info(f"[VERL DEBUG] self.sampling_params.compute_entropy = {self.sampling_params.compute_entropy}")

        self.pad_token_id = tokenizer.pad_token_id

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
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=False,
            )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            rollout_log_probs = []
            rollout_entropies = []  # 新增：存储熵值
            
            # 🆕 DEBUG: 打印 outputs 的完整结构
            # if self.compute_entropy and len(outputs) > 0:
                #logger.debug(f"\n{'='*80}")
                # logger.debug(f"[VERL DEBUG] 检查 outputs 结构:")
                # logger.debug(f"{'='*80}")
                # logger.debug(f"  type(outputs): {type(outputs)}")
                # logger.debug(f"  len(outputs): {len(outputs)}")
                # if hasattr(outputs[0], '__dict__'):
                #     logger.debug(f"  outputs[0].__dict__.keys(): {list(outputs[0].__dict__.keys())}")
                # logger.debug(f"  type(outputs[0]): {type(outputs[0])}")
                # available_attrs = [attr for attr in dir(outputs[0]) if not attr.startswith('_')]
                #logger.debug(f"  outputs[0] 可用属性: {available_attrs}")
                
                # if hasattr(outputs[0], 'outputs'):
                #     logger.debug(f"\n  outputs[0].outputs:")
                #     logger.debug(f"    len: {len(outputs[0].outputs)}")
                #     logger.debug(f"    type(outputs[0].outputs[0]): {type(outputs[0].outputs[0])}")
                #     output0_attrs = [attr for attr in dir(outputs[0].outputs[0]) if not attr.startswith('_')]
                #     logger.debug(f"    outputs[0].outputs[0] 可用属性: {output0_attrs}")
                    
                #     # 检查所有可能包含熵值的属性
                #     logger.debug(f"\n  🔍 查找熵值相关属性:")
                #     for attr in output0_attrs:
                #         if 'entrop' in attr.lower():
                #             val = getattr(outputs[0].outputs[0], attr)
                #             logger.debug(f"    ✅ outputs[0].outputs[0].{attr} = {val}")
                    
                    # 检查 entropies 属性
                #     if hasattr(outputs[0].outputs[0], 'entropies'):
                #         entropies_val = outputs[0].outputs[0].entropies
                #         logger.debug(f"\n  📊 entropies 详情:")
                #         logger.debug(f"    type: {type(entropies_val)}")
                #         logger.debug(f"    value: {entropies_val}")
                #         logger.debug(f"    is None: {entropies_val is None}")
                #         logger.debug(f"    bool(value): {bool(entropies_val) if entropies_val is not None else 'N/A'}")
                # logger.debug(f"{'='*80}\n")
            
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
            
            '''逐行代码解释：
            ## 📖 逐行详细解释

            1. for output in outputs: # 这个是barchsize维度的拆开

            **解释：**
            - 遍历 `outputs` 列表，这是 vLLM 返回的所有生成结果
            - `outputs` 的结构：`List[RequestOutput]`
            - 每个 `output` 是一个 `RequestOutput` 对象，对应一个输入 prompt 的生成结果

            **示例：**
            ```python
            # 如果 batch_size = 8，那么 outputs 有 8 个元素
            outputs = [
                RequestOutput(...),  # prompt 0 的生成结果
                RequestOutput(...),  # prompt 1 的生成结果
                ...
                RequestOutput(...),  # prompt 7 的生成结果
            ]
            ```

            2. for sample_id in range(len(output.outputs)): # 这个是一个样本的 rolloutn的拆开，一个问题rollout多次的结果
               sample_id表示一个问题的 第i次rollout
            **解释：**
            - 遍历当前 `output` 的所有采样结果
            - `output.outputs` 是一个列表，包含该 prompt 的所有采样（如果 `n > 1`）
            - `output.outputs` 的结构：`List[CompletionOutput]`
            - 在你的代码中，`n=1`（已经在 `ray_trainer` 中重复了），所以 `len(output.outputs) == 1`

            **示例：**
            ```python
            # 如果 sampling_params.n = 1（你的情况）
            output.outputs = [
                CompletionOutput(token_ids=[151644, 151645, ...], entropies=[0.12, 0.56, ...])
            ]

            # 如果 sampling_params.n = 3（每个 prompt 生成 3 个不同的回答）
            output.outputs = [
                CompletionOutput(...),  # 第 1 个采样
                CompletionOutput(...),  # 第 2 个采样
                CompletionOutput(...),  # 第 3 个采样
            ]
            ```
            
            3. response_ids = output.outputs[sample_id].token_ids

            **解释：**
            - 获取当前采样的所有生成 token IDs
            - `token_ids` 是一个列表：`List[int]`
            - 长度为 `response_length`（你的配置中是 128）

            **示例：**
            ```python
            response_ids = [151644, 151645, 151646, ..., 151643]  # 128 个 token IDs
            # 对应的文本可能是：["To", " solve", " this", ..., "<|endoftext|>"]
            ```

            4. response.append(response_ids)

            **解释：**
            - 将当前采样的 token IDs 添加到 `response` 列表
            - `response` 是一个二维列表：`List[List[int]]`
            - 最终 `response` 的形状：`[batch_size, response_length]`

            **示例：**
            ```python
            # 初始状态
            response = []

            # 第 1 次循环后
            response = [[151644, 151645, ...]]  # rollout 1

            # 第 2 次循环后
            response = [
                [151644, 151645, ...],  # rollout 1
                [151647, 151648, ...],  # rollout 2
            ]

            # 最终（batch_size=8）
            response = [
                [151644, 151645, ...],  # rollout 1
                [151647, 151648, ...],  # rollout 2
                ...
                [151699, 151700, ...],  # rollout 8
            ]
            ```

            4. if self.config.calculate_log_probs:

            **解释：**
            - 检查配置中是否启用了 log probability 计算
            - `calculate_log_probs` 是一个布尔值，默认为 `False`
            - 如果为 `True`，会提取每个 token 的对数概率

            5. curr_log_prob = []

            **解释：**
            - 初始化当前样本的 log probability 列表
            - 用于存储该样本每个 token 的对数概率


            6. for i, logprob in enumerate(output.outputs[sample_id].logprobs):

            **解释：**
            - 遍历当前采样的所有 token 的 logprobs
            - `output.outputs[sample_id].logprobs` 是一个列表：`List[Dict[int, Logprob]]`
            - 每个元素是一个字典，key 是 token ID，value 是 `Logprob` 对象

            **示例：**
            ```python
            output.outputs[0].logprobs = [
                {151644: Logprob(logprob=-0.123, rank=1, decoded_token="To")},      # token 0
                {151645: Logprob(logprob=-0.456, rank=1, decoded_token=" solve")},  # token 1
                {151646: Logprob(logprob=-0.789, rank=1, decoded_token=" this")},   # token 2
                ...
            ]
            ```

            7. curr_log_prob.append(logprob[response_ids[i]].logprob)

            **解释：**
            - 从 `logprob` 字典中提取当前 token ID 对应的对数概率
            - `logprob[response_ids[i]]` 返回一个 `Logprob` 对象
            - `.logprob` 获取其中的对数概率值（浮点数）

            **示例：**
            ```python
            # i = 0, response_ids[0] = 151644
            logprob = {151644: Logprob(logprob=-0.123, ...)}
            curr_log_prob.append(logprob[151644].logprob)  # 添加 -0.123

            # i = 1, response_ids[1] = 151645
            logprob = {151645: Logprob(logprob=-0.456, ...)}
            curr_log_prob.append(logprob[151645].logprob)  # 添加 -0.456

            # 最终
            curr_log_prob = [-0.123, -0.456, -0.789, ...]  # 128 个值
            ```


            8. rollout_log_probs.append(curr_log_prob)

            **解释：**
            - 将当前样本的 log probabilities 添加到 `rollout_log_probs` 列表
            - `rollout_log_probs` 是一个二维列表：`List[List[float]]`
            - 最终形状：`[batch_size, response_length]`

            **示例：**
            ```python
            rollout_log_probs = [
                [-0.123, -0.456, -0.789, ...],  # 样本 0 的 log probs
                [-0.234, -0.567, -0.890, ...],  # 样本 1 的 log probs
                ...
            ]
            ```

        # 新增：提取熵值
        9. if hasattr(output.outputs[sample_id], 'entropies') and output.outputs[sample_id].entropies:

        **解释：**
        - **第一个条件** `hasattr(output.outputs[sample_id], 'entropies')`：
        - 检查 `CompletionOutput` 对象是否有 `entropies` 属性
        - 如果 vLLM 没有修改，这个属性不存在，返回 `False`
        
        - **第二个条件** `output.outputs[sample_id].entropies`：
        - 检查 `entropies` 是否非空（不是 `None` 或空列表）
        - 这是一个**短路求值**，只有第一个条件为 `True` 时才会执行

        **为什么需要两个条件？**
        1. 如果没有 `hasattr` 检查，直接访问 `entropies` 会抛出 `AttributeError`
        2. 如果 `entropies` 存在但为 `None` 或 `[]`，第二个条件会过滤掉

        **示例：**
        ```python
        # 情况 1：vLLM 未修改
        hasattr(output.outputs[0], 'entropies')  # False
        # 短路，不执行第二个条件

        # 情况 2：vLLM 已修改，但熵值为空
        output.outputs[0].entropies = None
        hasattr(output.outputs[0], 'entropies')  # True
        output.outputs[0].entropies  # None（falsy）
        # 整体条件为 False

        # 情况 3：vLLM 已修改，熵值正常
        output.outputs[0].entropies = [0.12, 0.56, 0.34, ...]
        hasattr(output.outputs[0], 'entropies')  # True
        output.outputs[0].entropies  # [0.12, 0.56, ...]（truthy）
        # 整体条件为 True
        ```
        
        10. rollout_entropies.append(output.outputs[sample_id].entropies)

        **解释：**
        - 将当前样本的熵值列表添加到 `rollout_entropies`
        - `output.outputs[sample_id].entropies` 是一个列表：`List[float]`
        - `rollout_entropies` 是一个二维列表：`List[List[float]]`
        - 最终形状：`[batch_size, response_length]`

        **示例：**
        ```python
        # 单个样本的熵值
        output.outputs[0].entropies = [0.1234, 0.5678, 0.2345, ...]  # 128 个值

        # rollout_entropies 的构建过程
        rollout_entropies = []

        # 第 1 次循环
        rollout_entropies = [[0.1234, 0.5678, 0.2345, ...]]

        # 第 2 次循环
        rollout_entropies = [
            [0.1234, 0.5678, 0.2345, ...],  # 样本 0
            [0.2345, 0.6789, 0.3456, ...],  # 样本 1
        ]

        # 最终（batch_size=8）
        rollout_entropies = [
            [0.1234, 0.5678, ...],  # 样本 0 的熵值
            [0.2345, 0.6789, ...],  # 样本 1 的熵值
            ...
            [0.3456, 0.7890, ...],  # 样本 7 的熵值
        ]
        ```

            ## 🔍 完整数据流示例

            假设 `batch_size = 2`，`response_length = 4`（简化）：

            ```python
            # vLLM 返回的 outputs
            outputs = [
                RequestOutput(
                    outputs=[
                        CompletionOutput(
                            token_ids=[151644, 151645, 151646, 151647],
                            logprobs=[
                                {151644: Logprob(logprob=-0.1, ...)},
                                {151645: Logprob(logprob=-0.2, ...)},
                                {151646: Logprob(logprob=-0.3, ...)},
                                {151647: Logprob(logprob=-0.4, ...)},
                            ],
                            entropies=[0.12, 0.56, 0.34, 0.78]
                        )
                    ]
                ),
                RequestOutput(
                    outputs=[
                        CompletionOutput(
                            token_ids=[151648, 151649, 151650, 151651],
                            logprobs=[...],
                            entropies=[0.23, 0.67, 0.45, 0.89]
                        )
                    ]
                ),
            ]

            # 循环后的结果
            response = [
                [151644, 151645, 151646, 151647],  # 样本 0
                [151648, 151649, 151650, 151651],  # 样本 1
            ]

            rollout_log_probs = [
                [-0.1, -0.2, -0.3, -0.4],  # 样本 0
                [...],                      # 样本 1
            ]

            rollout_entropies = [
                [0.12, 0.56, 0.34, 0.78],  # 样本 0
                [0.23, 0.67, 0.45, 0.89],  # 样本 1
            ]
            ```

            ## 🎯 关键点总结

            1. **双层循环**：
            - 外层：遍历 batch 中的每个 prompt
            - 内层：遍历每个 prompt 的采样结果（通常 `n=1`）

            2. **数据结构**：
            - `response`: `List[List[int]]` - token IDs
            - `rollout_log_probs`: `List[List[float]]` - log probabilities
            - `rollout_entropies`: `List[List[float]]` - entropies

            3. **安全检查**：
            - `hasattr()` 确保属性存在
            - `and entropies` 确保值非空

            4. **最终形状**：
            - 所有列表的形状都是 `[batch_size, response_length]`
            - 例如：`[8, 128]`（8 个样本，每个 128 个 token）

            希望这个详细解释能帮助你理解这段代码的工作原理！
            '''
            
            
            # Debug: 检查熵值提取状态并打印生成内容
            if self.compute_entropy:
                logger.debug(f"\n{'='*80}")
                logger.info(f"[DEBUG Entropy] Rollout Entropy Debug Info")
                logger.debug(f"{'='*80}")
                # logger.debug(f"Total outputs: {len(outputs)}")
                # logger.debug(f"Collected entropies: {len(rollout_entropies)}")
                
                if rollout_entropies:
                    # 打印第一个样本的详细信息 
                    # 核对 是不是每个token都计算了熵值？
                    logger.info(f"\n[Sample 0] Generated {len(response[0])} tokens")
                    logger.info(f"[Sample 0] Entropy values ({len(rollout_entropies[0])} values):")
                    
                    # 解码生成的文本
                    from transformers import AutoTokenizer
                    try:
                        # 使用已有的 tokenizer
                        tokenizer = self.tokenizer
                        generated_text = tokenizer.decode(response[0], skip_special_tokens=False)
                        
                        # logger.info(f"\n[Sample 0] Generated text:")
                        # logger.info(f"{generated_text[:500]}...")  # 只打印前500个字符
                        
                        # 把前20个token的 解码文字 和 对应的熵值 打印出来
                        logger.info(f"\n[Sample 0] Token-by-token breakdown (first 20 tokens):")
                        logger.info(f"{'Token ID':<12} {'Token':<20} {'Entropy':<10}")
                        logger.info(f"{'-'*42}")
                        for i in range(min(20, len(response[0]))):
                            token_id = response[0][i]
                            token_text = tokenizer.decode([token_id], skip_special_tokens=False)
                            entropy_val = rollout_entropies[0][i] if i < len(rollout_entropies[0]) else 0.0
                            logger.info(f"{token_id:<12} {repr(token_text):<20} {entropy_val:.4f}")
                        
                        '''
                        逐行解释：
                        1. logger.info(f"{'Token ID':<12} {'Token':<20} {'Entropy':<10}")
                            打印表头
                            <12、<20、<10 是左对齐格式化，分别占 12、20、10 个字符宽度
                            输出类似：Token ID Token Entropy
                        2.logger.info(f"{'-'*42}")
                            打印分隔线，42 个 - 字符
                        3.for i in range(min(20, len(response[0])))
                            遍历前 20 个 token（如果不足 20 个，就遍历所有）
                        4.response[0] 是第一个样本的所有生成 token IDs
                          token_id = response[0][i]
                            获取第 i 个 token 的 ID（例如：151643）
                        5.token_text = tokenizer.decode([token_id], skip_special_tokens=False)
                            将 token ID 解码为文本
                        6.skip_special_tokens=False 保留特殊 token（如 <|endoftext|>）
                        entropy_val = rollout_entropies[0][i] if i < len(rollout_entropies[0]) else 0.0
                            获取第 i 个 token 的熵值
                            如果熵值列表不够长，默认为 0.0
                        7.logger.info(f"{token_id:<12} {repr(token_text):<20} {entropy_val:.4f}")
                            打印一行：token ID、token 文本、熵值
                            repr(token_text) 会显示转义字符（如 '\n' 而不是换行）
                            {entropy_val:.4f} 保留 4 位小数
                        
                        输出示例：
                        
                        Token ID     Token                Entropy   
                        ------------------------------------------
                        151644       'To'                 0.1234
                        151645       ' solve'             0.5678
                        151646       ' this'              0.2345
                        ...
                        
                        '''
                        
                        
                        # 统计信息
                        entropies_list = rollout_entropies[0][:len(response[0])]
                        avg_entropy = sum(entropies_list) / len(entropies_list) if entropies_list else 0.0
                        min_entropy = min(entropies_list) if entropies_list else 0.0
                        max_entropy = max(entropies_list) if entropies_list else 0.0
                        
                        logger.debug(f"\n[Sample 0] Entropy statistics:")
                        logger.debug(f"  Average: {avg_entropy:.4f}")
                        logger.debug(f"  Min:     {min_entropy:.4f}")
                        logger.debug(f"  Max:     {max_entropy:.4f}")
                        
                    except Exception as e:
                        logger.error(f"[DEBUG Entropy] Error decoding text: {e}")
                        logger.debug(f"[Sample 0] First 10 entropy values: {[f'{e:.4f}' for e in rollout_entropies[0][:10]]}")
                else:
                    logger.warning(f"\n⚠️  No entropies collected! vLLM may not be modified yet.")
                    logger.debug(f"\nChecking first output attributes:")
                    if outputs:
                        first_output = outputs[0].outputs[0]
                        available_attrs = [attr for attr in dir(first_output) if not attr.startswith('_')]
                        # logger.debug(f"  Available attributes: {available_attrs}")
                        # logger.debug(f"  Has 'entropies' attr: {hasattr(first_output, 'entropies')}")
                        
                        # 打印生成的文本（即使没有熵值）
                        #logger.debug(f"\n[Sample 0] Generated {len(response[0])} tokens (without entropy)")
                        from transformers import AutoTokenizer
                        try:
                            tokenizer = self.tokenizer
                            generated_text = tokenizer.decode(response[0], skip_special_tokens=False)
                            logger.info(f"[Sample 0] Generated text:")
                            logger.info(f"{generated_text[:500]}...")
                        except Exception as e:
                            logger.error(f"Error decoding: {e}")
                
                logger.info(f"{'='*80}\n")

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

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

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
