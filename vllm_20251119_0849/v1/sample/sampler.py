# SPDX-License-Identifier: Apache-2.0
"""A layer that samples the next tokens from the model's outputs."""


import logging
import torch
import torch.nn as nn

from vllm.v1.outputs import LogprobsTensors, SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.bad_words import apply_bad_words
from vllm.v1.sample.ops.penalties import (apply_all_penalties,
                                          apply_min_token_penalties)
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler


_SAMPLING_EPS = 1e-5
# 创建 logger
logger = logging.getLogger(__name__)


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()
        self.topk_topp_sampler = TopKTopPSampler()

    def forward(
        self,
        logits: torch.Tensor, # 这是大模型打分的原始分数（未归一化的），对于 词表中的每一个词 打分
        sampling_metadata: SamplingMetadata, # SamplingMetadata类 记录了很多参数，可以看 /opt/dhl_conda_envs/verl_fast/lib/python3.10/site-packages/vllm/v1/sample/metadata.py
    ) -> SamplerOutput:
        
        '''在代码上下文中，logits 是模型对下一个 token 的原始预测分数。     
        📚 技术定义
        logits = 模型对下一个 token 的偏好分数矩阵
        logits[i, j] = 模型认为第 i 个序列的下一个 token 是词汇表中第 j 个 token 的原始分数logits = 模型对下一个 token 的偏好分数矩阵
        维度: [当前序列数, 词汇表大小]  torch.Tensor  # 形状: [batch_size, vocab_size]
        含义: 每个分数代表模型对相应 token 的"喜爱程度"
        '''
        
        # 🆕 DEBUG: 进入 forward 方法
        #logger.debug(f"[V1 SAMPLER] 🚀 进入 Sampler.forward()")
        #logger.debug(f"[V1 SAMPLER]   logits.shape = {logits.shape}")
        # logits.shape = torch.Size([1024, 151936])
        if hasattr(sampling_metadata, 'compute_entropy'):
            logger.debug(f"[V1 SAMPLER]   sampling_metadata.compute_entropy = {sampling_metadata.compute_entropy}")
        
        # NOTE(woosuk): Use the original logits (before any penalties or
        # temperature scaling) for the top-k logprobs.
        # This is different from the V0 sampler, which uses the logits that
        # is used for sampling (after penalties and temperature scaling).
        # TODO(rob): provide option for logprobs post sampling.
        # See https://vllm-dev.slack.com/archives/C07UUL8E61Z/p1735907856007919 # noqa: E501
        '''# sampling_metadata.max_num_logprobs
           # None 表示不返回 logprobs
           # 0 表示只返回被采样token的logprob（单个token）
           # N>0 表示返回前N个最可能token的logprobs（包括被采样的token）
           # 情况2: max_num_logprobs = 0
            # 只返回实际被采样的那个token的logprob
            # 例如：采样得到token_id=1234，就只返回这个token的logprob
            # 返回格式: [{1234: -0.15}]  # 只有一个键值对
            情况3: max_num_logprobs = 5
            python
            # 返回前5个最可能token的logprobs（包括被采样的那个）
            # 例如：实际采样了token_id=1234，但还会返回其他4个高概率候选
            # 返回格式: [
            #   {1234: -0.15, 5678: -0.23, 9012: -0.45, 3456: -0.67, 7890: -0.89}]
            '''
        num_logprobs = sampling_metadata.max_num_logprobs # 这个是 每个token模型有规定 必须返回最高的几个 概率的词
        if num_logprobs is not None:
            raw_logprobs = self.compute_logprobs(logits) # 这里将所有 token惩罚前的值 进行了记录raw_logprobs

        # 先转换数据类型 以及 做一些词表过滤
        # Use float32 for the logits.
        logits = logits.to(torch.float32)
        # Apply allowed token ids.
        logits = self.apply_allowed_token_ids(logits, sampling_metadata)
        # Apply bad words exclusion.
        logits = self.apply_bad_words(logits, sampling_metadata)
        # Apply logits bias.
        logits = self.apply_logits_bias(logits, sampling_metadata)
        # Apply penalties (e.g., min_tokens, freq_penalties).
        logits = self.apply_penalties(logits, sampling_metadata)
        
        # 🆕 计算熵（在 sample 之前，使用处理后的 logits）
        
        entropies_tensor = None
        if hasattr(sampling_metadata, 'compute_entropy') and sampling_metadata.compute_entropy:
            #logger.info(f"[V1 SAMPLER] ✅ 开始计算熵")
            # 计算概率分布和对数概率
            probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
            logprobs_for_entropy = torch.log_softmax(logits, dim=-1, dtype=torch.float32)
            
            # logger.debug(f"[V1 SAMPLER]   probs.shape = {probs.shape}")
            # logger.debug(f"[V1 SAMPLER]   logprobs_for_entropy.shape = {logprobs_for_entropy.shape}")
            '''probs.shape = torch.Size([1, 151936]) logprobs_for_entropy.shape = torch.Size([1, 151936])'''
            # 计算熵: H = -sum(p * log(p))
            entropies_tensor = -torch.sum(probs * logprobs_for_entropy, dim=-1)
            
            #logger.debug(f"[V1 SAMPLER] 🎯 熵计算完成！")
            #logger.debug(f"[V1 SAMPLER]   entropies_tensor.shape = {entropies_tensor.shape}")
            #logger.debug(f"[V1 SAMPLER]   entropies_tensor.dtype = {entropies_tensor.dtype}")
            #logger.debug(f"[V1 SAMPLER]   entropies_tensor 前5个值 = {entropies_tensor[:min(5, len(entropies_tensor))].tolist()}")
            #logger.debug(f"[V1 SAMPLER]   entropies_tensor 平均值 = {entropies_tensor.mean().item():.4f}")
            #logger.debug(f"[V1 SAMPLER]   entropies_tensor 最小值 = {entropies_tensor.min().item():.4f}")
            #logger.debug(f"[V1 SAMPLER]   entropies_tensor 最大值 = {entropies_tensor.max().item():.4f}")
        else:
            compute_entropy_val = getattr(sampling_metadata, 'compute_entropy', False)
            logger.warning(f"[V1 SAMPLER] ⚠️  熵计算未执行")
            logger.debug(f"[V1 SAMPLER]   hasattr(sampling_metadata, 'compute_entropy') = {hasattr(sampling_metadata, 'compute_entropy')}")
            logger.debug(f"[V1 SAMPLER]   compute_entropy = {compute_entropy_val}")
        
        # Sample the next token.
        sampled = self.sample(logits, sampling_metadata)
        # Convert sampled token ids to int64 (long) type to ensure compatibility
        # with subsequent operations that may use these values as indices.
        # This conversion is necessary because FlashInfer sampling operations
        # return int32 (while PyTorch argmax and topk return int64).
        sampled = sampled.long()

        # Gather the logprobs of the topk and sampled token (if requested).
        # Get logprobs and rank tensors (if requested)
        logprobs_tensors = None if num_logprobs is None else \
            self.gather_logprobs(raw_logprobs, num_logprobs, token_ids=sampled)

        # Use int32 to reduce the tensor size.
        sampled = sampled.to(torch.int32)

        # These are GPU tensors.
        #logger.info(f"[V1 SAMPLER] 📦 构造 SamplerOutput...")
        #logger.debug(f"[V1 SAMPLER]   sampled.shape = {sampled.shape}")
        #logger.debug(f"[V1 SAMPLER]   entropies_tensor is None = {entropies_tensor is None}")
        
        sampler_output = SamplerOutput(
            # The sampled tokens are expanded to 2D tensor with shape
            # [num_requests, 1], where each row represents one generated
            # token per request.
            sampled_token_ids=sampled.unsqueeze(-1),
            logprobs_tensors=logprobs_tensors,
            entropies=entropies_tensor,  # 🆕 添加熵值
        )
        
        #logger.debug(f"[V1 SAMPLER] ✅ SamplerOutput 创建成功")
        if sampler_output.entropies is not None:
            logger.debug(f"[V1 SAMPLER]   sampler_output.entropies.shape = {sampler_output.entropies.shape}")
        #logger.info(f"[V1 SAMPLER] 🏁 退出 Sampler.forward()")
        
        return sampler_output

    def apply_temperature(
        self,
        logits: torch.Tensor,
        temp: torch.Tensor,
    ) -> torch.Tensor:
        # Use in-place division to avoid creating a new tensor.
        return logits.div_(temp.unsqueeze(dim=1))

    def greedy_sample(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1).view(-1)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        """Sample logits based on sampling metadata.

        The various logits processing functions called in this method
        may update the logits tensor in-place.
        """

        assert not (sampling_metadata.all_greedy
                    and sampling_metadata.all_random)
        if sampling_metadata.all_random:
            greedy_sampled = None
        else:
            greedy_sampled = self.greedy_sample(logits)
            if sampling_metadata.all_greedy:
                return greedy_sampled

        assert sampling_metadata.temperature is not None

        # Apply temperature.
        logits = self.apply_temperature(logits, sampling_metadata.temperature)

        # Apply min_p.
        if sampling_metadata.min_p is not None:
            logits = self.apply_min_p(logits, sampling_metadata.min_p)

        # Apply top_k and/or top_p.
        random_sampled = self.topk_topp_sampler(
            logits,
            sampling_metadata.generators,
            sampling_metadata.top_k,
            sampling_metadata.top_p,
        )

        if greedy_sampled is None:
            return random_sampled

        sampled = torch.where(
            sampling_metadata.temperature < _SAMPLING_EPS,
            greedy_sampled,
            random_sampled,
            out=greedy_sampled,  # Reuse tensor
        )
        return sampled

    def compute_logprobs(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.log_softmax(dim=-1, dtype=torch.float32)

    def gather_logprobs(
        self,
        logprobs: torch.Tensor,
        num_logprobs: int,
        token_ids: torch.Tensor,
    ) -> LogprobsTensors:
        """
        Gather logprobs for topk and sampled/prompt token.

        Args:
          logprobs: (num tokens) x (vocab) tensor
          num_logprobs: minimum number of logprobs to
                        retain per token
          token_ids: prompt tokens (if prompt logprobs)
                     or sampled tokens (if sampled
                     logprobs); 1D token ID tensor
                     with (num tokens) elements
                     Must be int64.

        Returns:
          Top-k int indices tensor, (num tokens) x (num_logprobs + 1)
          Top-k float logprobs tensor, (num tokens) x (num_logprobs + 1)
          Sampled token rank tensor, (num tokens)
        """
        assert token_ids.dtype == torch.int64
        # Find the topK values.
        topk_logprobs, topk_indices = torch.topk(logprobs,
                                                 num_logprobs,
                                                 dim=-1)

        # Get with the logprob of the prompt or sampled token.
        token_ids = token_ids.unsqueeze(-1)
        token_logprobs = logprobs.gather(-1, token_ids)

        # Compute the ranks of the actual token.
        token_ranks = (logprobs >= token_logprobs).sum(-1)

        # Concatenate together with the topk.
        indices = torch.cat((token_ids, topk_indices), dim=1)
        logprobs = torch.cat((token_logprobs, topk_logprobs), dim=1)

        # Use int32 to reduce the tensor size.
        indices = indices.to(torch.int32)

        return LogprobsTensors(indices, logprobs, token_ranks)

    def apply_penalties(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        if sampling_metadata.min_tokens:
            apply_min_token_penalties(logits,
                                      sampling_metadata.output_token_ids,
                                      sampling_metadata.min_tokens)
        if not sampling_metadata.no_penalties:
            assert sampling_metadata.prompt_token_ids is not None
            logits = apply_all_penalties(
                logits,
                sampling_metadata.prompt_token_ids,
                sampling_metadata.presence_penalties,
                sampling_metadata.frequency_penalties,
                sampling_metadata.repetition_penalties,
                sampling_metadata.output_token_ids,
            )
        return logits

    def apply_min_p(
        self,
        logits: torch.Tensor,
        min_p: torch.Tensor,
    ) -> torch.Tensor:
        """
        Filters logits using adaptive probability thresholding.
        """
        # Convert logits to probability distribution
        probability_values = torch.nn.functional.softmax(logits, dim=-1)
        # Calculate maximum probabilities per sequence
        max_probabilities = torch.amax(probability_values,
                                       dim=-1,
                                       keepdim=True)
        # Reshape min_p for broadcasting
        adjusted_min_p = min_p.unsqueeze(1) * max_probabilities
        # Identify valid tokens using threshold comparison
        valid_token_mask = probability_values >= adjusted_min_p
        # Apply mask using boolean indexing
        logits[~valid_token_mask] = -float('inf')
        return logits

    def apply_logits_bias(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        # TODO(houseroad): this implementation is extremely inefficient.
        # One idea is implement this as a PyTorch C++ op, and we may
        # even optimize the logit_bias layout.

        # Get vocabulary size from logits
        vocab_size = logits.shape[-1]

        for i, logit_bias in enumerate(sampling_metadata.logit_bias):
            if logit_bias:
                for token_id, bias in logit_bias.items():
                    # Check token_id bounds to ensure within vocabulary
                    if token_id < 0 or token_id >= vocab_size:
                        raise ValueError(
                            f"token_id {token_id} in logit_bias contains "
                            f"out-of-vocab token id. Vocabulary size: "
                            f"{vocab_size}")
                    logits[i, token_id] += bias
        return logits

    def apply_allowed_token_ids(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        if sampling_metadata.allowed_token_ids_mask is not None:
            logits.masked_fill_(sampling_metadata.allowed_token_ids_mask,
                                float("-inf"))
        return logits

    def apply_bad_words(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        if sampling_metadata.bad_words_token_ids:
            apply_bad_words(
                logits,
                sampling_metadata.bad_words_token_ids,
                sampling_metadata.output_token_ids,
            )
        return logits
