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
EntropyOutputWriter: 用于保存 token-level entropy 到 JSONL 文件
"""

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class EntropyOutputConfig:
    """熵输出配置"""
    enabled: bool = False  # 是否启用熵输出
    output_dir: str = "./entropy_outputs"  # 输出目录
    top_k: int = 10  # 标记熵值最高的 K 个 token
    save_interval: int = 10  # 每 N 个 step 保存一次
    mark_style: str = "both"  # "markdown" | "html" | "both"
    token_entropy_to_jsonl: bool = False  # DEBUG: 是否保存 token-level entropy


class EntropyOutputWriter:
    """
    用于收集和保存 token-level entropy 数据到 JSONL 文件
    
    功能：
    1. 收集每个样本的 prompt, response, entropy_list, data_source
    2. 计算 Top-K 高熵 token
    3. 生成 marked_response（用 **[token]** 标记高熵词）
    4. 计算统计信息（mean, min, max, std）
    5. 按 epoch + dataset 分开保存到 JSONL 文件
    """
    
    def __init__(
        self, 
        config: EntropyOutputConfig, 
        tokenizer, 
        rank: int = 0
    ):
        """
        Args:
            config: 熵输出配置
            tokenizer: HuggingFace tokenizer，用于解码 token
            rank: 当前进程的 rank（只在 rank 0 写入文件）
        """
        self.config = config
        self.tokenizer = tokenizer
        self.rank = rank
        self.enabled = (self.rank == 0) and config.enabled
        
        # 缓冲区：按 (epoch, data_source) 分组存储样本
        # 结构: {(epoch, data_source): [sample1, sample2, ...]}
        self.buffer: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
        
        # 统计信息
        self.total_samples = 0
        self.current_step = 0
        
        if self.enabled:
            # 创建输出目录
            os.makedirs(config.output_dir, exist_ok=True)
            logger.info(f"[EntropyOutputWriter] Initialized at rank {rank}")
            logger.info(f"  Output dir: {config.output_dir}")
            logger.info(f"  Top-K: {config.top_k}")
            logger.info(f"  Save interval: {config.save_interval} steps")
            logger.info(f"  Mark style: {config.mark_style}")
    
    def add_sample(self, sample_data: Dict[str, Any]) -> None:
        """
        添加单个样本到缓冲区
        
        Args:
            sample_data: 包含以下字段的字典
                - epoch: int
                - step: int
                - data_source: str (例如 "openai/gsm8k")
                - prompt: str
                - response: str
                - entropy_list: List[float]
        """
        if not self.enabled:
            return
        
        try:
            # 提取必要字段
            epoch = sample_data.get("epoch", 0)
            step = sample_data.get("step", 0)
            data_source = sample_data.get("data_source", "unknown")
            prompt = sample_data.get("prompt", "")
            response = sample_data.get("response", "")
            entropy_list = sample_data.get("entropy_list", [])
            
            # 验证数据
            if not entropy_list:
                logger.warning(f"[EntropyOutputWriter] Empty entropy_list for step {step}, skipping")
                return
            
            # 解码 response 为 token 列表（用于计算 Top-K）
            # 注意：这里需要逐个 token 解码，以便后续标记
            response_tokens = []
            if response:
                # 先编码再逐个解码，确保 token 边界正确
                response_token_ids = self.tokenizer.encode(response, add_special_tokens=False)
                response_tokens = [
                    self.tokenizer.decode([token_id], skip_special_tokens=False) 
                    for token_id in response_token_ids
                ]
            
            # 确保 entropy_list 长度与 token 数量匹配
            if len(entropy_list) != len(response_tokens):
                logger.warning(
                    f"[EntropyOutputWriter] Entropy list length ({len(entropy_list)}) "
                    f"!= token count ({len(response_tokens)}), truncating/padding"
                )
                # 截断或填充
                if len(entropy_list) > len(response_tokens):
                    entropy_list = entropy_list[:len(response_tokens)]
                else:
                    entropy_list = entropy_list + [0.0] * (len(response_tokens) - len(entropy_list))
            
            # 计算 Top-K 高熵 token
            top_k_data = self._calculate_top_k(entropy_list, response_tokens)
            
            # 生成 marked_response
            marked_response = self._generate_marked_response(
                response, 
                response_tokens,
                top_k_data["top_k_high_entropy_indices"]
            )
            
            # 计算统计信息
            entropy_stats = self._calculate_stats(entropy_list)
            
            # 构造完整的样本数据
            full_sample = {
                "epoch": epoch,
                "step": step,
                "data_source": data_source,
                "prompt": prompt,
                "response": response,
                "response_length": len(response),  # 🆕 添加 response 长度
                "entropy_list": entropy_list,
                "entropy_length": len(entropy_list),  # 🆕 添加 entropy_list 长度
                **top_k_data,
                "entropy_stats": entropy_stats,
                "marked_response": marked_response,
            }
            
            # 添加到缓冲区
            key = (epoch, data_source)
            self.buffer[key].append(full_sample)
            self.total_samples += 1
            self.current_step = step
            
            logger.debug(
                f"[EntropyOutputWriter] Added sample: epoch={epoch}, step={step}, "
                f"data_source={data_source}, tokens={len(response_tokens)}"
            )
            
        except Exception as e:
            logger.error(f"[EntropyOutputWriter] Error adding sample: {e}", exc_info=True)
    
    def _calculate_top_k(
        self, 
        entropy_list: List[float], 
        tokens: List[str]
    ) -> Dict[str, Any]:
        """
        计算 Top-K 高熵 token
        
        Returns:
            {
                "top_k_high_entropy_indices": [5, 12, 18, ...],
                "top_k_high_entropy_tokens": [
                    {"index": 5, "token": " calculate", "entropy": 0.8765},
                    ...
                ]
            }
        """
        if not entropy_list or not tokens:
            return {
                "top_k_high_entropy_indices": [],
                "top_k_high_entropy_tokens": []
            }
        
        # 创建 (index, entropy, token) 三元组
        indexed_entropies = [
            (i, entropy, tokens[i]) 
            for i, entropy in enumerate(entropy_list)
            if i < len(tokens)
        ]
        
        # 按 entropy 降序排序
        indexed_entropies.sort(key=lambda x: x[1], reverse=True)
        
        # 取 Top-K
        top_k = min(self.config.top_k, len(indexed_entropies))
        top_k_items = indexed_entropies[:top_k]
        
        # 提取 indices 和详细信息
        top_k_indices = [item[0] for item in top_k_items]
        top_k_tokens = [
            {
                "index": item[0],
                "token": item[2],
                "entropy": float(item[1])  # 转换为 Python float
            }
            for item in top_k_items
        ]
        
        return {
            "top_k_high_entropy_indices": top_k_indices,
            "top_k_high_entropy_tokens": top_k_tokens
        }
    
    def _generate_marked_response(
        self, 
        response: str, 
        tokens: List[str],
        top_k_indices: List[int]
    ) -> str:
        """
        生成标记了高熵词的 response
        
        使用 **[token]** 格式标记 Top-K 高熵 token
        
        Args:
            response: 原始 response 文本
            tokens: token 列表
            top_k_indices: Top-K 高熵 token 的 index 列表
        
        Returns:
            标记后的 response 文本
        """
        if not tokens or not top_k_indices:
            return response
        
        try:
            # 重建 response，标记高熵 token
            marked_tokens = []
            for i, token in enumerate(tokens):
                if i in top_k_indices:
                    marked_tokens.append(f"**[{token}]**")
                else:
                    marked_tokens.append(token)
            
            return "".join(marked_tokens)
        
        except Exception as e:
            logger.error(f"[EntropyOutputWriter] Error generating marked response: {e}")
            return response
    
    def _calculate_stats(self, entropy_list: List[float]) -> Dict[str, float]:
        """计算熵值统计信息"""
        if not entropy_list:
            return {
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
                "std": 0.0
            }
        
        entropy_array = np.array(entropy_list, dtype=np.float32)
        return {
            "mean": float(np.mean(entropy_array)),
            "min": float(np.min(entropy_array)),
            "max": float(np.max(entropy_array)),
            "std": float(np.std(entropy_array))
        }
    
    def flush(self) -> None:
        """批量写入缓冲区的数据到 JSONL 文件"""
        if not self.enabled or not self.buffer:
            return
        
        try:
            logger.info(
                f"[EntropyOutputWriter] Flushing {self.total_samples} samples "
                f"from {len(self.buffer)} groups at step {self.current_step}"
            )
            
            # 遍历所有 (epoch, data_source) 组
            for (epoch, data_source), samples in self.buffer.items():
                if not samples:
                    continue
                
                # 写入 JSONL 文件
                self._write_to_jsonl(epoch, data_source, samples)
            
            # 清空缓冲区
            self.buffer.clear()
            logger.info(f"[EntropyOutputWriter] Flush completed")
            
        except Exception as e:
            logger.error(f"[EntropyOutputWriter] Error during flush: {e}", exc_info=True)
    
    def _write_to_jsonl(
        self, 
        epoch: int, 
        data_source: str, 
        samples: List[Dict[str, Any]]
    ) -> None:
        """
        写入 JSONL 文件
        
        文件结构: {output_dir}/epoch_{epoch}/{dataset_name}.jsonl
        例如: ./entropy_outputs/epoch_0/gsm8k.jsonl
        """
        try:
            # 从 data_source 提取数据集名称
            # "openai/gsm8k" -> "gsm8k"
            # "lighteval/MATH" -> "math"
            dataset_name = data_source.split("/")[-1].lower()
            if dataset_name == "math":
                dataset_name = "math"
            elif dataset_name == "gsm8k":
                dataset_name = "gsm8k"
            else:
                dataset_name = dataset_name.replace("/", "_")
            
            # 创建 epoch 目录
            epoch_dir = Path(self.config.output_dir) / f"epoch_{epoch}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            
            # JSONL 文件路径
            jsonl_path = epoch_dir / f"{dataset_name}.jsonl"
            
            # 追加写入（如果文件已存在）
            with open(jsonl_path, "a", encoding="utf-8") as f:
                for sample in samples:
                    json_line = json.dumps(sample, ensure_ascii=False)
                    f.write(json_line + "\n")
            
            logger.info(
                f"[EntropyOutputWriter] Wrote {len(samples)} samples to {jsonl_path}"
            )
            
        except Exception as e:
            logger.error(
                f"[EntropyOutputWriter] Error writing to JSONL "
                f"(epoch={epoch}, data_source={data_source}): {e}",
                exc_info=True
            )
    
    def finalize(self) -> None:
        """训练结束时调用，确保所有数据都被写入"""
        if not self.enabled:
            return
        
        logger.info("[EntropyOutputWriter] Finalizing...")
        self.flush()
        logger.info(f"[EntropyOutputWriter] Total samples processed: {self.total_samples}")

