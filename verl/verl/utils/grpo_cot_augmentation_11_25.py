# # Copyright 2024 Bytedance Ltd. and/or its affiliates
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# """
# Utility for augmenting GRPO prompts with different COT examples per rollout.
# """

# import random
# from typing import List, Optional, Callable
# import numpy as np
# import torch
# from verl import DataProto
# from verl.utils.model import compute_position_id_with_mask


# class GRPOCOTAugmenter:
#     """
#     Augments repeated prompts in GRPO with different COT examples.
    
#     For each original prompt that gets repeated N times, this class adds
#     a different COT example to each repetition.
    
#     Example usage:
#         augmenter = GRPOCOTAugmenter(
#             cot_examples=["Example 1: ...", "Example 2: ...", "Example 3: ..."],
#             tokenizer=tokenizer,
#             num_repeats=4,
#             sampling_strategy="random_with_replacement"
#         )
#         gen_batch = augmenter.augment(gen_batch)
#     """
    
#     def __init__(
#         self,
#         cot_examples: Optional[List[str]] = None,
#         cot_examples_getter: Optional[Callable] = None,
#         tokenizer=None,
#         num_repeats: int = 1,
#         sampling_strategy: str = "random_with_replacement",
#         add_separator: bool = True,
#         separator: str = "\n\n",
#         enable: bool = True,
#         seed: Optional[int] = None,
#         debug_print_augmented_prompts: bool = True,
#         debug_num_samples: int = 3,
#         debug_print_full_prompt: bool = False,  # 🆕 是否打印完整prompt（不截断）
#     ):
#         """
#         Initialize the COT augmenter.
        
#         Args:
#             cot_examples: List of COT example strings to choose from.
#                          Can be None if cot_examples_getter is provided.
#             cot_examples_getter: Optional callable that returns COT examples dynamically.
#                                 Format: Callable[[batch_data], List[str]]
#                                 This allows generating different examples per prompt.
#             tokenizer: Tokenizer to encode the COT examples.
#             num_repeats: Number of times each prompt is repeated in GRPO.
#             sampling_strategy: How to sample COT examples:
#                 - "random_with_replacement": Random sampling (can repeat)
#                 - "random_without_replacement": Random sampling (no repeat, requires len(cot_examples) >= num_repeats)
#                 - "sequential": Use examples in order (循环使用)
#                 - "all_different_per_prompt": Each repetition gets a unique example per original prompt
#             add_separator: Whether to add a separator between prompt and COT example.
#             separator: The separator string (default: "\n\n").
#             enable: Whether to enable COT augmentation.
#             seed: Random seed for reproducibility.
#             debug_print_augmented_prompts: Whether to print augmented prompts for debugging.
#             debug_num_samples: Number of samples to print for debugging.
#         """
#         self.cot_examples = cot_examples
#         self.cot_examples_getter = cot_examples_getter
#         self.tokenizer = tokenizer
#         self.num_repeats = num_repeats
#         self.sampling_strategy = sampling_strategy
#         self.add_separator = add_separator
#         self.separator = separator
#         self.enable = enable
#         self.debug_print_augmented_prompts = debug_print_augmented_prompts
#         self.debug_num_samples = debug_num_samples
#         self.debug_print_full_prompt = debug_print_full_prompt  # 🆕 存储配置
        
#         if seed is not None:
#             random.seed(seed)
#             np.random.seed(seed)
        
#         # Validate configuration
#         if self.enable:
#             if self.cot_examples is None and self.cot_examples_getter is None:
#                 raise ValueError("Either cot_examples or cot_examples_getter must be provided when enable=True")
            
#             if self.tokenizer is None:
#                 raise ValueError("tokenizer must be provided when enable=True")
            
#             if self.sampling_strategy == "random_without_replacement" and self.cot_examples is not None:
#                 if len(self.cot_examples) < self.num_repeats:
#                     raise ValueError(
#                         f"For 'random_without_replacement' strategy, need at least {self.num_repeats} "
#                         f"COT examples, but only {len(self.cot_examples)} provided."
#                     )
    
#     def _remove_chat_template_roles(self, text: str) -> str:
#         """
#         移除聊天模板中的角色标记（如 'user', 'assistant' 等）。
        
#         常见的聊天模板格式：
#         - "user\n问题内容\nassistant"
#         - "user\n问题内容\nassistant\n"
        
#         我们需要保留问题内容，去掉角色标记。
        
#         Args:
#             text: 包含角色标记的文本
            
#         Returns:
#             清理后的纯文本问题
#         """
#         # 移除开头的 "user" 或 "user\n"
#         text = text.strip()
        
#         # 按行分割
#         lines = text.split('\n')
#         cleaned_lines = []
        
#         for line in lines:
#             line_stripped = line.strip()
#             # 跳过纯角色标记行（user, assistant, system等）
#             if line_stripped.lower() in ['user', 'assistant', 'system', '<|user|>', '<|assistant|>', '<|system|>']:
#                 continue
#             # 保留其他内容
#             if line_stripped:  # 跳过空行
#                 cleaned_lines.append(line)
        
#         # 重新组合
#         cleaned_text = '\n'.join(cleaned_lines).strip()
        
#         return cleaned_text
    
#     def _sample_cot_examples(self, num_samples: int, prompt_idx: Optional[int] = None) -> List[str]:
#         """
#         Sample COT examples based on the configured strategy.
        
#         Args:
#             num_samples: Number of examples to sample.
#             prompt_idx: Index of the original prompt (before repetition).
        
#         Returns:
#             List of sampled COT example strings.
#         """
#         cot_pool = self.cot_examples if self.cot_examples is not None else []
        
#         if self.sampling_strategy == "random_with_replacement":
#             return random.choices(cot_pool, k=num_samples)
        
#         elif self.sampling_strategy == "random_without_replacement":
#             return random.sample(cot_pool, k=num_samples)
        
#         elif self.sampling_strategy == "sequential":
#             # Use examples in circular order
#             result = []
#             for i in range(num_samples):
#                 idx = (prompt_idx + i) % len(cot_pool) if prompt_idx is not None else i % len(cot_pool)
#                 result.append(cot_pool[idx])
#             return result
        
#         elif self.sampling_strategy == "all_different_per_prompt":
#             # Each prompt gets its own set of examples
#             if prompt_idx is not None:
#                 start_idx = (prompt_idx * num_samples) % len(cot_pool)
#                 indices = [(start_idx + i) % len(cot_pool) for i in range(num_samples)]
#                 return [cot_pool[i] for i in indices]
#             else:
#                 return random.sample(cot_pool, k=min(num_samples, len(cot_pool)))
        
#         else:
#             raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
    
#     def augment(self, gen_batch: DataProto) -> DataProto:
#         """
#         Augment the repeated prompts with different COT examples.
        
#         Args:
#             gen_batch: DataProto after repeat() has been called.
#                       Assumes batch size = original_batch_size * num_repeats
        
#         Returns:
#             DataProto with augmented prompts.
#         """
#         if not self.enable:
#             return gen_batch
        
#         batch_size = gen_batch.batch["input_ids"].shape[0]
#         original_batch_size = batch_size // self.num_repeats
        
#         if batch_size % self.num_repeats != 0:
#             raise ValueError(
#                 f"Batch size {batch_size} is not divisible by num_repeats {self.num_repeats}. "
#                 "Make sure repeat() was called before augment()."
#             )
        
#         # Extract original input_ids and attention_mask
#         input_ids = gen_batch.batch["input_ids"]  # Shape: (batch_size, seq_len)
#         attention_mask = gen_batch.batch["attention_mask"]  # Shape: (batch_size, seq_len)
        
#         new_input_ids_list = []
#         new_attention_mask_list = []
#         new_position_ids_list = []
        
#         # Process each original prompt and its repetitions
#         for orig_idx in range(original_batch_size):
#             # Get COT examples for this prompt's repetitions
#             if self.cot_examples_getter is not None:
#                 # Use dynamic getter
#                 cot_examples_for_prompt = self.cot_examples_getter(gen_batch, orig_idx, self.num_repeats)
#             else:
#                 # Use static pool
#                 cot_examples_for_prompt = self._sample_cot_examples(self.num_repeats, prompt_idx=orig_idx)
            
#             # ⭐ Check if COT matching failed (empty list indicates skip)
#             skip_this_prompt = (len(cot_examples_for_prompt) == 0)
            
#             # Process each repetition
#             for rep_idx in range(self.num_repeats):
#                 batch_idx = orig_idx * self.num_repeats + rep_idx
                
#                 # Get the original prompt tokens
#                 orig_input_ids = input_ids[batch_idx]
#                 orig_attention_mask = attention_mask[batch_idx]
                
#                 # 🔑 修复：使用 mask 提取有效token（支持 left padding）
#                 # 对于 left padding: [pad, pad, token1, token2]
#                 # mask: [0, 0, 1, 1]
#                 # 需要提取: [token1, token2]
#                 valid_mask = orig_attention_mask.bool()
#                 prompt_tokens = orig_input_ids[valid_mask]
                
#                 # ⭐ If skipping, keep original prompt without COT
#                 if skip_this_prompt:
#                     # Keep original prompt without COT augmentation
#                     new_input_ids_list.append(orig_input_ids)
#                     new_attention_mask_list.append(orig_attention_mask)
#                     if rep_idx == 0:  # Log only once per prompt
#                         print(f"⚠️  No COT example found for prompt {orig_idx}, using original prompt without augmentation")
#                     continue
                
#                 # Get COT example for this repetition
#                 cot_text = cot_examples_for_prompt[rep_idx]
                
#                 # Decode original prompt to text（现在只包含有效token，没有padding）
#                 # 使用 skip_special_tokens=True 去掉特殊token
#                 prompt_text = self.tokenizer.decode(prompt_tokens, skip_special_tokens=True)
                
#                 # 🆕 移除聊天模板的角色标记（user, assistant等）
#                 # 这些通常以换行符分隔，我们需要清理掉
#                 prompt_text = self._remove_chat_template_roles(prompt_text)
                
#                 # ⭐ Construct augmented prompt: COT example FIRST, then the actual question
#                 # This helps the model understand that the COT is a reference example
#                 if self.add_separator:
#                     augmented_text = cot_text + self.separator + prompt_text
#                 else:
#                     augmented_text = cot_text + prompt_text
                
#                 # 📝 DEBUG: Print augmented prompt for verification
#                 # 🔑 打印所有rollout，展示不同的COT example
#                 if self.debug_print_augmented_prompts and orig_idx < self.debug_num_samples:
#                     # 获取dataset_name（如果有的话）
#                     dataset_name = "unknown"
#                     if hasattr(gen_batch, 'non_tensor_batch') and 'dataset_name' in gen_batch.non_tensor_batch:
#                         dataset_names = gen_batch.non_tensor_batch['dataset_name']
#                         if isinstance(dataset_names, np.ndarray):
#                             dataset_name = str(dataset_names[batch_idx])
#                         else:
#                             dataset_name = dataset_names[batch_idx]
                    
#                     print("\n" + "="*80)
#                     print(f"🔍 COT增强调试 - 样本 {orig_idx} | 数据集: {dataset_name} | Rollout {rep_idx}/{self.num_repeats}")
#                     print("="*80)
                    
#                     if self.debug_print_full_prompt:
#                         # 🆕 打印完整内容（不截断）
#                         print(f"\n📌 原始问题 ({len(prompt_text)} 字符):")
#                         print("-" * 80)
#                         print(prompt_text)
#                         print("-" * 80)
                        
#                         print(f"\n📚 示例COT ({len(cot_text)} 字符):")
#                         print("-" * 80)
#                         print(cot_text)
#                         print("-" * 80)
                        
#                         print(f"\n✨ 拼接后的完整Prompt ({len(augmented_text)} 字符):")
#                         print("⭐ 这就是最终喂给模型的内容！")
#                         print("-" * 80)
#                         print(augmented_text)
#                         print("-" * 80)
#                     else:
#                         # 原有的截断版本
#                         print(f"\n📌 原始问题 ({len(prompt_text)} 字符):")
#                         print(prompt_text[:200] + "..." if len(prompt_text) > 200 else prompt_text)
#                         print(f"\n📚 示例COT ({len(cot_text)} 字符):")
#                         print(cot_text[:300] + "..." if len(cot_text) > 300 else cot_text)
#                         print(f"\n✨ 拼接后的Prompt ({len(augmented_text)} 字符):")
#                         print(augmented_text[:500] + "..." if len(augmented_text) > 500 else augmented_text)
                    
#                     print("="*80 + "\n")
                
#                 # Tokenize the augmented prompt
#                 augmented_tokens = self.tokenizer.encode(augmented_text, add_special_tokens=False)
#                 augmented_input_ids = torch.tensor(augmented_tokens, dtype=torch.long, device=input_ids.device)
                
#                 # Create new attention mask
#                 augmented_attention_mask = torch.ones_like(augmented_input_ids)
                
#                 new_input_ids_list.append(augmented_input_ids)
#                 new_attention_mask_list.append(augmented_attention_mask)
        
#         # Pad all sequences to the same length
#         max_len = max(ids.shape[0] for ids in new_input_ids_list)
        
#         padded_input_ids = []
#         padded_attention_mask = []
        
#         pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        
#         for ids, mask in zip(new_input_ids_list, new_attention_mask_list):
#             # Left padding (standard for generation)
#             pad_length = max_len - ids.shape[0]
#             padded_ids = torch.cat([
#                 torch.full((pad_length,), pad_token_id, dtype=ids.dtype, device=ids.device),
#                 ids
#             ])
#             padded_mask = torch.cat([
#                 torch.zeros(pad_length, dtype=mask.dtype, device=mask.device),
#                 mask
#             ])
            
#             padded_input_ids.append(padded_ids)
#             padded_attention_mask.append(padded_mask)
        
#         # Stack into batch tensors
#         new_input_ids = torch.stack(padded_input_ids, dim=0)
#         new_attention_mask = torch.stack(padded_attention_mask, dim=0)
#         new_position_ids = compute_position_id_with_mask(new_attention_mask)
        
#         # Update the gen_batch
#         gen_batch.batch["input_ids"] = new_input_ids
#         gen_batch.batch["attention_mask"] = new_attention_mask
#         gen_batch.batch["position_ids"] = new_position_ids
        
#         return gen_batch


# def load_cot_examples_from_file(file_path: str) -> List[str]:
#     """
#     Load COT examples from a text file.
    
#     Supports:
#     - Plain text file with one example per line
#     - JSON file with a list of examples
#     - JSONL file with one example per line
    
#     Args:
#         file_path: Path to the file containing COT examples.
    
#     Returns:
#         List of COT example strings.
#     """
#     import json
    
#     if file_path.endswith('.json'):
#         with open(file_path, 'r', encoding='utf-8') as f:
#             examples = json.load(f)
#             if not isinstance(examples, list):
#                 raise ValueError(f"JSON file must contain a list, got {type(examples)}")
#             return examples
    
#     elif file_path.endswith('.jsonl'):
#         examples = []
#         with open(file_path, 'r', encoding='utf-8') as f:
#             for line in f:
#                 data = json.loads(line.strip())
#                 if isinstance(data, dict) and 'example' in data:
#                     examples.append(data['example'])
#                 elif isinstance(data, str):
#                     examples.append(data)
#                 else:
#                     examples.append(str(data))
#         return examples
    
#     else:
#         # Plain text file, one example per line
#         with open(file_path, 'r', encoding='utf-8') as f:
#             return [line.strip() for line in f if line.strip()]


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
Utility for augmenting GRPO prompts with different COT examples per rollout.
"""

import random
import logging # 🆕 导入 logging
from typing import List, Optional, Callable
import numpy as np
import torch
from verl import DataProto
from verl.utils.model import compute_position_id_with_mask

# 🆕 创建 logger
logger = logging.getLogger(__name__)


class GRPOCOTAugmenter:
    """
    Augments repeated prompts in GRPO with different COT examples.
    
    For each original prompt that gets repeated N times, this class adds
    a different COT example to each repetition.
    
    Example usage:
        augmenter = GRPOCOTAugmenter(
            cot_examples=["Example 1: ...", "Example 2: ...", "Example 3: ..."],
            tokenizer=tokenizer,
            num_repeats=4,
            sampling_strategy="random_with_replacement"
            max_prompt_length=1024  # 🆕 传入最大长度
        )
        gen_batch = augmenter.augment(gen_batch)
    """
    
    def __init__(
        self,
        cot_examples: Optional[List[str]] = None,
        cot_examples_getter: Optional[Callable] = None,
        tokenizer=None,
        num_repeats: int = 1,
        sampling_strategy: str = "random_with_replacement",
        add_separator: bool = True,
        separator: str = "\n\n",
        enable: bool = True,
        seed: Optional[int] = None,
        debug_print_augmented_prompts: bool = True,
        debug_num_samples: int = 3,
        debug_print_full_prompt: bool = False,  # 🆕 是否打印完整prompt（不截断）
        max_prompt_length: int = 2048,  # 🆕 接收 max_prompt_length
    ):
        """
        Initialize the COT augmenter.
        
        Args:
            cot_examples: List of COT example strings to choose from.
                         Can be None if cot_examples_getter is provided.
            cot_examples_getter: Optional callable that returns COT examples dynamically.
                                Format: Callable[[batch_data], List[str]]
                                This allows generating different examples per prompt.
            tokenizer: Tokenizer to encode the COT examples.
            num_repeats: Number of times each prompt is repeated in GRPO.
            sampling_strategy: How to sample COT examples:
                - "random_with_replacement": Random sampling (can repeat)
                - "random_without_replacement": Random sampling (no repeat, requires len(cot_examples) >= num_repeats)
                - "sequential": Use examples in order (循环使用)
                - "all_different_per_prompt": Each repetition gets a unique example per original prompt
            add_separator: Whether to add a separator between prompt and COT example.
            separator: The separator string (default: "\n\n").
            enable: Whether to enable COT augmentation.
            seed: Random seed for reproducibility.
            debug_print_augmented_prompts: Whether to print augmented prompts for debugging.
            debug_num_samples: Number of samples to print for debugging.
            max_prompt_length: The maximum token length allowed for a prompt.
        """
        self.cot_examples = cot_examples
        self.cot_examples_getter = cot_examples_getter
        self.tokenizer = tokenizer
        self.num_repeats = num_repeats
        self.sampling_strategy = sampling_strategy
        self.add_separator = add_separator
        self.separator = separator
        self.enable = enable
        self.debug_print_augmented_prompts = debug_print_augmented_prompts
        self.debug_num_samples = debug_num_samples
        self.debug_print_full_prompt = debug_print_full_prompt
        self.max_prompt_length = max_prompt_length  # 🆕 存储最大长度
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Validate configuration
        if self.enable:
            if self.cot_examples is None and self.cot_examples_getter is None:
                raise ValueError("Either cot_examples or cot_examples_getter must be provided when enable=True")
            
            if self.tokenizer is None:
                raise ValueError("tokenizer must be provided when enable=True")
            
            if self.sampling_strategy == "random_without_replacement" and self.cot_examples is not None:
                if len(self.cot_examples) < self.num_repeats:
                    raise ValueError(
                        f"For 'random_without_replacement' strategy, need at least {self.num_repeats} "
                        f"COT examples, but only {len(self.cot_examples)} provided."
                    )
    
    def _remove_chat_template_roles(self, text: str) -> str:
        """
        移除聊天模板中的角色标记（如 'user', 'assistant' 等）。
        
        常见的聊天模板格式：
        - "user\n问题内容\nassistant"
        - "user\n问题内容\nassistant\n"
        
        我们需要保留问题内容，去掉角色标记。
        
        Args:
            text: 包含角色标记的文本
            
        Returns:
            清理后的纯文本问题
        """
        # 移除开头的 "user" 或 "user\n"
        text = text.strip()
        
        # 按行分割
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line_stripped = line.strip()
            # 跳过纯角色标记行（user, assistant, system等）
            if line_stripped.lower() in ['user', 'assistant', 'system', '<|user|>', '<|assistant|>', '<|system|>']:
                continue
            # 保留其他内容
            if line_stripped:  # 跳过空行
                cleaned_lines.append(line)
        
        # 重新组合
        cleaned_text = '\n'.join(cleaned_lines).strip()
        
        return cleaned_text
    
    def _sample_cot_examples(self, num_samples: int, prompt_idx: Optional[int] = None) -> List[str]:
        """
        Sample COT examples based on the configured strategy.
        
        Args:
            num_samples: Number of examples to sample.
            prompt_idx: Index of the original prompt (before repetition).
        
        Returns:
            List of sampled COT example strings.
        """
        cot_pool = self.cot_examples if self.cot_examples is not None else []
        
        if self.sampling_strategy == "random_with_replacement":
            return random.choices(cot_pool, k=num_samples)
        
        elif self.sampling_strategy == "random_without_replacement":
            return random.sample(cot_pool, k=num_samples)
        
        elif self.sampling_strategy == "sequential":
            # Use examples in circular order
            result = []
            for i in range(num_samples):
                idx = (prompt_idx + i) % len(cot_pool) if prompt_idx is not None else i % len(cot_pool)
                result.append(cot_pool[idx])
            return result
        
        elif self.sampling_strategy == "all_different_per_prompt":
            # Each prompt gets its own set of examples
            if prompt_idx is not None:
                start_idx = (prompt_idx * num_samples) % len(cot_pool)
                indices = [(start_idx + i) % len(cot_pool) for i in range(num_samples)]
                return [cot_pool[i] for i in indices]
            else:
                return random.sample(cot_pool, k=min(num_samples, len(cot_pool)))
        
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
    
    def augment(self, gen_batch: DataProto) -> DataProto:
        """
        Augment the repeated prompts with different COT examples.
        
        Args:
            gen_batch: DataProto after repeat() has been called.
                      Assumes batch size = original_batch_size * num_repeats
        
        Returns:
            DataProto with augmented prompts.
        """
        if not self.enable:
            return gen_batch
        
        batch_size = gen_batch.batch["input_ids"].shape[0]
        original_batch_size = batch_size // self.num_repeats
        
        if batch_size % self.num_repeats != 0:
            raise ValueError(
                f"Batch size {batch_size} is not divisible by num_repeats {self.num_repeats}. "
                "Make sure repeat() was called before augment()."
            )
        
        # Extract original input_ids and attention_mask
        input_ids = gen_batch.batch["input_ids"]  # Shape: (batch_size, seq_len)
        attention_mask = gen_batch.batch["attention_mask"]  # Shape: (batch_size, seq_len)
        
        new_input_ids_list = []
        new_attention_mask_list = []
        new_position_ids_list = []
        
        # Process each original prompt and its repetitions
        for orig_idx in range(original_batch_size):
            # Get COT examples for this prompt's repetitions
            if self.cot_examples_getter is not None:
                # Use dynamic getter
                cot_examples_for_prompt = self.cot_examples_getter(gen_batch, orig_idx, self.num_repeats)
            else:
                # Use static pool
                cot_examples_for_prompt = self._sample_cot_examples(self.num_repeats, prompt_idx=orig_idx)
            
            # ⭐ Check if COT matching failed (empty list indicates skip)
            skip_this_prompt = (len(cot_examples_for_prompt) == 0)
            
            # Process each repetition
            for rep_idx in range(self.num_repeats):
                batch_idx = orig_idx * self.num_repeats + rep_idx
                
                # Get the original prompt tokens
                orig_input_ids = input_ids[batch_idx]
                orig_attention_mask = attention_mask[batch_idx]
                
                # 🔑 修复：使用 mask 提取有效token（支持 left padding）
                valid_mask = orig_attention_mask.bool()
                prompt_tokens = orig_input_ids[valid_mask]
                
                # Decode original prompt to text（现在只包含有效token，没有padding）
                # 使用 skip_special_tokens=True 去掉特殊token
                prompt_text = self.tokenizer.decode(prompt_tokens, skip_special_tokens=True)
                
                # 🆕 移除聊天模板的角色标记（user, assistant等）
                prompt_text = self._remove_chat_template_roles(prompt_text)
                
                # 🆕 --- 实施你的降级逻辑 (第 2 步 和 第 3 步) ---
                
                augmented_text = ""
                
                # ⭐ If skipping, keep original prompt without COT
                if skip_this_prompt:
                    if rep_idx == 0:  # Log only once per prompt
                        logger.debug(f"No COT example found for prompt {orig_idx}, using original prompt without augmentation")
                    augmented_text = prompt_text
                else:
                    # Get COT example for this repetition
                    cot_text = cot_examples_for_prompt[rep_idx]
                    
                    # ⭐ 步骤 1 & 2: 尝试拼接 CoT 并检查长度
                    if self.add_separator:
                        potential_text = cot_text + self.separator + prompt_text
                    else:
                        potential_text = cot_text + prompt_text
                    
                    potential_tokens = self.tokenizer.encode(potential_text, add_special_tokens=False)
                    
                    if len(potential_tokens) > self.max_prompt_length:
                        # 🆕 步骤 3: 降级 F1 (移除 CoT)
                        logger.warning(
                            f"样本 {orig_idx} (Rollout {rep_idx}) 的 CoT + Prompt 长度 ({len(potential_tokens)}) "
                            f"超过了 max_prompt_length ({self.max_prompt_length})。"
                            f"将移除 CoT 并降级到仅使用原 Prompt。"
                        )
                        augmented_text = prompt_text
                    else:
                        # 长度在范围内，使用增强后的 prompt
                        augmented_text = potential_text
                
                # 📝 DEBUG: Print augmented prompt for verification
                # 🔑 打印所有rollout，展示不同的COT example
                if self.debug_print_augmented_prompts and orig_idx < self.debug_num_samples:
                    # 获取dataset_name（如果有的话）
                    dataset_name = "unknown"
                    if hasattr(gen_batch, 'non_tensor_batch') and 'dataset_name' in gen_batch.non_tensor_batch:
                        dataset_names = gen_batch.non_tensor_batch['dataset_name']
                        if isinstance(dataset_names, np.ndarray):
                            dataset_name = str(dataset_names[batch_idx])
                        else:
                            dataset_name = dataset_names[batch_idx]
                    
                    print("\n" + "="*80)
                    print(f"🔍 COT增强调试 - 样本 {orig_idx} | 数据集: {dataset_name} | Rollout {rep_idx}/{self.num_repeats}")
                    print("="*80)
                    
                    if self.debug_print_full_prompt:
                        # 🆕 打印完整内容（不截断）
                        # 仅在未跳过时打印 CoT
                        if not skip_this_prompt:
                            print(f"\n📚 示例COT ({len(cot_text)} 字符):")
                            print("-" * 80)
                            print(cot_text)
                            print("-" * 80)
                        
                        print(f"\n✨ 拼接后的完整Prompt ({len(augmented_text)} 字符):")
                        print("⭐ 这就是最终喂给模型的内容！")
                        print("-" * 80)
                        print(augmented_text)
                        print("-" * 80)
                    else:
                        # 原有的截断版本
                        if not skip_this_prompt:
                            print(f"\n📚 示例COT ({len(cot_text)} 字符):")
                            print(cot_text[:300] + "..." if len(cot_text) > 300 else cot_text)
                        
                        print(f"\n✨ 拼接后的Prompt ({len(augmented_text)} 字符):")
                        print(augmented_text[:500] + "..." if len(augmented_text) > 500 else augmented_text)
                    
                    print("="*80 + "\n")
                
                # 🔧 修复：不要重新应用 chat template！
                # 原因：multi_dataset_with_cot.py 中已经使用硬编码模板构建了完整的 prompt
                # 原始 prompt_text 已经包含：<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n
                # COT 拼接后的 augmented_text 也已经包含完整的 chat template
                # 如果再用 apply_chat_template，会导致 system prompt 重复出现！
                
                # 直接 tokenize augmented_text，不添加任何 special tokens（因为已经包含在 template 中）
                augmented_tokens = self.tokenizer.encode(
                    augmented_text,
                    add_special_tokens=False  # ← 关键：原始 prompt 已包含所有 special tokens
                )
                
                # 🆕 增加一个最终的安全检查，以防万一
                if len(augmented_tokens) > self.max_prompt_length:
                     logger.error(
                        f"样本 {orig_idx} (Rollout {rep_idx}) 即使在降级后长度 ({len(augmented_tokens)}) "
                        f"仍然超过了 max_prompt_length ({self.max_prompt_length})！"
                        f"将强制从右侧截断。这不应该发生，请检查 multi_dataset_with_cot.py"
                    )
                     augmented_tokens = augmented_tokens[:self.max_prompt_length]

                augmented_input_ids = torch.tensor(augmented_tokens, dtype=torch.long, device=input_ids.device)
                
                # Create new attention mask
                augmented_attention_mask = torch.ones_like(augmented_input_ids)
                
                new_input_ids_list.append(augmented_input_ids)
                new_attention_mask_list.append(augmented_attention_mask)
        
        # Pad all sequences to the same length
        max_len = max(ids.shape[0] for ids in new_input_ids_list)
        
        padded_input_ids = []
        padded_attention_mask = []
        
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        
        for ids, mask in zip(new_input_ids_list, new_attention_mask_list):
            # Left padding (standard for generation)
            pad_length = max_len - ids.shape[0]
            padded_ids = torch.cat([
                torch.full((pad_length,), pad_token_id, dtype=ids.dtype, device=ids.device),
                ids
            ])
            padded_mask = torch.cat([
                torch.zeros(pad_length, dtype=mask.dtype, device=mask.device),
                mask
            ])
            
            padded_input_ids.append(padded_ids)
            padded_attention_mask.append(padded_mask)
        
        # Stack into batch tensors
        new_input_ids = torch.stack(padded_input_ids, dim=0)
        new_attention_mask = torch.stack(padded_attention_mask, dim=0)
        new_position_ids = compute_position_id_with_mask(new_attention_mask)
        
        # Update the gen_batch
        gen_batch.batch["input_ids"] = new_input_ids
        gen_batch.batch["attention_mask"] = new_attention_mask
        gen_batch.batch["position_ids"] = new_position_ids
        
        return gen_batch


def load_cot_examples_from_file(file_path: str) -> List[str]:
    """
    Load COT examples from a text file.
    
    Supports:
    - Plain text file with one example per line
    - JSON file with a list of examples
    - JSONL file with one example per line
    
    Args:
        file_path: Path to the file containing COT examples.
    
    Returns:
        List of COT example strings.
    """
    import json
    
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            examples = json.load(f)
            if not isinstance(examples, list):
                raise ValueError(f"JSON file must contain a list, got {type(examples)}")
            return examples
    
    elif file_path.endswith('.jsonl'):
        examples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                if isinstance(data, dict) and 'example' in data:
                    examples.append(data['example'])
                elif isinstance(data, str):
                    examples.append(data)
                else:
                    examples.append(str(data))
        return examples
    
    else:
        # Plain text file, one example per line
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
