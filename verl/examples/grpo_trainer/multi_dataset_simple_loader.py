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
# 多数据集COT加载器（适配实际数据格式）。

# 数据格式：
# - 训练数据: parquet，问题在 extra_info['question']
# - COT数据: JSONL，包含 selected_cots 字段

# 匹配策略：
# 1. 归一化匹配（去除空格、标点、大小写）
# 2. 精确匹配
# 3. 失败则跳过
# """

# import json
# import re
# from typing import Dict, List, Optional
# import numpy as np


# def normalize_question(question: str) -> str:
#     """归一化问题文本。"""
#     text = question.lower()
#     text = re.sub(r'\s+', ' ', text)
#     text = text.rstrip('.,!?;:')
#     return text.strip()


# class MultiDatasetSimpleCOTLoader:
#     """
#     多数据集COT加载器（简单匹配策略）。
    
#     每个数据集有自己的COT文件，根据dataset_name自动选择。
#     """
    
#     def __init__(
#         self,
#         cot_file_mapping: Dict[str, str],
#         cot_format_template: str = "Here's a similar example:\n\nQuestion: {question}\n\nStep-by-step solution:\n{rationale}\n\nFinal Answer: {final_answer}\n\nNow, let's solve the current problem:",
#         use_full_cot: bool = True,
#         skip_on_mismatch: bool = True,
#         verbose: bool = True,
#     ):
#         """
#         初始化多数据集COT加载器。
        
#         Args:
#             cot_file_mapping: 数据集名称 → COT文件路径的映射
#                 Example: {
#                     "gsm8k": "/path/to/gsm8k_cot.jsonl",
#                     "math": "/path/to/math_cot.jsonl"
#                 }
#             cot_format_template: COT格式化模板
#             use_full_cot: 是否使用完整COT
#             skip_on_mismatch: 匹配失败时是否跳过
#             verbose: 是否打印详细日志
#         """
#         self.cot_file_mapping = cot_file_mapping
#         self.cot_format_template = cot_format_template
#         self.use_full_cot = use_full_cot
#         self.skip_on_mismatch = skip_on_mismatch
#         self.verbose = verbose
        
#         # 每个数据集的COT数据: dataset_name -> {question -> [COT examples]}
#         self.dataset_cot_data: Dict[str, Dict[str, List[str]]] = {}
        
#         # 每个数据集的归一化映射: dataset_name -> {normalized -> original}
#         self.dataset_normalized_map: Dict[str, Dict[str, str]] = {}
        
#         # 统计信息
#         self.stats = {
#             "normalized_matches": 0,
#             "exact_matches": 0,
#             "failed_matches": 0,
#             "skipped": 0,
#         }
        
#         # 加载所有数据集的COT数据
#         self._load_all_cot_data()
        
#         print(f"✓ 多数据集COT加载器初始化完成")
#         print(f"  数据集数量: {len(self.dataset_cot_data)}")
#         for ds_name, cot_data in self.dataset_cot_data.items():
#             print(f"  - {ds_name}: {len(cot_data)} 个问题")
    
#     def _load_all_cot_data(self):
#         """加载所有数据集的COT数据。"""
#         for dataset_name, cot_file_path in self.cot_file_mapping.items():
#             print(f"\n加载 {dataset_name} 的COT数据: {cot_file_path}")
            
#             cot_data = {}
#             normalized_map = {}
            
#             with open(cot_file_path, 'r', encoding='utf-8') as f:
#                 for line in f:
#                     data = json.loads(line.strip())
                    
#                     original_question = data["question"]
#                     normalized = normalize_question(original_question)
                    
#                     # 建立归一化映射
#                     normalized_map[normalized] = original_question
                    
#                     # 提取和格式化COT例子
#                     selected_cots = data.get("selected_cots", [])
#                     formatted_cots = []
                    
#                     for cot in selected_cots:
#                         if self.use_full_cot:
#                             formatted_cot = self.cot_format_template.format(
#                                 question=cot.get("question", ""),
#                                 rationale=cot.get("rationale", ""),
#                                 final_answer=cot.get("final_answer", "")
#                             )
#                         else:
#                             formatted_cot = cot.get("rationale", "")
                        
#                         formatted_cots.append(formatted_cot)
                    
#                     cot_data[original_question] = formatted_cots
            
#             self.dataset_cot_data[dataset_name] = cot_data
#             self.dataset_normalized_map[dataset_name] = normalized_map
            
#             print(f"  ✓ 加载了 {len(cot_data)} 个问题")
    
#     def get_cot_examples(
#         self,
#         dataset_name: str,
#         question: str,
#         num_examples: Optional[int] = None
#     ) -> List[str]:
#         """
#         获取指定数据集的COT例子。
        
#         Args:
#             dataset_name: 数据集名称
#             question: 问题文本
#             num_examples: 需要的例子数量
        
#         Returns:
#             COT例子列表（失败则返回空列表）
#         """
#         # 检查数据集是否存在
#         if dataset_name not in self.dataset_cot_data:
#             print(f"❌ 未知数据集: {dataset_name}")
#             print(f"   可用数据集: {list(self.dataset_cot_data.keys())}")
#             return []
        
#         cot_data = self.dataset_cot_data[dataset_name]
#         normalized_map = self.dataset_normalized_map[dataset_name]
        
#         matched_question = None
#         match_type = None
        
#         # 策略1: 归一化匹配
#         normalized_query = normalize_question(question)
#         if normalized_query in normalized_map:
#             matched_question = normalized_map[normalized_query]
#             match_type = "normalized"
#             self.stats["normalized_matches"] += 1
        
#         # 策略2: 精确匹配
#         if matched_question is None:
#             if question in cot_data:
#                 matched_question = question
#                 match_type = "exact"
#                 self.stats["exact_matches"] += 1
        
#         # 匹配失败
#         if matched_question is None:
#             self.stats["failed_matches"] += 1
            
#             if self.verbose:
#                 print(f"❌ [{dataset_name}] 匹配失败 - 跳过:")
#                 print(f"   问题: {question[:100]}...")
            
#             if self.skip_on_mismatch:
#                 self.stats["skipped"] += 1
#                 return []
#             else:
#                 return []
        
#         # 匹配成功
#         if self.verbose:
#             if match_type == "normalized":
#                 print(f"✓ [{dataset_name}] 归一化匹配成功")
#             elif match_type == "exact":
#                 print(f"✓ [{dataset_name}] 精确匹配成功")
        
#         # 获取COT例子
#         cot_examples = cot_data[matched_question]
        
#         # 返回指定数量
#         if num_examples is None:
#             return cot_examples
#         elif num_examples <= len(cot_examples):
#             return cot_examples[:num_examples]
#         else:
#             # 循环使用
#             result = []
#             for i in range(num_examples):
#                 result.append(cot_examples[i % len(cot_examples)])
#             return result
    
#     def print_stats(self):
#         """打印统计信息。"""
#         total = sum(v for k, v in self.stats.items() if k != "skipped")
        
#         if total == 0:
#             print("还没有匹配尝试")
#             return
        
#         print("\n" + "=" * 60)
#         print("COT匹配统计")
#         print("=" * 60)
#         print(f"归一化匹配: {self.stats['normalized_matches']:5d} / {total} ({self.stats['normalized_matches']/total*100:5.2f}%)")
#         print(f"精确匹配:   {self.stats['exact_matches']:5d} / {total} ({self.stats['exact_matches']/total*100:5.2f}%)")
#         print(f"匹配失败:   {self.stats['failed_matches']:5d} / {total} ({self.stats['failed_matches']/total*100:5.2f}%)")
#         if self.skip_on_mismatch:
#             print(f"跳过数据:   {self.stats['skipped']:5d}")
#         print("=" * 60)
        
#         success_rate = (total - self.stats['failed_matches']) / total * 100
#         print(f"总体成功率: {success_rate:.2f}%")
#         print("=" * 60 + "\n")


# # 全局加载器
# _global_multi_simple_cot_loader: Optional[MultiDatasetSimpleCOTLoader] = None


# def initialize_multi_dataset_simple_cot_loader(
#     cot_file_mapping: Dict[str, str],
#     cot_format_template: str = "Here's a similar example:\n\nQuestion: {question}\n\nLet's solve it step by step:\n{rationale}\n\nFinal Answer: {final_answer}\n\nNow, let's solve the current problem:",
#     use_full_cot: bool = True,
#     skip_on_mismatch: bool = True,
#     verbose: bool = True,
# ):
#     """初始化全局多数据集COT加载器。"""
#     global _global_multi_simple_cot_loader
#     _global_multi_simple_cot_loader = MultiDatasetSimpleCOTLoader(
#         cot_file_mapping=cot_file_mapping,
#         cot_format_template=cot_format_template,
#         use_full_cot=use_full_cot,
#         skip_on_mismatch=skip_on_mismatch,
#         verbose=verbose,
#     )
#     return _global_multi_simple_cot_loader


# def get_multi_dataset_simple_cot_examples(batch, prompt_idx: int, num_repeats: int, tokenizer=None) -> List[str]:
#     """
#     多数据集COT获取函数（简单匹配）。
    
#     Args:
#         batch: DataProto batch
#         prompt_idx: 原始prompt索引
#         num_repeats: rollout次数
#         tokenizer: Tokenizer
    
#     Returns:
#         COT例子列表
#     """
#     global _global_multi_simple_cot_loader
    
#     if _global_multi_simple_cot_loader is None:
#         raise RuntimeError("多数据集COT加载器未初始化")
    
#     # 获取dataset_name
#     dataset_name = None
#     if "dataset_name" in batch.non_tensor_batch:
#         dataset_names = batch.non_tensor_batch["dataset_name"]
#         if isinstance(dataset_names, np.ndarray):
#             dataset_name = str(dataset_names[prompt_idx * num_repeats])
#         else:
#             dataset_name = dataset_names[prompt_idx * num_repeats]
    
#     # ⭐ 回退方案：如果只有一个数据集，使用唯一的 dataset_name
#     if dataset_name is None and len(_global_multi_simple_cot_loader.dataset_cot_data) == 1:
#         dataset_name = list(_global_multi_simple_cot_loader.dataset_cot_data.keys())[0]
#         if prompt_idx == 0:
#             print(f"⚠️  batch中没有dataset_name字段，自动使用唯一数据集: {dataset_name}")
    
#     if dataset_name is None:
#         # 🔍 调试：打印 batch 中实际包含的字段
#         if prompt_idx == 0:  # 只打印一次
#             print("❌ 错误: batch中没有dataset_name字段，且配置了多个数据集")
#             print(f"🔍 调试信息 - batch.non_tensor_batch 中的字段:")
#             print(f"   可用字段: {list(batch.non_tensor_batch.keys())}")
#             print(f"   配置的数据集: {list(_global_multi_simple_cot_loader.dataset_cot_data.keys())}")
#             print(f"\n💡 解决方案:")
#             print(f"   1. 使用 MultiDatasetWithCOT 数据集类")
#             print(f"   2. 或者，如果只有单个数据集，只配置一个 dataset_cot_mapping")
#             if hasattr(batch, 'meta_info'):
#                 print(f"   meta_info 字段: {list(batch.meta_info.keys())}")
#         return []
    
#     # 获取question
#     question = None
#     if "question" in batch.non_tensor_batch:
#         questions = batch.non_tensor_batch["question"]
#         if isinstance(questions, np.ndarray):
#             question = str(questions[prompt_idx * num_repeats])
#         else:
#             question = questions[prompt_idx * num_repeats]
    
#     if question is None and tokenizer is not None:
#         try:
#             input_ids = batch.batch["input_ids"][prompt_idx * num_repeats]
#             attention_mask = batch.batch["attention_mask"][prompt_idx * num_repeats]
#             prompt_length = attention_mask.sum().item()
#             prompt_tokens = input_ids[:prompt_length]
#             question = tokenizer.decode(prompt_tokens, skip_special_tokens=True)
#         except Exception as e:
#             print(f"⚠️  警告: 无法解码问题: {e}")
    
#     if question is None:
#         print("❌ 错误: 无法获取问题文本")
#         return []
    
#     # 获取COT例子
#     try:
#         cot_examples = _global_multi_simple_cot_loader.get_cot_examples(
#             dataset_name=dataset_name,
#             question=question,
#             num_examples=num_repeats
#         )
        
#         if len(cot_examples) == 0:
#             return []
        
#         if len(cot_examples) < num_repeats:
#             while len(cot_examples) < num_repeats:
#                 cot_examples.append(cot_examples[len(cot_examples) % len(cot_examples)])
        
#         return cot_examples[:num_repeats]
    
#     except Exception as e:
#         print(f"❌ 错误: 获取COT失败: {e}")
#         return []



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
多数据集COT加载器（适配实际数据格式）。

数据格式：
- 训练数据: parquet，问题在 extra_info['question']
- COT数据: JSONL，包含 selected_cots 字段

匹配策略：
1. 归一化匹配（去除空格、标点、大小写）
2. 精确匹配
3. 失败则跳过
"""

import json
import re
from typing import Dict, List, Optional
import numpy as np


def normalize_question(question: str) -> str:
    """归一化问题文本。"""
    text = question.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.rstrip('.,!?;:')
    return text.strip()


class MultiDatasetSimpleCOTLoader:
    """
    多数据集COT加载器（简单匹配策略）。
    
    每个数据集有自己的COT文件，根据dataset_name自动选择。
    """
    
    def __init__(
        self,
        cot_file_mapping: Dict[str, str],
        cot_format_template: str = "Here is a reference example that demonstrates the problem-solving approach:\n\n<Example>\nQuestion: {question}\n\nStep-by-step Solution:\n{rationale}\n\n#### {final_answer}\n</Example>\n\nNow, please solve the following problem using similar reasoning:",
        use_full_cot: bool = True,
        skip_on_mismatch: bool = True,
        verbose: bool = True,
    ):
        """
        初始化多数据集COT加载器。
        
        Args:
            cot_file_mapping: 数据集名称 → COT文件路径的映射
                Example: {
                    "gsm8k": "/path/to/gsm8k_cot.jsonl",
                    "math": "/path/to/math_cot.jsonl"
                }
            cot_format_template: COT格式化模板
            use_full_cot: 是否使用完整COT
            skip_on_mismatch: 匹配失败时是否跳过
            verbose: 是否打印详细日志
        """
        self.cot_file_mapping = cot_file_mapping
        self.cot_format_template = cot_format_template
        self.use_full_cot = use_full_cot
        self.skip_on_mismatch = skip_on_mismatch
        self.verbose = verbose
        
        # 每个数据集的COT数据: dataset_name -> {question -> [COT examples]}
        self.dataset_cot_data: Dict[str, Dict[str, List[str]]] = {}
        
        # 每个数据集的归一化映射: dataset_name -> {normalized -> original}
        self.dataset_normalized_map: Dict[str, Dict[str, str]] = {}
        
        # 统计信息
        self.stats = {
            "normalized_matches": 0,
            "exact_matches": 0,
            "failed_matches": 0,
            "skipped": 0,
        }
        
        # 加载所有数据集的COT数据
        self._load_all_cot_data()
        
        print(f"✓ 多数据集COT加载器初始化完成")
        print(f"  数据集数量: {len(self.dataset_cot_data)}")
        for ds_name, cot_data in self.dataset_cot_data.items():
            print(f"  - {ds_name}: {len(cot_data)} 个问题")
    
    def _load_all_cot_data(self):
        """加载所有数据集的COT数据。"""
        for dataset_name, cot_file_path in self.cot_file_mapping.items():
            print(f"\n加载 {dataset_name} 的COT数据: {cot_file_path}")
            
            cot_data = {}
            normalized_map = {}
            
            with open(cot_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    
                    original_question = data["question"]
                    normalized = normalize_question(original_question)
                    
                    # 建立归一化映射
                    normalized_map[normalized] = original_question
                    
                    # 提取和格式化COT例子
                    selected_cots = data.get("selected_cots", [])
                    formatted_cots = []
                    
                    for cot in selected_cots:
                        if self.use_full_cot:
                            formatted_cot = self.cot_format_template.format(
                                question=cot.get("question", ""),
                                rationale=cot.get("rationale", ""),
                                final_answer=cot.get("final_answer", "")
                            )
                        else:
                            formatted_cot = cot.get("rationale", "")
                        
                        formatted_cots.append(formatted_cot)
                    
                    cot_data[original_question] = formatted_cots
            
            self.dataset_cot_data[dataset_name] = cot_data
            self.dataset_normalized_map[dataset_name] = normalized_map
            
            print(f"  ✓ 加载了 {len(cot_data)} 个问题")
    
    def get_cot_examples(
        self,
        dataset_name: str,
        question: str,
        num_examples: Optional[int] = None
    ) -> List[str]:
        """
        获取指定数据集的COT例子。
        
        Args:
            dataset_name: 数据集名称
            question: 问题文本
            num_examples: 需要的例子数量
        
        Returns:
            COT例子列表（失败则返回空列表）
        """
        # 检查数据集是否存在
        if dataset_name not in self.dataset_cot_data:
            # print(f"❌ 未知数据集: {dataset_name}")
            # print(f"   可用数据集: {list(self.dataset_cot_data.keys())}")
            return []
        
        cot_data = self.dataset_cot_data[dataset_name]
        normalized_map = self.dataset_normalized_map[dataset_name]
        
        matched_question = None
        match_type = None
        
        # 策略1: 归一化匹配
        normalized_query = normalize_question(question)
        if normalized_query in normalized_map:
            matched_question = normalized_map[normalized_query]
            match_type = "normalized"
            self.stats["normalized_matches"] += 1
        
        # 策略2: 精确匹配
        if matched_question is None:
            if question in cot_data:
                matched_question = question
                match_type = "exact"
                self.stats["exact_matches"] += 1
        
        # 匹配失败
        if matched_question is None:
            self.stats["failed_matches"] += 1
            
            if self.verbose:
                print(f"❌ [{dataset_name}] 匹配失败 - 跳过:")
                print(f"   问题: {question[:100]}...")
            
            if self.skip_on_mismatch:
                self.stats["skipped"] += 1
                return []
            else:
                return []
        
        # 匹配成功
        if self.verbose:
            if match_type == "normalized":
                print(f"✓ [{dataset_name}] 归一化匹配成功")
            elif match_type == "exact":
                print(f"✓ [{dataset_name}] 精确匹配成功")
        
        # 获取COT例子
        cot_examples = cot_data[matched_question]
        
        # 返回指定数量
        if num_examples is None:
            return cot_examples
        elif num_examples <= len(cot_examples):
            return cot_examples[:num_examples]
        else:
            # 循环使用
            result = []
            for i in range(num_examples):
                result.append(cot_examples[i % len(cot_examples)])
            return result
    
    def print_stats(self):
        """打印统计信息。"""
        total = sum(v for k, v in self.stats.items() if k != "skipped")
        
        if total == 0:
            print("还没有匹配尝试")
            return
        
        print("\n" + "=" * 60)
        print("COT匹配统计")
        print("=" * 60)
        print(f"归一化匹配: {self.stats['normalized_matches']:5d} / {total} ({self.stats['normalized_matches']/total*100:5.2f}%)")
        print(f"精确匹配:   {self.stats['exact_matches']:5d} / {total} ({self.stats['exact_matches']/total*100:5.2f}%)")
        print(f"匹配失败:   {self.stats['failed_matches']:5d} / {total} ({self.stats['failed_matches']/total*100:5.2f}%)")
        if self.skip_on_mismatch:
            print(f"跳过数据:   {self.stats['skipped']:5d}")
        print("=" * 60)
        
        success_rate = (total - self.stats['failed_matches']) / total * 100
        print(f"总体成功率: {success_rate:.2f}%")
        print("=" * 60 + "\n")


# 全局加载器
_global_multi_simple_cot_loader: Optional[MultiDatasetSimpleCOTLoader] = None


def initialize_multi_dataset_simple_cot_loader(
    cot_file_mapping: Dict[str, str],
    cot_format_template: str = "Here is a reference example that demonstrates the problem-solving approach:\n\n<Example>\nQuestion: {question}\n\nStep-by-step Solution:\n{rationale}\n\n#### {final_answer}\n</Example>\n\nNow, please solve the following problem using similar reasoning:",
    use_full_cot: bool = True,
    skip_on_mismatch: bool = True,
    verbose: bool = True,
):
    """初始化全局多数据集COT加载器。"""
    global _global_multi_simple_cot_loader
    _global_multi_simple_cot_loader = MultiDatasetSimpleCOTLoader(
        cot_file_mapping=cot_file_mapping,
        cot_format_template=cot_format_template,
        use_full_cot=use_full_cot,
        skip_on_mismatch=skip_on_mismatch,
        verbose=verbose,
    )
    return _global_multi_simple_cot_loader


def get_multi_dataset_simple_cot_examples(batch, prompt_idx: int, num_repeats: int, tokenizer=None) -> List[str]:
    """
    多数据集COT获取函数（简单匹配）。
    
    Args:
        batch: DataProto batch
        prompt_idx: 原始prompt索引
        num_repeats: rollout次数
        tokenizer: Tokenizer
    
    Returns:
        COT例子列表
    """
    global _global_multi_simple_cot_loader
    
    if _global_multi_simple_cot_loader is None:
        raise RuntimeError("多数据集COT加载器未初始化")
    
    # 获取dataset_name
    dataset_name = None
    if "dataset_name" in batch.non_tensor_batch:
        dataset_names = batch.non_tensor_batch["dataset_name"]
        if isinstance(dataset_names, np.ndarray):
            dataset_name = str(dataset_names[prompt_idx * num_repeats])
        else:
            dataset_name = dataset_names[prompt_idx * num_repeats]
    
    # ⭐ 回退方案：如果只有一个数据集，使用唯一的 dataset_name
    if dataset_name is None and len(_global_multi_simple_cot_loader.dataset_cot_data) == 1:
        dataset_name = list(_global_multi_simple_cot_loader.dataset_cot_data.keys())[0]
        if prompt_idx == 0:
            print(f"⚠️  batch中没有dataset_name字段，自动使用唯一数据集: {dataset_name}")
    
    if dataset_name is None:
        # 🔍 调试：打印 batch 中实际包含的字段
        if prompt_idx == 0:  # 只打印一次
            print("❌ 错误: batch中没有dataset_name字段，且配置了多个数据集")
            print(f"🔍 调试信息 - batch.non_tensor_batch 中的字段:")
            print(f"   可用字段: {list(batch.non_tensor_batch.keys())}")
            print(f"   配置的数据集: {list(_global_multi_simple_cot_loader.dataset_cot_data.keys())}")
            print(f"\n💡 解决方案:")
            print(f"   1. 使用 MultiDatasetWithCOT 数据集类")
            print(f"   2. 或者，如果只有单个数据集，只配置一个 dataset_cot_mapping")
            if hasattr(batch, 'meta_info'):
                print(f"   meta_info 字段: {list(batch.meta_info.keys())}")
        return []
    
    # 获取question
    question = None
    if "question" in batch.non_tensor_batch:
        questions = batch.non_tensor_batch["question"]
        if isinstance(questions, np.ndarray):
            question = str(questions[prompt_idx * num_repeats])
        else:
            question = questions[prompt_idx * num_repeats]
    
    if question is None and tokenizer is not None:
        try:
            input_ids = batch.batch["input_ids"][prompt_idx * num_repeats]
            attention_mask = batch.batch["attention_mask"][prompt_idx * num_repeats]
            prompt_length = attention_mask.sum().item()
            prompt_tokens = input_ids[:prompt_length]
            question = tokenizer.decode(prompt_tokens, skip_special_tokens=True)
        except Exception as e:
            print(f"⚠️  警告: 无法解码问题: {e}")
    
    if question is None:
        print("❌ 错误: 无法获取问题文本")
        return []
    
    # 获取COT例子
    try:
        cot_examples = _global_multi_simple_cot_loader.get_cot_examples(
            dataset_name=dataset_name,
            question=question,
            num_examples=num_repeats
        )
        
        if len(cot_examples) == 0:
            return []
        
        if len(cot_examples) < num_repeats:
            while len(cot_examples) < num_repeats:
                cot_examples.append(cot_examples[len(cot_examples) % len(cot_examples)])
        
        return cot_examples[:num_repeats]
    
    except Exception as e:
        print(f"❌ 错误: 获取COT失败: {e}")
        return []

