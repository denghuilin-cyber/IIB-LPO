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
简单匹配COT加载器：只使用归一化匹配和精确匹配。

匹配策略：
1. 先尝试归一化匹配（去除空格、标点、大小写）
2. 失败则尝试精确匹配
3. 都失败则跳过数据

不使用相似度计算的模糊匹配。
"""

import json
import re
from typing import Dict, List, Optional, Tuple
import numpy as np


def normalize_question(question: str) -> str:
    """
    归一化问题文本（方法2）。
    
    处理步骤:
    1. 转小写
    2. 去除多余空格
    3. 去除末尾标点
    4. 去除首尾空格
    
    Args:
        question: 原始问题文本
    
    Returns:
        归一化后的问题文本
    """
    text = question.lower()
    text = re.sub(r'\s+', ' ', text)  # 多个空格 → 单个空格
    text = text.rstrip('.,!?;:')       # 去除末尾标点
    text = text.strip()                # 去除首尾空格
    return text


class SimpleMatchCOTLoader:
    """
    简单匹配策略的COT加载器。
    
    匹配策略（按顺序尝试）:
    1. 归一化匹配: 归一化后精确比较
    2. 精确匹配: 原始文本完全相同
    3. 失败: 跳过数据
    
    不使用模糊匹配（相似度计算）。
    """
    
    def __init__(
        self,
        cot_file_path: str,
        cot_format_template: str = "Here's a similar example:\n\nQuestion: {question}\n\nStep-by-step solution:\n{rationale}\n\nFinal Answer: {final_answer}\n\nNow, let's solve the current problem:",
        use_full_cot: bool = True,
        skip_on_mismatch: bool = True,
        verbose: bool = True,
    ):
        """
        初始化简单匹配COT加载器。
        
        Args:
            cot_file_path: COT JSONL文件路径
            cot_format_template: COT格式化模板
            use_full_cot: 是否使用完整COT
            skip_on_mismatch: 匹配失败时是否跳过
            verbose: 是否打印详细日志
        """
        self.cot_file_path = cot_file_path
        self.cot_format_template = cot_format_template
        self.use_full_cot = use_full_cot
        self.skip_on_mismatch = skip_on_mismatch
        self.verbose = verbose
        
        # 存储COT数据
        # 原始question → [COT examples]
        self.cot_data: Dict[str, List[str]] = {}
        
        # 归一化question → 原始question
        self.normalized_to_original: Dict[str, str] = {}
        
        # 统计信息
        self.stats = {
            "normalized_matches": 0,    # 归一化匹配成功
            "exact_matches": 0,          # 精确匹配成功
            "failed_matches": 0,         # 完全失败
            "skipped": 0,                # 跳过的数据
        }
        
        # 加载COT数据
        self._load_cot_data()
        
        print(f"✓ 加载了 {len(self.cot_data)} 个问题的COT数据")
        print(f"  匹配策略: 归一化匹配 → 精确匹配")
        print(f"  跳过失败: {skip_on_mismatch}")
    
    def _load_cot_data(self):
        """从JSONL文件加载COT数据。"""
        with open(self.cot_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                
                original_question = data["question"]
                
                # 建立归一化映射
                normalized = normalize_question(original_question)
                self.normalized_to_original[normalized] = original_question
                
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
                
                self.cot_data[original_question] = formatted_cots
    
    def get_cot_examples(
        self,
        question: str,
        num_examples: Optional[int] = None
    ) -> List[str]:
        """
        获取COT例子（归一化匹配 → 精确匹配）。
        
        匹配顺序:
        1. 归一化匹配
        2. 精确匹配
        3. 失败 → 跳过
        
        Args:
            question: 训练数据中的问题文本
            num_examples: 需要返回的例子数量
        
        Returns:
            COT例子列表（如果匹配失败，返回空列表）
        """
        matched_question = None
        match_type = None
        
        # 策略1: 归一化匹配
        normalized_query = normalize_question(question)
        if normalized_query in self.normalized_to_original:
            matched_question = self.normalized_to_original[normalized_query]
            match_type = "normalized"
            self.stats["normalized_matches"] += 1
        
        # 策略2: 精确匹配
        if matched_question is None:
            if question in self.cot_data:
                matched_question = question
                match_type = "exact"
                self.stats["exact_matches"] += 1
        
        # 所有策略都失败
        if matched_question is None:
            self.stats["failed_matches"] += 1
            
            if self.verbose:
                print(f"❌ 匹配失败 - 跳过数据:")
                print(f"   问题: {question[:100]}...")
            
            if self.skip_on_mismatch:
                self.stats["skipped"] += 1
                return []
            else:
                return []
        
        # 匹配成功 - 打印日志
        if self.verbose:
            if match_type == "normalized":
                print(f"✓ 归一化匹配成功")
            elif match_type == "exact":
                print(f"✓ 精确匹配成功")
        
        # 获取COT例子
        cot_examples = self.cot_data[matched_question]
        
        # 返回指定数量的例子
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
        """打印匹配统计信息。"""
        total = sum(self.stats.values()) - self.stats["skipped"]
        
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


# 全局加载器实例
_global_simple_cot_loader: Optional[SimpleMatchCOTLoader] = None


def initialize_simple_match_cot_loader(
    cot_file_path: str,
    cot_format_template: str = "Here's a similar example:\n\nQuestion: {question}\n\nLet's solve it step by step:\n{rationale}\n\nFinal Answer: {final_answer}\n\nNow, let's solve the current problem:",
    use_full_cot: bool = True,
    skip_on_mismatch: bool = True,
    verbose: bool = True,
):
    """初始化全局简单匹配COT加载器。"""
    global _global_simple_cot_loader
    _global_simple_cot_loader = SimpleMatchCOTLoader(
        cot_file_path=cot_file_path,
        cot_format_template=cot_format_template,
        use_full_cot=use_full_cot,
        skip_on_mismatch=skip_on_mismatch,
        verbose=verbose,
    )
    return _global_simple_cot_loader


def get_simple_match_cot_examples(batch, prompt_idx: int, num_repeats: int, tokenizer=None) -> List[str]:
    """COT例子获取函数（简单匹配策略）。"""
    global _global_simple_cot_loader
    
    if _global_simple_cot_loader is None:
        raise RuntimeError("简单匹配COT加载器未初始化")
    
    # 获取问题文本
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
    
    # 使用简单匹配策略获取COT
    try:
        cot_examples = _global_simple_cot_loader.get_cot_examples(
            question=question,
            num_examples=num_repeats
        )
        
        if len(cot_examples) == 0:
            return []
        
        if len(cot_examples) < num_repeats:
            print(f"⚠️  警告: COT例子数量 ({len(cot_examples)}) < rollout次数 ({num_repeats})")
            while len(cot_examples) < num_repeats:
                cot_examples.append(cot_examples[len(cot_examples) % len(cot_examples)])
        
        return cot_examples[:num_repeats]
    
    except Exception as e:
        print(f"❌ 错误: 获取COT例子失败: {e}")
        import traceback
        traceback.print_exc()
        return []

