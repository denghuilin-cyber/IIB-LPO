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
Hybrid matching COT loader: 先模糊匹配，再精确匹配，失败则跳过。

匹配策略:
1. 先尝试方法2：归一化/模糊匹配（更宽松）
2. 如果失败，尝试方法1：精确匹配
3. 如果都失败，跳过数据并打印日志

Use this for maximum robustness while ensuring quality.
"""

import json
import re
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher
import numpy as np


def normalize_question(question: str) -> str:
    """
    归一化问题文本。
    
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


def compute_similarity(text1: str, text2: str) -> float:
    """计算两个文本的相似度 (0-1)。"""
    norm1 = normalize_question(text1)
    norm2 = normalize_question(text2)
    return SequenceMatcher(None, norm1, norm2).ratio()


class HybridMatchCOTLoader:
    """
    混合匹配策略的COT加载器。
    
    匹配策略（按顺序尝试）:
    1. 归一化匹配（方法2）: 归一化后精确匹配
    2. 模糊匹配（方法2扩展）: 使用相似度阈值
    3. 精确匹配（方法1）: 原始文本完全匹配
    4. 失败: 跳过数据，打印日志，返回空
    
    特点:
    - 先尝试更robust的匹配（归一化）
    - 然后尝试精确匹配作为fallback
    - 完全匹配不到就跳过
    - 详细的匹配日志
    """
    
    def __init__(
        self,
        cot_file_path: str,
        cot_format_template: str = "Here's a similar example:\n\nQuestion: {question}\n\nStep-by-step solution:\n{rationale}\n\nFinal Answer: {final_answer}\n\nNow, let's solve the current problem:",
        fuzzy_threshold: float = 0.95,
        use_full_cot: bool = True,
        skip_on_mismatch: bool = True,
        verbose: bool = True,
    ):
        """
        初始化混合匹配COT加载器。
        
        Args:
            cot_file_path: COT JSONL文件路径
            cot_format_template: COT格式化模板
            fuzzy_threshold: 模糊匹配的相似度阈值 (0-1)
                           推荐: 0.95 (严格), 0.90 (中等), 0.85 (宽松)
            use_full_cot: 是否使用完整COT（问题+推理+答案）
            skip_on_mismatch: 匹配失败时是否跳过（返回空列表）
            verbose: 是否打印详细日志
        """
        self.cot_file_path = cot_file_path
        self.cot_format_template = cot_format_template
        self.fuzzy_threshold = fuzzy_threshold
        self.use_full_cot = use_full_cot
        self.skip_on_mismatch = skip_on_mismatch
        self.verbose = verbose
        
        # 存储COT数据
        # 原始question → [COT examples]
        self.cot_data: Dict[str, List[str]] = {}
        
        # 归一化question → 原始question
        self.normalized_to_original: Dict[str, str] = {}
        
        # 所有原始问题（用于模糊匹配）
        self.all_questions: List[str] = []
        
        # 统计信息
        self.stats = {
            "normalized_matches": 0,    # 归一化匹配成功
            "fuzzy_matches": 0,          # 模糊匹配成功
            "exact_matches": 0,          # 精确匹配成功
            "failed_matches": 0,         # 完全失败
            "skipped": 0,                # 跳过的数据
        }
        
        # 加载COT数据
        self._load_cot_data()
        
        print(f"✓ Loaded COT data for {len(self.cot_data)} questions")
        print(f"  Matching strategy: Normalized → Fuzzy → Exact")
        print(f"  Fuzzy threshold: {fuzzy_threshold}")
        print(f"  Skip on mismatch: {skip_on_mismatch}")
    
    def _load_cot_data(self):
        """从JSONL文件加载COT数据。"""
        with open(self.cot_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                
                original_question = data["question"]
                self.all_questions.append(original_question)
                
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
    
    def _try_normalized_match(self, query: str) -> Optional[Tuple[str, str]]:
        """
        尝试归一化匹配（方法2的第一步）。
        
        Returns:
            (matched_question, match_type) or None
        """
        normalized_query = normalize_question(query)
        
        if normalized_query in self.normalized_to_original:
            matched_original = self.normalized_to_original[normalized_query]
            return (matched_original, "normalized")
        
        return None
    
    def _try_fuzzy_match(self, query: str) -> Optional[Tuple[str, str, float]]:
        """
        尝试模糊匹配（方法2的第二步）。
        
        Returns:
            (matched_question, match_type, similarity) or None
        """
        best_match = None
        best_score = 0.0
        
        normalized_query = normalize_question(query)
        
        for candidate in self.all_questions:
            normalized_candidate = normalize_question(candidate)
            similarity = compute_similarity(normalized_query, normalized_candidate)
            
            if similarity > best_score and similarity >= self.fuzzy_threshold:
                best_score = similarity
                best_match = candidate
        
        if best_match is not None:
            return (best_match, "fuzzy", best_score)
        
        return None
    
    def _try_exact_match(self, query: str) -> Optional[Tuple[str, str]]:
        """
        尝试精确匹配（方法1）。
        
        Returns:
            (matched_question, match_type) or None
        """
        if query in self.cot_data:
            return (query, "exact")
        
        return None
    
    def get_cot_examples(
        self,
        question: str,
        num_examples: Optional[int] = None
    ) -> List[str]:
        """
        获取COT例子（使用混合匹配策略）。
        
        匹配顺序:
        1. 归一化匹配
        2. 模糊匹配
        3. 精确匹配
        4. 失败 → 跳过
        
        Args:
            question: 训练数据中的问题文本
            num_examples: 需要返回的例子数量
        
        Returns:
            COT例子列表（如果匹配失败且skip_on_mismatch=True，返回空列表）
        """
        matched_question = None
        match_type = None
        similarity = None
        
        # 策略1: 归一化匹配（方法2）
        result = self._try_normalized_match(question)
        if result is not None:
            matched_question, match_type = result
            self.stats["normalized_matches"] += 1
        
        # 策略2: 模糊匹配（方法2扩展）
        if matched_question is None:
            result = self._try_fuzzy_match(question)
            if result is not None:
                matched_question, match_type, similarity = result
                self.stats["fuzzy_matches"] += 1
        
        # 策略3: 精确匹配（方法1）
        if matched_question is None:
            result = self._try_exact_match(question)
            if result is not None:
                matched_question, match_type = result
                self.stats["exact_matches"] += 1
        
        # 所有策略都失败
        if matched_question is None:
            self.stats["failed_matches"] += 1
            
            if self.verbose:
                print(f"❌ 匹配失败 - 跳过数据:")
                print(f"   问题: {question[:100]}...")
            
            if self.skip_on_mismatch:
                self.stats["skipped"] += 1
                return []  # 返回空列表，表示跳过
            else:
                return []
        
        # 匹配成功 - 打印日志
        if self.verbose:
            if match_type == "normalized":
                print(f"✓ 归一化匹配成功")
            elif match_type == "fuzzy":
                print(f"≈ 模糊匹配成功 (相似度: {similarity:.3f})")
                if matched_question != question:
                    print(f"   查询: {question[:60]}...")
                    print(f"   匹配: {matched_question[:60]}...")
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
        total = sum(self.stats.values()) - self.stats["skipped"]  # skipped已包含在failed中
        
        if total == 0:
            print("还没有匹配尝试")
            return
        
        print("\n" + "=" * 60)
        print("COT匹配统计")
        print("=" * 60)
        print(f"归一化匹配: {self.stats['normalized_matches']:5d} / {total} ({self.stats['normalized_matches']/total*100:5.2f}%)")
        print(f"模糊匹配:   {self.stats['fuzzy_matches']:5d} / {total} ({self.stats['fuzzy_matches']/total*100:5.2f}%)")
        print(f"精确匹配:   {self.stats['exact_matches']:5d} / {total} ({self.stats['exact_matches']/total*100:5.2f}%)")
        print(f"匹配失败:   {self.stats['failed_matches']:5d} / {total} ({self.stats['failed_matches']/total*100:5.2f}%)")
        if self.skip_on_mismatch:
            print(f"跳过数据:   {self.stats['skipped']:5d}")
        print("=" * 60)
        
        success_rate = (total - self.stats['failed_matches']) / total * 100
        print(f"总体成功率: {success_rate:.2f}%")
        print("=" * 60 + "\n")


# 全局加载器实例
_global_hybrid_cot_loader: Optional[HybridMatchCOTLoader] = None


def initialize_hybrid_match_cot_loader(
    cot_file_path: str,
    cot_format_template: str = "Here's a similar example:\n\nQuestion: {question}\n\nLet's solve it step by step:\n{rationale}\n\nFinal Answer: {final_answer}\n\nNow, let's solve the current problem:",
    fuzzy_threshold: float = 0.95,
    use_full_cot: bool = True,
    skip_on_mismatch: bool = True,
    verbose: bool = True,
):
    """
    初始化全局混合匹配COT加载器。
    
    Args:
        cot_file_path: COT JSONL文件路径
        cot_format_template: COT格式化模板
        fuzzy_threshold: 模糊匹配阈值 (推荐: 0.95)
        use_full_cot: 是否使用完整COT
        skip_on_mismatch: 匹配失败时是否跳过
        verbose: 是否打印详细日志
    
    Returns:
        初始化的HybridMatchCOTLoader实例
    """
    global _global_hybrid_cot_loader
    _global_hybrid_cot_loader = HybridMatchCOTLoader(
        cot_file_path=cot_file_path,
        cot_format_template=cot_format_template,
        fuzzy_threshold=fuzzy_threshold,
        use_full_cot=use_full_cot,
        skip_on_mismatch=skip_on_mismatch,
        verbose=verbose,
    )
    return _global_hybrid_cot_loader


def get_hybrid_match_cot_examples(batch, prompt_idx: int, num_repeats: int, tokenizer=None) -> List[str]:
    """
    COT例子获取函数（混合匹配策略）。
    
    由GRPOCOTAugmenter调用。
    
    Args:
        batch: DataProto batch
        prompt_idx: 原始prompt索引（repeat前）
        num_repeats: repeat次数
        tokenizer: Tokenizer（可选）
    
    Returns:
        COT例子列表（如果匹配失败，返回空列表表示跳过）
    """
    global _global_hybrid_cot_loader
    
    if _global_hybrid_cot_loader is None:
        raise RuntimeError(
            "混合匹配COT加载器未初始化。请先调用 initialize_hybrid_match_cot_loader()"
        )
    
    # 获取问题文本
    question = None
    
    # 从non_tensor_batch获取
    if "question" in batch.non_tensor_batch:
        questions = batch.non_tensor_batch["question"]
        if isinstance(questions, np.ndarray):
            question = str(questions[prompt_idx * num_repeats])
        else:
            question = questions[prompt_idx * num_repeats]
    
    # Fallback: 从input_ids解码
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
        return []  # 跳过
    
    # 使用混合匹配策略获取COT
    try:
        cot_examples = _global_hybrid_cot_loader.get_cot_examples(
            question=question,
            num_examples=num_repeats
        )
        
        # 如果返回空列表，表示匹配失败且被跳过
        if len(cot_examples) == 0:
            return []  # 直接返回空，表示跳过这个问题
        
        # 补充到指定数量（循环使用）
        if len(cot_examples) < num_repeats:
            print(f"⚠️  警告: COT例子数量 ({len(cot_examples)}) < rollout次数 ({num_repeats})")
            while len(cot_examples) < num_repeats:
                cot_examples.append(cot_examples[len(cot_examples) % len(cot_examples)])
        
        return cot_examples[:num_repeats]
    
    except Exception as e:
        print(f"❌ 错误: 获取COT例子失败: {e}")
        import traceback
        traceback.print_exc()
        return []  # 失败时跳过

