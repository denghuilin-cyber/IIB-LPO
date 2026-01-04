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

import re

_SOLUTION_CLIP_CHARS = 300


def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]

    # Optimization: Regular expression matching on very long strings can be slow.
    # For math problems, the final answer is usually at the end.
    # We only match on the last 300 characters, which is a safe approximation for 300 tokens.
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        # 模型输出使用 \boxed{} 格式
        # Pattern matches \boxed{content} where content can contain nested braces
        boxed_pattern = r"\\boxed\{([^\}]+)\}"
        solutions = re.findall(boxed_pattern, solution_str)
        if len(solutions) == 0:
            final_answer = None
        else:
            # take the last solution and clean it
            final_answer = solutions[-1].replace(",", "").replace("$", "").strip()
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual
    Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text (模型输出，使用 \boxed{} 格式)
        ground_truth: the ground truth (标准答案，可能是纯数字 "72" 或完整格式 "...#### 72")
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
        
    Returns:
        dict: {"score": float, "acc": bool, "pred": str}
              与 DAPO/MATH 格式保持一致，避免 batch 混合时长度不匹配
    """
    # 1. 从模型输出中提取答案（使用 \boxed{} 格式）
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        return {
            "score": 0.0,
            "acc": False,
            "pred": ""
        }
    
    # 2. 从 ground_truth 中提取答案
    # 如果 ground_truth 包含 ####，提取 #### 后面的部分
    if isinstance(ground_truth, str) and "####" in ground_truth:
        gt_answer = ground_truth.split("####")[-1].strip()
        gt_answer = gt_answer.replace(",", "").strip()
    else:
        # 否则 ground_truth 就是纯数字（例如 "72"）
        gt_answer = str(ground_truth).strip()
    
    # 3. 比较提取的答案
    if answer == gt_answer:
        final_score = score
        is_correct = True
    else:
        final_score = format_score
        is_correct = False
    
    # 🔧 返回字典格式，与 DAPO/MATH 保持一致
    return {
        "score": final_score,
        "acc": is_correct,
        "pred": answer
    }
