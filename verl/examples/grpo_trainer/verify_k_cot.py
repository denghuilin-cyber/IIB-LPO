#!/usr/bin/env python3
"""
独立的COT验证脚本 - 验证8次rollout使用8个不同的COT

不依赖其他模块，所有代码都在这个文件里。

用法:
    python verify_cot_standalone.py
"""

import pandas as pd
import json
import re
from typing import Dict, List

# ============================================================================
# 配置区域 - 根据您的实际情况修改
# ============================================================================

NUM_ROLLOUTS = 8  # GRPO的group size，想测试几次rollout就设置多少

# 数据集配置
DATASETS = {
    "gsm8k": {
        "train_file": "/nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet",
        "cot_file": "/nas/dhl/CVAE/Datasets/GSM8K/processed/train_k_shot_GSM8K.jsonl",
        "data_source": "openai/gsm8k"
    },
    "math": {
        "train_file": "/nas/dhl/Datasets/my_Datasets/MATH/train.parquet",
        "cot_file": "/nas/dhl/CVAE/Datasets/MATH/processed/train_k_shot_MATH.jsonl",
        "data_source": "hendrycks/math"
    },
    "numina": {
        "train_file": "/nas/dhl/Datasets/my_Datasets/NuminaMath-CoT/train.parquet",
        "cot_file": "/nas/dhl/CVAE/Datasets/NuminaMath-CoT/processed/train_k_shot_NuminaMath-CoT.jsonl",
        "data_source": "AI-MO/NuminaMath-CoT"
    }
}

COT_FORMAT_TEMPLATE = """Here's a similar example:

Question: {question}

Step-by-step solution:
{rationale}

Final Answer: {final_answer}

Now, let's solve the current problem:"""

# ============================================================================
# 辅助函数
# ============================================================================

def normalize_question(question: str) -> str:
    """归一化问题文本"""
    text = question.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.rstrip('.,!?;:')
    return text.strip()


def load_cot_data(cot_file: str) -> tuple:
    """
    加载COT数据
    
    Returns:
        (cot_data, normalized_map)
        cot_data: {question → [formatted_cots]}
        normalized_map: {normalized_question → original_question}
    """
    cot_data = {}
    normalized_map = {}
    
    with open(cot_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            
            original_question = data["question"]
            normalized = normalize_question(original_question)
            normalized_map[normalized] = original_question
            
            # 格式化COT例子
            selected_cots = data.get("selected_cots", [])
            formatted_cots = []
            
            for cot in selected_cots:
                formatted_cot = COT_FORMAT_TEMPLATE.format(
                    question=cot.get("question", ""),
                    rationale=cot.get("rationale", ""),
                    final_answer=cot.get("final_answer", "")
                )
                formatted_cots.append(formatted_cot)
            
            cot_data[original_question] = formatted_cots
    
    return cot_data, normalized_map


def get_cot_examples(question: str, cot_data: dict, normalized_map: dict, num_examples: int) -> List[str]:
    """
    获取COT examples（归一化匹配 → 精确匹配）
    
    Returns:
        List of COT examples
    """
    matched_question = None
    match_type = None
    
    # 策略1: 归一化匹配
    normalized_query = normalize_question(question)
    if normalized_query in normalized_map:
        matched_question = normalized_map[normalized_query]
        match_type = "normalized"
    
    # 策略2: 精确匹配
    if matched_question is None and question in cot_data:
        matched_question = question
        match_type = "exact"
    
    # 匹配失败
    if matched_question is None:
        return []
    
    # 获取COT
    cot_examples = cot_data[matched_question]
    
    # 返回指定数量（循环使用）
    if num_examples <= len(cot_examples):
        return cot_examples[:num_examples]
    else:
        result = []
        for i in range(num_examples):
            result.append(cot_examples[i % len(cot_examples)])
        return result


# ============================================================================
# 主程序
# ============================================================================

def main():
    print("=" * 100)
    print(f"验证 {NUM_ROLLOUTS} 次 Rollout 的 COT Example".center(100))
    print("=" * 100)
    print()
    
    # 测试每个数据集
    for dataset_name, config in DATASETS.items():
        print("\n" + "=" * 100)
        print(f"数据集: {dataset_name}".center(100))
        print("=" * 100)
        
        try:
            # 加载COT数据
            print(f"加载COT数据: {config['cot_file']}")
            cot_data, normalized_map = load_cot_data(config['cot_file'])
            print(f"✓ 加载了 {len(cot_data)} 个问题的COT数据")
            
            # 加载训练数据
            print(f"加载训练数据: {config['train_file']}")
            df = pd.read_parquet(config['train_file'])
            print(f"✓ 加载了 {len(df)} 个训练样本")
            
            # 取第一个样本
            row = df.iloc[0]
            extra_info = row['extra_info']
            question = extra_info['question']
            data_source = row['data_source']
            
            print(f"\n测试问题:")
            print(f"  ID: {extra_info.get('index', 'N/A')}")
            print(f"  data_source: {data_source}")
            print(f"  问题: {question[:120]}...")
            print()
            
            # 获取COT examples
            print(f"获取 {NUM_ROLLOUTS} 个COT examples:")
            print("-" * 100)
            
            cot_examples = get_cot_examples(question, cot_data, normalized_map, NUM_ROLLOUTS)
            
            if len(cot_examples) == 0:
                print("❌ 匹配失败，跳过此数据集")
                continue
            
            print(f"✅ 成功获取 {len(cot_examples)} 个COT examples")
            print()
            
            # 显示每个rollout的COT
            for i, cot in enumerate(cot_examples, 1):
                print(f"{'▼' * 50}")
                print(f"Rollout {i}/{NUM_ROLLOUTS}:")
                print(f"{'▼' * 50}")
                
                # 显示COT内容（前500字符）
                print(cot[:500])
                
                if len(cot) > 500:
                    print(f"... (共 {len(cot)} 字符)")
                
                print()
            
            # 验证是否不同
            print("=" * 100)
            print("验证COT是否不同:")
            print("-" * 100)
            
            unique_cots = set(cot_examples)
            
            if len(unique_cots) == len(cot_examples):
                print(f"✅ 完美！{NUM_ROLLOUTS} 个COT examples 都不相同")
            elif len(unique_cots) == 1:
                print(f"❌ 所有COT都相同（这不应该发生）")
            else:
                print(f"⚠️  有 {len(unique_cots)} 个不同的COT（总共 {NUM_ROLLOUTS} 个rollout）")
                print(f"   说明: COT例子数量少于rollout次数，会循环使用")
                print(f"   这是正常的！")
            
            print("=" * 100)
            
        except FileNotFoundError as e:
            print(f"⚠️  文件不存在: {e}")
            print(f"   跳过此数据集")
        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 100)
    print("验证完成！")
    print("=" * 100)
    print()
    print("💡 说明:")
    print(f"   - 每个问题会rollout {NUM_ROLLOUTS} 次")
    print(f"   - 每次rollout会拼接不同的COT example到问题后面")
    print(f"   - 如果COT例子数量 < {NUM_ROLLOUTS}，会循环使用（这是正常的）")
    print()
    print("🚀 如果看到 '✅ 完美！' 或 '⚠️ 有X个不同的COT'，就可以开始训练了！")
    print()


if __name__ == "__main__":
    main()

