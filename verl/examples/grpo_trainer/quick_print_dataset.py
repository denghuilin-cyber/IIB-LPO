#!/usr/bin/env python3
"""
保存数据集样本为 JSONL 格式，并打印关键字段

用法:
    python save_dataset_samples.py
"""

import sys
sys.path.insert(0, '/nas/dhl/verl')

from examples.grpo_trainer.multi_dataset_with_cot import MultiDatasetWithCOT
from transformers import AutoTokenizer
import torch
import json
import numpy as np

# 配置
ACTOR_MODEL_PATH = "/nas/models/Qwen3-8B"
TRAIN_FILES_GSM8K = "/nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet"
TRAIN_FILES_MATH = "/nas/dhl/Datasets/my_Datasets/MATH/train.parquet"
OUTPUT_FILE = "/nas/dhl/verl/examples/grpo_trainer/dataset_samples.jsonl"
NUM_SAMPLES = 5  # 保存前5个样本


def make_serializable(obj):
    """将对象转换为JSON可序列化的格式"""
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj


def main():
    print("="*80)
    print("🔍 数据集内容保存工具")
    print("="*80)
    
    # 加载 tokenizer
    print("\n⏳ 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ACTOR_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer 加载成功")
    
    # 创建数据集
    print("\n⏳ 创建数据集...")
    config = {
        'gsm8k_path': TRAIN_FILES_GSM8K,
        'math_path': TRAIN_FILES_MATH,
        'max_prompt_length': 2048,
    }
    
    dataset = MultiDatasetWithCOT(
        tokenizer=tokenizer,
        config=config,
        is_train=True
    )
    print(f"✓ 数据集大小: {len(dataset)}")
    
    # 保存样本为 JSONL
    print(f"\n⏳ 保存前 {NUM_SAMPLES} 个样本到 {OUTPUT_FILE}...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for i in range(min(NUM_SAMPLES, len(dataset))):
            try:
                sample = dataset[i]
                # 转换为可序列化格式
                serializable_sample = make_serializable(sample)
                # 写入文件
                f.write(json.dumps(serializable_sample, ensure_ascii=False) + '\n')
                print(f"  ✓ 样本 {i}: {sample.get('dataset_name', 'N/A')}")
            except Exception as e:
                print(f"  ✗ 样本 {i} 失败: {e}")
    
    print(f"✓ 已保存完整样本到 {OUTPUT_FILE}")
    
    # 打印关键字段
    print(f"\n{'='*80}")
    print(f"📋 关键字段检查（前 {NUM_SAMPLES} 个样本）")
    print(f"{'='*80}\n")
    
    for i in range(min(NUM_SAMPLES, len(dataset))):
        try:
            sample = dataset[i]
            
            # 解码 input_ids 为文本
            input_ids = sample.get('input_ids')
            attention_mask = sample.get('attention_mask')
            
            if input_ids is not None and attention_mask is not None:
                valid_ids = input_ids[attention_mask.bool()]
                decoded_prompt = tokenizer.decode(valid_ids, skip_special_tokens=True)
            else:
                decoded_prompt = "N/A"
            
            # 提取关键字段
            question = sample.get('question', 'N/A')
            dataset_name = sample.get('dataset_name', 'N/A')
            reward_model = sample.get('reward_model', {})
            ground_truth = reward_model.get('ground_truth', 'N/A') if isinstance(reward_model, dict) else 'N/A'
            
            print(f"{'='*80}")
            print(f"样本 {i} ({dataset_name})")
            print(f"{'='*80}")
            
            print(f"\n📝 1. decoded_prompt (模型实际看到的):")
            print(f"{decoded_prompt}")
            
            print(f"\n📝 2. question (用于COT匹配):")
            print(f"{question}")
            
            print(f"\n📝 3. reward_model.ground_truth (正确答案):")
            print(f"{ground_truth}")
            
            # 检查关键点
            print(f"\n✅ 关键检查:")
            has_instruction = "Let's think step by step" in decoded_prompt
            question_has_instruction = "Let's think step by step" in str(question)
            
            print(f"  • decoded_prompt 包含指令: {'✅ 是' if has_instruction else '❌ 否'}")
            print(f"  • question 包含指令: {'⚠️ 是（错误）' if question_has_instruction else '✅ 否（正确）'}")
            print(f"  • ground_truth 存在: {'✅ 是' if ground_truth != 'N/A' else '❌ 否'}")
            
            print()
            
        except Exception as e:
            print(f"\n❌ 样本 {i} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 最终总结
    print("\n" + "="*80)
    print("🎯 最终总结")
    print("="*80)
    
    print(f"\n✅ 完整数据已保存到: {OUTPUT_FILE}")
    print(f"   查看: cat {OUTPUT_FILE} | jq .")
    
    print(f"\n📊 关键字段确认:")
    print(f"  1. ✅ decoded_prompt: 这是模型实际看到的文本")
    print(f"     → 应该包含: 'Let's think step by step and output the final answer after \"####\".'")
    print(f"     → 如果包含，说明指令已正确添加")
    
    print(f"\n  2. ✅ question: 这是用于 COT 匹配的纯问题")
    print(f"     → 不应该包含指令")
    print(f"     → 如果不包含，说明 COT 匹配会正常工作")
    
    print(f"\n  3. ✅ reward_model.ground_truth: 正确答案")
    print(f"     → 用于计算奖励")
    print(f"     → 必须正确，否则训练会学到错误的东西")
    
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()





'''关键检查：
✅ decoded_prompt: 模型实际看到的文本（应该包含指令）
✅ question: 用于 COT 匹配的纯问题（不应该包含指令）
✅ reward_model.ground_truth: 正确答'''