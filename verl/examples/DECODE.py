#!/usr/bin/env python3
"""
快速检查解码文本
"""
import sys
sys.path.insert(0, '/nas/dhl/verl')

from transformers import AutoTokenizer
from examples.grpo_trainer.multi_dataset_with_cot import MultiDatasetWithCOT
from verl.utils.dataset.rl_dataset import RLHFDataset

tokenizer = AutoTokenizer.from_pretrained('/nas/models/Qwen3-8B', trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("创建数据集...")

# 创建两个数据集
rlhf_dataset = RLHFDataset(
    data_files=['/nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet'],
    tokenizer=tokenizer,
    config={'max_prompt_length': 2048, 'truncation': 'error'}
)

custom_dataset = MultiDatasetWithCOT(
    tokenizer=tokenizer,
    config={'gsm8k_path': '/nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet', 'max_prompt_length': 2048},
    is_train=True
)

# 获取第一个样本
print("\n获取样本...")
rlhf_sample = rlhf_dataset[0]
custom_sample = custom_dataset[0]

# 解码对比
rlhf_input_ids = rlhf_sample['input_ids']
rlhf_mask = rlhf_sample['attention_mask']
rlhf_valid_ids = rlhf_input_ids[rlhf_mask.bool()]
rlhf_text = tokenizer.decode(rlhf_valid_ids, skip_special_tokens=True)

custom_input_ids = custom_sample['input_ids']
custom_mask = custom_sample['attention_mask']
custom_valid_ids = custom_input_ids[custom_mask.bool()]
custom_text = tokenizer.decode(custom_valid_ids, skip_special_tokens=True)

print('='*80)
print('🔍 解码文本详细对比（样本0）')
print('='*80)

print(f'\n🅰️ RLHFDataset:')
print(f'  有效token数: {rlhf_mask.sum().item()}')
print(f'  总长度: {len(rlhf_input_ids)}')
print(f'  Padding token数: {(~rlhf_mask.bool()).sum().item()}')
print(f'  Padding在开头: {(rlhf_mask.bool() == False).nonzero()[0].item() if (~rlhf_mask.bool()).any() else "无padding"}')
print(f'\n  完整文本:')
print(f'  {rlhf_text}')

print(f'\n🅱️ MultiDatasetWithCOT:')
print(f'  有效token数: {custom_mask.sum().item()}')
print(f'  总长度: {len(custom_input_ids)}')
print(f'  Padding token数: {(~custom_mask.bool()).sum().item()}')
print(f'  Padding在开头: {(custom_mask.bool() == False).nonzero()[0].item() if (~custom_mask.bool()).any() else "无padding"}')
print(f'\n  完整文本:')
print(f'  {custom_text}')

print(f'\n📊 对比结果:')
print(f'  文本完全相同: {rlhf_text.strip() == custom_text.strip()}')
print(f'  文本长度: {len(rlhf_text)} vs {len(custom_text)}')

if rlhf_text.strip() != custom_text.strip():
    print(f'\n  ❌ 文本不同！查找差异...')
    min_len = min(len(rlhf_text), len(custom_text))
    for i in range(min_len):
        if rlhf_text[i] != custom_text[i]:
            print(f'  首次不同位置: 字符 {i}')
            print(f'    RLHFDataset: ...{rlhf_text[max(0,i-30):i+30]}...')
            print(f'    MultiDatasetWithCOT: ...{custom_text[max(0,i-30):i+30]}...')
            break
else:
    print(f'\n  ✅ 文本完全相同！')
    print(f'  input_ids/attention_mask 不同是因为有效token数量不同：{rlhf_mask.sum().item()} vs {custom_mask.sum().item()}')

print('='*80)

