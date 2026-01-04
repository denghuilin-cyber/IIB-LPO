#!/usr/bin/env python3
"""
超简单测试 - 直接测试匹配

用法:
    python test_match_now.py
"""

import pandas as pd
import json
import re

# 文件路径
train_file = "/nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet"
cot_file = "/nas/dhl/CVAE/Datasets/GSM8K/processed/train_k_shot_GSM8K.jsonl"

print("=" * 80)
print("快速COT匹配测试")
print("=" * 80)
print()

# 加载数据
print("⏳ 加载数据...")
df = pd.read_parquet(train_file)
print(f"✓ 训练数据: {len(df)} 个样本")

cot_data = {}
with open(cot_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        cot_data[data['question']] = data
print(f"✓ COT数据: {len(cot_data)} 个问题")
print()

# 归一化函数
def normalize(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.rstrip('.,!?;:')
    return text.strip()

# 测试前5个
print("=" * 80)
print("测试前5个样本")
print("=" * 80)
print()

success = 0
fail = 0

for idx in range(min(5, len(df))):
    row = df.iloc[idx]
    
    # 从extra_info提取问题
    extra_info = row['extra_info']
    train_question = extra_info['question']
    train_id = extra_info.get('index', idx)
    
    print(f"样本 {idx + 1}:")
    print(f"  ID: {train_id}")
    print(f"  问题: {train_question[:80]}...")
    
    # 尝试匹配
    if train_question in cot_data:
        print(f"  ✅ 精确匹配成功!")
        print(f"     COT例子数: {len(cot_data[train_question]['selected_cots'])}")
        success += 1
    else:
        # 尝试归一化匹配
        norm_train = normalize(train_question)
        found = False
        
        for cot_q, cot_d in cot_data.items():
            if normalize(cot_q) == norm_train:
                print(f"  ✅ 归一化匹配成功!")
                print(f"     COT问题: {cot_q[:80]}...")
                print(f"     COT例子数: {len(cot_d['selected_cots'])}")
                success += 1
                found = True
                break
        
        if not found:
            print(f"  ❌ 匹配失败")
            fail += 1
    
    print()

print("=" * 80)
print(f"结果: 成功 {success}/5, 失败 {fail}/5")
print("=" * 80)

if success == 5:
    print("\n🎉 完美！所有样本都匹配成功！")
    print("✅ COT配置正确，可以开始训练！")
elif success >= 3:
    print("\n✓ 大部分样本匹配成功")
    print("建议: 检查失败的样本是否确实在COT文件中")
else:
    print("\n⚠️  匹配率较低")
    print("建议: 检查数据是否对应")

print()

