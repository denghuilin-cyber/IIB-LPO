#!/usr/bin/env python3
"""
检查训练数据中的data_source字段值

用法:
    python check_data_source.py
"""

import pandas as pd

datasets = [
    ("GSM8K", "/nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet"),
    ("MATH", "/nas/dhl/Datasets/my_Datasets/MATH/train.parquet"),
    ("NuminaMath-CoT", "/nas/dhl/Datasets/my_Datasets/NuminaMath-CoT/train.parquet"),
]

print("=" * 80)
print("检查data_source字段值")
print("=" * 80)
print()

for dataset_name, file_path in datasets:
    print(f"{'▼' * 40}")
    print(f"数据集: {dataset_name}")
    print(f"文件: {file_path}")
    print(f"{'▼' * 40}")
    
    try:
        df = pd.read_parquet(file_path)
        print(f"✓ 加载成功，共 {len(df)} 个样本")
        
        # 检查data_source字段
        if 'data_source' in df.columns:
            unique_sources = df['data_source'].unique()
            print(f"\ndata_source 字段的唯一值:")
            for source in unique_sources:
                count = (df['data_source'] == source).sum()
                print(f"  - '{source}': {count} 个样本")
        else:
            print(f"\n⚠️  没有 data_source 字段!")
            print(f"可用字段: {list(df.columns)}")
        
        # 显示第一个样本的完整信息
        print(f"\n第一个样本的 data_source:")
        row = df.iloc[0]
        print(f"  {row.get('data_source', 'N/A')}")
        
        print()
    
    except FileNotFoundError:
        print(f"❌ 文件不存在")
        print()
    except Exception as e:
        print(f"❌ 错误: {e}")
        print()

print("=" * 80)
print("检查完成")
print("=" * 80)
print()
print("💡 根据上面的结果，您需要在训练脚本中配置正确的映射关系")
print()

