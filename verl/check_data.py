#!/usr/bin/env python3
"""
快速检查数据集结构的脚本
用法: python3 check_data_structure.py
"""

import pandas as pd
import sys

# 数据文件路径（根据您的实际路径修改）
data_files = {
    'GSM8K': '/nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet',
    'MATH': '/nas/dhl/Datasets/my_Datasets/MATH/train.parquet',
}

def check_dataset(name, file_path):
    """检查单个数据集的结构"""
    print(f"\n{'='*80}")
    print(f"🔍 检查数据集: {name}")
    print(f"文件路径: {file_path}")
    print(f"{'='*80}")
    
    try:
        # 读取数据
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        elif file_path.endswith('.jsonl'):
            df = pd.read_json(file_path, lines=True)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            print(f"❌ 不支持的文件格式: {file_path}")
            return
        
        # 打印基本信息
        print(f"\n📊 基本信息:")
        print(f"  总行数: {len(df)}")
        print(f"  总列数: {len(df.columns)}")
        
        # 打印列名
        print(f"\n📋 所有列名 ({len(df.columns)} 个):")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col} ({df[col].dtype})")
        
        # 查找可能是问题的字段
        print(f"\n💡 可能包含问题的字段:")
        possible_question_fields = [
            col for col in df.columns 
            if any(keyword in col.lower() for keyword in ['question', 'prompt', 'problem', 'query', 'input'])
        ]
        if possible_question_fields:
            for field in possible_question_fields:
                print(f"  ✓ {field}")
        else:
            print(f"  ⚠️  未找到明显的问题字段")
        
        # 查找可能是答案的字段
        print(f"\n💡 可能包含答案的字段:")
        possible_answer_fields = [
            col for col in df.columns 
            if any(keyword in col.lower() for keyword in ['answer', 'solution', 'output', 'response'])
        ]
        if possible_answer_fields:
            for field in possible_answer_fields:
                print(f"  ✓ {field}")
        else:
            print(f"  ⚠️  未找到明显的答案字段")
        
        # 打印前3行数据（只显示前5列避免太宽）
        print(f"\n📄 前3行数据预览:")
        display_cols = list(df.columns[:5])
        print(df[display_cols].head(3).to_string())
        
        # 如果列太多，提示还有更多列
        if len(df.columns) > 5:
            print(f"\n  ... 还有 {len(df.columns) - 5} 列未显示")
        
        # 打印第一个样本的完整内容（前3个字段）
        if len(df) > 0:
            print(f"\n📝 第一个样本的详细内容（前3个字段）:")
            for col in list(df.columns[:3]):
                value = df.iloc[0][col]
                if isinstance(value, str) and len(value) > 200:
                    print(f"\n  {col}:")
                    print(f"    {value[:200]}...")
                else:
                    print(f"\n  {col}: {value}")
        
        print(f"\n{'='*80}\n")
        
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    print("=" * 80)
    print("数据集结构检查工具")
    print("=" * 80)
    
    for name, path in data_files.items():
        check_dataset(name, path)
    
    print("\n" + "=" * 80)
    print("✅ 检查完成！")
    print("\n💡 根据上面的输出，更新配置:")
    print("   如果问题字段叫 'query'，则使用: prompt_key='query'")
    print("   如果答案字段叫 'response'，则使用: answer_key='response'")
    print("=" * 80)

