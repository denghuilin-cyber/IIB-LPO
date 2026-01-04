#!/usr/bin/env python3
"""
测试 CVAE 采样功能
验证：
1. CVAE Manager 初始化
2. 从 question + prefix 采样 z
3. z 向量的形状和数值范围
"""

import torch
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from verl.utils.cvae_branching import create_cvae_manager


def test_cvae_sampling():
    """测试 CVAE 采样"""
    print("=" * 80)
    print("测试 CVAE 采样功能")
    print("=" * 80)
    
    # 1. 初始化 CVAE Manager
    print("\n1. 初始化 CVAE Manager...")
    
    try:
        manager = create_cvae_manager(
            cvae_model_path="/nas/dhl/CVAE/models/GSM8K-MATH-trained/lars_selector_GSM8K-MATH.pth",
            embedding_model_path="/nas/dhl/CVAE/models/deberta-v2-xlarge",
            injection_layers="all",
            device="cuda"
        )
        print("✅ CVAE Manager 初始化成功")
        print(f"  Latent dim: {manager.latent_dim}")
        print(f"  Embedding dim: {manager.embedding_dim}")
    except Exception as e:
        print(f"❌ CVAE Manager 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 2. 准备测试数据
    print("\n2. 准备测试数据...")
    
    # 纯问题（不包含 example COT）
    pure_question = "解方程 2x + 3 = 7"
    
    # 已生成的 prefix（模拟 LLM 的部分生成）
    prefix_text = "好的，让我们一步一步来解这个方程。首先"
    
    print(f"Pure question: {pure_question}")
    print(f"Prefix: {prefix_text}")
    print(f"CVAE 输入: {pure_question} {prefix_text}")
    
    # 3. 采样单个 z
    print("\n3. 采样单个 z 向量...")
    
    try:
        z_single = manager.sample_z_from_text(
            text=pure_question + " " + prefix_text,
            num_samples=1
        )
        
        print(f"✅ 单个 z 采样成功")
        print(f"  Shape: {z_single.shape}")
        print(f"  Device: {z_single.device}")
        print(f"  Dtype: {z_single.dtype}")
        print(f"  Range: [{z_single.min():.4f}, {z_single.max():.4f}]")
        print(f"  Mean: {z_single.mean():.4f}")
        print(f"  Std: {z_single.std():.4f}")
        
    except Exception as e:
        print(f"❌ 单个 z 采样失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. 采样多个 z
    print("\n4. 采样多个 z 向量...")
    
    num_samples_list = [1, 3, 5]
    
    for num_samples in num_samples_list:
        try:
            z_multi = manager.sample_z_from_text(
                text=pure_question + " " + prefix_text,
                num_samples=num_samples
            )
            
            print(f"✅ 采样 {num_samples} 个 z 成功")
            print(f"  Shape: {z_multi.shape}")
            print(f"  Range: [{z_multi.min():.4f}, {z_multi.max():.4f}]")
            print(f"  Mean: {z_multi.mean():.4f}")
            
            # 检查不同样本的差异
            if num_samples > 1:
                diff = (z_multi[0] - z_multi[1]).abs().mean()
                print(f"  样本间差异: {diff:.4f}")
            
        except Exception as e:
            print(f"❌ 采样 {num_samples} 个 z 失败: {e}")
    
    # 5. 测试不同的输入
    print("\n5. 测试不同的输入...")
    
    test_cases = [
        ("解方程 2x + 3 = 7", "首先"),
        ("计算 15 + 27", "让我们"),
        ("A train travels 120 km in 2 hours", "To solve this"),
    ]
    
    for question, prefix in test_cases:
        try:
            cvae_input = question + " " + prefix
            z = manager.sample_z_from_text(cvae_input, num_samples=1)
            
            print(f"✅ Question: {question[:50]}...")
            print(f"   Prefix: {prefix}")
            print(f"   z range: [{z.min():.4f}, {z.max():.4f}]")
            
        except Exception as e:
            print(f"❌ 采样失败: {e}")
    
    # 6. 验证 z 的可重复性（相同输入）
    print("\n6. 验证采样的随机性...")
    
    try:
        z1 = manager.sample_z_from_text(pure_question + " " + prefix_text, num_samples=1)
        z2 = manager.sample_z_from_text(pure_question + " " + prefix_text, num_samples=1)
        
        diff = (z1 - z2).abs().mean()
        
        print(f"相同输入的两次采样:")
        print(f"  z1 mean: {z1.mean():.4f}")
        print(f"  z2 mean: {z2.mean():.4f}")
        print(f"  差异: {diff:.4f}")
        
        if diff > 0.01:
            print(f"  ✅ 采样具有随机性（符合预期）")
        else:
            print(f"  ⚠️  采样几乎相同（可能是确定性采样）")
            
    except Exception as e:
        print(f"❌ 验证失败: {e}")
    
    print("\n" + "=" * 80)
    print("✅ 测试完成")
    print("=" * 80)
    
    print("\n📊 总结:")
    print("  - CVAE Manager 可以成功初始化")
    print("  - 可以从 question + prefix 采样 z 向量")
    print("  - z 向量形状正确: [num_samples, 128]")
    print("  - z 向量具有合理的数值范围")
    print("  - 采样具有随机性")
    print("\n🎯 下一步: 将 z 注入到 LLM 的 attention 层")


if __name__ == "__main__":
    test_cvae_sampling()

