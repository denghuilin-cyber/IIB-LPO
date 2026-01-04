#!/usr/bin/env python3
"""
测试 CVAE 分叉功能
验证：
1. CVAE 模型加载
2. Embedding 模型加载
3. 文本 -> 嵌入向量
4. 采样多个 z 向量
5. Hook 注册和注入
"""

import torch
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from verl.utils.cvae_branching import CVAEBranchingManager, create_cvae_manager


def test_cvae_loading():
    """测试 1: CVAE 模型加载"""
    print("=" * 80)
    print("测试 1: CVAE 模型加载")
    print("=" * 80)
    
    try:
        manager = create_cvae_manager(
            cvae_model_path="/nas/dhl/CVAE/models/GSM8K-MATH-trained/lars_selector_GSM8K-MATH.pth",
            embedding_model_path="/nas/dhl/CVAE/models/deberta-v2-xlarge",
            injection_layers="all",
            device="cuda"
        )
        print("✅ CVAE 管理器创建成功")
        print(f"  CVAE latent_dim: {manager.latent_dim}")
        print(f"  Embedding dim: {manager.embedding_dim}")
        print(f"  注入层配置: {manager.injection_layers}")
        return manager
    except Exception as e:
        print(f"❌ CVAE 管理器创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_text_to_embedding(manager):
    """测试 2: 文本转嵌入向量"""
    print("\n" + "=" * 80)
    print("测试 2: 文本转嵌入向量")
    print("=" * 80)
    
    test_texts = [
        "解方程 2x + 3 = 7",
        "A train travels 120 km in 2 hours. What is its speed?",
        "计算 15 + 27 的和"
    ]
    
    for text in test_texts:
        try:
            embedding = manager.text_to_embedding(text)
            print(f"✅ 文本: {text[:50]}...")
            print(f"  嵌入形状: {embedding.shape}")
            print(f"  嵌入范围: [{embedding.min():.4f}, {embedding.max():.4f}]")
        except Exception as e:
            print(f"❌ 文本转嵌入失败: {e}")
            import traceback
            traceback.print_exc()


def test_z_sampling(manager):
    """测试 3: 采样 z 向量"""
    print("\n" + "=" * 80)
    print("测试 3: 采样 z 向量")
    print("=" * 80)
    
    test_text = "解方程 2x + 3 = 7。让我们一步一步来："
    num_samples_list = [1, 4, 8]
    
    for num_samples in num_samples_list:
        try:
            z_samples = manager.sample_z_from_text(test_text, num_samples)
            print(f"✅ 采样 {num_samples} 个 z 向量")
            print(f"  z 形状: {z_samples.shape}")
            print(f"  z 范围: [{z_samples.min():.4f}, {z_samples.max():.4f}]")
            print(f"  z 均值: {z_samples.mean():.4f}")
            print(f"  z 标准差: {z_samples.std():.4f}")
            
            # 检查不同样本的差异
            if num_samples > 1:
                diff = (z_samples[0] - z_samples[1]).abs().mean()
                print(f"  样本间差异: {diff:.4f}")
        except Exception as e:
            print(f"❌ 采样失败: {e}")
            import traceback
            traceback.print_exc()


def test_hook_registration(manager):
    """测试 4: Hook 注册"""
    print("\n" + "=" * 80)
    print("测试 4: Hook 注册（模拟）")
    print("=" * 80)
    
    # 创建一个简单的模拟模型
    class MockAttentionLayer(torch.nn.Module):
        def __init__(self, hidden_dim=768):
            super().__init__()
            self.hidden_dim = hidden_dim
        
        def forward(self, x):
            # 模拟 attention 输出
            return x
    
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = torch.nn.ModuleDict({
                'attention': MockAttentionLayer(768)
            })
            self.layer2 = torch.nn.ModuleDict({
                'attention': MockAttentionLayer(768)
            })
            self.layer3 = torch.nn.ModuleDict({
                'attention': MockAttentionLayer(768)
            })
    
    mock_model = MockModel().cuda()
    
    # 测试不同的注入配置
    test_configs = [
        ("all", "所有层"),
        (2, "最后 2 层"),
        (1, "最后 1 层")
    ]
    
    # injection_layers 真正注册的层 desc是一段描述性的中文
    for injection_layers, desc in test_configs:
        print(f"\n--- 测试配置: {desc} ({injection_layers}) ---")
        
        # 创建新的管理器
        test_manager = CVAEBranchingManager(
            cvae_model_path="/nas/dhl/CVAE/models/GSM8K-MATH-trained/lars_selector_GSM8K-MATH.pth",
            embedding_model_path="/nas/dhl/CVAE/models/deberta-v2-xlarge",
            latent_dim=128,
            embedding_dim=1536,
            device="cuda",
            injection_layers=injection_layers
        )
        
        # 采样一个 z
        z = test_manager.sample_z_from_text("测试文本", num_samples=1)
        
        # 注册 hooks
        try:
            test_manager.register_attention_hooks(
                mock_model,
                z,
                injection_mode="add_to_last_token"
            )
            print(f"✅ 成功注册 {len(test_manager.hook_handles)} 个 hooks")
            
            # 测试 forward（验证 hook 是否工作）
            dummy_input = torch.randn(1, 10, 768).cuda()
            output = mock_model.layer1['attention'](dummy_input)
            print(f"  模拟 forward 成功，输出形状: {output.shape}")
            
            # 移除 hooks
            test_manager.remove_hooks()
            print(f"✅ 成功移除 hooks")
        except Exception as e:
            print(f"❌ Hook 注册失败: {e}")
            import traceback
            traceback.print_exc()


def test_z_projection(manager):
    """测试 5: z 投影层"""
    print("\n" + "=" * 80)
    print("测试 5: z 投影层")
    print("=" * 80)
    
    # 测试不同的 hidden_dim
    hidden_dims = [768, 1024, 2048, 4096]
    
    for hidden_dim in hidden_dims:
        try:
            projection_layer = manager.create_z_projection_layer(hidden_dim)
            print(f"✅ 创建投影层: {manager.latent_dim} -> {hidden_dim}")
            
            # 测试投影
            z = torch.randn(1, manager.latent_dim).cuda()
            z_projected = projection_layer(z)
            print(f"  投影前: {z.shape}")
            print(f"  投影后: {z_projected.shape}")
            print(f"  投影范围: [{z_projected.min():.4f}, {z_projected.max():.4f}]")
        except Exception as e:
            print(f"❌ 投影失败: {e}")


def main():
    """主测试流程"""
    print("\n" + "=" * 80)
    print("CVAE 分叉功能测试")
    print("=" * 80)
    
    # 测试 1: 加载 CVAE
    manager = test_cvae_loading()
    if manager is None:
        print("\n❌ CVAE 加载失败，终止测试")
        return
    
    # 测试 2: 文本转嵌入
    test_text_to_embedding(manager)
    
    # 测试 3: 采样 z
    test_z_sampling(manager)
    
    # 测试 4: Hook 注册
    test_hook_registration(manager)
    
    # 测试 5: z 投影
    test_z_projection(manager)
    
    print("\n" + "=" * 80)
    print("✅ 所有测试完成")
    print("=" * 80)


if __name__ == "__main__":
    main()

