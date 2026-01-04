#!/usr/bin/env python3
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# 🔧 在导入 verl 之前，先 patch hf_tokenizer 函数
from transformers import AutoTokenizer, AutoConfig

def patched_hf_tokenizer(name_or_path, **kwargs):
    """修补后的 tokenizer 加载函数"""
    if os.path.exists(name_or_path):
        kwargs['local_files_only'] = True
    return AutoTokenizer.from_pretrained(name_or_path, **kwargs)

# 导入 verl 模块
from verl.utils import tokenizer as tokenizer_module
# Monkey patch
tokenizer_module.hf_tokenizer = patched_hf_tokenizer

from vllm import LLM, SamplingParams
from verl.workers.config import RolloutConfig, HFModelConfig
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout
from torch.distributed.device_mesh import DeviceMesh


def create_mock_config():
    """创建模拟配置"""
    config = RolloutConfig(
        response_length=1024,
        prompt_length=2048,
        tensor_model_parallel_size=1,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        free_cache_engine=False,
        load_format="auto",
        compute_entropy=True,
        enable_cvae_branching=True,
        cvae_num_branches_per_path=3,
        cvae_model_path="/nas/dhl/CVAE/models/GSM8K-MATH-trained/lars_selector_GSM8K-MATH.pth",
        cvae_embedding_model_path="/nas/dhl/CVAE/models/deberta-v2-xlarge",
        cvae_injection_layers="all",
    )
    return config


def create_mock_model_config():
    """创建模拟模型配置"""
    model_path = "/nas/models/qwen2.5-math-1.5B_instruct"
    
    if not os.path.exists(model_path):
        raise ValueError(f"模型路径不存在: {model_path}")
    
    print(f"📁 加载本地模型: {model_path}")
    
    hf_config = AutoConfig.from_pretrained(
        model_path, 
        trust_remote_code=True,
        local_files_only=True
    )
    
    print("✅ 模型配置加载成功")
    
    model_config = HFModelConfig(
        local_path=model_path,
        local_tokenizer_path=model_path,
        hf_config=hf_config,
        trust_remote_code=True,
    )
    
    return model_config


def test_prepare_branching_data():
    """测试 _prepare_branching_data_for_single_path"""
    print("=" * 80)
    print("测试 _prepare_branching_data_for_single_path")
    print("=" * 80)
    
    print("\n1. 初始化 vLLMRollout...")
    config = create_mock_config()
    model_config = create_mock_model_config()
    
    device_mesh = DeviceMesh("cuda", torch.arange(1))
    
    rollout = vLLMRollout(
        config=config,
        model_config=model_config,
        device_mesh=device_mesh
    )
    print("✅ vLLMRollout 初始化完成")
    
    print("\n2. 进行第一次 rollout...")
    test_prompt = "解方程 2x + 3 = 7"
    prompt_ids = rollout.tokenizer.encode(test_prompt, add_special_tokens=False)
    
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=100,
        compute_entropy=True,
        logprobs=1,
    )
    
    outputs = rollout.inference_engine.generate(
        prompts=[{"prompt_token_ids": prompt_ids}],
        sampling_params=sampling_params,
        use_tqdm=False
    )
    
    output = outputs[0]
    print("✅ Rollout 完成")
    print(f"  生成 token 数: {len(output.outputs[0].token_ids)}")
    print(f"  熵值数量: {len(output.outputs[0].entropies) if output.outputs[0].entropies else 0}")
    
    print("\n3. 测试 _prepare_branching_data_for_single_path...")
    
    pure_question = "解方程 2x + 3 = 7"
    
    try:
        branching_data = rollout._prepare_branching_data_for_single_path(
            output=output,
            prompt_ids=prompt_ids,
            pure_question=pure_question,
            sample_idx=0
        )
        
        print("\n✅ 分叉数据准备成功！")
        print("\n" + "=" * 80)
        print("分叉数据详情:")
        print("=" * 80)
        
        print(f"\n📍 最高熵点:")
        print(f"  位置: token {branching_data['max_entropy_idx']}")
        print(f"  熵值: {branching_data['max_entropy_value']:.4f}")
        
        print(f"\n📝 Prefix 信息:")
        print(f"  Token 数: {len(branching_data['prefix_ids'])}")
        print(f"  文本: {branching_data['prefix_text'][:200]}...")
        
        print(f"\n🎲 采样的 z 向量:")
        print(f"  形状: {branching_data['z_samples'].shape}")
        print(f"  范围: [{branching_data['z_samples'].min():.4f}, {branching_data['z_samples'].max():.4f}]")
        print(f"  均值: {branching_data['z_samples'].mean():.4f}")
        print(f"  标准差: {branching_data['z_samples'].std():.4f}")
        
        print(f"\n📊 其他信息:")
        print(f"  纯问题: {branching_data['pure_question']}")
        print(f"  样本索引: {branching_data['sample_idx']}")
        print(f"  Prompt token 数: {len(branching_data['prompt_ids'])}")
        
        print("\n" + "=" * 80)
        print("验证 z 的随机性（采样两次对比）:")
        print("=" * 80)
        
        branching_data_2 = rollout._prepare_branching_data_for_single_path(
            output=output,
            prompt_ids=prompt_ids,
            pure_question=pure_question,
            sample_idx=1
        )
        
        z1 = branching_data['z_samples']
        z2 = branching_data_2['z_samples']
        
        diff = (z1 - z2).abs().mean()
        print(f"\n两次采样的平均差异: {diff:.4f}")
        
        if diff > 0.01:
            print("✅ z 采样具有随机性")
        else:
            print("⚠️  z 采样差异很小，可能存在问题")
        
        print("\n" + "=" * 80)
        print("✅ 测试完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_prepare_branching_data()
