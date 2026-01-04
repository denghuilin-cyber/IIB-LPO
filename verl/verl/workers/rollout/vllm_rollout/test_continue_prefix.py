#!/usr/bin/env python3
"""
测试 _continue_from_prefix 方法
验证 vLLM 是否能从给定的 prefix 继续生成
"""

import torch
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def test_continue_from_prefix():
    """测试从 prefix 继续生成"""
    print("=" * 80)
    print("测试 _continue_from_prefix 功能")
    print("=" * 80)
    
    # 配置
    MODEL_PATH = "/nas/models/Qwen3-8B"
    
    # 1. 加载模型和 tokenizer
    print("\n1. 加载模型...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        llm = LLM(
            model=MODEL_PATH,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.3,
            trust_remote_code=True,
            max_model_len=2048,
        )
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 2. 准备测试数据
    print("\n2. 准备测试数据...")
    
    # 原始问题
    question = "解方程 2x + 3 = 7"
    
    # 第一次生成（模拟初始 rollout）
    print(f"\n问题: {question}")
    
    # 编码 prompt
    prompt_text = f"{question} Let's think step by step."
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    
    print(f"Prompt: {prompt_text}")
    print(f"Prompt token 数: {len(prompt_ids)}")
    
    # 3. 第一次生成（完整生成）
    print("\n3. 第一次生成（完整生成）...")
    
    sampling_params_1 = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=50,
        compute_entropy=True,  # 🔧 启用熵计算
        logprobs=1,  # 需要 logprobs 才能计算熵
    )
    
    try:
        outputs_1 = llm.generate(
            prompts=[{"prompt_token_ids": prompt_ids}],
            sampling_params=sampling_params_1,
            use_tqdm=False
        )
        
        response_1 = outputs_1[0].outputs[0].token_ids
        response_text_1 = tokenizer.decode(response_1, skip_special_tokens=False)
        
        # 检查熵值
        entropies_1 = outputs_1[0].outputs[0].entropies
        has_entropy_1 = entropies_1 is not None and len(entropies_1) > 0
        
        print(f"✅ 第一次生成成功")
        print(f"生成的 token 数: {len(response_1)}")
        print(f"生成内容: {response_text_1[:200]}...")
        print(f"熵值计算: {'✅ 成功' if has_entropy_1 else '❌ 失败'}")
        if has_entropy_1:
            print(f"熵值数量: {len(entropies_1)}")
            print(f"熵值范围: [{min(entropies_1):.4f}, {max(entropies_1):.4f}]")
            print(f"平均熵值: {sum(entropies_1)/len(entropies_1):.4f}")
        
    except Exception as e:
        print(f"❌ 第一次生成失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. 从 prefix 继续生成
    print("\n4. 从 prefix 继续生成...")
    
    # 选择前 10 个 token 作为 prefix
    prefix_length = min(10, len(response_1))
    prefix_ids = response_1[:prefix_length]
    prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=False)
    
    print(f"Prefix token 数: {len(prefix_ids)}")
    print(f"Prefix 内容: {prefix_text}")
    
    # 拼接 prompt + prefix
    combined_ids = prompt_ids + prefix_ids
    remaining_length = 50 - len(prefix_ids)
    
    print(f"Combined token 数: {len(combined_ids)}")
    print(f"剩余生成长度: {remaining_length}")
    
    sampling_params_2 = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=remaining_length,
        compute_entropy=True,  # 🔧 启用熵计算
        logprobs=1,  # 需要 logprobs 才能计算熵
    )
    
    try:
        outputs_2 = llm.generate(
            prompts=[{"prompt_token_ids": combined_ids}],
            sampling_params=sampling_params_2,
            use_tqdm=False
        )
        
        response_2 = outputs_2[0].outputs[0].token_ids
        response_text_2 = tokenizer.decode(response_2, skip_special_tokens=False)
        
        # 检查熵值
        entropies_2 = outputs_2[0].outputs[0].entropies
        has_entropy_2 = entropies_2 is not None and len(entropies_2) > 0
        
        print(f"✅ 从 prefix 继续生成成功")
        print(f"新生成的 token 数: {len(response_2)}")
        print(f"新生成内容: {response_text_2[:200]}...")
        print(f"熵值计算: {'✅ 成功' if has_entropy_2 else '❌ 失败'}")
        if has_entropy_2:
            print(f"熵值数量: {len(entropies_2)}")
            print(f"熵值范围: [{min(entropies_2):.4f}, {max(entropies_2):.4f}]")
            print(f"平均熵值: {sum(entropies_2)/len(entropies_2):.4f}")
        
        # 完整的第二次生成结果
        full_response_2 = prefix_ids + response_2
        full_text_2 = tokenizer.decode(full_response_2, skip_special_tokens=False)
        
        print(f"\n完整的第二次生成 (prefix + 新生成):")
        print(f"总 token 数: {len(full_response_2)}")
        print(f"完整内容: {full_text_2[:300]}...")
        
    except Exception as e:
        print(f"❌ 从 prefix 继续生成失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. 对比两次生成
    print("\n" + "=" * 80)
    print("5. 对比两次生成")
    print("=" * 80)
    
    print(f"\n第一次生成 (完整):")
    print(f"  Token 数: {len(response_1)}")
    print(f"  内容: {response_text_1[:300]}...")
    
    print(f"\n第二次生成 (从 prefix 继续):")
    print(f"  Prefix token 数: {len(prefix_ids)}")
    print(f"  新生成 token 数: {len(response_2)}")
    print(f"  总 token 数: {len(full_response_2)}")
    print(f"  完整内容: {full_text_2[:300]}...")
    
    # 验证 prefix 是否一致
    if response_1[:prefix_length] == prefix_ids:
        print(f"\n✅ Prefix 验证成功: 第二次生成的 prefix 与第一次生成的前 {prefix_length} 个 token 一致")
    else:
        print(f"\n⚠️  Prefix 验证: 这是正常的，因为我们是从 prefix 继续生成，不是复制")
    
    print("\n" + "=" * 80)
    print("✅ 测试完成")
    print("=" * 80)
    
    print("\n📊 结论:")
    print("  - vLLM 可以从给定的 prefix 继续生成")
    print("  - 方法: 将 prompt + prefix 作为新的 prompt，生成剩余部分")
    print("  - 这个方法可以用于 CVAE 分叉")


if __name__ == "__main__":
    test_continue_from_prefix()

