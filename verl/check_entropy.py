#!/usr/bin/env python3
"""
检查 vLLM 的熵计算逻辑是否正确实现（修复版）
"""

import sys
import inspect

# 使用你的 vLLM 路径
sys.path.insert(0, '/opt/dhl_conda_envs/verl_fast/lib/python3.10/site-packages')

print("=" * 80)
print("🔍 检查 vLLM 熵计算逻辑（修复版）")
print("=" * 80)
print()

# ============================================================================
# 1. 检查 SamplingParams
# ============================================================================
print("1️⃣  检查 SamplingParams")
print("-" * 80)

try:
    from vllm.sampling_params import SamplingParams
    
    # 检查类定义
    sp_source = inspect.getsource(SamplingParams)
    
    has_compute_entropy_field = 'compute_entropy' in sp_source
    print(f"✅ SamplingParams 源码包含 compute_entropy: {has_compute_entropy_field}")
    
    # 测试实例化
    try:
        sp = SamplingParams(compute_entropy=True)
        print(f"✅ 可以实例化 SamplingParams(compute_entropy=True)")
        print(f"   sp.compute_entropy = {sp.compute_entropy}")
    except TypeError as e:
        print(f"❌ 无法实例化: {e}")
        
except Exception as e:
    print(f"❌ 检查失败: {e}")

print()

# ============================================================================
# 2. 检查 SamplingMetadata
# ============================================================================
print("2️⃣  检查 SamplingMetadata")
print("-" * 80)

try:
    from vllm.model_executor.sampling_metadata import SamplingMetadata
    
    # 检查 __init__ 方法
    init_source = inspect.getsource(SamplingMetadata.__init__)
    has_compute_entropy_param = 'compute_entropy' in init_source
    print(f"{'✅' if has_compute_entropy_param else '❌'} SamplingMetadata.__init__ 包含 compute_entropy 参数: {has_compute_entropy_param}")
    
    # 检查 prepare 方法
    if hasattr(SamplingMetadata, 'prepare'):
        prepare_source = inspect.getsource(SamplingMetadata.prepare)
        has_compute_entropy_extract = 'compute_entropy' in prepare_source
        print(f"{'✅' if has_compute_entropy_extract else '❌'} SamplingMetadata.prepare 提取 compute_entropy: {has_compute_entropy_extract}")
    else:
        print("❌ SamplingMetadata 没有 prepare 方法")
        
except Exception as e:
    print(f"❌ 检查失败: {e}")

print()

# ============================================================================
# 3. 检查 Sampler.forward() 熵计算逻辑
# ============================================================================
print("3️⃣  检查 Sampler.forward() 熵计算逻辑")
print("-" * 80)

try:
    from vllm.model_executor.layers.sampler import Sampler
    
    # 检查 forward 方法
    forward_source = inspect.getsource(Sampler.forward)
    
    # 关键检查点
    checks = {
        "计算 probs": "probs = torch.softmax" in forward_source,
        "计算 logprobs": "logprobs = torch.log_softmax" in forward_source,
        "检查 compute_entropy": "compute_entropy" in forward_source,
        "计算熵值": "entropies_tensor" in forward_source,
        "熵公式": "-torch.sum(probs * logprobs" in forward_source,
        "传递熵值给 _build_sampler_output": "entropies_tensor=" in forward_source,
    }
    
    print("关键代码检查：")
    for check_name, result in checks.items():
        print(f"  {'✅' if result else '❌'} {check_name}: {result}")
    
    print()
    
    # 如果缺少关键逻辑，打印相关代码片段
    if not checks["计算熵值"]:
        print("⚠️  Sampler.forward() 缺少熵计算逻辑！")
        print()
        print("应该包含类似以下代码：")
        print("""
        # 计算熵（如果需要）
        entropies_tensor = None
        if sampling_metadata.compute_entropy:
            entropies_tensor = -torch.sum(probs * logprobs, dim=-1)
        """)
    
    if not checks["传递熵值给 _build_sampler_output"]:
        print("⚠️  Sampler.forward() 没有传递 entropies_tensor 给 _build_sampler_output！")
        print()
        print("应该在 return 语句中包含：")
        print("""
        return _build_sampler_output(
            ...,
            entropies_tensor=entropies_tensor)
        """)
        
except Exception as e:
    print(f"❌ 检查失败: {e}")
    import traceback
    traceback.print_exc()

print()

# ============================================================================
# 4. 检查 _build_sampler_output 函数（关键！）
# ============================================================================
print("4️⃣  检查 _build_sampler_output() 熵值提取逻辑")
print("-" * 80)

try:
    from vllm.model_executor.layers import sampler
    
    # 获取 _build_sampler_output 函数的源码
    build_source = inspect.getsource(sampler._build_sampler_output)
    
    # 关键检查点
    checks = {
        "函数签名包含 entropies_tensor 参数": "entropies_tensor" in build_source.split('\n')[0],
        "初始化 entropy_idx": "entropy_idx = 0" in build_source,
        "提取 entropy_value": "entropy_value" in build_source,
        "检查 entropies_tensor 是否为空": "if entropies_tensor is not None" in build_source,
        "从 tensor 提取值": ".item()" in build_source,
        "传递给 SequenceOutput": "entropy=" in build_source,
        "返回 entropies": "entropies=" in build_source and "SamplerOutput" in build_source,
    }
    
    print("关键代码检查：")
    all_passed = True
    for check_name, result in checks.items():
        print(f"  {'✅' if result else '❌'} {check_name}: {result}")
        if not result:
            all_passed = False
    
    print()
    
    if all_passed:
        print("✅ _build_sampler_output() 包含完整的熵值提取逻辑")
    else:
        print("⚠️  _build_sampler_output() 缺少部分熵值提取逻辑")
        print()
        print("应该包含类似以下代码：")
        print("""
        def _build_sampler_output(
            ...,
            entropies_tensor: Optional[torch.Tensor] = None,  # 参数
        ) -> SamplerOutput:
            ...
            # 准备熵值索引
            entropy_idx = 0
            
            for ... in zip(...):
                ...
                for parent_id, next_token_id, logprobs in zip(...):
                    # 提取熵值
                    entropy_value = None
                    if entropies_tensor is not None and entropy_idx < len(entropies_tensor):
                        entropy_value = entropies_tensor[entropy_idx].item()
                        entropy_idx += 1
                    
                    seq_outputs.append(
                        SequenceOutput(..., entropy=entropy_value))
            
            return SamplerOutput(
                ...,
                entropies=entropies_tensor)
        """)
        
except Exception as e:
    print(f"❌ 检查失败: {e}")
    import traceback
    traceback.print_exc()

print()

# ============================================================================
# 5. 检查 Sequence.append_token_id
# ============================================================================
print("5️⃣  检查 Sequence.append_token_id")
print("-" * 80)

try:
    from vllm.sequence import Sequence
    
    append_source = inspect.getsource(Sequence.append_token_id)
    
    has_entropy_param = 'entropy' in append_source
    has_entropy_pass = 'entropy=' in append_source
    
    print(f"{'✅' if has_entropy_param else '❌'} append_token_id 有 entropy 参数: {has_entropy_param}")
    print(f"{'✅' if has_entropy_pass else '❌'} 传递 entropy 给 SequenceData: {has_entropy_pass}")
    
except Exception as e:
    print(f"❌ 检查失败: {e}")

print()

# ============================================================================
# 6. 检查 output_processor
# ============================================================================
print("6️⃣  检查 output_processor")
print("-" * 80)

try:
    from vllm.engine.output_processor.single_step import SingleStepOutputProcessor
    
    # 检查 _process_sequence_group_outputs
    if hasattr(SingleStepOutputProcessor, '_process_sequence_group_outputs'):
        process_source = inspect.getsource(SingleStepOutputProcessor._process_sequence_group_outputs)
        
        has_entropy_pass = 'entropy=' in process_source
        print(f"{'✅' if has_entropy_pass else '❌'} _process_sequence_group_outputs 传递 entropy: {has_entropy_pass}")
    else:
        print("❌ 找不到 _process_sequence_group_outputs 方法")
        
except Exception as e:
    print(f"❌ 检查失败: {e}")

print()

# ============================================================================
# 总结
# ============================================================================
print("=" * 80)
print("📊 诊断总结")
print("=" * 80)
print()
print("如果上面有任何 ❌，说明 vLLM 的熵计算逻辑不完整。")
print()
print("关键步骤：")
print("  1. Sampler.forward() 计算 entropies_tensor")
print("  2. Sampler.forward() 传递 entropies_tensor 给 _build_sampler_output()")
print("  3. _build_sampler_output() 从 entropies_tensor 提取每个 token 的熵值")
print("  4. _build_sampler_output() 将熵值传递给 SequenceOutput")
print("  5. _build_sampler_output() 返回 SamplerOutput 时包含 entropies")
print()
print("=" * 80)

