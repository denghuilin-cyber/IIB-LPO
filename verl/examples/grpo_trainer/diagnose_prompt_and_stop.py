#!/usr/bin/env python3
"""
诊断脚本：对比 VERL 原生数据集 vs 自定义数据集的 Prompt 和停止条件

目的：
1. 对比 RLHFDataset 和 MultiDatasetWithCOT 生成的 prompt
2. 检查 special tokens 是否正确
3. 检查 SamplingParams 的停止条件配置
4. 找出为什么分叉路径无法正常停止

使用方法：
    python diagnose_prompt_and_stop.py
"""

import os
import sys
import torch
from transformers import AutoTokenizer
from vllm import SamplingParams

print("="*100)
print("🔍 Prompt 和停止条件诊断脚本")
print("="*100)

# ============================================================================
# 1. 配置
# ============================================================================
print("\n[步骤 1] 加载配置...")

ACTOR_MODEL_PATH = "/nas/models/Qwen3-4B-Instruct-2507"
GSMA8K_PATH = "/nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet"
MATH_PATH = "/nas/dhl/Datasets/my_Datasets/MATH/train.parquet"

# ============================================================================
# 2. 加载 Tokenizer
# ============================================================================
print("\n[步骤 2] 加载 Tokenizer...")

try:
    tokenizer = AutoTokenizer.from_pretrained(ACTOR_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✅ Tokenizer 加载成功")
except Exception as e:
    print(f"❌ Tokenizer 加载失败: {e}")
    sys.exit(1)

# 打印 tokenizer 的关键信息
print("\n" + "-"*100)
print("📋 Tokenizer 配置信息:")
print("-"*100)
print(f"  模型路径: {ACTOR_MODEL_PATH}")
print(f"  vocab_size: {tokenizer.vocab_size}")
print(f"  pad_token: {repr(tokenizer.pad_token)} (ID: {tokenizer.pad_token_id})")
print(f"  eos_token: {repr(tokenizer.eos_token)} (ID: {tokenizer.eos_token_id})")
print(f"  bos_token: {repr(tokenizer.bos_token)} (ID: {getattr(tokenizer, 'bos_token_id', None)})")
print(f"  支持 chat_template: {hasattr(tokenizer, 'apply_chat_template')}")

# 检查 Qwen 特殊 tokens
qwen_special_tokens = ['<|im_start|>', '<|im_end|>', '<|endoftext|>']
print(f"\n  Qwen 特殊 tokens 检查:")
for token in qwen_special_tokens:
    try:
        token_id = tokenizer.encode(token, add_special_tokens=False)
        print(f"    {repr(token)}: {token_id}")
    except:
        print(f"    {repr(token)}: 不存在")

print("-"*100)

# ============================================================================
# 3. 测试 Chat Template
# ============================================================================
print("\n[步骤 3] 测试 Chat Template...")

test_question = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"

print(f"\n📝 测试问题: {test_question[:100]}...")

# 3.1 测试不带 COT 的 chat template
print("\n➡️  测试 1: 直接使用 chat template（不带 COT）")
print("-"*100)

if hasattr(tokenizer, 'apply_chat_template'):
    messages = [{"role": "user", "content": test_question}]
    prompt_text_1 = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print(f"生成的 prompt (前 500 字符):")
    print(prompt_text_1[:500])
    print(f"\n生成的 prompt (后 200 字符):")
    print(prompt_text_1[-200:])
    
    # 检查 special tokens
    print(f"\n包含 '<|im_start|>': {('<|im_start|>' in prompt_text_1)}")
    print(f"包含 '<|im_end|>': {('<|im_end|>' in prompt_text_1)}")
    
    # Tokenize
    tokens_1 = tokenizer.encode(prompt_text_1, add_special_tokens=False)
    print(f"\nToken 数量: {len(tokens_1)}")
    print(f"前 10 个 tokens: {tokens_1[:10]}")
    print(f"后 10 个 tokens: {tokens_1[-10:]}")
else:
    print("❌ Tokenizer 不支持 chat_template")

# 3.2 测试带 COT 的场景
print("\n➡️  测试 2: 模拟 COT 增强后的 prompt")
print("-"*100)

cot_example = """Here is a reference example that demonstrates the problem-solving approach:

<Example>
Question: John has 2 umbrellas in his house and 1 in the car. If they cost $8 each, how much did he pay in total?

Step-by-step Solution:
He has 2+1=3 umbrellas
That means he paid 3*8=$24

Final Answer: 24
</Example>

Now, please solve the following problem using similar reasoning:"""

# 模拟 COT 增强的拼接
augmented_text = cot_example + "\n\n" + test_question

print(f"COT 增强后的文本 (前 500 字符):")
print(augmented_text[:500])

# 方式 A: 直接 encode（原有方式 - 错误）
print("\n🔴 方式 A: 直接 encode (add_special_tokens=False) - 当前的错误方式")
tokens_2a = tokenizer.encode(augmented_text, add_special_tokens=False)
decoded_2a = tokenizer.decode(tokens_2a, skip_special_tokens=False)
print(f"  Token 数量: {len(tokens_2a)}")
print(f"  解码后包含 '<|im_start|>': {('<|im_start|>' in decoded_2a)}")
print(f"  解码后包含 '<|im_end|>': {('<|im_end|>' in decoded_2a)}")
print(f"  解码后的文本 (前 300 字符):")
print(f"  {decoded_2a[:300]}")

# 方式 B: 重新应用 chat template（修复方式）
print("\n🟢 方式 B: 重新应用 chat template - 修复方式")
if hasattr(tokenizer, 'apply_chat_template'):
    messages_2b = [{"role": "user", "content": augmented_text}]
    prompt_text_2b = tokenizer.apply_chat_template(
        messages_2b,
        tokenize=False,
        add_generation_prompt=True
    )
    tokens_2b = tokenizer.encode(prompt_text_2b, add_special_tokens=False)
    decoded_2b = tokenizer.decode(tokens_2b, skip_special_tokens=False)
    
    print(f"  Token 数量: {len(tokens_2b)}")
    print(f"  解码后包含 '<|im_start|>': {('<|im_start|>' in decoded_2b)}")
    print(f"  解码后包含 '<|im_end|>': {('<|im_end|>' in decoded_2b)}")
    print(f"  解码后的文本 (前 300 字符):")
    print(f"  {decoded_2b[:300]}")
    print(f"\n  解码后的文本 (后 200 字符):")
    print(f"  {decoded_2b[-200:]}")

print("-"*100)

# ============================================================================
# 4. 测试 SamplingParams
# ============================================================================
print("\n[步骤 4] 测试 SamplingParams 配置...")
print("-"*100)

# 4.1 默认配置（可能有问题）
print("\n🔴 配置 A: 默认 SamplingParams (可能缺少停止条件)")
params_a = SamplingParams(
    n=1,
    temperature=0.8,
    top_p=0.9,
    top_k=-1,
    max_tokens=8192,
)

print(f"  max_tokens: {params_a.max_tokens}")
print(f"  temperature: {params_a.temperature}")
print(f"  stop: {params_a.stop}")
print(f"  stop_token_ids: {params_a.stop_token_ids}")
print(f"  ignore_eos: {params_a.ignore_eos}")

if hasattr(params_a, '_all_stop_token_ids'):
    print(f"  _all_stop_token_ids: {params_a._all_stop_token_ids}")
else:
    print(f"  _all_stop_token_ids: 不存在")

# 4.2 明确设置 stop_token_ids（修复方式）
print("\n🟢 配置 B: 明确设置 stop_token_ids (修复方式)")
params_b = SamplingParams(
    n=1,
    temperature=0.8,
    top_p=0.9,
    top_k=-1,
    max_tokens=8192,
    stop_token_ids=[tokenizer.eos_token_id],  # ← 明确设置
)

print(f"  max_tokens: {params_b.max_tokens}")
print(f"  temperature: {params_b.temperature}")
print(f"  stop: {params_b.stop}")
print(f"  stop_token_ids: {params_b.stop_token_ids}")
print(f"  ignore_eos: {params_b.ignore_eos}")

if hasattr(params_b, '_all_stop_token_ids'):
    print(f"  _all_stop_token_ids: {params_b._all_stop_token_ids}")
    print(f"  包含 eos_token_id ({tokenizer.eos_token_id}): {tokenizer.eos_token_id in params_b._all_stop_token_ids}")
else:
    print(f"  _all_stop_token_ids: 不存在")

# 4.3 添加停止词
print("\n🟢 配置 C: 添加额外的停止词 (更安全)")
params_c = SamplingParams(
    n=1,
    temperature=0.8,
    top_p=0.9,
    top_k=-1,
    max_tokens=8192,
    stop_token_ids=[tokenizer.eos_token_id],
    stop=["####"],  # GSM8K 的答案格式
)

print(f"  max_tokens: {params_c.max_tokens}")
print(f"  stop: {params_c.stop}")
print(f"  stop_token_ids: {params_c.stop_token_ids}")

if hasattr(params_c, '_all_stop_token_ids'):
    print(f"  _all_stop_token_ids: {params_c._all_stop_token_ids}")

print("-"*100)

# ============================================================================
# 5. 加载并对比数据集
# ============================================================================
print("\n[步骤 5] 加载并对比数据集...")

# 5.1 加载自定义数据集
print("\n➡️  加载 MultiDatasetWithCOT...")
try:
    # 确保导入路径正确
    sys.path.insert(0, '/nas/dhl/verl')
    from examples.grpo_trainer.multi_dataset_with_cot import MultiDatasetWithCOT
    
    mock_config = {
        'gsm8k_path': GSMA8K_PATH,
        'math_path': MATH_PATH,
        'max_prompt_length': 2048,
    }
    
    custom_dataset = MultiDatasetWithCOT(
        tokenizer=tokenizer,
        config=mock_config,
        is_train=True
    )
    
    print(f"✅ MultiDatasetWithCOT 加载成功")
    print(f"  样本数量: {len(custom_dataset)}")
    
    # 获取一个样本
    sample_custom = custom_dataset[0]
    print(f"\n  样本字段: {list(sample_custom.keys())}")
    print(f"  dataset_name: {sample_custom.get('dataset_name', 'N/A')}")
    
    # 解码 input_ids
    if 'input_ids' in sample_custom:
        input_ids = sample_custom['input_ids']
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        
        # 找到非 padding 部分
        if 'attention_mask' in sample_custom:
            attention_mask = sample_custom['attention_mask']
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = attention_mask.tolist()
            # 找到第一个 1 的位置
            first_valid = next((i for i, v in enumerate(attention_mask) if v == 1), 0)
            valid_ids = input_ids[first_valid:]
        else:
            valid_ids = input_ids
        
        decoded = tokenizer.decode(valid_ids, skip_special_tokens=False)
        
        print(f"\n  📝 自定义数据集的 prompt (前 500 字符):")
        print(f"  {decoded[:500]}")
        print(f"\n  📝 自定义数据集的 prompt (后 200 字符):")
        print(f"  {decoded[-200:]}")
        
        print(f"\n  包含 '<|im_start|>': {('<|im_start|>' in decoded)}")
        print(f"  包含 '<|im_end|>': {('<|im_end|>' in decoded)}")
        print(f"  包含 '<|endoftext|>': {('<|endoftext|>' in decoded)}")

except Exception as e:
    print(f"❌ MultiDatasetWithCOT 加载失败: {e}")
    import traceback
    traceback.print_exc()

# 5.2 尝试加载 VERL 原生数据集（如果有的话）
print("\n➡️  对比 VERL 原生 RLHFDataset (如果可用)...")
try:
    from verl.utils.dataset import RLHFDataset
    
    # 这里需要你提供一个原生数据集的路径
    # native_dataset = RLHFDataset(...)
    # print(f"✅ RLHFDataset 加载成功")
    
    print("  ⚠️ 需要提供原生数据集路径才能对比")
    
except ImportError:
    print("  ⚠️ RLHFDataset 不可用")

# ============================================================================
# 6. 总结诊断结果
# ============================================================================
print("\n" + "="*100)
print("📊 诊断总结")
print("="*100)

print("""
关键发现：
1. Tokenizer 配置
   - EOS token ID 是否正确？
   - 是否支持 chat_template？

2. Prompt 格式
   - 方式 A (直接 encode): 缺少 special tokens
   - 方式 B (重新应用 chat template): 包含完整的 special tokens

3. SamplingParams 配置
   - 配置 A (默认): stop_token_ids 可能为空
   - 配置 B (明确设置): stop_token_ids 包含 EOS token
   - 配置 C (添加停止词): 更加安全

4. 自定义数据集
   - 检查生成的 prompt 是否包含 special tokens
   - 对比与原生数据集的差异

建议修复方案：
1. 在 vllm_rollout_spmd.py 的 SamplingParams 初始化时明确设置 stop_token_ids
2. 在 grpo_cot_augmentation.py 的 COT 增强后重新应用 chat template
3. 确保 _continue_from_prefix 正确继承停止条件
""")

print("="*100)
print("✅ 诊断完成！请查看上述输出，找出问题所在。")
print("="*100)

