# #!/usr/bin/env python3
# """
# 分析 COT 增强后的 prompt 长度

# 检查拼接后的 prompt 长度统计，并保存到 JSONL 文件

# 用法:
#     python analyze_cot_prompts.py
# """

# import sys
# sys.path.insert(0, '/nas/dhl/verl')

# from examples.grpo_trainer.multi_dataset_with_cot import MultiDatasetWithCOT
# from transformers import AutoTokenizer
# from torch.utils.data import DataLoader
# from verl.utils.dataset.rl_dataset import collate_fn
# from verl import DataProto
# import json
# import numpy as np

# # 配置
# ACTOR_MODEL_PATH = "/nas/models/Qwen3-8B"
# TRAIN_FILES_GSM8K = "/nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet"
# TRAIN_FILES_MATH = "/nas/dhl/Datasets/my_Datasets/MATH/train.parquet"
# GSM8K_COT = "/nas/dhl/CVAE/Datasets/GSM8K/processed/train_k_shot_GSM8K.jsonl"
# MATH_COT = "/nas/dhl/CVAE/Datasets/MATH/processed/train_k_shot_MATH.jsonl"
# OUTPUT_FILE = "/nas/dhl/verl/examples/grpo_trainer/cot_augmented_prompts.jsonl"
# NUM_BATCHES = 50  # 分析前10个batch（共80个样本）
# BATCH_SIZE = 8
# N_ROLLOUTS = 4


# def main():
#     print("="*80)
#     print("🔍 COT增强后的Prompt长度分析")
#     print("="*80)
    
#     # 加载 tokenizer
#     print("\n⏳ 加载 tokenizer...")
#     tokenizer = AutoTokenizer.from_pretrained(ACTOR_MODEL_PATH, trust_remote_code=True)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#     print("✓ Tokenizer 加载成功")
    
#     # 创建数据集
#     print("\n⏳ 创建数据集...")
#     config = {
#         'gsm8k_path': TRAIN_FILES_GSM8K,
#         'math_path': TRAIN_FILES_MATH,
#         'max_prompt_length': 2048,
#     }
    
#     dataset = MultiDatasetWithCOT(
#         tokenizer=tokenizer,
#         config=config,
#         is_train=True
#     )
#     print(f"✓ 数据集大小: {len(dataset)}")
    
#     # 创建 DataLoader
#     print("\n⏳ 创建 DataLoader...")
#     dataloader = DataLoader(
#         dataset=dataset,
#         batch_size=BATCH_SIZE,
#         collate_fn=collate_fn,
#         shuffle=False
#     )
    
#     # 初始化 COT 加载器
#     print("\n⏳ 初始化 COT 加载器...")
#     sys.path.insert(0, '/nas/dhl/verl/examples/grpo_trainer')
#     from multi_dataset_simple_loader import initialize_multi_dataset_simple_cot_loader, get_multi_dataset_simple_cot_examples
    
#     cot_loader = initialize_multi_dataset_simple_cot_loader(
#         cot_file_mapping={'gsm8k': GSM8K_COT, 'math': MATH_COT},
#         use_full_cot=True,
#         skip_on_mismatch=True,
#         verbose=False,
#     )
    
#     # 初始化 COT 增强器
#     from verl.utils.grpo_cot_augmentation import GRPOCOTAugmenter
    
#     def cot_getter_wrapper(batch, prompt_idx, num_repeats):
#         return get_multi_dataset_simple_cot_examples(batch, prompt_idx, num_repeats, tokenizer=tokenizer)
    
#     cot_augmenter = GRPOCOTAugmenter(
#         cot_examples_getter=cot_getter_wrapper,
#         tokenizer=tokenizer,
#         num_repeats=N_ROLLOUTS,
#         enable=True,
#         debug_print_augmented_prompts=False,  # 关闭调试打印
#     )
#     print("✓ COT 增强器初始化成功")
    
#     # 收集数据
#     print(f"\n⏳ 分析前 {NUM_BATCHES} 个batch...")
    
#     all_results = []
#     length_stats = {
#         'original': [],
#         'augmented': [],
#         'cot_example': []
#     }
    
#     batch_count = 0
#     for batch_dict in dataloader:
#         if batch_count >= NUM_BATCHES:
#             break
        
#         batch = DataProto.from_single_dict(batch_dict)
#         batch.non_tensor_batch["uid"] = ['uid'] * len(batch.batch)
        
#         # Pop出生成batch
#         reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()
#         batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
#         existing_batch_keys = [k for k in batch_keys_to_pop if k in batch.batch.keys()]
#         non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        
#         gen_batch = batch.pop(
#             batch_keys=existing_batch_keys,
#             non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
#         )
        
#         # 保存原始长度
#         original_input_ids = gen_batch.batch['input_ids'].clone()
#         original_masks = gen_batch.batch['attention_mask'].clone()
        
#         # Repeat
#         gen_batch = gen_batch.repeat(repeat_times=N_ROLLOUTS, interleave=True)
        
#         # COT 增强
#         gen_batch_augmented = cot_augmenter.augment(gen_batch)
        
#         # 分析每个样本
#         for i in range(len(original_input_ids)):
#             # 原始prompt
#             orig_mask = original_masks[i]
#             orig_valid_ids = original_input_ids[i][orig_mask.bool()]
#             orig_text = tokenizer.decode(orig_valid_ids, skip_special_tokens=True)
#             orig_length = len(orig_valid_ids)
            
#             # 增强后的prompt（第一个rollout）
#             aug_idx = i * N_ROLLOUTS  # 第一个rollout
#             aug_input_ids = gen_batch_augmented.batch['input_ids'][aug_idx]
#             aug_mask = gen_batch_augmented.batch['attention_mask'][aug_idx]
#             aug_valid_ids = aug_input_ids[aug_mask.bool()]
#             aug_text = tokenizer.decode(aug_valid_ids, skip_special_tokens=True)
#             aug_length = len(aug_valid_ids)
            
#             # COT example长度
#             cot_length = aug_length - orig_length
            
#             # 统计
#             length_stats['original'].append(orig_length)
#             length_stats['augmented'].append(aug_length)
#             length_stats['cot_example'].append(cot_length)
            
#             # 保存详细信息
#             dataset_name = gen_batch_augmented.non_tensor_batch['dataset_name'][aug_idx]
#             question = gen_batch_augmented.non_tensor_batch['question'][aug_idx]
            
#             result = {
#                 'sample_index': batch_count * BATCH_SIZE + i,
#                 'dataset_name': dataset_name,
#                 'question': question,
#                 'original_prompt': orig_text,
#                 'augmented_prompt': aug_text,
#                 'original_length': orig_length,
#                 'augmented_length': aug_length,
#                 'cot_length': cot_length,
#             }
#             all_results.append(result)
        
#         batch_count += 1
#         print(f"  处理batch {batch_count}/{NUM_BATCHES}...")
    
#     # 保存到 JSONL
#     print(f"\n⏳ 保存到 {OUTPUT_FILE}...")
#     with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
#         for result in all_results:
#             f.write(json.dumps(result, ensure_ascii=False) + '\n')
#     print(f"✓ 已保存 {len(all_results)} 个样本")
    
#     # 统计分析
#     print("\n" + "="*80)
#     print("📊 长度统计分析")
#     print("="*80)
    
#     original_lengths = np.array(length_stats['original'])
#     augmented_lengths = np.array(length_stats['augmented'])
#     cot_lengths = np.array(length_stats['cot_example'])
    
#     print(f"\n📌 原始Prompt长度（未增强）:")
#     print(f"   平均: {original_lengths.mean():.1f} tokens")
#     print(f"   最小: {original_lengths.min()} tokens")
#     print(f"   最大: {original_lengths.max()} tokens")
#     print(f"   中位数: {np.median(original_lengths):.1f} tokens")
    
#     print(f"\n✨ 增强后Prompt长度（COT + 原始问题）:")
#     print(f"   平均: {augmented_lengths.mean():.1f} tokens")
#     print(f"   最小: {augmented_lengths.min()} tokens")
#     print(f"   最大: {augmented_lengths.max()} tokens")
#     print(f"   中位数: {np.median(augmented_lengths):.1f} tokens")
    
#     print(f"\n📚 COT Example长度:")
#     print(f"   平均: {cot_lengths.mean():.1f} tokens")
#     print(f"   最小: {cot_lengths.min()} tokens")
#     print(f"   最大: {cot_lengths.max()} tokens")
#     print(f"   中位数: {np.median(cot_lengths):.1f} tokens")
    
#     # 按数据集分组统计
#     print(f"\n📊 按数据集分组:")
#     for ds_name in ['gsm8k', 'math']:
#         ds_results = [r for r in all_results if r['dataset_name'] == ds_name]
#         if ds_results:
#             ds_aug_lengths = [r['augmented_length'] for r in ds_results]
#             print(f"   {ds_name:10s}: 样本数={len(ds_results):3d}, 平均长度={np.mean(ds_aug_lengths):.1f} tokens")
    
#     # 长度分布
#     print(f"\n📊 增强后长度分布:")
#     bins = [0, 512, 1024, 1536, 2048, 3000, 5000, 10000]
#     for i in range(len(bins)-1):
#         count = np.sum((augmented_lengths >= bins[i]) & (augmented_lengths < bins[i+1]))
#         percentage = count / len(augmented_lengths) * 100
#         print(f"   {bins[i]:5d} - {bins[i+1]:5d} tokens: {count:4d} 样本 ({percentage:5.1f}%)")
    
#     # 超长样本
#     max_allowed = 2048
#     overlong = augmented_lengths > max_allowed
#     num_overlong = np.sum(overlong)
    
#     if num_overlong > 0:
#         print(f"\n⚠️  超过 {max_allowed} tokens 的样本:")
#         print(f"   数量: {num_overlong} / {len(augmented_lengths)} ({num_overlong/len(augmented_lengths)*100:.1f}%)")
#         print(f"   最长: {augmented_lengths.max()} tokens")
        
#         # 显示几个超长样本
#         overlong_indices = np.where(overlong)[0][:3]
#         for idx in overlong_indices:
#             r = all_results[idx]
#             print(f"\n   样本 {r['sample_index']} ({r['dataset_name']}):")
#             print(f"     长度: {r['augmented_length']} tokens")
#             print(f"     问题: {r['question'][:60]}...")
    
#     # 建议
#     print(f"\n" + "="*80)
#     print(f"💡 建议")
#     print(f"="*80)
    
#     avg_aug_length = augmented_lengths.mean()
    
#     if avg_aug_length > 2048:
#         print(f"\n⚠️  平均长度 {avg_aug_length:.0f} 超过了 max_prompt_length=2048")
#         print(f"   建议:")
#         print(f"   1. 增大 max_prompt_length 到 {int(augmented_lengths.max() * 1.1)}")
#         print(f"   2. 或者使用更短的 COT example")
#         print(f"   3. 或者过滤掉超长样本")
#     elif avg_aug_length > 1536:
#         print(f"\n✅ 平均长度 {avg_aug_length:.0f} 可以使用 max_prompt_length=2048")
#         print(f"   但建议设置为 {int(augmented_lengths.max() * 1.1)} 以容纳所有样本")
#     else:
#         print(f"\n✅ 平均长度 {avg_aug_length:.0f} 很合理")
#         print(f"   max_prompt_length=2048 足够")
#         if avg_aug_length < 1024:
#             print(f"   甚至可以降到 max_prompt_length=1536 节省内存")
    
#     print(f"\n" + "="*80)
#     print(f"✅ 分析完成")
#     print(f"="*80)
#     print(f"\n详细数据已保存到: {OUTPUT_FILE}")
#     print(f"\n查看示例:")
#     print(f"  cat {OUTPUT_FILE} | head -1 | jq .")
#     print(f"\n查看最长的样本:")
#     print(f"  cat {OUTPUT_FILE} | jq 'select(.augmented_length > 2000)' | head -3")
#     print(f"\n统计不同数据集:")
#     print(f"  cat {OUTPUT_FILE} | jq -r '.dataset_name' | sort | uniq -c")
#     print("="*80 + "\n")


# if __name__ == '__main__':
#     main()



#!/usr/bin/env python3
"""
分析 COT 增强后的 prompt 长度

检查拼接后的 prompt 长度统计，并保存到 JSONL 文件

用法:
    python analyze_cot_prompts.py
"""

import sys
sys.path.insert(0, '/nas/dhl/verl')

from examples.grpo_trainer.multi_dataset_with_cot import MultiDatasetWithCOT
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from verl.utils.dataset.rl_dataset import collate_fn
from verl import DataProto
import json
import numpy as np

# 配置
ACTOR_MODEL_PATH = "/nas/models/Qwen3-8B"
TRAIN_FILES_GSM8K = "/nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet"
TRAIN_FILES_MATH = "/nas/dhl/Datasets/my_Datasets/MATH/train.parquet"
GSM8K_COT = "/nas/dhl/CVAE/Datasets/GSM8K/processed/train_k_shot_GSM8K.jsonl"
MATH_COT = "/nas/dhl/CVAE/Datasets/MATH/processed/train_k_shot_MATH.jsonl"
OUTPUT_FILE = "/nas/dhl/verl/examples/grpo_trainer/cot_augmented_prompts_all.jsonl"
BATCH_SIZE = 8
N_ROLLOUTS = 4
ANALYZE_ALL = True  # True=分析所有样本, False=只分析前几个batch
NUM_BATCHES_SAMPLE = 10  # 如果ANALYZE_ALL=False，分析的batch数


def main():
    print("="*80)
    print("🔍 COT增强后的Prompt长度分析（所有数据集）")
    print("="*80)
    print(f"\n分析模式: {'所有样本' if ANALYZE_ALL else f'前{NUM_BATCHES_SAMPLE}个batch'}")
    
    # 加载 tokenizer
    print("\n⏳ 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ACTOR_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer 加载成功")
    
    # 创建数据集
    print("\n⏳ 创建数据集...")
    config = {
        'gsm8k_path': TRAIN_FILES_GSM8K,
        'math_path': TRAIN_FILES_MATH,
        'max_prompt_length': 2048,
    }
    
    dataset = MultiDatasetWithCOT(
        tokenizer=tokenizer,
        config=config,
        is_train=True
    )
    print(f"✓ 数据集大小: {len(dataset)}")
    
    # 创建 DataLoader
    print("\n⏳ 创建 DataLoader...")
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        shuffle=False
    )
    
    # 初始化 COT 加载器
    print("\n⏳ 初始化 COT 加载器...")
    sys.path.insert(0, '/nas/dhl/verl/examples/grpo_trainer')
    from multi_dataset_simple_loader import initialize_multi_dataset_simple_cot_loader, get_multi_dataset_simple_cot_examples
    
    cot_loader = initialize_multi_dataset_simple_cot_loader(
        cot_file_mapping={'gsm8k': GSM8K_COT, 'math': MATH_COT},
        use_full_cot=True,
        skip_on_mismatch=True,
        verbose=False,
    )
    
    # 初始化 COT 增强器
    from verl.utils.grpo_cot_augmentation import GRPOCOTAugmenter
    
    def cot_getter_wrapper(batch, prompt_idx, num_repeats):
        return get_multi_dataset_simple_cot_examples(batch, prompt_idx, num_repeats, tokenizer=tokenizer)
    
    cot_augmenter = GRPOCOTAugmenter(
        cot_examples_getter=cot_getter_wrapper,
        tokenizer=tokenizer,
        num_repeats=N_ROLLOUTS,
        enable=True,
        debug_print_augmented_prompts=False,  # 关闭调试打印
    )
    print("✓ COT 增强器初始化成功")
    
    # 收集数据
    total_batches = len(dataloader)
    max_batches = total_batches if ANALYZE_ALL else NUM_BATCHES_SAMPLE
    
    print(f"\n⏳ 分析{'所有' if ANALYZE_ALL else f'前{max_batches}个'} batch（共{total_batches}个batch，约{total_batches * BATCH_SIZE}个样本）...")
    
    all_results = []
    length_stats = {
        'original': [],
        'augmented': [],
        'cot_example': []
    }
    
    batch_count = 0
    for batch_dict in dataloader:
        if batch_count >= max_batches:
            break
        
        batch = DataProto.from_single_dict(batch_dict)
        batch.non_tensor_batch["uid"] = ['uid'] * len(batch.batch)
        
        # Pop出生成batch
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        existing_batch_keys = [k for k in batch_keys_to_pop if k in batch.batch.keys()]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        
        gen_batch = batch.pop(
            batch_keys=existing_batch_keys,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )
        
        # 保存原始长度
        original_input_ids = gen_batch.batch['input_ids'].clone()
        original_masks = gen_batch.batch['attention_mask'].clone()
        
        # Repeat
        gen_batch = gen_batch.repeat(repeat_times=N_ROLLOUTS, interleave=True)
        
        # COT 增强
        gen_batch_augmented = cot_augmenter.augment(gen_batch)
        
        # 分析每个样本
        for i in range(len(original_input_ids)):
            # 原始prompt
            orig_mask = original_masks[i]
            orig_valid_ids = original_input_ids[i][orig_mask.bool()]
            orig_text = tokenizer.decode(orig_valid_ids, skip_special_tokens=True)
            orig_length = len(orig_valid_ids)
            
            # 增强后的prompt（第一个rollout）
            aug_idx = i * N_ROLLOUTS  # 第一个rollout
            aug_input_ids = gen_batch_augmented.batch['input_ids'][aug_idx]
            aug_mask = gen_batch_augmented.batch['attention_mask'][aug_idx]
            aug_valid_ids = aug_input_ids[aug_mask.bool()]
            aug_text = tokenizer.decode(aug_valid_ids, skip_special_tokens=True)
            aug_length = len(aug_valid_ids)
            
            # COT example长度
            cot_length = aug_length - orig_length
            
            # 统计
            length_stats['original'].append(orig_length)
            length_stats['augmented'].append(aug_length)
            length_stats['cot_example'].append(cot_length)
            
            # 保存详细信息
            dataset_name = gen_batch_augmented.non_tensor_batch['dataset_name'][aug_idx]
            question = gen_batch_augmented.non_tensor_batch['question'][aug_idx]
            
            result = {
                'sample_index': batch_count * BATCH_SIZE + i,
                'dataset_name': dataset_name,
                'question': question,
                'original_prompt': orig_text,
                'augmented_prompt': aug_text,
                'original_length': orig_length,
                'augmented_length': aug_length,
                'cot_length': cot_length,
            }
            all_results.append(result)
        
        batch_count += 1
        # 显示进度
        if batch_count % 100 == 0 or batch_count == max_batches:
            print(f"  处理进度: {batch_count}/{max_batches} batches ({batch_count/max_batches*100:.1f}%)...")
    
    # 保存到 JSONL
    print(f"\n⏳ 保存到 {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(f"✓ 已保存 {len(all_results)} 个样本")
    
    # 统计分析
    print("\n" + "="*80)
    print("📊 长度统计分析")
    print("="*80)
    
    original_lengths = np.array(length_stats['original'])
    augmented_lengths = np.array(length_stats['augmented'])
    cot_lengths = np.array(length_stats['cot_example'])
    
    print(f"\n📌 原始Prompt长度（未增强）:")
    print(f"   平均: {original_lengths.mean():.1f} tokens")
    print(f"   最小: {original_lengths.min()} tokens")
    print(f"   最大: {original_lengths.max()} tokens")
    print(f"   中位数: {np.median(original_lengths):.1f} tokens")
    
    print(f"\n✨ 增强后Prompt长度（COT + 原始问题）:")
    print(f"   平均: {augmented_lengths.mean():.1f} tokens")
    print(f"   最小: {augmented_lengths.min()} tokens")
    print(f"   最大: {augmented_lengths.max()} tokens")
    print(f"   中位数: {np.median(augmented_lengths):.1f} tokens")
    
    print(f"\n📚 COT Example长度:")
    print(f"   平均: {cot_lengths.mean():.1f} tokens")
    print(f"   最小: {cot_lengths.min()} tokens")
    print(f"   最大: {cot_lengths.max()} tokens")
    print(f"   中位数: {np.median(cot_lengths):.1f} tokens")
    
    # 按数据集分组统计
    print(f"\n📊 按数据集分组统计:")
    for ds_name in ['gsm8k', 'math']:
        ds_results = [r for r in all_results if r['dataset_name'] == ds_name]
        if ds_results:
            ds_orig_lengths = np.array([r['original_length'] for r in ds_results])
            ds_aug_lengths = np.array([r['augmented_length'] for r in ds_results])
            ds_cot_lengths = np.array([r['cot_length'] for r in ds_results])
            
            print(f"\n  📁 {ds_name.upper()}:")
            print(f"     样本数: {len(ds_results)}")
            print(f"     原始prompt: 平均={ds_orig_lengths.mean():.1f}, 最大={ds_orig_lengths.max()}")
            print(f"     COT example: 平均={ds_cot_lengths.mean():.1f}, 最大={ds_cot_lengths.max()}")
            print(f"     增强后总长: 平均={ds_aug_lengths.mean():.1f}, 最大={ds_aug_lengths.max()}")
            
            # 超过2048的比例
            overlong_pct = np.sum(ds_aug_lengths > 2048) / len(ds_aug_lengths) * 100
            if overlong_pct > 0:
                print(f"     ⚠️ 超过2048: {overlong_pct:.1f}%")
    
    # 长度分布
    print(f"\n📊 增强后长度分布:")
    bins = [0, 512, 1024, 1536, 2048, 3000, 5000, 10000]
    for i in range(len(bins)-1):
        count = np.sum((augmented_lengths >= bins[i]) & (augmented_lengths < bins[i+1]))
        percentage = count / len(augmented_lengths) * 100
        print(f"   {bins[i]:5d} - {bins[i+1]:5d} tokens: {count:4d} 样本 ({percentage:5.1f}%)")
    
    # 超长样本
    max_allowed = 2048
    overlong = augmented_lengths > max_allowed
    num_overlong = np.sum(overlong)
    
    if num_overlong > 0:
        print(f"\n⚠️  超过 {max_allowed} tokens 的样本:")
        print(f"   数量: {num_overlong} / {len(augmented_lengths)} ({num_overlong/len(augmented_lengths)*100:.1f}%)")
        print(f"   最长: {augmented_lengths.max()} tokens")
        
        # 显示几个超长样本
        overlong_indices = np.where(overlong)[0][:3]
        for idx in overlong_indices:
            r = all_results[idx]
            print(f"\n   样本 {r['sample_index']} ({r['dataset_name']}):")
            print(f"     长度: {r['augmented_length']} tokens")
            print(f"     问题: {r['question'][:60]}...")
    
    # 建议
    print(f"\n" + "="*80)
    print(f"💡 建议")
    print(f"="*80)
    
    avg_aug_length = augmented_lengths.mean()
    
    if avg_aug_length > 2048:
        print(f"\n⚠️  平均长度 {avg_aug_length:.0f} 超过了 max_prompt_length=2048")
        print(f"   建议:")
        print(f"   1. 增大 max_prompt_length 到 {int(augmented_lengths.max() * 1.1)}")
        print(f"   2. 或者使用更短的 COT example")
        print(f"   3. 或者过滤掉超长样本")
    elif avg_aug_length > 1536:
        print(f"\n✅ 平均长度 {avg_aug_length:.0f} 可以使用 max_prompt_length=2048")
        print(f"   但建议设置为 {int(augmented_lengths.max() * 1.1)} 以容纳所有样本")
    else:
        print(f"\n✅ 平均长度 {avg_aug_length:.0f} 很合理")
        print(f"   max_prompt_length=2048 足够")
        if avg_aug_length < 1024:
            print(f"   甚至可以降到 max_prompt_length=1536 节省内存")
    
    print(f"\n" + "="*80)
    print(f"✅ 分析完成")
    print(f"="*80)
    
    print(f"\n📊 总结:")
    print(f"   分析样本数: {len(all_results)}")
    print(f"   GSM8K: {len([r for r in all_results if r['dataset_name']=='gsm8k'])} 个")
    print(f"   MATH: {len([r for r in all_results if r['dataset_name']=='math'])} 个")
    print(f"   平均增强后长度: {augmented_lengths.mean():.0f} tokens")
    print(f"   推荐 max_prompt_length: {int(np.percentile(augmented_lengths, 95))}")  # 95th percentile
    
    print(f"\n详细数据已保存到: {OUTPUT_FILE}")
    print(f"\n查看示例:")
    print(f"  cat {OUTPUT_FILE} | head -1 | jq .")
    print(f"\n查看最长的样本:")
    print(f"  cat {OUTPUT_FILE} | jq 'select(.augmented_length > 2000)' | head -3")
    print(f"\n统计不同数据集:")
    print(f"  cat {OUTPUT_FILE} | jq -r '.dataset_name' | sort | uniq -c")
    print(f"\n按长度排序查看:")
    print(f"  cat {OUTPUT_FILE} | jq -s 'sort_by(.augmented_length) | reverse | .[0:5]'")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()

