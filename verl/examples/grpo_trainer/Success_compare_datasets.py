# #!/usr/bin/env python3
# """
# 数据集格式对比工具

# 对比 RLHFDataset 和 MultiDatasetWithCOT 返回的样本格式
# 快速发现字段差异

# 用法:
#     python compare_datasets.py
# """

# import sys
# sys.path.insert(0, '/nas/dhl/verl')

# from verl.utils.dataset.rl_dataset import RLHFDataset
# from examples.grpo_trainer.multi_dataset_with_cot import MultiDatasetWithCOT
# from transformers import AutoTokenizer
# import torch
# from pprint import pprint

# # 配置
# ACTOR_MODEL_PATH = "/nas/models/Qwen3-8B"
# TRAIN_FILE = "/nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet"

# def format_value(value, max_len=50):
#     """格式化值用于打印"""
#     if isinstance(value, torch.Tensor):
#         return f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype})"
#     elif isinstance(value, str):
#         if len(value) > max_len:
#             return f'"{value[:max_len]}..."'
#         return f'"{value}"'
#     elif isinstance(value, (list, tuple)):
#         return f"{type(value).__name__}(len={len(value)})"
#     elif isinstance(value, dict):
#         return f"dict(keys={list(value.keys())})"
#     else:
#         return str(value)

# def print_sample(sample, title, index):
#     """打印单个样本的详细信息"""
#     print(f"\n{'='*80}")
#     print(f"{title} - 样本 {index}")
#     print(f"{'='*80}")
    
#     # 按类别分组
#     tensor_fields = {}
#     dict_fields = {}
#     simple_fields = {}
    
#     for key, value in sample.items():
#         if isinstance(value, torch.Tensor):
#             tensor_fields[key] = value
#         elif isinstance(value, dict):
#             dict_fields[key] = value
#         else:
#             simple_fields[key] = value
    
#     # 打印 Tensor 字段
#     if tensor_fields:
#         print("\n📊 Tensor 字段:")
#         for key, value in sorted(tensor_fields.items()):
#             print(f"  {key:20s} : {format_value(value)}")
    
#     # 打印 Dict 字段
#     if dict_fields:
#         print("\n📁 Dict 字段:")
#         for key, value in sorted(dict_fields.items()):
#             print(f"  {key:20s} : {format_value(value)}")
#             # 展开dict内容
#             if isinstance(value, dict) and len(value) < 10:
#                 for sub_key, sub_value in value.items():
#                     print(f"    ├─ {sub_key:16s} : {format_value(sub_value, max_len=40)}")
    
#     # 打印简单字段
#     if simple_fields:
#         print("\n📝 其他字段:")
#         for key, value in sorted(simple_fields.items()):
#             print(f"  {key:20s} : {format_value(value)}")
    
#     print(f"\n总字段数: {len(sample)}")
#     print(f"所有字段: {list(sample.keys())}")

# def compare_samples(rlhf_sample, custom_sample, index):
#     """对比两个样本，找出差异"""
#     print(f"\n{'='*80}")
#     print(f"⚖️  样本 {index} 对比分析")
#     print(f"{'='*80}")
    
#     rlhf_keys = set(rlhf_sample.keys())
#     custom_keys = set(custom_sample.keys())
    
#     # 找出差异
#     only_in_rlhf = rlhf_keys - custom_keys
#     only_in_custom = custom_keys - rlhf_keys
#     common_keys = rlhf_keys & custom_keys
    
#     print(f"\n📊 字段统计:")
#     print(f"  RLHFDataset 字段数: {len(rlhf_keys)}")
#     print(f"  MultiDatasetWithCOT 字段数: {len(custom_keys)}")
#     print(f"  共同字段数: {len(common_keys)}")
    
#     # 只在 RLHFDataset 中的字段（可能缺失）
#     if only_in_rlhf:
#         print(f"\n❌ 只在 RLHFDataset 中（MultiDatasetWithCOT 缺失）:")
#         for key in sorted(only_in_rlhf):
#             value = rlhf_sample[key]
#             print(f"  • {key:20s} : {format_value(value)}")
#             if key in ['reward_model', 'position_ids', 'input_ids', 'attention_mask']:
#                 print(f"    ⚠️  这是重要字段，缺失会导致错误！")
    
#     # 只在 MultiDatasetWithCOT 中的字段（额外字段）
#     if only_in_custom:
#         print(f"\n➕ 只在 MultiDatasetWithCOT 中（额外字段）:")
#         for key in sorted(only_in_custom):
#             value = custom_sample[key]
#             print(f"  • {key:20s} : {format_value(value)}")
#             if key in ['dataset_name', 'question']:
#                 print(f"    ✅ 这是 COT 增强需要的自定义字段")
    
#     # 共同字段（检查类型和形状是否一致）
#     print(f"\n✅ 共同字段:")
#     type_mismatch = []
#     shape_mismatch = []
    
#     for key in sorted(common_keys):
#         rlhf_val = rlhf_sample[key]
#         custom_val = custom_sample[key]
        
#         # 检查类型
#         rlhf_type = type(rlhf_val).__name__
#         custom_type = type(custom_val).__name__
        
#         status = "✓"
#         if rlhf_type != custom_type:
#             status = "⚠️ 类型不同"
#             type_mismatch.append(key)
#         elif isinstance(rlhf_val, torch.Tensor) and isinstance(custom_val, torch.Tensor):
#             if rlhf_val.shape != custom_val.shape:
#                 status = "⚠️ 形状不同"
#                 shape_mismatch.append(key)
        
#         print(f"  {status} {key:18s} : RLHFDataset={rlhf_type:12s}, MultiDatasetWithCOT={custom_type:12s}")
        
#         # 如果是 Tensor，显示形状
#         if isinstance(rlhf_val, torch.Tensor):
#             print(f"      {'':20s}   shape: {tuple(rlhf_val.shape)} vs {tuple(custom_val.shape)}")
    
#     # 总结
#     print(f"\n{'='*80}")
#     print(f"📋 对比总结:")
#     print(f"{'='*80}")
    
#     critical_missing = [k for k in only_in_rlhf if k in ['input_ids', 'attention_mask', 'position_ids', 'reward_model']]
    
#     if not only_in_rlhf and not type_mismatch and not shape_mismatch:
#         print("✅ 完美匹配！MultiDatasetWithCOT 完全符合标准。")
#     else:
#         if critical_missing:
#             print(f"❌ 缺少关键字段: {critical_missing}")
#             print(f"   这些字段缺失会导致训练失败！")
#         elif only_in_rlhf:
#             print(f"⚠️  缺少字段: {list(only_in_rlhf)}")
#             print(f"   可能不影响基本功能，但不符合完整标准。")
        
#         if type_mismatch:
#             print(f"⚠️  类型不匹配: {type_mismatch}")
        
#         if shape_mismatch:
#             print(f"⚠️  形状不匹配: {shape_mismatch}")
    
#     if only_in_custom:
#         print(f"➕ 额外字段: {list(only_in_custom)}")
#         print(f"   这些是自定义字段，用于特殊功能（如 COT 增强）。")
    
#     print(f"{'='*80}\n")


# def main():
#     print("\n" + "="*80)
#     print("🔍 数据集格式对比工具")
#     print("="*80)
#     print(f"对比文件: {TRAIN_FILE}")
#     print("="*80)
    
#     # 加载 tokenizer
#     print("\n⏳ 加载 tokenizer...")
#     tokenizer = AutoTokenizer.from_pretrained(ACTOR_MODEL_PATH, trust_remote_code=True)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#     print(f"✓ Tokenizer 加载成功")
    
#     # 创建 RLHFDataset
#     print("\n⏳ 创建 RLHFDataset...")
#     rlhf_config = {
#         'max_prompt_length': 2048,
#         'truncation': 'error',  # ← 必须是 'left', 'right', 'middle', 或 'error'，不能是 True
#     }
    
#     try:
#         rlhf_dataset = RLHFDataset(
#             data_files=[TRAIN_FILE],
#             tokenizer=tokenizer,
#             processor=None,
#             config=rlhf_config,
#         )
#         print(f"✓ RLHFDataset 创建成功，大小: {len(rlhf_dataset)}")
#     except Exception as e:
#         print(f"✗ RLHFDataset 创建失败: {e}")
#         import traceback
#         traceback.print_exc()
#         return
    
#     # 创建 MultiDatasetWithCOT
#     print("\n⏳ 创建 MultiDatasetWithCOT...")
#     custom_config = {
#         'gsm8k_path': TRAIN_FILE,
#         'max_prompt_length': 2048,
#     }
    
#     try:
#         custom_dataset = MultiDatasetWithCOT(
#             tokenizer=tokenizer,
#             processor=None,
#             config=custom_config,
#             is_train=True,
#         )
#         print(f"✓ MultiDatasetWithCOT 创建成功，大小: {len(custom_dataset)}")
#     except Exception as e:
#         print(f"✗ MultiDatasetWithCOT 创建失败: {e}")
#         import traceback
#         traceback.print_exc()
#         return
    
#     # 获取并打印前3个样本
#     print("\n" + "="*80)
#     print("📋 详细样本对比（前3个样本）")
#     print("="*80)
    
#     num_samples = min(3, len(rlhf_dataset), len(custom_dataset))
    
#     for i in range(num_samples):
#         try:
#             # 获取样本
#             rlhf_sample = rlhf_dataset[i]
#             custom_sample = custom_dataset[i]
            
#             # 打印 RLHFDataset 样本
#             print_sample(rlhf_sample, "🅰️ RLHFDataset", i)
            
#             # 打印 MultiDatasetWithCOT 样本
#             print_sample(custom_sample, "🅱️ MultiDatasetWithCOT", i)
            
#             # 对比分析
#             compare_samples(rlhf_sample, custom_sample, i)
            
#         except Exception as e:
#             print(f"\n❌ 处理样本 {i} 时出错: {e}")
#             import traceback
#             traceback.print_exc()
    
#     # 最终总结
#     print("\n" + "="*80)
#     print("🎯 最终建议")
#     print("="*80)
    
#     # 获取第一个样本做完整检查
#     rlhf_sample = rlhf_dataset[0]
#     custom_sample = custom_dataset[0]
    
#     rlhf_keys = set(rlhf_sample.keys())
#     custom_keys = set(custom_sample.keys())
    
#     critical_fields = {'input_ids', 'attention_mask', 'position_ids', 'reward_model'}
#     missing_critical = critical_fields - custom_keys
    
#     if missing_critical:
#         print(f"\n❌ MultiDatasetWithCOT 缺少关键字段:")
#         for field in missing_critical:
#             print(f"  • {field}")
#             print(f"    需要在 __getitem__() 中添加此字段")
#     else:
#         print(f"\n✅ MultiDatasetWithCOT 包含所有关键字段!")
    
#     # 检查类型匹配
#     print(f"\n📊 关键字段类型检查:")
#     for field in ['input_ids', 'attention_mask', 'position_ids']:
#         if field in rlhf_sample and field in custom_sample:
#             rlhf_val = rlhf_sample[field]
#             custom_val = custom_sample[field]
            
#             if isinstance(rlhf_val, torch.Tensor) and isinstance(custom_val, torch.Tensor):
#                 shape_match = rlhf_val.shape == custom_val.shape
#                 dtype_match = rlhf_val.dtype == custom_val.dtype
                
#                 status = "✅" if shape_match and dtype_match else "⚠️"
#                 print(f"  {status} {field:18s} : shape {tuple(rlhf_val.shape)} vs {tuple(custom_val.shape)}")
#                 if not dtype_match:
#                     print(f"      类型不同: {rlhf_val.dtype} vs {custom_val.dtype}")
#             else:
#                 rlhf_type = type(rlhf_val).__name__
#                 custom_type = type(custom_val).__name__
#                 status = "✅" if rlhf_type == custom_type else "❌"
#                 print(f"  {status} {field:18s} : {rlhf_type} vs {custom_type}")
    
#     # reward_model 特殊检查
#     if 'reward_model' in rlhf_sample and 'reward_model' in custom_sample:
#         print(f"\n🎁 reward_model 字段对比:")
#         rlhf_rm = rlhf_sample['reward_model']
#         custom_rm = custom_sample['reward_model']
        
#         if isinstance(rlhf_rm, dict) and isinstance(custom_rm, dict):
#             for key in ['style', 'ground_truth']:
#                 rlhf_has = key in rlhf_rm
#                 custom_has = key in custom_rm
#                 status = "✅" if (rlhf_has and custom_has) else "❌"
#                 print(f"  {status} {key:18s} : RLHFDataset={rlhf_has}, MultiDatasetWithCOT={custom_has}")
#                 if rlhf_has and custom_has:
#                     print(f"      值: {format_value(rlhf_rm[key], 30)} vs {format_value(custom_rm[key], 30)}")
    
#     print(f"\n{'='*80}")
#     print(f"如果所有关键字段都是 ✅，就可以开始训练了！")
#     print(f"{'='*80}\n")


# if __name__ == '__main__':
#     main()


#!/usr/bin/env python3
"""
数据集格式对比工具

对比 RLHFDataset 和 MultiDatasetWithCOT 返回的样本格式
快速发现字段差异

用法:
    python compare_datasets.py
"""

import sys
sys.path.insert(0, '/nas/dhl/verl')

from verl.utils.dataset.rl_dataset import RLHFDataset
from examples.grpo_trainer.multi_dataset_with_cot import MultiDatasetWithCOT
from transformers import AutoTokenizer
import torch
from pprint import pprint

# 配置
ACTOR_MODEL_PATH = "/nas/models/Qwen3-8B"
TRAIN_FILE = "/nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet"

def format_value(value, max_len=50):
    """格式化值用于打印"""
    if isinstance(value, torch.Tensor):
        return f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype})"
    elif isinstance(value, str):
        if len(value) > max_len:
            return f'"{value[:max_len]}..."'
        return f'"{value}"'
    elif isinstance(value, (list, tuple)):
        return f"{type(value).__name__}(len={len(value)})"
    elif isinstance(value, dict):
        return f"dict(keys={list(value.keys())})"
    else:
        return str(value)

def print_sample(sample, title, index):
    """打印单个样本的详细信息"""
    print(f"\n{'='*80}")
    print(f"{title} - 样本 {index}")
    print(f"{'='*80}")
    
    # 按类别分组
    tensor_fields = {}
    dict_fields = {}
    simple_fields = {}
    
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            tensor_fields[key] = value
        elif isinstance(value, dict):
            dict_fields[key] = value
        else:
            simple_fields[key] = value
    
    # 打印 Tensor 字段
    if tensor_fields:
        print("\n📊 Tensor 字段:")
        for key, value in sorted(tensor_fields.items()):
            print(f"  {key:20s} : {format_value(value)}")
    
    # 打印 Dict 字段
    if dict_fields:
        print("\n📁 Dict 字段:")
        for key, value in sorted(dict_fields.items()):
            print(f"  {key:20s} : {format_value(value)}")
            # 展开dict内容
            if isinstance(value, dict) and len(value) < 10:
                for sub_key, sub_value in value.items():
                    print(f"    ├─ {sub_key:16s} : {format_value(sub_value, max_len=40)}")
    
    # 打印简单字段
    if simple_fields:
        print("\n📝 其他字段:")
        for key, value in sorted(simple_fields.items()):
            print(f"  {key:20s} : {format_value(value)}")
    
    print(f"\n总字段数: {len(sample)}")
    print(f"所有字段: {list(sample.keys())}")

def compare_samples(rlhf_sample, custom_sample, index):
    """对比两个样本，找出差异"""
    print(f"\n{'='*80}")
    print(f"⚖️  样本 {index} 对比分析")
    print(f"{'='*80}")
    
    rlhf_keys = set(rlhf_sample.keys())
    custom_keys = set(custom_sample.keys())
    
    # 找出差异
    only_in_rlhf = rlhf_keys - custom_keys
    only_in_custom = custom_keys - rlhf_keys
    common_keys = rlhf_keys & custom_keys
    
    print(f"\n📊 字段统计:")
    print(f"  RLHFDataset 字段数: {len(rlhf_keys)}")
    print(f"  MultiDatasetWithCOT 字段数: {len(custom_keys)}")
    print(f"  共同字段数: {len(common_keys)}")
    
    # 只在 RLHFDataset 中的字段（可能缺失）
    if only_in_rlhf:
        print(f"\n❌ 只在 RLHFDataset 中（MultiDatasetWithCOT 缺失）:")
        for key in sorted(only_in_rlhf):
            value = rlhf_sample[key]
            print(f"  • {key:20s} : {format_value(value)}")
            if key in ['reward_model', 'position_ids', 'input_ids', 'attention_mask']:
                print(f"    ⚠️  这是重要字段，缺失会导致错误！")
    
    # 只在 MultiDatasetWithCOT 中的字段（额外字段）
    if only_in_custom:
        print(f"\n➕ 只在 MultiDatasetWithCOT 中（额外字段）:")
        for key in sorted(only_in_custom):
            value = custom_sample[key]
            print(f"  • {key:20s} : {format_value(value)}")
            if key in ['dataset_name', 'question']:
                print(f"    ✅ 这是 COT 增强需要的自定义字段")
    
    # 共同字段（检查类型、形状和内容是否一致）
    print(f"\n✅ 共同字段（类型、形状、内容对比）:")
    type_mismatch = []
    shape_mismatch = []
    content_mismatch = []
    
    for key in sorted(common_keys):
        rlhf_val = rlhf_sample[key]
        custom_val = custom_sample[key]
        
        # 检查类型
        rlhf_type = type(rlhf_val).__name__
        custom_type = type(custom_val).__name__
        
        status = "✓"
        content_match = None  # None = 未检查，True = 一致，False = 不一致
        
        if rlhf_type != custom_type:
            status = "⚠️ 类型不同"
            type_mismatch.append(key)
        elif isinstance(rlhf_val, torch.Tensor) and isinstance(custom_val, torch.Tensor):
            # Tensor 类型：检查形状和内容
            if rlhf_val.shape != custom_val.shape:
                status = "⚠️ 形状不同"
                shape_mismatch.append(key)
            else:
                # 形状相同，检查内容
                content_match = torch.equal(rlhf_val, custom_val)
                if not content_match:
                    status = "⚠️ 内容不同"
                    content_mismatch.append(key)
        elif isinstance(rlhf_val, dict) and isinstance(custom_val, dict):
            # Dict 类型：检查关键子字段
            if key == 'reward_model':
                # reward_model 特殊检查 ground_truth
                rlhf_gt = rlhf_val.get('ground_truth')
                custom_gt = custom_val.get('ground_truth')
                content_match = (rlhf_gt == custom_gt)
                if not content_match:
                    status = "❌ ground_truth 不同"
                    content_mismatch.append(key)
        elif isinstance(rlhf_val, (str, int, float)):
            # 简单类型：直接比较
            content_match = (rlhf_val == custom_val)
            if not content_match:
                status = "⚠️ 内容不同"
                content_mismatch.append(key)
        
        print(f"  {status} {key:18s} : RLHFDataset={rlhf_type:12s}, MultiDatasetWithCOT={custom_type:12s}")
        
        # 如果是 Tensor，显示形状和内容匹配情况
        if isinstance(rlhf_val, torch.Tensor):
            print(f"      {'':20s}   shape: {tuple(rlhf_val.shape)} vs {tuple(custom_val.shape)}")
            if content_match is not None:
                match_str = "✅ 完全相同" if content_match else "❌ 内容不同"
                print(f"      {'':20s}   内容: {match_str}")
                if not content_match and key in ['input_ids', 'attention_mask']:
                    # 显示第一个不同的位置
                    diff_mask = (rlhf_val != custom_val)
                    if diff_mask.any():
                        first_diff_idx = diff_mask.nonzero()[0].item()
                        print(f"      {'':20s}   首次不同位置: index={first_diff_idx}")
                        print(f"      {'':20s}     RLHFDataset[{first_diff_idx}] = {rlhf_val[first_diff_idx].item()}")
                        print(f"      {'':20s}     MultiDatasetWithCOT[{first_diff_idx}] = {custom_val[first_diff_idx].item()}")
        elif key == 'reward_model' and isinstance(rlhf_val, dict) and isinstance(custom_val, dict):
            # reward_model 显示详细对比
            if content_match is not None:
                match_str = "✅ ground_truth 相同" if content_match else "❌ ground_truth 不同"
                print(f"      {'':20s}   {match_str}")
                if not content_match:
                    print(f"      {'':20s}     RLHFDataset: {rlhf_val.get('ground_truth')}")
                    print(f"      {'':20s}     MultiDatasetWithCOT: {custom_val.get('ground_truth')}")
        elif content_match is not None:
            # 其他类型显示匹配情况
            if not content_match:
                print(f"      {'':20s}   RLHFDataset: {format_value(rlhf_val, 30)}")
                print(f"      {'':20s}   MultiDatasetWithCOT: {format_value(custom_val, 30)}")
    
    # 总结
    print(f"\n{'='*80}")
    print(f"📋 对比总结:")
    print(f"{'='*80}")
    
    critical_missing = [k for k in only_in_rlhf if k in ['input_ids', 'attention_mask', 'position_ids', 'reward_model']]
    
    # 检查是否完美匹配
    is_perfect = (
        not only_in_rlhf and 
        not type_mismatch and 
        not shape_mismatch and 
        not content_mismatch
    )
    
    if is_perfect:
        print("🎉 完美匹配！MultiDatasetWithCOT 完全符合标准，且内容完全一致！")
    else:
        # 按严重程度报告问题
        if critical_missing:
            print(f"❌ 缺少关键字段: {critical_missing}")
            print(f"   这些字段缺失会导致训练失败！")
        elif only_in_rlhf:
            print(f"⚠️  缺少字段: {list(only_in_rlhf)}")
            print(f"   可能不影响基本功能，但不符合完整标准。")
        
        if type_mismatch:
            print(f"⚠️  类型不匹配: {type_mismatch}")
            print(f"   需要检查数据提取逻辑。")
        
        if shape_mismatch:
            print(f"⚠️  形状不匹配: {shape_mismatch}")
            print(f"   需要检查 tokenization 参数。")
        
        if content_mismatch:
            print(f"❌ 内容不匹配: {content_mismatch}")
            print(f"   字段存在且类型正确，但内容不同！")
            print(f"   这是最严重的问题，说明数据提取逻辑有误！")
            for field in content_mismatch:
                if field in ['input_ids', 'attention_mask']:
                    print(f"   → {field}: tokenization 逻辑可能不同")
                elif field == 'reward_model':
                    print(f"   → {field}: ground_truth 提取逻辑可能有误")
    
    if only_in_custom:
        print(f"\n➕ 额外字段: {list(only_in_custom)}")
        print(f"   这些是自定义字段，用于特殊功能（如 COT 增强）。")
    
    # 给出建议
    if content_mismatch:
        print(f"\n💡 修复建议:")
        print(f"   内容不匹配说明提取逻辑有问题，请检查:")
        print(f"   1. tokenization 参数是否完全相同")
        print(f"   2. 是否使用了相同的 chat template")
        print(f"   3. reward_model 是否从原始数据正确读取")
    
    print(f"{'='*80}\n")


def main():
    print("\n" + "="*80)
    print("🔍 数据集格式对比工具")
    print("="*80)
    print(f"对比文件: {TRAIN_FILE}")
    print("="*80)
    
    # 加载 tokenizer
    print("\n⏳ 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ACTOR_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✓ Tokenizer 加载成功")
    
    # 创建 RLHFDataset
    print("\n⏳ 创建 RLHFDataset...")
    rlhf_config = {
        'max_prompt_length': 2048,
        'truncation': 'error',  # ← 必须是 'left', 'right', 'middle', 或 'error'，不能是 True
    }
    
    try:
        rlhf_dataset = RLHFDataset(
            data_files=[TRAIN_FILE],
            tokenizer=tokenizer,
            processor=None,
            config=rlhf_config,
        )
        print(f"✓ RLHFDataset 创建成功，大小: {len(rlhf_dataset)}")
    except Exception as e:
        print(f"✗ RLHFDataset 创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 创建 MultiDatasetWithCOT
    print("\n⏳ 创建 MultiDatasetWithCOT...")
    custom_config = {
        'gsm8k_path': TRAIN_FILE,
        'max_prompt_length': 2048,
    }
    
    try:
        custom_dataset = MultiDatasetWithCOT(
            tokenizer=tokenizer,
            processor=None,
            config=custom_config,
            is_train=True,
        )
        print(f"✓ MultiDatasetWithCOT 创建成功，大小: {len(custom_dataset)}")
    except Exception as e:
        print(f"✗ MultiDatasetWithCOT 创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 获取并打印前3个样本
    print("\n" + "="*80)
    print("📋 详细样本对比（前3个样本）")
    print("="*80)
    
    num_samples = min(3, len(rlhf_dataset), len(custom_dataset))
    
    for i in range(num_samples):
        try:
            # 获取样本
            rlhf_sample = rlhf_dataset[i]
            custom_sample = custom_dataset[i]
            
            # 打印 RLHFDataset 样本
            print_sample(rlhf_sample, "🅰️ RLHFDataset", i)
            
            # 打印 MultiDatasetWithCOT 样本
            print_sample(custom_sample, "🅱️ MultiDatasetWithCOT", i)
            
            # 对比分析
            compare_samples(rlhf_sample, custom_sample, i)
            
        except Exception as e:
            print(f"\n❌ 处理样本 {i} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 最终总结
    print("\n" + "="*80)
    print("🎯 最终建议")
    print("="*80)
    
    # 获取第一个样本做完整检查
    rlhf_sample = rlhf_dataset[0]
    custom_sample = custom_dataset[0]
    
    rlhf_keys = set(rlhf_sample.keys())
    custom_keys = set(custom_sample.keys())
    
    critical_fields = {'input_ids', 'attention_mask', 'position_ids', 'reward_model'}
    missing_critical = critical_fields - custom_keys
    
    if missing_critical:
        print(f"\n❌ MultiDatasetWithCOT 缺少关键字段:")
        for field in missing_critical:
            print(f"  • {field}")
            print(f"    需要在 __getitem__() 中添加此字段")
    else:
        print(f"\n✅ MultiDatasetWithCOT 包含所有关键字段!")
    
    # 检查类型匹配
    print(f"\n📊 关键字段类型检查:")
    for field in ['input_ids', 'attention_mask', 'position_ids']:
        if field in rlhf_sample and field in custom_sample:
            rlhf_val = rlhf_sample[field]
            custom_val = custom_sample[field]
            
            if isinstance(rlhf_val, torch.Tensor) and isinstance(custom_val, torch.Tensor):
                shape_match = rlhf_val.shape == custom_val.shape
                dtype_match = rlhf_val.dtype == custom_val.dtype
                
                status = "✅" if shape_match and dtype_match else "⚠️"
                print(f"  {status} {field:18s} : shape {tuple(rlhf_val.shape)} vs {tuple(custom_val.shape)}")
                if not dtype_match:
                    print(f"      类型不同: {rlhf_val.dtype} vs {custom_val.dtype}")
            else:
                rlhf_type = type(rlhf_val).__name__
                custom_type = type(custom_val).__name__
                status = "✅" if rlhf_type == custom_type else "❌"
                print(f"  {status} {field:18s} : {rlhf_type} vs {custom_type}")
    
    # reward_model 特殊检查
    if 'reward_model' in rlhf_sample and 'reward_model' in custom_sample:
        print(f"\n🎁 reward_model 字段对比:")
        rlhf_rm = rlhf_sample['reward_model']
        custom_rm = custom_sample['reward_model']
        
        if isinstance(rlhf_rm, dict) and isinstance(custom_rm, dict):
            for key in ['style', 'ground_truth']:
                rlhf_has = key in rlhf_rm
                custom_has = key in custom_rm
                status = "✅" if (rlhf_has and custom_has) else "❌"
                print(f"  {status} {key:18s} : RLHFDataset={rlhf_has}, MultiDatasetWithCOT={custom_has}")
                if rlhf_has and custom_has:
                    print(f"      值: {format_value(rlhf_rm[key], 30)} vs {format_value(custom_rm[key], 30)}")
    
    print(f"\n{'='*80}")
    print(f"如果所有关键字段都是 ✅，就可以开始训练了！")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

