#!/usr/bin/env python3
"""
模拟训练环境的诊断工具

这个脚本模拟真实训练时的配置，包括多进程、DataProto等
"""

import sys
sys.path.insert(0, '/nas/dhl/verl')

from examples.grpo_trainer.multi_dataset_with_cot import MultiDatasetWithCOT
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader
from verl.utils.dataset.rl_dataset import collate_fn
from verl import DataProto
import numpy as np
import uuid

# 配置（与 test_k_cot.sh 保持一致）
ACTOR_MODEL_PATH = "/nas/models/Qwen3-8B"
TRAIN_FILES_GSM8K = "/nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet"
TRAIN_FILES_MATH = "/nas/dhl/Datasets/my_Datasets/MATH/train.parquet"
BATCH_SIZE = 8
N_ROLLOUTS = 4  # 每个问题生成4次

def main():
    print("\n" + "="*80)
    print("🔍 训练环境诊断工具（模拟真实训练配置）")
    print("="*80)
    
    # Step 1: 加载 tokenizer
    print("\nStep 1: 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ACTOR_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  ✓ Tokenizer 加载成功")
    
    # Step 2: 创建数据集（模拟训练配置）
    print("\nStep 2: 创建训练数据集...")
    config = {
        'gsm8k_path': TRAIN_FILES_GSM8K,
        'math_path': TRAIN_FILES_MATH,
        'max_prompt_length': 2048,
        'max_response_length': 2048,
        'dataloader_num_workers': 0,  # ← 先用0测试
        'shuffle': True,
        'seed': 1,
    }
    
    train_dataset = MultiDatasetWithCOT(
        tokenizer=tokenizer,
        config=config,
        is_train=True
    )
    print(f"  ✓ 训练数据集大小: {len(train_dataset)}")
    
    # Step 3: 创建验证数据集（模拟空验证集）
    print("\nStep 3: 创建验证数据集（空）...")
    val_config = config.copy()
    val_dataset = MultiDatasetWithCOT(
        tokenizer=tokenizer,
        config=val_config,
        is_train=False  # ← 验证模式，但没有数据路径
    )
    print(f"  验证数据集大小: {len(val_dataset)}")
    
    if len(val_dataset) > 0:
        print("  ⚠️  警告: 验证数据集不为空！这可能不符合预期。")
    
    # Step 4: 创建 DataLoader（使用 StatefulDataLoader）
    print("\nStep 4: 创建 StatefulDataLoader...")
    
    # 训练 DataLoader
    from torch.utils.data import RandomSampler
    import torch
    
    train_dataloader_generator = torch.Generator()
    train_dataloader_generator.manual_seed(config.get("seed", 1))
    sampler = RandomSampler(data_source=train_dataset, generator=train_dataloader_generator)
    
    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=config['dataloader_num_workers'],
        drop_last=True,
        collate_fn=collate_fn,
        sampler=sampler,
    )
    print(f"  ✓ 训练 DataLoader: {len(train_dataloader)} batches")
    
    # 验证 DataLoader
    val_batch_size = 8  # 显式设置
    
    # 🔑 修复：StatefulDataLoader 不支持空数据集
    if len(val_dataset) == 0:
        print(f"  ⚠️  验证集为空，创建虚拟的空 DataLoader")
        val_dataloader = []  # 空列表代替
    else:
        val_dataloader = StatefulDataLoader(
            dataset=val_dataset,
            batch_size=val_batch_size,
            num_workers=config['dataloader_num_workers'],
            shuffle=config.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )
    
    print(f"  验证 DataLoader: {len(val_dataloader)} batches")
    
    # Step 5: 模拟训练循环（测试前几个batch）
    print("\nStep 5: 模拟训练循环...")
    
    num_test_batches = min(5, len(train_dataloader))
    success_count = 0
    empty_count = 0
    
    for i, batch_dict in enumerate(train_dataloader):
        if i >= num_test_batches:
            break
        
        print(f"\n  Batch {i}:")
        
        # ⭐ 模拟真实训练的处理流程
        try:
            # 🔍 先检查 batch_dict 的内容
            print(f"    🔍 batch_dict 调试:")
            print(f"       类型: {type(batch_dict)}")
            print(f"       字段: {list(batch_dict.keys()) if isinstance(batch_dict, dict) else 'N/A'}")
            
            if 'input_ids' in batch_dict:
                input_ids_val = batch_dict['input_ids']
                print(f"       input_ids 类型: {type(input_ids_val)}")
                if hasattr(input_ids_val, 'shape'):
                    print(f"       input_ids shape: {input_ids_val.shape}")
                elif hasattr(input_ids_val, '__len__'):
                    print(f"       input_ids len: {len(input_ids_val)}")
                    if len(input_ids_val) > 0:
                        first_item = input_ids_val[0]
                        print(f"       input_ids[0] 类型: {type(first_item)}")
                        if hasattr(first_item, '__len__'):
                            print(f"       input_ids[0] len: {len(first_item)}")
            
            # Step 5.1: 转换为 DataProto
            print(f"    🔄 转换为 DataProto...")
            batch = DataProto.from_single_dict(batch_dict)
            
            print(f"    🔍 DataProto 结果:")
            print(f"       batch.batch 类型: {type(batch.batch)}")
            print(f"       batch.batch 内容: {batch.batch}")
            # TensorDict 有 keys() 方法
            if batch.batch is not None and hasattr(batch.batch, 'keys'):
                print(f"       batch.batch 字段: {list(batch.batch.keys())}")
            
            # Step 5.2: 检查是否为空（这是真实训练中的检查）
            # 注意：TensorDict 不能直接用 if 判断，需要检查是否为 None 或长度为0
            if batch.batch is None or len(batch.batch) == 0:
                print(f"    ❌ 检测到空batch! (这是训练时的检查)")
                empty_count += 1
                continue
            
            # Step 5.3: 添加 uid
            batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(batch.batch))], 
                dtype=object
            )
            
            # Step 5.4: 模拟 repeat 操作（GRPO需要）
            batch_repeated = batch.repeat(repeat_times=N_ROLLOUTS, interleave=True)
            
            print(f"    ✓ Batch 正常")
            print(f"      原始 batch_size: {len(batch.batch['input_ids'])}")
            print(f"      repeat 后 batch_size: {len(batch_repeated.batch['input_ids'])}")
            print(f"      dataset_names: {batch_repeated.non_tensor_batch['dataset_name'][:8]}")
            success_count += 1
            
        except Exception as e:
            print(f"    ❌ 处理batch时异常: {e}")
            import traceback
            traceback.print_exc()
    
    # Step 6: 测试验证集 DataLoader（如果非空）
    if len(val_dataloader) > 0:
        print("\nStep 6: 测试验证集 DataLoader...")
        val_empty_count = 0
        for i, val_batch_dict in enumerate(val_dataloader):
            if i >= 3:
                break
            
            print(f"\n  Validation Batch {i}:")
            
            try:
                val_batch = DataProto.from_single_dict(val_batch_dict)
                
                if not val_batch.batch or len(val_batch.batch) == 0:
                    print(f"    ❌ 验证集空batch!")
                    val_empty_count += 1
                else:
                    print(f"    ✓ 验证 batch 正常")
            except Exception as e:
                print(f"    ❌ 异常: {e}")
        
        if val_empty_count > 0:
            print(f"\n  ⚠️  警告: 验证集有 {val_empty_count} 个空batch")
    else:
        print("\nStep 6: 跳过验证集测试（验证集为空）")
    
    # 总结
    print("\n" + "="*80)
    print("📋 诊断总结")
    print("="*80)
    print(f"  训练数据集大小: {len(train_dataset)}")
    print(f"  验证数据集大小: {len(val_dataset)}")
    print(f"  训练 batches: {len(train_dataloader)}")
    print(f"  验证 batches: {len(val_dataloader)}")
    print(f"  测试结果: {success_count}/{num_test_batches} 成功, {empty_count} 空batch")
    
    if empty_count == 0:
        print("\n✅ 所有测试通过！训练环境配置正常。")
        print("   如果训练时仍然出现空batch，可能是:")
        print("   1. 多进程问题 (dataloader_num_workers > 0)")
        print("   2. Ray分布式环境的问题")
        print("   3. COT增强过程中的问题")
    else:
        print(f"\n❌ 检测到 {empty_count} 个空batch!")
        print("   这可能是问题根源。")
    
    print("="*80 + "\n")

if __name__ == '__main__':
    main()

