# #!/usr/bin/env python3
# """
# 快速测试 ray_trainer.fit() 的数据流
# 无需 Ray 初始化，直接测试数据处理逻辑

# 用法:
#     python quick_fit_test.py
# """

# import sys
# sys.path.insert(0, '/nas/dhl/verl')

# from examples.grpo_trainer.multi_dataset_with_cot import MultiDatasetWithCOT
# from transformers import AutoTokenizer
# from torch.utils.data import DataLoader
# from verl.utils.dataset.rl_dataset import collate_fn
# from verl import DataProto
# import numpy as np
# import uuid
# import torch

# # 配置
# ACTOR_MODEL_PATH = "/nas/models/Qwen3-8B"
# TRAIN_FILES_GSM8K = "/nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet"
# TRAIN_FILES_MATH = "/nas/dhl/Datasets/my_Datasets/MATH/train.parquet"
# BATCH_SIZE = 8
# N_ROLLOUTS = 4

# def simulate_get_gen_batch(batch: DataProto, async_rollout_mode=False):
#     """
#     模拟 _get_gen_batch 方法（ray_trainer.py 第656-671行）
#     """
#     print("\n🔍 模拟 _get_gen_batch()...")
    
#     # 检查当前 batch 中有哪些字段
#     print(f"  batch.batch 字段: {list(batch.batch.keys())}")
#     print(f"  batch.non_tensor_batch 字段: {list(batch.non_tensor_batch.keys())}")
    
#     reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()
#     print(f"  reward_model_keys: {reward_model_keys}")
    
#     # pop those keys for generation
#     batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
    
#     # 🔑 关键检查：哪些key存在，哪些不存在
#     print(f"  尝试pop的batch_keys: {batch_keys_to_pop}")
#     for key in batch_keys_to_pop:
#         exists = key in batch.batch.keys()
#         print(f"    {key}: {'✓ 存在' if exists else '✗ 不存在'}")
    
#     # 🔑 修复：只pop存在的key
#     existing_batch_keys = [k for k in batch_keys_to_pop if k in batch.batch.keys()]
#     print(f"  实际pop的batch_keys: {existing_batch_keys}")
    
#     non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
#     print(f"  non_tensor_batch_keys_to_pop: {non_tensor_batch_keys_to_pop}")
    
#     try:
#         gen_batch = batch.pop(
#             batch_keys=existing_batch_keys,  # ← 只pop存在的key
#             non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
#         )
#         print(f"  ✓ pop 成功")
#         print(f"  gen_batch.batch 字段: {list(gen_batch.batch.keys())}")
#         print(f"  gen_batch.non_tensor_batch 字段: {list(gen_batch.non_tensor_batch.keys())}")
#         return gen_batch
#     except Exception as e:
#         print(f"  ✗ pop 失败: {e}")
#         import traceback
#         traceback.print_exc()
#         return None


# def main():
#     print("\n" + "="*80)
#     print("🚀 快速 fit() 流程测试（无需Ray初始化）")
#     print("="*80)
    
#     # Step 1: 加载 tokenizer
#     print("\nStep 1: 加载 tokenizer...")
#     tokenizer = AutoTokenizer.from_pretrained(ACTOR_MODEL_PATH, trust_remote_code=True)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#     print(f"  ✓ Tokenizer 加载成功")
    
#     # Step 2: 创建数据集
#     print("\nStep 2: 创建数据集...")
#     config = {
#         'gsm8k_path': TRAIN_FILES_GSM8K,
#         'math_path': TRAIN_FILES_MATH,
#         'max_prompt_length': 2048,
#         'dataloader_num_workers': 0,
#         'shuffle': True,
#         'seed': 1,
#     }
    
#     dataset = MultiDatasetWithCOT(
#         tokenizer=tokenizer,
#         config=config,
#         is_train=True
#     )
#     print(f"  ✓ 数据集大小: {len(dataset)}")
    
#     # Step 3: 创建 DataLoader
#     print("\nStep 3: 创建 DataLoader...")
#     from torch.utils.data import RandomSampler
    
#     train_dataloader_generator = torch.Generator()
#     train_dataloader_generator.manual_seed(config['seed'])
#     sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    
#     dataloader = DataLoader(
#         dataset=dataset,
#         batch_size=BATCH_SIZE,
#         num_workers=0,
#         drop_last=True,
#         collate_fn=collate_fn,
#         sampler=sampler,
#     )
#     print(f"  ✓ DataLoader: {len(dataloader)} batches")
    
#     # Step 4: 模拟训练循环（只测试第一个batch）
#     print("\nStep 4: 模拟 fit() 训练循环...")
    
#     for i, batch_dict in enumerate(dataloader):
#         if i >= 1:  # 只测试第一个batch
#             break
        
#         print(f"\n{'='*80}")
#         print(f"处理 Batch {i}")
#         print(f"{'='*80}")
        
#         # Step 4.1: 转换为 DataProto
#         print("\n📦 Step 4.1: DataProto.from_single_dict()...")
#         print(f"  batch_dict keys: {list(batch_dict.keys())}")
        
#         batch = DataProto.from_single_dict(batch_dict)
        
#         if batch.batch is None or len(batch.batch) == 0:
#             print(f"  ✗ batch.batch 为空!")
#             continue
        
#         print(f"  ✓ batch.batch 创建成功")
#         print(f"    batch.batch 字段: {list(batch.batch.keys())}")
#         print(f"    batch.batch['input_ids'] shape: {batch.batch['input_ids'].shape}")
#         print(f"    batch.non_tensor_batch 字段: {list(batch.non_tensor_batch.keys())}")
        
#         # Step 4.2: 添加 uid（真实训练中会做）
#         print("\n📝 Step 4.2: 添加 uid...")
#         batch.non_tensor_batch["uid"] = np.array(
#             [str(uuid.uuid4()) for _ in range(len(batch.batch))], 
#             dtype=object
#         )
#         print(f"  ✓ uid 已添加")
        
#         # Step 4.3: 模拟 _get_gen_batch()
#         print("\n🎯 Step 4.3: 模拟 _get_gen_batch()...")
#         gen_batch = simulate_get_gen_batch(batch, async_rollout_mode=False)
        
#         if gen_batch is None:
#             print("  ✗ _get_gen_batch 失败，停止测试")
#             break
        
#         # Step 4.4: repeat（GRPO需要）
#         print(f"\n🔄 Step 4.4: repeat(n={N_ROLLOUTS})...")
#         print(f"  repeat 前 batch_size: {len(gen_batch.batch['input_ids'])}")
        
#         gen_batch_repeated = gen_batch.repeat(repeat_times=N_ROLLOUTS, interleave=True)
        
#         print(f"  repeat 后 batch_size: {len(gen_batch_repeated.batch['input_ids'])}")
#         print(f"  gen_batch_repeated.batch 字段: {list(gen_batch_repeated.batch.keys())}")
#         print(f"  gen_batch_repeated.non_tensor_batch 字段: {list(gen_batch_repeated.non_tensor_batch.keys())}")
        
#         # 检查 dataset_name 和 question 是否保留
#         if 'dataset_name' in gen_batch_repeated.non_tensor_batch:
#             dataset_names = gen_batch_repeated.non_tensor_batch['dataset_name']
#             print(f"  ✓ dataset_name 已保留: {dataset_names[:8]}")
#         else:
#             print(f"  ✗ dataset_name 丢失!")
        
#         if 'question' in gen_batch_repeated.non_tensor_batch:
#             questions = gen_batch_repeated.non_tensor_batch['question']
#             print(f"  ✓ question 已保留: {len(questions)} 个")
#             print(f"    示例: {questions[0][:50]}...")
#         else:
#             print(f"  ✗ question 丢失!")
        
#         # Step 4.5: 模拟 COT 增强（不真正调用，只检查数据是否可用）
#         print(f"\n🎨 Step 4.5: 检查 COT 增强所需数据...")
        
#         # COT增强需要的字段
#         required_for_cot = ['dataset_name', 'question']
#         all_present = True
#         for field in required_for_cot:
#             if field in gen_batch_repeated.non_tensor_batch:
#                 print(f"  ✓ {field}: 存在")
#             else:
#                 print(f"  ✗ {field}: 缺失!")
#                 all_present = False
        
#         if all_present:
#             print(f"\n  ✅ COT增强所需的所有字段都存在!")
#             print(f"  可以调用: get_multi_dataset_simple_cot_examples()")
#         else:
#             print(f"\n  ❌ 缺少COT增强所需字段，会导致COT增强失败")
        
#         # 总结
#         print(f"\n{'='*80}")
#         print(f"✅ 第一个batch处理成功!")
#         print(f"{'='*80}")
#         print(f"关键检查点:")
#         print(f"  ✓ DataProto 转换成功")
#         print(f"  ✓ _get_gen_batch() 成功")
#         print(f"  ✓ repeat() 成功")
#         print(f"  ✓ COT增强所需字段完整" if all_present else "  ✗ COT增强所需字段缺失")
#         print(f"\n下一步: 可以尝试真实训练!")
#         print(f"{'='*80}\n")
        
#         break  # 只测试一个batch
    
#     print("\n" + "="*80)
#     print("📋 快速测试完成")
#     print("="*80)
#     print("\n如果上面所有步骤都成功，说明数据处理流程正常。")
#     print("可以运行真实训练: bash examples/grpo_trainer/test_k_cot.sh")
#     print("="*80 + "\n")


# if __name__ == '__main__':
#     main()




#!/usr/bin/env python3
"""
快速测试 ray_trainer.fit() 的数据流
无需 Ray 初始化，直接测试数据处理逻辑

用法:
    python quick_fit_test.py
"""

import sys
sys.path.insert(0, '/nas/dhl/verl')

from examples.grpo_trainer.multi_dataset_with_cot import MultiDatasetWithCOT
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from verl.utils.dataset.rl_dataset import collate_fn
from verl import DataProto
import numpy as np
import uuid
import torch

# 配置
ACTOR_MODEL_PATH = "/nas/models/Qwen3-8B"
TRAIN_FILES_GSM8K = "/nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet"
TRAIN_FILES_MATH = "/nas/dhl/Datasets/my_Datasets/MATH/train.parquet"
BATCH_SIZE = 8
N_ROLLOUTS = 4

def simulate_get_gen_batch(batch: DataProto, async_rollout_mode=False):
    """
    模拟 _get_gen_batch 方法（ray_trainer.py 第656-671行）
    """
    print("\n🔍 模拟 _get_gen_batch()...")
    
    # 检查当前 batch 中有哪些字段
    print(f"  batch.batch 字段: {list(batch.batch.keys())}")
    print(f"  batch.non_tensor_batch 字段: {list(batch.non_tensor_batch.keys())}")
    
    reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()
    print(f"  reward_model_keys: {reward_model_keys}")
    
    # pop those keys for generation
    batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
    
    # 🔑 关键检查：哪些key存在，哪些不存在
    print(f"  尝试pop的batch_keys: {batch_keys_to_pop}")
    for key in batch_keys_to_pop:
        exists = key in batch.batch.keys()
        print(f"    {key}: {'✓ 存在' if exists else '✗ 不存在'}")
    
    # 🔑 修复：只pop存在的key
    existing_batch_keys = [k for k in batch_keys_to_pop if k in batch.batch.keys()]
    print(f"  实际pop的batch_keys: {existing_batch_keys}")
    
    non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
    print(f"  non_tensor_batch_keys_to_pop: {non_tensor_batch_keys_to_pop}")
    
    try:
        gen_batch = batch.pop(
            batch_keys=existing_batch_keys,  # ← 只pop存在的key
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )
        print(f"  ✓ pop 成功")
        print(f"  gen_batch.batch 字段: {list(gen_batch.batch.keys())}")
        print(f"  gen_batch.non_tensor_batch 字段: {list(gen_batch.non_tensor_batch.keys())}")
        return gen_batch
    except Exception as e:
        print(f"  ✗ pop 失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("\n" + "="*80)
    print("🚀 快速 fit() 流程测试（无需Ray初始化）")
    print("="*80)
    
    # Step 1: 加载 tokenizer
    print("\nStep 1: 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ACTOR_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  ✓ Tokenizer 加载成功")
    
    # Step 2: 创建数据集
    print("\nStep 2: 创建数据集...")
    config = {
        'gsm8k_path': TRAIN_FILES_GSM8K,
        'math_path': TRAIN_FILES_MATH,
        'max_prompt_length': 2048,
        'dataloader_num_workers': 0,
        'shuffle': True,
        'seed': 1,
    }
    
    dataset = MultiDatasetWithCOT(
        tokenizer=tokenizer,
        config=config,
        is_train=True
    )
    print(f"  ✓ 数据集大小: {len(dataset)}")
    
    # Step 3: 创建 DataLoader
    print("\nStep 3: 创建 DataLoader...")
    from torch.utils.data import RandomSampler
    
    train_dataloader_generator = torch.Generator()
    train_dataloader_generator.manual_seed(config['seed'])
    sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        drop_last=True,
        collate_fn=collate_fn,
        sampler=sampler,
    )
    print(f"  ✓ DataLoader: {len(dataloader)} batches")
    
    # Step 4: 模拟训练循环（只测试第一个batch）
    print("\nStep 4: 模拟 fit() 训练循环...")
    
    for i, batch_dict in enumerate(dataloader):
        if i >= 1:  # 只测试第一个batch
            break
        
        print(f"\n{'='*80}")
        print(f"处理 Batch {i}")
        print(f"{'='*80}")
        
        # Step 4.1: 转换为 DataProto
        print("\n📦 Step 4.1: DataProto.from_single_dict()...")
        print(f"  batch_dict keys: {list(batch_dict.keys())}")
        
        batch = DataProto.from_single_dict(batch_dict)
        
        if batch.batch is None or len(batch.batch) == 0:
            print(f"  ✗ batch.batch 为空!")
            continue
        
        print(f"  ✓ batch.batch 创建成功")
        print(f"    batch.batch 字段: {list(batch.batch.keys())}")
        print(f"    batch.batch['input_ids'] shape: {batch.batch['input_ids'].shape}")
        print(f"    batch.non_tensor_batch 字段: {list(batch.non_tensor_batch.keys())}")
        
        # Step 4.2: 添加 uid（真实训练中会做）
        print("\n📝 Step 4.2: 添加 uid...")
        batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(batch.batch))], 
            dtype=object
        )
        print(f"  ✓ uid 已添加")
        
        # Step 4.3: 模拟 _get_gen_batch()
        print("\n🎯 Step 4.3: 模拟 _get_gen_batch()...")
        gen_batch = simulate_get_gen_batch(batch, async_rollout_mode=False)
        
        if gen_batch is None:
            print("  ✗ _get_gen_batch 失败，停止测试")
            break
        
        # Step 4.4: repeat（GRPO需要）
        print(f"\n🔄 Step 4.4: repeat(n={N_ROLLOUTS})...")
        print(f"  repeat 前 batch_size: {len(gen_batch.batch['input_ids'])}")
        
        gen_batch_repeated = gen_batch.repeat(repeat_times=N_ROLLOUTS, interleave=True)
        
        print(f"  repeat 后 batch_size: {len(gen_batch_repeated.batch['input_ids'])}")
        print(f"  gen_batch_repeated.batch 字段: {list(gen_batch_repeated.batch.keys())}")
        print(f"  gen_batch_repeated.non_tensor_batch 字段: {list(gen_batch_repeated.non_tensor_batch.keys())}")
        
        # 检查 dataset_name 和 question 是否保留
        if 'dataset_name' in gen_batch_repeated.non_tensor_batch:
            dataset_names = gen_batch_repeated.non_tensor_batch['dataset_name']
            print(f"  ✓ dataset_name 已保留: {dataset_names[:8]}")
        else:
            print(f"  ✗ dataset_name 丢失!")
        
        if 'question' in gen_batch_repeated.non_tensor_batch:
            questions = gen_batch_repeated.non_tensor_batch['question']
            print(f"  ✓ question 已保留: {len(questions)} 个")
            print(f"    示例: {questions[0][:50]}...")
        else:
            print(f"  ✗ question 丢失!")
        
        # Step 4.5: 检查所有关键字段
        print(f"\n🔍 Step 4.5: 检查所有关键字段...")
        
        # 检查 batch 中的 tensor 字段
        print(f"\n  Tensor 字段检查:")
        tensor_fields = ['input_ids', 'attention_mask', 'position_ids']
        for field in tensor_fields:
            if field in batch.batch:
                print(f"    ✓ {field}: 存在 (shape={batch.batch[field].shape})")
            else:
                print(f"    ✗ {field}: 缺失!")
        
        # 检查 non_tensor 字段
        print(f"\n  Non-tensor 字段检查:")
        
        # reward_model 检查（训练必需）
        if 'reward_model' in batch.non_tensor_batch:
            print(f"    ✓ reward_model: 存在")
            # 检查 reward_model 的内容
            rm = batch.non_tensor_batch['reward_model'][0]
            if isinstance(rm, dict):
                has_gt = 'ground_truth' in rm
                has_style = 'style' in rm
                print(f"      - style: {'✓ 存在' if has_style else '✗ 缺失'}")
                print(f"      - ground_truth: {'✓ 存在' if has_gt else '✗ 缺失'}")
        else:
            print(f"    ✗ reward_model: 缺失! (会导致 reward_fn 失败)")
        
        # COT 字段检查
        print(f"\n  COT 增强字段检查:")
        cot_fields = ['dataset_name', 'question']
        cot_all_present = True
        for field in cot_fields:
            if field in gen_batch_repeated.non_tensor_batch:
                print(f"    ✓ {field}: 存在")
            else:
                print(f"    ✗ {field}: 缺失!")
                cot_all_present = False
        
        if cot_all_present:
            print(f"\n  ✅ COT增强所需字段完整")
        else:
            print(f"\n  ❌ COT增强字段缺失")
        
        # 最终总结
        print(f"\n{'='*80}")
        print(f"📋 第一个batch处理总结")
        print(f"{'='*80}")
        
        # 检查所有关键步骤
        checks = {
            "DataProto 转换": True,
            "_get_gen_batch()": gen_batch is not None,
            "repeat() 操作": True,
            "reward_model 字段": 'reward_model' in batch.non_tensor_batch,
            "COT 所需字段": cot_all_present,
        }
        
        all_pass = all(checks.values())
        
        for check_name, passed in checks.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {check_name}")
        
        print(f"\n{'='*80}")
        if all_pass:
            print(f"✅✅✅ 所有检查通过！可以开始训练！")
            print(f"\n运行: bash examples/grpo_trainer/test_k_cot.sh")
        else:
            print(f"❌ 部分检查失败，请修复后再训练")
            print(f"\n使用对比工具查看详细差异:")
            print(f"  python examples/grpo_trainer/compare_datasets.py")
        print(f"{'='*80}\n")
        
        break  # 只测试一个batch
    
    print("\n" + "="*80)
    print("📋 快速测试完成")
    print("="*80)
    print("\n如果上面所有步骤都成功，说明数据处理流程正常。")
    print("可以运行真实训练: bash examples/grpo_trainer/test_k_cot.sh")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()

