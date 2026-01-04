# #!/bin/bash
# # 快速测试：验证多数据集COT增强
# # 只运行少量步骤，查看输出格式

# set -x

# # 配置
# export WANDB_MODE="offline"
# ACTOR_MODEL_PATH="/nas/models/Qwen3-8B"
# N_ROLLOUTS=4  # 减少rollout次数加快测试
# BATCH_SIZE=8  # 小batch加快测试
# EPOCHS=1


# # 假设你的项目根目录是 /nas/dhl/verl
# export PYTHONPATH="/nas/dhl/verl:$PYTHONPATH"


# # 数据文件
# TRAIN_FILES_GSM8K="/nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet"
# TRAIN_FILES_MATH="/nas/dhl/Datasets/my_Datasets/MATH/train.parquet"
# #VAL_FILES="/nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet"
# VAL_FILES=""

# # COT文件路径
# GSM8K_COT="/nas/dhl/CVAE/Datasets/GSM8K/processed/train_k_shot_GSM8K.jsonl"
# MATH_COT="/nas/dhl/CVAE/Datasets/MATH/processed/train_k_shot_MATH.jsonl"

# # GPU配置
# N_GPUS_PER_NODE=8           # 使用8张GPU
# TENSOR_PARALLEL=1           # Tensor Parallelism大小（1=不分割模型，最大吞吐量）
# '''假设模型是一个大蛋糕：

# tensor_model_parallel_size=1:
# ┌────────────────┐
# │  完整模型       │  GPU 0
# └────────────────┘

# tensor_model_parallel_size=2:
# ┌────────┐┌────────┐
# │模型左半 ││模型右半 │
# │  部分  ││  部分  │
# └────────┘└────────┘
#   GPU 0     GPU 1

# tensor_model_parallel_size=4:
# ┌───┐┌───┐┌───┐┌───┐
# │1/4││2/4││3/4││4/4│
# └───┘└───┘└───┘└───┘
# GPU0  GPU1  GPU2  GPU3'''
# # 输出
# OUTPUT_DIR="/nas/dhl/outputs/test_cot_output"

# python3 -m verl.trainer.main_ppo \
#   ++data.custom_cls.path=pkg://examples.grpo_trainer.multi_dataset_with_cot \
#   ++data.custom_cls.name=MultiDatasetWithCOT \
#   ++data.gsm8k_path=$TRAIN_FILES_GSM8K \
#   ++data.math_path=$TRAIN_FILES_MATH \
#   \
#   data.val_files=$VAL_FILES \
#   data.train_batch_size=$BATCH_SIZE \
#   data.val_batch_size=8 \
#   data.max_prompt_length=2048 \
#   data.max_response_length=2048 \
#   data.shuffle=True \
#   trainer.test_freq=0 \
#   \
#   actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
#   actor_rollout_ref.actor.strategy=fsdp \
#   actor_rollout_ref.actor.optim.lr=1e-6 \
#   actor_rollout_ref.actor.ppo_mini_batch_size=8 \
#   actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
#   actor_rollout_ref.actor.use_kl_loss=True \
#   actor_rollout_ref.actor.kl_loss_coef=0.001 \
#   actor_rollout_ref.actor.entropy_coeff=0 \
#   actor_rollout_ref.model.enable_gradient_checkpointing=True \
#   actor_rollout_ref.actor.fsdp_config.param_offload=False \
#   actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
#   actor_rollout_ref.rollout.name=vllm \
#   actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
#   actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_PARALLEL \
#   actor_rollout_ref.rollout.n=$N_ROLLOUTS \
#   actor_rollout_ref.rollout.temperature=0.7 \
#   actor_rollout_ref.rollout.top_p=0.9 \
#   actor_rollout_ref.rollout.response_length=256 \
#   actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
#   actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
#   actor_rollout_ref.ref.fsdp_config.param_offload=True \
#   \
#   ++actor_rollout_ref.rollout.cot_augmentation.enable=true \
#   ++actor_rollout_ref.rollout.cot_augmentation.use_multi_dataset=true \
#   ++actor_rollout_ref.rollout.cot_augmentation.dataset_cot_mapping.gsm8k=$GSM8K_COT \
#   ++actor_rollout_ref.rollout.cot_augmentation.dataset_cot_mapping.math=$MATH_COT \
#   ++actor_rollout_ref.rollout.cot_augmentation.skip_on_mismatch=true \
#   ++actor_rollout_ref.rollout.cot_augmentation.verbose=true \
#   ++actor_rollout_ref.rollout.cot_augmentation.use_full_cot=true \
#   ++actor_rollout_ref.rollout.cot_augmentation.debug_print_augmented_prompts=true \
#   ++actor_rollout_ref.rollout.cot_augmentation.debug_num_samples=5 \
#   ++actor_rollout_ref.rollout.cot_augmentation.debug_print_full_prompt=true \
#   \
#   algorithm.adv_estimator=grpo \
#   algorithm.gamma=1.0 \
#   algorithm.use_kl_in_reward=False \
#   \
#   trainer.total_epochs=$EPOCHS \
#   trainer.critic_warmup=0 \
#   trainer.project_name=test_cot \
#   trainer.experiment_name=test_multi_dataset_cot \
#   trainer.logger='[console]' \
#   trainer.default_local_dir=$OUTPUT_DIR \
#   trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
#   trainer.nnodes=1 \
#   trainer.save_freq=1000 \
#   trainer.test_freq=0   


# '''
# 为什么这样设计？
# ✅ 灵活性：每个rollout可以使用不同的COT example
# ✅ 解耦：数据集只负责加载原始数据，COT增强是训练策略
# ✅ 可控性：可以轻松开启/关闭COT增强
# 📊 完整的数据流总结
# 1. 数据文件（train.parquet）
#    ↓ 包含原始问题和答案
   
# 2. MultiDatasetWithCOT.__getitem__()
#    ↓ 读取并 tokenize 原始问题
#    ↓ 返回: {input_ids, dataset_name, question, reward_model, ...}
#    ↓ 注意：这里的 input_ids 是原始问题的 token
   
# 3. DataLoader
#    ↓ collate_fn 合并成 batch
#    ↓ batch_size = 8
   
# 4. ray_trainer.fit()
#    ↓ batch = DataProto.from_single_dict(batch_dict)
#    ↓ gen_batch = self._get_gen_batch(batch)
#    ↓ gen_batch = gen_batch.repeat(n=4)  # batch_size变成32
   
# 5. COT 增强（⭐ 关键步骤）
#    ↓ gen_batch = self.cot_augmenter.augment(gen_batch)
#    ↓ 对每个原始问题:
#    ↓   - 根据 dataset_name 选择COT库
#    ↓   - 根据 question 匹配COT example
#    ↓   - 拼接: COT + "\n\n" + 原始问题
#    ↓   - 重新 tokenize
#    ↓   - 更新 gen_batch 的 input_ids
#    ↓
#    ↓ 现在 gen_batch 的 input_ids 是增强后的 prompt
   
# 6. 模型生成
#    ↓ gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
#    ↓ 模型看到的是：[COT Example] + [原始问题]

# '''




#!/bin/bash
# 快速测试：验证多数据集COT增强
# 只运行少量步骤，查看输出格式
set -x

# 配置
export WANDB_MODE="offline"
ACTOR_MODEL_PATH="/nas/models/Qwen3-8B"
N_ROLLOUTS=4  # 减少rollout次数加快测试
BATCH_SIZE=8  # 小batch加快测试
EPOCHS=1


# 假设你的项目根目录是 /nas/dhl/verl
export PYTHONPATH="/nas/dhl/verl:$PYTHONPATH"


# 数据文件
TRAIN_FILES_GSM8K="/nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet"
TRAIN_FILES_MATH="/nas/dhl/Datasets/my_Datasets/MATH/train.parquet"
#VAL_FILES="/nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet"
VAL_FILES=""

# COT文件路径
GSM8K_COT="/nas/dhl/CVAE/Datasets/GSM8K/processed/train_k_shot_GSM8K.jsonl"
MATH_COT="/nas/dhl/CVAE/Datasets/MATH/processed/train_k_shot_MATH.jsonl"

# GPU配置
N_GPUS_PER_NODE=8           # 使用8张GPU
TENSOR_PARALLEL=1           # Tensor Parallelism大小（1=不分割模型，最大吞吐量）


# 输出目录和日志
OUTPUT_DIR="/nas/dhl/outputs/test_cot_output"
LOG_FILE="${OUTPUT_DIR}/training_$(date +%Y%m%d_%H%M%S).log"


echo "================================"
echo "🚀 开始训练"
echo "================================"
echo "输出目录: ${OUTPUT_DIR}"
echo "日志文件: ${LOG_FILE}"
echo "Batch配置: batch_size=${BATCH_SIZE}, rollouts=${N_ROLLOUTS}, total_responses=$((BATCH_SIZE * N_ROLLOUTS))"
echo "================================"


python3 -m verl.trainer.main_ppo \
  ++data.custom_cls.path=pkg://examples.grpo_trainer.multi_dataset_with_cot \
  ++data.custom_cls.name=MultiDatasetWithCOT \
  ++data.gsm8k_path=$TRAIN_FILES_GSM8K \
  ++data.math_path=$TRAIN_FILES_MATH \
  \
  data.val_files=$VAL_FILES \
  data.val_batch_size=8 \
  data.train_batch_size=$BATCH_SIZE \
  data.max_prompt_length=2048 \
  data.max_response_length=1024 \
  data.shuffle=True \
  trainer.test_freq=0 \
  \
  actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_PARALLEL \
  actor_rollout_ref.rollout.n=$N_ROLLOUTS \
  actor_rollout_ref.rollout.temperature=0.7 \
  actor_rollout_ref.rollout.top_p=0.9 \
  actor_rollout_ref.rollout.response_length=256 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  \
  ++actor_rollout_ref.rollout.cot_augmentation.enable=true \
  ++actor_rollout_ref.rollout.cot_augmentation.use_multi_dataset=true \
  ++actor_rollout_ref.rollout.cot_augmentation.dataset_cot_mapping.gsm8k=$GSM8K_COT \
  ++actor_rollout_ref.rollout.cot_augmentation.dataset_cot_mapping.math=$MATH_COT \
  ++actor_rollout_ref.rollout.cot_augmentation.skip_on_mismatch=true \
  ++actor_rollout_ref.rollout.cot_augmentation.verbose=true \
  ++actor_rollout_ref.rollout.cot_augmentation.use_full_cot=true \
  ++actor_rollout_ref.rollout.cot_augmentation.debug_print_augmented_prompts=true \
  ++actor_rollout_ref.rollout.cot_augmentation.debug_num_samples=3 \
  ++actor_rollout_ref.rollout.cot_augmentation.debug_print_full_prompt=true \
  \
  algorithm.adv_estimator=grpo \
  algorithm.gamma=1.0 \
  algorithm.use_kl_in_reward=False \
  \
  trainer.total_epochs=$EPOCHS \
  trainer.critic_warmup=0 \
  trainer.project_name=test_cot \
  trainer.experiment_name=test_multi_dataset_cot \
  trainer.logger='[console]' \
  trainer.default_local_dir=$OUTPUT_DIR \
  trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
  trainer.nnodes=1 \
  trainer.save_freq=1000 \
  trainer.test_freq=0 > ${LOG_FILE} 2>&1





#tail -f /nas/dhl/outputs/test_cot_output/training_*.log