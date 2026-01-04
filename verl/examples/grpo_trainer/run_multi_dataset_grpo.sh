#!/bin/bash
# GSM8K + MATH 混合训练（带COT增强）
# 修复版：使用 MultiDatasetWithCOT 数据集类

set -x

# 配置
export WANDB_MODE="offline"
ACTOR_MODEL_PATH="/nas/models/Qwen3-8B"
N_ROLLOUTS=8
BATCH_SIZE=256
EPOCHS=3
LR=1e-6

# 数据文件
TRAIN_FILES_GSM8K="/nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet"
TRAIN_FILES_MATH="/nas/dhl/Datasets/my_Datasets/MATH/train.parquet"
VAL_FILES="/nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet"

# COT文件路径
GSM8K_COT="/nas/dhl/CVAE/Datasets/GSM8K/processed/train_k_shot_GSM8K.jsonl"
MATH_COT="/nas/dhl/CVAE/Datasets/MATH/processed/train_k_shot_MATH.jsonl"

# GPU
N_GPUS_PER_NODE=8

# 输出
OUTPUT_DIR="./checkpoints/gsm8k_math_grpo_cot"

python3 -m verl.trainer.main_ppo \
  ++data.custom_cls.path=pkg://examples.grpo_trainer.multi_dataset_with_cot \
  ++data.custom_cls.name=MultiDatasetWithCOT \
  ++data.gsm8k_path=$TRAIN_FILES_GSM8K \
  ++data.math_path=$TRAIN_FILES_MATH \
  \
  data.val_files=$VAL_FILES \
  data.train_batch_size=$BATCH_SIZE \
  data.val_batch_size=512 \
  data.max_prompt_length=4096 \
  data.max_response_length=4096 \
  data.shuffle=True \
  \
  actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.optim.lr=$LR \
  actor_rollout_ref.actor.ppo_mini_batch_size=256 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.n=$N_ROLLOUTS \
  actor_rollout_ref.rollout.temperature=0.7 \
  actor_rollout_ref.rollout.top_p=0.9 \
  actor_rollout_ref.rollout.response_length=512 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
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
  ++actor_rollout_ref.rollout.cot_augmentation.debug_print_full_prompt=false \
  \
  algorithm.adv_estimator=grpo \
  algorithm.gamma=1.0 \
  algorithm.use_kl_in_reward=False \
  \
  trainer.total_epochs=$EPOCHS \
  trainer.critic_warmup=0 \
  trainer.project_name=gsm8k_math_grpo \
  trainer.experiment_name=grpo_8rollouts_cot \
  trainer.logger='[console,wandb]' \
  trainer.default_local_dir=$OUTPUT_DIR \
  trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
  trainer.nnodes=1 \
  trainer.save_freq=50 \
  trainer.test_freq=10

# ============================================================================
# 🔑 关键修改说明:
#   1. ✅ 使用 MultiDatasetWithCOT 数据集类（必需！）
#      ++data.custom_cls.path=pkg://examples.grpo_trainer.multi_dataset_with_cot
#      ++data.custom_cls.name=MultiDatasetWithCOT
#   
#   2. ✅ 使用简化的配置方式（Hydra友好）
#      ++data.gsm8k_path=$TRAIN_FILES_GSM8K
#      ++data.math_path=$TRAIN_FILES_MATH
#      这样每个样本会自动添加 dataset_name 字段
#   
#   3. ✅ 配置 COT 映射
#      dataset_cot_mapping.gsm8k → GSM8K 的 COT 文件
#      dataset_cot_mapping.math → MATH 的 COT 文件
#   
#   4. ✅ 启用调试输出
#      debug_print_augmented_prompts=true → 打印增强后的 prompt
#      debug_num_samples=3 → 打印前3个样本
#      debug_print_full_prompt=false → 打印截断版本（节省日志空间）
#                                      设为 true 可打印完整prompt
#      verbose=true → 详细日志
# ============================================================================
#
# 📝 期望看到的输出:
#   ✅ Loading dataset: gsm8k
#      Loaded 7473 samples
#   ✅ Loading dataset: math
#      Loaded xxxx samples
#   ✅ Loaded xxxx total samples from 2 datasets
#   ✅ Initializing multi-dataset COT loader...
#   ✅ 🎯 增强后的Prompt (样本 0, 数据集: gsm8k):
#      [示例COT] + [原始问题]
# ============================================================================

