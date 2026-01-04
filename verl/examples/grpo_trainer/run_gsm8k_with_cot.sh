#!/bin/bash
# Example script for running GRPO on GSM8K with COT augmentation
# This script shows how to use separate COT examples for each rollout

set -x

# Paths
TRAIN_DATA="/nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet"
VAL_DATA="/nas/dhl/Datasets/my_Datasets/gsm8k/test.parquet"  # Adjust as needed
COT_DATA="/nas/dhl/CVAE/Datasets/GSM8K/processed/train_k_shot_GSM8K.jsonl"

# Model paths
ACTOR_MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"
CRITIC_MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"  # If using critic
REWARD_MODEL_PATH=""  # If using reward model

# Training configuration
N_ROLLOUTS=4  # Number of rollouts per prompt in GRPO
BATCH_SIZE=256
EPOCHS=3
LR=1e-6

# COT configuration
COT_FORMAT="Here's a similar example to help you:\n\nQuestion: {question}\n\nStep-by-step solution:\n{rationale}\n\nFinal Answer: {final_answer}\n\n---\n\nNow solve the following problem step by step:"
COT_MATCH_BY="question"  # or "id" if your data has matching IDs
COT_SAMPLING="sequential"  # sequential, random_with_replacement, random_without_replacement

python3 -m verl.trainer.main_ppo \
  data.train_files=$TRAIN_DATA \
  data.val_files=$VAL_DATA \
  data.train_batch_size=$BATCH_SIZE \
  data.val_batch_size=1312 \
  data.max_prompt_length=512 \
  data.max_response_length=512 \
  data.shuffle=True \
  data.prompt_key=question \
  +data.custom_cls.path=examples.grpo_trainer.gsm8k_dataset_with_cot \
  +data.custom_cls.name=GSM8KParquetDatasetWithCOT \
  actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
  actor_rollout_ref.actor.optim.lr=$LR \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.n=$N_ROLLOUTS \
  actor_rollout_ref.rollout.temperature=0.7 \
  actor_rollout_ref.rollout.top_p=0.9 \
  +actor_rollout_ref.rollout.cot_augmentation.enable=true \
  +actor_rollout_ref.rollout.cot_augmentation.cot_file_path=$COT_DATA \
  +actor_rollout_ref.rollout.cot_augmentation.cot_format_template="$COT_FORMAT" \
  +actor_rollout_ref.rollout.cot_augmentation.match_by=$COT_MATCH_BY \
  +actor_rollout_ref.rollout.cot_augmentation.sampling_strategy=$COT_SAMPLING \
  +actor_rollout_ref.rollout.cot_augmentation.add_separator=true \
  +actor_rollout_ref.rollout.cot_augmentation.separator="\n\n" \
  +actor_rollout_ref.rollout.cot_augmentation.use_full_cot=true \
  algorithm.adv_estimator=grpo \
  algorithm.gamma=1.0 \
  trainer.total_epochs=$EPOCHS \
  trainer.project_name=gsm8k_grpo_cot \
  trainer.experiment_name=grpo_with_cot_examples \
  trainer.logger=['console','wandb'] \
  trainer.default_local_dir=./checkpoints/gsm8k_grpo_cot \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1

# Note: Adjust paths, GPU settings, and other hyperparameters according to your setup

