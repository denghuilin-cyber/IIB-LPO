#!/bin/bash
# 多数据集 + COT增强训练脚本
# 支持 GSM8K + MATH 混合训练，动态COT example匹配

set -x

# 配置
export WANDB_MODE="offline"
ACTOR_MODEL_PATH="/nas/models/Qwen3-8B"

# ⭐ Batch Size 配置（参考官方文档）
BATCH_SIZE=4                # data.train_batch_size: 每次采样8个prompt
N_ROLLOUTS=4                # rollout.n: 每个prompt生成4次
                            # → 总responses = 8 × 4 = 32
PPO_MINI_BATCH=8            # ppo_mini_batch_size: 32÷8 = 4次mini batch更新
PPO_MICRO_BATCH_PER_GPU=4   # ppo_micro_batch_size_per_gpu: 每个GPU处理4个

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
N_GPUS_PER_NODE=8
TENSOR_PARALLEL=1

# 输出目录和日志
OUTPUT_DIR="/nas/dhl/outputs/test_cot_output"
LOG_FILE="${OUTPUT_DIR}/training_$(date +%Y%m%d_%H%M%S).log"

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

echo "================================"
echo "🚀 开始训练"
echo "================================"
echo "输出目录: ${OUTPUT_DIR}"
echo "日志文件: ${LOG_FILE}"
echo "Batch配置: batch_size=${BATCH_SIZE}, rollouts=${N_ROLLOUTS}, total_responses=$((BATCH_SIZE * N_ROLLOUTS))"
echo "================================"

# 使用 nohup 后台运行，并记录日志
python3 -m verl.trainer.main_ppo \
  ++data.custom_cls.path=pkg://examples.grpo_trainer.multi_dataset_with_cot \
  ++data.custom_cls.name=MultiDatasetWithCOT \
  ++data.gsm8k_path=$TRAIN_FILES_GSM8K \
  ++data.math_path=$TRAIN_FILES_MATH \
  \
  data.val_files=$VAL_FILES \
  data.val_batch_size=8 \
  data.train_batch_size=$BATCH_SIZE \
  data.max_prompt_length=1024 \
  data.max_response_length=1024 \
  data.shuffle=True \
  trainer.test_freq=0 \
  \
  actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_PER_GPU \
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
  trainer.test_freq=0 > ${LOG_FILE} 2>&1 &

# 获取进程ID
#TRAIN_PID=$!

echo ""
echo "============================================================"
echo "🎉 训练已启动（后台运行）"
echo ""
echo "📊 配置信息："
echo "   GPU数量: $N_GPUS_PER_NODE"
echo "   Tensor Parallel: $TENSOR_PARALLEL (每个GPU独立运行完整模型)"
echo "   Rollout次数: $N_ROLLOUTS (每个问题生成4个不同的回答)"
echo "   Batch Size: $BATCH_SIZE"
echo ""
echo "⚠️ 注意：已禁用验证阶段 (trainer.test_freq=0) 以避免验证集数据加载问题"
echo "   这样可以专注于查看COT增强的效果"
echo ""
echo "📝 每个样本显示3部分："
echo "   📌 原始问题 - 从数据集读取的问题"
echo "   📚 示例COT - 从COT文件匹配的推理示例"
echo "   ✨ 拼接后的完整Prompt - 最终喂给模型的内容"
echo ""
echo "⭐ 重点关注 '✨ 拼接后的完整Prompt' 部分！"
echo "   这就是模型实际看到的输入："
echo "   [示例COT] + \\n\\n + [原始问题]"
echo ""
echo "============================================================"

