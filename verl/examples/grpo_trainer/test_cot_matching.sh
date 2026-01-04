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

# 数据文件
TRAIN_FILES_GSM8K="/nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet"
TRAIN_FILES_MATH="/nas/dhl/Datasets/my_Datasets/MATH/train.parquet"
VAL_FILES="/nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet"

# COT文件路径
GSM8K_COT="/nas/dhl/CVAE/Datasets/GSM8K/processed/train_k_shot_GSM8K.jsonl"
MATH_COT="/nas/dhl/CVAE/Datasets/MATH/processed/train_k_shot_MATH.jsonl"

# GPU
N_GPUS_PER_NODE=1  # 只用1个GPU测试

# 输出
OUTPUT_DIR="./test_cot_output"

python3 -m verl.trainer.main_ppo \
    ++data.custom_cls.path=pkg://examples.grpo_trainer.multi_dataset_with_cot \
    ++data.custom_cls.name=MultiDatasetWithCOT \
    ++data.gsm8k_path=$TRAIN_FILES_GSM8K \
    ++data.math_path=$TRAIN_FILES_MATH \
    \
    data.val_files=$VAL_FILES \
    data.train_batch_size=$BATCH_SIZE \
    data.val_batch_size=8 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.shuffle=True \
    \
    actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
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
    ++actor_rollout_ref.rollout.cot_augmentation.debug_num_samples=5 \
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
    trainer.test_freq=1

    echo ""
    echo "============================================================"
    echo "🎉 测试完成！请查看上面的输出"
    echo ""
    echo "期望看到的关键信息："
    echo "✅ Loading dataset: gsm8k"
    echo "✅ Loading dataset: math"
    echo "✅ Loaded XXX total samples from 2 datasets"
    echo "✅ Initializing multi-dataset COT loader..."
    echo ""
    echo "📝 完整的拼接结果（每个样本会打印3部分）："
    echo "   📌 Original Prompt - 原始问题（完整版）"
    echo "   📚 COT Example - 示例COT（完整版）"
    echo "   ✨ Augmented Prompt - 拼接后的完整prompt（这就是喂给模型的内容！）"
    echo ""
    echo "⭐ 重点关注 'Augmented Prompt' 部分，这就是最终喂给模型的prompt！"
    echo "============================================================"

