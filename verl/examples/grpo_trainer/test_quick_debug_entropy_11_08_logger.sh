#!/bin/bash
# 🚀 快速调试脚本：只运行 1 个 step，立即看到 DEBUG 输出
# 用于验证 vLLM 熵计算是否正常工作

# 🆕 必须在最开始设置 Ray 环境变量（在 set -x 之前）
export RAY_ENABLE_DASHBOARD=0
export RAY_DASHBOARD_HOST=""

set -x

# 配置
export WANDB_MODE="offline"
export TRANSFORMERS_OFFLINE=0
export HF_DATASETS_OFFLINE=0

# 🔥 使用最小配置加快调试
ACTOR_MODEL_PATH="/nas/models/qwen2.5-math-1.5B_instruct"
N_ROLLOUTS=1  # 🔥 只生成 1 个 rollout
BATCH_SIZE=8  # ⚠️ 真实训练 batch size 需 ≥ 最小值 8
EPOCHS=1
MAX_STEPS=1   # 🔥 只运行 1 个 step

# 项目路径
export PYTHONPATH="/nas/dhl/verl:$PYTHONPATH"

# 数据文件
TRAIN_FILES_GSM8K="/nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet"
TRAIN_FILES_MATH="/nas/dhl/Datasets/my_Datasets/MATH/train.parquet"
VAL_FILES=""

# COT文件路径
GSM8K_COT="/nas/dhl/CVAE/Datasets/GSM8K/processed/train_k_shot_GSM8K.jsonl"
MATH_COT="/nas/dhl/CVAE/Datasets/MATH/processed/train_k_shot_MATH.jsonl"

# GPU配置
N_GPUS_PER_NODE=8
TENSOR_PARALLEL=1

# 输出目录
OUTPUT_DIR="/nas/dhl/outputs/quick_debug_entropy_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/training_$(date +%Y%m%d_%H%M%S).log"


export WANDB_DIR="${OUTPUT_DIR}/wandb"
mkdir -p "${WANDB_DIR}"



export RAY_ENABLE_DASHBOARD=0
export RAY_DASHBOARD_HOST=""


# 2. 清理 vLLM 的所有 .pyc 缓存
find /opt/dhl_conda_envs/verl_fast/lib/python3.10/site-packages/vllm -name "*.pyc" -delete
find /opt/dhl_conda_envs/verl_fast/lib/python3.10/site-packages/vllm -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# 3. 强制重新编译（可选，但推荐）
python3 -m compileall -f /opt/dhl_conda_envs/verl_fast/lib/python3.10/site-packages/vllm


# 🆕 清理环境
echo "================================"
echo "🧹 清理环境"
echo "================================"
ray stop --force 2>/dev/null || true
sleep 2
rm -rf /tmp/ray/* 2>/dev/null || true

# 清理 vLLM 缓存
VLLM_PATH="/opt/dhl_conda_envs/verl_fast/lib/python3.10/site-packages/vllm"
if [ -d "$VLLM_PATH" ]; then
    echo "  清理 vLLM 缓存..."
    find "$VLLM_PATH" -name "*.pyc" -delete 2>/dev/null || true
    find "$VLLM_PATH" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    python3 -m compileall -q -f "$VLLM_PATH" 2>/dev/null || true
    echo "  ✅ vLLM 缓存已清理"
fi

echo "✅ 环境清理完成"



# 设置日志级别为 DEBUG
export VLLM_LOGGING_LEVEL=DEBUG
export VERL_LOGGING_LEVEL=DEBUG
# 如果 不想输出token-level的熵值 改成INFO即可
echo "VLLM_LOGGING_LEVEL: $VLLM_LOGGING_LEVEL"
echo "VERL_LOGGING_LEVEL: $VERL_LOGGING_LEVEL"


# 🔥 直接输出到终端（不用 tee），实时看到 DEBUG 信息
python3 -m verl.trainer.main_ppo \
  ++data.custom_cls.path=pkg://examples.grpo_trainer.multi_dataset_with_cot \
  ++data.custom_cls.name=MultiDatasetWithCOT \
  ++data.gsm8k_path=$TRAIN_FILES_GSM8K \
  ++data.math_path=$TRAIN_FILES_MATH \
  \
  data.val_files=$VAL_FILES \
  data.val_batch_size=1 \
  data.train_batch_size=$BATCH_SIZE \
  data.max_prompt_length=512 \
  data.max_response_length=128 \
  data.shuffle=False \
  \
  actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
  ++actor_rollout_ref.model.hf_config.torch_dtype=bfloat16 \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_PARALLEL \
  actor_rollout_ref.rollout.n=$N_ROLLOUTS \
  actor_rollout_ref.rollout.temperature=0.7 \
  actor_rollout_ref.rollout.top_p=0.9 \
  actor_rollout_ref.rollout.response_length=128 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.compute_entropy=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  \
  ++actor_rollout_ref.rollout.cot_augmentation.enable=true \
  ++actor_rollout_ref.rollout.cot_augmentation.use_multi_dataset=true \
  ++actor_rollout_ref.rollout.cot_augmentation.dataset_cot_mapping.gsm8k=$GSM8K_COT \
  ++actor_rollout_ref.rollout.cot_augmentation.dataset_cot_mapping.math=$MATH_COT \
  ++actor_rollout_ref.rollout.cot_augmentation.skip_on_mismatch=true \
  ++actor_rollout_ref.rollout.cot_augmentation.use_full_cot=true \
  ++actor_rollout_ref.rollout.cot_augmentation.verbose=false \
  ++actor_rollout_ref.rollout.cot_augmentation.debug_print_augmented_prompts=false \
  ++actor_rollout_ref.rollout.cot_augmentation.debug_num_samples=0 \
  ++actor_rollout_ref.rollout.cot_augmentation.debug_print_full_prompt=false \
  \
  algorithm.adv_estimator=grpo \
  algorithm.gamma=1.0 \
  algorithm.use_kl_in_reward=False \
  \
  trainer.total_epochs=1 \
  trainer.critic_warmup=0 \
  trainer.project_name=quick_debug_entropy \
  trainer.experiment_name=debug_1_step \
  trainer.logger='[console]' \
  trainer.default_local_dir=$OUTPUT_DIR \
  trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
  trainer.nnodes=1 \
  trainer.save_freq=999999 \
  trainer.test_freq=0 \
  +trainer.load_checkpoint=False \
  ++trainer.max_steps=$MAX_STEPS 2>&1 | tee ${LOG_FILE}

echo "================================"
echo "✅ 调试完成"
echo "================================"
echo "如果看到 DEBUG 输出，说明 vLLM 熵计算正常工作！"
echo "如果没有看到 DEBUG 输出，请检查："
echo "  1. vLLM 文件是否正确更新"
echo "  2. Python 缓存是否清理"
echo "  3. Ray workers 是否重启"
echo "================================"

