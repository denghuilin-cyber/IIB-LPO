#!/bin/bash
# 🔥 版本 1: 只训练 MATH 数据集
# 特点：使用 COT 增强

# 🆕 必须在最开始设置 Ray 环境变量（在 set -x 之前）
export RAY_ENABLE_DASHBOARD=0
export RAY_DASHBOARD_HOST=""


# 让ray的报错出现
export HYDRA_FULL_ERROR=1

set -x

# 配置
export TRANSFORMERS_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export HYDRA_FULL_ERROR=1

# 模型路径
ACTOR_MODEL_PATH="/nas/models/Qwen3-8B"
EPOCHS=3

# GPU配置
N_GPUS_PER_NODE=4
export CUDA_VISIBLE_DEVICES="0,1,2,3"
TENSOR_PARALLEL=1
N_ROLLOUTS=2

# --- 这里是修正点 ---
BATCH_SIZE=4
ppo_mini_batch_size=4

# 🆕 新增定义这个缺失的变量
# 因为 Global Batch=4, GPU=4, 所以每张卡分到 1
batch_size_per_gpu=1

# 项目路径
export PYTHONPATH="/nas/dhl/verl:$PYTHONPATH"

# ✅ 只使用 MATH 数据集
TRAIN_FILES_MATH="/nas/dhl/Datasets/my_Datasets/MATH/train.parquet"
VAL_FILES=""

# ✅ 只配置 MATH 的 COT 文件
MATH_COT="/nas/dhl/CVAE/Datasets/MATH/processed/train_k_shot_MATH.jsonl"

# 🔥 CVAE 分叉配置
ENABLE_CVAE_BRANCHING=True
CVAE_BRANCHING_MODE="psa"
CVAE_NUM_BRANCHES=3
CVAE_INJECTION_LAYERS=4
CVAE_LATENT_DIM=128
CVAE_EMBEDDING_DIM=1536

# CVAE 模型路径
CVAE_MODEL_PATH="/nas/dhl/CVAE/models/GSM8K-MATH-trained/lars_selector_GSM8K-MATH.pth"
CVAE_EMBEDDING_MODEL_PATH="/nas/dhl/CVAE/models/deberta-v2-xlarge"

# 输出目录
OUTPUT_DIR="/nas/dhl/outputs/train_math_only_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/training.log"

# 熵输出配置
ENTROPY_OUTPUT_DIR="${OUTPUT_DIR}/Entropy_out"
ENTROPY_TOP_K=10
ENTROPY_SAVE_INTERVAL=10
ENTROPY_ENABLED=true

# wandb设置
export WANDB_MODE="offline"
export WANDB_DIR="${OUTPUT_DIR}/wandb"
export WANDB_SAVE_DIR="${OUTPUT_DIR}/wandb"
mkdir -p "${WANDB_DIR}"

# 清理 vLLM 缓存
find /opt/dhl_conda_envs/verl_fast/lib/python3.10/site-packages/vllm -name "*.pyc" -delete
find /opt/dhl_conda_envs/verl_fast/lib/python3.10/site-packages/vllm -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# 设置日志级别
export VLLM_LOGGING_LEVEL=INFO
export VERL_LOGGING_LEVEL=INFO

echo "================================"
echo "🎯 训练配置: 只使用 MATH 数据集"
echo "================================"
echo "数据集: MATH"
echo "MATH 数据: $TRAIN_FILES_MATH"
echo "MATH COT: $MATH_COT"
echo "输出目录: $OUTPUT_DIR"
echo "================================"
echo ""

# 🔥 启动训练
python3 -m verl.trainer.main_ppo \
  ++data.custom_cls.path=pkg://examples.grpo_trainer.multi_dataset_with_cot \
  ++data.custom_cls.name=MultiDatasetWithCOT \
  ++data.math_path=$TRAIN_FILES_MATH \
  \
  data.val_files=$VAL_FILES \
  data.val_batch_size=1 \
  data.train_batch_size=$BATCH_SIZE \
  actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$batch_size_per_gpu \
  data.max_prompt_length=2048 \
  data.max_response_length=8192 \
  actor_rollout_ref.rollout.response_length=8192 \
  actor_rollout_ref.rollout.max_num_batched_tokens=10240 \
  ++reward_model.num_examine=3 \
  data.shuffle=False \
  \
  actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
  ++actor_rollout_ref.model.hf_config.torch_dtype=bfloat16 \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_PARALLEL \
  actor_rollout_ref.rollout.n=$N_ROLLOUTS \
  actor_rollout_ref.rollout.temperature=0.8 \
  actor_rollout_ref.rollout.top_p=0.9 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.compute_entropy=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  \
  +actor_rollout_ref.rollout.enable_cvae_branching=${ENABLE_CVAE_BRANCHING} \
  +actor_rollout_ref.rollout.cvae_num_branches_per_path=${CVAE_NUM_BRANCHES} \
  +actor_rollout_ref.rollout.cvae_branching_mode=${CVAE_BRANCHING_MODE} \
  +actor_rollout_ref.rollout.cvae_model_path="${CVAE_MODEL_PATH}" \
  +actor_rollout_ref.rollout.cvae_embedding_model_path="${CVAE_EMBEDDING_MODEL_PATH}" \
  +actor_rollout_ref.rollout.cvae_injection_layers=${CVAE_INJECTION_LAYERS} \
  +actor_rollout_ref.rollout.cvae_latent_dim=${CVAE_LATENT_DIM} \
  +actor_rollout_ref.rollout.cvae_embedding_dim=${CVAE_EMBEDDING_DIM} \
  \
  ++actor_rollout_ref.rollout.entropy_output.enabled=${ENTROPY_ENABLED} \
  ++actor_rollout_ref.rollout.entropy_output.output_dir="${ENTROPY_OUTPUT_DIR}" \
  ++actor_rollout_ref.rollout.entropy_output.top_k=${ENTROPY_TOP_K} \
  ++actor_rollout_ref.rollout.entropy_output.save_interval=${ENTROPY_SAVE_INTERVAL} \
  ++actor_rollout_ref.rollout.entropy_output.mark_style="both" \
  ++actor_rollout_ref.rollout.entropy_output.token_entropy_to_jsonl=true \
  \
  ++actor_rollout_ref.rollout.cot_augmentation.enable=true \
  ++actor_rollout_ref.rollout.cot_augmentation.use_multi_dataset=true \
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
  algorithm.use_kl_in_reward=True \
  \
  trainer.total_epochs=$EPOCHS \
  trainer.critic_warmup=0 \
  trainer.project_name=train_math_only \
  trainer.experiment_name=math_only \
  trainer.logger='[console,wandb]' \
  trainer.default_local_dir=$OUTPUT_DIR \
  trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
  trainer.nnodes=1 \
  trainer.save_freq=100 \
  trainer.test_freq=0 \
  +trainer.load_checkpoint=False 2>&1 | tee ${LOG_FILE}

echo "================================"
echo "✅ 训练完成 (MATH only)"
echo "================================"

