


# #!/bin/bash
# # 🔥 版本 2: 训练 MATH + DAPO 两个数据集
# # 特点：MATH 使用 COT 增强，DAPO 自动跳过（skip_on_mismatch=true）

# # 🆕 必须在最开始设置 Ray 环境变量（在 set -x 之前）
# export RAY_ENABLE_DASHBOARD=0
# export RAY_DASHBOARD_HOST=""

# # 让ray的报错出现
# export HYDRA_FULL_ERROR=1

# set -x

# # 配置
# export TRANSFORMERS_OFFLINE=0
# export HF_DATASETS_OFFLINE=0
# export HYDRA_FULL_ERROR=1

# # 模型路径
# ACTOR_MODEL_PATH="/nas/models/Qwen3-8B"

# # GPU配置
# N_GPUS_PER_NODE=4
# export CUDA_VISIBLE_DEVICES="0,1,2,3"
# TENSOR_PARALLEL=1
# N_ROLLOUTS=2              # ✅ 每个 prompt 生成 4 个响应（GRPO 推荐最小值）
# BATCH_SIZE=4              # ✅ 4 个问题
# ppo_mini_batch_size=4     # ✅ mini batch size
# batch_size_per_gpu=1      # ✅ 每个 GPU 的微批次（用于梯度累积）
# EPOCHS=3

# # 📊 计算验证:
# # - 总样本数 = BATCH_SIZE × N_ROLLOUTS = 4 × 4 = 16
# # - 每个 GPU = 16 / 4 = 4 个样本
# # - CVAE 分叉后: 4 × (1 + 3) = 16 条路径 per GPU

# # 项目路径
# export PYTHONPATH="/nas/dhl/verl:$PYTHONPATH"

# # ✅ 使用所有三个数据集
# #TRAIN_FILES_GSM8K="/nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet"
# TRAIN_FILES_MATH="/nas/dhl/Datasets/my_Datasets/MATH/train.parquet"
# TRAIN_FILES_DAPO="/nas/dhl/Datasets/my_Datasets/dapo-math-17k.parquet"
# VAL_FILES=""

# # ✅ 配置 GSM8K 和 MATH 的 COT 文件（DAPO 没有，会自动跳过）
# #GSM8K_COT="/nas/dhl/CVAE/Datasets/GSM8K/processed/train_k_shot_GSM8K.jsonl"
# MATH_COT="/nas/dhl/CVAE/Datasets/MATH/processed/train_k_shot_MATH.jsonl"

# # 🔥 CVAE 分叉配置
# ENABLE_CVAE_BRANCHING=True
# CVAE_BRANCHING_MODE="psa"
# CVAE_NUM_BRANCHES=3
# CVAE_INJECTION_LAYERS=4
# CVAE_LATENT_DIM=128
# CVAE_EMBEDDING_DIM=1536
# top_k=-1  # 0 for HF rollout, -1 for vLLM rollout

# # CVAE 模型路径
# CVAE_MODEL_PATH="/nas/dhl/CVAE/models/GSM8K-MATH-trained/lars_selector_GSM8K-MATH.pth"
# CVAE_EMBEDDING_MODEL_PATH="/nas/dhl/CVAE/models/deberta-v2-xlarge"

# # 输出目录
# OUTPUT_DIR="/nas/dhl/outputs/train_math_dapo_$(date +%Y%m%d_%H%M%S)"
# mkdir -p "${OUTPUT_DIR}"
# LOG_FILE="${OUTPUT_DIR}/training.log"

# # Rollout 调试目录（用于验证 reward 计算正确性）
# # 🎯 会保存每个 step 的前5个样本到 JSONL 文件
# # 格式: {step}.jsonl，包含 input, output, ground_truth, score, data_source
# ROLLOUT_DEBUG_DIR="${OUTPUT_DIR}/rollout_debug"
# mkdir -p "${ROLLOUT_DEBUG_DIR}"

# # 熵输出配置
# ENTROPY_OUTPUT_DIR="${OUTPUT_DIR}/Entropy_out"
# ENTROPY_TOP_K=10
# ENTROPY_SAVE_INTERVAL=10
# ENTROPY_ENABLED=true

# # wandb设置
# export WANDB_MODE="offline"
# export WANDB_DIR="${OUTPUT_DIR}/wandb"
# export WANDB_SAVE_DIR="${OUTPUT_DIR}/wandb"
# mkdir -p "${WANDB_DIR}"

# # 清理 vLLM 缓存
# # find /opt/dhl_conda_envs/verl_fast/lib/python3.10/site-packages/vllm -name "*.pyc" -delete
# # find /opt/dhl_conda_envs/verl_fast/lib/python3.10/site-packages/vllm -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# # 设置日志级别
# export VLLM_LOGGING_LEVEL=INFO
# export VERL_LOGGING_LEVEL=INFO

# echo "================================"
# echo "🎯 训练配置: MATH + DAPO"
# echo "================================"
# echo "数据集 1: MATH"
# echo "  数据: $TRAIN_FILES_MATH"
# echo "  COT: $MATH_COT ✅"
# echo ""
# echo "数据集 2: DAPO"
# echo "  数据: $TRAIN_FILES_DAPO"
# echo "  COT: 无（自动跳过）⚠️"
# echo ""
# echo "输出目录: $OUTPUT_DIR"
# echo ""
# echo "📊 Reward 验证 (自动启用):"
# echo "  输出目录: $ROLLOUT_DEBUG_DIR"
# echo "  ├─ num_examine: 3 (控制台打印)"
# echo "  ├─ num_examine_rollout: 5 (保存到JSONL)"
# echo "  ├─ 文件: {step}.jsonl"
# echo "  └─ 字段: input, output, ground_truth, score, data_source"
# echo "================================"
# echo ""

# #++data.gsm8k_path=$TRAIN_FILES_GSM8K \

# # 🔥 启动训练
# python3 -m verl.trainer.main_ppo \
#   ++data.custom_cls.path=pkg://examples.grpo_trainer.multi_dataset_with_cot \
#   ++data.custom_cls.name=MultiDatasetWithCOT \
#   ++data.math_path=$TRAIN_FILES_MATH \
#   ++data.dapo_path=$TRAIN_FILES_DAPO \
#   ++data.dapo_prompt_key=prompt \
#   ++data.dapo_answer_key=reward_model.ground_truth \
#   \
#   data.val_files=$VAL_FILES \
#   data.val_batch_size=1 \
#   data.train_batch_size=$BATCH_SIZE \
#   actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
#   actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$batch_size_per_gpu \
#   data.max_prompt_length=2048 \
#   data.max_response_length=8192 \
#   actor_rollout_ref.rollout.response_length=8192 \
#   actor_rollout_ref.rollout.max_num_batched_tokens=12288 \
#   data.shuffle=False \
#   \
#   actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
#   ++actor_rollout_ref.model.hf_config.torch_dtype=bfloat16 \
#   actor_rollout_ref.actor.strategy=fsdp \
#   actor_rollout_ref.actor.optim.lr=1e-6 \
#   actor_rollout_ref.actor.use_kl_loss=False \
#   actor_rollout_ref.actor.entropy_coeff=0 \
#   actor_rollout_ref.model.enable_gradient_checkpointing=False \
#   actor_rollout_ref.actor.fsdp_config.param_offload=False \
#   actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
#   actor_rollout_ref.rollout.name=vllm \
#   actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
#   actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_PARALLEL \
#   actor_rollout_ref.rollout.n=$N_ROLLOUTS \
#   actor_rollout_ref.actor.kl_loss_coef=0.0 \
#   actor_rollout_ref.rollout.temperature=0.9 \
#   actor_rollout_ref.rollout.top_p=0.9 \
#   actor_rollout_ref.rollout.top_k="${top_k}" \
#   actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
#   actor_rollout_ref.rollout.compute_entropy=True \
#   actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
#   actor_rollout_ref.ref.fsdp_config.param_offload=True \
#   \
#   +actor_rollout_ref.rollout.enable_cvae_branching=${ENABLE_CVAE_BRANCHING} \
#   +actor_rollout_ref.rollout.cvae_num_branches_per_path=${CVAE_NUM_BRANCHES} \
#   +actor_rollout_ref.rollout.cvae_branching_mode=${CVAE_BRANCHING_MODE} \
#   +actor_rollout_ref.rollout.cvae_model_path="${CVAE_MODEL_PATH}" \
#   +actor_rollout_ref.rollout.cvae_embedding_model_path="${CVAE_EMBEDDING_MODEL_PATH}" \
#   +actor_rollout_ref.rollout.cvae_injection_layers=${CVAE_INJECTION_LAYERS} \
#   +actor_rollout_ref.rollout.cvae_latent_dim=${CVAE_LATENT_DIM} \
#   +actor_rollout_ref.rollout.cvae_embedding_dim=${CVAE_EMBEDDING_DIM} \
#   \
#   ++actor_rollout_ref.rollout.entropy_output.enabled=${ENTROPY_ENABLED} \
#   ++actor_rollout_ref.rollout.entropy_output.output_dir="${ENTROPY_OUTPUT_DIR}" \
#   ++actor_rollout_ref.rollout.entropy_output.top_k=${ENTROPY_TOP_K} \
#   ++actor_rollout_ref.rollout.entropy_output.save_interval=${ENTROPY_SAVE_INTERVAL} \
#   ++actor_rollout_ref.rollout.entropy_output.mark_style="both" \
#   ++actor_rollout_ref.rollout.entropy_output.token_entropy_to_jsonl=true \
#   \
#   ++actor_rollout_ref.rollout.cot_augmentation.enable=true \
#   ++actor_rollout_ref.rollout.cot_augmentation.use_multi_dataset=true \
#   ++actor_rollout_ref.rollout.cot_augmentation.dataset_cot_mapping.math=$MATH_COT \
#   ++actor_rollout_ref.rollout.cot_augmentation.skip_on_mismatch=true \
#   ++actor_rollout_ref.rollout.cot_augmentation.use_full_cot=true \
#   ++actor_rollout_ref.rollout.cot_augmentation.verbose=false \
#   ++actor_rollout_ref.rollout.cot_augmentation.debug_print_augmented_prompts=false \
#   ++actor_rollout_ref.rollout.cot_augmentation.debug_num_samples=0 \
#   ++actor_rollout_ref.rollout.cot_augmentation.debug_print_full_prompt=false \
#   \
#   algorithm.adv_estimator=grpo \
#   algorithm.gamma=1.0 \
#   algorithm.use_kl_in_reward=True \
#   \
#   ++reward_model.num_examine=3 \
#   ++reward_model.num_examine_rollout=5 \
#   ++trainer.rollout_data_dir=$ROLLOUT_DEBUG_DIR \
#   \
#   trainer.total_epochs=$EPOCHS \
#   trainer.critic_warmup=0 \
#   trainer.project_name=train_math_dapo \
#   trainer.experiment_name=math_dapo \
#   trainer.logger='[console,wandb]' \
#   trainer.default_local_dir=$OUTPUT_DIR \
#   trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
#   trainer.nnodes=1 \
#   trainer.save_freq=100 \
#   trainer.test_freq=0 \
#   +trainer.load_checkpoint=False 2>&1 | tee ${LOG_FILE}

# echo "================================"
# echo "✅ 训练完成 (MATH + DAPO)"
# echo "================================"


#!/bin/bash
# 🔥 版本 2: 训练 MATH + DAPO 两个数据集
# 特点：MATH 使用 COT 增强，DAPO 自动跳过（skip_on_mismatch=true）

# 清理 vLLM
# VLLM_PATH="/opt/dhl_conda_envs/verl_fast/lib/python3.10/site-packages/vllm"
# if [ -d "$VLLM_PATH" ]; then
#     echo "    清理 vLLM: $VLLM_PATH"
#     find "$VLLM_PATH" -name "*.pyc" -delete 2>/dev/null || true
#     find "$VLLM_PATH" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
# fi

# # 5. 重新编译
# # echo "  5/5: 重新编译..."
# python3 -m compileall -q -f "$CURRENT_PROJECT" 2>/dev/null || true
# python3 -m compileall -q -f "$VLLM_PATH" 2>/dev/null || true


# # 额外保险：删掉 ray 自己的缓存
# rm -rf /tmp/ray/session_latest
# rm -rf /tmp/ray/worker*

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
#ACTOR_MODEL_PATH="/nas/models/Qwen3-8B"
#ACTOR_MODEL_PATH="/nas/models/Qwen2.5-7B"
ACTOR_MODEL_PATH="/opt/nvidia/Qwen2.5-7B"

# GPU配置
N_GPUS_PER_NODE=8
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
TENSOR_PARALLEL=1
N_ROLLOUTS=2              # ✅ 每个 prompt 生成 4 个响应（GRPO 推荐最小值）
BATCH_SIZE=32             # ✅ 4 个问题
ppo_mini_batch_size=32     # ✅ mini batch size
batch_size_per_gpu=2      # ✅ 每个 GPU 的微批次（用于梯度累积）
EPOCHS=2

# 📊 计算验证:
# - 总样本数 = BATCH_SIZE × N_ROLLOUTS = 4 × 4 = 16
# - 每个 GPU = 16 / 4 = 4 个样本
# - CVAE 分叉后: 4 × (1 + 3) = 16 条路径 per GPU

# 项目路径
export PYTHONPATH="/nas/dhl/verl:$PYTHONPATH"

# ✅ 使用所有三个数据集
#TRAIN_FILES_GSM8K="/nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet"
# TRAIN_FILES_MATH="/nas/dhl/Datasets/my_Datasets/MATH/train.parquet"
# TRAIN_FILES_DAPO="/nas/dhl/Datasets/my_Datasets/dapo-math-17k-deduplicated-cleaned.parquet"

TRAIN_FILES_DAPO="/nas/dhl/Datasets/filtered/final_parquet/dapo-final-filtered.parquet"
TRAIN_FILES_MATH="/nas/dhl/Datasets/filtered/final_parquet/MATH-train-final-filtered.parquet"


VAL_FILES=""

# ✅ 配置 GSM8K 和 MATH 的 COT 文件（DAPO 没有，会自动跳过）
#GSM8K_COT="/nas/dhl/CVAE/Datasets/GSM8K/processed/train_k_shot_GSM8K.jsonl"
MATH_COT="/nas/dhl/CVAE/Datasets/MATH/processed/train_k_shot_MATH.jsonl"

# 🔥 CVAE 分叉配置
ENABLE_CVAE_BRANCHING=True
CVAE_BRANCHING_MODE="psa"
CVAE_NUM_BRANCHES=3
CVAE_INJECTION_LAYERS=4
CVAE_LATENT_DIM=128
CVAE_EMBEDDING_DIM=1536
top_k=-1  # 0 for HF rollout, -1 for vLLM rollout

# CVAE 模型路径
CVAE_MODEL_PATH="/nas/dhl/CVAE/models/GSM8K-MATH-trained/lars_selector_GSM8K-MATH.pth"
CVAE_EMBEDDING_MODEL_PATH="/nas/dhl/CVAE/models/deberta-v2-xlarge"

# 🆕 CVAE 投影层训练配置
CVAE_PROJECTION_LR=5e-7           # 投影层学习率
MAX_CVAE_CKPT_TO_KEEP=3           # CVAE checkpoint 保留数量（与 actor 一致）

# 输出目录
OUTPUT_DIR="/nas/dhl/outputs/qwen2.5_7b_train_math_dapo_ib_cvaemlp_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/training.log"

# Rollout 调试目录（用于验证 reward 计算正确性）
# 🎯 会保存每个 step 的前5个样本到 JSONL 文件
# 格式: {step}.jsonl，包含 input, output, ground_truth, score, data_source
ROLLOUT_DEBUG_DIR="${OUTPUT_DIR}/rollout_debug"
mkdir -p "${ROLLOUT_DEBUG_DIR}"

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
# find /opt/dhl_conda_envs/verl_fast/lib/python3.10/site-packages/vllm -name "*.pyc" -delete
# find /opt/dhl_conda_envs/verl_fast/lib/python3.10/site-packages/vllm -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# 设置日志级别
export VLLM_LOGGING_LEVEL=INFO
export VERL_LOGGING_LEVEL=INFO

echo "================================"
echo "🎯 训练配置: MATH + DAPO"
echo "================================"
echo "数据集 1: MATH"
echo "  数据: $TRAIN_FILES_MATH"
echo "  COT: $MATH_COT ✅"
echo ""
echo "数据集 2: DAPO"
echo "  数据: $TRAIN_FILES_DAPO"
echo "  COT: 无（自动跳过）⚠️"
echo ""
echo "输出目录: $OUTPUT_DIR"
echo ""
echo "🌿 CVAE 配置:"
echo "  启用: ${ENABLE_CVAE_BRANCHING}"
echo "  分叉模式: ${CVAE_BRANCHING_MODE}"
echo "  分叉次数: ${CVAE_NUM_BRANCHES}"
echo "  注入层数: ${CVAE_INJECTION_LAYERS}"
echo "  投影层学习率: ${CVAE_PROJECTION_LR} ✅"
echo "  保存频率: 与 Actor 一致 (每 100 步) ✅"
echo "  最多保留: ${MAX_CVAE_CKPT_TO_KEEP} 个 checkpoint ✅"
echo ""
echo "📊 Reward 验证 (自动启用):"
echo "  输出目录: $ROLLOUT_DEBUG_DIR"
echo "  ├─ num_examine: 3 (控制台打印)"
echo "  ├─ num_examine_rollout: 5 (保存到JSONL)"
echo "  ├─ 文件: {step}.jsonl"
echo "  └─ 字段: input, output, ground_truth, score, data_source"
echo ""
echo "🔍 验证目录存在:"
if [ -d "$ROLLOUT_DEBUG_DIR" ]; then
    echo "  ✅ $ROLLOUT_DEBUG_DIR 已创建"
else
    echo "  ❌ $ROLLOUT_DEBUG_DIR 不存在（将在训练时创建）"
fi
echo "================================"
echo ""

#++data.gsm8k_path=$TRAIN_FILES_GSM8K \

# 🔥 启动训练
python3 -m verl.trainer.main_ppo \
  ++actor_rollout_ref.actor.use_ib_regularization=True \
  actor_rollout_ref.actor.entropy_coeff=0.005 \
  ++data.custom_cls.path=pkg://examples.grpo_trainer.multi_dataset_with_cot \
  ++data.custom_cls.name=MultiDatasetWithCOT \
  ++data.math_path=$TRAIN_FILES_MATH \
  ++data.dapo_path=$TRAIN_FILES_DAPO \
  ++data.dapo_prompt_key=prompt \
  ++data.dapo_answer_key=reward_model.ground_truth \
  \
  data.val_files=$VAL_FILES \
  data.val_batch_size=1 \
  data.train_batch_size=$BATCH_SIZE \
  actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$batch_size_per_gpu \
  data.max_prompt_length=2048 \
  data.max_response_length=4096 \
  actor_rollout_ref.rollout.response_length=4096 \
  actor_rollout_ref.rollout.max_num_batched_tokens=10088 \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
  data.shuffle=True \
  \
  actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
  ++actor_rollout_ref.model.hf_config.torch_dtype=bfloat16 \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  ++actor_rollout_ref.actor.cvae_projection_lr=${CVAE_PROJECTION_LR} \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_PARALLEL \
  actor_rollout_ref.rollout.n=$N_ROLLOUTS \
  actor_rollout_ref.rollout.temperature=0.9 \
  actor_rollout_ref.rollout.top_p=0.9 \
  actor_rollout_ref.rollout.top_k="${top_k}" \
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
  algorithm.use_kl_in_reward=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  \
  ++reward_model.num_examine=3 \
  ++reward_model.num_examine_rollout=5 \
  ++trainer.rollout_data_dir=$ROLLOUT_DEBUG_DIR \
  \
  trainer.total_epochs=$EPOCHS \
  trainer.critic_warmup=0 \
  trainer.project_name=train_math_dapo \
  trainer.experiment_name=math_dapo \
  trainer.logger='[console,wandb]' \
  trainer.default_local_dir=$OUTPUT_DIR \
  trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
  trainer.nnodes=1 \
  trainer.save_freq=32 \
  trainer.test_freq=0 \
  ++trainer.max_cvae_ckpt_to_keep=${MAX_CVAE_CKPT_TO_KEEP} \
  ++trainer.load_checkpoint=False 2>&1 | tee ${LOG_FILE}

echo "================================"
echo "✅ 训练完成 (MATH + DAPO)"
echo "================================"

