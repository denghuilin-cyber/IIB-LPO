#!/bin/bash
# 🚀 快速调试脚本：只运行 1 个 step，立即看到 DEBUG 输出
# 用于验证 vLLM 熵计算是否正常工作

# 🆕 必须在最开始设置 Ray 环境变量（在 set -x 之前）
export RAY_ENABLE_DASHBOARD=0
export RAY_DASHBOARD_HOST=""

# 让ray的报错出现
export HYDRA_FULL_ERROR=1

set -x

# 配置
export TRANSFORMERS_OFFLINE=0
export HF_DATASETS_OFFLINE=0
# 让ray的报错出现
export HYDRA_FULL_ERROR=1

# 🔥 使用最小配置加快调试
#ACTOR_MODEL_PATH="/nas/models/qwen2.5-math-1.5B_instruct"
#ACTOR_MODEL_PATH="/nas/models/Qwen3-8B"
ACTOR_MODEL_PATH="/nas/models/Qwen3-4B-Instruct-2507"



# GPU配置
N_GPUS_PER_NODE=1
export CUDA_VISIBLE_DEVICES="4"
TENSOR_PARALLEL=1
N_ROLLOUTS=1  # 🔥 只生成 1 个 rollout
BATCH_SIZE=1  # ⚠️ 真实训练 batch size 需 ≥ 最小值 8
ppo_mini_batch_size=1
batch_size_per_gpu=1
EPOCHS=3



# 项目路径
export PYTHONPATH="/nas/dhl/verl:$PYTHONPATH"

# 数据文件
TRAIN_FILES_GSM8K="/nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet"
TRAIN_FILES_MATH="/nas/dhl/Datasets/my_Datasets/MATH/train.parquet"
VAL_FILES=""

# COT文件路径
GSM8K_COT="/nas/dhl/CVAE/Datasets/GSM8K/processed/train_k_shot_GSM8K.jsonl"
MATH_COT="/nas/dhl/CVAE/Datasets/MATH/processed/train_k_shot_MATH.jsonl"


# 🔥 CVAE 分叉配置
ENABLE_CVAE_BRANCHING=True   # 是否启用 CVAE 分叉
CVAE_BRANCHING_MODE="psa"  # 分叉模式: random, input, psa, softmax
CVAE_NUM_BRANCHES=3           # 每个路径分叉次数
CVAE_INJECTION_LAYERS=4       # 注入层数：4（最后4层）, 8（最后8层）, "all"（所有层）
CVAE_LATENT_DIM=128           # 潜在向量维度（CVAE 的 z 向量维度）
CVAE_EMBEDDING_DIM=1536       # 嵌入向量维度（DeBERTa-v2-xlarge 的输出维度）
CVAE_MODEL_PATH="/nas/dhl/CVAE/models/GSM8K-MATH-trained/lars_selector_GSM8K-MATH.pth"
CVAE_EMBEDDING_MODEL_PATH="/nas/dhl/CVAE/models/deberta-v2-xlarge"

# 输出目录
OUTPUT_DIR="/nas/dhl/outputs/quick_debug_entropy_${CVAE_BRANCHING_MODE}_${CVAE_NUM_BRANCHES}_${CVAE_INJECTION_LAYERS}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/training_$(date +%Y%m%d_%H%M%S).log"


# 🆕 熵输出配置
ENTROPY_OUTPUT_DIR="${OUTPUT_DIR}/Entropy_out"
ENTROPY_TOP_K=10              # 标记熵值最高的 K 个 token
ENTROPY_SAVE_INTERVAL=10       # 🔥 每 1 个 step 保存一次（调试用）
ENTROPY_ENABLED=true          # 是否启用熵输出到 JSONL

# wandb设置
export WANDB_MODE="offline"
export WANDB_DIR="${OUTPUT_DIR}/wandb"
export WANDB_SAVE_DIR="${OUTPUT_DIR}/wandb"
mkdir -p "${WANDB_DIR}"

export RAY_ENABLE_DASHBOARD=0
export RAY_DASHBOARD_HOST=""

# 🔥 项目路径（确保使用当前项目，不是旧版本）
export PYTHONPATH="/nas/dhl/verl:$PYTHONPATH"

# 🔥 验证路径
echo "================================"
echo "🔍 验证项目路径"
echo "================================"
echo "PYTHONPATH: $PYTHONPATH"
echo "当前目录: $(pwd)"
echo "verl 模块位置: $(python3 -c 'import verl; print(verl.__file__)' 2>/dev/null || echo '未找到')"
echo "================================"
echo ""

#______________________________
# # 2. 清理 vLLM 的所有 .pyc 缓存
# find /opt/dhl_conda_envs/verl_fast/lib/python3.10/site-packages/vllm -name "*.pyc" -delete
# find /opt/dhl_conda_envs/verl_fast/lib/python3.10/site-packages/vllm -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# 3. 强制重新编译（可选，但推荐）
# python3 -m compileall -f /opt/dhl_conda_envs/verl_fast/lib/python3.10/site-packages/vllm
#________________________________

# # 🆕 清理环境
# echo "================================"
# echo "🧹 清理环境"
# echo "================================"
# ray stop --force 2>/dev/null || true
# sleep 2
# rm -rf /tmp/ray/* 2>/dev/null || true

# # 清理 vLLM 缓存
# VLLM_PATH="/opt/dhl_conda_envs/verl_fast/lib/python3.10/site-packages/vllm"
# if [ -d "$VLLM_PATH" ]; then
#     echo "  清理 vLLM 缓存..."
#     find "$VLLM_PATH" -name "*.pyc" -delete 2>/dev/null || true
#     find "$VLLM_PATH" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
#     python3 -m compileall -q -f "$VLLM_PATH" 2>/dev/null || true
#     echo "  ✅ vLLM 缓存已清理"
# fi

# echo "✅ 环境清理完成"


# 🆕 完全清理环境（解决 Ray 使用旧代码路径的问题）
# echo "================================"
# echo "🧹 完全清理环境"
# echo "================================"

# # 1. 强制停止 Ray
# echo "  1/5: 停止 Ray..."
# ray stop --force 2>/dev/null || true
# sleep 3

# # 2. 杀死残留进程
# echo "  2/5: 杀死残留进程..."
# pkill -9 -f "ray::" 2>/dev/null || true
# pkill -9 -f "raylet" 2>/dev/null || true
# pkill -9 -f "gcs_server" 2>/dev/null || true
# sleep 2

# # 3. 清理 Ray 临时文件
# echo "  3/5: 清理 Ray 临时文件..."
# rm -rf /tmp/ray/* 2>/dev/null || true
# rm -rf /tmp/ray_* 2>/dev/null || true
# rm -rf ~/.ray/* 2>/dev/null || true

# # 4. 清理 Python 缓存（关键！）
# echo "  4/5: 清理 Python 缓存..."

# # 清理当前项目
# CURRENT_PROJECT="/nas/dhl/verl"
# if [ -d "$CURRENT_PROJECT" ]; then
#     echo "    清理当前项目: $CURRENT_PROJECT"
#     find "$CURRENT_PROJECT" -name "*.pyc" -delete 2>/dev/null || true
#     find "$CURRENT_PROJECT" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
# fi

# 清理旧项目（避免冲突）
# OLD_PROJECT="/nas/dhl/verl_success_entropy_token_vis_11_10改verl能跑"
# if [ -d "$OLD_PROJECT" ]; then
#     echo "    清理旧项目: $OLD_PROJECT"
#     find "$OLD_PROJECT" -name "*.pyc" -delete 2>/dev/null || true
#     find "$OLD_PROJECT" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
# fi

# # 清理 vLLM
# VLLM_PATH="/opt/dhl_conda_envs/verl_fast/lib/python3.10/site-packages/vllm"
# if [ -d "$VLLM_PATH" ]; then
#     echo "    清理 vLLM: $VLLM_PATH"
#     find "$VLLM_PATH" -name "*.pyc" -delete 2>/dev/null || true
#     find "$VLLM_PATH" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
# fi

# 5. 重新编译
# echo "  5/5: 重新编译..."
# python3 -m compileall -q -f "$CURRENT_PROJECT" 2>/dev/null || true
# python3 -m compileall -q -f "$VLLM_PATH" 2>/dev/null || true

# 清理旧项目（避免被导入）
# find /nas/dhl/verl_success_entropy_token_vis_11_10改verl能跑 -name "*.pyc" -delete 2>/dev/null || true
# find /nas/dhl/verl_success_entropy_token_vis_11_10改verl能跑 -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# # 6. 验证 verl 模块位置
# export PYTHONPATH="/nas/dhl/verl:$PYTHONPATH"
# python3 -c "import sys; print('PYTHONPATH:', sys.path[:3])"
# python3 -c "import verl; print('verl location:', verl.__file__)" 2>/dev/null || echo "verl 未安装（正常）"

# echo "✅ 环境清理完成"
# echo ""

unset PYTHONPATH
export PYTHONPATH="/nas/dhl/verl"

# 设置日志级别为 DEBUG
export VLLM_LOGGING_LEVEL=INFO
export VERL_LOGGING_LEVEL=INFO
# 如果 不想输出token-level的熵值 改成INFO即可
echo "VLLM_LOGGING_LEVEL: $VLLM_LOGGING_LEVEL"
echo "VERL_LOGGING_LEVEL: $VERL_LOGGING_LEVEL"


# 🔥 验证路径
echo "================================"
echo "🔍 验证项目路径"
echo "================================"
echo "PYTHONPATH: $PYTHONPATH"
echo "当前目录: $(pwd)"
echo "verl 模块位置: $(python3 -c 'import verl; print(verl.__file__)' 2>/dev/null || echo '未找到')"
echo "================================"
echo ""

'''
data.max_response_length=2048  这个是sft训练用的参数 过长数据会被直接过滤
只有 actor_rollout_ref.rollout.response_length=2048 这个参数和长度相关
长度 至少要 4096，最好 8196因为长度很影响性能， 8196的话，最大batchsize是8
qwen3是长文本训练过的，最好考虑qwen2.5的模型

'''



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
  actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$batch_size_per_gpu \
  data.max_prompt_length=2048 \
  data.max_response_length=8192 \
  actor_rollout_ref.rollout.response_length=8192 \
  actor_rollout_ref.rollout.max_num_batched_tokens=10240 \
  data.shuffle=False \
  \
  actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
  ++actor_rollout_ref.model.hf_config.torch_dtype=bfloat16 \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
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
  algorithm.use_kl_in_reward=True \
  \
  trainer.total_epochs=$EPOCHS \
  trainer.critic_warmup=0 \
  trainer.project_name=quick_debug_entropy \
  trainer.experiment_name=debug_1_step \
  trainer.logger='[console,wandb]' \
  trainer.default_local_dir=$OUTPUT_DIR \
  trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
  trainer.nnodes=1 \
  trainer.save_freq=100 \
  trainer.test_freq=0 \
  +trainer.load_checkpoint=False 2>&1 | tee ${LOG_FILE}

echo "================================"
echo "✅ 调试完成"
echo "================================"
echo "如果看到 DEBUG 输出，说明 vLLM 熵计算正常工作！"
echo "如果没有看到 DEBUG 输出，请检查："
echo "  1. vLLM 文件是否正确更新"
echo "  2. Python 缓存是否清理"
echo "  3. Ray workers 是否重启"
echo "================================"
echo ""
echo "🎯 熵输出文件位置："
echo "  ${ENTROPY_OUTPUT_DIR}/"
echo "  ├── epoch_0/"
echo "  │   ├── gsm8k.jsonl"
echo "  │   └── math.jsonl"
echo "================================"


#   actor_rollout_ref.model.enable_gradient_checkpointing=True \ 为了方便恢复训练