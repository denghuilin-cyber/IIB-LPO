#!/bin/bash
# 快速测试：验证多数据集COT增强 + 熵计算
# 在推理阶段计算并输出每个token的熵值
# 🆕 必须在最开始设置 Ray 环境变量（在 set -x 之前）
export RAY_ENABLE_DASHBOARD=0
export RAY_DASHBOARD_HOST=""


# 🆕 清理 Ray 环境和 Python 缓存（避免残留导致初始化失败）
echo "================================"
echo "🧹 清理 Ray 环境和 Python 缓存"
echo "================================"

# 1. 停止 Ray
echo "  停止 Ray..."
ray stop --force 2>/dev/null || true
sleep 2

# 2. 清理 Ray 临时文件
echo "  清理 Ray 临时文件..."
rm -rf /tmp/ray/* 2>/dev/null || true

# 3. 清理 vLLM Python 缓存（关键！）
echo "  清理 vLLM Python 缓存..."
VLLM_PATH="/opt/dhl_conda_envs/verl_fast/lib/python3.10/site-packages/vllm"
if [ -d "$VLLM_PATH" ]; then
    # 删除所有 .pyc 文件
    find "$VLLM_PATH" -name "*.pyc" -delete 2>/dev/null || true
    # 删除所有 __pycache__ 目录
    find "$VLLM_PATH" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    echo "  ✅ vLLM 缓存已清理"
else
    echo "  ⚠️  vLLM 路径不存在: $VLLM_PATH"
fi

# 4. 强制重新编译 vLLM（可选，但推荐）
echo "  重新编译 vLLM Python 文件..."
python3 -m compileall -q -f "$VLLM_PATH" 2>/dev/null || true

echo "✅ 环境清理完成"

# 2. 清理 vLLM 的所有 .pyc 缓存
find /opt/dhl_conda_envs/verl_fast/lib/python3.10/site-packages/vllm -name "*.pyc" -delete
find /opt/dhl_conda_envs/verl_fast/lib/python3.10/site-packages/vllm -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# 3. 强制重新编译（可选，但推荐）
python3 -m compileall -f /opt/dhl_conda_envs/verl_fast/lib/python3.10/site-packages/vllm



set -x

# '''
# 新增功能：熵计算与输出
# ================================
# 1. 在 rollout 阶段计算每个 token 的熵值
# 2. 输出格式：
#    - 控制台打印：[Rollout Entropy] Step X: Y.ZZZZ
#    - wandb/tensorboard：rollout/entropy 指标
# 3. 熵值含义：
#    - 范围：[0, log(vocab_size)] ≈ [0, 10.4]
# '''

export WANDB_MODE="offline"

# debug用小模型加载更快
#ACTOR_MODEL_PATH="/nas/models/qwen2.5-math-1.5B_instruct"
ACTOR_MODEL_PATH="/nas/models/Qwen3-8B" 
N_ROLLOUTS=4  # 减少rollout次数加快测试
BATCH_SIZE=1  # 小batch加快测试
EPOCHS=1

# 假设你的项目根目录是 /nas/dhl/verl
export PYTHONPATH="/nas/dhl/verl:$PYTHONPATH"

# 数据文件
TRAIN_FILES_GSM8K="/nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet"
TRAIN_FILES_MATH="/nas/dhl/Datasets/my_Datasets/MATH/train.parquet"
VAL_FILES=""

# COT文件路径
GSM8K_COT="/nas/dhl/CVAE/Datasets/GSM8K/processed/train_k_shot_GSM8K.jsonl"
MATH_COT="/nas/dhl/CVAE/Datasets/MATH/processed/train_k_shot_MATH.jsonl"

# GPU配置
N_GPUS_PER_NODE=8           # 使用8张GPU
TENSOR_PARALLEL=1           # Tensor Parallelism大小（1=不分割模型，最大吞吐量）

# 输出目录和日志
#OUTPUT_DIR="/nas/dhl/outputs/test_cot_entropy_output_qwen2_5_1_5B"
#OUTPUT_DIR="/nas/dhl/outputs/test_cot_entropy_output"
OUTPUT_DIR="/nas/dhl/outputs/test_entropy"
LOG_FILE="${OUTPUT_DIR}/training_$(date +%Y%m%d_%H%M%S).log"

# 🆕 设置 wandb 目录
export WANDB_DIR="${OUTPUT_DIR}/wandb"
mkdir -p "${WANDB_DIR}"

echo "================================"
echo "🚀 开始训练（带熵计算）"
echo "================================"
echo "输出目录: ${OUTPUT_DIR}"
echo "日志文件: ${LOG_FILE}"
echo "Batch配置: batch_size=${BATCH_SIZE}, rollouts=${N_ROLLOUTS}, total_responses=$((BATCH_SIZE * N_ROLLOUTS))"
echo "⚠️  注意：需要先修改 vLLM 源码以支持熵计算（见 VLLM_ENTROPY_MODIFICATIONS.md）"
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
  ++actor_rollout_ref.model.hf_config.torch_dtype=bfloat16 \
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
  actor_rollout_ref.rollout.compute_entropy=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
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
  trainer.total_epochs=$EPOCHS \
  trainer.critic_warmup=0 \
  trainer.project_name=test_cot_entropy \
  trainer.experiment_name=test_multi_dataset_cot_with_entropy \
  trainer.logger='[console]' \
  trainer.default_local_dir=$OUTPUT_DIR \
  trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
  trainer.nnodes=1 \
  trainer.save_freq=1000 \
  trainer.test_freq=0 \
  trainer.load_checkpoint=False \
  trainer.ckpt_path=null 2>&1 | tee ${LOG_FILE}

echo "================================"
echo "✅ 训练完成"
echo "================================"
echo "查看日志: tail -f ${LOG_FILE}"
echo "查看熵值: grep 'Rollout Entropy' ${LOG_FILE}"
echo "================================"

# 快速查看熵值输出示例
echo ""
echo "📊 熵值输出示例："
echo "================================"
grep "Rollout Entropy" ${LOG_FILE} | head -20 || echo "（训练尚未开始或日志文件不存在）"


