#!/bin/bash
# Branching 可视化示例脚本

echo "==================================="
echo "Branching 可视化工具示例"
echo "==================================="
echo ""


# 示例2：可视化 epoch_0 的 branching.jsonl

 
python3 /nas/dhl/verl/examples/grpo_trainer/branching_vis/visualize_branching.py \
        /nas/dhl/outputs/qwen2.5_7b_train_math_dapo_ib_cvaemlp_20251123_155428/Entropy_out/epoch_0/branching.jsonl
echo "==================================="
echo "✅ 完成！"
echo "==================================="

