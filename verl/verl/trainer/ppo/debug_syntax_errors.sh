#!/bin/bash
# 诊断脚本：检查语法错误并清理缓存

set -e

echo "================================"
echo "🔍 开始诊断"
echo "================================"

# 1. 检查 Python 版本
echo ""
echo "1️⃣ 检查 Python 版本"
python3 --version

# 2. 检查文件是否存在
echo ""
echo "2️⃣ 检查文件是否存在"
FILES=(
    "verl/utils/entropy_output_writer.py"
    "verl/workers/config/rollout.py"
    "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py"
    "verl/trainer/ppo/ray_trainer.py"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (不存在)"
    fi
done

# 3. 运行 Python 诊断脚本
echo ""
echo "3️⃣ 运行语法检查"
python3 /nas/dhl/verl/verl/trainer/ppo/debug_syntax_errors.py

# 4. 清理 Python 缓存
echo ""
echo "4️⃣ 清理 Python 缓存"
echo "  清理 __pycache__..."
find verl -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "  清理 .pyc 文件..."
find verl -name "*.pyc" -delete 2>/dev/null || true
echo "  ✅ 缓存清理完成"

# 5. 重新编译
echo ""
echo "5️⃣ 重新编译 Python 文件"
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  编译 $file..."
        python3 -m py_compile "$file"
        if [ $? -eq 0 ]; then
            echo "    ✅ 编译成功"
        else
            echo "    ❌ 编译失败"
            exit 1
        fi
    fi
done

# 6. 检查 Ray 状态
echo ""
echo "6️⃣ 检查 Ray 状态"
ray status 2>/dev/null || echo "  ⚠️  Ray 未运行或未安装"

echo ""
echo "================================"
echo "✅ 诊断完成"
echo "================================"
echo ""
echo "如果所有检查都通过，但 Ray 仍然报错，请尝试："
echo "  1. ray stop --force"
echo "  2. rm -rf /tmp/ray/*"
echo "  3. 重新运行训练脚本"

