#!/bin/bash

echo "=========================================="
echo "检查 gpu_model_runner.py"
echo "=========================================="

FILE="/opt/dhl_conda_envs/verl_fast/lib/python3.10/site-packages/vllm/v1/worker/gpu_model_runner.py"

echo ""
echo "1. 文件是否存在？"
if [ -f "$FILE" ]; then
    echo "✅ 文件存在"
else
    echo "❌ 文件不存在"
    exit 1
fi

echo ""
echo "2. 文件大小："
ls -lh "$FILE"

echo ""
echo "3. 文件行数："
wc -l "$FILE"

echo ""
echo "4. 检查 GPUModelRunner 类定义："
grep -n "^class GPUModelRunner" "$FILE"

echo ""
echo "5. 检查文件末尾（最后 10 行）："
tail -10 "$FILE"

echo ""
echo "6. Python 语法检查："
python3 -m py_compile "$FILE" 2>&1
if [ $? -eq 0 ]; then
    echo "✅ 语法正确"
else
    echo "❌ 语法错误"
fi

echo ""
echo "7. 尝试导入 GPUModelRunner："
python3 -c "
import sys
sys.path.insert(0, '/opt/dhl_conda_envs/verl_fast/lib/python3.10/site-packages')
try:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    print('✅ 导入成功')
    print(f'   GPUModelRunner 类型: {type(GPUModelRunner)}')
except Exception as e:
    print(f'❌ 导入失败: {e}')
    import traceback
    traceback.print_exc()
"

echo ""
echo "=========================================="
echo "检查完成"
echo "=========================================="

