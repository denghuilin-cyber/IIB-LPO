# 1. 停止 Ray
ray stop --force
sleep 3

# 2. 清理 Ray 临时文件
rm -rf /tmp/ray/*
rm -rf /tmp/ray_*
rm -rf ~/.ray/*
rm -rf /dev/shm/ray_*

# 3. 清理新项目的 Python 缓存
find /nas/dhl/verl -name "*.pyc" -delete
find /nas/dhl/verl -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# 4. 清理旧项目的 Python 缓存（防止被误导入）
find /nas/dhl/verl_success_entropy_token_vis_11_10改verl能跑 -name "*.pyc" -delete
find /nas/dhl/verl_success_entropy_token_vis_11_10改verl能跑 -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# 5. 清理 vLLM 缓存
find /opt/dhl_conda_envs/verl_fast/lib/python3.10/site-packages/vllm -name "*.pyc" -delete
find /opt/dhl_conda_envs/verl_fast/lib/python3.10/site-packages/vllm -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# 6. 重新编译新项目（忽略 .sh 文件错误）
python3 -m compileall -q /nas/dhl/verl 2>&1 | grep -v "test_cot_matching.py" | grep -v "SyntaxError" || true

# 7. 验证 Python 会导入哪个 verl
unset PYTHONPATH
export PYTHONPATH="/nas/dhl/verl"
python3 -c "import sys; print('PYTHONPATH:', [p for p in sys.path if 'verl' in p])"