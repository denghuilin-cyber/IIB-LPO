#!/bin/bash

# 分步骤环境安装脚本 - 修正版
# 使用方法: ./install_env.sh [步骤号]

# ============ 配置区 ============
ENV_PATH="/opt/dhl_conda_envs/verl_fast"
CONDA_SH="/nas/dhl/envs/bin/activate"
PYTHON_VERSION="3.10"

# 本地whl文件路径
FLASH_ATTN_WHL="/nas/dhl/envs/DHL_git/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
FLASHINFER_WHL="/nas/dhl/envs/DHL_git/flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl"

# 源配置 - 清华源优先
TSINGHUA="https://pypi.tuna.tsinghua.edu.cn/simple"
ALIYUN="https://mirrors.aliyun.com/pypi/simple/"

LOG_DIR="./install_logs"
mkdir -p "$LOG_DIR"

# ============ 工具函数 ============
log_info() {
    echo "[INFO] $1"
}

log_error() {
    echo "[ERROR] $1"
}

log_warn() {
    echo "[WARN] $1"
}

log_step() {
    echo ""
    echo "=========================================="
    echo "  $1"
    echo "=========================================="
}

'''
# 这部分自己运行，剩余的包 用脚本一锅端
# 创建环境
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge 
conda config --set show_channel_urls yes
conda create -p /opt/dhl_conda_envs/verl_fast python=3.10 -y
# 激活环境
conda activate /opt/dhl_conda_envs/verl_fast
conda activate /opt/dhl_conda_envs/verl_fast 
conda config --append envs_dirs /opt/dhl_conda_envs
'''

mark_done() {
    touch "$LOG_DIR/step_$1_done"
    log_info "✓ 步骤 $1 完成"
}

# 自己自定义一个安装函数
pi() {
    log_info "安装: $@"
    
    if pip install -i "$TSINGHUA" "$@" 2>&1 | tee -a "$LOG_DIR/current_step.log"; then
        log_info "✓ 安装成功 (清华源)"
        return 0
    fi
    
    log_warn "清华源失败，尝试阿里云源..."
    if pip install -i "$ALIYUN" "$@" 2>&1 | tee -a "$LOG_DIR/current_step.log"; then
        log_info "✓ 安装成功 (阿里云源)"
        return 0
    fi
    
    log_error "✗ 安装失败: $@"
    log_warn "继续执行..."
    return 1
}

# ============ 安装步骤 ============

step_1() {
    # 接受conda TOS
    log_info "接受conda服务条款..."
    conda config --set channel_priority flexible
    conda config --add channels conda-forge
    
    conda activate "$ENV_PATH"
    log_info "Python版本: $(python --version)"
    
    mark_done 1
}

step_2() {
    log_step "步骤2: 安装PyTorch"
    check_done 2 && { log_info "已完成，跳过"; return 0; }
    #activate_env
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/
    python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
    mark_done 2
}

step_3() {
    log_step "步骤3: Transformers生态"
    check_done 3 && { log_info "已完成，跳过"; return 0; }
    #activate_env
    pi transformers==4.51.1 tokenizers==0.21.4 accelerate==1.10.1 peft==0.17.1 \
       safetensors==0.6.2 huggingface-hub==0.36.0 hf-transfer==0.1.9 sentencepiece==0.2.1 tiktoken==0.12.0
    mark_done 3
}

step_4() {
    log_step "步骤4: Flash Attention (本地)"
    check_done 4 && { log_info "已完成，跳过"; return 0; }
    #activate_env
    if [ -f "$FLASH_ATTN_WHL" ]; then
        pip install "$FLASH_ATTN_WHL"
    else
        log_error "文件不存在: $FLASH_ATTN_WHL"
    fi
    mark_done 4
}

step_5() {
    log_step "步骤5: xformers"
    check_done 5 && { log_info "已完成，跳过"; return 0; }
    #activate_env
    pi xformers==0.0.29.post2
    mark_done 5
}

step_6() {
    log_step "步骤6: flashinfer (本地)"
    check_done 6 && { log_info "已完成，跳过"; return 0; }
    activate_env
    if [ -f "$FLASHINFER_WHL" ]; then
        pip install "$FLASHINFER_WHL"
    else
        log_error "文件不存在: $FLASHINFER_WHL"
    fi
    mark_done 6
}

step_7() {
    log_step "步骤7: vLLM"
    check_done 7 && { log_info "已完成，跳过"; return 0; }
    #activate_env
    pi vllm==0.8.5.post1
    mark_done 7
}

step_8() {
    log_step "步骤8: SGLang"
    check_done 8 && { log_info "已完成，跳过"; return 0; }
    #activate_env
    pi sglang==0.4.6.post1
    mark_done 8
}

step_9() {
    log_step "步骤9: 数据科学库"
    check_done 9 && { log_info "已完成，跳过"; return 0; }
    #activate_env
    pi numpy==2.2.6 pandas==2.3.3 scipy==1.15.3 datasets==4.3.0 pyarrow==22.0.0
    cd /nas/dhl/verl && pip install -e .
    pi numpy==2.2.6 opentelemetry-sdk==1.26.0  opentelemetry-api==1.26.0
    mark_done 9
}

step_10() {
    log_step "步骤10: 机器学习工具"
    check_done 10 && { log_info "已完成，跳过"; return 0; }
    #activate_env
    pi ray==2.49.1 wandb==0.22.3 tensorboard==2.20.0 tensordict==0.9.1 torchdata==0.11.0 einops==0.8.1
    mark_done 10
}

step_11() {
    log_step "步骤11: Web框架"
    check_done 11 && { log_info "已完成，跳过"; return 0; }
    #activate_env
    pi fastapi==0.120.4 uvicorn==0.38.0 pydantic==2.12.3 httpx==0.28.1 aiohttp==3.13.2
    mark_done 11
}

step_12() {
    log_step "步骤12: 配置管理"
    check_done 12 && { log_info "已完成，跳过"; return 0; }
    #activate_env
    pi hydra-core==1.3.2 omegaconf==2.3.0 PyYAML==6.0.3
    mark_done 12
}

step_13() {
    log_step "步骤13: 视觉工具"
    check_done 13 && { log_info "已完成，跳过"; return 0; }
    activate_env
    pi opencv-python-headless==4.12.0.88 pillow==11.3.0 av==16.0.1 qwen-vl-utils==0.0.11
    mark_done 13
}

step_14() {
    log_step "步骤14: 优化库"
    check_done 14 && { log_info "已完成，跳过"; return 0; }
    activate_env
    pi triton==3.2.0 liger-kernel==0.6.3 torchao==0.13.0 compressed-tensors==0.9.3
    mark_done 14
}

step_15() {
    log_step "步骤15: 其他工具"
    check_done 15 && { log_info "已完成，跳过"; return 0; }
    #activate_env
    pi tqdm==4.67.1 psutil==7.1.2 pytest==8.4.2 ruff==0.14.3 \
       cupy-cuda12x==13.6.0 numba==0.61.2 openai==2.6.1 ipython==8.37.0 \
       GitPython==3.1.45 protobuf==4.25.8 ninja==1.13.0 uvloop==0.21.0 wrapt==2.0.0 \
       virtualenv==20.35.4 tomli==2.2.1 smart_open==7.3.0.post1 sentry-sdk==2.43.0 \
       codetiming==1.4.0 colorful==0.5.8 torch_memory_saver==0.0.8 setuptools==78.1.1 \
       rsa==4.9.1 pylatexenc==2.10 pynvml==12.0.0 pyasn1==0.6.1 pyasn1_modules==0.4.2 \
       pybind11==3.0.1 rich==14.2.0 rignore==0.7.3 pyext==0.7 pre_commit==4.3.0 \
       nvidia-nvshmem-cu12==3.3.20 nvidia-cufile-cu12==1.13.1.3 nodeenv==1.9.1 mathruler==0.1.0 \
       cfgv==3.4.0 opencv-fixer==0.2.5  optree==0.17.0 rpds-py==0.28.0 rich-toolkit==0.16.0
    pip uninstall opentelemetry-exporter-prometheus
    mark_done 15
}

# ============ 主程序 ============
main() {
    local start=${1:-0}
    local end=${2:-15}
    
    log_info "环境路径: $ENV_PATH"
    log_info "步骤范围: $start - $end"
    
    for i in $(seq $start $end); do
        step_$i || log_warn "步骤 $i 有警告，继续..."
    done
    
    log_step "安装完成！"
}

# 执行入口
if [ $# -eq 1 ]; then
    main 1 15
else
    main $1 $2
fi
