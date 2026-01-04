#!/bin/bash

################################################################################
# vLLM 自定义文件同步脚本 (增强版)
# 功能：将 vllm_my 中指定的文件夹和文件，全量覆盖到 vllm 运行环境中
################################################################################

set -e  # 遇到错误立即退出

# --- 颜色定义 ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# --- 路径定义 ---
TARGET_ROOT="/opt/dhl_conda_envs/verl_fast/lib/python3.10/site-packages/vllm"
SOURCE_ROOT="/opt/dhl_conda_envs/verl_fast/lib/python3.10/site-packages/vllm_my"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_ROOT="${TARGET_ROOT}_backups/backup_${TIMESTAMP}"

# ==============================================================================
# 📝在此处配置你要替换的内容
# ==============================================================================

# 1. 需要【全量替换】的文件夹列表 (相对于 vllm 根目录)
# 注意：这些文件夹在目标位置会被先删除再复制，确保没有残留旧文件
DIRS_TO_SYNC=(
    "v1"
    "worker"
    "model_executor/models"
)

# 2. 需要【单独替换】的文件列表 (相对于 vllm 根目录)
FILES_TO_SYNC=(
    "entrypoints/llm.py"
)

# ==============================================================================

echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}                           vLLM 代码增量同步工具${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo -e "   源目录: ${SOURCE_ROOT}"
echo -e "   目标目录: ${TARGET_ROOT}"
echo -e "   备份目录: ${BACKUP_ROOT}"
echo ""

################################################################################
# 步骤 1: 环境检查
################################################################################
echo -e "${YELLOW}[步骤 1/4] 检查源文件是否存在...${NC}"

if [ ! -d "$SOURCE_ROOT" ]; then
    echo -e "${RED}❌ 源目录不存在: ${SOURCE_ROOT}${NC}"
    exit 1
fi

if [ ! -d "$TARGET_ROOT" ]; then
    echo -e "${RED}❌ 目标目录不存在: ${TARGET_ROOT}${NC}"
    exit 1
fi

# 检查所有配置的文件夹
for dir in "${DIRS_TO_SYNC[@]}"; do
    if [ ! -d "${SOURCE_ROOT}/${dir}" ]; then
        echo -e "${RED}❌ 配置错误: 源文件夹不存在 -> ${SOURCE_ROOT}/${dir}${NC}"
        exit 1
    fi
done

# 检查所有配置的文件
for file in "${FILES_TO_SYNC[@]}"; do
    if [ ! -f "${SOURCE_ROOT}/${file}" ]; then
        echo -e "${RED}❌ 配置错误: 源文件不存在 -> ${SOURCE_ROOT}/${file}${NC}"
        exit 1
    fi
done

echo -e "${GREEN}✅ 所有源文件检查通过${NC}"
echo ""

################################################################################
# 步骤 2: 备份 (Backup)
################################################################################
echo -e "${YELLOW}[步骤 2/4] 正在备份目标文件...${NC}"
mkdir -p "$BACKUP_ROOT"

# 备份文件夹
for dir in "${DIRS_TO_SYNC[@]}"; do
    if [ -d "${TARGET_ROOT}/${dir}" ]; then
        # 保持目录结构备份
        mkdir -p "${BACKUP_ROOT}/$(dirname "${dir}")"
        cp -r "${TARGET_ROOT}/${dir}" "${BACKUP_ROOT}/${dir}"
        echo -e "   📦 已备份文件夹: ${dir}"
    else
        echo -e "   ⚠️  目标文件夹不存在(跳过备份): ${dir}"
    fi
done

# 备份文件
for file in "${FILES_TO_SYNC[@]}"; do
    if [ -f "${TARGET_ROOT}/${file}" ]; then
        mkdir -p "${BACKUP_ROOT}/$(dirname "${file}")"
        cp "${TARGET_ROOT}/${file}" "${BACKUP_ROOT}/${file}"
        echo -e "   📄 已备份文件: ${file}"
    else
        echo -e "   ⚠️  目标文件不存在(跳过备份): ${file}"
    fi
done

echo -e "${GREEN}✅ 备份完成${NC}"
echo ""

################################################################################
# 步骤 3: 替换 (Replace)
################################################################################
echo -e "${YELLOW}[步骤 3/4] 执行替换操作...${NC}"

# 1. 处理文件夹 (先删后拷，确保完全一致)
for dir in "${DIRS_TO_SYNC[@]}"; do
    TARGET_DIR="${TARGET_ROOT}/${dir}"
    SOURCE_DIR="${SOURCE_ROOT}/${dir}"
    
    echo -e "   📂 [文件夹] 同步: ${dir} ..."
    
    # 删除旧目录
    rm -rf "$TARGET_DIR"
    
    # 确保父目录存在
    mkdir -p "$(dirname "$TARGET_DIR")"
    
    # 复制新目录
    cp -r "$SOURCE_DIR" "$TARGET_DIR"
done

# 2. 处理单文件 (直接覆盖)
for file in "${FILES_TO_SYNC[@]}"; do
    TARGET_FILE="${TARGET_ROOT}/${file}"
    SOURCE_FILE="${SOURCE_ROOT}/${file}"
    
    echo -e "   📝 [文件]   同步: ${file} ..."
    
    cp "$SOURCE_FILE" "$TARGET_FILE"
done

echo -e "${GREEN}✅ 替换完成${NC}"
echo ""

################################################################################
# 步骤 4: 清理缓存 (Cleanup)
################################################################################
echo -e "${YELLOW}[步骤 4/4] 清理编译缓存...${NC}"

# 清理目标目录下的所有 .pyc 和 __pycache__
# 为了安全起见，只清理刚才涉及到的目录
for dir in "${DIRS_TO_SYNC[@]}"; do
    find "${TARGET_ROOT}/${dir}" -name "*.pyc" -delete 2>/dev/null || true
    find "${TARGET_ROOT}/${dir}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
done

# 重新编译 (可选，防止 import 错误)
echo -e "   正在重新编译 Python 字节码..."
python3 -m compileall -q "${TARGET_ROOT}" > /dev/null 2>&1 || true

echo -e "${GREEN}✅ 缓存清理完成${NC}"
echo ""

################################################################################
# 结束
################################################################################
echo -e "${BLUE}================================================================================================${NC}"
echo -e "${GREEN}🎉 同步操作成功！${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo -e "   备份路径: ${BACKUP_ROOT}"
echo -e "   请务必执行以下命令重启 Ray 以生效:"
echo -e "   ${BLUE}ray stop --force && sleep 2${NC}"
echo ""