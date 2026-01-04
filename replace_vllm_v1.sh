#!/bin/bash

################################################################################
# vLLM v1 文件替换脚本
# 功能：备份原始 v1 文件夹，然后用修改后的文件替换
################################################################################

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 路径定义
TARGET_VLLM="/opt/dhl_conda_envs/verl_fast/lib/python3.10/site-packages/vllm"
SOURCE_VLLM="/opt/dhl_conda_envs/verl_fast/lib/python3.10/site-packages/vllm_my"
BACKUP_DIR="${TARGET_VLLM}/v1_backup_$(date +%Y%m%d_%H%M%S)"

echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}                           vLLM v1 文件替换脚本${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo ""

# 整个vllm根目录下的文件 都替换掉
mkdir -p "/opt/dhl_conda_envs/verl_fast/lib/python3.10/site-packages/vllm_$(date+ %Y%m%d_%H%M)"

cp $TARGET_VLLM/*.py "/opt/dhl_conda_envs/verl_fast/lib/python3.10/site-packages/vllm_$(date+ %Y%m%d_%H%M)/"
# 把整个根目录下的文件 都拷贝一遍
cp $SOURCE_VLLM/*.py $TARGET_VLLM


TARGET_FILE="${TARGET_VLLM}/sampling_params.py"
SOURCE_FILE="${SOURCE_VLLM}/sampling_params.py"
BACKUP_FILE="${TARGET_VLLM}/sampling_params_backup.py"
# 备份
cp "${TARGET_FILE}" "${BACKUP_FILE}"
echo "✅ 已备份: ${BACKUP_FILE}"
# 替换
cp "${SOURCE_FILE}" "${TARGET_FILE}"
echo "✅ 已替换: ${TARGET_FILE}"


TARGET_FILE="${TARGET_VLLM}/outputs.py"
SOURCE_FILE="${SOURCE_VLLM}/outputs.py"
BACKUP_FILE="${TARGET_VLLM}/outputs_backup.py"
# 备份
cp "${TARGET_FILE}" "${BACKUP_FILE}"
echo "✅ 已备份: ${BACKUP_FILE}"
# 替换
cp "${SOURCE_FILE}" "${TARGET_FILE}"
echo "✅ 已替换: ${TARGET_FILE}"


TARGET_FILE="${TARGET_VLLM}/forward_context.py"
SOURCE_FILE="${SOURCE_VLLM}/forward_context.py"
BACKUP_FILE="${TARGET_VLLM}/forward_context_backup.py"
# 备份
cp "${TARGET_FILE}" "${BACKUP_FILE}"
echo "✅ 已备份: ${BACKUP_FILE}"
# 替换
cp "${SOURCE_FILE}" "${TARGET_FILE}"
echo "✅ 已替换: ${TARGET_FILE}"


TARGET_FILE="${TARGET_VLLM}/sequence.py"
SOURCE_FILE="${SOURCE_VLLM}/sequence.py"
BACKUP_FILE="${TARGET_VLLM}/sequence_backup.py"
# 备份
cp "${TARGET_FILE}" "${BACKUP_FILE}"
echo "✅ 已备份: ${BACKUP_FILE}"
# 替换
cp "${SOURCE_FILE}" "${TARGET_FILE}"
echo "✅ 已替换: ${TARGET_FILE}"

################################################################################
# 步骤 1: 检查路径是否存在
################################################################################
echo -e "${YELLOW}[步骤 1/5] 检查路径...${NC}"

if [ ! -d "$TARGET_VLLM" ]; then
    echo -e "${RED}❌ 错误: 目标路径不存在: $TARGET_VLLM${NC}"
    exit 1
fi

if [ ! -d "$SOURCE_VLLM" ]; then
    echo -e "${RED}❌ 错误: 源路径不存在: $SOURCE_VLLM${NC}"
    exit 1
fi

if [ ! -d "${TARGET_VLLM}/v1" ]; then
    echo -e "${RED}❌ 错误: v1 文件夹不存在: ${TARGET_VLLM}/v1${NC}"
    exit 1
fi

if [ ! -d "${SOURCE_VLLM}/v1" ]; then
    echo -e "${RED}❌ 错误: 修改后的 v1 文件夹不存在: ${SOURCE_VLLM}/v1${NC}"
    exit 1
fi

echo -e "${GREEN}✅ 所有路径检查通过${NC}"
echo -e "   目标 vLLM: ${TARGET_VLLM}"
echo -e "   源 vLLM:   ${SOURCE_VLLM}"
echo ""

################################################################################
# 步骤 2: 显示将要替换的文件
################################################################################
echo -e "${YELLOW}[步骤 2/5] 统计将要替换的文件...${NC}"

# 统计 Python 文件数量
TARGET_PY_COUNT=$(find "${TARGET_VLLM}/v1" -name "*.py" -type f | wc -l)
SOURCE_PY_COUNT=$(find "${SOURCE_VLLM}/v1" -name "*.py" -type f | wc -l)

echo -e "   目标 v1 文件夹中的 Python 文件: ${TARGET_PY_COUNT} 个"
echo -e "   源 v1 文件夹中的 Python 文件:   ${SOURCE_PY_COUNT} 个"
echo ""

# 列出修改过的文件（包含 "entropies" 或 "compute_entropy" 的文件）
echo -e "${BLUE}📝 检测到以下文件可能被修改过（包含熵相关代码）:${NC}"
grep -r "compute_entropy\|entropies" "${SOURCE_VLLM}" --include="*.py" -l 2>/dev/null | \
    sed "s|${SOURCE_VLLM}/||" | sort | while read -r file; do
    echo -e "   ${GREEN}✓${NC} $file"
done
echo ""

################################################################################
# 步骤 3: 用户确认
################################################################################
echo -e "${YELLOW}[步骤 3/5] 确认操作...${NC}"
echo -e "${RED}⚠️  警告: 此操作将会:${NC}"
echo -e "   1. 备份整个 v1 文件夹到: ${BACKUP_DIR}"
echo -e "   2. 删除目标 v1 文件夹中的所有 Python 文件"
echo -e "   3. 用修改后的文件替换"
echo ""


################################################################################
# 步骤 4: 备份原始 v1 文件夹
################################################################################
echo -e "${YELLOW}[步骤 4/5] 备份原始 v1 文件夹...${NC}"

# 创建备份目录
mkdir -p "$BACKUP_DIR"

# 复制整个 v1 文件夹
echo -e "   正在复制 v1 文件夹..."
cp -r "${TARGET_VLLM}/v1" "${BACKUP_DIR}/"

# 验证备份
if [ -d "${BACKUP_DIR}/v1" ]; then
    BACKUP_PY_COUNT=$(find "${BACKUP_DIR}/v1" -name "*.py" -type f | wc -l)
    echo -e "${GREEN}✅ 备份成功${NC}"
    echo -e "   备份路径: ${BACKUP_DIR}/v1"
    echo -e "   备份文件数: ${BACKUP_PY_COUNT} 个 Python 文件"
else
    echo -e "${RED}❌ 备份失败！操作终止${NC}"
    exit 1
fi
echo ""

################################################################################
# 步骤 5: 替换 v1 文件夹
################################################################################
echo -e "${YELLOW}[步骤 5/5] 替换 v1 文件夹...${NC}"

# 删除目标 v1 文件夹
echo -e "   正在删除目标 v1 文件夹..."
rm -rf "${TARGET_VLLM}/v1"

# 复制修改后的 v1 文件夹
echo -e "   正在复制修改后的 v1 文件夹..."
cp -r "${SOURCE_VLLM}/v1" "${TARGET_VLLM}/"

# 验证替换
if [ -d "${TARGET_VLLM}/v1" ]; then
    NEW_PY_COUNT=$(find "${TARGET_VLLM}/v1" -name "*.py" -type f | wc -l)
    echo -e "${GREEN}✅ 替换成功${NC}"
    echo -e "   新 v1 文件夹中的 Python 文件: ${NEW_PY_COUNT} 个"
else
    echo -e "${RED}❌ 替换失败！正在恢复备份...${NC}"
    cp -r "${BACKUP_DIR}/v1" "${TARGET_VLLM}/"
    echo -e "${GREEN}✅ 已恢复备份${NC}"
    exit 1
fi
echo ""

################################################################################
# 步骤 6: 清理缓存并重新编译
################################################################################
echo -e "${YELLOW}[步骤 6/5] 清理缓存并重新编译...${NC}"

# 清理 .pyc 文件
echo -e "   正在清理 .pyc 文件..."
find "${TARGET_VLLM}/v1" -name "*.pyc" -delete 2>/dev/null || true

# 清理 __pycache__ 目录
echo -e "   正在清理 __pycache__ 目录..."
find "${TARGET_VLLM}/v1" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# 重新编译
echo -e "   正在重新编译 Python 文件..."
python3 -m compileall -q "${TARGET_VLLM}/v1" 2>/dev/null || true

echo -e "${GREEN}✅ 缓存清理和重新编译完成${NC}"
echo ""

################################################################################
# 完成
################################################################################
echo -e "${BLUE}================================================================================================${NC}"
echo -e "${GREEN}🎉 替换完成！${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo ""
echo -e "${GREEN}✅ 操作总结:${NC}"
echo -e "   1. 原始 v1 文件夹已备份到: ${BACKUP_DIR}/v1"
echo -e "   2. 已用修改后的文件替换目标 v1 文件夹"
echo -e "   3. 已清理缓存并重新编译"
echo ""
echo -e "${YELLOW}📝 下一步操作:${NC}"
echo -e "   1. 重启 Ray: ${BLUE}ray stop --force && sleep 2${NC}"
echo -e "   2. 运行训练脚本验证熵值计算是否正常"
echo ""
echo -e "${YELLOW}🔄 如需恢复备份:${NC}"
echo -e "   ${BLUE}rm -rf ${TARGET_VLLM}/v1${NC}"
echo -e "   ${BLUE}cp -r ${BACKUP_DIR}/v1 ${TARGET_VLLM}/${NC}"
echo -e "   ${BLUE}find ${TARGET_VLLM}/v1 -name '*.pyc' -delete${NC}"
echo -e "   ${BLUE}find ${TARGET_VLLM}/v1 -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true${NC}"
echo ""
echo -e "${GREEN}✨ 完成！${NC}"

