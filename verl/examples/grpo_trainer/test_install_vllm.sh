# #!/bin/bash
# # 安装修改后的 vLLM（支持熵计算）
# # 使用方法：bash install_modified_vllm.sh

# set -e  # 遇到错误立即退出

# echo "================================"
# echo "🔧 安装修改后的 vLLM"
# echo "================================"

# # 1. 检查当前 vLLM 位置
# # echo ""
# # echo "步骤 1: 检查当前 vLLM 安装位置"
# # echo "--------------------------------"
# # python3 -c "
# # import vllm
# # print('当前 vLLM 位置:', vllm.__file__)
# # print('当前 vLLM 版本:', vllm.__version__)
# # "

# # # 2. 卸载旧的 vLLM
# # echo ""
# # echo "步骤 2: 卸载旧的 vLLM"
# # echo "--------------------------------"
# # pip uninstall vllm -y || echo "（vLLM 未安装或已卸载）"

# # 3. 找到修改后的 vLLM 源码目录
# VLLM_SOURCE_DIR="/nas/dhl/vllm"

# if [ ! -d "$VLLM_SOURCE_DIR" ]; then
#     echo "❌ 错误：找不到 vLLM 源码目录: $VLLM_SOURCE_DIR"
#     exit 1
# fi

# echo "vLLM 源码目录: $VLLM_SOURCE_DIR"

# # 4. 检查关键修改是否存在
# echo ""
# echo "步骤 3: 验证 vLLM 修改"
# echo "--------------------------------"

# check_file() {
#     local file=$1
#     local pattern=$2
#     local description=$3
    
#     if grep -q "$pattern" "$file"; then
#         echo "  ✅ $description"
#         return 0
#     else
#         echo "  ❌ $description - 未找到修改"
#         return 1
#     fi
# }

# all_checks_passed=true

# check_file "$VLLM_SOURCE_DIR/sampling_params.py" "compute_entropy" "sampling_params.py 有 compute_entropy" || all_checks_passed=false
# check_file "$VLLM_SOURCE_DIR/outputs.py" "entropies: Optional\[list\[float\]\]" "outputs.py 有 entropies 字段" || all_checks_passed=false
# check_file "$VLLM_SOURCE_DIR/sequence.py" "_output_entropies" "sequence.py 有 _output_entropies" || all_checks_passed=false
# check_file "$VLLM_SOURCE_DIR/model_executor/layers/sampler.py" "entropies_tensor" "sampler.py 有熵计算" || all_checks_passed=false

# if [ "$all_checks_passed" = false ]; then
#     echo ""
#     echo "❌ 错误：vLLM 源码修改不完整！"
#     echo "请确保已经按照 VLLM_MODIFICATIONS_COMPLETE.md 完成所有修改。"
#     exit 1
# fi

# # 5. 创建 setup.py（如果不存在）
# echo ""
# echo "步骤 4: 创建 setup.py"
# echo "--------------------------------"

# if [ ! -f "$VLLM_SOURCE_DIR/setup.py" ]; then
#     echo "创建 setup.py..."
#     cat > "$VLLM_SOURCE_DIR/setup.py" << 'EOF'
# from setuptools import setup, find_packages

# setup(
#     name="vllm",
#     version="0.8.5.post1+entropy",
#     packages=find_packages(),
#     python_requires=">=3.8",
#     install_requires=[
#         "torch>=2.0.0",
#         "transformers>=4.30.0",
#         "numpy",
#     ],
# )
# EOF
#     echo "  ✅ setup.py 已创建"
# else
#     echo "  ✅ setup.py 已存在"
# fi

# # 6. 以开发模式安装
# echo ""
# echo "步骤 5: 以开发模式安装 vLLM"
# echo "--------------------------------"
# cd "$VLLM_SOURCE_DIR"
# pip install -e . --no-build-isolation

# # 7. 验证安装
# echo ""
# echo "步骤 6: 验证安装"
# echo "--------------------------------"

# python3 << 'EOF'
# import sys

# print("=" * 80)
# print("vLLM 熵计算功能验证")
# print("=" * 80)

# # 测试 1: 检查 vLLM 位置
# import vllm
# print(f"\n✅ vLLM 位置: {vllm.__file__}")
# print(f"✅ vLLM 版本: {vllm.__version__}")

# # 测试 2: SamplingParams
# from vllm.sampling_params import SamplingParams
# try:
#     sp = SamplingParams(compute_entropy=True)
#     print(f"✅ SamplingParams.compute_entropy = {sp.compute_entropy}")
# except Exception as e:
#     print(f"❌ SamplingParams 错误: {e}")
#     sys.exit(1)

# # 测试 3: CompletionOutput
# from vllm.outputs import CompletionOutput
# try:
#     output = CompletionOutput(
#         index=0, text="test", token_ids=[1,2,3],
#         cumulative_logprob=None, logprobs=None,
#         entropies=[0.3, 0.25, 0.4]
#     )
#     print(f"✅ CompletionOutput.entropies = {output.entropies}")
# except Exception as e:
#     print(f"❌ CompletionOutput 错误: {e}")
#     sys.exit(1)

# # 测试 4: SequenceData
# from vllm.sequence import SequenceData
# from array import array
# try:
#     seq_data = SequenceData(array('l', [1, 2, 3]))
#     seq_data.append_token_id(4, 0.5, entropy=0.3)
#     entropies = seq_data.get_output_entropies()
#     print(f"✅ SequenceData.get_output_entropies() = {entropies}")
# except Exception as e:
#     print(f"❌ SequenceData 错误: {e}")
#     sys.exit(1)

# # 测试 5: SequenceOutput
# from vllm.sequence import SequenceOutput, Logprob
# try:
#     seq_out = SequenceOutput(
#         parent_seq_id=0, output_token=123,
#         logprobs={123: Logprob(logprob=0.0, rank=1, decoded_token="test")},
#         entropy=0.456
#     )
#     print(f"✅ SequenceOutput.entropy = {seq_out.entropy}")
# except Exception as e:
#     print(f"❌ SequenceOutput 错误: {e}")
#     sys.exit(1)

# print("\n" + "=" * 80)
# print("🎉 所有测试通过！vLLM 熵计算功能已正确安装！")
# print("=" * 80)
# EOF

# if [ $? -eq 0 ]; then
#     echo ""
#     echo "================================"
#     echo "✅ vLLM 安装成功！"
#     echo "================================"
#     echo ""
#     echo "下一步："
#     echo "1. 删除旧的训练输出: rm -rf /nas/dhl/outputs/test_cot_entropy_output_qwen2_5_1_5B"
#     echo "2. 重新运行训练: bash examples/grpo_trainer/test_k_cot_entropy.sh"
#     echo ""
# else
#     echo ""
#     echo "================================"
#     echo "❌ vLLM 安装失败！"
#     echo "================================"
#     exit 1
# fi


(verl) ctyun172022236164# echo "=== 尝试导入 vLLM ==="
                          python3 -m pip show vllm
=== 尝试导入 vLLM ===
Name: vllm
Version: 0.8.5.post1+entropy
Summary: 
Home-page: 
Author: 
Author-email: 
License: 
Location: /nas/dhl/vllm
Editable project location: /nas/dhl/vllm
Requires: numpy, torch, transformers
Required-by: 
(verl) ctyun172022236164# 

python3 -c "import sys; print('Python path:', sys.executable)"
python3 -c "import vllm; print('vLLM 位置:', vllm.__file__)"

Python path: /nas/dhl/envs/envs/verl/bin/python3
INFO 11-01 14:46:29 [__init__.py:239] Automatically detected platform cuda.
vLLM 位置: /nas/dhl/vllm/__init__.py
