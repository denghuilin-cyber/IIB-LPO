import json
import matplotlib.pyplot as plt
import numpy as np
import os

# 请将此处修改为您文件的实际路径
# 注意：您提供的路径中包含引号，请确认路径是否正确
file_path = '/nas/dhl/Eval/math-eval/“/nas/dhl/Eval/Eval_Output/qwen3-8192”/MATH-500-test-temp_0.0-top_p_0.95-top_k_-1.jsonl'

# 如果路径中的引号是多余的，请使用下面的格式（示例）：
# file_path = '/nas/dhl/Eval/math-eval/Eval_Output/Qwen3-4B-Instruct-8192/MATH-500-test-temp_0.0-top_p_0.95-top_k_-1.jsonl'

def analyze_response_lengths(path):
    lengths = []
    
    # 检查文件是否存在
    if not os.path.exists(path):
        print(f"错误: 找不到文件 {path}")
        return

    print(f"正在读取文件: {path} ...")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # 获取 vanilla_response 字段
                    if 'vanilla_response' in data:
                        response_text = data['vanilla_response']
                        lengths.append(len(response_text))
                except json.JSONDecodeError:
                    print("跳过无法解析的行")
                    continue
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return

    if not lengths:
        print("未找到有效的数据或 vanilla_response 字段。")
        return

    # 1. 计算统计数据
    avg_len = np.mean(lengths)
    max_len = np.max(lengths)
    min_len = np.min(lengths)
    
    print("-" * 30)
    print(f"统计结果 (基于 {len(lengths)} 条数据):")
    print(f"平均长度 (Average): {avg_len:.2f}")
    print(f"最大长度 (Max):     {max_len}")
    print(f"最小长度 (Min):     {min_len}")
    print("-" * 30)

    # 2. 可视化：绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Model Response Lengths')
    plt.xlabel('Length (characters)')
    plt.ylabel('Frequency')
    
    # 添加平均值的竖线
    plt.axvline(avg_len, color='red', linestyle='dashed', linewidth=1, label=f'Avg: {avg_len:.2f}')
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    
    # 保存图片
    output_img = 'response_length_dist.png'
    plt.savefig(os.path.join( os.path.dirname(file_path) ,output_img))
    print(f"可视化图表已保存为: {output_img}")
    plt.show()

if __name__ == "__main__":
    # 如果您想测试，可以创建一个包含您示例数据的临时文件
    # analyze_response_lengths("test.jsonl") 
    
    # 运行主逻辑
    analyze_response_lengths(file_path)