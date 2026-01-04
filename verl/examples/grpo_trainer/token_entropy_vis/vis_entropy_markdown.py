#!/usr/bin/env python3
"""
将熵输出的 JSONL 文件转换为 Markdown 可视化文件

功能：
1. 读取 JSONL 文件（每行一个样本）
2. 提取 epoch, step, marked_response（已标记高熵词的答案）
3. 生成美观的 Markdown 文件

使用方法：
    python visualize_entropy_to_markdown.py <jsonl_file>
    
    或者处理整个目录：
    python visualize_entropy_to_markdown.py <entropy_output_dir>

输出：
    在同一目录下生成 vis_markdown_{dataset_name}.md 文件
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import argparse


def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """读取 JSONL 文件"""
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                samples.append(sample)
            except json.JSONDecodeError as e:
                print(f"⚠️  警告：第 {line_num} 行 JSON 解析失败: {e}")
                continue
    return samples


def generate_markdown(samples: List[Dict[str, Any]], dataset_name: str) -> str:
    """生成 Markdown 内容"""
    
    # Markdown 头部
    md_lines = [
        f"# 熵可视化：{dataset_name}",
        "",
        f"**总样本数**: {len(samples)}",
        "",
        "---",
        "",
    ]
    
    # 按 (epoch, step) 分组
    grouped = {}
    for sample in samples:
        epoch = sample.get("epoch", 0)
        step = sample.get("step", 0)
        key = (epoch, step)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(sample)
    
    # 按 epoch 和 step 排序
    sorted_keys = sorted(grouped.keys())
    
    # 生成每个 (epoch, step) 的内容
    for epoch, step in sorted_keys:
        samples_in_group = grouped[(epoch, step)]
        
        md_lines.append(f"## Epoch {epoch} - Step {step}")
        md_lines.append("")
        md_lines.append(f"**样本数**: {len(samples_in_group)}")
        md_lines.append("")
        
        # 遍历该组中的每个样本
        for i, sample in enumerate(samples_in_group, 1):
            md_lines.append(f"### 样本 {i}")
            md_lines.append("")
            
            # 提取字段
            prompt = sample.get("prompt", "")
            marked_response = sample.get("marked_response", "")
            entropy_stats = sample.get("entropy_stats", {})
            response_length = sample.get("response_length", 0)
            entropy_length = sample.get("entropy_length", 0)
            top_k_tokens = sample.get("top_k_high_entropy_tokens", [])
            
            # 显示统计信息
            md_lines.append("**统计信息**:")
            md_lines.append("")
            md_lines.append(f"- **Response 长度**: {response_length} 字符")
            md_lines.append(f"- **Token 数量**: {entropy_length} tokens")
            md_lines.append(f"- **平均熵**: {entropy_stats.get('mean', 0):.4f}")
            md_lines.append(f"- **最小熵**: {entropy_stats.get('min', 0):.6f}")
            md_lines.append(f"- **最大熵**: {entropy_stats.get('max', 0):.4f}")
            md_lines.append(f"- **熵标准差**: {entropy_stats.get('std', 0):.4f}")
            md_lines.append("")
            
            # 显示 Top-K 高熵词
            if top_k_tokens:
                md_lines.append("**Top-K 高熵词**:")
                md_lines.append("")
                md_lines.append("| 排名 | Token | 熵值 | 位置 |")
                md_lines.append("|------|-------|------|------|")
                for rank, token_info in enumerate(top_k_tokens, 1):
                    token = token_info.get("token", "")
                    entropy = token_info.get("entropy", 0)
                    index = token_info.get("index", 0)
                    # 转义 Markdown 特殊字符
                    token_escaped = token.replace("|", "\\|").replace("\n", "\\n")
                    md_lines.append(f"| {rank} | `{token_escaped}` | {entropy:.4f} | {index} |")
                md_lines.append("")
            
            # 显示 Prompt（折叠）
            md_lines.append("<details>")
            md_lines.append("<summary><b>📝 点击查看 Prompt</b></summary>")
            md_lines.append("")
            md_lines.append("```")
            md_lines.append(prompt)
            md_lines.append("```")
            md_lines.append("")
            md_lines.append("</details>")
            md_lines.append("")
            
            # 显示标记后的答案（高熵词用 **[word]** 标记）
            md_lines.append("**🎯 答案（高熵词已标记）**:")
            md_lines.append("")
            md_lines.append(marked_response)
            md_lines.append("")
            
            # 分隔线
            md_lines.append("---")
            md_lines.append("")
    
    return "\n".join(md_lines)


def process_jsonl_file(jsonl_path: str) -> str:
    """处理单个 JSONL 文件"""
    print(f"📖 读取文件: {jsonl_path}")
    
    # 读取 JSONL
    samples = read_jsonl(jsonl_path)
    if not samples:
        print(f"⚠️  警告：文件为空或无有效样本")
        return None
    
    print(f"✅ 读取了 {len(samples)} 个样本")
    
    # 提取 dataset_name（从文件名）
    file_name = Path(jsonl_path).stem  # 例如 "gsm8k"
    dataset_name = file_name
    
    # 生成 Markdown
    print(f"🎨 生成 Markdown...")
    markdown_content = generate_markdown(samples, dataset_name)
    
    # 确定输出路径
    output_dir = Path(jsonl_path).parent
    output_path = output_dir / f"vis_markdown_{dataset_name}.md"
    
    # 写入文件
    print(f"💾 保存到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"✅ 完成！")
    return str(output_path)


def process_directory(entropy_output_dir: str):
    """处理整个目录（递归查找所有 JSONL 文件）"""
    print(f"📂 扫描目录: {entropy_output_dir}")
    
    entropy_dir = Path(entropy_output_dir)
    if not entropy_dir.exists():
        print(f"❌ 错误：目录不存在: {entropy_output_dir}")
        return
    
    # 查找所有 JSONL 文件（排除已生成的可视化文件）
    jsonl_files = []
    for jsonl_file in entropy_dir.rglob("*.jsonl"):
        # 跳过可视化文件
        if "vis_" in jsonl_file.name:
            continue
        jsonl_files.append(jsonl_file)
    
    if not jsonl_files:
        print(f"⚠️  警告：未找到任何 JSONL 文件")
        return
    
    print(f"✅ 找到 {len(jsonl_files)} 个 JSONL 文件")
    print("")
    
    # 处理每个文件
    for i, jsonl_file in enumerate(jsonl_files, 1):
        print(f"[{i}/{len(jsonl_files)}] 处理: {jsonl_file.relative_to(entropy_dir)}")
        try:
            output_path = process_jsonl_file(str(jsonl_file))
            if output_path:
                print(f"    ✅ 生成: {Path(output_path).relative_to(entropy_dir)}")
        except Exception as e:
            print(f"    ❌ 错误: {e}")
        print("")


def main():
    parser = argparse.ArgumentParser(
        description="将熵输出的 JSONL 文件转换为 Markdown 可视化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 处理单个文件
  python visualize_entropy_to_markdown.py /path/to/epoch_0/gsm8k.jsonl
  
  # 处理整个目录（递归）
  python visualize_entropy_to_markdown.py /path/to/Entropy_out/
  
  # 处理当前目录
  python visualize_entropy_to_markdown.py .
        """
    )
    
    parser.add_argument(
        "path",
        help="JSONL 文件路径或包含 JSONL 文件的目录"
    )
    
    args = parser.parse_args()
    
    path = Path(args.path)
    
    if not path.exists():
        print(f"❌ 错误：路径不存在: {args.path}")
        sys.exit(1)
    
    print("=" * 80)
    print("🎨 熵可视化：JSONL → Markdown")
    print("=" * 80)
    print("")
    
    if path.is_file():
        # 处理单个文件
        if path.suffix != ".jsonl":
            print(f"❌ 错误：不是 JSONL 文件: {args.path}")
            sys.exit(1)
        process_jsonl_file(str(path))
    elif path.is_dir():
        # 处理整个目录
        process_directory(str(path))
    else:
        print(f"❌ 错误：无效的路径: {args.path}")
        sys.exit(1)
    
    print("")
    print("=" * 80)
    print("✅ 所有任务完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()

