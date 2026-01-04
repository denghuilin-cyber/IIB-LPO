#!/usr/bin/env python3
"""
可视化 branching.jsonl 文件
将分叉树结构转换为 Markdown 格式

使用方法:
    python visualize_branching.py <branching.jsonl路径>
    
输出:
    在同目录下生成 branching_visualization.md
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any


def load_branching_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    """加载 branching.jsonl 文件"""
    records = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def format_content_preview(content: str, max_length: int = 100) -> str:
    """格式化内容预览（截取前 max_length 个字符）"""
    if len(content) <= max_length:
        return content
    return content[:max_length] + "..."


def generate_markdown_for_single_sample(record: Dict[str, Any]) -> str:
    """为单个样本生成 Markdown（简洁版）"""
    sample_idx = record.get("sample_idx", "N/A")
    pure_question = record.get("pure_question", "N/A")
    gt = record.get("gt", "N/A")  # 🆕 获取 ground truth
    branches = record.get("branches", [])
    forking_vis = record.get("forking_vis_markdown", "")
    
    # 开始构建 Markdown
    md_lines = []
    
    # 标题
    md_lines.append(f"## Sample {sample_idx}")
    md_lines.append("")
    
    # 问题
    md_lines.append(f"**Question:**")
    md_lines.append(f"> {pure_question}")
    md_lines.append("")
    
    # 分叉树可视化
    md_lines.append(f"**Branching Tree:**")
    md_lines.append("")
    md_lines.append("```")
    md_lines.append(forking_vis)
    md_lines.append("```")
    md_lines.append("")
    
    # 直接显示每个分支（不要 "Branch Details" 标题）
    for branch in branches:
        branch_name = branch.get("branch", "N/A")
        parent_branch = branch.get("parent_branch", "None (root)")
        branching_point = branch.get("branching_point_token_idx", "N/A")
        tokens = branch.get("tokens", "N/A")
        avg_entropy = branch.get("avg_entropy", 0.0)
        reward = branch.get("reward", "N/A")  # 🆕 获取 reward
        content = branch.get("content", "")
        
        # 分支标题
        md_lines.append(f"**Branch `{branch_name}`**")
        md_lines.append("")
        
        # 分支元数据
        md_lines.append(f"- **Parent Branch:** {parent_branch}")
        if branching_point != "N/A":
            md_lines.append(f"- **Branching Point:** Token {branching_point}")
        md_lines.append(f"- **Tokens:** {tokens}")
        md_lines.append(f"- **Average Entropy:** {avg_entropy:.4f}")
        md_lines.append(f"- **Ground Truth:** {gt}")  # 🆕 添加 ground truth
        md_lines.append(f"- **Reward:** {reward}")  # 🆕 添加 reward
        md_lines.append("")
        
        # 完整内容（不折叠，直接显示）
        md_lines.append(f"**Full Content:**")
        md_lines.append("")
        md_lines.append(f"```")
        md_lines.append(content)
        md_lines.append(f"```")
        md_lines.append("")
    
    md_lines.append("---")
    md_lines.append("")
    
    return "\n".join(md_lines)


def generate_markdown_visualization(records: List[Dict[str, Any]]) -> str:
    """生成完整的 Markdown 可视化"""
    md_lines = []
    
    # 文档标题
    md_lines.append("# Branching Visualization")
    md_lines.append("")
    md_lines.append(f"**Total Samples:** {len(records)}")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    
    # 为每个样本生成 Markdown
    for record in records:
        sample_md = generate_markdown_for_single_sample(record)
        md_lines.append(sample_md)
    
    return "\n".join(md_lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_branching.py <branching.jsonl路径>")
        print("Example: python visualize_branching.py ./output/epoch_0/branching.jsonl")
        sys.exit(1)
    
    # 获取输入文件路径
    input_path = Path(sys.argv[1])
    
    if not input_path.exists():
        print(f"❌ 错误: 文件不存在: {input_path}")
        sys.exit(1)
    
    if not input_path.is_file():
        print(f"❌ 错误: 不是文件: {input_path}")
        sys.exit(1)
    
    print(f"📖 读取文件: {input_path}")
    
    # 加载数据
    try:
        records = load_branching_jsonl(str(input_path))
        print(f"✅ 成功加载 {len(records)} 个样本")
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 生成 Markdown
    print(f"🔨 生成 Markdown 可视化...")
    try:
        markdown_content = generate_markdown_visualization(records)
    except Exception as e:
        print(f"❌ 生成 Markdown 失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 保存到同目录下
    output_path = input_path.parent / "branching_visualization.md"
    print(f"💾 保存到: {output_path}")
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        print(f"✅ 可视化完成！")
        print(f"📄 输出文件: {output_path}")
    except Exception as e:
        print(f"❌ 保存文件失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

