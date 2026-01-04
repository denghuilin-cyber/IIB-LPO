#!/usr/bin/env python3
"""
检查代码与 Ray 的兼容性
Ray 的 SyntaxError 通常是由于：
1. 实际的语法错误
2. 导入错误
3. 配置错误（Hydra/OmegaConf）
4. 类型注解问题
"""

import sys
import ast
import traceback
from pathlib import Path
from typing import List, Tuple

def check_file_syntax(filepath: str) -> Tuple[bool, str]:
    """使用 AST 检查文件语法"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # 尝试解析 AST
        ast.parse(code, filename=filepath)
        return True, "语法正确"
    
    except SyntaxError as e:
        error_msg = f"语法错误 at line {e.lineno}, col {e.offset}: {e.msg}\n"
        if e.text:
            error_msg += f"  代码: {e.text.strip()}\n"
            if e.offset:
                error_msg += f"  位置: {' ' * (e.offset - 1)}^\n"
        return False, error_msg
    
    except Exception as e:
        return False, f"解析错误: {e}"

def check_imports(filepath: str) -> Tuple[bool, str]:
    """检查文件中的导入语句"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        
        tree = ast.parse(code, filename=filepath)
        
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        
        return True, f"找到 {len(imports)} 个导入语句"
    
    except Exception as e:
        return False, f"检查导入失败: {e}"

def check_dataclass_fields(filepath: str) -> Tuple[bool, str]:
    """检查 dataclass 字段定义"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        
        tree = ast.parse(code, filename=filepath)
        
        issues = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # 检查是否有 @dataclass 装饰器
                has_dataclass = any(
                    (isinstance(d, ast.Name) and d.id == 'dataclass') or
                    (isinstance(d, ast.Attribute) and d.attr == 'dataclass')
                    for d in node.decorator_list
                )
                
                if has_dataclass:
                    # 检查字段定义
                    for item in node.body:
                        if isinstance(item, ast.AnnAssign):
                            # 检查是否有类型注解
                            if item.annotation is None:
                                issues.append(f"类 {node.name} 的字段缺少类型注解")
        
        if issues:
            return False, "\n".join(issues)
        return True, "dataclass 字段定义正确"
    
    except Exception as e:
        return False, f"检查 dataclass 失败: {e}"

def check_hydra_config(filepath: str) -> Tuple[bool, str]:
    """检查 Hydra 配置相关的问题"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # 检查常见的 Hydra 配置问题
        issues = []
        
        # 检查是否有未闭合的字符串
        if code.count('"') % 2 != 0:
            issues.append("可能有未闭合的双引号")
        if code.count("'") % 2 != 0:
            issues.append("可能有未闭合的单引号")
        
        # 检查是否有未闭合的括号
        if code.count('(') != code.count(')'):
            issues.append(f"括号不匹配: ( {code.count('(')} vs ) {code.count(')')}")
        if code.count('[') != code.count(']'):
            issues.append(f"方括号不匹配: [ {code.count('[')} vs ] {code.count(']')}")
        if code.count('{') != code.count('}'):
            issues.append(f"花括号不匹配: {{ {code.count('{')} vs }} {code.count('}')}")
        
        if issues:
            return False, "\n".join(issues)
        return True, "Hydra 配置检查通过"
    
    except Exception as e:
        return False, f"检查 Hydra 配置失败: {e}"

def main():
    files_to_check = [
        "verl/utils/entropy_output_writer.py",
        "verl/workers/config/rollout.py",
        "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py",
        "verl/trainer/ppo/ray_trainer.py",
    ]
    
    print("="*80)
    print("🔍 Ray 兼容性检查")
    print("="*80)
    
    all_passed = True
    
    for filepath in files_to_check:
        print(f"\n{'='*80}")
        print(f"📄 检查文件: {filepath}")
        print(f"{'='*80}")
        
        path = Path(filepath)
        if not path.exists():
            print(f"❌ 文件不存在")
            all_passed = False
            continue
        
        # 1. 语法检查
        print("\n1️⃣ 语法检查...")
        ok, msg = check_file_syntax(filepath)
        if ok:
            print(f"  ✅ {msg}")
        else:
            print(f"  ❌ {msg}")
            all_passed = False
            continue  # 如果语法错误，后续检查没有意义
        
        # 2. 导入检查
        print("\n2️⃣ 导入检查...")
        ok, msg = check_imports(filepath)
        if ok:
            print(f"  ✅ {msg}")
        else:
            print(f"  ⚠️  {msg}")
        
        # 3. Dataclass 检查
        if 'config' in filepath or 'writer' in filepath:
            print("\n3️⃣ Dataclass 字段检查...")
            ok, msg = check_dataclass_fields(filepath)
            if ok:
                print(f"  ✅ {msg}")
            else:
                print(f"  ⚠️  {msg}")
        
        # 4. Hydra 配置检查
        print("\n4️⃣ Hydra 配置检查...")
        ok, msg = check_hydra_config(filepath)
        if ok:
            print(f"  ✅ {msg}")
        else:
            print(f"  ❌ {msg}")
            all_passed = False
    
    # 总结
    print(f"\n{'='*80}")
    print("📊 检查总结")
    print(f"{'='*80}")
    
    if all_passed:
        print("✅ 所有检查通过！")
        print("\n如果 Ray 仍然报 SyntaxError，可能的原因：")
        print("  1. 服务器上的文件版本不一致")
        print("  2. Ray workers 使用了旧的缓存")
        print("  3. 环境变量或配置问题")
        print("\n建议操作：")
        print("  1. 确认文件已上传到服务器")
        print("  2. 运行: find verl -name '*.pyc' -delete")
        print("  3. 运行: ray stop --force && rm -rf /tmp/ray/*")
        print("  4. 重新运行训练")
        return 0
    else:
        print("❌ 发现问题，请修复后重试")
        return 1

if __name__ == "__main__":
    sys.exit(main())

