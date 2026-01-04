#!/usr/bin/env python3
"""
诊断脚本：检查所有修改的文件是否有语法错误
"""

import sys
import py_compile
import traceback
from pathlib import Path

# 需要检查的文件列表
files_to_check = [
    "verl/utils/entropy_output_writer.py",
    "verl/workers/config/rollout.py",
    "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py",
    "verl/trainer/ppo/ray_trainer.py",
]

def check_syntax(filepath):
    """检查单个文件的语法"""
    print(f"\n{'='*80}")
    print(f"检查文件: {filepath}")
    print(f"{'='*80}")
    
    try:
        # 尝试编译
        py_compile.compile(filepath, doraise=True)
        print(f"✅ 语法检查通过")
        return True
    except SyntaxError as e:
        print(f"❌ 语法错误!")
        print(f"  文件: {e.filename}")
        print(f"  行号: {e.lineno}")
        print(f"  列号: {e.offset}")
        print(f"  错误: {e.msg}")
        print(f"  代码: {e.text}")
        print(f"\n完整错误信息:")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"❌ 其他错误: {e}")
        traceback.print_exc()
        return False

def check_imports(filepath):
    """尝试导入模块"""
    print(f"\n尝试导入模块...")
    
    try:
        # 将文件路径转换为模块路径
        module_path = filepath.replace("/", ".").replace(".py", "")
        
        # 尝试导入
        exec(f"import {module_path}")
        print(f"✅ 导入成功")
        return True
    except ImportError as e:
        print(f"⚠️  导入错误 (可能是依赖问题): {e}")
        return True  # 导入错误不一定是语法问题
    except SyntaxError as e:
        print(f"❌ 导入时发现语法错误!")
        print(f"  文件: {e.filename}")
        print(f"  行号: {e.lineno}")
        print(f"  错误: {e.msg}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"⚠️  其他错误: {e}")
        return True

def main():
    print("="*80)
    print("🔍 开始诊断语法错误")
    print("="*80)
    
    all_passed = True
    failed_files = []
    
    for filepath in files_to_check:
        path = Path(filepath)
        
        if not path.exists():
            print(f"\n❌ 文件不存在: {filepath}")
            all_passed = False
            failed_files.append(filepath)
            continue
        
        # 检查语法
        syntax_ok = check_syntax(filepath)
        
        if not syntax_ok:
            all_passed = False
            failed_files.append(filepath)
            continue
        
        # 尝试导入
        # import_ok = check_imports(filepath)
        # if not import_ok:
        #     all_passed = False
        #     failed_files.append(filepath)
    
    # 总结
    print(f"\n{'='*80}")
    print("📊 诊断总结")
    print(f"{'='*80}")
    
    if all_passed:
        print("✅ 所有文件语法检查通过！")
        print("\n如果 Ray 仍然报错，可能的原因：")
        print("  1. 服务器上的文件未更新")
        print("  2. Python 缓存文件 (.pyc) 未清理")
        print("  3. Ray workers 未重启")
        print("  4. 依赖版本不匹配")
        return 0
    else:
        print(f"❌ 发现 {len(failed_files)} 个文件有问题:")
        for f in failed_files:
            print(f"  - {f}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

