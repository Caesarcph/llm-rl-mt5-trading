#!/usr/bin/env python3
"""
系统诊断命令行工具
快速诊断系统问题并提供解决方案
"""

import sys
import argparse
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.diagnostic_tool import run_diagnostics
from src.utils.performance_profiler import get_profiler
from src.core.environment import check_environment


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="LLM-RL MT5 Trading System 诊断工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python diagnose.py              # 运行完整诊断
  python diagnose.py --env        # 仅检查环境
  python diagnose.py --perf       # 显示性能报告
  python diagnose.py --quick      # 快速检查
        """
    )
    
    parser.add_argument(
        '--env',
        action='store_true',
        help='仅检查环境和依赖'
    )
    
    parser.add_argument(
        '--perf',
        action='store_true',
        help='显示性能分析报告'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='快速检查（跳过耗时检查）'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='详细输出'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("LLM-RL MT5 Trading System 诊断工具")
    print("=" * 60 + "\n")
    
    try:
        if args.env:
            # 仅环境检查
            print("正在检查环境...")
            status = check_environment()
            
            if status.is_valid:
                print("\n✓ 环境检查通过")
                return 0
            else:
                print("\n✗ 环境检查失败")
                return 1
        
        elif args.perf:
            # 性能报告
            print("正在生成性能报告...")
            profiler = get_profiler()
            
            # 收集当前指标
            profiler.collect_metrics()
            
            # 打印报告
            report = profiler.generate_report()
            print(report)
            
            return 0
        
        else:
            # 完整诊断
            report = run_diagnostics()
            
            # 返回状态码
            if report.overall_status == 'healthy':
                return 0
            elif report.overall_status == 'warning':
                return 1
            else:
                return 2
    
    except KeyboardInterrupt:
        print("\n\n诊断已取消。")
        return 130
    
    except Exception as e:
        print(f"\n✗ 诊断失败: {e}")
        
        if args.verbose:
            import traceback
            traceback.print_exc()
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
