#!/usr/bin/env python3
"""
LLM-RL MT5 Trading System 安装脚本
自动检测环境、安装依赖、配置系统
"""

import sys
import os
import subprocess
import shutil
from pathlib import Path


def print_header(text):
    """打印标题"""
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def print_step(step, text):
    """打印步骤"""
    print(f"\n[{step}] {text}")


def run_command(command, check=True):
    """运行命令"""
    print(f"  执行: {command}")
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True
    )
    
    if check and result.returncode != 0:
        print(f"  错误: {result.stderr}")
        return False
    
    if result.stdout:
        print(f"  {result.stdout}")
    
    return True


def check_python_version():
    """检查Python版本"""
    print_step(1, "检查Python版本")
    
    version = sys.version_info
    print(f"  当前版本: {version.major}.{version.minor}.{version.micro}")
    
    if version < (3, 9):
        print("  ✗ Python版本过低，需要3.9+")
        return False
    
    print("  ✓ Python版本符合要求")
    return True


def create_directories():
    """创建必要的目录"""
    print_step(2, "创建目录结构")
    
    directories = [
        "data",
        "logs",
        "logs/reports",
        "logs/rl",
        "models",
        "models/rl",
        "models/rl_optimizer",
        "ea31337",
        "demo_ea31337",
        "config/symbols",
        "test_config",
        "test_logs",
        "test_logs/rl",
    ]
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory}")
    
    print("  ✓ 目录结构创建完成")
    return True


def install_dependencies():
    """安装依赖包"""
    print_step(3, "安装依赖包")
    
    if not Path("requirements.txt").exists():
        print("  ✗ requirements.txt 不存在")
        return False
    
    print("  安装必需依赖...")
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt"):
        print("  ✗ 依赖安装失败")
        return False
    
    print("  ✓ 依赖安装完成")
    return True


def setup_config():
    """设置配置文件"""
    print_step(4, "配置系统")
    
    config_file = Path("config/config.yaml")
    
    if not config_file.exists():
        print("  ✗ 配置文件不存在")
        return False
    
    print("  ✓ 配置文件已存在")
    
    # 检查是否需要配置MT5
    print("\n  是否需要配置MT5连接? (y/n): ", end="")
    response = input().strip().lower()
    
    if response == 'y':
        print("\n  请输入MT5配置信息:")
        
        server = input("    服务器地址: ").strip()
        login = input("    登录账号: ").strip()
        password = input("    密码: ").strip()
        
        # 更新配置文件
        import yaml
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        config['mt5']['server'] = server
        config['mt5']['login'] = int(login) if login else 0
        config['mt5']['password'] = password
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        
        print("  ✓ MT5配置已更新")
    
    return True


def verify_installation():
    """验证安装"""
    print_step(5, "验证安装")
    
    # 添加src到路径
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    
    try:
        from src.core.environment import check_environment
        
        status = check_environment()
        
        if not status.is_valid:
            print("\n  ✗ 环境验证失败")
            print("\n  错误:")
            for error in status.errors:
                print(f"    - {error}")
            
            if status.warnings:
                print("\n  警告:")
                for warning in status.warnings:
                    print(f"    - {warning}")
            
            return False
        
        print("\n  ✓ 环境验证通过")
        
        if status.warnings:
            print("\n  警告:")
            for warning in status.warnings:
                print(f"    - {warning}")
        
        return True
        
    except Exception as e:
        print(f"\n  ✗ 验证失败: {e}")
        return False


def create_startup_script():
    """创建启动脚本"""
    print_step(6, "创建启动脚本")
    
    # Windows批处理脚本
    if sys.platform == 'win32':
        script_content = f"""@echo off
echo Starting LLM-RL MT5 Trading System...
"{sys.executable}" main.py
pause
"""
        script_path = Path("start.bat")
        with open(script_path, 'w') as f:
            f.write(script_content)
        print("  ✓ 已创建 start.bat")
    
    # Linux/Mac shell脚本
    else:
        script_content = f"""#!/bin/bash
echo "Starting LLM-RL MT5 Trading System..."
{sys.executable} main.py
"""
        script_path = Path("start.sh")
        with open(script_path, 'w') as f:
            f.write(script_content)
        script_path.chmod(0o755)
        print("  ✓ 已创建 start.sh")
    
    return True


def main():
    """主安装流程"""
    print_header("LLM-RL MT5 Trading System 安装程序")
    
    print("\n欢迎使用 LLM-RL MT5 Trading System!")
    print("本程序将自动完成系统安装和配置。\n")
    
    # 执行安装步骤
    steps = [
        ("检查Python版本", check_python_version),
        ("创建目录结构", create_directories),
        ("安装依赖包", install_dependencies),
        ("配置系统", setup_config),
        ("验证安装", verify_installation),
        ("创建启动脚本", create_startup_script),
    ]
    
    for step_name, step_func in steps:
        if not step_func():
            print(f"\n✗ 安装失败: {step_name}")
            print("\n请检查错误信息并重试。")
            return 1
    
    print_header("安装完成!")
    
    print("\n系统已成功安装。")
    print("\n下一步:")
    print("  1. 编辑 config/config.yaml 配置文件")
    print("  2. 确保MT5已安装并运行")
    if sys.platform == 'win32':
        print("  3. 运行 start.bat 启动系统")
    else:
        print("  3. 运行 ./start.sh 启动系统")
    
    print("\n文档:")
    print("  - README.md: 系统概述")
    print("  - docs/: 详细文档")
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n安装已取消。")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n安装异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
