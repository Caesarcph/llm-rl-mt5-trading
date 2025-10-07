#!/usr/bin/env python3
"""
LLM-RL MT5 Trading System
主程序入口点
"""

import sys
import os
import asyncio
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.config import load_config, get_config
from src.core.logging import setup_logging, get_logger
from src.core.exceptions import TradingSystemException
from src.core.trading_system import TradingSystem


def main():
    """主程序入口"""
    try:
        # 加载配置
        print("正在加载配置...")
        config = load_config()
        
        # 设置日志系统
        print("正在初始化日志系统...")
        setup_logging(config.logging)
        logger = get_logger()
        
        logger.info("=" * 60)
        logger.info("LLM-RL MT5 Trading System 启动")
        logger.info("=" * 60)
        
        # 验证配置
        if not config.validate():
            logger.error("配置验证失败，系统退出")
            return 1
        
        logger.info("配置验证通过")
        
        # 创建必要的目录
        create_directories()
        
        # 检查依赖
        if not check_dependencies():
            logger.error("依赖检查失败，系统退出")
            return 1
        
        logger.info("依赖检查通过")
        
        if config.simulation_mode:
            logger.info("系统运行在模拟模式")
        else:
            logger.warning("系统运行在实盘模式 - 请确保已充分测试")
        
        # 创建并启动交易系统
        logger.info("创建交易系统实例...")
        trading_system = TradingSystem()
        
        # 运行交易系统
        logger.info("启动交易系统...")
        asyncio.run(trading_system.start())
        
        return 0
        
    except TradingSystemException as e:
        print(f"交易系统异常: {e}")
        return 1
    except Exception as e:
        print(f"未知异常: {e}")
        import traceback
        traceback.print_exc()
        return 1


def create_directories():
    """创建必要的目录"""
    directories = [
        "data",
        "logs", 
        "models",
        "ea31337",
        "tests",
        "config/symbols"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger = get_logger()
    logger.info("目录结构创建完成")


def check_dependencies():
    """检查系统依赖"""
    logger = get_logger()
    
    try:
        # 检查Python版本
        if sys.version_info < (3, 9):
            logger.error(f"Python版本过低: {sys.version_info}, 需要3.9+")
            return False
        
        # 检查关键依赖包
        required_packages = [
            'pandas',
            'numpy', 
            'yaml',
            'MetaTrader5'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"缺少依赖包: {missing_packages}")
            logger.info("请运行: pip install -r requirements.txt")
            return False
        
        logger.info("所有依赖包检查通过")
        return True
        
    except Exception as e:
        logger.error(f"依赖检查异常: {e}")
        return False


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)