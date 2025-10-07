#!/usr/bin/env python3
"""
系统设置验证脚本
验证项目基础结构和核心接口是否正确建立
"""

import sys
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

def validate_project_structure():
    """验证项目目录结构"""
    print("🔍 验证项目目录结构...")
    
    required_dirs = [
        "src/core",
        "src/data", 
        "src/strategies",
        "src/agents",
        "src/bridge",
        "src/utils",
        "config",
        "config/symbols",
        "tests",
        "data",
        "logs"
    ]
    
    missing_dirs = []
    for directory in required_dirs:
        if not Path(directory).exists():
            missing_dirs.append(directory)
    
    if missing_dirs:
        print(f"❌ 缺少目录: {missing_dirs}")
        return False
    
    print("✅ 项目目录结构完整")
    return True


def validate_core_modules():
    """验证核心模块"""
    print("🔍 验证核心模块...")
    
    try:
        # 测试数据模型
        from src.core.models import MarketData, Signal, Account, Position, Trade
        from src.core.models import DataProvider, OrderExecutor, Strategy
        print("✅ 数据模型导入成功")
        
        # 测试配置管理
        from src.core.config import ConfigManager, SystemConfig, get_config
        config = get_config()
        assert config.validate(), "配置验证失败"
        print("✅ 配置管理系统正常")
        
        # 测试日志系统
        from src.core.logging import setup_logging, get_logger
        logger = get_logger()
        logger.info("日志系统测试")
        print("✅ 日志系统正常")
        
        # 测试异常处理
        from src.core.exceptions import TradingSystemException, DataValidationException
        print("✅ 异常处理框架正常")
        
        return True
        
    except Exception as e:
        print(f"❌ 核心模块验证失败: {e}")
        return False


def validate_configuration_files():
    """验证配置文件"""
    print("🔍 验证配置文件...")
    
    required_configs = [
        "config/config.yaml",
        "config/symbols/eurusd.yaml",
        "config/symbols/xauusd.yaml", 
        "config/symbols/usoil.yaml"
    ]
    
    missing_configs = []
    for config_file in required_configs:
        if not Path(config_file).exists():
            missing_configs.append(config_file)
    
    if missing_configs:
        print(f"❌ 缺少配置文件: {missing_configs}")
        return False
    
    # 测试配置加载
    try:
        from src.core.config import load_config
        config = load_config()
        assert len(config.trading.symbols) > 0, "交易品种配置为空"
        print("✅ 配置文件完整且可正常加载")
        return True
    except Exception as e:
        print(f"❌ 配置文件验证失败: {e}")
        return False


def validate_data_models():
    """验证数据模型功能"""
    print("🔍 验证数据模型功能...")
    
    try:
        from datetime import datetime
        import pandas as pd
        from src.core.models import MarketData, Signal, Account, Position, PositionType
        
        # 测试MarketData
        sample_ohlcv = pd.DataFrame({
            'time': [datetime.now()],
            'open': [1.1000],
            'high': [1.1010], 
            'low': [1.0990],
            'close': [1.1005],
            'volume': [1000]
        })
        
        market_data = MarketData(
            symbol="EURUSD",
            timeframe="H1",
            timestamp=datetime.now(),
            ohlcv=sample_ohlcv
        )
        assert market_data.symbol == "EURUSD"
        
        # 测试Signal
        signal = Signal(
            strategy_id="test",
            symbol="EURUSD",
            direction=1,
            strength=0.8,
            entry_price=1.1000,
            sl=1.0950,
            tp=1.1050,
            size=0.1,
            confidence=0.9,
            timestamp=datetime.now()
        )
        assert signal.direction == 1
        
        # 测试Account
        account = Account(
            account_id="12345",
            balance=10000.0,
            equity=9800.0,
            margin=500.0,
            free_margin=9300.0,
            margin_level=1960.0,
            currency="USD",
            leverage=100
        )
        assert account.can_open_position(1000.0)
        
        # 测试Position
        position = Position(
            position_id="pos_001",
            symbol="EURUSD",
            type=PositionType.LONG,
            volume=1.0,
            open_price=1.1000,
            current_price=1.1050
        )
        pnl = position.calculate_unrealized_pnl()
        assert abs(pnl - 0.005) < 0.001
        
        print("✅ 数据模型功能正常")
        return True
        
    except Exception as e:
        print(f"❌ 数据模型验证失败: {e}")
        return False


def validate_interfaces():
    """验证接口定义"""
    print("🔍 验证接口定义...")
    
    try:
        from src.core.models import DataProvider, OrderExecutor, Strategy
        
        # 检查接口方法
        data_provider_methods = ['get_market_data', 'get_account_info', 'get_positions', 'get_trades_history']
        for method in data_provider_methods:
            assert hasattr(DataProvider, method), f"DataProvider缺少方法: {method}"
        
        order_executor_methods = ['send_order', 'close_position', 'modify_position']
        for method in order_executor_methods:
            assert hasattr(OrderExecutor, method), f"OrderExecutor缺少方法: {method}"
        
        strategy_methods = ['generate_signal', 'update_parameters', 'get_performance_metrics']
        for method in strategy_methods:
            assert hasattr(Strategy, method), f"Strategy缺少方法: {method}"
        
        print("✅ 接口定义完整")
        return True
        
    except Exception as e:
        print(f"❌ 接口验证失败: {e}")
        return False


def main():
    """主验证函数"""
    print("🚀 开始验证LLM-RL MT5交易系统基础结构...")
    print("=" * 60)
    
    validation_results = []
    
    # 执行各项验证
    validation_results.append(validate_project_structure())
    validation_results.append(validate_core_modules())
    validation_results.append(validate_configuration_files())
    validation_results.append(validate_data_models())
    validation_results.append(validate_interfaces())
    
    print("=" * 60)
    
    # 汇总结果
    passed = sum(validation_results)
    total = len(validation_results)
    
    if passed == total:
        print(f"🎉 验证完成! 所有 {total} 项检查都通过了")
        print("✅ 项目基础结构和核心接口已成功建立")
        print("\n📋 任务完成情况:")
        print("  ✅ 创建项目目录结构")
        print("  ✅ 定义核心数据模型接口")
        print("  ✅ 实现基础配置管理系统")
        print("  ✅ 创建日志系统和异常处理框架")
        print("\n🎯 下一步: 可以开始实现任务2 - MT5连接和数据管道")
        return 0
    else:
        print(f"❌ 验证失败! {passed}/{total} 项检查通过")
        print("请修复上述问题后重新验证")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)