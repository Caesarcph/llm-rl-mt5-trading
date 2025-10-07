#!/usr/bin/env python3
"""
EA31337桥接器演示脚本
展示如何使用EA31337Bridge和StrategyConfigManager
"""

import sys
import os
import json
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.bridge.ea31337_bridge import EA31337Bridge
from src.bridge.set_config_manager import StrategyConfigManager
from src.core.models import Signal


def demo_bridge_functionality():
    """演示桥接器基本功能"""
    print("=== EA31337桥接器功能演示 ===\n")
    
    # 创建桥接器实例
    with EA31337Bridge(config_path="demo_ea31337") as bridge:
        print(f"1. 桥接器初始化完成，配置路径: {bridge.config_path}")
        
        # 检查连接状态
        is_connected = bridge.is_connected()
        print(f"2. EA31337连接状态: {'已连接' if is_connected else '未连接'}")
        
        # 创建模拟信号文件
        demo_signals = {
            "signals": [
                {
                    "strategy_id": "trend_following_eurusd",
                    "symbol": "EURUSD",
                    "direction": 1,
                    "strength": 0.85,
                    "entry_price": 1.1000,
                    "sl": 1.0950,
                    "tp": 1.1100,
                    "size": 0.01,
                    "confidence": 0.80,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": {
                        "source": "demo",
                        "ma_fast": 12,
                        "ma_slow": 26
                    }
                },
                {
                    "strategy_id": "scalping_gbpusd",
                    "symbol": "GBPUSD",
                    "direction": -1,
                    "strength": 0.70,
                    "entry_price": 1.2500,
                    "sl": 1.2520,
                    "tp": 1.2480,
                    "size": 0.02,
                    "confidence": 0.75,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": {
                        "source": "demo",
                        "rsi": 75
                    }
                }
            ]
        }
        
        # 写入信号文件
        with open(bridge.signal_file, 'w', encoding='utf-8') as f:
            json.dump(demo_signals, f, ensure_ascii=False, indent=2)
        
        print("3. 创建模拟信号文件")
        
        # 获取信号
        signals = bridge.get_signals()
        print(f"4. 获取到 {len(signals)} 个交易信号:")
        for i, signal in enumerate(signals, 1):
            direction_text = "买入" if signal.direction == 1 else "卖出"
            print(f"   信号{i}: {signal.symbol} {direction_text}, 强度: {signal.strength:.2f}, 置信度: {signal.confidence:.2f}")
        
        # 创建模拟状态文件
        demo_status = {
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "strategies": {
                "trend_following_eurusd": {"active": True, "signals": 15, "profit": 125.50},
                "scalping_gbpusd": {"active": True, "signals": 8, "profit": -23.10}
            },
            "account": {
                "balance": 10000.0,
                "equity": 10102.40,
                "margin": 150.0,
                "free_margin": 9952.40
            }
        }
        
        with open(bridge.status_file, 'w', encoding='utf-8') as f:
            json.dump(demo_status, f, ensure_ascii=False, indent=2)
        
        print("5. 创建模拟状态文件")
        
        # 获取状态
        status = bridge.get_status()
        if status:
            print(f"6. EA状态: {status['status']}")
            print(f"   账户余额: ${status['account']['balance']:.2f}")
            print(f"   账户净值: ${status['account']['equity']:.2f}")
            print(f"   活跃策略数: {len(status['strategies'])}")
        
        # 发送命令
        print("7. 发送控制命令:")
        commands = [
            ("start_strategy", {"strategy": "breakout_xauusd"}),
            ("update_parameters", {"strategy": "trend_following_eurusd", "parameters": {"Lots": 0.02, "TakeProfit": 120}}),
            ("stop_strategy", {"strategy": "scalping_gbpusd"})
        ]
        
        for command, params in commands:
            success = bridge.send_command(command, params)
            print(f"   {command}: {'成功' if success else '失败'}")
        
        # 健康检查
        health = bridge.health_check()
        print("8. 系统健康检查:")
        print(f"   桥接器连接: {'正常' if health['bridge_connected'] else '异常'}")
        print(f"   写入权限: {'正常' if health['write_permission'] else '异常'}")
        print(f"   信号文件: {'存在' if health['signal_file_exists'] else '不存在'}")
        print(f"   状态文件: {'存在' if health['status_file_exists'] else '不存在'}")


def demo_config_manager():
    """演示配置管理器功能"""
    print("\n=== 策略配置管理器演示 ===\n")
    
    # 创建配置管理器
    config_manager = StrategyConfigManager(config_dir="demo_configs")
    print(f"1. 配置管理器初始化完成，配置目录: {config_manager.config_dir}")
    
    # 显示可用模板
    templates = config_manager.get_template_list()
    print(f"2. 可用策略模板: {', '.join(templates)}")
    
    # 基于模板创建配置
    print("3. 基于模板创建策略配置:")
    
    strategies_to_create = [
        ("trend_following", "eurusd_trend", "EURUSD", {"Lots": 0.01, "MaxRisk": 1.5}),
        ("scalping", "gbpusd_scalp", "GBPUSD", {"Lots": 0.02, "MaxSpread": 1.0}),
        ("breakout", "xauusd_breakout", "XAUUSD", {"Lots": 0.01, "MaxRisk": 2.5})
    ]
    
    for template, strategy_name, symbol, custom_params in strategies_to_create:
        success = config_manager.create_from_template(template, strategy_name, symbol, custom_params)
        print(f"   {strategy_name} ({symbol}): {'成功' if success else '失败'}")
    
    # 显示创建的配置
    config_list = config_manager.get_config_list()
    print(f"4. 已创建的配置: {', '.join(config_list)}")
    
    # 加载和显示配置详情
    print("5. 配置详情:")
    for strategy_name in config_list:
        config = config_manager.load_config(strategy_name)
        if config:
            print(f"   {strategy_name}:")
            print(f"     手数: {config.get('Lots', 'N/A')}")
            print(f"     止盈: {config.get('TakeProfit', 'N/A')}")
            print(f"     止损: {config.get('StopLoss', 'N/A')}")
            print(f"     最大风险: {config.get('MaxRisk', 'N/A')}%")
    
    # 参数优化演示
    print("6. 参数优化演示:")
    optimization_results = {
        "Lots": 0.015,
        "TakeProfit": 110,
        "StopLoss": 45,
        "MA_Period_Fast": 10
    }
    
    optimized_config = config_manager.optimize_parameters("eurusd_trend", "EURUSD", optimization_results)
    if optimized_config:
        print("   优化后的参数:")
        for param, value in optimization_results.items():
            if param in optimized_config:
                print(f"     {param}: {optimized_config[param]}")
    
    # 配置验证
    print("7. 配置验证:")
    for strategy_name in config_list[:2]:  # 验证前两个配置
        config = config_manager.load_config(strategy_name)
        errors = config_manager.validate_config(config)
        if errors:
            print(f"   {strategy_name}: 发现 {len(errors)} 个错误")
            for error in errors:
                print(f"     - {error}")
        else:
            print(f"   {strategy_name}: 配置有效")


def main():
    """主函数"""
    print("EA31337集成桥接器演示程序")
    print("=" * 50)
    
    try:
        # 演示桥接器功能
        demo_bridge_functionality()
        
        # 演示配置管理器功能
        demo_config_manager()
        
        print("\n=== 演示完成 ===")
        print("生成的文件:")
        print("- demo_ea31337/: EA31337通信文件目录")
        print("- demo_configs/: 策略配置文件目录")
        print("\n可以查看这些目录中的文件来了解系统的工作方式。")
        
    except Exception as e:
        print(f"演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()