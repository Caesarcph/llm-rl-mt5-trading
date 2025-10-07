#!/usr/bin/env python3
"""
EA31337桥接器集成测试
"""

import unittest
import json
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.bridge.ea31337_bridge import EA31337Bridge
from src.bridge.set_config_manager import StrategyConfigManager
from src.core.models import Signal
from src.core.exceptions import BridgeError, ConfigurationException


class TestEA31337Bridge(unittest.TestCase):
    """EA31337桥接器测试"""
    
    def setUp(self):
        """测试初始化"""
        self.temp_dir = tempfile.mkdtemp()
        self.bridge = EA31337Bridge(config_path=self.temp_dir, signal_timeout=10)
        
        # 创建测试信号数据
        self.test_signals = [
            {
                "strategy_id": "test_strategy_1",
                "symbol": "EURUSD",
                "direction": 1,
                "strength": 0.8,
                "entry_price": 1.1000,
                "sl": 1.0950,
                "tp": 1.1100,
                "size": 0.01,
                "confidence": 0.75,
                "timestamp": datetime.now().isoformat(),
                "metadata": {"source": "test"}
            },
            {
                "strategy_id": "test_strategy_2",
                "symbol": "GBPUSD",
                "direction": -1,
                "strength": 0.6,
                "entry_price": 1.2500,
                "sl": 1.2550,
                "tp": 1.2400,
                "size": 0.02,
                "confidence": 0.65,
                "timestamp": datetime.now().isoformat(),
                "metadata": {"source": "test"}
            }
        ]
        
        # 创建测试状态数据
        self.test_status = {
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "strategies": {
                "test_strategy_1": {"active": True, "signals": 5},
                "test_strategy_2": {"active": True, "signals": 3}
            },
            "account": {
                "balance": 10000.0,
                "equity": 10050.0,
                "margin": 100.0
            }
        }
    
    def tearDown(self):
        """测试清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_bridge_initialization(self):
        """测试桥接器初始化"""
        self.assertTrue(Path(self.temp_dir).exists())
        self.assertEqual(self.bridge.config_path, Path(self.temp_dir))
        self.assertEqual(self.bridge.signal_timeout, 10)
        
        # 检查通信文件路径
        self.assertTrue(self.bridge.signal_file.parent.exists())
        self.assertTrue(self.bridge.status_file.parent.exists())
    
    def test_get_signals_empty(self):
        """测试获取空信号列表"""
        signals = self.bridge.get_signals()
        self.assertEqual(signals, [])
    
    def test_get_signals_with_data(self):
        """测试获取信号数据"""
        # 写入测试信号文件
        signal_data = {"signals": self.test_signals}
        with open(self.bridge.signal_file, 'w', encoding='utf-8') as f:
            json.dump(signal_data, f, ensure_ascii=False, indent=2)
        
        # 获取信号
        signals = self.bridge.get_signals()
        
        self.assertEqual(len(signals), 2)
        self.assertIsInstance(signals[0], Signal)
        self.assertEqual(signals[0].strategy_id, "test_strategy_1")
        self.assertEqual(signals[0].symbol, "EURUSD")
        self.assertEqual(signals[0].direction, 1)
        self.assertEqual(signals[1].strategy_id, "test_strategy_2")
        self.assertEqual(signals[1].symbol, "GBPUSD")
        self.assertEqual(signals[1].direction, -1)
    
    def test_get_signals_expired(self):
        """测试过期信号过滤"""
        # 创建过期信号
        expired_signal = self.test_signals[0].copy()
        expired_signal["timestamp"] = (datetime.now() - timedelta(seconds=60)).isoformat()
        
        signal_data = {"signals": [expired_signal]}
        with open(self.bridge.signal_file, 'w', encoding='utf-8') as f:
            json.dump(signal_data, f, ensure_ascii=False, indent=2)
        
        # 获取信号（应该为空，因为信号已过期）
        signals = self.bridge.get_signals()
        self.assertEqual(len(signals), 0)
    
    def test_get_status_empty(self):
        """测试获取空状态"""
        status = self.bridge.get_status()
        self.assertIsNone(status)
    
    def test_get_status_with_data(self):
        """测试获取状态数据"""
        # 写入测试状态文件
        with open(self.bridge.status_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_status, f, ensure_ascii=False, indent=2)
        
        # 获取状态
        status = self.bridge.get_status()
        
        self.assertIsNotNone(status)
        self.assertEqual(status["status"], "running")
        self.assertIn("strategies", status)
        self.assertEqual(len(status["strategies"]), 2)
    
    def test_send_command(self):
        """测试发送命令"""
        command = "start_strategy"
        params = {"strategy": "test_strategy"}
        
        result = self.bridge.send_command(command, params)
        self.assertTrue(result)
        
        # 检查命令文件是否创建
        self.assertTrue(self.bridge.command_file.exists())
        
        # 验证命令内容
        with open(self.bridge.command_file, 'r', encoding='utf-8') as f:
            command_data = json.load(f)
        
        self.assertEqual(command_data["command"], command)
        self.assertEqual(command_data["params"], params)
        self.assertIn("timestamp", command_data)
        self.assertIn("id", command_data)
    
    def test_update_parameters(self):
        """测试更新策略参数"""
        strategy = "test_strategy"
        params = {"Lots": 0.02, "TakeProfit": 150}
        
        result = self.bridge.update_parameters(strategy, params)
        self.assertTrue(result)
        
        # 验证命令文件内容
        with open(self.bridge.command_file, 'r', encoding='utf-8') as f:
            command_data = json.load(f)
        
        self.assertEqual(command_data["command"], "update_parameters")
        self.assertEqual(command_data["params"]["strategy"], strategy)
        self.assertEqual(command_data["params"]["parameters"], params)
    
    def test_is_connected_no_status(self):
        """测试连接状态检查（无状态文件）"""
        self.assertFalse(self.bridge.is_connected())
    
    def test_is_connected_with_recent_status(self):
        """测试连接状态检查（最近状态）"""
        # 写入最近的状态文件
        with open(self.bridge.status_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_status, f, ensure_ascii=False, indent=2)
        
        self.assertTrue(self.bridge.is_connected())
    
    def test_is_connected_with_old_status(self):
        """测试连接状态检查（过期状态）"""
        # 写入过期状态文件
        old_status = self.test_status.copy()
        old_status["timestamp"] = (datetime.now() - timedelta(minutes=10)).isoformat()
        
        with open(self.bridge.status_file, 'w', encoding='utf-8') as f:
            json.dump(old_status, f, ensure_ascii=False, indent=2)
        
        self.assertFalse(self.bridge.is_connected())
    
    def test_strategy_control_commands(self):
        """测试策略控制命令"""
        strategy = "test_strategy"
        
        # 测试启动策略
        result = self.bridge.start_strategy(strategy)
        self.assertTrue(result)
        
        # 测试停止策略
        result = self.bridge.stop_strategy(strategy)
        self.assertTrue(result)
        
        # 测试紧急停止
        result = self.bridge.emergency_stop()
        self.assertTrue(result)
    
    def test_get_strategy_status(self):
        """测试获取策略状态"""
        # 写入状态文件
        with open(self.bridge.status_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_status, f, ensure_ascii=False, indent=2)
        
        # 获取策略状态
        strategy_status = self.bridge.get_strategy_status("test_strategy_1")
        self.assertIsNotNone(strategy_status)
        self.assertTrue(strategy_status["active"])
        self.assertEqual(strategy_status["signals"], 5)
        
        # 获取不存在的策略状态
        strategy_status = self.bridge.get_strategy_status("nonexistent")
        self.assertIsNone(strategy_status)
    
    def test_health_check(self):
        """测试健康检查"""
        health = self.bridge.health_check()
        
        self.assertIn("bridge_connected", health)
        self.assertIn("signal_file_exists", health)
        self.assertIn("status_file_exists", health)
        self.assertIn("write_permission", health)
        self.assertIn("timestamp", health)
        
        # 写入状态文件后再次检查
        with open(self.bridge.status_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_status, f, ensure_ascii=False, indent=2)
        
        health = self.bridge.health_check()
        self.assertIn("ea_status", health)
        self.assertEqual(health["ea_status"], "running")
    
    def test_cleanup(self):
        """测试清理功能"""
        # 创建临时文件
        self.bridge.command_file.write_text("test")
        self.bridge.response_file.write_text("test")
        
        self.assertTrue(self.bridge.command_file.exists())
        self.assertTrue(self.bridge.response_file.exists())
        
        # 执行清理
        self.bridge.cleanup()
        
        self.assertFalse(self.bridge.command_file.exists())
        self.assertFalse(self.bridge.response_file.exists())
    
    def test_context_manager(self):
        """测试上下文管理器"""
        with EA31337Bridge(config_path=self.temp_dir) as bridge:
            self.assertIsInstance(bridge, EA31337Bridge)
            
            # 创建临时文件
            bridge.command_file.write_text("test")
            self.assertTrue(bridge.command_file.exists())
        
        # 退出上下文后应该清理文件
        self.assertFalse(bridge.command_file.exists())


class TestStrategyConfigManager(unittest.TestCase):
    """策略配置管理器测试"""
    
    def setUp(self):
        """测试初始化"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = StrategyConfigManager(config_dir=self.temp_dir)
        
        # 创建测试配置
        self.test_config = {
            "Lots": 0.01,
            "TakeProfit": 100,
            "StopLoss": 50,
            "MaxSpread": 3.0,
            "MA_Period_Fast": 12,
            "MA_Period_Slow": 26,
            "MaxRisk": 2.0
        }
    
    def tearDown(self):
        """测试清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_config_manager_initialization(self):
        """测试配置管理器初始化"""
        self.assertTrue(Path(self.temp_dir).exists())
        self.assertEqual(self.config_manager.config_dir, Path(self.temp_dir))
        self.assertGreater(len(self.config_manager.templates), 0)
    
    def test_save_and_load_config(self):
        """测试保存和加载配置"""
        strategy_name = "test_strategy"
        
        # 保存配置
        result = self.config_manager.save_config(strategy_name, self.test_config)
        self.assertTrue(result)
        
        # 检查文件是否创建
        set_file = Path(self.temp_dir) / f"{strategy_name}.set"
        self.assertTrue(set_file.exists())
        
        # 加载配置
        loaded_config = self.config_manager.load_config(strategy_name)
        self.assertEqual(loaded_config["Lots"], 0.01)
        self.assertEqual(loaded_config["TakeProfit"], 100)
        self.assertEqual(loaded_config["MaxRisk"], 2.0)
    
    def test_load_nonexistent_config(self):
        """测试加载不存在的配置"""
        config = self.config_manager.load_config("nonexistent")
        self.assertEqual(config, {})
    
    def test_create_from_template(self):
        """测试基于模板创建配置"""
        template_name = "trend_following"
        strategy_name = "test_trend_strategy"
        symbol = "EURUSD"
        custom_params = {"Lots": 0.02, "MaxRisk": 1.5}
        
        result = self.config_manager.create_from_template(
            template_name, strategy_name, symbol, custom_params
        )
        self.assertTrue(result)
        
        # 验证创建的配置
        config = self.config_manager.load_config(strategy_name)
        self.assertEqual(config["Lots"], 0.02)
        self.assertEqual(config["MaxRisk"], 1.5)
        self.assertIn("TakeProfit", config)
        self.assertIn("StopLoss", config)
    
    def test_create_from_invalid_template(self):
        """测试使用无效模板创建配置"""
        result = self.config_manager.create_from_template(
            "invalid_template", "test_strategy", "EURUSD"
        )
        self.assertFalse(result)
    
    def test_optimize_parameters(self):
        """测试参数优化"""
        strategy_name = "test_strategy"
        
        # 先保存基础配置
        self.config_manager.save_config(strategy_name, self.test_config)
        
        # 优化参数
        optimization_results = {
            "Lots": 0.03,
            "TakeProfit": 120,
            "MA_Period_Fast": 15
        }
        
        optimized_config = self.config_manager.optimize_parameters(
            strategy_name, "EURUSD", optimization_results
        )
        
        self.assertEqual(optimized_config["Lots"], 0.03)
        self.assertEqual(optimized_config["TakeProfit"], 120)
        self.assertEqual(optimized_config["MA_Period_Fast"], 15)
        self.assertEqual(optimized_config["StopLoss"], 50)  # 未优化的参数保持不变
    
    def test_get_template_list(self):
        """测试获取模板列表"""
        templates = self.config_manager.get_template_list()
        self.assertIsInstance(templates, list)
        self.assertIn("trend_following", templates)
        self.assertIn("scalping", templates)
        self.assertIn("breakout", templates)
    
    def test_get_config_list(self):
        """测试获取配置列表"""
        # 创建几个配置文件
        self.config_manager.save_config("strategy1", self.test_config)
        self.config_manager.save_config("strategy2", self.test_config)
        
        config_list = self.config_manager.get_config_list()
        self.assertIn("strategy1", config_list)
        self.assertIn("strategy2", config_list)
    
    def test_delete_config(self):
        """测试删除配置"""
        strategy_name = "test_strategy"
        
        # 创建配置
        self.config_manager.save_config(strategy_name, self.test_config)
        set_file = Path(self.temp_dir) / f"{strategy_name}.set"
        self.assertTrue(set_file.exists())
        
        # 删除配置
        result = self.config_manager.delete_config(strategy_name)
        self.assertTrue(result)
        self.assertFalse(set_file.exists())
        
        # 删除不存在的配置
        result = self.config_manager.delete_config("nonexistent")
        self.assertFalse(result)
    
    def test_validate_config(self):
        """测试配置验证"""
        # 有效配置
        errors = self.config_manager.validate_config(self.test_config)
        self.assertEqual(len(errors), 0)
        
        # 无效配置（缺少必需参数）
        invalid_config = {"Lots": 0.01}
        errors = self.config_manager.validate_config(invalid_config)
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("TakeProfit" in error for error in errors))
        
        # 无效配置（参数值错误）
        invalid_config = {"Lots": -0.01, "TakeProfit": 100, "StopLoss": 50}
        errors = self.config_manager.validate_config(invalid_config)
        self.assertGreater(len(errors), 0)
    
    def test_symbol_specific_adjustments(self):
        """测试品种特定调整"""
        # 测试黄金配置调整
        result = self.config_manager.create_from_template(
            "trend_following", "xauusd_strategy", "XAUUSD"
        )
        self.assertTrue(result)
        
        config = self.config_manager.load_config("xauusd_strategy")
        # 黄金的点差和止盈止损应该被放大
        self.assertGreater(config["MaxSpread"], 3.0)
        self.assertGreater(config["TakeProfit"], 100)
        
        # 测试日元对配置调整
        result = self.config_manager.create_from_template(
            "trend_following", "usdjpy_strategy", "USDJPY"
        )
        self.assertTrue(result)
        
        config = self.config_manager.load_config("usdjpy_strategy")
        # 日元对的止盈止损应该被放大100倍
        self.assertGreater(config["TakeProfit"], 1000)
        self.assertGreater(config["StopLoss"], 1000)


if __name__ == '__main__':
    # 设置日志级别
    logging.basicConfig(level=logging.DEBUG)
    
    # 运行测试
    unittest.main(verbosity=2)