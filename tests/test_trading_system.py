#!/usr/bin/env python3
"""
交易系统集成测试
测试主交易系统类的初始化、运行和关闭流程
"""

import unittest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.trading_system import TradingSystem, SystemState, SystemStats
from src.core.exceptions import TradingSystemException


class TestTradingSystem(unittest.TestCase):
    """交易系统测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.trading_system = None
    
    def tearDown(self):
        """测试后清理"""
        if self.trading_system:
            try:
                asyncio.run(self.trading_system.shutdown())
            except:
                pass
    
    def test_system_creation(self):
        """测试系统创建"""
        system = TradingSystem()
        
        self.assertIsNotNone(system)
        self.assertEqual(system.state, SystemState.STOPPED)
        self.assertIsInstance(system.stats, SystemStats)
        self.assertIsNone(system.stats.start_time)
    
    def test_system_state_transitions(self):
        """测试系统状态转换"""
        system = TradingSystem()
        
        # 初始状态
        self.assertEqual(system.state, SystemState.STOPPED)
        
        # 状态转换
        system.state = SystemState.STARTING
        self.assertEqual(system.state, SystemState.STARTING)
        
        system.state = SystemState.RUNNING
        self.assertEqual(system.state, SystemState.RUNNING)
        
        system.state = SystemState.PAUSED
        self.assertEqual(system.state, SystemState.PAUSED)
        
        system.state = SystemState.STOPPING
        self.assertEqual(system.state, SystemState.STOPPING)
        
        system.state = SystemState.STOPPED
        self.assertEqual(system.state, SystemState.STOPPED)
    
    @patch('src.core.trading_system.MT5Connection')
    @patch('src.core.trading_system.DataPipeline')
    @patch('src.core.trading_system.MonitoringSystem')
    @patch('src.core.trading_system.AlertSystem')
    def test_system_initialization(self, mock_alert, mock_monitoring, mock_pipeline, mock_mt5):
        """测试系统初始化"""
        # 模拟MT5连接
        mock_mt5_instance = Mock()
        mock_mt5_instance.is_connected = True
        mock_mt5.return_value = mock_mt5_instance
        
        # 模拟监控系统
        mock_monitoring_instance = Mock()
        mock_monitoring_instance.start = asyncio.coroutine(lambda: None)
        mock_monitoring.return_value = mock_monitoring_instance
        
        # 模拟告警系统
        mock_alert_instance = Mock()
        mock_alert_instance.initialize = asyncio.coroutine(lambda: None)
        mock_alert_instance.send_alert = asyncio.coroutine(lambda *args, **kwargs: None)
        mock_alert.return_value = mock_alert_instance
        
        system = TradingSystem()
        
        # 运行初始化
        result = asyncio.run(system.initialize())
        
        self.assertTrue(result)
        self.assertIsNotNone(system.stats.start_time)
        self.assertIsNotNone(system.mt5_connection)
        self.assertIsNotNone(system.data_pipeline)
        self.assertIsNotNone(system.monitoring)
        self.assertIsNotNone(system.alert_system)
    
    def test_system_stats_initialization(self):
        """测试系统统计初始化"""
        system = TradingSystem()
        stats = system.stats
        
        self.assertIsNone(stats.start_time)
        self.assertEqual(stats.uptime, timedelta())
        self.assertEqual(stats.total_signals, 0)
        self.assertEqual(stats.total_trades, 0)
        self.assertEqual(stats.successful_trades, 0)
        self.assertEqual(stats.failed_trades, 0)
        self.assertEqual(stats.total_profit, 0.0)
        self.assertEqual(stats.current_drawdown, 0.0)
        self.assertEqual(stats.max_drawdown, 0.0)
        self.assertEqual(stats.errors_count, 0)
        self.assertEqual(stats.recovery_count, 0)
    
    def test_pause_and_resume(self):
        """测试暂停和恢复功能"""
        system = TradingSystem()
        
        # 暂停系统
        system.pause()
        self.assertEqual(system.state, SystemState.PAUSED)
        self.assertTrue(system._pause_event.is_set())
        
        # 恢复系统
        system.resume()
        self.assertEqual(system.state, SystemState.RUNNING)
        self.assertFalse(system._pause_event.is_set())
    
    def test_get_status(self):
        """测试获取系统状态"""
        system = TradingSystem()
        system.stats.start_time = datetime.now()
        system.stats.total_trades = 10
        system.stats.successful_trades = 8
        system.stats.failed_trades = 2
        
        status = system.get_status()
        
        self.assertIn('state', status)
        self.assertIn('stats', status)
        self.assertIn('components', status)
        
        self.assertEqual(status['stats']['total_trades'], 10)
        self.assertEqual(status['stats']['successful_trades'], 8)
        self.assertEqual(status['stats']['failed_trades'], 2)
        self.assertEqual(status['stats']['success_rate'], 0.8)
    
    def test_collect_metrics(self):
        """测试收集系统指标"""
        system = TradingSystem()
        system.stats.start_time = datetime.now()
        system.stats.total_signals = 50
        system.stats.total_trades = 20
        system.stats.successful_trades = 15
        
        metrics = system._collect_metrics()
        
        self.assertIn('timestamp', metrics)
        self.assertIn('state', metrics)
        self.assertIn('uptime', metrics)
        self.assertIn('total_signals', metrics)
        self.assertIn('total_trades', metrics)
        self.assertIn('success_rate', metrics)
        
        self.assertEqual(metrics['total_signals'], 50)
        self.assertEqual(metrics['total_trades'], 20)
        self.assertEqual(metrics['successful_trades'], 15)
        self.assertEqual(metrics['success_rate'], 0.75)
    
    def test_is_trading_time(self):
        """测试交易时间检查"""
        system = TradingSystem()
        
        # 如果未启用交易时间限制，应该总是返回True
        if not system.config.trading_hours.enabled:
            self.assertTrue(system._is_trading_time())
    
    @patch('src.core.trading_system.MT5Connection')
    def test_mt5_connection_initialization(self, mock_mt5):
        """测试MT5连接初始化"""
        mock_mt5_instance = Mock()
        mock_mt5_instance.is_connected = True
        mock_mt5_instance.connect = Mock()
        mock_mt5.return_value = mock_mt5_instance
        
        system = TradingSystem()
        connection = system._initialize_mt5_connection()
        
        self.assertIsNotNone(connection)
        mock_mt5_instance.connect.assert_called_once()
    
    def test_update_stats(self):
        """测试统计更新"""
        system = TradingSystem()
        
        # 模拟仓位管理器
        system.position_manager = Mock()
        system.position_manager.positions = {}
        
        # 模拟资金管理器
        system.fund_manager = Mock()
        system.fund_manager.get_account_info = Mock(return_value={
            'equity': 9500,
            'balance': 10000
        })
        
        system._update_stats()
        
        # 验证回撤计算
        self.assertGreater(system.stats.current_drawdown, 0)
        self.assertGreater(system.stats.max_drawdown, 0)


class TestSystemIntegration(unittest.TestCase):
    """系统集成测试"""
    
    @patch('src.core.trading_system.MT5Connection')
    @patch('src.core.trading_system.DataPipeline')
    @patch('src.core.trading_system.StrategyManager')
    @patch('src.core.trading_system.OrderExecutor')
    @patch('src.core.trading_system.PositionManager')
    @patch('src.core.trading_system.RiskControlSystem')
    @patch('src.core.trading_system.MonitoringSystem')
    @patch('src.core.trading_system.AlertSystem')
    def test_full_system_integration(self, mock_alert, mock_monitoring, mock_risk,
                                     mock_position, mock_order, mock_strategy,
                                     mock_pipeline, mock_mt5):
        """测试完整系统集成"""
        # 设置所有模拟对象
        mock_mt5_instance = Mock()
        mock_mt5_instance.is_connected = True
        mock_mt5_instance.connect = Mock()
        mock_mt5.return_value = mock_mt5_instance
        
        mock_monitoring_instance = Mock()
        mock_monitoring_instance.start = asyncio.coroutine(lambda: None)
        mock_monitoring_instance.stop = asyncio.coroutine(lambda: None)
        mock_monitoring.return_value = mock_monitoring_instance
        
        mock_alert_instance = Mock()
        mock_alert_instance.initialize = asyncio.coroutine(lambda: None)
        mock_alert_instance.send_alert = asyncio.coroutine(lambda *args, **kwargs: None)
        mock_alert_instance.close = asyncio.coroutine(lambda: None)
        mock_alert.return_value = mock_alert_instance
        
        # 创建系统
        system = TradingSystem()
        
        # 初始化
        result = asyncio.run(system.initialize())
        self.assertTrue(result)
        
        # 验证所有组件都已创建
        self.assertIsNotNone(system.mt5_connection)
        self.assertIsNotNone(system.data_pipeline)
        self.assertIsNotNone(system.strategy_manager)
        self.assertIsNotNone(system.order_executor)
        self.assertIsNotNone(system.position_manager)
        self.assertIsNotNone(system.risk_control)
        self.assertIsNotNone(system.monitoring)
        self.assertIsNotNone(system.alert_system)
        
        # 关闭系统
        asyncio.run(system.shutdown())
        self.assertEqual(system.state, SystemState.STOPPED)


if __name__ == '__main__':
    unittest.main()
