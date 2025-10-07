"""
品种配置管理器测试模块
"""

import unittest
import os
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import yaml

from src.core.symbol_config_manager import (
    SymbolConfigManager,
    SymbolConfig,
    TradingHours,
    RiskParams,
    EIAMonitoring,
    OPECMonitoring,
    EventMonitoring
)


class TestTradingHours(unittest.TestCase):
    """交易时段测试"""
    
    def test_is_trading_time_weekday(self):
        """测试工作日交易时间"""
        trading_hours = TradingHours(
            monday="09:00-17:00",
            tuesday="09:00-17:00"
        )
        
        # 周一 10:00 - 应该在交易时间内
        test_time = datetime(2024, 1, 1, 10, 0)  # Monday
        self.assertTrue(trading_hours.is_trading_time(test_time))
        
        # 周一 08:00 - 应该不在交易时间内
        test_time = datetime(2024, 1, 1, 8, 0)
        self.assertFalse(trading_hours.is_trading_time(test_time))
        
        # 周一 18:00 - 应该不在交易时间内
        test_time = datetime(2024, 1, 1, 18, 0)
        self.assertFalse(trading_hours.is_trading_time(test_time))
    
    def test_is_trading_time_weekend(self):
        """测试周末交易时间"""
        trading_hours = TradingHours()
        
        # 周六 - 应该不在交易时间内
        test_time = datetime(2024, 1, 6, 10, 0)  # Saturday
        self.assertFalse(trading_hours.is_trading_time(test_time))


class TestSymbolConfig(unittest.TestCase):
    """品种配置测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.config = SymbolConfig(
            symbol="EURUSD",
            spread_limit=2.0,
            min_lot=0.01,
            max_lot=10.0,
            strategies=["ma_cross", "macd"],
            timeframes=["M5", "H1"],
            trading_hours=TradingHours(monday="00:00-23:59"),
            risk_params=RiskParams(stop_loss_pips=20, take_profit_pips=40)
        )
    
    def test_is_trading_time(self):
        """测试交易时间检查"""
        # 周一应该可以交易
        test_time = datetime(2024, 1, 1, 10, 0)  # Monday
        self.assertTrue(self.config.is_trading_time(test_time))
    
    def test_get_optimized_params(self):
        """测试获取优化参数"""
        self.config.optimize_params = {
            'ma_fast_period': (5, 50, 5),
            'ma_slow_period': (20, 200, 10),
            'macd_fast': (8, 16, 2)
        }
        
        ma_params = self.config.get_optimized_params('ma')
        self.assertIn('ma_fast_period', ma_params)
        self.assertIn('ma_slow_period', ma_params)
        
        macd_params = self.config.get_optimized_params('macd')
        self.assertIn('macd_fast', macd_params)
    
    def test_eia_monitoring(self):
        """测试EIA库存数据监控"""
        self.config.eia_monitoring = EIAMonitoring(
            enabled=True,
            release_time="15:30",
            pre_release_stop=30,
            post_release_wait=15
        )
        
        # 周三 15:00 - 应该停止交易（发布前30分钟）
        test_time = datetime(2024, 1, 3, 15, 0)  # Wednesday
        should_stop, reason = self.config.should_stop_trading_for_event(test_time)
        self.assertTrue(should_stop)
        self.assertIn("EIA", reason)
        
        # 周三 14:00 - 应该可以交易
        test_time = datetime(2024, 1, 3, 14, 0)
        should_stop, _ = self.config.should_stop_trading_for_event(test_time)
        self.assertFalse(should_stop)
    
    def test_opec_monitoring(self):
        """测试OPEC会议监控"""
        self.config.opec_monitoring = OPECMonitoring(
            enabled=True,
            meeting_dates=["2024-01-15"],
            pre_meeting_stop=120,  # 2小时
            post_meeting_wait=60   # 1小时
        )
        
        # 会议前1小时 - 应该停止交易
        test_time = datetime(2024, 1, 14, 23, 0)
        should_stop, reason = self.config.should_stop_trading_for_event(test_time)
        self.assertTrue(should_stop)
        self.assertIn("OPEC", reason)
        
        # 会议前3小时 - 应该可以交易
        test_time = datetime(2024, 1, 14, 21, 0)
        should_stop, _ = self.config.should_stop_trading_for_event(test_time)
        self.assertFalse(should_stop)


class TestSymbolConfigManager(unittest.TestCase):
    """品种配置管理器测试"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建临时配置目录
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / "symbols"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建测试配置文件
        self._create_test_configs()
        
        # 初始化管理器
        self.manager = SymbolConfigManager(str(self.config_dir))
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_configs(self):
        """创建测试配置文件"""
        # EURUSD配置
        eurusd_config = {
            'symbol': 'EURUSD',
            'spread_limit': 2.0,
            'min_lot': 0.01,
            'max_lot': 10.0,
            'lot_step': 0.01,
            'risk_multiplier': 1.0,
            'strategies': ['ma_cross', 'macd'],
            'timeframes': ['M5', 'H1'],
            'trading_hours': {
                'monday': '00:00-23:59',
                'tuesday': '00:00-23:59',
                'wednesday': '00:00-23:59',
                'thursday': '00:00-23:59',
                'friday': '00:00-21:00'
            },
            'optimize_params': {
                'ma_fast_period': [5, 50, 5],
                'ma_slow_period': [20, 200, 10]
            },
            'risk_params': {
                'max_spread': 3.0,
                'min_equity': 1000.0,
                'max_slippage': 3,
                'stop_loss_pips': 20,
                'take_profit_pips': 40
            }
        }
        
        # XAUUSD配置
        xauusd_config = {
            'symbol': 'XAUUSD',
            'spread_limit': 5.0,
            'min_lot': 0.01,
            'max_lot': 5.0,
            'lot_step': 0.01,
            'risk_multiplier': 0.8,
            'strategies': ['breakout'],
            'timeframes': ['H1', 'H4'],
            'trading_hours': {
                'monday': '07:00-23:59',
                'tuesday': '00:00-23:59',
                'wednesday': '00:00-23:59',
                'thursday': '00:00-23:59',
                'friday': '00:00-21:00'
            },
            'optimize_params': {},
            'risk_params': {
                'max_spread': 8.0,
                'min_equity': 2000.0,
                'max_slippage': 5,
                'stop_loss_pips': 100,
                'take_profit_pips': 200
            },
            'event_monitoring': ['FOMC', 'NFP', 'CPI']
        }
        
        # USOIL配置
        usoil_config = {
            'symbol': 'USOIL',
            'spread_limit': 4.0,
            'min_lot': 0.01,
            'max_lot': 3.0,
            'lot_step': 0.01,
            'risk_multiplier': 0.7,
            'strategies': ['news_trading'],
            'timeframes': ['H1', 'H4'],
            'trading_hours': {
                'monday': '13:00-23:59',
                'tuesday': '00:00-23:59',
                'wednesday': '00:00-23:59',
                'thursday': '00:00-23:59',
                'friday': '00:00-20:00'
            },
            'optimize_params': {},
            'risk_params': {
                'max_spread': 6.0,
                'min_equity': 3000.0,
                'max_slippage': 8,
                'stop_loss_pips': 150,
                'take_profit_pips': 300
            },
            'eia_monitoring': {
                'enabled': True,
                'release_time': '15:30',
                'pre_release_stop': 30,
                'post_release_wait': 15
            },
            'opec_monitoring': {
                'enabled': True,
                'meeting_dates': ['2024-06-01', '2024-12-01'],
                'pre_meeting_stop': 120,
                'post_meeting_wait': 60
            }
        }
        
        # 保存配置文件
        with open(self.config_dir / 'eurusd.yaml', 'w') as f:
            yaml.dump(eurusd_config, f)
        
        with open(self.config_dir / 'xauusd.yaml', 'w') as f:
            yaml.dump(xauusd_config, f)
        
        with open(self.config_dir / 'usoil.yaml', 'w') as f:
            yaml.dump(usoil_config, f)
    
    def test_load_all_configs(self):
        """测试加载所有配置"""
        self.assertEqual(len(self.manager.configs), 3)
        self.assertIn('EURUSD', self.manager.configs)
        self.assertIn('XAUUSD', self.manager.configs)
        self.assertIn('USOIL', self.manager.configs)
    
    def test_get_config(self):
        """测试获取品种配置"""
        config = self.manager.get_config('EURUSD')
        self.assertIsNotNone(config)
        self.assertEqual(config.symbol, 'EURUSD')
        self.assertEqual(config.spread_limit, 2.0)
        
        # 测试不存在的品种
        config = self.manager.get_config('INVALID')
        self.assertIsNone(config)
    
    def test_get_all_symbols(self):
        """测试获取所有品种"""
        symbols = self.manager.get_all_symbols()
        self.assertEqual(len(symbols), 3)
        self.assertIn('EURUSD', symbols)
        self.assertIn('XAUUSD', symbols)
        self.assertIn('USOIL', symbols)
    
    def test_is_symbol_configured(self):
        """测试检查品种是否已配置"""
        self.assertTrue(self.manager.is_symbol_configured('EURUSD'))
        self.assertFalse(self.manager.is_symbol_configured('INVALID'))
    
    def test_get_active_symbols(self):
        """测试获取活跃品种"""
        # 周一 10:00 UTC
        test_time = datetime(2024, 1, 1, 10, 0)
        active_symbols = self.manager.get_active_symbols(test_time)
        
        # EURUSD应该活跃（00:00-23:59）
        self.assertIn('EURUSD', active_symbols)
        
        # XAUUSD应该活跃（07:00开始，现在是10:00）
        self.assertIn('XAUUSD', active_symbols)
        
        # USOIL不应该活跃（13:00开始）
        self.assertNotIn('USOIL', active_symbols)
        
        # 测试早上6点 - XAUUSD不应该活跃
        test_time_early = datetime(2024, 1, 1, 6, 0)
        active_symbols_early = self.manager.get_active_symbols(test_time_early)
        self.assertNotIn('XAUUSD', active_symbols_early)
    
    def test_get_risk_multiplier(self):
        """测试获取风险倍数"""
        self.assertEqual(self.manager.get_risk_multiplier('EURUSD'), 1.0)
        self.assertEqual(self.manager.get_risk_multiplier('XAUUSD'), 0.8)
        self.assertEqual(self.manager.get_risk_multiplier('USOIL'), 0.7)
        self.assertEqual(self.manager.get_risk_multiplier('INVALID'), 1.0)
    
    def test_get_strategies_for_symbol(self):
        """测试获取品种策略"""
        strategies = self.manager.get_strategies_for_symbol('EURUSD')
        self.assertEqual(len(strategies), 2)
        self.assertIn('ma_cross', strategies)
        self.assertIn('macd', strategies)
        
        strategies = self.manager.get_strategies_for_symbol('INVALID')
        self.assertEqual(len(strategies), 0)
    
    def test_validate_lot_size(self):
        """测试验证手数"""
        # 正常手数
        valid, adjusted = self.manager.validate_lot_size('EURUSD', 0.5)
        self.assertTrue(valid)
        self.assertEqual(adjusted, 0.5)
        
        # 超过最大手数
        valid, adjusted = self.manager.validate_lot_size('EURUSD', 15.0)
        self.assertTrue(valid)
        self.assertEqual(adjusted, 10.0)  # 应该被限制到max_lot
        
        # 小于最小手数
        valid, adjusted = self.manager.validate_lot_size('EURUSD', 0.005)
        self.assertTrue(valid)
        self.assertEqual(adjusted, 0.01)  # 应该被调整到min_lot
        
        # 不是lot_step的倍数
        valid, adjusted = self.manager.validate_lot_size('EURUSD', 0.123)
        self.assertTrue(valid)
        self.assertAlmostEqual(adjusted, 0.12, places=2)
        
        # 无效品种
        valid, adjusted = self.manager.validate_lot_size('INVALID', 0.5)
        self.assertFalse(valid)
        self.assertEqual(adjusted, 0.0)
    
    def test_check_spread(self):
        """测试检查点差"""
        # 正常点差
        self.assertTrue(self.manager.check_spread('EURUSD', 1.5))
        
        # 超过限制
        self.assertFalse(self.manager.check_spread('EURUSD', 3.5))
        
        # 边界值
        self.assertTrue(self.manager.check_spread('EURUSD', 2.0))
        
        # 无效品种
        self.assertFalse(self.manager.check_spread('INVALID', 1.0))
    
    def test_save_config(self):
        """测试保存配置"""
        # 创建新配置
        new_config = SymbolConfig(
            symbol='GBPUSD',
            spread_limit=2.5,
            min_lot=0.01,
            max_lot=8.0,
            strategies=['momentum'],
            timeframes=['M15', 'H1'],
            trading_hours=TradingHours(),
            risk_params=RiskParams()
        )
        
        # 保存配置
        success = self.manager.save_config('GBPUSD', new_config)
        self.assertTrue(success)
        
        # 验证文件已创建
        config_file = self.config_dir / 'gbpusd.yaml'
        self.assertTrue(config_file.exists())
        
        # 验证可以重新加载
        loaded_config = self.manager.get_config('GBPUSD')
        self.assertIsNotNone(loaded_config)
        self.assertEqual(loaded_config.symbol, 'GBPUSD')
        self.assertEqual(loaded_config.spread_limit, 2.5)
    
    def test_create_default_config(self):
        """测试创建默认配置"""
        default_config = self.manager.create_default_config('TESTPAIR')
        
        self.assertEqual(default_config.symbol, 'TESTPAIR')
        self.assertEqual(default_config.spread_limit, 3.0)
        self.assertEqual(default_config.min_lot, 0.01)
        self.assertEqual(default_config.max_lot, 10.0)
        self.assertIn('ma_cross', default_config.strategies)
    
    def test_event_monitoring_integration(self):
        """测试事件监控集成"""
        # 测试XAUUSD的事件监控
        config = self.manager.get_config('XAUUSD')
        self.assertIsNotNone(config.event_monitoring)
        self.assertIn('FOMC', config.event_monitoring.events)
        
        # 测试USOIL的EIA监控
        config = self.manager.get_config('USOIL')
        self.assertIsNotNone(config.eia_monitoring)
        self.assertTrue(config.eia_monitoring.enabled)
        
        # 测试USOIL的OPEC监控
        self.assertIsNotNone(config.opec_monitoring)
        self.assertTrue(config.opec_monitoring.enabled)
        self.assertEqual(len(config.opec_monitoring.meeting_dates), 2)


if __name__ == '__main__':
    unittest.main()
