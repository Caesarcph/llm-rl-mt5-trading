"""
策略管理器测试用例
测试策略注册、加载、信号聚合和冲突解决功能
"""

import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.strategies.strategy_manager import (
    StrategyManager, ConflictResolutionMethod, SignalAggregationMethod,
    StrategyRegistration
)
from src.strategies.base_strategies import (
    BaseStrategy, StrategyConfig, StrategyType,
    TrendFollowingStrategy, MeanReversionStrategy, BreakoutStrategy
)
from src.core.models import MarketData, Signal


class TestStrategyManager(unittest.TestCase):
    """策略管理器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.strategy_manager = StrategyManager(
            conflict_resolution=ConflictResolutionMethod.HIGHEST_STRENGTH,
            aggregation_method=SignalAggregationMethod.WEIGHTED_AVERAGE
        )
        
        # 创建测试市场数据
        self.market_data = self._create_test_market_data()
    
    def _create_test_market_data(self) -> MarketData:
        """创建测试市场数据"""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='1H')
        
        # 生成模拟价格数据
        np.random.seed(42)
        close_prices = 1.1000 + np.cumsum(np.random.randn(100) * 0.0001)
        
        ohlcv = pd.DataFrame({
            'time': dates,
            'open': close_prices + np.random.randn(100) * 0.0001,
            'high': close_prices + abs(np.random.randn(100) * 0.0002),
            'low': close_prices - abs(np.random.randn(100) * 0.0002),
            'close': close_prices,
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        return MarketData(
            symbol="EURUSD",
            timeframe="H1",
            timestamp=datetime.now(),
            ohlcv=ohlcv,
            indicators={},
            spread=0.00002
        )
    
    def test_register_strategy(self):
        """测试策略注册"""
        config = StrategyConfig(
            name="test_trend",
            strategy_type=StrategyType.TREND_FOLLOWING,
            enabled=True
        )
        
        result = self.strategy_manager.register_strategy(
            name="test_trend",
            strategy_class=TrendFollowingStrategy,
            config=config,
            weight=1.0,
            auto_load=True
        )
        
        self.assertTrue(result)
        self.assertIn("test_trend", self.strategy_manager.strategy_registrations)
        self.assertIn("test_trend", self.strategy_manager.strategies)
    
    def test_load_strategy(self):
        """测试策略加载"""
        config = StrategyConfig(
            name="test_mean_reversion",
            strategy_type=StrategyType.MEAN_REVERSION,
            enabled=True
        )
        
        # 先注册但不自动加载
        self.strategy_manager.register_strategy(
            name="test_mean_reversion",
            strategy_class=MeanReversionStrategy,
            config=config,
            auto_load=False
        )
        
        # 手动加载
        result = self.strategy_manager.load_strategy("test_mean_reversion")
        
        self.assertTrue(result)
        self.assertIn("test_mean_reversion", self.strategy_manager.strategies)
    
    def test_unload_strategy(self):
        """测试策略卸载"""
        config = StrategyConfig(
            name="test_breakout",
            strategy_type=StrategyType.BREAKOUT,
            enabled=True
        )
        
        self.strategy_manager.register_strategy(
            name="test_breakout",
            strategy_class=BreakoutStrategy,
            config=config,
            auto_load=True
        )
        
        # 卸载策略
        result = self.strategy_manager.unload_strategy("test_breakout")
        
        self.assertTrue(result)
        self.assertNotIn("test_breakout", self.strategy_manager.strategies)
        # 注册信息应该还在
        self.assertIn("test_breakout", self.strategy_manager.strategy_registrations)
    
    def test_unregister_strategy(self):
        """测试策略注销"""
        config = StrategyConfig(
            name="test_strategy",
            strategy_type=StrategyType.TREND_FOLLOWING,
            enabled=True
        )
        
        self.strategy_manager.register_strategy(
            name="test_strategy",
            strategy_class=TrendFollowingStrategy,
            config=config
        )
        
        # 注销策略
        result = self.strategy_manager.unregister_strategy("test_strategy")
        
        self.assertTrue(result)
        self.assertNotIn("test_strategy", self.strategy_manager.strategies)
        self.assertNotIn("test_strategy", self.strategy_manager.strategy_registrations)
    
    def test_generate_signals(self):
        """测试信号生成"""
        # 注册多个策略
        strategies = [
            ("trend_strategy", TrendFollowingStrategy, StrategyType.TREND_FOLLOWING),
            ("mean_reversion", MeanReversionStrategy, StrategyType.MEAN_REVERSION),
            ("breakout", BreakoutStrategy, StrategyType.BREAKOUT)
        ]
        
        for name, strategy_class, strategy_type in strategies:
            config = StrategyConfig(
                name=name,
                strategy_type=strategy_type,
                enabled=True,
                min_signal_strength=0.3
            )
            
            self.strategy_manager.register_strategy(
                name=name,
                strategy_class=strategy_class,
                config=config,
                weight=1.0
            )
        
        # 生成信号
        signals = self.strategy_manager.generate_signals(self.market_data)
        
        # 验证信号
        self.assertIsInstance(signals, list)
        # 至少应该有一些信号（取决于市场数据）
        for signal in signals:
            self.assertIsInstance(signal, Signal)
            self.assertEqual(signal.symbol, "EURUSD")
    
    def test_detect_conflicts(self):
        """测试冲突检测"""
        # 创建冲突信号
        signal1 = Signal(
            strategy_id="strategy1",
            symbol="EURUSD",
            direction=1,  # 买入
            strength=0.8,
            entry_price=1.1000,
            sl=1.0950,
            tp=1.1100,
            size=0.1,
            confidence=0.7,
            timestamp=datetime.now()
        )
        
        signal2 = Signal(
            strategy_id="strategy2",
            symbol="EURUSD",
            direction=-1,  # 卖出
            strength=0.6,
            entry_price=1.1000,
            sl=1.1050,
            tp=1.0900,
            size=0.1,
            confidence=0.6,
            timestamp=datetime.now()
        )
        
        # 检测冲突
        conflicts = self.strategy_manager._detect_conflicts([signal1, signal2])
        self.assertTrue(conflicts)
        
        # 无冲突情况
        signal3 = Signal(
            strategy_id="strategy3",
            symbol="EURUSD",
            direction=1,  # 同向
            strength=0.7,
            entry_price=1.1000,
            sl=1.0950,
            tp=1.1100,
            size=0.1,
            confidence=0.65,
            timestamp=datetime.now()
        )
        
        no_conflicts = self.strategy_manager._detect_conflicts([signal1, signal3])
        self.assertFalse(no_conflicts)
    
    def test_resolve_conflicts_highest_strength(self):
        """测试冲突解决 - 最高强度"""
        self.strategy_manager.conflict_resolution = ConflictResolutionMethod.HIGHEST_STRENGTH
        
        signals = [
            Signal("s1", "EURUSD", 1, 0.6, 1.1000, 1.0950, 1.1100, 0.1, 0.7, datetime.now()),
            Signal("s2", "EURUSD", -1, 0.8, 1.1000, 1.1050, 1.0900, 0.1, 0.6, datetime.now()),
            Signal("s3", "EURUSD", 1, 0.5, 1.1000, 1.0950, 1.1100, 0.1, 0.65, datetime.now())
        ]
        
        result = self.strategy_manager._resolve_conflicts(signals)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.strength, 0.8)
        self.assertEqual(result.strategy_id, "s2")
    
    def test_resolve_conflicts_highest_confidence(self):
        """测试冲突解决 - 最高置信度"""
        self.strategy_manager.conflict_resolution = ConflictResolutionMethod.HIGHEST_CONFIDENCE
        
        signals = [
            Signal("s1", "EURUSD", 1, 0.6, 1.1000, 1.0950, 1.1100, 0.1, 0.7, datetime.now()),
            Signal("s2", "EURUSD", -1, 0.8, 1.1000, 1.1050, 1.0900, 0.1, 0.6, datetime.now()),
            Signal("s3", "EURUSD", 1, 0.5, 1.1000, 1.0950, 1.1100, 0.1, 0.75, datetime.now())
        ]
        
        result = self.strategy_manager._resolve_conflicts(signals)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.confidence, 0.75)
        self.assertEqual(result.strategy_id, "s3")
    
    def test_resolve_conflicts_cancel_all(self):
        """测试冲突解决 - 取消所有"""
        self.strategy_manager.conflict_resolution = ConflictResolutionMethod.CANCEL_ALL
        
        signals = [
            Signal("s1", "EURUSD", 1, 0.6, 1.1000, 1.0950, 1.1100, 0.1, 0.7, datetime.now()),
            Signal("s2", "EURUSD", -1, 0.8, 1.1000, 1.1050, 1.0900, 0.1, 0.6, datetime.now())
        ]
        
        result = self.strategy_manager._resolve_conflicts(signals)
        
        self.assertIsNone(result)
    
    def test_aggregate_signals_simple_average(self):
        """测试信号聚合 - 简单平均"""
        self.strategy_manager.aggregation_method = SignalAggregationMethod.SIMPLE_AVERAGE
        
        signals = [
            Signal("s1", "EURUSD", 1, 0.6, 1.1000, 1.0950, 1.1100, 0.1, 0.7, datetime.now()),
            Signal("s2", "EURUSD", 1, 0.8, 1.1010, 1.0960, 1.1110, 0.15, 0.8, datetime.now()),
            Signal("s3", "EURUSD", 1, 0.7, 1.0990, 1.0940, 1.1090, 0.12, 0.75, datetime.now())
        ]
        
        result = self.strategy_manager._simple_average_signal(signals)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.direction, 1)
        self.assertAlmostEqual(result.strength, 0.7, places=2)
        self.assertAlmostEqual(result.confidence, 0.75, places=2)
    
    def test_aggregate_signals_weighted_average(self):
        """测试信号聚合 - 加权平均"""
        signals = [
            Signal("s1", "EURUSD", 1, 0.6, 1.1000, 1.0950, 1.1100, 0.1, 0.5, datetime.now()),
            Signal("s2", "EURUSD", 1, 0.8, 1.1010, 1.0960, 1.1110, 0.15, 0.9, datetime.now()),
            Signal("s3", "EURUSD", 1, 0.7, 1.0990, 1.0940, 1.1090, 0.12, 0.6, datetime.now())
        ]
        
        result = self.strategy_manager._weighted_average_signal(signals)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.direction, 1)
        # 加权平均应该更接近高置信度的信号
        self.assertGreater(result.strength, 0.7)
    
    def test_aggregate_signals_majority_vote(self):
        """测试信号聚合 - 多数投票"""
        signals = [
            Signal("s1", "EURUSD", 1, 0.6, 1.1000, 1.0950, 1.1100, 0.1, 0.7, datetime.now()),
            Signal("s2", "EURUSD", 1, 0.8, 1.1010, 1.0960, 1.1110, 0.15, 0.8, datetime.now()),
            Signal("s3", "EURUSD", -1, 0.9, 1.1000, 1.1050, 1.0900, 0.1, 0.85, datetime.now())
        ]
        
        result = self.strategy_manager._majority_vote_signal(signals)
        
        self.assertIsNotNone(result)
        # 多数是买入信号
        self.assertEqual(result.direction, 1)
    
    def test_aggregate_signals_full_flow(self):
        """测试完整的信号聚合流程"""
        signals = [
            Signal("s1", "EURUSD", 1, 0.6, 1.1000, 1.0950, 1.1100, 0.1, 0.7, datetime.now()),
            Signal("s2", "EURUSD", 1, 0.8, 1.1010, 1.0960, 1.1110, 0.15, 0.8, datetime.now())
        ]
        
        result = self.strategy_manager.aggregate_signals(signals, symbol="EURUSD")
        
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.aggregated_signal)
        self.assertEqual(len(result.original_signals), 2)
        self.assertFalse(result.conflicts_detected)
        self.assertEqual(result.aggregation_metadata['num_signals'], 2)
    
    def test_update_strategy_weights(self):
        """测试策略权重更新"""
        config = StrategyConfig(
            name="test_strategy",
            strategy_type=StrategyType.TREND_FOLLOWING
        )
        
        self.strategy_manager.register_strategy(
            name="test_strategy",
            strategy_class=TrendFollowingStrategy,
            config=config,
            weight=1.0
        )
        
        # 更新权重
        self.strategy_manager.update_strategy_weights({
            "test_strategy": 1.5
        })
        
        self.assertEqual(self.strategy_manager.strategy_weights["test_strategy"], 1.5)
    
    def test_enable_disable_strategy(self):
        """测试启用/禁用策略"""
        config = StrategyConfig(
            name="test_strategy",
            strategy_type=StrategyType.TREND_FOLLOWING
        )
        
        self.strategy_manager.register_strategy(
            name="test_strategy",
            strategy_class=TrendFollowingStrategy,
            config=config
        )
        
        # 禁用策略
        self.strategy_manager.disable_strategy("test_strategy")
        self.assertFalse(self.strategy_manager.strategies["test_strategy"].is_enabled)
        
        # 启用策略
        self.strategy_manager.enable_strategy("test_strategy")
        self.assertTrue(self.strategy_manager.strategies["test_strategy"].is_enabled)
    
    def test_get_active_strategies(self):
        """测试获取活跃策略"""
        strategies = [
            ("strategy1", True),
            ("strategy2", False),
            ("strategy3", True)
        ]
        
        for name, enabled in strategies:
            config = StrategyConfig(
                name=name,
                strategy_type=StrategyType.TREND_FOLLOWING,
                enabled=True  # Config always enabled
            )
            
            self.strategy_manager.register_strategy(
                name=name,
                strategy_class=TrendFollowingStrategy,
                config=config,
                enabled=enabled  # But registration controls actual enabled state
            )
        
        active = self.strategy_manager.get_active_strategies()
        
        self.assertEqual(len(active), 2)
        self.assertIn("strategy1", active)
        self.assertIn("strategy3", active)
        self.assertNotIn("strategy2", active)
    
    def test_get_strategy_info(self):
        """测试获取策略信息"""
        config = StrategyConfig(
            name="test_strategy",
            strategy_type=StrategyType.TREND_FOLLOWING
        )
        
        self.strategy_manager.register_strategy(
            name="test_strategy",
            strategy_class=TrendFollowingStrategy,
            config=config,
            weight=1.2,
            metadata={"test_key": "test_value"}
        )
        
        info = self.strategy_manager.get_strategy_info("test_strategy")
        
        self.assertIsNotNone(info)
        self.assertEqual(info['name'], "test_strategy")
        self.assertEqual(info['weight'], 1.2)
        self.assertIn('config', info)
        self.assertIn('performance', info)
        self.assertEqual(info['metadata']['test_key'], "test_value")
    
    def test_signal_history(self):
        """测试信号历史记录"""
        signal = Signal(
            strategy_id="test",
            symbol="EURUSD",
            direction=1,
            strength=0.8,
            entry_price=1.1000,
            sl=1.0950,
            tp=1.1100,
            size=0.1,
            confidence=0.7,
            timestamp=datetime.now()
        )
        
        self.strategy_manager._add_to_signal_history(signal)
        
        history = self.strategy_manager.get_signal_history()
        
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].strategy_id, "test")
        
        # 测试过滤
        signal2 = Signal(
            strategy_id="test2",
            symbol="GBPUSD",
            direction=-1,
            strength=0.6,
            entry_price=1.2500,
            sl=1.2550,
            tp=1.2400,
            size=0.1,
            confidence=0.65,
            timestamp=datetime.now()
        )
        
        self.strategy_manager._add_to_signal_history(signal2)
        
        # 按品种过滤
        eurusd_history = self.strategy_manager.get_signal_history(symbol="EURUSD")
        self.assertEqual(len(eurusd_history), 1)
        self.assertEqual(eurusd_history[0].symbol, "EURUSD")
        
        # 按策略过滤
        test2_history = self.strategy_manager.get_signal_history(strategy_id="test2")
        self.assertEqual(len(test2_history), 1)
        self.assertEqual(test2_history[0].strategy_id, "test2")


if __name__ == '__main__':
    unittest.main()
