"""
RL策略优化器测试
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime

from src.rl.trading_env import TradingEnvironment
from src.rl.rl_trainer import RLTrainer, TrainingConfig
from src.rl.rl_strategy_optimizer import RLStrategyOptimizer, RLOptimizerConfig
from src.core.models import Signal, MarketData


class TestRLStrategyOptimizer(unittest.TestCase):
    """测试RL策略优化器"""
    
    @classmethod
    def setUpClass(cls):
        """类级别设置"""
        # 创建测试数据
        dates = pd.date_range(start='2024-01-01', periods=500, freq='h')
        np.random.seed(42)
        
        close_prices = 100 + np.cumsum(np.random.randn(500) * 0.5)
        
        cls.test_data = pd.DataFrame({
            'open': close_prices + np.random.randn(500) * 0.1,
            'high': close_prices + np.abs(np.random.randn(500) * 0.2),
            'low': close_prices - np.abs(np.random.randn(500) * 0.2),
            'close': close_prices,
            'volume': np.random.randint(1000, 10000, 500),
            'sma_20': close_prices,
            'rsi': np.random.uniform(30, 70, 500),
            'macd': np.random.randn(500) * 0.5
        }, index=dates)
        
        # 创建环境和训练器
        cls.env = TradingEnvironment(
            symbol='EURUSD',
            data=cls.test_data,
            initial_balance=10000.0
        )
        
        cls.config = TrainingConfig(algorithm="PPO", total_timesteps=1000)
        cls.trainer = RLTrainer(cls.env, cls.config)
    
    @classmethod
    def tearDownClass(cls):
        """类级别清理"""
        cls.trainer.train_env.close()
        cls.trainer.eval_env.close()
        cls.env.close()
    
    def test_optimizer_initialization(self):
        """测试优化器初始化"""
        optimizer = RLStrategyOptimizer(self.trainer)
        
        self.assertIsNotNone(optimizer.trainer)
        self.assertGreater(optimizer.rl_weight, 0)
        self.assertLess(optimizer.rl_weight, 1)
        self.assertEqual(optimizer.rl_weight + optimizer.traditional_weight, 1.0)
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = RLOptimizerConfig(
            initial_rl_weight=0.5,
            max_rl_weight=0.8,
            min_rl_weight=0.2
        )
        
        optimizer = RLStrategyOptimizer(self.trainer, config)
        
        self.assertEqual(optimizer.rl_weight, 0.5)
        self.assertEqual(optimizer.config.max_rl_weight, 0.8)
        self.assertEqual(optimizer.config.min_rl_weight, 0.2)
    
    def test_action_to_signal_buy(self):
        """测试买入动作转换"""
        optimizer = RLStrategyOptimizer(self.trainer)
        
        market_data = MarketData(
            symbol='EURUSD',
            timeframe='H1',
            timestamp=datetime.now(),
            ohlcv=self.test_data.iloc[:100]
        )
        
        # 动作1 = BUY
        signal = optimizer._action_to_signal(1, market_data)
        
        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, 1)
        self.assertEqual(signal.strategy_id, 'rl_model')
        self.assertGreater(signal.entry_price, 0)
    
    def test_action_to_signal_sell(self):
        """测试卖出动作转换"""
        optimizer = RLStrategyOptimizer(self.trainer)
        
        market_data = MarketData(
            symbol='EURUSD',
            timeframe='H1',
            timestamp=datetime.now(),
            ohlcv=self.test_data.iloc[:100]
        )
        
        # 动作2 = SELL
        signal = optimizer._action_to_signal(2, market_data)
        
        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, -1)
        self.assertEqual(signal.strategy_id, 'rl_model')
    
    def test_action_to_signal_hold(self):
        """测试持有动作转换"""
        optimizer = RLStrategyOptimizer(self.trainer)
        
        market_data = MarketData(
            symbol='EURUSD',
            timeframe='H1',
            timestamp=datetime.now(),
            ohlcv=self.test_data.iloc[:100]
        )
        
        # 动作0 = HOLD
        signal = optimizer._action_to_signal(0, market_data)
        
        self.assertIsNone(signal)  # HOLD不生成信号
    
    def test_aggregate_traditional_signals(self):
        """测试传统信号聚合"""
        optimizer = RLStrategyOptimizer(self.trainer)
        
        # 创建测试信号
        signals = [
            Signal(
                strategy_id='strategy1',
                symbol='EURUSD',
                direction=1,
                strength=0.8,
                entry_price=1.1000,
                sl=1.0950,
                tp=1.1100,
                size=0.1,
                confidence=0.7,
                timestamp=datetime.now()
            ),
            Signal(
                strategy_id='strategy2',
                symbol='EURUSD',
                direction=1,
                strength=0.6,
                entry_price=1.1005,
                sl=1.0955,
                tp=1.1105,
                size=0.1,
                confidence=0.6,
                timestamp=datetime.now()
            )
        ]
        
        aggregated = optimizer._aggregate_traditional_signals(signals)
        
        self.assertIsNotNone(aggregated)
        self.assertEqual(aggregated.direction, 1)
        self.assertEqual(aggregated.strategy_id, 'traditional_aggregated')
        self.assertIn('num_signals', aggregated.metadata)
    
    def test_aggregate_empty_signals(self):
        """测试空信号列表聚合"""
        optimizer = RLStrategyOptimizer(self.trainer)
        
        aggregated = optimizer._aggregate_traditional_signals([])
        
        self.assertIsNone(aggregated)
    
    def test_fuse_signals_same_direction(self):
        """测试同方向信号融合"""
        optimizer = RLStrategyOptimizer(self.trainer)
        
        market_data = MarketData(
            symbol='EURUSD',
            timeframe='H1',
            timestamp=datetime.now(),
            ohlcv=self.test_data.iloc[:100]
        )
        
        rl_signal = Signal(
            strategy_id='rl_model',
            symbol='EURUSD',
            direction=1,
            strength=0.8,
            entry_price=1.1000,
            sl=1.0950,
            tp=1.1100,
            size=0.1,
            confidence=0.8,
            timestamp=datetime.now()
        )
        
        traditional_signal = Signal(
            strategy_id='traditional',
            symbol='EURUSD',
            direction=1,
            strength=0.7,
            entry_price=1.1005,
            sl=1.0955,
            tp=1.1105,
            size=0.1,
            confidence=0.7,
            timestamp=datetime.now()
        )
        
        fused = optimizer._fuse_signals(rl_signal, traditional_signal, market_data)
        
        self.assertIsNotNone(fused)
        self.assertEqual(fused.direction, 1)
        self.assertEqual(fused.strategy_id, 'rl_traditional_combined')
        self.assertIn('source', fused.metadata)
        self.assertEqual(fused.metadata['source'], 'combined')
    
    def test_fuse_signals_opposite_direction(self):
        """测试反方向信号融合"""
        optimizer = RLStrategyOptimizer(self.trainer)
        
        market_data = MarketData(
            symbol='EURUSD',
            timeframe='H1',
            timestamp=datetime.now(),
            ohlcv=self.test_data.iloc[:100]
        )
        
        rl_signal = Signal(
            strategy_id='rl_model',
            symbol='EURUSD',
            direction=1,
            strength=0.8,
            entry_price=1.1000,
            sl=1.0950,
            tp=1.1100,
            size=0.1,
            confidence=0.8,
            timestamp=datetime.now()
        )
        
        traditional_signal = Signal(
            strategy_id='traditional',
            symbol='EURUSD',
            direction=-1,
            strength=0.6,
            entry_price=1.1005,
            sl=1.1055,
            tp=1.0905,
            size=0.1,
            confidence=0.6,
            timestamp=datetime.now()
        )
        
        fused = optimizer._fuse_signals(rl_signal, traditional_signal, market_data)
        
        self.assertIsNotNone(fused)
        # 应该选择得分更高的信号
        self.assertIn('source', fused.metadata)
    
    def test_update_performance(self):
        """测试性能更新"""
        optimizer = RLStrategyOptimizer(self.trainer)
        
        # 添加性能数据
        for i in range(10):
            optimizer.update_performance(
                rl_performance=0.6 + i * 0.01,
                traditional_performance=0.5,
                combined_performance=0.55
            )
        
        self.assertEqual(len(optimizer.rl_performance_history), 10)
        self.assertEqual(len(optimizer.traditional_performance_history), 10)
        self.assertEqual(len(optimizer.combined_performance_history), 10)
    
    def test_weight_adjustment(self):
        """测试权重调整"""
        config = RLOptimizerConfig(
            initial_rl_weight=0.5,
            performance_window=10
        )
        optimizer = RLStrategyOptimizer(self.trainer, config)
        
        initial_weight = optimizer.rl_weight
        
        # 模拟RL表现更好
        for _ in range(15):
            optimizer.update_performance(
                rl_performance=0.7,
                traditional_performance=0.5,
                combined_performance=0.6
            )
        
        # RL权重应该增加
        self.assertGreater(optimizer.rl_weight, initial_weight)
    
    def test_add_online_sample(self):
        """测试添加在线样本"""
        config = RLOptimizerConfig(online_learning_enabled=True)
        optimizer = RLStrategyOptimizer(self.trainer, config)
        
        obs = np.random.randn(100)
        next_obs = np.random.randn(100)
        
        optimizer.add_online_sample(obs, 1, 0.5, next_obs)
        
        self.assertEqual(len(optimizer.online_samples), 1)
        self.assertEqual(optimizer.steps_since_update, 1)
    
    def test_online_learning_disabled(self):
        """测试禁用在线学习"""
        config = RLOptimizerConfig(online_learning_enabled=False)
        optimizer = RLStrategyOptimizer(self.trainer, config)
        
        obs = np.random.randn(100)
        next_obs = np.random.randn(100)
        
        optimizer.add_online_sample(obs, 1, 0.5, next_obs)
        
        self.assertEqual(len(optimizer.online_samples), 0)
    
    def test_get_statistics(self):
        """测试获取统计信息"""
        optimizer = RLStrategyOptimizer(self.trainer)
        
        # 模拟一些预测
        optimizer.total_predictions = 100
        optimizer.rl_predictions = 60
        optimizer.traditional_predictions = 40
        
        stats = optimizer.get_statistics()
        
        self.assertIn('rl_weight', stats)
        self.assertIn('total_predictions', stats)
        self.assertIn('rl_prediction_rate', stats)
        self.assertEqual(stats['total_predictions'], 100)
        self.assertEqual(stats['rl_predictions'], 60)
        self.assertAlmostEqual(stats['rl_prediction_rate'], 0.6)
    
    def test_reset_statistics(self):
        """测试重置统计信息"""
        optimizer = RLStrategyOptimizer(self.trainer)
        
        # 设置一些统计数据
        optimizer.total_predictions = 100
        optimizer.rl_predictions = 60
        optimizer.rl_performance_history = [0.5, 0.6, 0.7]
        
        # 重置
        optimizer.reset_statistics()
        
        self.assertEqual(optimizer.total_predictions, 0)
        self.assertEqual(optimizer.rl_predictions, 0)
        self.assertEqual(len(optimizer.rl_performance_history), 0)
    
    def test_get_combined_signal_no_traditional(self):
        """测试无传统信号时的组合信号"""
        optimizer = RLStrategyOptimizer(self.trainer)
        
        market_data = MarketData(
            symbol='EURUSD',
            timeframe='H1',
            timestamp=datetime.now(),
            ohlcv=self.test_data.iloc[:100]
        )
        
        obs, _ = self.env.reset()
        
        signal = optimizer.get_combined_signal(market_data, [], obs)
        
        # 可能返回None（如果RL选择HOLD）或RL信号
        if signal is not None:
            self.assertEqual(signal.strategy_id, 'rl_model')


if __name__ == '__main__':
    unittest.main()
