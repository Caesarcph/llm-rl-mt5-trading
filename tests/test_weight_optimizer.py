"""
策略权重优化器测试用例
"""

import unittest
from datetime import datetime, timedelta
import numpy as np

from src.strategies.weight_optimizer import (
    StrategyWeightOptimizer, WeightOptimizationConfig,
    WeightOptimizationMethod, StrategyMetrics
)
from src.core.models import Trade, TradeType


class TestStrategyWeightOptimizer(unittest.TestCase):
    """策略权重优化器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.config = WeightOptimizationConfig(
            method=WeightOptimizationMethod.PERFORMANCE_BASED,
            min_weight=0.1,
            max_weight=2.0,
            min_trades_required=5
        )
        self.optimizer = StrategyWeightOptimizer(self.config)
        
        # 创建测试交易数据
        self.test_trades = self._create_test_trades()
    
    def _create_test_trades(self) -> dict:
        """创建测试交易数据"""
        trades = {}
        
        # 策略1：高胜率，中等盈利
        strategy1_trades = []
        for i in range(20):
            profit = 100 if i < 14 else -50  # 70%胜率
            trade = Trade(
                trade_id=f"s1_{i}",
                symbol="EURUSD",
                type=TradeType.BUY,
                volume=0.1,
                open_price=1.1000,
                close_price=1.1000 + profit/10000,
                profit=profit,
                open_time=datetime.now() - timedelta(days=20-i),
                close_time=datetime.now() - timedelta(days=20-i, hours=-1),
                strategy_id="strategy1"
            )
            strategy1_trades.append(trade)
        trades['strategy1'] = strategy1_trades
        
        # 策略2：中等胜率，高盈利
        strategy2_trades = []
        for i in range(20):
            profit = 200 if i < 12 else -80  # 60%胜率，但盈利更高
            trade = Trade(
                trade_id=f"s2_{i}",
                symbol="EURUSD",
                type=TradeType.BUY,
                volume=0.1,
                open_price=1.1000,
                close_price=1.1000 + profit/10000,
                profit=profit,
                open_time=datetime.now() - timedelta(days=20-i),
                close_time=datetime.now() - timedelta(days=20-i, hours=-1),
                strategy_id="strategy2"
            )
            strategy2_trades.append(trade)
        trades['strategy2'] = strategy2_trades
        
        # 策略3：低胜率，低盈利
        strategy3_trades = []
        for i in range(20):
            profit = 50 if i < 8 else -60  # 40%胜率
            trade = Trade(
                trade_id=f"s3_{i}",
                symbol="EURUSD",
                type=TradeType.BUY,
                volume=0.1,
                open_price=1.1000,
                close_price=1.1000 + profit/10000,
                profit=profit,
                open_time=datetime.now() - timedelta(days=20-i),
                close_time=datetime.now() - timedelta(days=20-i, hours=-1),
                strategy_id="strategy3"
            )
            strategy3_trades.append(trade)
        trades['strategy3'] = strategy3_trades
        
        return trades
    
    def test_update_strategy_metrics(self):
        """测试更新策略指标"""
        self.optimizer.update_strategy_metrics("strategy1", self.test_trades['strategy1'])
        
        self.assertIn("strategy1", self.optimizer.strategy_metrics)
        metrics = self.optimizer.strategy_metrics["strategy1"]
        
        self.assertEqual(metrics.total_trades, 20)
        self.assertEqual(metrics.winning_trades, 14)
        self.assertEqual(metrics.losing_trades, 6)
        self.assertAlmostEqual(metrics.win_rate, 0.7, places=2)
    
    def test_strategy_metrics_calculation(self):
        """测试策略指标计算"""
        metrics = StrategyMetrics(strategy_name="test")
        metrics.calculate_metrics(self.test_trades['strategy1'])
        
        self.assertEqual(metrics.total_trades, 20)
        self.assertGreater(metrics.total_profit, 0)
        self.assertGreater(metrics.total_loss, 0)
        self.assertGreater(metrics.profit_factor, 1.0)  # 盈利策略
        self.assertGreater(metrics.win_rate, 0.5)
    
    def test_performance_score(self):
        """测试性能评分"""
        # 更新所有策略指标
        for strategy_name, trades in self.test_trades.items():
            self.optimizer.update_strategy_metrics(strategy_name, trades)
        
        # 获取性能评分
        score1 = self.optimizer.strategy_metrics['strategy1'].get_performance_score()
        score2 = self.optimizer.strategy_metrics['strategy2'].get_performance_score()
        score3 = self.optimizer.strategy_metrics['strategy3'].get_performance_score()
        
        # 验证评分在0-1之间
        self.assertGreaterEqual(score1, 0)
        self.assertLessEqual(score1, 1)
        
        # 策略2应该有最高评分（高盈利）
        self.assertGreater(score2, score3)
    
    def test_equal_weights(self):
        """测试等权重计算"""
        self.optimizer.config.method = WeightOptimizationMethod.EQUAL_WEIGHT
        
        strategy_names = ['strategy1', 'strategy2', 'strategy3']
        weights = self.optimizer.calculate_weights(strategy_names, force_update=True)
        
        # 所有权重应该相等
        self.assertEqual(weights['strategy1'], 1.0)
        self.assertEqual(weights['strategy2'], 1.0)
        self.assertEqual(weights['strategy3'], 1.0)
    
    def test_performance_based_weights(self):
        """测试基于性能的权重计算"""
        # 更新策略指标
        for strategy_name, trades in self.test_trades.items():
            self.optimizer.update_strategy_metrics(strategy_name, trades)
        
        self.optimizer.config.method = WeightOptimizationMethod.PERFORMANCE_BASED
        
        strategy_names = ['strategy1', 'strategy2', 'strategy3']
        weights = self.optimizer.calculate_weights(strategy_names, force_update=True)
        
        # 验证权重
        self.assertIn('strategy1', weights)
        self.assertIn('strategy2', weights)
        self.assertIn('strategy3', weights)
        
        # 表现好的策略应该有更高的权重
        self.assertGreater(weights['strategy2'], weights['strategy3'])
    
    def test_sharpe_based_weights(self):
        """测试基于夏普比率的权重计算"""
        # 更新策略指标
        for strategy_name, trades in self.test_trades.items():
            self.optimizer.update_strategy_metrics(strategy_name, trades)
        
        self.optimizer.config.method = WeightOptimizationMethod.SHARPE_RATIO
        
        strategy_names = ['strategy1', 'strategy2', 'strategy3']
        weights = self.optimizer.calculate_weights(strategy_names, force_update=True)
        
        # 验证权重存在
        for name in strategy_names:
            self.assertIn(name, weights)
            self.assertGreater(weights[name], 0)
    
    def test_win_rate_based_weights(self):
        """测试基于胜率的权重计算"""
        # 更新策略指标
        for strategy_name, trades in self.test_trades.items():
            self.optimizer.update_strategy_metrics(strategy_name, trades)
        
        self.optimizer.config.method = WeightOptimizationMethod.WIN_RATE
        
        strategy_names = ['strategy1', 'strategy2', 'strategy3']
        weights = self.optimizer.calculate_weights(strategy_names, force_update=True)
        
        # 高胜率策略应该有更高权重
        self.assertGreater(weights['strategy1'], weights['strategy3'])
    
    def test_profit_factor_based_weights(self):
        """测试基于盈利因子的权重计算"""
        # 更新策略指标
        for strategy_name, trades in self.test_trades.items():
            self.optimizer.update_strategy_metrics(strategy_name, trades)
        
        self.optimizer.config.method = WeightOptimizationMethod.PROFIT_FACTOR
        
        strategy_names = ['strategy1', 'strategy2', 'strategy3']
        weights = self.optimizer.calculate_weights(strategy_names, force_update=True)
        
        # 验证权重
        for name in strategy_names:
            self.assertIn(name, weights)
    
    def test_ensemble_weights(self):
        """测试集成方法权重计算"""
        # 更新策略指标
        for strategy_name, trades in self.test_trades.items():
            self.optimizer.update_strategy_metrics(strategy_name, trades)
        
        self.optimizer.config.method = WeightOptimizationMethod.ENSEMBLE
        
        strategy_names = ['strategy1', 'strategy2', 'strategy3']
        weights = self.optimizer.calculate_weights(strategy_names, force_update=True)
        
        # 验证权重
        for name in strategy_names:
            self.assertIn(name, weights)
            self.assertGreater(weights[name], 0)
    
    def test_weight_constraints(self):
        """测试权重约束"""
        weights = {
            'strategy1': 0.05,  # 低于最小值
            'strategy2': 2.5,   # 高于最大值
            'strategy3': 1.0    # 正常范围
        }
        
        constrained = self.optimizer._apply_weight_constraints(weights)
        
        self.assertEqual(constrained['strategy1'], self.config.min_weight)
        self.assertEqual(constrained['strategy2'], self.config.max_weight)
        self.assertEqual(constrained['strategy3'], 1.0)
    
    def test_weight_history(self):
        """测试权重历史记录"""
        # 更新策略指标
        for strategy_name, trades in self.test_trades.items():
            self.optimizer.update_strategy_metrics(strategy_name, trades)
        
        strategy_names = ['strategy1', 'strategy2', 'strategy3']
        
        # 计算权重多次
        for _ in range(3):
            self.optimizer.calculate_weights(strategy_names, force_update=True)
        
        # 获取历史
        history = self.optimizer.get_weight_history()
        
        self.assertGreater(len(history), 0)
        self.assertIn('timestamp', history[0])
        self.assertIn('weights', history[0])
        self.assertIn('method', history[0])
    
    def test_get_strategy_weight_history(self):
        """测试获取特定策略的权重历史"""
        # 更新策略指标
        for strategy_name, trades in self.test_trades.items():
            self.optimizer.update_strategy_metrics(strategy_name, trades)
        
        strategy_names = ['strategy1', 'strategy2', 'strategy3']
        
        # 计算权重多次
        for _ in range(3):
            self.optimizer.calculate_weights(strategy_names, force_update=True)
        
        # 获取特定策略历史
        history = self.optimizer.get_weight_history(strategy_name='strategy1')
        
        self.assertGreater(len(history), 0)
        for record in history:
            self.assertIn('timestamp', record)
            self.assertIn('weight', record)
    
    def test_update_frequency(self):
        """测试更新频率控制"""
        # 更新策略指标
        for strategy_name, trades in self.test_trades.items():
            self.optimizer.update_strategy_metrics(strategy_name, trades)
        
        strategy_names = ['strategy1', 'strategy2', 'strategy3']
        
        # 第一次计算
        weights1 = self.optimizer.calculate_weights(strategy_names, force_update=True)
        
        # 立即再次计算（不强制更新）
        weights2 = self.optimizer.calculate_weights(strategy_names, force_update=False)
        
        # 应该返回相同的权重（因为未到更新时间）
        self.assertEqual(weights1, weights2)
    
    def test_insufficient_trades(self):
        """测试交易数据不足的情况"""
        # 创建少量交易
        few_trades = self.test_trades['strategy1'][:3]
        
        self.optimizer.update_strategy_metrics("strategy_few", few_trades)
        
        weights = self.optimizer.calculate_weights(['strategy_few'], force_update=True)
        
        # 数据不足应该给予中性权重（归一化后为1.0，因为只有一个策略）
        self.assertAlmostEqual(weights['strategy_few'], 1.0, places=1)
    
    def test_no_metrics(self):
        """测试没有指标数据的情况"""
        weights = self.optimizer.calculate_weights(['unknown_strategy'], force_update=True)
        
        # 无数据应该给予中性权重（归一化后为1.0，因为只有一个策略）
        self.assertAlmostEqual(weights['unknown_strategy'], 1.0, places=1)
    
    def test_get_strategy_metrics(self):
        """测试获取策略指标"""
        self.optimizer.update_strategy_metrics("strategy1", self.test_trades['strategy1'])
        
        metrics = self.optimizer.get_strategy_metrics("strategy1")
        
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.strategy_name, "strategy1")
        self.assertGreater(metrics.total_trades, 0)
    
    def test_get_all_metrics(self):
        """测试获取所有策略指标"""
        for strategy_name, trades in self.test_trades.items():
            self.optimizer.update_strategy_metrics(strategy_name, trades)
        
        all_metrics = self.optimizer.get_all_metrics()
        
        self.assertEqual(len(all_metrics), 3)
        self.assertIn('strategy1', all_metrics)
        self.assertIn('strategy2', all_metrics)
        self.assertIn('strategy3', all_metrics)
    
    def test_rl_optimized_weights_fallback(self):
        """测试RL优化权重的回退机制"""
        # 更新策略指标
        for strategy_name, trades in self.test_trades.items():
            self.optimizer.update_strategy_metrics(strategy_name, trades)
        
        self.optimizer.config.method = WeightOptimizationMethod.RL_OPTIMIZED
        
        strategy_names = ['strategy1', 'strategy2', 'strategy3']
        
        # 没有足够历史数据时应该回退到性能基础方法
        weights = self.optimizer.calculate_weights(strategy_names, force_update=True)
        
        # 验证权重存在且合理
        for name in strategy_names:
            self.assertIn(name, weights)
            self.assertGreater(weights[name], 0)


if __name__ == '__main__':
    unittest.main()
