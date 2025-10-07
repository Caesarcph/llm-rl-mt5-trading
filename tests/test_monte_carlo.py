"""
蒙特卡洛模拟器测试
"""

import unittest
import numpy as np
from datetime import datetime, timedelta
from typing import List

from src.core.models import Trade, TradeType
from src.strategies.monte_carlo import (
    MonteCarloSimulator,
    MonteCarloConfig,
    MonteCarloResult,
    StressTestScenario,
    StressTestResult
)
from src.strategies.backtest import BacktestResult


class TestMonteCarloConfig(unittest.TestCase):
    """测试蒙特卡洛配置"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = MonteCarloConfig()
        self.assertEqual(config.n_simulations, 1000)
        self.assertEqual(config.confidence_level, 0.95)
        self.assertTrue(config.validate())
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = MonteCarloConfig(
            n_simulations=500,
            confidence_level=0.99,
            random_seed=42,
            parallel=False
        )
        self.assertEqual(config.n_simulations, 500)
        self.assertEqual(config.confidence_level, 0.99)
        self.assertEqual(config.random_seed, 42)
        self.assertFalse(config.parallel)
    
    def test_invalid_config(self):
        """测试无效配置"""
        with self.assertRaises(ValueError):
            config = MonteCarloConfig(n_simulations=0)
            config.validate()
        
        with self.assertRaises(ValueError):
            config = MonteCarloConfig(confidence_level=1.5)
            config.validate()


class TestMonteCarloSimulator(unittest.TestCase):
    """测试蒙特卡洛模拟器"""
    
    def setUp(self):
        """设置测试环境"""
        self.config = MonteCarloConfig(
            n_simulations=100,  # 减少模拟次数以加快测试
            random_seed=42,
            parallel=False  # 测试时使用顺序执行
        )
        self.simulator = MonteCarloSimulator(self.config)
        self.trades = self._create_sample_trades()
    
    def _create_sample_trades(self, n_trades: int = 50) -> List[Trade]:
        """创建示例交易"""
        trades = []
        np.random.seed(42)
        
        for i in range(n_trades):
            # 60%胜率
            profit = np.random.normal(100, 50) if np.random.random() < 0.6 else np.random.normal(-80, 40)
            
            trade = Trade(
                trade_id=f"trade_{i}",
                symbol="EURUSD",
                type=TradeType.BUY if i % 2 == 0 else TradeType.SELL,
                volume=0.1,
                open_price=1.1000 + np.random.normal(0, 0.001),
                close_price=1.1000 + np.random.normal(0, 0.001),
                profit=profit,
                commission=2.0,
                swap=0.5,
                open_time=datetime.now() - timedelta(hours=n_trades - i),
                close_time=datetime.now() - timedelta(hours=n_trades - i - 1),
                strategy_id="test_strategy"
            )
            trades.append(trade)
        
        return trades
    
    def test_simulate_from_trades(self):
        """测试基于交易的模拟"""
        result = self.simulator.simulate_from_trades(self.trades, initial_balance=10000.0)
        
        # 验证结果结构
        self.assertIsInstance(result, MonteCarloResult)
        self.assertEqual(result.n_simulations, 100)
        self.assertEqual(result.confidence_level, 0.95)
        
        # 验证统计指标
        self.assertIsInstance(result.mean_return, float)
        self.assertIsInstance(result.median_return, float)
        self.assertIsInstance(result.std_return, float)
        
        # 验证风险指标
        self.assertIsInstance(result.var, float)
        self.assertIsInstance(result.cvar, float)
        self.assertLessEqual(result.cvar, result.var)  # CVaR应该小于等于VaR
        
        # 验证概率
        self.assertGreaterEqual(result.prob_profit, 0)
        self.assertLessEqual(result.prob_profit, 1)
        self.assertAlmostEqual(result.prob_profit + result.prob_loss, 1.0, places=2)
        
        # 验证分位数
        self.assertEqual(len(result.percentiles), 9)
        self.assertIn(0.50, result.percentiles)
        self.assertAlmostEqual(result.percentiles[0.50], result.median_return, places=2)
    
    def test_simulate_from_backtest(self):
        """测试基于回测结果的模拟"""
        backtest_result = BacktestResult(
            strategy_name="Test Strategy",
            symbol="EURUSD",
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            initial_balance=10000.0,
            final_balance=11000.0,
            total_return=0.10,
            total_trades=50,
            winning_trades=30,
            losing_trades=20,
            win_rate=0.6,
            profit_factor=1.5,
            max_drawdown=500.0,
            max_drawdown_percent=0.05,
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            calmar_ratio=2.0,
            average_trade=20.0,
            average_win=50.0,
            average_loss=-30.0,
            largest_win=200.0,
            largest_loss=-150.0,
            consecutive_wins=5,
            consecutive_losses=3,
            trades=self.trades
        )
        
        result = self.simulator.simulate_from_backtest(backtest_result)
        
        self.assertIsInstance(result, MonteCarloResult)
        self.assertEqual(result.n_simulations, 100)
    
    def test_empty_trades(self):
        """测试空交易列表"""
        with self.assertRaises(ValueError):
            self.simulator.simulate_from_trades([], initial_balance=10000.0)
    
    def test_var_calculation(self):
        """测试VaR计算"""
        returns = np.array([-0.05, -0.03, -0.01, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15])
        
        var_95 = self.simulator._calculate_var(returns, 0.95)
        self.assertLess(var_95, 0)  # VaR应该是负数
        
        var_99 = self.simulator._calculate_var(returns, 0.99)
        self.assertLess(var_99, var_95)  # 99% VaR应该更极端
    
    def test_cvar_calculation(self):
        """测试CVaR计算"""
        returns = np.array([-0.10, -0.08, -0.05, -0.03, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15])
        
        cvar = self.simulator._calculate_cvar(returns, 0.95)
        var = self.simulator._calculate_var(returns, 0.95)
        
        self.assertLessEqual(cvar, var)  # CVaR应该小于等于VaR
    
    def test_parallel_vs_sequential(self):
        """测试并行和顺序执行的一致性"""
        # 顺序执行
        config_seq = MonteCarloConfig(n_simulations=50, random_seed=42, parallel=False)
        simulator_seq = MonteCarloSimulator(config_seq)
        result_seq = simulator_seq.simulate_from_trades(self.trades, initial_balance=10000.0)
        
        # 并行执行
        config_par = MonteCarloConfig(n_simulations=50, random_seed=42, parallel=True, max_workers=2)
        simulator_par = MonteCarloSimulator(config_par)
        result_par = simulator_par.simulate_from_trades(self.trades, initial_balance=10000.0)
        
        # 结果应该相似（由于随机性，不会完全相同）
        self.assertAlmostEqual(result_seq.mean_return, result_par.mean_return, delta=0.05)
        self.assertAlmostEqual(result_seq.std_return, result_par.std_return, delta=0.05)


class TestStressTest(unittest.TestCase):
    """测试压力测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.config = MonteCarloConfig(n_simulations=100, random_seed=42, parallel=False)
        self.simulator = MonteCarloSimulator(self.config)
        self.trades = self._create_sample_trades()
    
    def _create_sample_trades(self, n_trades: int = 50) -> List[Trade]:
        """创建示例交易"""
        trades = []
        np.random.seed(42)
        
        for i in range(n_trades):
            profit = np.random.normal(50, 30) if np.random.random() < 0.6 else np.random.normal(-40, 20)
            
            trade = Trade(
                trade_id=f"trade_{i}",
                symbol="EURUSD",
                type=TradeType.BUY,
                volume=0.1,
                open_price=1.1000,
                close_price=1.1010,
                profit=profit,
                commission=2.0,
                swap=0.5,
                open_time=datetime.now() - timedelta(hours=n_trades - i),
                close_time=datetime.now() - timedelta(hours=n_trades - i - 1),
                strategy_id="test_strategy"
            )
            trades.append(trade)
        
        return trades
    
    def test_stress_test_default_scenarios(self):
        """测试默认压力场景"""
        results = self.simulator.stress_test(self.trades, initial_balance=10000.0)
        
        # 应该有5个默认场景
        self.assertEqual(len(results), 5)
        
        # 验证每个结果
        for result in results:
            self.assertIsInstance(result, StressTestResult)
            self.assertIsInstance(result.scenario_name, str)
            self.assertIsInstance(result.stressed_return, float)
            self.assertIsInstance(result.survival_probability, float)
            self.assertGreaterEqual(result.survival_probability, 0)
            self.assertLessEqual(result.survival_probability, 1)
    
    def test_stress_test_custom_scenario(self):
        """测试自定义压力场景"""
        custom_scenario = StressTestScenario(
            name="自定义场景",
            description="测试场景",
            return_shock=-0.30,
            volatility_multiplier=1.5,
            win_rate_adjustment=-0.10,
            drawdown_multiplier=2.0
        )
        
        results = self.simulator.stress_test(
            self.trades,
            scenarios=[custom_scenario],
            initial_balance=10000.0
        )
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].scenario_name, "自定义场景")
        
        # 压力后收益应该低于原始收益
        self.assertLess(results[0].stressed_return, results[0].original_return)
    
    def test_stress_test_impact(self):
        """测试压力测试影响"""
        results = self.simulator.stress_test(self.trades, initial_balance=10000.0)
        
        # 找到"市场崩盘"场景
        crash_result = next(r for r in results if "崩盘" in r.scenario_name)
        
        # 崩盘场景应该有显著的负面影响
        self.assertLess(crash_result.return_impact, 0)
        self.assertLess(crash_result.drawdown_impact, 0)
        
        # 生存概率应该是0到1之间的值
        self.assertGreaterEqual(crash_result.survival_probability, 0)
        self.assertLessEqual(crash_result.survival_probability, 1)
    
    def test_empty_trades_stress_test(self):
        """测试空交易列表的压力测试"""
        with self.assertRaises(ValueError):
            self.simulator.stress_test([], initial_balance=10000.0)


class TestScenarioAnalysis(unittest.TestCase):
    """测试情景分析"""
    
    def setUp(self):
        """设置测试环境"""
        self.config = MonteCarloConfig(n_simulations=50, random_seed=42, parallel=False)
        self.simulator = MonteCarloSimulator(self.config)
        self.trades = self._create_sample_trades()
    
    def _create_sample_trades(self, n_trades: int = 30) -> List[Trade]:
        """创建示例交易"""
        trades = []
        np.random.seed(42)
        
        for i in range(n_trades):
            profit = np.random.normal(40, 25) if np.random.random() < 0.6 else np.random.normal(-30, 15)
            
            trade = Trade(
                trade_id=f"trade_{i}",
                symbol="EURUSD",
                type=TradeType.BUY,
                volume=0.1,
                open_price=1.1000,
                close_price=1.1010,
                profit=profit,
                commission=2.0,
                swap=0.5,
                open_time=datetime.now() - timedelta(hours=n_trades - i),
                close_time=datetime.now() - timedelta(hours=n_trades - i - 1),
                strategy_id="test_strategy"
            )
            trades.append(trade)
        
        return trades
    
    def test_scenario_analysis_default(self):
        """测试默认情景分析"""
        results = self.simulator.scenario_analysis(self.trades, initial_balance=10000.0)
        
        # 应该有3个默认场景
        self.assertEqual(len(results), 3)
        self.assertIn('基准情景', results)
        self.assertIn('乐观情景', results)
        self.assertIn('悲观情景', results)
        
        # 验证每个结果
        for scenario_name, result in results.items():
            self.assertIsInstance(result, MonteCarloResult)
            self.assertEqual(result.n_simulations, 50)
    
    def test_scenario_analysis_ordering(self):
        """测试情景分析结果排序"""
        results = self.simulator.scenario_analysis(self.trades, initial_balance=10000.0)
        
        # 乐观情景应该有最高的平均收益
        # 悲观情景应该有最低的平均收益
        optimistic = results['乐观情景']
        pessimistic = results['悲观情景']
        baseline = results['基准情景']
        
        self.assertGreater(optimistic.mean_return, baseline.mean_return)
        self.assertLess(pessimistic.mean_return, baseline.mean_return)
    
    def test_scenario_analysis_custom(self):
        """测试自定义情景分析"""
        custom_scenarios = {
            '高增长': {'return_multiplier': 1.5, 'volatility_multiplier': 1.0},
            '低增长': {'return_multiplier': 0.8, 'volatility_multiplier': 1.2}
        }
        
        results = self.simulator.scenario_analysis(
            self.trades,
            initial_balance=10000.0,
            scenarios=custom_scenarios
        )
        
        self.assertEqual(len(results), 2)
        self.assertIn('高增长', results)
        self.assertIn('低增长', results)
        
        # 高增长场景应该有更高的收益
        self.assertGreater(
            results['高增长'].mean_return,
            results['低增长'].mean_return
        )


class TestRiskMetrics(unittest.TestCase):
    """测试风险指标计算"""
    
    def setUp(self):
        """设置测试环境"""
        self.config = MonteCarloConfig(n_simulations=100, random_seed=42, parallel=False)
        self.simulator = MonteCarloSimulator(self.config)
        self.trades = self._create_sample_trades()
    
    def _create_sample_trades(self, n_trades: int = 40) -> List[Trade]:
        """创建示例交易"""
        trades = []
        np.random.seed(42)
        
        for i in range(n_trades):
            profit = np.random.normal(60, 35) if np.random.random() < 0.65 else np.random.normal(-45, 25)
            
            trade = Trade(
                trade_id=f"trade_{i}",
                symbol="EURUSD",
                type=TradeType.BUY,
                volume=0.1,
                open_price=1.1000,
                close_price=1.1010,
                profit=profit,
                commission=2.0,
                swap=0.5,
                open_time=datetime.now() - timedelta(hours=n_trades - i),
                close_time=datetime.now() - timedelta(hours=n_trades - i - 1),
                strategy_id="test_strategy"
            )
            trades.append(trade)
        
        return trades
    
    def test_calculate_risk_metrics(self):
        """测试风险指标计算"""
        metrics = self.simulator.calculate_risk_metrics(self.trades, initial_balance=10000.0)
        
        # 验证所有指标都存在
        self.assertIsInstance(metrics.var_1d, float)
        self.assertIsInstance(metrics.var_5d, float)
        self.assertIsInstance(metrics.max_drawdown, float)
        self.assertIsInstance(metrics.sharpe_ratio, float)
        self.assertIsInstance(metrics.sortino_ratio, float)
        self.assertIsInstance(metrics.calmar_ratio, float)
        self.assertIsInstance(metrics.win_rate, float)
        self.assertIsInstance(metrics.profit_factor, float)
        
        # 验证合理性
        self.assertGreaterEqual(metrics.win_rate, 0)
        self.assertLessEqual(metrics.win_rate, 1)
        self.assertGreaterEqual(metrics.profit_factor, 0)
        
        # 5日VaR应该大于1日VaR（绝对值）
        self.assertGreater(abs(metrics.var_5d), abs(metrics.var_1d))
    
    def test_risk_acceptability(self):
        """测试风险可接受性"""
        metrics = self.simulator.calculate_risk_metrics(self.trades, initial_balance=10000.0)
        
        # 测试默认阈值
        is_acceptable = metrics.is_risk_acceptable()
        self.assertTrue(isinstance(is_acceptable, (bool, np.bool_)))
        
        # 测试自定义阈值
        is_acceptable_strict = metrics.is_risk_acceptable(max_var=0.01, max_dd=0.10)
        self.assertTrue(isinstance(is_acceptable_strict, (bool, np.bool_)))


class TestExportResults(unittest.TestCase):
    """测试结果导出"""
    
    def setUp(self):
        """设置测试环境"""
        self.config = MonteCarloConfig(n_simulations=50, random_seed=42, parallel=False)
        self.simulator = MonteCarloSimulator(self.config)
        self.trades = self._create_sample_trades()
    
    def _create_sample_trades(self, n_trades: int = 30) -> List[Trade]:
        """创建示例交易"""
        trades = []
        np.random.seed(42)
        
        for i in range(n_trades):
            profit = np.random.normal(50, 30)
            
            trade = Trade(
                trade_id=f"trade_{i}",
                symbol="EURUSD",
                type=TradeType.BUY,
                volume=0.1,
                open_price=1.1000,
                close_price=1.1010,
                profit=profit,
                commission=2.0,
                swap=0.5,
                open_time=datetime.now() - timedelta(hours=n_trades - i),
                close_time=datetime.now() - timedelta(hours=n_trades - i - 1),
                strategy_id="test_strategy"
            )
            trades.append(trade)
        
        return trades
    
    def test_export_json(self):
        """测试JSON导出"""
        result = self.simulator.simulate_from_trades(self.trades, initial_balance=10000.0)
        
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filepath = f.name
        
        try:
            success = self.simulator.export_results(result, filepath, format='json')
            self.assertTrue(success)
            self.assertTrue(os.path.exists(filepath))
            
            # 验证文件内容
            import json
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.assertIn('simulation_id', data)
            self.assertIn('mean_return', data)
            self.assertIn('var', data)
        
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    
    def test_export_csv(self):
        """测试CSV导出"""
        result = self.simulator.simulate_from_trades(self.trades, initial_balance=10000.0)
        
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            filepath = f.name
        
        try:
            success = self.simulator.export_results(result, filepath, format='csv')
            self.assertTrue(success)
            self.assertTrue(os.path.exists(filepath))
            
            # 验证文件内容
            import pandas as pd
            df = pd.read_csv(filepath)
            
            self.assertIn('simulation', df.columns)
            self.assertIn('return', df.columns)
            self.assertIn('drawdown', df.columns)
            self.assertEqual(len(df), result.n_simulations)
        
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    
    def test_export_invalid_format(self):
        """测试无效格式导出"""
        result = self.simulator.simulate_from_trades(self.trades, initial_balance=10000.0)
        
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            filepath = f.name
        
        try:
            success = self.simulator.export_results(result, filepath, format='invalid')
            self.assertFalse(success)
        finally:
            import os
            if os.path.exists(filepath):
                os.remove(filepath)


if __name__ == '__main__':
    unittest.main()
