"""
策略性能跟踪器测试用例
"""

import unittest
from datetime import datetime, timedelta
import pandas as pd

from src.strategies.performance_tracker import (
    PerformanceTracker, PerformancePeriod,
    StrategyPerformanceMetrics
)
from src.core.models import Trade, TradeType, Signal


class TestPerformanceTracker(unittest.TestCase):
    """性能跟踪器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.tracker = PerformanceTracker()
        
        # 创建测试交易数据
        self.test_trades = self._create_test_trades()
        
        # 记录交易
        for strategy_name, trades in self.test_trades.items():
            for trade in trades:
                self.tracker.record_trade(trade)
    
    def _create_test_trades(self) -> dict:
        """创建测试交易数据"""
        trades = {}
        
        # 策略1：盈利策略
        strategy1_trades = []
        for i in range(10):
            profit = 100 if i < 7 else -50  # 70%胜率
            trade = Trade(
                trade_id=f"s1_{i}",
                symbol="EURUSD",
                type=TradeType.BUY,
                volume=0.1,
                open_price=1.1000,
                close_price=1.1000 + profit/10000,
                profit=profit,
                commission=2.0,
                swap=0.5,
                open_time=datetime.now() - timedelta(days=10-i),
                close_time=datetime.now() - timedelta(days=10-i, hours=-2),
                strategy_id="strategy1"
            )
            strategy1_trades.append(trade)
        trades['strategy1'] = strategy1_trades
        
        # 策略2：亏损策略
        strategy2_trades = []
        for i in range(10):
            profit = 50 if i < 3 else -80  # 30%胜率
            trade = Trade(
                trade_id=f"s2_{i}",
                symbol="GBPUSD",
                type=TradeType.SELL,
                volume=0.1,
                open_price=1.2500,
                close_price=1.2500 + profit/10000,
                profit=profit,
                commission=2.0,
                swap=-0.5,
                open_time=datetime.now() - timedelta(days=10-i),
                close_time=datetime.now() - timedelta(days=10-i, hours=-3),
                strategy_id="strategy2"
            )
            strategy2_trades.append(trade)
        trades['strategy2'] = strategy2_trades
        
        return trades
    
    def test_record_trade(self):
        """测试记录交易"""
        trade = Trade(
            trade_id="test_1",
            symbol="EURUSD",
            type=TradeType.BUY,
            volume=0.1,
            open_price=1.1000,
            close_price=1.1010,
            profit=100,
            open_time=datetime.now(),
            close_time=datetime.now(),
            strategy_id="test_strategy"
        )
        
        self.tracker.record_trade(trade)
        
        self.assertIn("test_strategy", self.tracker.trades_by_strategy)
        self.assertEqual(len(self.tracker.trades_by_strategy["test_strategy"]), 1)
    
    def test_record_signal(self):
        """测试记录信号"""
        signal = Signal(
            strategy_id="test_strategy",
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
        
        self.tracker.record_signal(signal)
        
        self.assertIn("test_strategy", self.tracker.signals_by_strategy)
        self.assertEqual(len(self.tracker.signals_by_strategy["test_strategy"]), 1)
    
    def test_calculate_metrics(self):
        """测试计算性能指标"""
        metrics = self.tracker.calculate_metrics("strategy1")
        
        self.assertEqual(metrics.strategy_name, "strategy1")
        self.assertEqual(metrics.total_trades, 10)
        self.assertEqual(metrics.winning_trades, 7)
        self.assertEqual(metrics.losing_trades, 3)
        self.assertAlmostEqual(metrics.win_rate, 0.7, places=2)
        self.assertGreater(metrics.net_profit, 0)
    
    def test_metrics_calculation_details(self):
        """测试详细指标计算"""
        metrics = self.tracker.calculate_metrics("strategy1")
        
        # 验证盈亏统计
        self.assertGreater(metrics.gross_profit, 0)
        self.assertGreater(metrics.gross_loss, 0)
        self.assertEqual(metrics.net_profit, metrics.gross_profit - metrics.gross_loss)
        
        # 验证比率指标
        self.assertGreater(metrics.profit_factor, 1.0)  # 盈利策略
        
        # 验证交易质量
        self.assertGreater(metrics.avg_win, 0)
        self.assertGreater(metrics.avg_loss, 0)
        self.assertNotEqual(metrics.avg_profit_per_trade, 0)
    
    def test_filter_trades_by_date(self):
        """测试按日期过滤交易"""
        start_date = datetime.now() - timedelta(days=5)
        end_date = datetime.now()
        
        metrics = self.tracker.calculate_metrics("strategy1", start_date, end_date)
        
        # 应该只包含最近5天的交易
        self.assertLessEqual(metrics.total_trades, 10)
        self.assertGreater(metrics.total_trades, 0)
    
    def test_consecutive_wins_losses(self):
        """测试连续盈亏计算"""
        metrics = self.tracker.calculate_metrics("strategy1")
        
        # 策略1有7个盈利交易，应该有连续盈利
        self.assertGreater(metrics.max_consecutive_wins, 0)
    
    def test_generate_report(self):
        """测试生成性能报告"""
        report = self.tracker.generate_report(
            period=PerformancePeriod.WEEKLY,
            strategy_names=['strategy1', 'strategy2']
        )
        
        self.assertIsNotNone(report)
        self.assertEqual(len(report.strategies), 2)
        self.assertIn('rankings', report.to_dict())
        self.assertIn('summary', report.to_dict())
    
    def test_report_rankings(self):
        """测试报告排名"""
        report = self.tracker.generate_report(
            period=PerformancePeriod.ALL_TIME,
            strategy_names=['strategy1', 'strategy2']
        )
        
        # 验证排名存在
        self.assertIn('net_profit', report.rankings)
        self.assertIn('win_rate', report.rankings)
        self.assertIn('profit_factor', report.rankings)
        
        # 策略1应该在净利润排名中靠前
        net_profit_ranking = report.rankings['net_profit']
        self.assertEqual(net_profit_ranking[0][0], 'strategy1')
    
    def test_report_summary(self):
        """测试报告摘要"""
        report = self.tracker.generate_report(
            period=PerformancePeriod.ALL_TIME,
            strategy_names=['strategy1', 'strategy2']
        )
        
        summary = report.summary
        
        self.assertEqual(summary['total_strategies'], 2)
        self.assertEqual(summary['total_trades'], 20)
        self.assertIn('total_net_profit', summary)
        self.assertIn('avg_win_rate', summary)
    
    def test_get_strategy_ranking(self):
        """测试获取策略排名"""
        ranking = self.tracker.get_strategy_ranking(
            metric='net_profit',
            period=PerformancePeriod.ALL_TIME
        )
        
        self.assertEqual(len(ranking), 2)
        # 策略1应该排在前面（更高的净利润）
        self.assertEqual(ranking[0][0], 'strategy1')
    
    def test_get_performance_comparison(self):
        """测试获取性能对比"""
        comparison = self.tracker.get_performance_comparison(
            strategy_names=['strategy1', 'strategy2'],
            period=PerformancePeriod.ALL_TIME
        )
        
        self.assertIsInstance(comparison, pd.DataFrame)
        self.assertEqual(len(comparison), 2)
        self.assertIn('Strategy', comparison.columns)
        self.assertIn('Win Rate', comparison.columns)
        self.assertIn('Net Profit', comparison.columns)
    
    def test_get_equity_curve(self):
        """测试获取权益曲线"""
        equity_curve = self.tracker.get_equity_curve('strategy1')
        
        self.assertIsInstance(equity_curve, pd.DataFrame)
        self.assertEqual(len(equity_curve), 10)
        self.assertIn('cumulative_profit', equity_curve.columns)
        self.assertIn('profit', equity_curve.columns)
        
        # 验证累计利润是递增的（对于盈利策略）
        self.assertGreater(
            equity_curve['cumulative_profit'].iloc[-1],
            equity_curve['cumulative_profit'].iloc[0]
        )
    
    def test_metrics_cache(self):
        """测试指标缓存"""
        # 第一次计算
        metrics1 = self.tracker.calculate_metrics("strategy1", use_cache=True)
        
        # 第二次应该使用缓存
        metrics2 = self.tracker.calculate_metrics("strategy1", use_cache=True)
        
        # 应该是同一个对象
        self.assertEqual(id(metrics1), id(metrics2))
        
        # 清除缓存
        self.tracker.clear_cache()
        
        # 再次计算应该是新对象
        metrics3 = self.tracker.calculate_metrics("strategy1", use_cache=True)
        self.assertNotEqual(id(metrics1), id(metrics3))
    
    def test_report_history(self):
        """测试报告历史"""
        # 生成多个报告
        for _ in range(3):
            self.tracker.generate_report(period=PerformancePeriod.DAILY)
        
        history = self.tracker.get_report_history(limit=2)
        
        self.assertEqual(len(history), 2)
        self.assertIsNotNone(history[0].report_id)
    
    def test_export_report(self):
        """测试导出报告"""
        report = self.tracker.generate_report(period=PerformancePeriod.WEEKLY)
        
        filepath = "test_report.json"
        result = self.tracker.export_report(report, filepath)
        
        self.assertTrue(result)
        
        # 清理测试文件
        import os
        if os.path.exists(filepath):
            os.remove(filepath)
    
    def test_empty_strategy(self):
        """测试空策略"""
        metrics = self.tracker.calculate_metrics("nonexistent_strategy")
        
        self.assertEqual(metrics.total_trades, 0)
        self.assertEqual(metrics.net_profit, 0)
    
    def test_max_drawdown_calculation(self):
        """测试最大回撤计算"""
        metrics = self.tracker.calculate_metrics("strategy1")
        
        # 应该有回撤（因为有亏损交易）
        self.assertGreaterEqual(metrics.max_drawdown, 0)
    
    def test_sharpe_ratio_calculation(self):
        """测试夏普比率计算"""
        metrics = self.tracker.calculate_metrics("strategy1")
        
        # 盈利策略应该有正的夏普比率
        self.assertGreater(metrics.sharpe_ratio, 0)
    
    def test_report_to_json(self):
        """测试报告转JSON"""
        report = self.tracker.generate_report(period=PerformancePeriod.DAILY)
        
        json_str = report.to_json()
        
        self.assertIsInstance(json_str, str)
        self.assertIn('report_id', json_str)
        self.assertIn('strategies', json_str)
    
    def test_metrics_to_dict(self):
        """测试指标转字典"""
        metrics = self.tracker.calculate_metrics("strategy1")
        
        metrics_dict = metrics.to_dict()
        
        self.assertIsInstance(metrics_dict, dict)
        self.assertIn('strategy_name', metrics_dict)
        self.assertIn('total_trades', metrics_dict)
        self.assertIn('win_rate', metrics_dict)


if __name__ == '__main__':
    unittest.main()
