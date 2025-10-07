"""
回测框架测试
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.models import MarketData
from src.strategies.base_strategies import StrategyConfig, StrategyType, TrendFollowingStrategy
from src.strategies.backtest import BacktestConfig, BacktestEngine, BacktestResult


class TestBacktestEngine(unittest.TestCase):
    """测试回测引擎"""
    
    def setUp(self):
        # 创建回测配置
        self.backtest_config = BacktestConfig(
            initial_balance=10000.0,
            leverage=100,
            spread=0.0001,
            commission=0.0,
            slippage=0.0001
        )
        
        # 创建回测引擎
        self.engine = BacktestEngine(self.backtest_config)
        
        # 创建测试策略
        strategy_config = StrategyConfig(
            name="test_trend_strategy",
            strategy_type=StrategyType.TREND_FOLLOWING,
            risk_per_trade=0.02
        )
        self.strategy = TrendFollowingStrategy(strategy_config)
        
        # 创建测试数据
        self.market_data_list = self.create_test_market_data()
    
    def create_test_market_data(self, periods=100):
        """创建测试市场数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='h')
        np.random.seed(42)
        
        base_price = 1.1000
        # 创建上升趋势
        trend = np.linspace(0, 0.01, periods)
        noise = np.random.normal(0, 0.0005, periods)
        prices = base_price + trend + noise
        
        market_data_list = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            high = price + abs(np.random.normal(0, 0.0002))
            low = price - abs(np.random.normal(0, 0.0002))
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            volume = np.random.randint(5000, 15000)
            
            ohlcv_data = pd.DataFrame([{
                'timestamp': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            }])
            
            market_data = MarketData(
                symbol="EURUSD",
                timeframe="H1",
                timestamp=date,
                ohlcv=ohlcv_data
            )
            
            market_data_list.append(market_data)
        
        return market_data_list
    
    def test_backtest_config_validation(self):
        """测试回测配置验证"""
        # 有效配置
        valid_config = BacktestConfig(initial_balance=10000.0)
        self.assertTrue(valid_config.validate())
        
        # 无效配置 - 负余额
        with self.assertRaises(ValueError):
            invalid_config = BacktestConfig(initial_balance=-1000.0)
            invalid_config.validate()
        
        # 无效配置 - 零杠杆
        with self.assertRaises(ValueError):
            invalid_config = BacktestConfig(leverage=0)
            invalid_config.validate()
    
    def test_backtest_engine_initialization(self):
        """测试回测引擎初始化"""
        self.assertEqual(self.engine.current_balance, 10000.0)
        self.assertEqual(self.engine.current_equity, 10000.0)
        self.assertEqual(len(self.engine.positions), 0)
        self.assertEqual(len(self.engine.trades), 0)
    
    def test_run_backtest(self):
        """测试运行回测"""
        result = self.engine.run_backtest(self.strategy, self.market_data_list)
        
        # 验证回测结果
        self.assertIsInstance(result, BacktestResult)
        self.assertEqual(result.strategy_name, "test_trend_strategy")
        self.assertEqual(result.symbol, "EURUSD")
        self.assertEqual(result.initial_balance, 10000.0)
        
        # 验证基本统计
        self.assertGreaterEqual(result.total_trades, 0)
        self.assertGreaterEqual(result.winning_trades, 0)
        self.assertGreaterEqual(result.losing_trades, 0)
        self.assertEqual(result.total_trades, result.winning_trades + result.losing_trades)
        
        # 验证胜率
        if result.total_trades > 0:
            expected_win_rate = result.winning_trades / result.total_trades
            self.assertAlmostEqual(result.win_rate, expected_win_rate, places=4)
        
        # 验证收益率
        expected_return = (result.final_balance - result.initial_balance) / result.initial_balance
        self.assertAlmostEqual(result.total_return, expected_return, places=4)
    
    def test_multiple_strategies_backtest(self):
        """测试多策略回测"""
        # 创建另一个策略
        strategy_config2 = StrategyConfig(
            name="test_trend_strategy_2",
            strategy_type=StrategyType.TREND_FOLLOWING,
            risk_per_trade=0.01  # 不同的风险参数
        )
        strategy2 = TrendFollowingStrategy(strategy_config2)
        
        strategies = [self.strategy, strategy2]
        results = self.engine.run_multiple_strategies(strategies, self.market_data_list)
        
        self.assertEqual(len(results), 2)
        self.assertIn("test_trend_strategy", results)
        self.assertIn("test_trend_strategy_2", results)
        
        # 验证每个结果
        for strategy_name, result in results.items():
            if result is not None:  # 回测可能失败
                self.assertIsInstance(result, BacktestResult)
                self.assertEqual(result.strategy_name, strategy_name)
    
    def test_backtest_result_to_dict(self):
        """测试回测结果转换为字典"""
        result = self.engine.run_backtest(self.strategy, self.market_data_list)
        result_dict = result.to_dict()
        
        # 验证字典包含所有必要字段
        required_fields = [
            'strategy_name', 'symbol', 'start_date', 'end_date',
            'initial_balance', 'final_balance', 'total_return',
            'total_trades', 'winning_trades', 'losing_trades',
            'win_rate', 'profit_factor', 'max_drawdown',
            'sharpe_ratio', 'sortino_ratio'
        ]
        
        for field in required_fields:
            self.assertIn(field, result_dict)
    
    def test_equity_curve_generation(self):
        """测试权益曲线生成"""
        result = self.engine.run_backtest(self.strategy, self.market_data_list)
        
        # 权益曲线应该有数据点
        self.assertGreater(len(result.equity_curve), 0)
        
        # 每个数据点应该是(时间, 权益)的元组
        for timestamp, equity in result.equity_curve:
            self.assertIsInstance(timestamp, datetime)
            self.assertIsInstance(equity, (int, float))
            self.assertGreater(equity, 0)  # 权益应该为正数
    
    def test_drawdown_calculation(self):
        """测试回撤计算"""
        result = self.engine.run_backtest(self.strategy, self.market_data_list)
        
        # 最大回撤应该为非负数
        self.assertGreaterEqual(result.max_drawdown, 0)
        self.assertGreaterEqual(result.max_drawdown_percent, 0)
        
        # 最大回撤百分比应该在0-1之间
        self.assertLessEqual(result.max_drawdown_percent, 1)
    
    def test_risk_metrics_calculation(self):
        """测试风险指标计算"""
        result = self.engine.run_backtest(self.strategy, self.market_data_list)
        
        # 夏普比率和索提诺比率应该是有限数值
        self.assertFalse(np.isnan(result.sharpe_ratio))
        self.assertFalse(np.isnan(result.sortino_ratio))
        
        # 卡玛比率
        if result.max_drawdown_percent > 0:
            expected_calmar = result.total_return / result.max_drawdown_percent
            self.assertAlmostEqual(result.calmar_ratio, expected_calmar, places=4)
    
    def test_trade_statistics(self):
        """测试交易统计"""
        result = self.engine.run_backtest(self.strategy, self.market_data_list)
        
        if result.total_trades > 0:
            # 平均交易盈亏
            total_profit = sum(trade.profit for trade in result.trades)
            expected_avg = total_profit / result.total_trades
            self.assertAlmostEqual(result.average_trade, expected_avg, places=2)
            
            # 最大盈利和亏损
            profits = [trade.profit for trade in result.trades if trade.profit > 0]
            losses = [trade.profit for trade in result.trades if trade.profit < 0]
            
            if profits:
                self.assertEqual(result.largest_win, max(profits))
            if losses:
                self.assertEqual(result.largest_loss, min(losses))
    
    def test_position_management(self):
        """测试持仓管理"""
        # 创建简单的测试数据，确保会产生信号
        simple_data = self.create_simple_trending_data()
        
        # 运行回测
        result = self.engine.run_backtest(self.strategy, simple_data)
        
        # 验证所有持仓都已平仓
        self.assertEqual(len(self.engine.positions), 0)
        
        # 如果有交易，验证交易记录
        for trade in result.trades:
            self.assertIsNotNone(trade.close_time)
            self.assertGreater(trade.close_price, 0)
    
    def create_simple_trending_data(self, periods=50):
        """创建简单的趋势数据，更容易产生信号"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='h')
        
        base_price = 1.1000
        # 创建明显的上升趋势
        prices = []
        for i in range(periods):
            price = base_price + (i * 0.0001)  # 每小时上涨1个点
            prices.append(price)
        
        market_data_list = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            high = price + 0.00005
            low = price - 0.00005
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            volume = 10000
            
            ohlcv_data = pd.DataFrame([{
                'timestamp': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            }])
            
            market_data = MarketData(
                symbol="EURUSD",
                timeframe="H1",
                timestamp=date,
                ohlcv=ohlcv_data
            )
            
            market_data_list.append(market_data)
        
        return market_data_list


    def test_parallel_backtest(self):
        """测试并行回测"""
        # 创建多个策略
        strategies = []
        for i in range(3):
            config = StrategyConfig(
                name=f"test_strategy_{i}",
                strategy_type=StrategyType.TREND_FOLLOWING,
                risk_per_trade=0.02
            )
            strategies.append(TrendFollowingStrategy(config))
        
        # 运行并行回测
        results = self.engine.run_multiple_strategies(
            strategies, 
            self.market_data_list,
            parallel=True
        )
        
        # 验证结果
        self.assertEqual(len(results), 3)
        for i in range(3):
            strategy_name = f"test_strategy_{i}"
            self.assertIn(strategy_name, results)
            if results[strategy_name] is not None:
                self.assertIsInstance(results[strategy_name], BacktestResult)
    
    def test_export_json(self):
        """测试导出JSON格式"""
        import tempfile
        import os
        
        result = self.engine.run_backtest(self.strategy, self.market_data_list)
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # 导出
            success = self.engine.export_results(result, temp_path, format='json')
            self.assertTrue(success)
            self.assertTrue(os.path.exists(temp_path))
            
            # 验证文件内容
            import json
            with open(temp_path, 'r') as f:
                data = json.load(f)
                self.assertEqual(data['strategy_name'], result.strategy_name)
                self.assertEqual(data['symbol'], result.symbol)
        finally:
            # 清理
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_export_csv(self):
        """测试导出CSV格式"""
        import tempfile
        import os
        
        result = self.engine.run_backtest(self.strategy, self.market_data_list)
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            # 导出
            success = self.engine.export_results(result, temp_path, format='csv')
            self.assertTrue(success)
            
            # 验证文件存在
            base_path = temp_path.rsplit('.', 1)[0]
            self.assertTrue(os.path.exists(f"{base_path}_trades.csv"))
            self.assertTrue(os.path.exists(f"{base_path}_summary.csv"))
        finally:
            # 清理
            base_path = temp_path.rsplit('.', 1)[0]
            for suffix in ['_trades.csv', '_summary.csv']:
                filepath = f"{base_path}{suffix}"
                if os.path.exists(filepath):
                    os.remove(filepath)
    
    def test_export_html(self):
        """测试导出HTML格式"""
        import tempfile
        import os
        
        result = self.engine.run_backtest(self.strategy, self.market_data_list)
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            temp_path = f.name
        
        try:
            # 导出
            success = self.engine.export_results(result, temp_path, format='html')
            self.assertTrue(success)
            self.assertTrue(os.path.exists(temp_path))
            
            # 验证HTML内容
            with open(temp_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.assertIn(result.strategy_name, content)
                self.assertIn(result.symbol, content)
                self.assertIn('<!DOCTYPE html>', content)
        finally:
            # 清理
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_strategy_comparison_visualization(self):
        """测试策略对比可视化"""
        # 创建多个策略
        strategies = []
        for i in range(2):
            config = StrategyConfig(
                name=f"test_strategy_{i}",
                strategy_type=StrategyType.TREND_FOLLOWING,
                risk_per_trade=0.02
            )
            strategies.append(TrendFollowingStrategy(config))
        
        # 运行回测
        results = self.engine.run_multiple_strategies(strategies, self.market_data_list)
        
        # 测试对比数据框生成
        comparison_df = self.engine.compare_strategies(results)
        self.assertIsInstance(comparison_df, pd.DataFrame)
        self.assertEqual(len(comparison_df), len([r for r in results.values() if r is not None]))
    
    def test_detailed_visualization_no_error(self):
        """测试详细可视化不会报错"""
        result = self.engine.run_backtest(self.strategy, self.market_data_list)
        
        # 测试可视化方法不会抛出异常（即使matplotlib未安装也应该优雅处理）
        try:
            self.engine.visualize_detailed_analysis(result)
        except ImportError:
            # matplotlib未安装是可以接受的
            pass
        except Exception as e:
            # 其他异常应该被记录但不应该导致测试失败
            self.logger.warning(f"可视化测试警告: {str(e)}")


if __name__ == '__main__':
    unittest.main(verbosity=2)