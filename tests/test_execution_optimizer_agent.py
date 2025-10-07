"""
执行优化Agent测试
测试执行优化Agent的各项功能
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

from src.agents.execution_optimizer import (
    ExecutionOptimizerAgent, SlippagePredictor, LiquidityAnalyzer, 
    TimingOptimizer, OrderSplitter, ExecutionMethod, LiquidityLevel,
    SlippageAnalysis, LiquidityAnalysis, TimingAnalysis, OrderSplit
)
from src.core.models import MarketData, Signal, Account
from src.core.exceptions import OrderException


class TestSlippagePredictor(unittest.TestCase):
    """滑点预测器测试"""
    
    def setUp(self):
        self.predictor = SlippagePredictor(lookback_period=50)
        self.market_data = self._create_mock_market_data()
    
    def _create_mock_market_data(self) -> MarketData:
        """创建模拟市场数据"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        ohlcv = pd.DataFrame({
            'open': np.random.uniform(1.1000, 1.1100, 100),
            'high': np.random.uniform(1.1050, 1.1150, 100),
            'low': np.random.uniform(1.0950, 1.1050, 100),
            'close': np.random.uniform(1.1000, 1.1100, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        }, index=dates)
        
        return MarketData(
            symbol="EURUSD",
            timeframe="H1",
            timestamp=datetime.now(),
            ohlcv=ohlcv,
            indicators={'atr': 0.0015},
            spread=0.0002
        )
    
    def test_predict_slippage_basic(self):
        """测试基础滑点预测"""
        result = self.predictor.predict_slippage("EURUSD", self.market_data, 1.0, 1)
        
        self.assertIsInstance(result, SlippageAnalysis)
        self.assertGreater(result.expected_slippage, 0)
        self.assertGreaterEqual(result.slippage_probability, 0)
        self.assertLessEqual(result.slippage_probability, 1)
        self.assertGreater(result.worst_case_slippage, result.expected_slippage)
    
    def test_predict_slippage_with_volume_impact(self):
        """测试成交量对滑点的影响"""
        small_volume_result = self.predictor.predict_slippage("EURUSD", self.market_data, 1.0, 1)
        large_volume_result = self.predictor.predict_slippage("EURUSD", self.market_data, 10.0, 1)
        
        # 大成交量应该有更高的滑点
        self.assertGreater(large_volume_result.expected_slippage, small_volume_result.expected_slippage)
    
    def test_record_actual_slippage(self):
        """测试记录实际滑点"""
        symbol = "EURUSD"
        expected_price = 1.1000
        actual_price = 1.1002
        timestamp = datetime.now()
        
        self.predictor.record_actual_slippage(symbol, expected_price, actual_price, timestamp)
        
        # 检查历史记录
        self.assertIn(symbol, self.predictor.slippage_history)
        self.assertEqual(len(self.predictor.slippage_history[symbol]), 1)
        
        record = self.predictor.slippage_history[symbol][0]
        self.assertEqual(record[0], timestamp)
        self.assertEqual(record[1], expected_price)
        self.assertEqual(record[2], abs(actual_price - expected_price))


class TestLiquidityAnalyzer(unittest.TestCase):
    """流动性分析器测试"""
    
    def setUp(self):
        self.analyzer = LiquidityAnalyzer()
        self.market_data = self._create_mock_market_data()
    
    def _create_mock_market_data(self) -> MarketData:
        """创建模拟市场数据"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        ohlcv = pd.DataFrame({
            'open': np.random.uniform(1.1000, 1.1100, 100),
            'high': np.random.uniform(1.1050, 1.1150, 100),
            'low': np.random.uniform(1.0950, 1.1050, 100),
            'close': np.random.uniform(1.1000, 1.1100, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        }, index=dates)
        
        return MarketData(
            symbol="EURUSD",
            timeframe="H1",
            timestamp=datetime.now(),
            ohlcv=ohlcv,
            spread=0.0002
        )
    
    def test_analyze_liquidity_basic(self):
        """测试基础流动性分析"""
        result = self.analyzer.analyze_liquidity("EURUSD", self.market_data)
        
        self.assertIsInstance(result, LiquidityAnalysis)
        self.assertIsInstance(result.liquidity_level, LiquidityLevel)
        self.assertGreater(result.bid_ask_spread, 0)
        self.assertIn('bid_depth', result.market_depth)
        self.assertIn('ask_depth', result.market_depth)
        self.assertGreater(result.impact_cost, 0)
    
    def test_liquidity_level_determination(self):
        """测试流动性水平判断"""
        # 测试高流动性场景
        high_liquidity_data = self.market_data
        high_liquidity_data.spread = 0.0001  # 很小的点差
        
        result = self.analyzer.analyze_liquidity("EURUSD", high_liquidity_data)
        
        # 应该是中等或高流动性
        self.assertIn(result.liquidity_level, [LiquidityLevel.MEDIUM, LiquidityLevel.HIGH])


class TestTimingOptimizer(unittest.TestCase):
    """时机优化器测试"""
    
    def setUp(self):
        self.optimizer = TimingOptimizer()
        self.signal = self._create_mock_signal()
        self.market_data = self._create_mock_market_data()
    
    def _create_mock_signal(self) -> Signal:
        """创建模拟信号"""
        return Signal(
            strategy_id="test_strategy",
            symbol="EURUSD",
            direction=1,
            strength=0.8,
            entry_price=1.1000,
            sl=1.0950,
            tp=1.1100,
            size=1.0,
            confidence=0.7,
            timestamp=datetime.now()
        )
    
    def _create_mock_market_data(self) -> MarketData:
        """创建模拟市场数据"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        ohlcv = pd.DataFrame({
            'open': np.random.uniform(1.1000, 1.1100, 100),
            'high': np.random.uniform(1.1050, 1.1150, 100),
            'low': np.random.uniform(1.0950, 1.1050, 100),
            'close': np.random.uniform(1.1000, 1.1100, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        }, index=dates)
        
        return MarketData(
            symbol="EURUSD",
            timeframe="H1",
            timestamp=datetime.now(),
            ohlcv=ohlcv,
            indicators={'atr': 0.0015}
        )
    
    def test_analyze_timing_basic(self):
        """测试基础时机分析"""
        result = self.optimizer.analyze_timing(self.signal, self.market_data)
        
        self.assertIsInstance(result, TimingAnalysis)
        self.assertIsInstance(result.optimal_entry_time, datetime)
        self.assertGreaterEqual(result.entry_score, 0)
        self.assertLessEqual(result.entry_score, 100)
        self.assertGreaterEqual(result.market_momentum, -1)
        self.assertLessEqual(result.market_momentum, 1)
        self.assertGreater(result.volatility_forecast, 0)
        self.assertGreaterEqual(result.execution_urgency, 0)
        self.assertLessEqual(result.execution_urgency, 1)
    
    def test_timing_with_high_confidence_signal(self):
        """测试高置信度信号的时机分析"""
        high_confidence_signal = self.signal
        high_confidence_signal.confidence = 0.9
        high_confidence_signal.strength = 0.9
        
        result = self.optimizer.analyze_timing(high_confidence_signal, self.market_data)
        
        # 高置信度信号应该有较高的紧急程度
        self.assertGreater(result.execution_urgency, 0.5)


class TestOrderSplitter(unittest.TestCase):
    """订单拆分器测试"""
    
    def setUp(self):
        self.splitter = OrderSplitter()
        self.signal = self._create_mock_signal()
        self.liquidity_analysis = self._create_mock_liquidity_analysis()
    
    def _create_mock_signal(self) -> Signal:
        """创建模拟信号"""
        return Signal(
            strategy_id="test_strategy",
            symbol="EURUSD",
            direction=1,
            strength=0.8,
            entry_price=1.1000,
            sl=1.0950,
            tp=1.1100,
            size=20.0,  # 大订单
            confidence=0.7,
            timestamp=datetime.now()
        )
    
    def _create_mock_liquidity_analysis(self) -> LiquidityAnalysis:
        """创建模拟流动性分析"""
        return LiquidityAnalysis(
            liquidity_level=LiquidityLevel.MEDIUM,
            bid_ask_spread=0.0002,
            market_depth={'bid_depth': 1000, 'ask_depth': 1000, 'total_depth': 2000},
            volume_profile={'average_volume': 2000, 'volume_std': 400, 
                          'current_volume': 2000, 'volume_ratio': 1.0},
            impact_cost=0.0001
        )
    
    def test_split_large_order(self):
        """测试大订单拆分"""
        result = self.splitter.split_order(self.signal, self.liquidity_analysis, max_chunk_size=5.0)
        
        self.assertIsInstance(result, OrderSplit)
        self.assertGreater(len(result.child_orders), 1)  # 应该被拆分
        self.assertEqual(result.total_volume, self.signal.size)
        self.assertEqual(len(result.child_orders), len(result.execution_schedule))
        
        # 检查子订单总量
        total_child_volume = sum(order['volume'] for order in result.child_orders)
        self.assertAlmostEqual(total_child_volume, self.signal.size, places=6)
    
    def test_split_small_order(self):
        """测试小订单不拆分"""
        small_signal = self.signal
        small_signal.size = 3.0  # 小订单
        
        result = self.splitter.split_order(small_signal, self.liquidity_analysis, max_chunk_size=5.0)
        
        self.assertEqual(len(result.child_orders), 1)  # 不应该被拆分
        self.assertEqual(result.child_orders[0]['volume'], small_signal.size)


class TestExecutionOptimizerAgent(unittest.TestCase):
    """执行优化Agent测试"""
    
    def setUp(self):
        self.agent = ExecutionOptimizerAgent()
        self.signal = self._create_mock_signal()
        self.market_data = self._create_mock_market_data()
        self.account = self._create_mock_account()
    
    def _create_mock_signal(self) -> Signal:
        """创建模拟信号"""
        return Signal(
            strategy_id="test_strategy",
            symbol="EURUSD",
            direction=1,
            strength=0.8,
            entry_price=1.1000,
            sl=1.0950,
            tp=1.1100,
            size=5.0,
            confidence=0.7,
            timestamp=datetime.now()
        )
    
    def _create_mock_market_data(self) -> MarketData:
        """创建模拟市场数据"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        ohlcv = pd.DataFrame({
            'open': np.random.uniform(1.1000, 1.1100, 100),
            'high': np.random.uniform(1.1050, 1.1150, 100),
            'low': np.random.uniform(1.0950, 1.1050, 100),
            'close': np.random.uniform(1.1000, 1.1100, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        }, index=dates)
        
        return MarketData(
            symbol="EURUSD",
            timeframe="H1",
            timestamp=datetime.now(),
            ohlcv=ohlcv,
            indicators={'atr': 0.0015},
            spread=0.0002
        )
    
    def _create_mock_account(self) -> Account:
        """创建模拟账户"""
        return Account(
            account_id="12345",
            balance=10000.0,
            equity=10000.0,
            margin=1000.0,
            free_margin=9000.0,
            margin_level=1000.0,
            currency="USD",
            leverage=100
        )
    
    def test_agent_initialization(self):
        """测试Agent初始化"""
        agent = ExecutionOptimizerAgent()
        
        self.assertIsNotNone(agent.slippage_predictor)
        self.assertIsNotNone(agent.liquidity_analyzer)
        self.assertIsNotNone(agent.timing_optimizer)
        self.assertIsNotNone(agent.order_splitter)
        self.assertEqual(len(agent.execution_history), 0)
    
    def test_agent_initialization_with_config(self):
        """测试带配置的Agent初始化"""
        config = {
            'slippage_lookback': 200,
            'max_chunk_size': 20.0,
            'enable_order_splitting': False
        }
        
        agent = ExecutionOptimizerAgent(config)
        
        self.assertEqual(agent.config['slippage_lookback'], 200)
        self.assertEqual(agent.config['max_chunk_size'], 20.0)
        self.assertFalse(agent.config['enable_order_splitting'])
    
    def test_optimize_execution_basic(self):
        """测试基础执行优化"""
        execution_plan = self.agent.optimize_execution(self.signal, self.market_data, self.account)
        
        self.assertIsNotNone(execution_plan)
        self.assertEqual(execution_plan.signal, self.signal)
        self.assertIsInstance(execution_plan.method, ExecutionMethod)
        self.assertIsNotNone(execution_plan.slippage_analysis)
        self.assertIsNotNone(execution_plan.liquidity_analysis)
        self.assertIsNotNone(execution_plan.timing_analysis)
        self.assertGreater(execution_plan.estimated_cost, 0)
        self.assertGreaterEqual(execution_plan.execution_priority, 1)
        self.assertLessEqual(execution_plan.execution_priority, 10)
    
    def test_optimize_execution_with_large_order(self):
        """测试大订单执行优化"""
        large_signal = self.signal
        large_signal.size = 25.0  # 大订单
        
        execution_plan = self.agent.optimize_execution(large_signal, self.market_data, self.account)
        
        # 大订单应该被拆分
        if self.agent.config.get('enable_order_splitting', True):
            self.assertGreater(len(execution_plan.order_splits), 0)
    
    def test_should_delay_execution(self):
        """测试延迟执行判断"""
        execution_plan = self.agent.optimize_execution(self.signal, self.market_data, self.account)
        
        should_delay = self.agent.should_delay_execution(execution_plan)
        
        self.assertIsInstance(should_delay, bool)
    
    def test_record_execution_result(self):
        """测试记录执行结果"""
        executed_price = 1.1002
        execution_time = datetime.now()
        slippage = 0.0002
        
        initial_history_length = len(self.agent.execution_history)
        
        self.agent.record_execution_result(self.signal, executed_price, execution_time, slippage)
        
        # 检查历史记录增加
        self.assertEqual(len(self.agent.execution_history), initial_history_length + 1)
        
        # 检查记录内容
        latest_record = self.agent.execution_history[-1]
        self.assertEqual(latest_record['symbol'], self.signal.symbol)
        self.assertEqual(latest_record['executed_price'], executed_price)
        self.assertEqual(latest_record['slippage'], slippage)
    
    def test_get_execution_statistics_no_data(self):
        """测试无数据时的执行统计"""
        stats = self.agent.get_execution_statistics()
        
        self.assertEqual(stats['status'], 'no_data')
    
    def test_get_execution_statistics_with_data(self):
        """测试有数据时的执行统计"""
        # 添加一些执行记录
        for i in range(5):
            self.agent.record_execution_result(
                self.signal, 
                1.1000 + i * 0.0001, 
                datetime.now() - timedelta(days=i),
                i * 0.0001
            )
        
        stats = self.agent.get_execution_statistics("EURUSD")
        
        self.assertEqual(stats['status'], 'success')
        self.assertEqual(stats['total_executions'], 5)
        self.assertIn('average_slippage', stats)
        self.assertIn('median_slippage', stats)
        self.assertIn('max_slippage', stats)
        self.assertIn('min_slippage', stats)
    
    def test_get_execution_statistics_filtered_by_symbol(self):
        """测试按品种过滤的执行统计"""
        # 添加不同品种的执行记录
        eurusd_signal = self.signal
        gbpusd_signal = Signal(
            strategy_id="test_strategy",
            symbol="GBPUSD",
            direction=1,
            strength=0.8,
            entry_price=1.2500,
            sl=1.2450,
            tp=1.2600,
            size=1.0,
            confidence=0.7,
            timestamp=datetime.now()
        )
        
        self.agent.record_execution_result(eurusd_signal, 1.1002, datetime.now(), 0.0002)
        self.agent.record_execution_result(gbpusd_signal, 1.2502, datetime.now(), 0.0002)
        
        eurusd_stats = self.agent.get_execution_statistics("EURUSD")
        gbpusd_stats = self.agent.get_execution_statistics("GBPUSD")
        
        self.assertEqual(eurusd_stats['total_executions'], 1)
        self.assertEqual(gbpusd_stats['total_executions'], 1)
    
    def test_clear_history(self):
        """测试清空历史记录"""
        # 添加一些记录
        self.agent.record_execution_result(self.signal, 1.1002, datetime.now(), 0.0002)
        
        self.assertGreater(len(self.agent.execution_history), 0)
        
        self.agent.clear_history()
        
        self.assertEqual(len(self.agent.execution_history), 0)
        self.assertEqual(len(self.agent.slippage_predictor.slippage_history), 0)
    
    def test_optimize_execution_exception_handling(self):
        """测试执行优化异常处理"""
        # 使用patch来模拟内部方法抛出异常
        with patch.object(self.agent.slippage_predictor, 'predict_slippage', side_effect=Exception("Test exception")):
            with self.assertRaises(OrderException):
                self.agent.optimize_execution(self.signal, self.market_data, self.account)


if __name__ == '__main__':
    unittest.main()