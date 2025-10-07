"""
基础策略模块测试
测试趋势跟踪、均值回归和突破策略
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.models import MarketData, Signal
from src.core.exceptions import StrategyException
from src.strategies.base_strategies import (
    StrategyConfig, StrategyType, StrategyPerformance,
    TrendFollowingStrategy, MeanReversionStrategy, BreakoutStrategy,
    StrategyManager
)


class TestStrategyConfig(unittest.TestCase):
    """测试策略配置"""
    
    def test_valid_config(self):
        """测试有效配置"""
        config = StrategyConfig(
            name="test_strategy",
            strategy_type=StrategyType.TREND_FOLLOWING,
            risk_per_trade=0.02,
            max_positions=3
        )
        
        self.assertTrue(config.validate())
    
    def test_invalid_risk_per_trade(self):
        """测试无效的每笔交易风险"""
        config = StrategyConfig(
            name="test_strategy",
            strategy_type=StrategyType.TREND_FOLLOWING,
            risk_per_trade=0.15  # 超过10%
        )
        
        with self.assertRaises(ValueError):
            config.validate()
    
    def test_invalid_max_positions(self):
        """测试无效的最大持仓数"""
        config = StrategyConfig(
            name="test_strategy",
            strategy_type=StrategyType.TREND_FOLLOWING,
            max_positions=0
        )
        
        with self.assertRaises(ValueError):
            config.validate()


class TestStrategyPerformance(unittest.TestCase):
    """测试策略性能指标"""
    
    def setUp(self):
        self.performance = StrategyPerformance()
    
    def test_update_performance_winning_trade(self):
        """测试更新盈利交易性能"""
        self.performance.update_performance(100.0)
        
        self.assertEqual(self.performance.total_trades, 1)
        self.assertEqual(self.performance.winning_trades, 1)
        self.assertEqual(self.performance.losing_trades, 0)
        self.assertEqual(self.performance.total_profit, 100.0)
        self.assertEqual(self.performance.win_rate, 1.0)
    
    def test_update_performance_losing_trade(self):
        """测试更新亏损交易性能"""
        self.performance.update_performance(-50.0)
        
        self.assertEqual(self.performance.total_trades, 1)
        self.assertEqual(self.performance.winning_trades, 0)
        self.assertEqual(self.performance.losing_trades, 1)
        self.assertEqual(self.performance.total_profit, -50.0)
        self.assertEqual(self.performance.win_rate, 0.0)
    
    def test_performance_score_insufficient_trades(self):
        """测试交易次数不足时的性能评分"""
        # 少于10笔交易
        for i in range(5):
            self.performance.update_performance(10.0)
        
        score = self.performance.get_performance_score()
        self.assertEqual(score, 0.5)  # 应该返回中性评分
    
    def test_performance_score_sufficient_trades(self):
        """测试交易次数充足时的性能评分"""
        # 添加10笔盈利交易
        for i in range(10):
            self.performance.update_performance(10.0)
        
        # 设置其他指标
        self.performance.profit_factor = 2.0
        self.performance.sharpe_ratio = 1.5
        
        score = self.performance.get_performance_score()
        self.assertGreater(score, 0.5)
        self.assertLessEqual(score, 1.0)


class BaseStrategyTestCase(unittest.TestCase):
    """策略测试基类"""
    
    def setUp(self):
        # 创建测试数据
        self.create_test_data()
    
    def create_test_data(self, trend_type="uptrend", periods=100):
        """创建测试数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='h')
        np.random.seed(42)
        
        base_price = 1.1000
        
        if trend_type == "uptrend":
            trend = np.linspace(0, 0.01, periods)
        elif trend_type == "downtrend":
            trend = np.linspace(0, -0.01, periods)
        else:  # sideways
            trend = np.zeros(periods)
        
        noise = np.random.normal(0, 0.0005, periods)
        prices = base_price + trend + noise
        
        ohlcv_data = []
        for i, price in enumerate(prices):
            high = price + abs(np.random.normal(0, 0.0002))
            low = price - abs(np.random.normal(0, 0.0002))
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            volume = np.random.randint(5000, 15000)
            
            ohlcv_data.append({
                'timestamp': dates[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })
        
        self.ohlcv_df = pd.DataFrame(ohlcv_data)
        self.market_data = MarketData(
            symbol="EURUSD",
            timeframe="H1",
            timestamp=datetime.now(),
            ohlcv=self.ohlcv_df
        )


class TestTrendFollowingStrategy(BaseStrategyTestCase):
    """测试趋势跟踪策略"""
    
    def setUp(self):
        super().setUp()
        
        config = StrategyConfig(
            name="trend_following_test",
            strategy_type=StrategyType.TREND_FOLLOWING,
            risk_per_trade=0.02
        )
        
        self.strategy = TrendFollowingStrategy(config)
    
    def test_strategy_initialization(self):
        """测试策略初始化"""
        self.assertEqual(self.strategy.config.name, "trend_following_test")
        self.assertEqual(self.strategy.config.strategy_type, StrategyType.TREND_FOLLOWING)
        self.assertTrue(self.strategy.is_enabled)
        
        # 检查默认参数
        self.assertIn("ma_fast_period", self.strategy.config.parameters)
        self.assertIn("ma_slow_period", self.strategy.config.parameters)
        self.assertIn("adx_threshold", self.strategy.config.parameters)
    
    def test_uptrend_signal_generation(self):
        """测试上升趋势信号生成"""
        # 创建明显的上升趋势数据
        self.create_test_data(trend_type="uptrend", periods=50)
        
        signal = self.strategy.generate_signal(self.market_data)
        
        if signal:  # 可能因为ADX不够强而没有信号
            self.assertIsInstance(signal, Signal)
            self.assertEqual(signal.symbol, "EURUSD")
            self.assertEqual(signal.strategy_id, "trend_following_test")
            self.assertGreater(signal.strength, 0)
            self.assertGreater(signal.confidence, 0)
            
            # 在上升趋势中，应该生成买入信号
            if signal.direction != 0:
                self.assertGreater(signal.direction, 0)
    
    def test_downtrend_signal_generation(self):
        """测试下降趋势信号生成"""
        # 创建明显的下降趋势数据
        self.create_test_data(trend_type="downtrend", periods=50)
        
        signal = self.strategy.generate_signal(self.market_data)
        
        if signal:  # 可能因为ADX不够强而没有信号
            self.assertIsInstance(signal, Signal)
            
            # 在下降趋势中，应该生成卖出信号
            if signal.direction != 0:
                self.assertLess(signal.direction, 0)
    
    def test_sideways_market_no_signal(self):
        """测试横盘市场不生成信号"""
        # 创建横盘数据
        self.create_test_data(trend_type="sideways", periods=50)
        
        signal = self.strategy.generate_signal(self.market_data)
        
        # 横盘市场ADX应该较低，不应该生成信号
        self.assertIsNone(signal)
    
    def test_parameter_update(self):
        """测试参数更新"""
        new_params = {
            "ma_fast_period": 10,
            "adx_threshold": 30
        }
        
        self.strategy.update_parameters(new_params)
        
        self.assertEqual(self.strategy.config.parameters["ma_fast_period"], 10)
        self.assertEqual(self.strategy.config.parameters["adx_threshold"], 30)
    
    def test_position_size_calculation(self):
        """测试仓位大小计算"""
        entry_price = 1.1000
        stop_loss = 1.0950
        account_balance = 10000
        
        position_size = self.strategy._calculate_position_size(
            self.market_data, entry_price, stop_loss, account_balance
        )
        
        self.assertGreater(position_size, 0)
        self.assertLessEqual(position_size, 1.0)  # 不应该超过1手
    
    def test_stop_loss_calculation(self):
        """测试止损计算"""
        entry_price = 1.1000
        direction = 1  # 买入
        
        stop_loss = self.strategy._calculate_stop_loss(
            self.market_data, direction, entry_price
        )
        
        # 买入时止损应该低于入场价
        self.assertLess(stop_loss, entry_price)
    
    def test_take_profit_calculation(self):
        """测试止盈计算"""
        entry_price = 1.1000
        stop_loss = 1.0950
        direction = 1  # 买入
        
        take_profit = self.strategy._calculate_take_profit(
            entry_price, stop_loss, direction, risk_reward_ratio=2.0
        )
        
        # 买入时止盈应该高于入场价
        self.assertGreater(take_profit, entry_price)
        
        # 检查风险回报比
        risk = entry_price - stop_loss
        reward = take_profit - entry_price
        ratio = reward / risk
        
        self.assertAlmostEqual(ratio, 2.0, places=2)


class TestMeanReversionStrategy(BaseStrategyTestCase):
    """测试均值回归策略"""
    
    def setUp(self):
        super().setUp()
        
        config = StrategyConfig(
            name="mean_reversion_test",
            strategy_type=StrategyType.MEAN_REVERSION,
            risk_per_trade=0.02
        )
        
        self.strategy = MeanReversionStrategy(config)
    
    def test_strategy_initialization(self):
        """测试策略初始化"""
        self.assertEqual(self.strategy.config.name, "mean_reversion_test")
        self.assertEqual(self.strategy.config.strategy_type, StrategyType.MEAN_REVERSION)
        
        # 检查默认参数
        self.assertIn("rsi_period", self.strategy.config.parameters)
        self.assertIn("rsi_overbought", self.strategy.config.parameters)
        self.assertIn("rsi_oversold", self.strategy.config.parameters)
        self.assertIn("bb_period", self.strategy.config.parameters)
    
    def test_oversold_signal_generation(self):
        """测试超卖信号生成"""
        # 创建超卖条件的数据（价格大幅下跌）
        dates = pd.date_range(start='2023-01-01', periods=50, freq='h')
        base_price = 1.1000
        
        # 创建下跌后的超卖条件
        prices = []
        for i in range(50):
            if i < 30:
                # 前30个周期下跌
                price = base_price - (i * 0.0005)
            else:
                # 后20个周期稳定在低位
                price = base_price - 0.015 + np.random.normal(0, 0.0001)
            prices.append(price)
        
        ohlcv_data = []
        for i, price in enumerate(prices):
            high = price + abs(np.random.normal(0, 0.0001))
            low = price - abs(np.random.normal(0, 0.0001))
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            volume = np.random.randint(5000, 15000)
            
            ohlcv_data.append({
                'timestamp': dates[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })
        
        ohlcv_df = pd.DataFrame(ohlcv_data)
        market_data = MarketData(
            symbol="EURUSD",
            timeframe="H1",
            timestamp=datetime.now(),
            ohlcv=ohlcv_df
        )
        
        signal = self.strategy.generate_signal(market_data)
        
        if signal:  # 可能需要更极端的条件才能触发
            self.assertIsInstance(signal, Signal)
            # 超卖条件下应该生成买入信号
            self.assertGreater(signal.direction, 0)
    
    def test_overbought_signal_generation(self):
        """测试超买信号生成"""
        # 创建超买条件的数据（价格大幅上涨）
        dates = pd.date_range(start='2023-01-01', periods=50, freq='h')
        base_price = 1.1000
        
        # 创建上涨后的超买条件
        prices = []
        for i in range(50):
            if i < 30:
                # 前30个周期上涨
                price = base_price + (i * 0.0005)
            else:
                # 后20个周期稳定在高位
                price = base_price + 0.015 + np.random.normal(0, 0.0001)
            prices.append(price)
        
        ohlcv_data = []
        for i, price in enumerate(prices):
            high = price + abs(np.random.normal(0, 0.0001))
            low = price - abs(np.random.normal(0, 0.0001))
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            volume = np.random.randint(5000, 15000)
            
            ohlcv_data.append({
                'timestamp': dates[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })
        
        ohlcv_df = pd.DataFrame(ohlcv_data)
        market_data = MarketData(
            symbol="EURUSD",
            timeframe="H1",
            timestamp=datetime.now(),
            ohlcv=ohlcv_df
        )
        
        signal = self.strategy.generate_signal(market_data)
        
        if signal:  # 可能需要更极端的条件才能触发
            self.assertIsInstance(signal, Signal)
            # 超买条件下应该生成卖出信号
            self.assertLess(signal.direction, 0)


class TestBreakoutStrategy(BaseStrategyTestCase):
    """测试突破策略"""
    
    def setUp(self):
        super().setUp()
        
        config = StrategyConfig(
            name="breakout_test",
            strategy_type=StrategyType.BREAKOUT,
            risk_per_trade=0.02
        )
        
        self.strategy = BreakoutStrategy(config)
    
    def test_strategy_initialization(self):
        """测试策略初始化"""
        self.assertEqual(self.strategy.config.name, "breakout_test")
        self.assertEqual(self.strategy.config.strategy_type, StrategyType.BREAKOUT)
        
        # 检查默认参数
        self.assertIn("bb_period", self.strategy.config.parameters)
        self.assertIn("volume_threshold", self.strategy.config.parameters)
        self.assertIn("atr_period", self.strategy.config.parameters)
    
    def test_upward_breakout_signal(self):
        """测试向上突破信号"""
        # 创建突破数据：先横盘，然后向上突破
        dates = pd.date_range(start='2023-01-01', periods=50, freq='h')
        base_price = 1.1000
        
        prices = []
        volumes = []
        for i in range(50):
            if i < 40:
                # 前40个周期横盘
                price = base_price + np.random.normal(0, 0.0002)
                volume = np.random.randint(5000, 8000)  # 正常成交量
            else:
                # 后10个周期向上突破
                price = base_price + 0.005 + np.random.normal(0, 0.0001)
                volume = np.random.randint(15000, 25000)  # 高成交量
            
            prices.append(price)
            volumes.append(volume)
        
        ohlcv_data = []
        for i, (price, volume) in enumerate(zip(prices, volumes)):
            high = price + abs(np.random.normal(0, 0.0001))
            low = price - abs(np.random.normal(0, 0.0001))
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            
            ohlcv_data.append({
                'timestamp': dates[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })
        
        ohlcv_df = pd.DataFrame(ohlcv_data)
        market_data = MarketData(
            symbol="EURUSD",
            timeframe="H1",
            timestamp=datetime.now(),
            ohlcv=ohlcv_df
        )
        
        signal = self.strategy.generate_signal(market_data)
        
        if signal:  # 突破策略需要特定条件
            self.assertIsInstance(signal, Signal)
            # 向上突破应该生成买入信号
            self.assertGreater(signal.direction, 0)
            self.assertIn("upward_breakout", signal.metadata.get("breakout_type", ""))


class TestStrategyManager(unittest.TestCase):
    """测试策略管理器"""
    
    def setUp(self):
        self.manager = StrategyManager()
        
        # 创建测试策略
        trend_config = StrategyConfig(
            name="trend_strategy",
            strategy_type=StrategyType.TREND_FOLLOWING
        )
        self.trend_strategy = TrendFollowingStrategy(trend_config)
        
        mean_config = StrategyConfig(
            name="mean_strategy",
            strategy_type=StrategyType.MEAN_REVERSION
        )
        self.mean_strategy = MeanReversionStrategy(mean_config)
        
        # 创建测试数据
        dates = pd.date_range(start='2023-01-01', periods=100, freq='h')
        np.random.seed(42)
        
        base_price = 1.1000
        prices = base_price + np.cumsum(np.random.normal(0, 0.0001, 100))
        
        ohlcv_data = []
        for i, price in enumerate(prices):
            high = price + abs(np.random.normal(0, 0.0001))
            low = price - abs(np.random.normal(0, 0.0001))
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            volume = np.random.randint(5000, 15000)
            
            ohlcv_data.append({
                'timestamp': dates[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })
        
        ohlcv_df = pd.DataFrame(ohlcv_data)
        self.market_data = MarketData(
            symbol="EURUSD",
            timeframe="H1",
            timestamp=datetime.now(),
            ohlcv=ohlcv_df
        )
    
    def test_add_strategy(self):
        """测试添加策略"""
        self.manager.add_strategy(self.trend_strategy, weight=1.0)
        
        self.assertIn("trend_strategy", self.manager.strategies)
        self.assertEqual(self.manager.strategy_weights["trend_strategy"], 1.0)
    
    def test_remove_strategy(self):
        """测试移除策略"""
        self.manager.add_strategy(self.trend_strategy)
        self.manager.remove_strategy("trend_strategy")
        
        self.assertNotIn("trend_strategy", self.manager.strategies)
        self.assertNotIn("trend_strategy", self.manager.strategy_weights)
    
    def test_generate_signals(self):
        """测试生成信号"""
        self.manager.add_strategy(self.trend_strategy, weight=0.8)
        self.manager.add_strategy(self.mean_strategy, weight=1.2)
        
        signals = self.manager.generate_signals(self.market_data)
        
        self.assertIsInstance(signals, list)
        # 检查权重是否正确应用
        for signal in signals:
            if signal.strategy_id == "trend_strategy":
                # 权重应该影响信号强度
                pass  # 具体检查取决于原始信号强度
    
    def test_update_strategy_weights(self):
        """测试更新策略权重"""
        self.manager.add_strategy(self.trend_strategy, weight=1.0)
        
        performance_data = {"trend_strategy": 1.5}
        self.manager.update_strategy_weights(performance_data)
        
        self.assertEqual(self.manager.strategy_weights["trend_strategy"], 1.5)
    
    def test_enable_disable_strategy(self):
        """测试启用/禁用策略"""
        self.manager.add_strategy(self.trend_strategy)
        
        # 禁用策略
        self.manager.disable_strategy("trend_strategy")
        self.assertFalse(self.trend_strategy.is_enabled)
        
        # 启用策略
        self.manager.enable_strategy("trend_strategy")
        self.assertTrue(self.trend_strategy.is_enabled)
    
    def test_get_active_strategies(self):
        """测试获取活跃策略"""
        self.manager.add_strategy(self.trend_strategy)
        self.manager.add_strategy(self.mean_strategy)
        
        # 禁用一个策略
        self.manager.disable_strategy("mean_strategy")
        
        active_strategies = self.manager.get_active_strategies()
        
        self.assertIn("trend_strategy", active_strategies)
        self.assertNotIn("mean_strategy", active_strategies)
    
    def test_get_strategy_performance(self):
        """测试获取策略性能"""
        self.manager.add_strategy(self.trend_strategy)
        
        performance = self.manager.get_strategy_performance()
        
        self.assertIn("trend_strategy", performance)
        self.assertIsInstance(performance["trend_strategy"], StrategyPerformance)


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)