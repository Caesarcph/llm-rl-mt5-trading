"""
技术指标模块测试
测试指标计算准确性和缓存机制
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

from src.core.models import MarketData
from src.core.exceptions import DataValidationError, IndicatorCalculationError
from src.strategies.indicators import (
    TechnicalIndicators, IndicatorCache, IndicatorResult, 
    IndicatorType, IndicatorConfig
)
from src.strategies.custom_indicators import MultiTimeframeAnalyzer, CustomIndicators


class TestIndicatorCache(unittest.TestCase):
    """测试指标缓存"""
    
    def setUp(self):
        self.cache = IndicatorCache(max_size=5)
    
    def test_cache_set_and_get(self):
        """测试缓存设置和获取"""
        # 创建测试结果
        result = IndicatorResult(
            name="SMA",
            symbol="EURUSD",
            timeframe="H1",
            timestamp=datetime.now(),
            values={"value": np.array([1.1000, 1.1010, 1.1020])}
        )
        
        # 设置缓存
        self.cache.set("EURUSD", "H1", "sma", {"period": 20}, result)
        
        # 获取缓存
        cached_result = self.cache.get("EURUSD", "H1", "sma", {"period": 20}, ttl=300)
        
        self.assertIsNotNone(cached_result)
        self.assertEqual(cached_result.name, "SMA")
        self.assertEqual(cached_result.symbol, "EURUSD")
    
    def test_cache_expiry(self):
        """测试缓存过期"""
        result = IndicatorResult(
            name="SMA",
            symbol="EURUSD",
            timeframe="H1",
            timestamp=datetime.now(),
            values={"value": np.array([1.1000])}
        )
        
        self.cache.set("EURUSD", "H1", "sma", {"period": 20}, result)
        
        # 使用很短的TTL
        cached_result = self.cache.get("EURUSD", "H1", "sma", {"period": 20}, ttl=0)
        
        self.assertIsNone(cached_result)
    
    def test_cache_max_size(self):
        """测试缓存最大容量"""
        # 添加超过最大容量的缓存项
        for i in range(10):
            result = IndicatorResult(
                name="SMA",
                symbol=f"SYMBOL{i}",
                timeframe="H1",
                timestamp=datetime.now(),
                values={"value": np.array([1.0])}
            )
            self.cache.set(f"SYMBOL{i}", "H1", "sma", {"period": 20}, result)
        
        # 检查缓存大小不超过最大值
        stats = self.cache.get_cache_stats()
        self.assertEqual(stats["size"], 5)
        self.assertEqual(stats["max_size"], 5)


class TestTechnicalIndicators(unittest.TestCase):
    """测试技术指标计算"""
    
    def setUp(self):
        self.indicators = TechnicalIndicators(use_cache=False)
        
        # 创建测试数据
        dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
        np.random.seed(42)  # 确保可重复性
        
        # 生成模拟价格数据
        base_price = 1.1000
        price_changes = np.random.normal(0, 0.0001, 100)
        prices = base_price + np.cumsum(price_changes)
        
        # 创建OHLCV数据
        ohlcv_data = []
        for i, price in enumerate(prices):
            high = price + abs(np.random.normal(0, 0.0001))
            low = price - abs(np.random.normal(0, 0.0001))
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            volume = np.random.randint(1000, 10000)
            
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
    
    def test_data_validation(self):
        """测试数据验证"""
        # 测试空数据
        empty_data = MarketData(
            symbol="EURUSD",
            timeframe="H1",
            timestamp=datetime.now(),
            ohlcv=pd.DataFrame()
        )
        
        with self.assertRaises(DataValidationError):
            self.indicators.calculate_sma(empty_data)
        
        # 测试缺少列的数据
        incomplete_df = pd.DataFrame({'open': [1.1], 'high': [1.1]})
        incomplete_data = MarketData(
            symbol="EURUSD",
            timeframe="H1",
            timestamp=datetime.now(),
            ohlcv=incomplete_df
        )
        
        with self.assertRaises(DataValidationError):
            self.indicators.calculate_sma(incomplete_data)
    
    def test_sma_calculation(self):
        """测试SMA计算"""
        result = self.indicators.calculate_sma(self.market_data, period=20)
        
        self.assertIsInstance(result, IndicatorResult)
        self.assertEqual(result.name, "SMA")
        self.assertEqual(result.symbol, "EURUSD")
        self.assertIn("value", result.values)
        
        # 验证SMA值的合理性
        sma_values = result.values["value"]
        self.assertEqual(len(sma_values), len(self.ohlcv_df))
        
        # 前19个值应该是NaN
        self.assertTrue(np.isnan(sma_values[:19]).all())
        
        # 第20个值应该是前20个收盘价的平均值
        expected_sma_20 = self.ohlcv_df['close'][:20].mean()
        self.assertAlmostEqual(sma_values[19], expected_sma_20, places=5)
    
    def test_ema_calculation(self):
        """测试EMA计算"""
        result = self.indicators.calculate_ema(self.market_data, period=20)
        
        self.assertIsInstance(result, IndicatorResult)
        self.assertEqual(result.name, "EMA")
        self.assertIn("value", result.values)
        
        # EMA值应该比SMA更快响应价格变化
        ema_values = result.values["value"]
        self.assertFalse(np.isnan(ema_values[-1]))
    
    def test_macd_calculation(self):
        """测试MACD计算"""
        result = self.indicators.calculate_macd(self.market_data)
        
        self.assertIsInstance(result, IndicatorResult)
        self.assertEqual(result.name, "MACD")
        self.assertIn("macd", result.values)
        self.assertIn("signal", result.values)
        self.assertIn("histogram", result.values)
        
        # 验证MACD组件
        macd = result.values["macd"]
        signal = result.values["signal"]
        histogram = result.values["histogram"]
        
        self.assertEqual(len(macd), len(self.ohlcv_df))
        self.assertEqual(len(signal), len(self.ohlcv_df))
        self.assertEqual(len(histogram), len(self.ohlcv_df))
    
    def test_rsi_calculation(self):
        """测试RSI计算"""
        result = self.indicators.calculate_rsi(self.market_data, period=14)
        
        self.assertIsInstance(result, IndicatorResult)
        self.assertEqual(result.name, "RSI")
        self.assertIn("value", result.values)
        
        # RSI值应该在0-100之间
        rsi_values = result.values["value"]
        valid_rsi = rsi_values[~np.isnan(rsi_values)]
        
        self.assertTrue((valid_rsi >= 0).all())
        self.assertTrue((valid_rsi <= 100).all())
    
    def test_bollinger_bands_calculation(self):
        """测试布林带计算"""
        result = self.indicators.calculate_bollinger_bands(self.market_data)
        
        self.assertIsInstance(result, IndicatorResult)
        self.assertEqual(result.name, "Bollinger Bands")
        self.assertIn("upper", result.values)
        self.assertIn("middle", result.values)
        self.assertIn("lower", result.values)
        self.assertIn("bandwidth", result.values)
        
        # 验证布林带关系
        upper = result.values["upper"]
        middle = result.values["middle"]
        lower = result.values["lower"]
        
        # 上轨应该大于中轨，中轨应该大于下轨
        valid_indices = ~(np.isnan(upper) | np.isnan(middle) | np.isnan(lower))
        self.assertTrue((upper[valid_indices] >= middle[valid_indices]).all())
        self.assertTrue((middle[valid_indices] >= lower[valid_indices]).all())
    
    def test_adx_calculation(self):
        """测试ADX计算"""
        result = self.indicators.calculate_adx(self.market_data)
        
        self.assertIsInstance(result, IndicatorResult)
        self.assertEqual(result.name, "ADX")
        self.assertIn("adx", result.values)
        self.assertIn("plus_di", result.values)
        self.assertIn("minus_di", result.values)
        
        # ADX值应该在0-100之间
        adx_values = result.values["adx"]
        valid_adx = adx_values[~np.isnan(adx_values)]
        
        if len(valid_adx) > 0:
            self.assertTrue((valid_adx >= 0).all())
            self.assertTrue((valid_adx <= 100).all())
    
    def test_stochastic_calculation(self):
        """测试随机指标计算"""
        result = self.indicators.calculate_stochastic(self.market_data)
        
        self.assertIsInstance(result, IndicatorResult)
        self.assertEqual(result.name, "Stochastic")
        self.assertIn("k", result.values)
        self.assertIn("d", result.values)
        
        # 随机指标值应该在0-100之间
        k_values = result.values["k"]
        d_values = result.values["d"]
        
        valid_k = k_values[~np.isnan(k_values)]
        valid_d = d_values[~np.isnan(d_values)]
        
        if len(valid_k) > 0:
            self.assertTrue((valid_k >= 0).all())
            self.assertTrue((valid_k <= 100).all())
        
        if len(valid_d) > 0:
            self.assertTrue((valid_d >= 0).all())
            self.assertTrue((valid_d <= 100).all())
    
    def test_atr_calculation(self):
        """测试ATR计算"""
        result = self.indicators.calculate_atr(self.market_data)
        
        self.assertIsInstance(result, IndicatorResult)
        self.assertEqual(result.name, "ATR")
        self.assertIn("value", result.values)
        
        # ATR值应该为正数
        atr_values = result.values["value"]
        valid_atr = atr_values[~np.isnan(atr_values)]
        
        if len(valid_atr) > 0:
            self.assertTrue((valid_atr >= 0).all())
    
    def test_cci_calculation(self):
        """测试CCI计算"""
        result = self.indicators.calculate_cci(self.market_data)
        
        self.assertIsInstance(result, IndicatorResult)
        self.assertEqual(result.name, "CCI")
        self.assertIn("value", result.values)
        
        # CCI值通常在-100到+100之间，但可以超出这个范围
        cci_values = result.values["value"]
        self.assertFalse(np.isnan(cci_values[-1]))  # 最后一个值不应该是NaN
    
    def test_indicator_summary(self):
        """测试指标汇总"""
        summary = self.indicators.get_indicator_summary(self.market_data)
        
        self.assertIn("symbol", summary)
        self.assertIn("timeframe", summary)
        self.assertIn("indicators", summary)
        
        indicators = summary["indicators"]
        self.assertIn("trend", indicators)
        self.assertIn("momentum", indicators)
        self.assertIn("volatility", indicators)
        
        # 检查趋势指标
        trend = indicators["trend"]
        self.assertIn("sma_20", trend)
        self.assertIn("ema_20", trend)
        self.assertIn("macd", trend)
        self.assertIn("trend_strength", trend)
        
        # 检查动量指标
        momentum = indicators["momentum"]
        self.assertIn("rsi", momentum)
        self.assertIn("momentum_signal", momentum)
        
        # 检查波动率指标
        volatility = indicators["volatility"]
        self.assertIn("atr", volatility)
        self.assertIn("volatility_level", volatility)
    
    def test_multiple_timeframes(self):
        """测试多周期计算"""
        # 创建多个周期的数据
        market_data_dict = {
            "H1": self.market_data,
            "H4": self.market_data  # 简化测试，使用相同数据
        }
        
        results = self.indicators.calculate_multiple_timeframes(
            market_data_dict, "sma", period=20
        )
        
        self.assertIn("H1", results)
        self.assertIn("H4", results)
        
        for timeframe, result in results.items():
            self.assertIsInstance(result, IndicatorResult)
            self.assertEqual(result.name, "SMA")
    
    def test_cache_functionality(self):
        """测试缓存功能"""
        # 启用缓存的指标计算器
        cached_indicators = TechnicalIndicators(use_cache=True, cache_ttl=300)
        
        # 第一次计算
        result1 = cached_indicators.calculate_sma(self.market_data, period=20)
        
        # 第二次计算（应该从缓存获取）
        result2 = cached_indicators.calculate_sma(self.market_data, period=20)
        
        # 结果应该相同
        self.assertEqual(result1.name, result2.name)
        self.assertEqual(result1.symbol, result2.symbol)
        
        # 检查缓存统计
        stats = cached_indicators.get_cache_stats()
        self.assertIn("size", stats)
        self.assertGreater(stats["size"], 0)


class TestCustomIndicators(unittest.TestCase):
    """测试自定义指标"""
    
    def setUp(self):
        self.custom_indicators = CustomIndicators()
        
        # 创建测试数据
        dates = pd.date_range(start='2023-01-01', periods=50, freq='H')
        np.random.seed(42)
        
        # 生成更复杂的价格数据
        base_price = 1.1000
        trend = np.linspace(0, 0.01, 50)  # 上升趋势
        noise = np.random.normal(0, 0.0005, 50)
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
    
    def test_market_structure_calculation(self):
        """测试市场结构计算"""
        result = self.custom_indicators.calculate_market_structure(self.market_data)
        
        self.assertIsInstance(result, IndicatorResult)
        self.assertEqual(result.name, "Market Structure")
        self.assertIn("swing_highs", result.values)
        self.assertIn("swing_lows", result.values)
        self.assertIn("structure", result.values)
        self.assertIn("trend", result.values)
        
        # 验证摆动点
        swing_highs = result.values["swing_highs"]
        swing_lows = result.values["swing_lows"]
        
        self.assertIsInstance(swing_highs, list)
        self.assertIsInstance(swing_lows, list)
        
        # 验证结构分析
        structure = result.values["structure"]
        trend = result.values["trend"]
        
        self.assertIsInstance(structure, str)
        self.assertIsInstance(trend, str)
    
    def test_volume_profile_calculation(self):
        """测试成交量分布计算"""
        result = self.custom_indicators.calculate_volume_profile(self.market_data, bins=10)
        
        self.assertIsInstance(result, IndicatorResult)
        self.assertEqual(result.name, "Volume Profile")
        self.assertIn("volume_profile", result.values)
        self.assertIn("price_bins", result.values)
        self.assertIn("poc_price", result.values)
        self.assertIn("value_area_high", result.values)
        self.assertIn("value_area_low", result.values)
        
        # 验证成交量分布
        volume_profile = result.values["volume_profile"]
        price_bins = result.values["price_bins"]
        
        self.assertEqual(len(volume_profile), 10)
        self.assertEqual(len(price_bins), 11)  # bins + 1
        
        # 验证POC价格在合理范围内
        poc_price = result.values["poc_price"]
        min_price = self.ohlcv_df['low'].min()
        max_price = self.ohlcv_df['high'].max()
        
        self.assertGreaterEqual(poc_price, min_price)
        self.assertLessEqual(poc_price, max_price)
    
    def test_order_flow_calculation(self):
        """测试订单流计算"""
        result = self.custom_indicators.calculate_order_flow(self.market_data)
        
        self.assertIsInstance(result, IndicatorResult)
        self.assertEqual(result.name, "Order Flow")
        self.assertIn("buying_pressure", result.values)
        self.assertIn("selling_pressure", result.values)
        self.assertIn("cumulative_delta", result.values)
        self.assertIn("flow_strength", result.values)
        
        # 验证订单流数据
        buying_pressure = result.values["buying_pressure"]
        selling_pressure = result.values["selling_pressure"]
        cumulative_delta = result.values["cumulative_delta"]
        flow_strength = result.values["flow_strength"]
        
        self.assertEqual(len(buying_pressure), len(self.ohlcv_df) - 1)  # 少一个，因为需要前一个收盘价
        self.assertEqual(len(selling_pressure), len(self.ohlcv_df) - 1)
        self.assertEqual(len(cumulative_delta), len(self.ohlcv_df) - 1)
        
        # 验证流强度在合理范围内
        self.assertGreaterEqual(flow_strength, -1)
        self.assertLessEqual(flow_strength, 1)


class TestMultiTimeframeAnalyzer(unittest.TestCase):
    """测试多周期分析器"""
    
    def setUp(self):
        self.indicators = TechnicalIndicators(use_cache=False)
        self.analyzer = MultiTimeframeAnalyzer(self.indicators)
        
        # 创建多个周期的测试数据
        self.market_data_dict = {}
        
        for timeframe in ["M15", "H1", "H4"]:
            dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
            np.random.seed(42)
            
            base_price = 1.1000
            prices = base_price + np.cumsum(np.random.normal(0, 0.0001, 100))
            
            ohlcv_data = []
            for i, price in enumerate(prices):
                high = price + abs(np.random.normal(0, 0.0001))
                low = price - abs(np.random.normal(0, 0.0001))
                open_price = prices[i-1] if i > 0 else price
                close_price = price
                volume = np.random.randint(1000, 10000)
                
                ohlcv_data.append({
                    'timestamp': dates[i],
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close_price,
                    'volume': volume
                })
            
            ohlcv_df = pd.DataFrame(ohlcv_data)
            self.market_data_dict[timeframe] = MarketData(
                symbol="EURUSD",
                timeframe=timeframe,
                timestamp=datetime.now(),
                ohlcv=ohlcv_df
            )
    
    def test_trend_confluence_analysis(self):
        """测试趋势一致性分析"""
        result = self.analyzer.analyze_trend_confluence(self.market_data_dict)
        
        self.assertIn("timeframe_analysis", result)
        self.assertIn("weighted_trend_score", result)
        self.assertIn("overall_trend", result)
        self.assertIn("confluence_strength", result)
        
        # 验证每个周期的分析
        timeframe_analysis = result["timeframe_analysis"]
        for timeframe in ["M15", "H1", "H4"]:
            self.assertIn(timeframe, timeframe_analysis)
            
            tf_data = timeframe_analysis[timeframe]
            if "score" in tf_data:  # 如果没有错误
                self.assertIn("score", tf_data)
                self.assertIn("strength", tf_data)
                self.assertIn("signals", tf_data)
                self.assertIn("weight", tf_data)
        
        # 验证加权趋势评分
        weighted_score = result["weighted_trend_score"]
        self.assertIsInstance(weighted_score, (int, float))
        
        # 验证整体趋势
        overall_trend = result["overall_trend"]
        self.assertIn(overall_trend, ["strong_bullish", "bullish", "neutral", "bearish", "strong_bearish"])
        
        # 验证一致性强度
        confluence_strength = result["confluence_strength"]
        self.assertGreaterEqual(confluence_strength, 0)
        self.assertLessEqual(confluence_strength, 1)
    
    def test_support_resistance_analysis(self):
        """测试支撑阻力分析"""
        result = self.analyzer.analyze_support_resistance(self.market_data_dict)
        
        self.assertIn("timeframe_sr", result)
        self.assertIn("consolidated_support", result)
        self.assertIn("consolidated_resistance", result)
        self.assertIn("key_levels", result)
        
        # 验证每个周期的支撑阻力
        timeframe_sr = result["timeframe_sr"]
        for timeframe in ["M15", "H1", "H4"]:
            self.assertIn(timeframe, timeframe_sr)
            
            tf_data = timeframe_sr[timeframe]
            if "support_levels" in tf_data:  # 如果没有错误
                self.assertIn("support_levels", tf_data)
                self.assertIn("resistance_levels", tf_data)
                self.assertIn("current_price", tf_data)
        
        # 验证合并后的支撑阻力
        consolidated_support = result["consolidated_support"]
        consolidated_resistance = result["consolidated_resistance"]
        key_levels = result["key_levels"]
        
        self.assertIsInstance(consolidated_support, list)
        self.assertIsInstance(consolidated_resistance, list)
        self.assertIsInstance(key_levels, list)


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)