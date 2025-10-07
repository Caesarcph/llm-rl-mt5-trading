"""
市场分析Agent测试用例
测试MarketAnalystAgent的各项功能
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.agents.market_analyst import (
    MarketAnalystAgent, TrendType, MarketRegime, VolatilityLevel,
    TrendAnalysis, VolatilityAnalysis, CorrelationMatrix, MarketAnalysis,
    MarketRegimeDetector, CorrelationAnalyzer
)
from src.core.models import MarketData, MarketState
from src.strategies.indicators import IndicatorResult


class TestMarketAnalystAgent(unittest.TestCase):
    """市场分析Agent测试类"""
    
    def setUp(self):
        """测试初始化"""
        self.agent = MarketAnalystAgent()
        self.sample_market_data = self._create_sample_market_data()
        self.sample_indicators = self._create_sample_indicators()
    
    def _create_sample_market_data(self) -> MarketData:
        """创建示例市场数据"""
        # 创建100个交易日的OHLCV数据
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        # 生成模拟价格数据（上升趋势）
        base_price = 1.1000
        price_trend = np.linspace(0, 0.0200, 100)  # 2%的上升趋势
        noise = np.random.normal(0, 0.0020, 100)   # 0.2%的随机噪声
        
        close_prices = base_price + price_trend + noise
        
        # 生成OHLC数据
        ohlcv_data = []
        for i, close in enumerate(close_prices):
            high = close + abs(np.random.normal(0, 0.0010))
            low = close - abs(np.random.normal(0, 0.0010))
            open_price = close + np.random.normal(0, 0.0005)
            volume = np.random.randint(1000, 10000)
            
            ohlcv_data.append({
                'time': dates[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        ohlcv_df = pd.DataFrame(ohlcv_data)
        ohlcv_df.set_index('time', inplace=True)
        
        return MarketData(
            symbol="EURUSD",
            timeframe="D1",
            timestamp=datetime.now(),
            ohlcv=ohlcv_df,
            indicators={},
            spread=0.00015,
            liquidity=1.0
        )
    
    def _create_sample_indicators(self) -> dict:
        """创建示例技术指标"""
        indicators = {}
        
        # RSI指标
        rsi_values = np.random.uniform(30, 70, 100)
        indicators['rsi'] = IndicatorResult(
            name="RSI",
            symbol="EURUSD",
            timeframe="D1",
            timestamp=datetime.now(),
            values={"value": rsi_values}
        )
        
        # MACD指标
        macd_values = np.random.uniform(-0.001, 0.001, 100)
        signal_values = np.random.uniform(-0.001, 0.001, 100)
        indicators['macd'] = IndicatorResult(
            name="MACD",
            symbol="EURUSD",
            timeframe="D1",
            timestamp=datetime.now(),
            values={
                "macd": macd_values,
                "signal": signal_values,
                "histogram": macd_values - signal_values
            }
        )
        
        # ADX指标
        adx_values = np.random.uniform(15, 35, 100)
        indicators['adx'] = IndicatorResult(
            name="ADX",
            symbol="EURUSD",
            timeframe="D1",
            timestamp=datetime.now(),
            values={
                "adx": adx_values,
                "plus_di": np.random.uniform(10, 30, 100),
                "minus_di": np.random.uniform(10, 30, 100)
            }
        )
        
        # 布林带指标
        close_prices = self.sample_market_data.ohlcv['close'].values
        middle = np.mean(close_prices[-20:])
        std = np.std(close_prices[-20:])
        indicators['bollinger_bands'] = IndicatorResult(
            name="Bollinger Bands",
            symbol="EURUSD",
            timeframe="D1",
            timestamp=datetime.now(),
            values={
                "upper": np.full(100, middle + 2*std),
                "middle": np.full(100, middle),
                "lower": np.full(100, middle - 2*std)
            }
        )
        
        # ATR指标
        atr_values = np.random.uniform(0.0010, 0.0030, 100)
        indicators['atr'] = IndicatorResult(
            name="ATR",
            symbol="EURUSD",
            timeframe="D1",
            timestamp=datetime.now(),
            values={"value": atr_values}
        )
        
        return indicators
    
    def test_agent_initialization(self):
        """测试Agent初始化"""
        agent = MarketAnalystAgent()
        
        self.assertIsNotNone(agent.technical_indicators)
        self.assertIsNotNone(agent.regime_detector)
        self.assertIsNotNone(agent.correlation_analyzer)
        self.assertEqual(len(agent.correlation_history), 0)
        self.assertEqual(len(agent.analysis_history), 0)
    
    def test_agent_initialization_with_config(self):
        """测试带配置的Agent初始化"""
        config = {
            'use_cache': False,
            'cache_ttl': 600,
            'regime_lookback': 30,
            'correlation_lookback': 50
        }
        
        agent = MarketAnalystAgent(config)
        self.assertEqual(agent.config['cache_ttl'], 600)
        self.assertEqual(agent.config['regime_lookback'], 30)
    
    @patch('src.agents.market_analyst.MarketAnalystAgent._calculate_key_indicators')
    def test_analyze_market_state(self, mock_indicators):
        """测试市场状态分析"""
        mock_indicators.return_value = self.sample_indicators
        
        market_state = self.agent.analyze_market_state("EURUSD", self.sample_market_data)
        
        self.assertIsInstance(market_state, MarketState)
        self.assertIn(market_state.trend, ['uptrend', 'downtrend', 'sideways'])
        self.assertIn(market_state.regime, ['trending', 'ranging', 'breakout', 'volatile'])
        self.assertIsInstance(market_state.volatility, float)
        self.assertIn('support', market_state.support_resistance)
        self.assertIn('resistance', market_state.support_resistance)
    
    @patch('src.agents.market_analyst.MarketAnalystAgent._calculate_key_indicators')
    def test_generate_comprehensive_analysis(self, mock_indicators):
        """测试综合市场分析"""
        mock_indicators.return_value = self.sample_indicators
        
        analysis = self.agent.generate_comprehensive_analysis("EURUSD", self.sample_market_data)
        
        self.assertIsInstance(analysis, MarketAnalysis)
        self.assertEqual(analysis.symbol, "EURUSD")
        self.assertIsInstance(analysis.trend_analysis, TrendAnalysis)
        self.assertIsInstance(analysis.volatility_analysis, VolatilityAnalysis)
        self.assertIsInstance(analysis.market_regime, MarketRegime)
        self.assertIsInstance(analysis.market_score, float)
        self.assertTrue(0 <= analysis.market_score <= 100)
        
        # 检查历史记录是否保存
        self.assertIn("EURUSD", self.agent.analysis_history)
        self.assertEqual(len(self.agent.analysis_history["EURUSD"]), 1)
    
    def test_analyze_multi_symbol_correlations(self):
        """测试多品种相关性分析"""
        # 创建多个品种的市场数据
        market_data_dict = {
            "EURUSD": self.sample_market_data,
            "GBPUSD": self._create_sample_market_data_gbp(),
            "USDJPY": self._create_sample_market_data_jpy()
        }
        
        correlation_matrix = self.agent.analyze_multi_symbol_correlations(market_data_dict)
        
        self.assertIsInstance(correlation_matrix, CorrelationMatrix)
        self.assertEqual(len(correlation_matrix.symbols), 3)
        self.assertIn("EURUSD", correlation_matrix.symbols)
        self.assertIn("GBPUSD", correlation_matrix.symbols)
        self.assertIn("USDJPY", correlation_matrix.symbols)
        
        # 检查相关性矩阵的对角线应该为1
        for symbol in correlation_matrix.symbols:
            self.assertAlmostEqual(
                correlation_matrix.get_correlation(symbol, symbol), 1.0, places=2
            )
        
        # 检查历史记录是否保存
        self.assertEqual(len(self.agent.correlation_history), 1)
    
    def _create_sample_market_data_gbp(self) -> MarketData:
        """创建GBP示例数据"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        base_price = 1.2500
        price_trend = np.linspace(0, 0.0150, 100)
        noise = np.random.normal(0, 0.0025, 100)
        close_prices = base_price + price_trend + noise
        
        ohlcv_data = []
        for i, close in enumerate(close_prices):
            high = close + abs(np.random.normal(0, 0.0012))
            low = close - abs(np.random.normal(0, 0.0012))
            open_price = close + np.random.normal(0, 0.0006)
            volume = np.random.randint(800, 8000)
            
            ohlcv_data.append({
                'time': dates[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        ohlcv_df = pd.DataFrame(ohlcv_data)
        ohlcv_df.set_index('time', inplace=True)
        
        return MarketData(
            symbol="GBPUSD",
            timeframe="D1",
            timestamp=datetime.now(),
            ohlcv=ohlcv_df
        )
    
    def _create_sample_market_data_jpy(self) -> MarketData:
        """创建JPY示例数据"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        base_price = 150.00
        price_trend = np.linspace(0, -2.00, 100)  # 下降趋势
        noise = np.random.normal(0, 0.50, 100)
        close_prices = base_price + price_trend + noise
        
        ohlcv_data = []
        for i, close in enumerate(close_prices):
            high = close + abs(np.random.normal(0, 0.30))
            low = close - abs(np.random.normal(0, 0.30))
            open_price = close + np.random.normal(0, 0.15)
            volume = np.random.randint(1200, 12000)
            
            ohlcv_data.append({
                'time': dates[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        ohlcv_df = pd.DataFrame(ohlcv_data)
        ohlcv_df.set_index('time', inplace=True)
        
        return MarketData(
            symbol="USDJPY",
            timeframe="D1",
            timestamp=datetime.now(),
            ohlcv=ohlcv_df
        )
    
    def test_detect_correlation_regime_changes(self):
        """测试相关性状态变化检测"""
        # 先添加一些相关性历史数据
        market_data_dict = {
            "EURUSD": self.sample_market_data,
            "GBPUSD": self._create_sample_market_data_gbp()
        }
        
        # 添加两个时间点的相关性数据
        self.agent.analyze_multi_symbol_correlations(market_data_dict)
        self.agent.analyze_multi_symbol_correlations(market_data_dict)
        
        regime_changes = self.agent.detect_correlation_regime_changes()
        
        self.assertIn('status', regime_changes)
        self.assertIn(regime_changes['status'], ['analyzed', 'insufficient_data'])
        
        if regime_changes['status'] == 'analyzed':
            self.assertIn('average_change', regime_changes)
            self.assertIn('max_change', regime_changes)
            self.assertIn('regime_shift_detected', regime_changes)
    
    def test_trend_analysis(self):
        """测试趋势分析"""
        trend_analysis = self.agent._analyze_trend(self.sample_market_data, self.sample_indicators)
        
        self.assertIsInstance(trend_analysis, TrendAnalysis)
        self.assertIsInstance(trend_analysis.trend_type, TrendType)
        self.assertTrue(0 <= trend_analysis.strength <= 1)
        self.assertTrue(0 <= trend_analysis.confidence <= 1)
        self.assertIsInstance(trend_analysis.duration_bars, int)
        self.assertIsInstance(trend_analysis.slope, float)
        self.assertTrue(0 <= trend_analysis.r_squared <= 1)
    
    def test_volatility_analysis(self):
        """测试波动率分析"""
        volatility_analysis = self.agent._analyze_volatility(self.sample_market_data, self.sample_indicators)
        
        self.assertIsInstance(volatility_analysis, VolatilityAnalysis)
        self.assertIsInstance(volatility_analysis.current_volatility, float)
        self.assertIsInstance(volatility_analysis.volatility_level, VolatilityLevel)
        self.assertTrue(0 <= volatility_analysis.volatility_percentile <= 1)
        self.assertIsInstance(volatility_analysis.atr_ratio, float)
    
    def test_support_resistance_levels(self):
        """测试支撑阻力位分析"""
        sr_levels = self.agent._find_support_resistance_levels(self.sample_market_data)
        
        self.assertIsInstance(sr_levels, dict)
        self.assertIn('support', sr_levels)
        self.assertIn('resistance', sr_levels)
        self.assertIsInstance(sr_levels['support'], list)
        self.assertIsInstance(sr_levels['resistance'], list)
        
        # 检查支撑位和阻力位的合理性（由于随机数据，不强制要求严格的大小关系）
        if sr_levels['support'] and sr_levels['resistance']:
            # 只检查数据类型和数量合理性
            self.assertTrue(all(isinstance(level, float) for level in sr_levels['support']))
            self.assertTrue(all(isinstance(level, float) for level in sr_levels['resistance']))
            self.assertLessEqual(len(sr_levels['support']), 5)
            self.assertLessEqual(len(sr_levels['resistance']), 5)
    
    def test_technical_signals_generation(self):
        """测试技术信号生成"""
        signals = self.agent._generate_technical_signals(self.sample_indicators)
        
        self.assertIsInstance(signals, dict)
        
        # 检查信号值范围
        for signal_name, signal_value in signals.items():
            self.assertIsInstance(signal_value, float)
            if 'signal' in signal_name:
                self.assertTrue(-1 <= signal_value <= 1)
    
    def test_market_score_calculation(self):
        """测试市场评分计算"""
        trend_analysis = TrendAnalysis(
            trend_type=TrendType.UPTREND,
            strength=0.8,
            confidence=0.7,
            duration_bars=50,
            slope=0.001,
            r_squared=0.6,
            support_resistance={}
        )
        
        volatility_analysis = VolatilityAnalysis(
            current_volatility=0.02,
            volatility_level=VolatilityLevel.MEDIUM,
            volatility_percentile=0.5,
            atr_ratio=0.015
        )
        
        technical_signals = {
            'rsi_signal': 0.5,
            'macd_signal': 1.0,
            'trend_strength': 0.8
        }
        
        market_score = self.agent._calculate_market_score(
            trend_analysis, volatility_analysis, technical_signals
        )
        
        self.assertIsInstance(market_score, float)
        self.assertTrue(0 <= market_score <= 100)
    
    def test_get_analysis_summary(self):
        """测试分析汇总获取"""
        # 先生成一个分析
        with patch('src.agents.market_analyst.MarketAnalystAgent._calculate_key_indicators') as mock_indicators:
            mock_indicators.return_value = self.sample_indicators
            self.agent.generate_comprehensive_analysis("EURUSD", self.sample_market_data)
        
        summary = self.agent.get_analysis_summary("EURUSD")
        
        self.assertIsNotNone(summary)
        self.assertEqual(summary['symbol'], "EURUSD")
        self.assertIn('trend', summary)
        self.assertIn('trend_strength', summary)
        self.assertIn('market_regime', summary)
        self.assertIn('volatility_level', summary)
        self.assertIn('market_score', summary)
        self.assertIn('recommendation', summary)
        self.assertIn('support_levels', summary)
        self.assertIn('resistance_levels', summary)
    
    def test_get_analysis_summary_no_data(self):
        """测试无数据时的分析汇总"""
        summary = self.agent.get_analysis_summary("NONEXISTENT")
        self.assertIsNone(summary)
    
    def test_clear_cache(self):
        """测试缓存清理"""
        # 先生成一些数据
        with patch('src.agents.market_analyst.MarketAnalystAgent._calculate_key_indicators') as mock_indicators:
            mock_indicators.return_value = self.sample_indicators
            self.agent.generate_comprehensive_analysis("EURUSD", self.sample_market_data)
        
        market_data_dict = {"EURUSD": self.sample_market_data}
        self.agent.analyze_multi_symbol_correlations(market_data_dict)
        
        # 确认有数据
        self.assertTrue(len(self.agent.analysis_history) > 0)
        self.assertTrue(len(self.agent.correlation_history) > 0)
        
        # 清理缓存
        self.agent.clear_cache()
        
        # 确认数据被清理
        self.assertEqual(len(self.agent.analysis_history), 0)
        self.assertEqual(len(self.agent.correlation_history), 0)
    
    def test_error_handling_invalid_data(self):
        """测试无效数据的错误处理"""
        # 创建包含无效数据的市场数据（但不是完全空的）
        invalid_ohlcv = pd.DataFrame({
            'open': [1.1000],
            'high': [1.1000],
            'low': [1.1000], 
            'close': [1.1000],
            'volume': [0]
        })
        
        invalid_market_data = MarketData(
            symbol="TEST",
            timeframe="D1",
            timestamp=datetime.now(),
            ohlcv=invalid_ohlcv,
            indicators={}
        )
        
        # 应该返回默认的市场状态而不是抛出异常
        market_state = self.agent.analyze_market_state("TEST", invalid_market_data)
        
        self.assertIsInstance(market_state, MarketState)
        self.assertEqual(market_state.trend, "sideways")
        self.assertEqual(market_state.regime, "ranging")


class TestMarketRegimeDetector(unittest.TestCase):
    """市场状态检测器测试类"""
    
    def setUp(self):
        """测试初始化"""
        self.detector = MarketRegimeDetector(lookback_period=30)
        self.sample_market_data = self._create_sample_market_data()
    
    def _create_sample_market_data(self) -> MarketData:
        """创建示例市场数据"""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        base_price = 1.1000
        close_prices = base_price + np.random.normal(0, 0.001, 50)
        
        ohlcv_data = []
        for i, close in enumerate(close_prices):
            ohlcv_data.append({
                'time': dates[i],
                'open': close + np.random.normal(0, 0.0005),
                'high': close + abs(np.random.normal(0, 0.001)),
                'low': close - abs(np.random.normal(0, 0.001)),
                'close': close,
                'volume': np.random.randint(1000, 10000)
            })
        
        ohlcv_df = pd.DataFrame(ohlcv_data)
        ohlcv_df.set_index('time', inplace=True)
        
        return MarketData(
            symbol="EURUSD",
            timeframe="D1",
            timestamp=datetime.now(),
            ohlcv=ohlcv_df
        )
    
    def test_detect_regime(self):
        """测试市场状态检测"""
        indicators = {
            'adx': IndicatorResult(
                name="ADX",
                symbol="EURUSD",
                timeframe="D1",
                timestamp=datetime.now(),
                values={"adx": np.full(50, 30)}  # 强趋势
            )
        }
        
        regime = self.detector.detect_regime(self.sample_market_data, indicators)
        
        self.assertIsInstance(regime, MarketRegime)
        self.assertIn(regime, [MarketRegime.TRENDING, MarketRegime.RANGING, 
                              MarketRegime.BREAKOUT, MarketRegime.VOLATILE])
    
    def test_detect_regime_error_handling(self):
        """测试错误处理"""
        # 传入空的指标字典
        regime = self.detector.detect_regime(self.sample_market_data, {})
        
        # 应该返回有效的市场状态而不是抛出异常
        self.assertIsInstance(regime, MarketRegime)
        self.assertIn(regime, [MarketRegime.TRENDING, MarketRegime.RANGING, 
                              MarketRegime.BREAKOUT, MarketRegime.VOLATILE])


class TestCorrelationAnalyzer(unittest.TestCase):
    """相关性分析器测试类"""
    
    def setUp(self):
        """测试初始化"""
        self.analyzer = CorrelationAnalyzer(lookback_period=50)
    
    def test_analyze_correlations(self):
        """测试相关性分析"""
        # 创建相关的市场数据
        market_data_dict = {
            "EURUSD": self._create_correlated_data(base=1.1000, correlation=0.8),
            "GBPUSD": self._create_correlated_data(base=1.2500, correlation=0.8),
            "USDJPY": self._create_correlated_data(base=150.00, correlation=-0.6)
        }
        
        correlation_matrix = self.analyzer.analyze_correlations(market_data_dict)
        
        self.assertIsInstance(correlation_matrix, CorrelationMatrix)
        self.assertEqual(len(correlation_matrix.symbols), 3)
        
        # 检查对角线元素为1
        for symbol in correlation_matrix.symbols:
            self.assertAlmostEqual(
                correlation_matrix.get_correlation(symbol, symbol), 1.0, places=2
            )
    
    def _create_correlated_data(self, base: float, correlation: float) -> MarketData:
        """创建相关的市场数据"""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        
        # 生成基础随机序列
        base_returns = np.random.normal(0, 0.01, 50)
        
        # 生成相关的收益率序列
        correlated_returns = correlation * base_returns + \
                           np.sqrt(1 - correlation**2) * np.random.normal(0, 0.01, 50)
        
        # 转换为价格序列
        prices = [base]
        for ret in correlated_returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        ohlcv_data = []
        for i, close in enumerate(prices):
            ohlcv_data.append({
                'time': dates[i],
                'open': close + np.random.normal(0, 0.0005),
                'high': close + abs(np.random.normal(0, 0.001)),
                'low': close - abs(np.random.normal(0, 0.001)),
                'close': close,
                'volume': np.random.randint(1000, 10000)
            })
        
        ohlcv_df = pd.DataFrame(ohlcv_data)
        ohlcv_df.set_index('time', inplace=True)
        
        return MarketData(
            symbol="TEST",
            timeframe="D1",
            timestamp=datetime.now(),
            ohlcv=ohlcv_df
        )
    
    def test_detect_regime_changes(self):
        """测试相关性状态变化检测"""
        # 创建两个相关性矩阵
        symbols = ["EURUSD", "GBPUSD"]
        
        # 第一个矩阵
        matrix1 = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], 
                              index=symbols, columns=symbols)
        corr1 = CorrelationMatrix(
            symbols=symbols,
            correlation_matrix=matrix1,
            timestamp=datetime.now() - timedelta(hours=1),
            lookback_period=50
        )
        
        # 第二个矩阵（相关性发生显著变化）
        matrix2 = pd.DataFrame([[1.0, 0.9], [0.9, 1.0]], 
                              index=symbols, columns=symbols)
        corr2 = CorrelationMatrix(
            symbols=symbols,
            correlation_matrix=matrix2,
            timestamp=datetime.now(),
            lookback_period=50
        )
        
        correlation_history = [corr1, corr2]
        
        changes = self.analyzer.detect_regime_changes(correlation_history)
        
        self.assertEqual(changes['status'], 'analyzed')
        self.assertIn('average_change', changes)
        self.assertIn('max_change', changes)
        self.assertIn('regime_shift_detected', changes)
    
    def test_detect_regime_changes_insufficient_data(self):
        """测试数据不足时的状态变化检测"""
        changes = self.analyzer.detect_regime_changes([])
        self.assertEqual(changes['status'], 'insufficient_data')


if __name__ == '__main__':
    unittest.main()