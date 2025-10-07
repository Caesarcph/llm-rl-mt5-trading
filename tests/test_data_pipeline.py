#!/usr/bin/env python3
"""
数据管道单元测试
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime, timedelta
import asyncio

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data.pipeline import DataPipeline, DataPipelineError
from src.data.mt5_connection import ConnectionConfig
from src.data.models import TimeFrame, MarketData, Tick


class TestDataPipeline(unittest.TestCase):
    """数据管道测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.connection_config = ConnectionConfig(
            login=12345,
            password="test_password",
            server="test_server"
        )
        
        # 创建模拟的OHLCV数据
        self.mock_ohlcv = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=100, freq='1h'),
            'open': [1.1000 + i * 0.0001 for i in range(100)],
            'high': [1.1005 + i * 0.0001 for i in range(100)],
            'low': [1.0995 + i * 0.0001 for i in range(100)],
            'close': [1.1002 + i * 0.0001 for i in range(100)],
            'tick_volume': [1000 + i * 10 for i in range(100)],
            'real_volume': [1000 + i * 10 for i in range(100)]
        })
        
        # 创建模拟的MT5 rates数据
        self.mock_rates = []
        for i in range(100):
            self.mock_rates.append({
                'time': int((datetime(2024, 1, 1) + timedelta(hours=i)).timestamp()),
                'open': 1.1000 + i * 0.0001,
                'high': 1.1005 + i * 0.0001,
                'low': 1.0995 + i * 0.0001,
                'close': 1.1002 + i * 0.0001,
                'tick_volume': 1000 + i * 10,
                'real_volume': 1000 + i * 10
            })
    
    @patch('src.data.pipeline.mt5')
    @patch('src.data.mt5_connection.mt5')
    def test_pipeline_initialization(self, mock_mt5_conn, mock_mt5_pipeline):
        """测试数据管道初始化"""
        pipeline = DataPipeline(self.connection_config)
        
        self.assertIsNotNone(pipeline.mt5_connection)
        self.assertIsNotNone(pipeline.data_validator)
        self.assertIsNotNone(pipeline.cache)
        self.assertIsNotNone(pipeline.executor)
    
    @patch('src.data.pipeline.mt5')
    @patch('src.data.mt5_connection.mt5')
    def test_get_realtime_data_success(self, mock_mt5_conn, mock_mt5_pipeline):
        """测试成功获取实时数据"""
        # 模拟MT5连接成功
        mock_mt5_conn.initialize.return_value = True
        mock_mt5_conn.account_info.return_value = Mock()
        mock_mt5_conn.terminal_info.return_value = Mock()
        
        # 模拟获取数据成功
        mock_mt5_pipeline.copy_rates_from_pos.return_value = self.mock_rates
        mock_mt5_pipeline.symbol_info_tick.return_value = Mock(ask=1.1005, bid=1.1000)
        mock_mt5_pipeline.TIMEFRAME_H1 = 16385
        
        pipeline = DataPipeline(self.connection_config)
        
        # 获取实时数据
        data = pipeline.get_realtime_data("EURUSD", TimeFrame.H1, count=100)
        
        self.assertIsNotNone(data)
        self.assertIsInstance(data, MarketData)
        self.assertEqual(data.symbol, "EURUSD")
        self.assertEqual(data.timeframe, TimeFrame.H1)
        self.assertFalse(data.ohlcv.empty)
        self.assertAlmostEqual(data.spread, 0.0005, places=4)  # ask - bid
    
    @patch('src.data.pipeline.mt5')
    @patch('src.data.mt5_connection.mt5')
    def test_get_historical_data_success(self, mock_mt5_conn, mock_mt5_pipeline):
        """测试成功获取历史数据"""
        # 模拟MT5连接成功
        mock_mt5_conn.initialize.return_value = True
        mock_mt5_conn.account_info.return_value = Mock()
        mock_mt5_conn.terminal_info.return_value = Mock()
        
        # 模拟获取数据成功
        mock_mt5_pipeline.copy_rates_range.return_value = self.mock_rates
        mock_mt5_pipeline.TIMEFRAME_H1 = 16385
        
        pipeline = DataPipeline(self.connection_config)
        
        # 获取历史数据
        start_time = datetime(2024, 1, 1)
        end_time = datetime(2024, 1, 5)
        data = pipeline.get_historical_data("EURUSD", TimeFrame.H1, start_time, end_time)
        
        self.assertIsNotNone(data)
        self.assertIsInstance(data, MarketData)
        self.assertEqual(data.symbol, "EURUSD")
        self.assertEqual(data.timeframe, TimeFrame.H1)
        self.assertFalse(data.ohlcv.empty)
    
    @patch('src.data.pipeline.mt5')
    @patch('src.data.mt5_connection.mt5')
    def test_get_tick_data_success(self, mock_mt5_conn, mock_mt5_pipeline):
        """测试成功获取Tick数据"""
        # 模拟MT5连接成功
        mock_mt5_conn.initialize.return_value = True
        mock_mt5_conn.account_info.return_value = Mock()
        mock_mt5_conn.terminal_info.return_value = Mock()
        
        # 模拟Tick数据
        mock_ticks = []
        for i in range(10):
            mock_ticks.append({
                'time': int((datetime.now() - timedelta(seconds=i)).timestamp()),
                'bid': 1.1000 + i * 0.00001,
                'ask': 1.1005 + i * 0.00001,
                'last': 1.1002 + i * 0.00001,
                'volume_real': 100,
                'flags': 0
            })
        
        mock_mt5_pipeline.copy_ticks_from.return_value = mock_ticks
        mock_mt5_pipeline.COPY_TICKS_ALL = 3
        
        pipeline = DataPipeline(self.connection_config)
        
        # 获取Tick数据
        ticks = pipeline.get_tick_data("EURUSD", count=10)
        
        self.assertIsNotNone(ticks)
        self.assertIsInstance(ticks, list)
        self.assertEqual(len(ticks), 10)
        
        for tick in ticks:
            self.assertIsInstance(tick, Tick)
            self.assertEqual(tick.symbol, "EURUSD")
            self.assertGreater(tick.ask, tick.bid)
    
    @patch('src.data.pipeline.mt5')
    @patch('src.data.mt5_connection.mt5')
    def test_data_caching(self, mock_mt5_conn, mock_mt5_pipeline):
        """测试数据缓存功能"""
        # 模拟MT5连接成功
        mock_mt5_conn.initialize.return_value = True
        mock_mt5_conn.account_info.return_value = Mock()
        mock_mt5_conn.terminal_info.return_value = Mock()
        
        # 模拟获取数据成功
        mock_mt5_pipeline.copy_rates_from_pos.return_value = self.mock_rates
        mock_mt5_pipeline.TIMEFRAME_H1 = 16385
        
        pipeline = DataPipeline(self.connection_config)
        
        # 第一次获取数据
        data1 = pipeline.get_realtime_data("EURUSD", TimeFrame.H1, count=100)
        self.assertIsNotNone(data1)
        
        # 第二次获取相同数据，应该从缓存获取
        data2 = pipeline.get_realtime_data("EURUSD", TimeFrame.H1, count=100)
        self.assertIsNotNone(data2)
        
        # 验证缓存统计
        stats = pipeline.stats
        self.assertGreater(stats['cache_hits'], 0)
    
    @patch('src.data.pipeline.mt5')
    @patch('src.data.mt5_connection.mt5')
    def test_connection_failure(self, mock_mt5_conn, mock_mt5_pipeline):
        """测试连接失败处理"""
        # 模拟MT5连接失败
        mock_mt5_conn.initialize.return_value = False
        
        pipeline = DataPipeline(self.connection_config)
        
        # 尝试获取数据
        data = pipeline.get_realtime_data("EURUSD", TimeFrame.H1, count=100)
        
        self.assertIsNone(data)
        
        # 验证统计信息
        stats = pipeline.stats
        self.assertGreater(stats['requests_failed'], 0)
    
    @patch('src.data.pipeline.mt5')
    @patch('src.data.mt5_connection.mt5')
    def test_no_data_returned(self, mock_mt5_conn, mock_mt5_pipeline):
        """测试MT5返回空数据的处理"""
        # 模拟MT5连接成功
        mock_mt5_conn.initialize.return_value = True
        mock_mt5_conn.account_info.return_value = Mock()
        mock_mt5_conn.terminal_info.return_value = Mock()
        
        # 模拟返回空数据
        mock_mt5_pipeline.copy_rates_from_pos.return_value = None
        mock_mt5_pipeline.TIMEFRAME_H1 = 16385
        
        pipeline = DataPipeline(self.connection_config)
        
        # 尝试获取数据
        data = pipeline.get_realtime_data("EURUSD", TimeFrame.H1, count=100)
        
        self.assertIsNone(data)
    
    @patch('src.data.pipeline.mt5')
    @patch('src.data.mt5_connection.mt5')
    def test_indicator_calculation(self, mock_mt5_conn, mock_mt5_pipeline):
        """测试技术指标计算"""
        # 模拟MT5连接成功
        mock_mt5_conn.initialize.return_value = True
        mock_mt5_conn.account_info.return_value = Mock()
        mock_mt5_conn.terminal_info.return_value = Mock()
        
        # 模拟获取数据成功
        mock_mt5_pipeline.copy_rates_from_pos.return_value = self.mock_rates
        mock_mt5_pipeline.TIMEFRAME_H1 = 16385
        
        pipeline = DataPipeline(self.connection_config)
        
        # 获取包含指标的数据
        data = pipeline.get_realtime_data("EURUSD", TimeFrame.H1, count=100, include_indicators=True)
        
        self.assertIsNotNone(data)
        self.assertIsInstance(data.indicators, dict)
        
        # 验证常见指标存在
        expected_indicators = ['sma_20', 'ema_12', 'ema_26', 'rsi', 'bb_upper', 'bb_lower']
        for indicator in expected_indicators:
            if indicator in data.indicators:
                self.assertIsInstance(data.indicators[indicator], (int, float))
    
    @patch('src.data.pipeline.mt5')
    @patch('src.data.mt5_connection.mt5')
    def test_unsupported_timeframe(self, mock_mt5_conn, mock_mt5_pipeline):
        """测试不支持的时间周期"""
        # 模拟MT5连接成功
        mock_mt5_conn.initialize.return_value = True
        mock_mt5_conn.account_info.return_value = Mock()
        mock_mt5_conn.terminal_info.return_value = Mock()
        
        pipeline = DataPipeline(self.connection_config)
        
        # 清空时间周期映射来模拟不支持的周期
        pipeline.timeframe_map = {}
        
        # 尝试获取数据
        data = pipeline.get_realtime_data("EURUSD", TimeFrame.H1, count=100)
        
        self.assertIsNone(data)
    
    @patch('src.data.pipeline.mt5')
    @patch('src.data.mt5_connection.mt5')
    def test_async_multiple_symbols(self, mock_mt5_conn, mock_mt5_pipeline):
        """测试异步获取多个品种数据"""
        # 模拟MT5连接成功
        mock_mt5_conn.initialize.return_value = True
        mock_mt5_conn.account_info.return_value = Mock()
        mock_mt5_conn.terminal_info.return_value = Mock()
        
        # 模拟获取数据成功
        mock_mt5_pipeline.copy_rates_from_pos.return_value = self.mock_rates
        mock_mt5_pipeline.TIMEFRAME_H1 = 16385
        
        pipeline = DataPipeline(self.connection_config)
        
        # 异步获取多个品种数据
        symbols = ["EURUSD", "GBPUSD", "USDJPY"]
        
        async def test_async():
            results = await pipeline.get_multiple_symbols_data(symbols, TimeFrame.H1, count=100)
            return results
        
        # 运行异步测试
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(test_async())
            
            self.assertEqual(len(results), 3)
            for symbol in symbols:
                self.assertIn(symbol, results)
                self.assertIsNotNone(results[symbol])
        finally:
            loop.close()
    
    @patch('src.data.pipeline.mt5')
    @patch('src.data.mt5_connection.mt5')
    def test_data_stats(self, mock_mt5_conn, mock_mt5_pipeline):
        """测试数据统计功能"""
        # 模拟MT5连接成功
        mock_mt5_conn.initialize.return_value = True
        mock_mt5_conn.account_info.return_value = Mock()
        mock_mt5_conn.terminal_info.return_value = Mock()
        
        # 模拟获取数据成功
        mock_mt5_pipeline.copy_rates_from_pos.return_value = self.mock_rates
        mock_mt5_pipeline.TIMEFRAME_H1 = 16385
        
        pipeline = DataPipeline(self.connection_config)
        
        # 获取数据统计
        stats = pipeline.get_data_stats("EURUSD", TimeFrame.H1)
        
        self.assertIsNotNone(stats)
        self.assertEqual(stats.symbol, "EURUSD")
        self.assertEqual(stats.timeframe, TimeFrame.H1)
        self.assertGreater(stats.total_bars, 0)
    
    @patch('src.data.pipeline.mt5')
    @patch('src.data.mt5_connection.mt5')
    def test_pipeline_stats(self, mock_mt5_conn, mock_mt5_pipeline):
        """测试管道统计信息"""
        # 模拟MT5连接成功
        mock_mt5_conn.initialize.return_value = True
        mock_mt5_conn.account_info.return_value = Mock()
        mock_mt5_conn.terminal_info.return_value = Mock()
        
        # 模拟获取数据成功
        mock_mt5_pipeline.copy_rates_from_pos.return_value = self.mock_rates
        mock_mt5_pipeline.TIMEFRAME_H1 = 16385
        
        pipeline = DataPipeline(self.connection_config)
        
        # 执行一些操作
        data = pipeline.get_realtime_data("EURUSD", TimeFrame.H1, count=100)
        
        # 获取统计信息
        stats = pipeline.stats
        
        self.assertIn('requests_total', stats)
        self.assertIn('requests_successful', stats)
        self.assertIn('connection_status', stats)
        self.assertGreater(stats['requests_total'], 0)
    
    @patch('src.data.pipeline.mt5')
    @patch('src.data.mt5_connection.mt5')
    def test_context_manager(self, mock_mt5_conn, mock_mt5_pipeline):
        """测试上下文管理器"""
        # 模拟MT5连接成功
        mock_mt5_conn.initialize.return_value = True
        mock_mt5_conn.account_info.return_value = Mock()
        mock_mt5_conn.terminal_info.return_value = Mock()
        
        with DataPipeline(self.connection_config) as pipeline:
            self.assertIsNotNone(pipeline)
        
        # 验证资源已清理（通过检查线程池是否关闭）
        self.assertTrue(pipeline.executor._shutdown)


if __name__ == '__main__':
    unittest.main()