#!/usr/bin/env python3
"""
数据库管理器单元测试
"""

import unittest
import tempfile
import shutil
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data.database import DatabaseManager, DatabaseError
from src.data.models import MarketData, Tick, TimeFrame


class TestDatabaseManager(unittest.TestCase):
    """数据库管理器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建临时目录和数据库
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.db_manager = DatabaseManager(str(self.db_path))
        
        # 创建测试数据
        self.test_market_data = self._create_test_market_data()
        self.test_ticks = self._create_test_ticks()
        self.test_trade_data = self._create_test_trade_data()
    
    def tearDown(self):
        """测试后清理"""
        self.db_manager.close()
        shutil.rmtree(self.temp_dir)
    
    def _create_test_market_data(self) -> MarketData:
        """创建测试市场数据"""
        ohlcv = pd.DataFrame({
            'open': [1.1000, 1.1010, 1.1020],
            'high': [1.1005, 1.1015, 1.1025],
            'low': [1.0995, 1.1005, 1.1015],
            'close': [1.1002, 1.1012, 1.1022],
            'volume': [1000, 1100, 1200],
            'real_volume': [1000, 1100, 1200]
        }, index=pd.date_range('2024-01-01 10:00:00', periods=3, freq='1h'))
        
        return MarketData(
            symbol="EURUSD",
            timeframe=TimeFrame.H1,
            timestamp=datetime.now(),
            ohlcv=ohlcv,
            indicators={'sma_20': 1.1010, 'rsi': 55.5},
            spread=0.0002
        )
    
    def _create_test_ticks(self) -> list:
        """创建测试Tick数据"""
        ticks = []
        base_time = datetime.now()
        
        for i in range(5):
            tick = Tick(
                symbol="EURUSD",
                timestamp=base_time + timedelta(seconds=i),
                bid=1.1000 + i * 0.00001,
                ask=1.1002 + i * 0.00001,
                last=1.1001 + i * 0.00001,
                volume=100 + i * 10,
                flags=0
            )
            ticks.append(tick)
        
        return ticks
    
    def _create_test_trade_data(self) -> dict:
        """创建测试交易数据"""
        return {
            'trade_id': 'TEST_001',
            'symbol': 'EURUSD',
            'trade_type': 'BUY',
            'volume': 0.1,
            'open_price': 1.1000,
            'close_price': 1.1020,
            'sl': 1.0980,
            'tp': 1.1050,
            'profit': 20.0,
            'commission': 0.5,
            'swap': 0.0,
            'open_time': datetime.now() - timedelta(hours=1),
            'close_time': datetime.now(),
            'strategy_id': 'test_strategy',
            'comment': 'Test trade'
        }
    
    def test_database_initialization(self):
        """测试数据库初始化"""
        self.assertTrue(self.db_path.exists())
        
        # 检查数据库统计
        stats = self.db_manager.get_database_stats()
        self.assertIn('market_data_count', stats)
        self.assertIn('tick_data_count', stats)
        self.assertIn('trades_count', stats)
        self.assertEqual(stats['market_data_count'], 0)  # 初始应该为空
    
    def test_save_and_get_market_data(self):
        """测试保存和获取市场数据"""
        # 保存市场数据
        result = self.db_manager.save_market_data(self.test_market_data)
        self.assertTrue(result)
        
        # 获取市场数据
        retrieved_data = self.db_manager.get_market_data("EURUSD", TimeFrame.H1)
        
        self.assertIsNotNone(retrieved_data)
        self.assertEqual(len(retrieved_data), 3)
        self.assertIn('open', retrieved_data.columns)
        self.assertIn('high', retrieved_data.columns)
        self.assertIn('low', retrieved_data.columns)
        self.assertIn('close', retrieved_data.columns)
        self.assertIn('volume', retrieved_data.columns)
        
        # 验证数据值
        self.assertAlmostEqual(retrieved_data['open'].iloc[0], 1.1000, places=4)
        self.assertAlmostEqual(retrieved_data['close'].iloc[-1], 1.1022, places=4)
    
    def test_save_and_get_tick_data(self):
        """测试保存和获取Tick数据"""
        # 保存Tick数据
        result = self.db_manager.save_tick_data(self.test_ticks)
        self.assertTrue(result)
        
        # 获取Tick数据
        retrieved_ticks = self.db_manager.get_tick_data("EURUSD")
        
        self.assertIsNotNone(retrieved_ticks)
        self.assertEqual(len(retrieved_ticks), 5)
        
        # 验证第一个Tick
        first_tick = retrieved_ticks[-1]  # 数据是按时间倒序返回的
        self.assertEqual(first_tick.symbol, "EURUSD")
        self.assertAlmostEqual(first_tick.bid, 1.1000, places=5)
        self.assertAlmostEqual(first_tick.ask, 1.1002, places=5)
    
    def test_save_and_get_trade_records(self):
        """测试保存和获取交易记录"""
        # 保存交易记录
        result = self.db_manager.save_trade_record(self.test_trade_data)
        self.assertTrue(result)
        
        # 获取交易记录
        retrieved_trades = self.db_manager.get_trade_records()
        
        self.assertIsNotNone(retrieved_trades)
        self.assertEqual(len(retrieved_trades), 1)
        
        # 验证交易数据
        trade = retrieved_trades.iloc[0]
        self.assertEqual(trade['trade_id'], 'TEST_001')
        self.assertEqual(trade['symbol'], 'EURUSD')
        self.assertEqual(trade['trade_type'], 'BUY')
        self.assertAlmostEqual(trade['volume'], 0.1, places=2)
        self.assertAlmostEqual(trade['profit'], 20.0, places=2)
    
    def test_get_trade_records_with_filters(self):
        """测试带过滤条件的交易记录查询"""
        # 保存多个交易记录
        trade_data_2 = self.test_trade_data.copy()
        trade_data_2['trade_id'] = 'TEST_002'
        trade_data_2['symbol'] = 'GBPUSD'
        trade_data_2['strategy_id'] = 'another_strategy'
        
        self.db_manager.save_trade_record(self.test_trade_data)
        self.db_manager.save_trade_record(trade_data_2)
        
        # 按品种过滤
        eurusd_trades = self.db_manager.get_trade_records(symbol='EURUSD')
        self.assertEqual(len(eurusd_trades), 1)
        self.assertEqual(eurusd_trades.iloc[0]['symbol'], 'EURUSD')
        
        # 按策略过滤
        strategy_trades = self.db_manager.get_trade_records(strategy_id='test_strategy')
        self.assertEqual(len(strategy_trades), 1)
        self.assertEqual(strategy_trades.iloc[0]['strategy_id'], 'test_strategy')
    
    def test_save_and_get_strategy_parameters(self):
        """测试保存和获取策略参数"""
        # 保存策略参数
        parameters = {
            'ma_period': 20,
            'rsi_period': 14,
            'risk_percent': 2.0
        }
        
        performance_metrics = {
            'total_return': 15.5,
            'sharpe_ratio': 1.2,
            'max_drawdown': -5.0
        }
        
        result = self.db_manager.save_strategy_parameters(
            'test_strategy', 'EURUSD', parameters, performance_metrics
        )
        self.assertTrue(result)
        
        # 获取策略参数
        retrieved_params = self.db_manager.get_strategy_parameters('test_strategy', 'EURUSD')
        
        self.assertIsNotNone(retrieved_params)
        self.assertEqual(len(retrieved_params), 1)
        
        # 验证参数
        param_row = retrieved_params.iloc[0]
        self.assertEqual(param_row['strategy_name'], 'test_strategy')
        self.assertEqual(param_row['symbol'], 'EURUSD')
        self.assertEqual(param_row['parameters']['ma_period'], 20)
        self.assertEqual(param_row['performance_metrics']['total_return'], 15.5)
    
    def test_market_data_with_time_range(self):
        """测试按时间范围获取市场数据"""
        # 保存市场数据
        self.db_manager.save_market_data(self.test_market_data)
        
        # 设置时间范围
        start_time = datetime(2024, 1, 1, 10, 30)
        end_time = datetime(2024, 1, 1, 12, 30)
        
        # 获取指定时间范围的数据
        filtered_data = self.db_manager.get_market_data(
            "EURUSD", TimeFrame.H1, start_time, end_time
        )
        
        self.assertIsNotNone(filtered_data)
        # 应该只有2条记录在这个时间范围内
        self.assertEqual(len(filtered_data), 2)
    
    def test_market_data_with_limit(self):
        """测试限制返回条数的市场数据查询"""
        # 保存市场数据
        self.db_manager.save_market_data(self.test_market_data)
        
        # 限制返回1条记录
        limited_data = self.db_manager.get_market_data("EURUSD", TimeFrame.H1, limit=1)
        
        self.assertIsNotNone(limited_data)
        self.assertEqual(len(limited_data), 1)
    
    def test_database_backup_and_restore(self):
        """测试数据库备份和恢复"""
        # 保存一些数据
        self.db_manager.save_market_data(self.test_market_data)
        
        # 备份数据库
        backup_path = Path(self.temp_dir) / "backup.db"
        result = self.db_manager.backup_database(str(backup_path))
        self.assertTrue(result)
        self.assertTrue(backup_path.exists())
        
        # 清空原数据库（模拟数据丢失）
        self.db_manager.close()
        self.db_path.unlink()
        
        # 重新创建数据库管理器
        self.db_manager = DatabaseManager(str(self.db_path))
        
        # 验证数据为空
        empty_data = self.db_manager.get_market_data("EURUSD", TimeFrame.H1)
        self.assertIsNone(empty_data)
        
        # 恢复数据库
        result = self.db_manager.restore_database(str(backup_path))
        self.assertTrue(result)
        
        # 验证数据已恢复
        restored_data = self.db_manager.get_market_data("EURUSD", TimeFrame.H1)
        self.assertIsNotNone(restored_data)
        self.assertEqual(len(restored_data), 3)
    
    def test_database_stats(self):
        """测试数据库统计信息"""
        # 保存一些数据
        self.db_manager.save_market_data(self.test_market_data)
        self.db_manager.save_tick_data(self.test_ticks)
        self.db_manager.save_trade_record(self.test_trade_data)
        
        # 获取统计信息
        stats = self.db_manager.get_database_stats()
        
        self.assertIn('market_data_count', stats)
        self.assertIn('tick_data_count', stats)
        self.assertIn('trades_count', stats)
        self.assertIn('db_size_bytes', stats)
        self.assertIn('db_size_mb', stats)
        
        self.assertEqual(stats['market_data_count'], 3)  # 3条K线数据
        self.assertEqual(stats['tick_data_count'], 5)    # 5条Tick数据
        self.assertEqual(stats['trades_count'], 1)       # 1条交易记录
        self.assertGreater(stats['db_size_bytes'], 0)
    
    def test_cleanup_old_data(self):
        """测试清理旧数据"""
        # 创建一些旧的Tick数据
        old_ticks = []
        old_time = datetime.now() - timedelta(days=35)  # 35天前的数据
        
        for i in range(3):
            tick = Tick(
                symbol="EURUSD",
                timestamp=old_time + timedelta(seconds=i),
                bid=1.1000,
                ask=1.1002,
                last=1.1001,
                volume=100,
                flags=0
            )
            old_ticks.append(tick)
        
        # 保存旧数据和新数据
        self.db_manager.save_tick_data(old_ticks)
        self.db_manager.save_tick_data(self.test_ticks)
        
        # 验证数据总数
        all_ticks = self.db_manager.get_tick_data("EURUSD")
        self.assertEqual(len(all_ticks), 8)  # 3条旧数据 + 5条新数据
        
        # 清理30天前的数据
        result = self.db_manager.cleanup_old_data(days_to_keep=30)
        self.assertTrue(result)
        
        # 验证旧数据已被清理
        remaining_ticks = self.db_manager.get_tick_data("EURUSD")
        self.assertEqual(len(remaining_ticks), 5)  # 只剩下5条新数据
    
    def test_context_manager(self):
        """测试上下文管理器"""
        temp_db_path = Path(self.temp_dir) / "context_test.db"
        
        with DatabaseManager(str(temp_db_path)) as db:
            # 保存一些数据
            result = db.save_market_data(self.test_market_data)
            self.assertTrue(result)
            
            # 验证数据存在
            data = db.get_market_data("EURUSD", TimeFrame.H1)
            self.assertIsNotNone(data)
        
        # 上下文退出后，数据库文件应该仍然存在
        self.assertTrue(temp_db_path.exists())
    
    def test_invalid_backup_restore(self):
        """测试无效的备份恢复操作"""
        # 尝试从不存在的备份文件恢复
        result = self.db_manager.restore_database("nonexistent_backup.db")
        self.assertFalse(result)
    
    def test_duplicate_trade_id(self):
        """测试重复交易ID的处理"""
        # 保存交易记录
        result1 = self.db_manager.save_trade_record(self.test_trade_data)
        self.assertTrue(result1)
        
        # 保存相同ID的交易记录（应该替换原记录）
        modified_trade = self.test_trade_data.copy()
        modified_trade['profit'] = 30.0  # 修改利润
        
        result2 = self.db_manager.save_trade_record(modified_trade)
        self.assertTrue(result2)
        
        # 验证只有一条记录，且利润已更新
        trades = self.db_manager.get_trade_records()
        self.assertEqual(len(trades), 1)
        self.assertAlmostEqual(trades.iloc[0]['profit'], 30.0, places=2)


if __name__ == '__main__':
    unittest.main()