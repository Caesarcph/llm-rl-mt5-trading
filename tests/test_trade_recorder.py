"""
交易记录系统测试
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta
import tempfile
import shutil
from pathlib import Path
import json

from src.core.trade_recorder import (
    TradeRecorder, TradeRecord, TradeStatus, TradeOutcome
)
from src.core.models import Position, PositionType
from src.data.database import DatabaseManager


class TestTradeRecord(unittest.TestCase):
    """测试TradeRecord数据类"""
    
    def test_trade_record_creation(self):
        """测试交易记录创建"""
        trade = TradeRecord(
            trade_id="12345",
            symbol="EURUSD",
            trade_type="BUY",
            volume=0.1,
            open_price=1.1000,
            open_time=datetime.now(),
            strategy_id="test_strategy"
        )
        
        self.assertEqual(trade.trade_id, "12345")
        self.assertEqual(trade.symbol, "EURUSD")
        self.assertEqual(trade.status, TradeStatus.OPEN)
    
    def test_trade_record_to_dict(self):
        """测试转换为字典"""
        trade = TradeRecord(
            trade_id="12345",
            symbol="EURUSD",
            trade_type="BUY",
            volume=0.1,
            open_price=1.1000,
            open_time=datetime.now(),
            strategy_id="test_strategy"
        )
        
        trade_dict = trade.to_dict()
        
        self.assertIsInstance(trade_dict, dict)
        self.assertEqual(trade_dict['trade_id'], "12345")
        self.assertEqual(trade_dict['status'], 'open')
    
    def test_trade_record_from_dict(self):
        """测试从字典创建"""
        data = {
            'trade_id': "12345",
            'symbol': "EURUSD",
            'trade_type': "BUY",
            'volume': 0.1,
            'open_price': 1.1000,
            'open_time': datetime.now().isoformat(),
            'strategy_id': "test_strategy",
            'status': 'open'
        }
        
        trade = TradeRecord.from_dict(data)
        
        self.assertEqual(trade.trade_id, "12345")
        self.assertEqual(trade.status, TradeStatus.OPEN)


class TestTradeRecorder(unittest.TestCase):
    """测试TradeRecorder类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建模拟数据库管理器
        self.mock_db = Mock(spec=DatabaseManager)
        self.mock_db.save_trade_record = Mock(return_value=True)
        self.mock_db.get_trade_records = Mock(return_value=None)
        
        # 创建交易记录器
        self.recorder = TradeRecorder(
            db_manager=self.mock_db,
            config={'log_dir': self.temp_dir}
        )
    
    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir)
    
    def test_recorder_initialization(self):
        """测试记录器初始化"""
        self.assertIsNotNone(self.recorder)
        self.assertEqual(len(self.recorder.active_trades), 0)
        self.assertEqual(len(self.recorder.recent_closed_trades), 0)
    
    def test_record_trade_open(self):
        """测试记录开仓"""
        position = Position(
            position_id="12345",
            symbol="EURUSD",
            type=PositionType.LONG,
            volume=0.1,
            open_price=1.1000,
            current_price=1.1000,
            sl=1.0950,
            tp=1.1100,
            profit=0.0,
            swap=0.0,
            commission=0.0,
            open_time=datetime.now(),
            comment="Test trade",
            magic_number=123
        )
        
        trade_record = self.recorder.record_trade_open(position, "test_strategy")
        
        self.assertIsNotNone(trade_record)
        self.assertEqual(trade_record.trade_id, "12345")
        self.assertEqual(trade_record.status, TradeStatus.OPEN)
        self.assertIn("12345", self.recorder.active_trades)
        self.mock_db.save_trade_record.assert_called_once()
    
    def test_record_trade_close(self):
        """测试记录平仓"""
        # 先记录开仓
        position = Position(
            position_id="12345",
            symbol="EURUSD",
            type=PositionType.LONG,
            volume=0.1,
            open_price=1.1000,
            current_price=1.1050,
            sl=1.0950,
            tp=1.1100,
            profit=50.0,
            swap=0.0,
            commission=0.0,
            open_time=datetime.now() - timedelta(hours=2),
            comment="Test trade",
            magic_number=123
        )
        
        self.recorder.record_trade_open(position, "test_strategy")
        
        # 记录平仓
        trade_record = self.recorder.record_trade_close(
            position=position,
            close_price=1.1050,
            profit=50.0,
            commission=2.0,
            swap=1.0
        )
        
        self.assertIsNotNone(trade_record)
        self.assertEqual(trade_record.status, TradeStatus.CLOSED)
        self.assertEqual(trade_record.profit, 50.0)
        self.assertEqual(trade_record.commission, 2.0)
        self.assertNotIn("12345", self.recorder.active_trades)
        self.assertIn(trade_record, self.recorder.recent_closed_trades)
    
    def test_calculate_trade_metrics(self):
        """测试计算交易指标"""
        trade_record = TradeRecord(
            trade_id="12345",
            symbol="EURUSD",
            trade_type="BUY",
            volume=0.1,
            open_price=1.1000,
            open_time=datetime.now() - timedelta(hours=2),
            close_price=1.1050,
            close_time=datetime.now(),
            profit=50.0,
            commission=2.0,
            swap=1.0,
            strategy_id="test_strategy",
            sl=1.0950,
            tp=1.1100
        )
        
        self.recorder._calculate_trade_metrics(trade_record)
        
        self.assertIsNotNone(trade_record.pips)
        self.assertIsNotNone(trade_record.duration_hours)
        self.assertEqual(trade_record.outcome, TradeOutcome.WIN)
        self.assertIsNotNone(trade_record.risk_reward_ratio)
    
    def test_get_trade_statistics_empty(self):
        """测试获取空统计信息"""
        stats = self.recorder.get_trade_statistics()
        
        self.assertEqual(stats['total_trades'], 0)
    
    def test_get_trade_statistics_with_trades(self):
        """测试获取交易统计"""
        # 模拟数据库返回交易记录
        import pandas as pd
        
        mock_df = pd.DataFrame([
            {
                'trade_id': '1',
                'symbol': 'EURUSD',
                'trade_type': 'BUY',
                'volume': 0.1,
                'open_price': 1.1000,
                'close_price': 1.1050,
                'sl': 1.0950,
                'tp': 1.1100,
                'profit': 50.0,
                'commission': 2.0,
                'swap': 1.0,
                'open_time': datetime.now() - timedelta(hours=2),
                'close_time': datetime.now(),
                'strategy_id': 'test_strategy',
                'comment': 'Test'
            },
            {
                'trade_id': '2',
                'symbol': 'EURUSD',
                'trade_type': 'SELL',
                'volume': 0.1,
                'open_price': 1.1050,
                'close_price': 1.1100,
                'sl': 1.1150,
                'tp': 1.1000,
                'profit': -50.0,
                'commission': 2.0,
                'swap': 1.0,
                'open_time': datetime.now() - timedelta(hours=1),
                'close_time': datetime.now(),
                'strategy_id': 'test_strategy',
                'comment': 'Test'
            }
        ])
        
        self.mock_db.get_trade_records.return_value = mock_df
        
        stats = self.recorder.get_trade_statistics()
        
        self.assertEqual(stats['closed_trades'], 2)
        self.assertEqual(stats['winning_trades'], 1)
        self.assertEqual(stats['losing_trades'], 1)
        self.assertEqual(stats['win_rate'], 50.0)
    
    def test_generate_daily_report(self):
        """测试生成每日报告"""
        # 模拟空数据
        self.mock_db.get_trade_records.return_value = None
        
        report = self.recorder.generate_daily_report()
        
        self.assertIsNotNone(report)
        self.assertIn('date', report)
        self.assertIn('summary', report)
        self.assertIn('trades', report)
    
    def test_generate_weekly_report(self):
        """测试生成每周报告"""
        self.mock_db.get_trade_records.return_value = None
        
        report = self.recorder.generate_weekly_report()
        
        self.assertIsNotNone(report)
        self.assertIn('week_start', report)
        self.assertIn('week_end', report)
        self.assertIn('summary', report)
        self.assertIn('daily_breakdown', report)
    
    def test_generate_monthly_report(self):
        """测试生成每月报告"""
        self.mock_db.get_trade_records.return_value = None
        
        report = self.recorder.generate_monthly_report()
        
        self.assertIsNotNone(report)
        self.assertIn('month', report)
        self.assertIn('summary', report)
        self.assertIn('weekly_breakdown', report)
    
    def test_write_trade_log(self):
        """测试写入交易日志"""
        trade_record = TradeRecord(
            trade_id="12345",
            symbol="EURUSD",
            trade_type="BUY",
            volume=0.1,
            open_price=1.1000,
            open_time=datetime.now(),
            strategy_id="test_strategy"
        )
        
        self.recorder._write_trade_log(trade_record, "OPEN")
        
        log_file = Path(self.temp_dir) / "trades.log"
        self.assertTrue(log_file.exists())
        
        # 验证日志内容
        with open(log_file, 'r', encoding='utf-8') as f:
            log_line = f.readline()
            log_entry = json.loads(log_line)
            
            self.assertEqual(log_entry['action'], 'OPEN')
            self.assertEqual(log_entry['trade_id'], '12345')
    
    def test_get_audit_log(self):
        """测试获取审计日志"""
        # 先写入一些日志
        trade_record = TradeRecord(
            trade_id="12345",
            symbol="EURUSD",
            trade_type="BUY",
            volume=0.1,
            open_price=1.1000,
            open_time=datetime.now(),
            strategy_id="test_strategy"
        )
        
        self.recorder._write_trade_log(trade_record, "OPEN")
        
        # 获取审计日志
        audit_logs = self.recorder.get_audit_log()
        
        self.assertIsInstance(audit_logs, list)
        self.assertGreater(len(audit_logs), 0)
        self.assertEqual(audit_logs[0]['trade_id'], '12345')
    
    def test_get_audit_log_with_filter(self):
        """测试带过滤条件的审计日志"""
        # 写入多条日志
        for i in range(3):
            trade_record = TradeRecord(
                trade_id=f"1234{i}",
                symbol="EURUSD",
                trade_type="BUY",
                volume=0.1,
                open_price=1.1000,
                open_time=datetime.now(),
                strategy_id="test_strategy"
            )
            self.recorder._write_trade_log(trade_record, "OPEN")
        
        # 按trade_id过滤
        audit_logs = self.recorder.get_audit_log(trade_id="12341")
        
        self.assertEqual(len(audit_logs), 1)
        self.assertEqual(audit_logs[0]['trade_id'], '12341')
    
    def test_export_trades_to_csv(self):
        """测试导出交易记录到CSV"""
        import pandas as pd
        
        # 模拟交易数据
        mock_df = pd.DataFrame([
            {
                'trade_id': '1',
                'symbol': 'EURUSD',
                'trade_type': 'BUY',
                'volume': 0.1,
                'open_price': 1.1000,
                'close_price': 1.1050,
                'sl': 1.0950,
                'tp': 1.1100,
                'profit': 50.0,
                'commission': 2.0,
                'swap': 1.0,
                'open_time': datetime.now(),
                'close_time': datetime.now(),
                'strategy_id': 'test_strategy',
                'comment': 'Test'
            }
        ])
        
        self.mock_db.get_trade_records.return_value = mock_df
        
        csv_path = Path(self.temp_dir) / "trades.csv"
        result = self.recorder.export_trades_to_csv(str(csv_path))
        
        self.assertTrue(result)
        self.assertTrue(csv_path.exists())
    
    def test_get_performance_summary(self):
        """测试获取性能摘要"""
        self.mock_db.get_trade_records.return_value = None
        
        summary = self.recorder.get_performance_summary()
        
        self.assertIsNotNone(summary)
        self.assertIn('all_time', summary)
        self.assertIn('last_30_days', summary)
        self.assertIn('today', summary)
        self.assertIn('active_trades', summary)
    
    def test_calculate_consecutive_trades(self):
        """测试计算连续盈亏"""
        trades = [
            TradeRecord(
                trade_id=f"{i}",
                symbol="EURUSD",
                trade_type="BUY",
                volume=0.1,
                open_price=1.1000,
                open_time=datetime.now() - timedelta(hours=i),
                close_time=datetime.now() - timedelta(hours=i-1),
                strategy_id="test",
                outcome=TradeOutcome.WIN if i % 2 == 0 else TradeOutcome.LOSS
            )
            for i in range(10)
        ]
        
        max_wins, max_losses = self.recorder._calculate_consecutive_trades(trades)
        
        self.assertIsInstance(max_wins, int)
        self.assertIsInstance(max_losses, int)
    
    def test_cleanup_old_logs(self):
        """测试清理旧日志"""
        # 写入一些日志
        old_time = datetime.now() - timedelta(days=100)
        recent_time = datetime.now()
        
        log_file = Path(self.temp_dir) / "trades.log"
        
        with open(log_file, 'w', encoding='utf-8') as f:
            # 旧日志
            old_log = {
                'timestamp': old_time.isoformat(),
                'action': 'OPEN',
                'trade_id': 'old_trade'
            }
            f.write(json.dumps(old_log) + '\n')
            
            # 新日志
            new_log = {
                'timestamp': recent_time.isoformat(),
                'action': 'OPEN',
                'trade_id': 'new_trade'
            }
            f.write(json.dumps(new_log) + '\n')
        
        # 清理旧日志（保留30天）
        result = self.recorder.cleanup_old_logs(days_to_keep=30)
        
        self.assertTrue(result)
        
        # 验证只保留了新日志
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 1)
            log_entry = json.loads(lines[0])
            self.assertEqual(log_entry['trade_id'], 'new_trade')
    
    def test_find_best_and_worst_day(self):
        """测试找出最佳和最差交易日"""
        trades = [
            TradeRecord(
                trade_id="1",
                symbol="EURUSD",
                trade_type="BUY",
                volume=0.1,
                open_price=1.1000,
                open_time=datetime(2024, 1, 1, 10, 0),
                close_time=datetime(2024, 1, 1, 12, 0),
                profit=100.0,
                strategy_id="test",
                status=TradeStatus.CLOSED
            ),
            TradeRecord(
                trade_id="2",
                symbol="EURUSD",
                trade_type="BUY",
                volume=0.1,
                open_price=1.1000,
                open_time=datetime(2024, 1, 2, 10, 0),
                close_time=datetime(2024, 1, 2, 12, 0),
                profit=-50.0,
                strategy_id="test",
                status=TradeStatus.CLOSED
            )
        ]
        
        best_day = self.recorder._find_best_day(trades)
        worst_day = self.recorder._find_worst_day(trades)
        
        self.assertEqual(best_day['date'], '2024-01-01')
        self.assertEqual(best_day['profit'], 100.0)
        self.assertEqual(worst_day['date'], '2024-01-02')
        self.assertEqual(worst_day['profit'], -50.0)


class TestTradeRecorderIntegration(unittest.TestCase):
    """交易记录器集成测试"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建真实的数据库管理器
        db_path = Path(self.temp_dir) / "test.db"
        self.db_manager = DatabaseManager(str(db_path))
        
        # 创建交易记录器
        self.recorder = TradeRecorder(
            db_manager=self.db_manager,
            config={'log_dir': self.temp_dir}
        )
    
    def tearDown(self):
        """测试后清理"""
        self.db_manager.close()
        shutil.rmtree(self.temp_dir)
    
    def test_full_trade_lifecycle(self):
        """测试完整的交易生命周期"""
        # 创建持仓
        position = Position(
            position_id="12345",
            symbol="EURUSD",
            type=PositionType.LONG,
            volume=0.1,
            open_price=1.1000,
            current_price=1.1000,
            sl=1.0950,
            tp=1.1100,
            profit=0.0,
            swap=0.0,
            commission=0.0,
            open_time=datetime.now(),
            comment="Integration test",
            magic_number=123
        )
        
        # 记录开仓
        open_record = self.recorder.record_trade_open(position, "test_strategy")
        self.assertIsNotNone(open_record)
        
        # 更新持仓价格
        position.current_price = 1.1050
        position.profit = 50.0
        
        # 记录平仓
        close_record = self.recorder.record_trade_close(
            position=position,
            close_price=1.1050,
            profit=50.0,
            commission=2.0,
            swap=1.0
        )
        self.assertIsNotNone(close_record)
        
        # 获取交易历史
        trades = self.recorder.get_trade_history(symbol="EURUSD")
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].trade_id, "12345")
        
        # 获取统计信息
        stats = self.recorder.get_trade_statistics(symbol="EURUSD")
        self.assertEqual(stats['closed_trades'], 1)
        self.assertEqual(stats['winning_trades'], 1)


if __name__ == '__main__':
    unittest.main()
