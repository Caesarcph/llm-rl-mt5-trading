"""
仓位管理器测试
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import MetaTrader5 as mt5

from src.core.position_manager import PositionManager, TrailingStopType
from src.core.models import Position, PositionType, Account


class TestPositionManager(unittest.TestCase):
    """仓位管理器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.config = {
            'max_positions': 10,
            'max_risk_per_position': 0.02,
            'trailing_stop_enabled': True,
            'partial_close_enabled': True
        }
        self.manager = PositionManager(self.config)
        
        # 创建测试持仓
        self.test_position = Position(
            position_id='12345',
            symbol='EURUSD',
            type=PositionType.LONG,
            volume=0.1,
            open_price=1.1000,
            current_price=1.1050,
            sl=1.0950,
            tp=1.1100,
            profit=50.0,
            open_time=datetime.now() - timedelta(hours=2)
        )
        
        # 创建测试账户
        self.test_account = Account(
            account_id='test_account',
            balance=10000.0,
            equity=10050.0,
            margin=100.0,
            free_margin=9950.0,
            margin_level=10050.0,
            currency='USD',
            leverage=100
        )
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.manager.max_positions, 10)
        self.assertEqual(self.manager.max_risk_per_position, 0.02)
        self.assertTrue(self.manager.trailing_stop_enabled)
        self.assertTrue(self.manager.partial_close_enabled)
        self.assertIsInstance(self.manager.positions, dict)
    
    def test_get_position(self):
        """测试获取持仓"""
        self.manager.positions['12345'] = self.test_position
        
        position = self.manager.get_position('12345')
        self.assertIsNotNone(position)
        self.assertEqual(position.position_id, '12345')
    
    def test_get_position_not_found(self):
        """测试获取不存在的持仓"""
        position = self.manager.get_position('99999')
        self.assertIsNone(position)
    
    def test_get_positions_by_symbol(self):
        """测试按品种获取持仓"""
        pos1 = Position(
            position_id='1',
            symbol='EURUSD',
            type=PositionType.LONG,
            volume=0.1,
            open_price=1.1000,
            current_price=1.1050
        )
        pos2 = Position(
            position_id='2',
            symbol='GBPUSD',
            type=PositionType.LONG,
            volume=0.1,
            open_price=1.3000,
            current_price=1.3050
        )
        
        self.manager.positions['1'] = pos1
        self.manager.positions['2'] = pos2
        
        eurusd_positions = self.manager.get_positions_by_symbol('EURUSD')
        self.assertEqual(len(eurusd_positions), 1)
        self.assertEqual(eurusd_positions[0].symbol, 'EURUSD')
    
    def test_get_total_exposure(self):
        """测试获取总敞口"""
        pos1 = Position(
            position_id='1',
            symbol='EURUSD',
            type=PositionType.LONG,
            volume=0.1,
            open_price=1.1000,
            current_price=1.1050
        )
        pos2 = Position(
            position_id='2',
            symbol='EURUSD',
            type=PositionType.LONG,
            volume=0.2,
            open_price=1.1000,
            current_price=1.1050
        )
        
        self.manager.positions['1'] = pos1
        self.manager.positions['2'] = pos2
        
        total = self.manager.get_total_exposure('EURUSD')
        self.assertAlmostEqual(total, 0.3, places=5)
    
    def test_get_net_exposure(self):
        """测试获取净敞口"""
        pos1 = Position(
            position_id='1',
            symbol='EURUSD',
            type=PositionType.LONG,
            volume=0.3,
            open_price=1.1000,
            current_price=1.1050
        )
        pos2 = Position(
            position_id='2',
            symbol='EURUSD',
            type=PositionType.SHORT,
            volume=0.1,
            open_price=1.1000,
            current_price=1.0950
        )
        
        self.manager.positions['1'] = pos1
        self.manager.positions['2'] = pos2
        
        net = self.manager.get_net_exposure('EURUSD')
        self.assertAlmostEqual(net, 0.2, places=5)  # 0.3 - 0.1
    
    def test_enable_trailing_stop(self):
        """测试启用追踪止损"""
        self.manager.positions['12345'] = self.test_position
        
        result = self.manager.enable_trailing_stop(
            '12345',
            TrailingStopType.FIXED,
            50.0,
            10.0
        )
        
        self.assertTrue(result)
        self.assertIn('12345', self.manager.trailing_stops)
        self.assertEqual(self.manager.trailing_stops['12345']['type'], TrailingStopType.FIXED)
        self.assertEqual(self.manager.trailing_stops['12345']['value'], 50.0)
    
    def test_enable_trailing_stop_invalid_position(self):
        """测试对不存在的持仓启用追踪止损"""
        result = self.manager.enable_trailing_stop('99999', TrailingStopType.FIXED, 50.0)
        self.assertFalse(result)
    
    def test_monitor_position_risk(self):
        """测试监控持仓风险"""
        risk_metrics = self.manager.monitor_position_risk(
            self.test_position,
            self.test_account
        )
        
        self.assertEqual(risk_metrics['position_id'], '12345')
        self.assertEqual(risk_metrics['symbol'], 'EURUSD')
        self.assertEqual(risk_metrics['current_profit'], 50.0)
        self.assertTrue(risk_metrics['has_sl'])
        self.assertTrue(risk_metrics['has_tp'])
        self.assertGreater(risk_metrics['duration_hours'], 0)
    
    def test_monitor_position_risk_losing_position(self):
        """测试监控亏损持仓风险"""
        losing_position = Position(
            position_id='12345',
            symbol='EURUSD',
            type=PositionType.LONG,
            volume=0.1,
            open_price=1.1000,
            current_price=1.0950,
            sl=1.0900,
            tp=1.1100,
            profit=-50.0,
            open_time=datetime.now()
        )
        
        risk_metrics = self.manager.monitor_position_risk(
            losing_position,
            self.test_account
        )
        
        self.assertEqual(risk_metrics['risk_amount'], 50.0)
        self.assertGreater(risk_metrics['risk_percentage'], 0)
    
    def test_get_portfolio_risk_metrics_empty(self):
        """测试空组合风险指标"""
        metrics = self.manager.get_portfolio_risk_metrics(self.test_account)
        
        self.assertEqual(metrics['total_positions'], 0)
        self.assertEqual(metrics['total_profit'], 0)
        self.assertEqual(metrics['total_risk'], 0)
    
    def test_get_portfolio_risk_metrics(self):
        """测试组合风险指标"""
        pos1 = Position(
            position_id='1',
            symbol='EURUSD',
            type=PositionType.LONG,
            volume=0.1,
            open_price=1.1000,
            current_price=1.1050,
            profit=50.0,
            open_time=datetime.now()
        )
        pos2 = Position(
            position_id='2',
            symbol='GBPUSD',
            type=PositionType.LONG,
            volume=0.1,
            open_price=1.3000,
            current_price=1.2950,
            profit=-50.0,
            open_time=datetime.now()
        )
        
        self.manager.positions['1'] = pos1
        self.manager.positions['2'] = pos2
        
        metrics = self.manager.get_portfolio_risk_metrics(self.test_account)
        
        self.assertEqual(metrics['total_positions'], 2)
        self.assertEqual(metrics['total_profit'], 0.0)  # 50 - 50
        self.assertEqual(metrics['total_risk'], 50.0)
        self.assertGreater(metrics['risk_percentage'], 0)
    
    def test_check_position_limits_ok(self):
        """测试持仓限制检查（正常）"""
        # 添加少量持仓
        for i in range(5):
            pos = Position(
                position_id=str(i),
                symbol='EURUSD',
                type=PositionType.LONG,
                volume=0.1,
                open_price=1.1000,
                current_price=1.1050
            )
            self.manager.positions[str(i)] = pos
        
        ok, error = self.manager.check_position_limits()
        self.assertTrue(ok)
        self.assertIsNone(error)
    
    def test_check_position_limits_exceeded(self):
        """测试持仓限制检查（超限）"""
        # 添加超过限制的持仓
        for i in range(10):
            pos = Position(
                position_id=str(i),
                symbol='EURUSD',
                type=PositionType.LONG,
                volume=0.1,
                open_price=1.1000,
                current_price=1.1050
            )
            self.manager.positions[str(i)] = pos
        
        ok, error = self.manager.check_position_limits()
        self.assertFalse(ok)
        self.assertIsNotNone(error)
    
    def test_archive_position(self):
        """测试归档持仓"""
        self.manager._archive_position(self.test_position)
        
        self.assertEqual(len(self.manager.position_history), 1)
        self.assertEqual(self.manager.position_history[0]['position_id'], '12345')
    
    def test_archive_position_limit(self):
        """测试归档持仓数量限制"""
        # 添加超过1000条记录
        for i in range(1100):
            pos = Position(
                position_id=str(i),
                symbol='EURUSD',
                type=PositionType.LONG,
                volume=0.1,
                open_price=1.1000,
                current_price=1.1050,
                open_time=datetime.now()
            )
            self.manager._archive_position(pos)
        
        # 应该只保留最近1000条
        self.assertEqual(len(self.manager.position_history), 1000)
    
    def test_get_position_statistics_empty(self):
        """测试空统计"""
        stats = self.manager.get_position_statistics()
        self.assertEqual(stats['count'], 0)
    
    def test_get_position_statistics(self):
        """测试持仓统计"""
        # 添加一些历史记录
        for i in range(10):
            self.manager.position_history.append({
                'position_id': str(i),
                'symbol': 'EURUSD',
                'type': 'LONG',
                'volume': 0.1,
                'open_price': 1.1000,
                'close_price': 1.1050 if i % 2 == 0 else 1.0950,
                'profit': 50.0 if i % 2 == 0 else -50.0,
                'open_time': datetime.now(),
                'close_time': datetime.now(),
                'duration_hours': 2.0
            })
        
        stats = self.manager.get_position_statistics()
        
        self.assertEqual(stats['count'], 10)
        self.assertEqual(stats['win_rate'], 50.0)
        self.assertAlmostEqual(stats['avg_profit'], 0.0, places=1)
        self.assertEqual(stats['avg_duration'], 2.0)
    
    def test_get_position_statistics_by_symbol(self):
        """测试按品种统计"""
        # 添加不同品种的历史记录
        self.manager.position_history.append({
            'position_id': '1',
            'symbol': 'EURUSD',
            'profit': 50.0,
            'duration_hours': 2.0
        })
        self.manager.position_history.append({
            'position_id': '2',
            'symbol': 'GBPUSD',
            'profit': 30.0,
            'duration_hours': 1.0
        })
        
        stats = self.manager.get_position_statistics('EURUSD')
        self.assertEqual(stats['count'], 1)
        self.assertEqual(stats['total_profit'], 50.0)
    
    @patch('MetaTrader5.initialize')
    @patch('MetaTrader5.order_send')
    def test_close_position_success(self, mock_order_send, mock_init):
        """测试成功关闭持仓"""
        mock_init.return_value = True
        
        # 模拟成功的平仓结果
        mock_result = Mock()
        mock_result.retcode = mt5.TRADE_RETCODE_DONE
        mock_order_send.return_value = mock_result
        
        self.manager.positions['12345'] = self.test_position
        
        result = self.manager.close_position('12345')
        self.assertTrue(result)
    
    def test_close_position_not_found(self):
        """测试关闭不存在的持仓"""
        result = self.manager.close_position('99999')
        self.assertFalse(result)
    
    @patch('MetaTrader5.initialize')
    @patch('MetaTrader5.order_send')
    def test_modify_position_success(self, mock_order_send, mock_init):
        """测试成功修改持仓"""
        mock_init.return_value = True
        
        # 模拟成功的修改结果
        mock_result = Mock()
        mock_result.retcode = mt5.TRADE_RETCODE_DONE
        mock_order_send.return_value = mock_result
        
        self.manager.positions['12345'] = self.test_position
        
        result = self.manager.modify_position('12345', sl=1.0960, tp=1.1110)
        self.assertTrue(result)
        self.assertEqual(self.test_position.sl, 1.0960)
        self.assertEqual(self.test_position.tp, 1.1110)
    
    def test_modify_position_not_found(self):
        """测试修改不存在的持仓"""
        result = self.manager.modify_position('99999', sl=1.0950)
        self.assertFalse(result)
    
    @patch('MetaTrader5.initialize')
    @patch('MetaTrader5.symbol_info')
    def test_calculate_trailing_stop_long_fixed(self, mock_symbol_info, mock_init):
        """测试计算多头固定追踪止损"""
        mock_init.return_value = True
        
        # 模拟品种信息
        mock_info = Mock()
        mock_info.point = 0.00001
        mock_symbol_info.return_value = mock_info
        
        trailing_config = {
            'type': TrailingStopType.FIXED,
            'value': 50.0,
            'highest_price': 1.1050
        }
        
        new_sl = self.manager._calculate_trailing_stop(self.test_position, trailing_config)
        
        self.assertIsNotNone(new_sl)
        self.assertLess(new_sl, 1.1050)
    
    def test_calculate_trailing_stop_long_percentage(self):
        """测试计算多头百分比追踪止损"""
        trailing_config = {
            'type': TrailingStopType.PERCENTAGE,
            'value': 1.0,  # 1%
            'highest_price': 1.1050
        }
        
        new_sl = self.manager._calculate_trailing_stop(self.test_position, trailing_config)
        
        self.assertIsNotNone(new_sl)
        expected = 1.1050 * 0.99
        self.assertAlmostEqual(new_sl, expected, places=4)
    
    def test_partial_close_position_invalid_percentage(self):
        """测试无效的部分平仓百分比"""
        self.manager.positions['12345'] = self.test_position
        
        result = self.manager.partial_close_position('12345', 0)
        self.assertFalse(result)
        
        result = self.manager.partial_close_position('12345', 100)
        self.assertFalse(result)
        
        result = self.manager.partial_close_position('12345', 150)
        self.assertFalse(result)
    
    def test_partial_close_disabled(self):
        """测试部分平仓功能禁用"""
        manager = PositionManager({'partial_close_enabled': False})
        manager.positions['12345'] = self.test_position
        
        result = manager.partial_close_position('12345', 50)
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
