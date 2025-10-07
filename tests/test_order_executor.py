"""
订单执行器测试
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import MetaTrader5 as mt5

from src.core.order_executor import OrderExecutor, SlippageModel
from src.core.models import Signal


class TestOrderExecutor(unittest.TestCase):
    """订单执行器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.config = {
            'max_retries': 3,
            'retry_delay': 0.1,
            'max_slippage': 0.001,
            'slippage_model': 'dynamic',
            'default_slippage': 0.0002
        }
        self.executor = OrderExecutor(self.config)
        
        # 创建测试信号
        self.test_signal = Signal(
            strategy_id='test_strategy',
            symbol='EURUSD',
            direction=1,
            strength=0.8,
            entry_price=1.1000,
            sl=1.0950,
            tp=1.1100,
            size=0.1,
            confidence=0.85,
            timestamp=datetime.now(),
            metadata={'magic_number': 12345}
        )
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.executor.max_retries, 3)
        self.assertEqual(self.executor.max_slippage, 0.001)
        self.assertEqual(self.executor.slippage_model, SlippageModel.DYNAMIC)
        self.assertIsInstance(self.executor.pending_orders, dict)
        self.assertIsInstance(self.executor.order_history, list)
    
    def test_validate_signal_valid(self):
        """测试有效信号验证"""
        result = self.executor._validate_signal(self.test_signal)
        self.assertTrue(result)
    
    def test_validate_signal_invalid_size(self):
        """测试无效手数"""
        # Signal模型在__post_init__中验证，所以创建时会抛出异常
        with self.assertRaises(ValueError):
            invalid_signal = Signal(
                strategy_id='test',
                symbol='EURUSD',
                direction=1,
                strength=0.8,
                entry_price=1.1000,
                sl=1.0950,
                tp=1.1100,
                size=0.0,  # 无效手数
                confidence=0.85,
                timestamp=datetime.now()
            )
    
    def test_validate_signal_invalid_sl_buy(self):
        """测试买入订单无效止损"""
        invalid_signal = Signal(
            strategy_id='test',
            symbol='EURUSD',
            direction=1,
            strength=0.8,
            entry_price=1.1000,
            sl=1.1050,  # 止损高于入场价
            tp=1.1100,
            size=0.1,
            confidence=0.85,
            timestamp=datetime.now()
        )
        result = self.executor._validate_signal(invalid_signal)
        self.assertFalse(result)
    
    def test_validate_signal_invalid_tp_sell(self):
        """测试卖出订单无效止盈"""
        invalid_signal = Signal(
            strategy_id='test',
            symbol='EURUSD',
            direction=-1,
            strength=0.8,
            entry_price=1.1000,
            sl=1.1050,
            tp=1.1100,  # 止盈高于入场价
            size=0.1,
            confidence=0.85,
            timestamp=datetime.now()
        )
        result = self.executor._validate_signal(invalid_signal)
        self.assertFalse(result)
    
    def test_predict_slippage_fixed(self):
        """测试固定滑点预测"""
        executor = OrderExecutor({'slippage_model': 'fixed', 'fixed_slippage': 0.0001})
        slippage = executor.predict_slippage('EURUSD', 0.1)
        self.assertEqual(slippage, 0.0001)
    
    def test_predict_slippage_historical(self):
        """测试历史滑点预测"""
        executor = OrderExecutor({'slippage_model': 'historical', 'default_slippage': 0.0002})
        
        # 添加历史数据
        executor.slippage_history['EURUSD'] = [0.0001, 0.0002, 0.0003]
        
        slippage = executor.predict_slippage('EURUSD', 0.1)
        self.assertGreater(slippage, 0)
        self.assertLess(slippage, 0.001)
    
    def test_predict_slippage_no_history(self):
        """测试无历史数据时的滑点预测"""
        executor = OrderExecutor({'slippage_model': 'historical', 'default_slippage': 0.0002})
        slippage = executor.predict_slippage('GBPUSD', 0.1)
        self.assertEqual(slippage, 0.0002)
    
    def test_adjust_price_for_slippage_buy(self):
        """测试买入价格调整"""
        price = 1.1000
        direction = 1
        slippage = 0.0001
        
        adjusted = self.executor._adjust_price_for_slippage(price, direction, slippage)
        self.assertGreater(adjusted, price)
        self.assertAlmostEqual(adjusted, 1.1001, places=4)
    
    def test_adjust_price_for_slippage_sell(self):
        """测试卖出价格调整"""
        price = 1.1000
        direction = -1
        slippage = 0.0001
        
        adjusted = self.executor._adjust_price_for_slippage(price, direction, slippage)
        self.assertLess(adjusted, price)
        self.assertAlmostEqual(adjusted, 1.0999, places=4)
    
    def test_calculate_actual_slippage(self):
        """测试实际滑点计算"""
        expected = 1.1000
        actual = 1.1002
        
        slippage = self.executor._calculate_actual_slippage(expected, actual)
        self.assertAlmostEqual(slippage, 0.0002 / 1.1000, places=6)
    
    def test_record_slippage(self):
        """测试滑点记录"""
        self.executor._record_slippage('EURUSD', 0.0001)
        self.executor._record_slippage('EURUSD', 0.0002)
        
        self.assertIn('EURUSD', self.executor.slippage_history)
        self.assertEqual(len(self.executor.slippage_history['EURUSD']), 2)
    
    def test_record_slippage_limit(self):
        """测试滑点记录数量限制"""
        # 添加超过100条记录
        for i in range(150):
            self.executor._record_slippage('EURUSD', 0.0001)
        
        # 应该只保留最近100条
        self.assertEqual(len(self.executor.slippage_history['EURUSD']), 100)
    
    def test_should_retry_retriable_error(self):
        """测试可重试错误"""
        # 网络错误等可重试
        result = self.executor._should_retry(10018)
        self.assertTrue(result)
    
    def test_should_retry_non_retriable_error(self):
        """测试不可重试错误"""
        # 资金不足不应重试
        result = self.executor._should_retry(10004)
        self.assertFalse(result)
        
        # 无效手数不应重试
        result = self.executor._should_retry(10013)
        self.assertFalse(result)
    
    def test_prepare_order_request_buy(self):
        """测试准备买入订单请求"""
        request = self.executor._prepare_order_request(self.test_signal, 1.1000)
        
        self.assertEqual(request['symbol'], 'EURUSD')
        self.assertEqual(request['volume'], 0.1)
        self.assertEqual(request['type'], mt5.ORDER_TYPE_BUY)
        self.assertEqual(request['price'], 1.1000)
        self.assertEqual(request['sl'], 1.0950)
        self.assertEqual(request['tp'], 1.1100)
        self.assertEqual(request['magic'], 12345)
    
    def test_prepare_order_request_sell(self):
        """测试准备卖出订单请求"""
        sell_signal = Signal(
            strategy_id='test_strategy',
            symbol='EURUSD',
            direction=-1,
            strength=0.8,
            entry_price=1.1000,
            sl=1.1050,
            tp=1.0900,
            size=0.1,
            confidence=0.85,
            timestamp=datetime.now()
        )
        
        request = self.executor._prepare_order_request(sell_signal, 1.1000)
        self.assertEqual(request['type'], mt5.ORDER_TYPE_SELL)
    
    def test_create_order_result(self):
        """测试创建订单结果"""
        result = self.executor._create_order_result(
            success=True,
            signal=self.test_signal,
            order_id='12345',
            fill_price=1.1001
        )
        
        self.assertTrue(result['success'])
        self.assertEqual(result['symbol'], 'EURUSD')
        self.assertEqual(result['order_id'], '12345')
        self.assertEqual(result['fill_price'], 1.1001)
        self.assertIn('timestamp', result)
    
    def test_add_to_history(self):
        """测试添加到历史记录"""
        result = {'order_id': '12345', 'success': True}
        self.executor._add_to_history(result)
        
        self.assertEqual(len(self.executor.order_history), 1)
        self.assertEqual(self.executor.order_history[0]['order_id'], '12345')
    
    def test_add_to_history_limit(self):
        """测试历史记录数量限制"""
        # 添加超过1000条记录
        for i in range(1100):
            self.executor._add_to_history({'order_id': str(i)})
        
        # 应该只保留最近1000条
        self.assertEqual(len(self.executor.order_history), 1000)
    
    def test_get_order_status(self):
        """测试获取订单状态"""
        result = {'order_id': '12345', 'success': True, 'symbol': 'EURUSD'}
        self.executor._add_to_history(result)
        
        status = self.executor.get_order_status('12345')
        self.assertIsNotNone(status)
        self.assertEqual(status['order_id'], '12345')
    
    def test_get_order_status_not_found(self):
        """测试获取不存在的订单状态"""
        status = self.executor.get_order_status('99999')
        self.assertIsNone(status)
    
    def test_get_slippage_statistics_single_symbol(self):
        """测试单个品种滑点统计"""
        self.executor._record_slippage('EURUSD', 0.0001)
        self.executor._record_slippage('EURUSD', 0.0002)
        self.executor._record_slippage('EURUSD', 0.0003)
        
        stats = self.executor.get_slippage_statistics('EURUSD')
        
        self.assertEqual(stats['symbol'], 'EURUSD')
        self.assertEqual(stats['count'], 3)
        self.assertAlmostEqual(stats['avg'], 0.0002, places=6)
        self.assertEqual(stats['min'], 0.0001)
        self.assertEqual(stats['max'], 0.0003)
    
    def test_get_slippage_statistics_all_symbols(self):
        """测试所有品种滑点统计"""
        self.executor._record_slippage('EURUSD', 0.0001)
        self.executor._record_slippage('GBPUSD', 0.0002)
        
        stats = self.executor.get_slippage_statistics()
        
        self.assertIn('EURUSD', stats)
        self.assertIn('GBPUSD', stats)
        self.assertEqual(stats['EURUSD']['count'], 1)
        self.assertEqual(stats['GBPUSD']['count'], 1)
    
    def test_get_time_factor(self):
        """测试时间因子"""
        factor = self.executor._get_time_factor()
        self.assertGreaterEqual(factor, 1.0)
        self.assertLessEqual(factor, 1.5)
    
    @patch('MetaTrader5.initialize')
    @patch('MetaTrader5.symbol_info')
    def test_calculate_dynamic_slippage(self, mock_symbol_info, mock_init):
        """测试动态滑点计算"""
        mock_init.return_value = True
        
        # 模拟品种信息
        mock_info = Mock()
        mock_info.spread = 10
        mock_info.point = 0.00001
        mock_symbol_info.return_value = mock_info
        
        slippage = self.executor._calculate_dynamic_slippage('EURUSD', 0.1)
        
        self.assertGreater(slippage, 0)
        self.assertLessEqual(slippage, self.executor.max_slippage)
    
    @patch('MetaTrader5.initialize')
    @patch('MetaTrader5.order_send')
    def test_send_order_success(self, mock_order_send, mock_init):
        """测试成功发送订单"""
        mock_init.return_value = True
        
        # 模拟成功的订单结果
        mock_result = Mock()
        mock_result.retcode = mt5.TRADE_RETCODE_DONE
        mock_result.order = 12345
        mock_result.price = 1.1001
        mock_result.volume = 0.1
        mock_result.comment = "Success"
        mock_order_send.return_value = mock_result
        
        result = self.executor.send_order(self.test_signal)
        
        self.assertTrue(result['success'])
        self.assertEqual(result['order_id'], '12345')
    
    @patch('MetaTrader5.initialize')
    def test_send_order_connection_failure(self, mock_init):
        """测试连接失败"""
        mock_init.return_value = False
        
        result = self.executor.send_order(self.test_signal)
        
        self.assertFalse(result['success'])
        self.assertIn('error', result)
    
    def test_send_order_invalid_signal(self):
        """测试无效信号"""
        # 创建一个有效的信号，但修改其属性使其无效
        invalid_signal = Signal(
            strategy_id='test',
            symbol='EURUSD',
            direction=1,
            strength=0.8,
            entry_price=1.1000,
            sl=1.1050,  # 无效止损（高于入场价）
            tp=1.1100,
            size=0.1,
            confidence=0.85,
            timestamp=datetime.now()
        )
        
        result = self.executor.send_order(invalid_signal)
        
        self.assertFalse(result['success'])
        self.assertIn('验证失败', result['error'])


if __name__ == '__main__':
    unittest.main()
