"""
风险控制系统测试
"""

import unittest
from datetime import datetime, timedelta
from src.core.risk_control_system import (
    RiskControlSystem, StopLossConfig, CircuitBreakerConfig,
    RiskLevel, CircuitBreakerStatus
)
from src.core.models import Account, Trade, TradeType


class TestRiskControlSystem(unittest.TestCase):
    """风险控制系统测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.initial_balance = 100000.0
        self.risk_control = RiskControlSystem(self.initial_balance)
        self.account = Account(
            account_id="123456",
            balance=self.initial_balance,
            equity=self.initial_balance,
            margin=0.0,
            free_margin=self.initial_balance,
            margin_level=0.0,
            currency="USD",
            leverage=100
        )
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.risk_control.initial_balance, self.initial_balance)
        self.assertEqual(self.risk_control.current_balance, self.initial_balance)
        self.assertEqual(self.risk_control.peak_balance, self.initial_balance)
        self.assertEqual(self.risk_control.consecutive_losses, 0)
        self.assertFalse(self.risk_control.trading_halted)
        self.assertEqual(self.risk_control.circuit_breaker_status, CircuitBreakerStatus.INACTIVE)
    
    def test_record_losing_trade(self):
        """测试记录亏损交易"""
        losing_trade = Trade(
            trade_id="T001",
            symbol="EURUSD",
            type=TradeType.BUY,
            volume=0.1,
            open_price=1.1000,
            close_price=1.0950,
            profit=-500.0,
            open_time=datetime.now(),
            close_time=datetime.now()
        )
        
        self.risk_control.record_trade_result(losing_trade)
        
        self.assertEqual(self.risk_control.consecutive_losses, 1)
        self.assertEqual(len(self.risk_control.daily_losses), 1)
        self.assertEqual(len(self.risk_control.weekly_losses), 1)
        self.assertEqual(len(self.risk_control.monthly_losses), 1)
    
    def test_record_winning_trade_resets_consecutive_losses(self):
        """测试盈利交易重置连续亏损"""
        # 先记录亏损
        self.risk_control.consecutive_losses = 2
        
        winning_trade = Trade(
            trade_id="T002",
            symbol="EURUSD",
            type=TradeType.BUY,
            volume=0.1,
            open_price=1.1000,
            close_price=1.1050,
            profit=500.0,
            open_time=datetime.now(),
            close_time=datetime.now()
        )
        
        self.risk_control.record_trade_result(winning_trade)
        self.assertEqual(self.risk_control.consecutive_losses, 0)

    
    def test_daily_stop_loss_trigger(self):
        """测试日止损触发"""
        # 创建超过日止损限制的亏损（5%）
        daily_loss = self.initial_balance * 0.06  # 6%
        
        losing_trade = Trade(
            trade_id="T003",
            symbol="EURUSD",
            type=TradeType.BUY,
            volume=1.0,
            open_price=1.1000,
            close_price=1.0940,
            profit=-daily_loss,
            open_time=datetime.now(),
            close_time=datetime.now()
        )
        
        self.risk_control.record_trade_result(losing_trade)
        can_trade, alerts = self.risk_control.check_risk_limits(self.account)
        
        self.assertFalse(can_trade)
        self.assertTrue(self.risk_control.trading_halted)
        self.assertTrue(any(alert.category == "daily" for alert in alerts))
    
    def test_consecutive_losses_trigger(self):
        """测试连续亏损触发"""
        # 记录3笔连续亏损
        for i in range(3):
            losing_trade = Trade(
                trade_id=f"T{i:03d}",
                symbol="EURUSD",
                type=TradeType.BUY,
                volume=0.1,
                open_price=1.1000,
                close_price=1.0950,
                profit=-500.0,
                open_time=datetime.now(),
                close_time=datetime.now()
            )
            self.risk_control.record_trade_result(losing_trade)
        
        can_trade, alerts = self.risk_control.check_risk_limits(self.account)
        
        self.assertFalse(can_trade)
        self.assertTrue(self.risk_control.trading_halted)
        self.assertEqual(self.risk_control.consecutive_losses, 3)
        self.assertTrue(any(alert.category == "consecutive" for alert in alerts))
    
    def test_circuit_breaker_trigger(self):
        """测试熔断触发"""
        # 设置当前余额低于峰值8%以上
        self.risk_control.peak_balance = 100000.0
        self.risk_control.current_balance = 91000.0  # 9%回撤
        
        can_trade, alerts = self.risk_control.check_risk_limits(self.account)
        
        self.assertFalse(can_trade)
        self.assertEqual(self.risk_control.circuit_breaker_status, CircuitBreakerStatus.ACTIVE)
        self.assertTrue(any(alert.category == "circuit_breaker" for alert in alerts))
    
    def test_position_size_multiplier(self):
        """测试仓位大小乘数"""
        # 默认应该是1.0
        self.assertEqual(self.risk_control.get_position_size_multiplier(), 1.0)
        
        # 激活仓位缩减
        self.risk_control._reduce_position_size(0.5)
        self.assertEqual(self.risk_control.get_position_size_multiplier(), 0.5)
        
        # 重置
        self.risk_control.reset_position_reduction()
        self.assertEqual(self.risk_control.get_position_size_multiplier(), 1.0)
    
    def test_force_resume_trading(self):
        """测试强制恢复交易"""
        # 设置暂停状态
        self.risk_control.trading_halted = True
        self.risk_control.circuit_breaker_status = CircuitBreakerStatus.ACTIVE
        
        # 强制恢复
        self.risk_control.force_resume_trading()
        
        self.assertFalse(self.risk_control.trading_halted)
        self.assertEqual(self.risk_control.circuit_breaker_status, CircuitBreakerStatus.INACTIVE)
    
    def test_get_risk_status(self):
        """测试获取风险状态"""
        status = self.risk_control.get_risk_status()
        
        self.assertIn('current_balance', status)
        self.assertIn('peak_balance', status)
        self.assertIn('current_drawdown', status)
        self.assertIn('daily_loss', status)
        self.assertIn('weekly_loss', status)
        self.assertIn('monthly_loss', status)
        self.assertIn('consecutive_losses', status)
        self.assertIn('circuit_breaker', status)
        self.assertIn('trading_halted', status)
    
    def test_weekly_stop_loss_position_reduction(self):
        """测试周止损触发仓位缩减"""
        # 创建超过周止损限制的亏损（10%）
        weekly_loss = self.initial_balance * 0.11  # 11%
        
        losing_trade = Trade(
            trade_id="T004",
            symbol="EURUSD",
            type=TradeType.BUY,
            volume=1.0,
            open_price=1.1000,
            close_price=1.0890,
            profit=-weekly_loss,
            open_time=datetime.now(),
            close_time=datetime.now()
        )
        
        self.risk_control.record_trade_result(losing_trade)
        can_trade, alerts = self.risk_control.check_risk_limits(self.account)
        
        self.assertTrue(self.risk_control.position_reduction_active)
        self.assertEqual(self.risk_control.position_reduction_percentage, 0.5)
        self.assertTrue(any(alert.category == "weekly" for alert in alerts))


if __name__ == '__main__':
    unittest.main()
