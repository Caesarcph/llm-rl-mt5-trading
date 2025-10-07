"""
资金管理器测试
"""

import unittest
from datetime import datetime, timedelta
from src.core.fund_manager import FundManager, FundStage, StageConfig, StagePerformance
from src.core.models import Account, Trade, TradeType


class TestFundManager(unittest.TestCase):
    """资金管理器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.total_capital = 100000.0
        self.fund_manager = FundManager(self.total_capital)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.fund_manager.total_capital, self.total_capital)
        self.assertEqual(self.fund_manager.reserved_capital, self.total_capital * 0.5)
        self.assertEqual(self.fund_manager.available_capital, self.total_capital * 0.5)
        self.assertEqual(self.fund_manager.current_stage, FundStage.TESTING)
    
    def test_stage_configs(self):
        """测试阶段配置"""
        configs = self.fund_manager.stage_configs
        self.assertIn(FundStage.TESTING, configs)
        self.assertIn(FundStage.STABLE, configs)
        self.assertIn(FundStage.FULL, configs)
        
        # 测试阶段配置
        testing_config = configs[FundStage.TESTING]
        self.assertEqual(testing_config.allocation_percentage, 0.10)
        self.assertEqual(testing_config.min_trades, 20)
        self.assertEqual(testing_config.min_win_rate, 0.50)
    
    def test_get_allocated_capital(self):
        """测试获取分配资金"""
        # 测试阶段: 10%
        allocated = self.fund_manager.get_allocated_capital()
        expected = self.total_capital * 0.5 * 0.10  # 50% available * 10% allocation
        self.assertEqual(allocated, expected)
    
    def test_get_max_position_size(self):
        """测试获取最大仓位"""
        max_position = self.fund_manager.get_max_position_size(risk_per_trade=0.02)
        allocated = self.fund_manager.get_allocated_capital()
        expected = allocated * 0.02
        self.assertEqual(max_position, expected)

    
    def test_record_trade(self):
        """测试记录交易"""
        # 创建盈利交易
        winning_trade = Trade(
            trade_id="T001",
            symbol="EURUSD",
            type=TradeType.BUY,
            volume=0.1,
            open_price=1.1000,
            close_price=1.1050,
            profit=50.0,
            open_time=datetime.now(),
            close_time=datetime.now()
        )
        
        self.fund_manager.record_trade(winning_trade)
        self.assertEqual(self.fund_manager.current_performance.total_trades, 1)
        self.assertEqual(self.fund_manager.current_performance.winning_trades, 1)
        self.assertEqual(self.fund_manager.current_performance.total_profit, 50.0)
        
        # 创建亏损交易
        losing_trade = Trade(
            trade_id="T002",
            symbol="EURUSD",
            type=TradeType.SELL,
            volume=0.1,
            open_price=1.1000,
            close_price=1.1030,
            profit=-30.0,
            open_time=datetime.now(),
            close_time=datetime.now()
        )
        
        self.fund_manager.record_trade(losing_trade)
        self.assertEqual(self.fund_manager.current_performance.total_trades, 2)
        self.assertEqual(self.fund_manager.current_performance.losing_trades, 1)
        self.assertEqual(self.fund_manager.current_performance.total_loss, 30.0)
    
    def test_update_drawdown(self):
        """测试更新回撤"""
        self.fund_manager.update_drawdown(-0.05)
        self.assertEqual(self.fund_manager.current_performance.max_drawdown, -0.05)
        
        # 更大的回撤
        self.fund_manager.update_drawdown(-0.08)
        self.assertEqual(self.fund_manager.current_performance.max_drawdown, -0.08)
        
        # 较小的回撤不应更新
        self.fund_manager.update_drawdown(-0.03)
        self.assertEqual(self.fund_manager.current_performance.max_drawdown, -0.08)
    
    def test_stage_progression_insufficient_trades(self):
        """测试交易次数不足时无法晋级"""
        # 只记录10笔交易（需要20笔）
        for i in range(10):
            trade = Trade(
                trade_id=f"T{i:03d}",
                symbol="EURUSD",
                type=TradeType.BUY,
                volume=0.1,
                open_price=1.1000,
                close_price=1.1050,
                profit=50.0,
                open_time=datetime.now(),
                close_time=datetime.now()
            )
            self.fund_manager.record_trade(trade)
        
        can_progress, next_stage, reasons = self.fund_manager.evaluate_stage_progression()
        self.assertFalse(can_progress)
        self.assertIsNone(next_stage)
        self.assertTrue(any("交易次数不足" in r for r in reasons))

    
    def test_stage_progression_success(self):
        """测试成功晋级"""
        # 模拟满足晋级条件
        self.fund_manager.current_performance.total_trades = 25
        self.fund_manager.current_performance.winning_trades = 15
        self.fund_manager.current_performance.losing_trades = 10
        self.fund_manager.current_performance.total_profit = 1500.0
        self.fund_manager.current_performance.total_loss = 1000.0
        self.fund_manager.current_performance.max_drawdown = -0.08
        self.fund_manager.current_performance.duration_days = 15
        
        # 评估晋级
        can_progress, next_stage, reasons = self.fund_manager.evaluate_stage_progression()
        self.assertTrue(can_progress)
        self.assertEqual(next_stage, FundStage.STABLE)
        self.assertEqual(len(reasons), 0)
        
        # 执行晋级
        result = self.fund_manager.progress_to_next_stage()
        self.assertTrue(result)
        self.assertEqual(self.fund_manager.current_stage, FundStage.STABLE)
    
    def test_demote_stage(self):
        """测试降级"""
        # 先晋级到STABLE
        self.fund_manager.current_stage = FundStage.STABLE
        
        # 执行降级
        result = self.fund_manager.demote_stage("测试降级")
        self.assertTrue(result)
        self.assertEqual(self.fund_manager.current_stage, FundStage.TESTING)
    
    def test_auto_adjust_allocation_demote(self):
        """测试自动调整-降级"""
        # 设置为STABLE阶段
        self.fund_manager.current_stage = FundStage.STABLE
        
        # 创建账户
        account = Account(
            account_id="123456",
            balance=100000.0,
            equity=90000.0,
            margin=5000.0,
            free_margin=85000.0,
            margin_level=1800.0,
            currency="USD",
            leverage=100
        )
        
        # 回撤超过阈值（STABLE阈值是12%）
        adjusted = self.fund_manager.auto_adjust_allocation(account, -0.15)
        self.assertTrue(adjusted)
        self.assertEqual(self.fund_manager.current_stage, FundStage.TESTING)
    
    def test_get_stage_status(self):
        """测试获取阶段状态"""
        status = self.fund_manager.get_stage_status()
        
        self.assertEqual(status['current_stage'], FundStage.TESTING.value)
        self.assertIn('allocated_capital', status)
        self.assertIn('allocation_percentage', status)
        self.assertIn('win_rate', status)
        self.assertIn('profit_factor', status)
        self.assertIn('can_progress', status)
    
    def test_get_risk_assessment(self):
        """测试获取风险评估"""
        assessment = self.fund_manager.get_risk_assessment()
        
        self.assertEqual(assessment['total_capital'], self.total_capital)
        self.assertIn('allocated_capital', assessment)
        self.assertIn('reserved_capital', assessment)
        self.assertIn('allocation_ratio', assessment)
        self.assertIn('reserve_ratio', assessment)
        self.assertIn('stage_performance', assessment)


if __name__ == '__main__':
    unittest.main()
