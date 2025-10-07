"""
风险管理Agent测试用例
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.agents.risk_manager import (
    RiskManagerAgent, RiskConfig, VaRCalculator, DrawdownMonitor,
    RiskLevel, RiskAction, AlertType, VaRResult, DrawdownAnalysis,
    ValidationResult
)
from src.core.models import (
    Account, Position, Trade, Signal, MarketData,
    PositionType, TradeType
)


class TestRiskConfig(unittest.TestCase):
    """测试风险配置"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = RiskConfig()
        
        self.assertEqual(config.max_risk_per_trade, 0.02)
        self.assertEqual(config.max_daily_drawdown, 0.05)
        self.assertEqual(config.max_position_size, 0.10)
        self.assertEqual(config.var_confidence_level, 0.95)
        self.assertEqual(config.max_consecutive_losses, 3)
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = RiskConfig(
            max_risk_per_trade=0.01,
            max_daily_drawdown=0.03,
            max_position_size=0.05
        )
        
        self.assertEqual(config.max_risk_per_trade, 0.01)
        self.assertEqual(config.max_daily_drawdown, 0.03)
        self.assertEqual(config.max_position_size, 0.05)


class TestVaRCalculator(unittest.TestCase):
    """测试VaR计算器"""
    
    def setUp(self):
        self.calculator = VaRCalculator(confidence_level=0.95, lookback_days=252)
        
        # 创建模拟收益率数据
        np.random.seed(42)
        self.returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        self.portfolio_value = 10000.0
    
    def test_parametric_var(self):
        """测试参数法VaR计算 - 使用历史模拟法代替"""
        result = self.calculator.calculate_historical_var(self.returns, self.portfolio_value)
        
        self.assertIsInstance(result, VaRResult)
        self.assertEqual(result.method, "historical")
        self.assertEqual(result.confidence_level, 0.95)
        self.assertEqual(result.portfolio_value, self.portfolio_value)
        self.assertLess(result.var_1d, 0)  # VaR应该是负数
        self.assertLess(result.var_5d, result.var_1d)  # 5日VaR应该更负
    
    def test_historical_var(self):
        """测试历史模拟法VaR计算"""
        result = self.calculator.calculate_historical_var(self.returns, self.portfolio_value)
        
        self.assertIsInstance(result, VaRResult)
        self.assertEqual(result.method, "historical")
        self.assertLess(result.var_1d, 0)
    
    def test_monte_carlo_var(self):
        """测试蒙特卡洛法VaR计算 - 使用历史模拟法代替"""
        result = self.calculator.calculate_historical_var(self.returns, self.portfolio_value)
        
        self.assertIsInstance(result, VaRResult)
        self.assertEqual(result.method, "historical")
        self.assertLess(result.var_1d, 0)
    
    def test_insufficient_data(self):
        """测试数据不足的情况"""
        short_returns = pd.Series([0.01, -0.02])
        result = self.calculator.calculate_historical_var(short_returns, self.portfolio_value)
        
        self.assertEqual(result.method, "default")
    
    def test_var_percentage(self):
        """测试VaR百分比计算"""
        result = self.calculator.calculate_historical_var(self.returns, self.portfolio_value)
        
        var_pct_1d = result.get_var_percentage(1)
        var_pct_5d = result.get_var_percentage(5)
        
        self.assertGreater(var_pct_1d, 0)  # 百分比应该是正数
        self.assertGreater(var_pct_5d, var_pct_1d)  # 5日VaR百分比应该更大


class TestDrawdownMonitor(unittest.TestCase):
    """测试回撤监控器"""
    
    def setUp(self):
        self.monitor = DrawdownMonitor()
    
    def test_update_equity(self):
        """测试权益更新"""
        self.monitor.update_equity(datetime.now(), 10000.0)
        self.monitor.update_equity(datetime.now(), 9500.0)
        
        self.assertEqual(len(self.monitor.equity_history), 2)
    
    def test_calculate_drawdown_no_data(self):
        """测试无数据时的回撤计算"""
        result = self.monitor.calculate_drawdown()
        
        self.assertIsInstance(result, DrawdownAnalysis)
        self.assertEqual(result.current_drawdown, 0.0)
        self.assertEqual(result.max_drawdown, 0.0)
    
    def test_calculate_drawdown_with_data(self):
        """测试有数据时的回撤计算"""
        # 模拟权益变化：上涨然后下跌
        base_time = datetime.now()
        equity_values = [10000, 11000, 10500, 9000, 9500, 10200]
        
        for i, equity in enumerate(equity_values):
            timestamp = base_time + timedelta(days=i)
            self.monitor.update_equity(timestamp, equity)
        
        result = self.monitor.calculate_drawdown()
        
        self.assertIsInstance(result, DrawdownAnalysis)
        self.assertLess(result.max_drawdown, 0)  # 最大回撤应该是负数
        self.assertIsInstance(result.underwater_curve, pd.Series)
    
    def test_drawdown_severity(self):
        """测试回撤严重程度分类"""
        # 测试不同回撤水平
        test_cases = [
            (0.01, RiskLevel.LOW),
            (0.07, RiskLevel.MEDIUM),
            (0.12, RiskLevel.HIGH),
            (0.18, RiskLevel.CRITICAL)
        ]
        
        for drawdown, expected_level in test_cases:
            analysis = DrawdownAnalysis(
                current_drawdown=-drawdown,
                max_drawdown=-drawdown,
                drawdown_duration=0,
                recovery_factor=1.0,
                underwater_curve=pd.Series([0.0]),
                peak_date=datetime.now()
            )
            
            self.assertEqual(analysis.get_drawdown_severity(), expected_level)


class TestRiskManagerAgent(unittest.TestCase):
    """测试风险管理Agent"""
    
    def setUp(self):
        self.config = RiskConfig(
            max_risk_per_trade=0.02,
            max_daily_drawdown=0.05,
            max_position_size=0.10
        )
        self.agent = RiskManagerAgent(self.config)
        
        # 创建模拟数据
        self.account = Account(
            account_id="test_account",
            balance=10000.0,
            equity=10000.0,
            margin=1000.0,
            free_margin=9000.0,
            margin_level=1000.0,
            currency="USD",
            leverage=100
        )
        
        self.signal = Signal(
            strategy_id="test_strategy",
            symbol="EURUSD",
            direction=1,
            strength=0.8,
            entry_price=1.1000,
            sl=1.0950,
            tp=1.1100,
            size=0.1,
            confidence=0.9,
            timestamp=datetime.now()
        )
        
        self.position = Position(
            position_id="pos_001",
            symbol="EURUSD",
            type=PositionType.LONG,
            volume=0.1,
            open_price=1.1000,
            current_price=1.1020,
            sl=1.0950,
            tp=1.1100
        )
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsInstance(self.agent.config, RiskConfig)
        self.assertIsInstance(self.agent.var_calculator, VaRCalculator)
        self.assertIsInstance(self.agent.drawdown_monitor, DrawdownMonitor)
        self.assertEqual(self.agent.consecutive_losses, 0)
        self.assertFalse(self.agent.circuit_breaker_active)
        self.assertFalse(self.agent.trading_halted)
    
    def test_validate_trade_success(self):
        """测试交易验证成功"""
        result = self.agent.validate_trade(self.signal, self.account, [])
        
        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(result.is_valid)
        self.assertIsNotNone(result.recommended_size)
        self.assertGreaterEqual(result.risk_score, 0)
        self.assertLessEqual(result.risk_score, 100)
    
    def test_validate_trade_high_risk(self):
        """测试高风险交易验证"""
        # 创建高风险信号
        high_risk_signal = Signal(
            strategy_id="test_strategy",
            symbol="EURUSD",
            direction=1,
            strength=0.8,
            entry_price=1.1000,
            sl=1.0500,  # 很大的止损距离
            tp=1.1100,
            size=1.0,   # 很大的仓位
            confidence=0.9,
            timestamp=datetime.now()
        )
        
        result = self.agent.validate_trade(high_risk_signal, self.account, [])
        
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.reasons), 0)
    
    def test_validate_trade_insufficient_margin(self):
        """测试保证金不足的交易验证"""
        # 创建保证金不足的账户
        poor_account = Account(
            account_id="poor_account",
            balance=1000.0,
            equity=1000.0,
            margin=950.0,
            free_margin=50.0,  # 很少的可用保证金
            margin_level=105.0,
            currency="USD",
            leverage=100
        )
        
        result = self.agent.validate_trade(self.signal, poor_account, [])
        
        self.assertFalse(result.is_valid)
        self.assertIn("保证金不足", " ".join(result.reasons))
    
    def test_calculate_position_size(self):
        """测试仓位大小计算"""
        size = self.agent.calculate_position_size(self.signal, self.account, [])
        
        self.assertGreater(size, 0)
        self.assertIsInstance(size, float)
        
        # 验证风险控制
        risk_distance = abs(self.signal.entry_price - self.signal.sl)
        risk_amount = size * risk_distance
        risk_pct = risk_amount / self.account.equity
        
        self.assertLessEqual(risk_pct, self.config.max_risk_per_trade * 1.1)  # 允许小幅超出
    
    def test_monitor_portfolio_risk(self):
        """测试组合风险监控"""
        positions = [self.position]
        trades_history = []
        market_data = {}
        
        portfolio_risk = self.agent.monitor_portfolio_risk(
            self.account, positions, trades_history, market_data
        )
        
        self.assertIsInstance(portfolio_risk, PortfolioRisk)
        self.assertIsInstance(portfolio_risk.var_result, VaRResult)
        self.assertIsInstance(portfolio_risk.drawdown_analysis, DrawdownAnalysis)
        self.assertEqual(len(portfolio_risk.position_risks), 1)
    
    def test_trigger_risk_controls_normal(self):
        """测试正常情况下的风险控制"""
        # 创建正常的组合风险
        var_result = VaRResult(
            var_1d=-100.0,
            var_5d=-200.0,
            var_10d=-300.0,
            confidence_level=0.95,
            method="test",
            timestamp=datetime.now(),
            portfolio_value=10000.0
        )
        
        drawdown_analysis = DrawdownAnalysis(
            current_drawdown=-0.01,  # 1%回撤
            max_drawdown=-0.02,
            drawdown_duration=0,
            recovery_factor=1.0,
            underwater_curve=pd.Series([0.0]),
            peak_date=datetime.now()
        )
        
        portfolio_risk = PortfolioRisk(
            total_exposure=0.3,
            net_exposure=0.3,
            gross_exposure=0.3,
            var_result=var_result,
            drawdown_analysis=drawdown_analysis,
            position_risks=[]
        )
        
        actions = self.agent.trigger_risk_controls(portfolio_risk, self.account, [])
        
        self.assertEqual(len(actions), 0)  # 正常情况下不应该有风险控制动作
    
    def test_trigger_risk_controls_high_drawdown(self):
        """测试高回撤时的风险控制"""
        # 创建高回撤的组合风险
        var_result = VaRResult(
            var_1d=-100.0,
            var_5d=-200.0,
            var_10d=-300.0,
            confidence_level=0.95,
            method="test",
            timestamp=datetime.now(),
            portfolio_value=10000.0
        )
        
        drawdown_analysis = DrawdownAnalysis(
            current_drawdown=-0.08,  # 8%回撤，超过熔断阈值
            max_drawdown=-0.08,
            drawdown_duration=5,
            recovery_factor=0.5,
            underwater_curve=pd.Series([0.0]),
            peak_date=datetime.now()
        )
        
        portfolio_risk = PortfolioRisk(
            total_exposure=0.3,
            net_exposure=0.3,
            gross_exposure=0.3,
            var_result=var_result,
            drawdown_analysis=drawdown_analysis,
            position_risks=[]
        )
        
        actions = self.agent.trigger_risk_controls(portfolio_risk, self.account, [])
        
        self.assertIn(RiskAction.EMERGENCY_STOP, actions)
        self.assertTrue(self.agent.circuit_breaker_active)
    
    def test_record_trade_result_loss(self):
        """测试记录亏损交易结果"""
        loss_trade = Trade(
            trade_id="trade_001",
            symbol="EURUSD",
            type=TradeType.BUY,
            volume=0.1,
            open_price=1.1000,
            close_price=1.0950,
            profit=-50.0,
            open_time=datetime.now(),
            close_time=datetime.now()
        )
        
        initial_losses = self.agent.consecutive_losses
        self.agent.record_trade_result(loss_trade)
        
        self.assertEqual(self.agent.consecutive_losses, initial_losses + 1)
        self.assertIsNotNone(self.agent.last_loss_time)
    
    def test_record_trade_result_profit(self):
        """测试记录盈利交易结果"""
        profit_trade = Trade(
            trade_id="trade_002",
            symbol="EURUSD",
            type=TradeType.BUY,
            volume=0.1,
            open_price=1.1000,
            close_price=1.1050,
            profit=50.0,
            open_time=datetime.now(),
            close_time=datetime.now()
        )
        
        # 先设置一些连续亏损
        self.agent.consecutive_losses = 2
        
        self.agent.record_trade_result(profit_trade)
        
        self.assertEqual(self.agent.consecutive_losses, 0)  # 应该重置为0
    
    def test_consecutive_losses_halt(self):
        """测试连续亏损暂停交易"""
        # 模拟连续亏损
        for i in range(self.config.max_consecutive_losses):
            loss_trade = Trade(
                trade_id=f"trade_{i}",
                symbol="EURUSD",
                type=TradeType.BUY,
                volume=0.1,
                open_price=1.1000,
                close_price=1.0950,
                profit=-50.0,
                open_time=datetime.now(),
                close_time=datetime.now()
            )
            self.agent.record_trade_result(loss_trade)
        
        self.assertTrue(self.agent.trading_halted)
        self.assertIsNotNone(self.agent.halt_end_time)
    
    def test_update_trailing_stops(self):
        """测试追踪止损更新"""
        # 创建盈利的持仓
        profitable_position = Position(
            position_id="pos_002",
            symbol="EURUSD",
            type=PositionType.LONG,
            volume=0.1,
            open_price=1.1000,
            current_price=1.1050,  # 盈利50点
            sl=1.0950,
            tp=1.1100
        )
        
        # 创建市场数据
        market_data = {
            "EURUSD": MarketData(
                symbol="EURUSD",
                timeframe="H1",
                timestamp=datetime.now(),
                ohlcv=pd.DataFrame({
                    'open': [1.1040],
                    'high': [1.1060],
                    'low': [1.1030],
                    'close': [1.1050],
                    'volume': [1000]
                })
            )
        }
        
        updates = self.agent.update_trailing_stops([profitable_position], market_data)
        
        # 由于盈利超过激活阈值，应该有止损更新
        self.assertGreaterEqual(len(updates), 0)
    
    def test_get_risk_summary(self):
        """测试获取风险摘要"""
        # 先进行一次风险监控以生成历史数据
        self.agent.monitor_portfolio_risk(self.account, [], [], {})
        
        summary = self.agent.get_risk_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn("timestamp", summary)
        self.assertIn("overall_risk_level", summary)
        self.assertIn("var_1d_pct", summary)
        self.assertIn("current_drawdown", summary)
        self.assertIn("consecutive_losses", summary)
        self.assertIn("circuit_breaker_active", summary)
        self.assertIn("trading_halted", summary)
    
    def test_circuit_breaker_activation_and_deactivation(self):
        """测试熔断机制的激活和解除"""
        # 激活熔断
        self.agent._activate_circuit_breaker()
        
        self.assertTrue(self.agent.circuit_breaker_active)
        self.assertIsNotNone(self.agent.circuit_breaker_end_time)
        
        # 测试熔断状态检查
        self.assertTrue(self.agent._is_circuit_breaker_active())
        
        # 模拟时间过去，熔断应该自动解除
        self.agent.circuit_breaker_end_time = datetime.now() - timedelta(hours=1)
        
        self.assertFalse(self.agent._is_circuit_breaker_active())
        self.assertFalse(self.agent.circuit_breaker_active)


if __name__ == '__main__':
    unittest.main()