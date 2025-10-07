"""
渐进式风险管理系统演示
展示FundManager和RiskControlSystem的集成使用
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from src.core.fund_manager import FundManager, FundStage
from src.core.risk_control_system import RiskControlSystem, StopLossConfig, CircuitBreakerConfig
from src.core.models import Account, Trade, TradeType


def print_separator():
    print("\n" + "="*80 + "\n")


def demo_fund_manager():
    """演示资金管理器"""
    print("=== 资金管理器演示 ===")
    
    # 初始化资金管理器
    total_capital = 100000.0
    fund_manager = FundManager(total_capital)
    
    print(f"初始资金: ${total_capital:,.2f}")
    print(f"当前阶段: {fund_manager.current_stage.value}")
    print(f"分配资金: ${fund_manager.get_allocated_capital():,.2f}")
    
    # 模拟交易记录
    print("\n模拟20笔交易（满足测试阶段要求）...")
    for i in range(20):
        profit = 100.0 if i % 2 == 0 else -50.0  # 胜率60%
        trade = Trade(
            trade_id=f"T{i:03d}",
            symbol="EURUSD",
            type=TradeType.BUY,
            volume=0.1,
            open_price=1.1000,
            close_price=1.1010 if profit > 0 else 1.0990,
            profit=profit,
            open_time=datetime.now(),
            close_time=datetime.now()
        )
        fund_manager.record_trade(trade)
    
    # 设置满足晋级条件
    fund_manager.current_performance.duration_days = 15
    
    # 检查阶段状态
    status = fund_manager.get_stage_status()
    print(f"\n阶段状态:")
    print(f"  交易次数: {status['total_trades']}")
    print(f"  胜率: {status['win_rate']:.2%}")
    print(f"  盈利因子: {status['profit_factor']:.2f}")
    print(f"  运行天数: {status['duration_days']}")
    print(f"  可以晋级: {status['can_progress']}")
    
    # 尝试晋级
    if fund_manager.progress_to_next_stage():
        print(f"\n✓ 成功晋级到: {fund_manager.current_stage.value}")
        print(f"  新分配资金: ${fund_manager.get_allocated_capital():,.2f}")
    
    print_separator()


def demo_risk_control_system():
    """演示风险控制系统"""
    print("=== 风险控制系统演示 ===")
    
    # 初始化风险控制系统
    initial_balance = 100000.0
    risk_control = RiskControlSystem(initial_balance)
    
    account = Account(
        account_id="123456",
        balance=initial_balance,
        equity=initial_balance,
        margin=0.0,
        free_margin=initial_balance,
        margin_level=0.0,
        currency="USD",
        leverage=100
    )
    
    print(f"初始余额: ${initial_balance:,.2f}")
    
    # 场景1: 正常交易
    print("\n场景1: 正常交易")
    winning_trade = Trade(
        trade_id="T001",
        symbol="EURUSD",
        type=TradeType.BUY,
        volume=0.1,
        open_price=1.1000,
        close_price=1.1050,
        profit=500.0,
        open_time=datetime.now(),
        close_time=datetime.now()
    )
    risk_control.record_trade_result(winning_trade)
    can_trade, alerts = risk_control.check_risk_limits(account)
    print(f"  可以交易: {can_trade}")
    print(f"  告警数量: {len(alerts)}")
    
    # 场景2: 连续亏损
    print("\n场景2: 连续亏损触发暂停")
    for i in range(3):
        losing_trade = Trade(
            trade_id=f"T{i+2:03d}",
            symbol="EURUSD",
            type=TradeType.BUY,
            volume=0.1,
            open_price=1.1000,
            close_price=1.0950,
            profit=-500.0,
            open_time=datetime.now(),
            close_time=datetime.now()
        )
        risk_control.record_trade_result(losing_trade)
    
    can_trade, alerts = risk_control.check_risk_limits(account)
    print(f"  连续亏损: {risk_control.consecutive_losses}次")
    print(f"  可以交易: {can_trade}")
    print(f"  交易暂停: {risk_control.trading_halted}")
    if alerts:
        print(f"  告警: {alerts[0].message}")
    
    # 场景3: 熔断触发
    print("\n场景3: 回撤触发熔断")
    risk_control2 = RiskControlSystem(initial_balance)
    risk_control2.peak_balance = 100000.0
    risk_control2.current_balance = 91000.0  # 9%回撤
    
    can_trade, alerts = risk_control2.check_risk_limits(account)
    print(f"  当前回撤: {abs(risk_control2._calculate_current_drawdown()):.2%}")
    print(f"  熔断状态: {risk_control2.circuit_breaker_status.value}")
    print(f"  可以交易: {can_trade}")
    
    # 获取风险状态
    print("\n风险状态摘要:")
    status = risk_control.get_risk_status()
    print(f"  当前余额: ${status['current_balance']:,.2f}")
    print(f"  峰值余额: ${status['peak_balance']:,.2f}")
    print(f"  日亏损使用率: {status['daily_loss']['utilization']:.2%}")
    print(f"  仓位乘数: {status['position_size_multiplier']:.2f}")
    
    print_separator()


def demo_integrated_system():
    """演示集成系统"""
    print("=== 集成系统演示 ===")
    
    # 初始化两个系统
    total_capital = 100000.0
    fund_manager = FundManager(total_capital)
    risk_control = RiskControlSystem(total_capital)
    
    print(f"总资金: ${total_capital:,.2f}")
    print(f"资金阶段: {fund_manager.current_stage.value}")
    print(f"分配资金: ${fund_manager.get_allocated_capital():,.2f}")
    
    # 模拟交易流程
    print("\n模拟交易流程:")
    
    account = Account(
        account_id="123456",
        balance=total_capital,
        equity=total_capital,
        margin=0.0,
        free_margin=total_capital,
        margin_level=0.0,
        currency="USD",
        leverage=100
    )
    
    # 检查风险限制
    can_trade, alerts = risk_control.check_risk_limits(account)
    print(f"1. 检查风险限制: {'通过' if can_trade else '未通过'}")
    
    if can_trade:
        # 计算仓位大小
        allocated_capital = fund_manager.get_allocated_capital()
        position_size_multiplier = risk_control.get_position_size_multiplier()
        max_position = fund_manager.get_max_position_size() * position_size_multiplier
        
        print(f"2. 分配资金: ${allocated_capital:,.2f}")
        print(f"3. 仓位乘数: {position_size_multiplier:.2f}")
        print(f"4. 最大仓位: ${max_position:,.2f}")
        
        # 执行交易
        trade = Trade(
            trade_id="T001",
            symbol="EURUSD",
            type=TradeType.BUY,
            volume=0.1,
            open_price=1.1000,
            close_price=1.1050,
            profit=500.0,
            open_time=datetime.now(),
            close_time=datetime.now()
        )
        
        # 记录到两个系统
        fund_manager.record_trade(trade)
        risk_control.record_trade_result(trade)
        
        print(f"5. 交易执行: {trade.trade_id}, 盈亏: ${trade.profit:,.2f}")
        
        # 更新余额
        new_balance = total_capital + trade.profit
        risk_control.update_balance(new_balance)
        
        print(f"6. 更新余额: ${new_balance:,.2f}")
    
    # 显示综合状态
    print("\n综合状态:")
    fund_status = fund_manager.get_stage_status()
    risk_status = risk_control.get_risk_status()
    
    print(f"  资金阶段: {fund_status['current_stage']}")
    print(f"  阶段交易: {fund_status['total_trades']}")
    print(f"  阶段胜率: {fund_status['win_rate']:.2%}")
    print(f"  连续亏损: {risk_status['consecutive_losses']}")
    print(f"  交易状态: {'正常' if not risk_status['trading_halted'] else '暂停'}")
    
    print_separator()


if __name__ == "__main__":
    print("\n渐进式风险管理系统演示\n")
    
    demo_fund_manager()
    demo_risk_control_system()
    demo_integrated_system()
    
    print("演示完成！")
