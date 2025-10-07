#!/usr/bin/env python3
"""
风险管理Agent演示脚本
展示RiskManagerAgent的核心功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from src.agents import RiskManagerAgent, RiskConfig
from src.core.models import Account, Signal, Position, Trade, PositionType, TradeType


def main():
    """演示风险管理Agent功能"""
    print("=== 风险管理Agent演示 ===\n")
    
    # 1. 创建风险管理配置
    print("1. 创建风险管理配置")
    config = RiskConfig(
        max_risk_per_trade=0.02,  # 单笔最大风险2%
        max_daily_drawdown=0.05,  # 日最大回撤5%
        max_position_size=0.10,   # 单个品种最大仓位10%
        max_consecutive_losses=3  # 最大连续亏损3次
    )
    print(f"配置: 单笔风险{config.max_risk_per_trade:.1%}, 日回撤{config.max_daily_drawdown:.1%}")
    
    # 2. 初始化风险管理Agent
    print("\n2. 初始化风险管理Agent")
    risk_manager = RiskManagerAgent(config)
    print("风险管理Agent初始化完成")
    
    # 3. 创建模拟账户
    print("\n3. 创建模拟账户")
    account = Account(
        account_id="demo_account",
        balance=10000.0,
        equity=10000.0,
        margin=1000.0,
        free_margin=9000.0,
        margin_level=1000.0,
        currency="USD",
        leverage=100
    )
    print(f"账户: 余额${account.balance:,.2f}, 权益${account.equity:,.2f}, 保证金水平{account.margin_level:.1f}%")
    
    # 4. 创建交易信号
    print("\n4. 创建交易信号")
    signal = Signal(
        strategy_id="demo_strategy",
        symbol="EURUSD",
        direction=1,  # 买入
        strength=0.8,
        entry_price=1.1000,
        sl=1.0950,    # 50点止损
        tp=1.1100,    # 100点止盈
        size=0.1,     # 0.1手
        confidence=0.9,
        timestamp=datetime.now()
    )
    print(f"信号: {signal.symbol} {'买入' if signal.direction > 0 else '卖出'} {signal.size}手")
    print(f"入场价: {signal.entry_price}, 止损: {signal.sl}, 止盈: {signal.tp}")
    
    # 5. 验证交易信号
    print("\n5. 验证交易信号")
    validation_result = risk_manager.validate_trade(signal, account, [])
    
    print(f"验证结果: {'通过' if validation_result.is_valid else '拒绝'}")
    print(f"风险评分: {validation_result.risk_score:.1f}/100")
    
    if validation_result.recommended_size:
        print(f"推荐仓位: {validation_result.recommended_size:.3f}手")
    
    if validation_result.reasons:
        print("拒绝原因:")
        for reason in validation_result.reasons:
            print(f"  - {reason}")
    
    if validation_result.warnings:
        print("警告:")
        for warning in validation_result.warnings:
            print(f"  - {warning}")
    
    # 6. 计算仓位大小
    print("\n6. 计算仓位大小")
    position_size = risk_manager.calculate_position_size(signal, account, [])
    print(f"计算的仓位大小: {position_size:.3f}手")
    
    # 计算风险金额
    risk_distance = abs(signal.entry_price - signal.sl)
    risk_amount = position_size * risk_distance
    risk_percentage = risk_amount / account.equity
    print(f"风险距离: {risk_distance:.5f}")
    print(f"风险金额: ${risk_amount:.2f}")
    print(f"风险百分比: {risk_percentage:.2%}")
    
    # 7. 模拟连续亏损
    print("\n7. 模拟连续亏损场景")
    print("模拟3笔连续亏损交易...")
    
    for i in range(3):
        loss_trade = Trade(
            trade_id=f"loss_trade_{i+1}",
            symbol="EURUSD",
            type=TradeType.BUY,
            volume=0.1,
            open_price=1.1000,
            close_price=1.0950,
            profit=-50.0,  # 亏损$50
            open_time=datetime.now(),
            close_time=datetime.now()
        )
        
        risk_manager.record_trade_result(loss_trade)
        print(f"记录第{i+1}笔亏损交易, 连续亏损次数: {risk_manager.consecutive_losses}")
    
    # 8. 检查交易状态
    print("\n8. 检查交易状态")
    print(f"交易是否暂停: {'是' if risk_manager.trading_halted else '否'}")
    print(f"熔断是否激活: {'是' if risk_manager.circuit_breaker_active else '否'}")
    
    # 9. 获取风险摘要
    print("\n9. 获取风险摘要")
    risk_summary = risk_manager.get_risk_summary()
    print("风险摘要:")
    for key, value in risk_summary.items():
        if key == "timestamp":
            print(f"  {key}: {value.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print(f"  {key}: {value}")
    
    # 10. 测试盈利交易重置连续亏损
    print("\n10. 测试盈利交易重置连续亏损")
    profit_trade = Trade(
        trade_id="profit_trade_1",
        symbol="EURUSD",
        type=TradeType.BUY,
        volume=0.1,
        open_price=1.1000,
        close_price=1.1050,
        profit=50.0,  # 盈利$50
        open_time=datetime.now(),
        close_time=datetime.now()
    )
    
    risk_manager.record_trade_result(profit_trade)
    print(f"记录盈利交易后, 连续亏损次数: {risk_manager.consecutive_losses}")
    print(f"交易是否暂停: {'是' if risk_manager.trading_halted else '否'}")
    
    print("\n=== 演示完成 ===")


if __name__ == "__main__":
    main()