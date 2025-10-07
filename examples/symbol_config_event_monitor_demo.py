"""
品种配置和经济事件监控演示
展示如何使用SymbolConfigManager和EconomicEventMonitor
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.symbol_config_manager import SymbolConfigManager
from src.core.economic_event_monitor import (
    EconomicEventMonitor,
    EconomicEvent,
    EIAInventoryEvent,
    OPECMeeting,
    EventType,
    EventImpact
)


def demo_symbol_config_manager():
    """演示品种配置管理器功能"""
    print("=" * 80)
    print("品种配置管理器演示")
    print("=" * 80)
    
    # 初始化管理器
    manager = SymbolConfigManager("config/symbols")
    
    # 1. 获取所有配置的品种
    print("\n1. 所有配置的品种:")
    symbols = manager.get_all_symbols()
    for symbol in symbols:
        print(f"   - {symbol}")
    
    # 2. 获取特定品种的配置
    print("\n2. EURUSD配置详情:")
    eurusd_config = manager.get_config("EURUSD")
    if eurusd_config:
        print(f"   品种: {eurusd_config.symbol}")
        print(f"   点差限制: {eurusd_config.spread_limit}")
        print(f"   最小手数: {eurusd_config.min_lot}")
        print(f"   最大手数: {eurusd_config.max_lot}")
        print(f"   风险倍数: {eurusd_config.risk_multiplier}")
        print(f"   启用策略: {', '.join(eurusd_config.strategies)}")
    
    # 3. 检查交易时间
    print("\n3. 检查当前交易时间:")
    current_time = datetime.utcnow()
    print(f"   当前UTC时间: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    active_symbols = manager.get_active_symbols(current_time)
    print(f"   当前可交易品种: {', '.join(active_symbols)}")
    
    # 4. 验证手数
    print("\n4. 手数验证:")
    test_lots = [0.005, 0.1, 0.123, 5.0, 15.0]
    for lot in test_lots:
        valid, adjusted = manager.validate_lot_size("EURUSD", lot)
        print(f"   请求手数 {lot:.3f} -> 调整后 {adjusted:.2f} (有效: {valid})")
    
    # 5. 检查点差
    print("\n5. 点差检查:")
    test_spreads = [1.0, 2.0, 3.5]
    for spread in test_spreads:
        is_valid = manager.check_spread("EURUSD", spread)
        status = "✓ 可接受" if is_valid else "✗ 超出限制"
        print(f"   点差 {spread} -> {status}")
    
    # 6. 品种特殊配置
    print("\n6. 品种特殊配置:")
    
    # XAUUSD - 黄金
    xauusd_config = manager.get_config("XAUUSD")
    if xauusd_config:
        print(f"\n   XAUUSD (黄金):")
        print(f"   - 风险倍数: {xauusd_config.risk_multiplier} (降低风险)")
        print(f"   - 止损点数: {xauusd_config.risk_params.stop_loss_pips}")
        if xauusd_config.event_monitoring:
            print(f"   - 监控事件: {', '.join(xauusd_config.event_monitoring.events)}")
    
    # USOIL - 原油
    usoil_config = manager.get_config("USOIL")
    if usoil_config:
        print(f"\n   USOIL (原油):")
        print(f"   - 风险倍数: {usoil_config.risk_multiplier} (最低风险)")
        print(f"   - 止损点数: {usoil_config.risk_params.stop_loss_pips}")
        if usoil_config.eia_monitoring:
            print(f"   - EIA监控: 启用")
            print(f"   - 发布时间: {usoil_config.eia_monitoring.release_time} UTC")
        if usoil_config.opec_monitoring:
            print(f"   - OPEC监控: 启用")


def demo_economic_event_monitor():
    """演示经济事件监控器功能"""
    print("\n" + "=" * 80)
    print("经济事件监控器演示")
    print("=" * 80)
    
    # 初始化监控器
    monitor = EconomicEventMonitor("config/economic_calendar.json")
    
    # 1. 添加测试事件
    print("\n1. 添加即将发生的测试事件:")
    
    # 添加一个2小时后的NFP事件
    nfp_event = EconomicEvent(
        event_id="test_nfp",
        name="非农就业数据 (NFP)",
        country="US",
        event_type=EventType.EMPLOYMENT,
        impact=EventImpact.CRITICAL,
        scheduled_time=datetime.utcnow() + timedelta(hours=2),
        forecast_value=180000.0,
        previous_value=175000.0,
        affected_symbols=["EURUSD", "XAUUSD", "GBPUSD"],
        description="美国非农就业人数变化"
    )
    monitor.calendar.add_event(nfp_event)
    print(f"   ✓ 添加事件: {nfp_event.name}")
    print(f"     时间: {nfp_event.scheduled_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"     影响: {nfp_event.impact.value}")
    print(f"     影响品种: {', '.join(nfp_event.affected_symbols)}")
    
    # 2. 获取即将发生的事件
    print("\n2. 即将发生的重要事件 (未来24小时):")
    upcoming_events = monitor.calendar.get_upcoming_events(
        hours_ahead=24,
        min_impact=EventImpact.MEDIUM
    )
    
    if upcoming_events:
        for event in upcoming_events:
            time_diff = (event.scheduled_time - datetime.utcnow()).total_seconds() / 3600
            print(f"\n   事件: {event.name}")
            print(f"   时间: {event.scheduled_time.strftime('%Y-%m-%d %H:%M:%S')} UTC ({time_diff:.1f}小时后)")
            print(f"   影响级别: {event.impact.value}")
            print(f"   影响品种: {', '.join(event.affected_symbols)}")
    else:
        print("   暂无即将发生的重要事件")
    
    # 3. 检查交易限制
    print("\n3. 检查各品种的交易限制:")
    test_symbols = ["EURUSD", "XAUUSD", "USOIL"]
    
    for symbol in test_symbols:
        should_restrict, reason = monitor.check_trading_restrictions(symbol)
        if should_restrict:
            print(f"   {symbol}: ⚠️  限制交易 - {reason}")
        else:
            print(f"   {symbol}: ✓ 可以交易")
    
    # 4. 获取事件驱动的策略调整
    print("\n4. 事件驱动的策略调整建议:")
    
    for symbol in ["EURUSD", "XAUUSD"]:
        adjustments = monitor.get_event_driven_adjustments(symbol)
        print(f"\n   {symbol}:")
        print(f"   - 减少仓位: {'是' if adjustments['reduce_position'] else '否'}")
        print(f"   - 增加止损: {'是' if adjustments['increase_stop_loss'] else '否'}")
        print(f"   - 避免新交易: {'是' if adjustments['avoid_new_trades'] else '否'}")
        print(f"   - 风险倍数: {adjustments['risk_multiplier']}")
        if adjustments['reasons']:
            print(f"   - 原因: {'; '.join(adjustments['reasons'])}")
    
    # 5. EIA库存监控
    print("\n5. EIA库存数据监控:")
    
    next_release = monitor.eia_monitor.get_next_release_time()
    print(f"   下次发布时间: {next_release.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
    is_near, release_time = monitor.eia_monitor.is_near_release(
        minutes_before=30,
        minutes_after=15
    )
    
    if is_near:
        print(f"   ⚠️  临近EIA发布时间，建议暂停USOIL交易")
    else:
        time_diff = (next_release - datetime.utcnow()).total_seconds() / 3600
        print(f"   ✓ 距离下次发布还有 {time_diff:.1f} 小时")
    
    # 添加历史库存数据
    print("\n   添加历史库存数据...")
    for i in range(4):
        inventory_event = EIAInventoryEvent(
            release_date=datetime.utcnow() - timedelta(weeks=i),
            crude_oil_change=1.5 - i * 0.5,  # 递减趋势
            forecast_crude=1.0,
            previous_crude=0.5
        )
        monitor.eia_monitor.add_inventory_data(inventory_event)
    
    # 分析趋势
    trend = monitor.eia_monitor.analyze_trend(weeks=4)
    if trend:
        print(f"\n   库存趋势分析 (最近4周):")
        print(f"   - 平均变化: {trend['average_change']:.2f} 百万桶")
        print(f"   - 总变化: {trend['total_change']:.2f} 百万桶")
        print(f"   - 趋势: {trend['trend']} ({'库存增加' if trend['trend'] == 'building' else '库存减少'})")
    
    # 6. OPEC会议监控
    print("\n6. OPEC会议监控:")
    
    next_meeting = monitor.opec_monitor.get_next_meeting()
    if next_meeting:
        print(f"   下次会议时间: {next_meeting.meeting_date.strftime('%Y-%m-%d')}")
        print(f"   会议类型: {next_meeting.meeting_type}")
        
        is_near, meeting = monitor.opec_monitor.is_near_meeting(
            hours_before=2,
            hours_after=1
        )
        
        if is_near:
            print(f"   ⚠️  临近OPEC会议，建议暂停USOIL交易")
        else:
            days_diff = (next_meeting.meeting_date - datetime.utcnow()).days
            print(f"   ✓ 距离下次会议还有 {days_diff} 天")
    else:
        print("   暂无已计划的OPEC会议")
    
    # 7. 市场情绪分析
    print("\n7. 基于事件的市场情绪分析:")
    sentiment = monitor.get_market_sentiment_from_events()
    
    if sentiment:
        for symbol, mood in sentiment.items():
            mood_cn = "看涨" if mood == "bullish" else "看跌"
            print(f"   {symbol}: {mood_cn} ({mood})")
    else:
        print("   暂无明确的市场情绪信号")


def demo_integrated_workflow():
    """演示集成工作流程"""
    print("\n" + "=" * 80)
    print("集成工作流程演示")
    print("=" * 80)
    
    # 初始化两个管理器
    symbol_manager = SymbolConfigManager("config/symbols")
    event_monitor = EconomicEventMonitor("config/economic_calendar.json")
    
    print("\n模拟交易决策流程:")
    print("-" * 80)
    
    # 要交易的品种
    symbol = "EURUSD"
    requested_lot = 0.5
    current_spread = 1.8
    
    print(f"\n交易请求:")
    print(f"  品种: {symbol}")
    print(f"  请求手数: {requested_lot}")
    print(f"  当前点差: {current_spread}")
    
    # 步骤1: 检查品种是否已配置
    print(f"\n步骤1: 检查品种配置...")
    if not symbol_manager.is_symbol_configured(symbol):
        print(f"  ✗ 品种 {symbol} 未配置，拒绝交易")
        return
    print(f"  ✓ 品种已配置")
    
    # 步骤2: 检查交易时间
    print(f"\n步骤2: 检查交易时间...")
    config = symbol_manager.get_config(symbol)
    if not config.is_trading_time():
        print(f"  ✗ 当前不在交易时间内，拒绝交易")
        return
    print(f"  ✓ 在交易时间内")
    
    # 步骤3: 检查点差
    print(f"\n步骤3: 检查点差...")
    if not symbol_manager.check_spread(symbol, current_spread):
        print(f"  ✗ 点差 {current_spread} 超过限制 {config.spread_limit}，拒绝交易")
        return
    print(f"  ✓ 点差可接受")
    
    # 步骤4: 验证手数
    print(f"\n步骤4: 验证手数...")
    valid, adjusted_lot = symbol_manager.validate_lot_size(symbol, requested_lot)
    if not valid:
        print(f"  ✗ 手数验证失败")
        return
    print(f"  ✓ 手数有效，调整为: {adjusted_lot}")
    
    # 步骤5: 检查经济事件限制
    print(f"\n步骤5: 检查经济事件...")
    should_restrict, reason = event_monitor.check_trading_restrictions(symbol)
    if should_restrict:
        print(f"  ⚠️  事件限制: {reason}")
        print(f"  建议: 暂停交易或减少仓位")
    else:
        print(f"  ✓ 无事件限制")
    
    # 步骤6: 获取风险调整建议
    print(f"\n步骤6: 获取风险调整...")
    adjustments = event_monitor.get_event_driven_adjustments(symbol)
    risk_multiplier = config.risk_multiplier * adjustments['risk_multiplier']
    
    print(f"  基础风险倍数: {config.risk_multiplier}")
    print(f"  事件调整倍数: {adjustments['risk_multiplier']}")
    print(f"  最终风险倍数: {risk_multiplier}")
    
    if adjustments['reduce_position']:
        print(f"  ⚠️  建议减少仓位")
    if adjustments['increase_stop_loss']:
        print(f"  ⚠️  建议增加止损距离")
    
    # 步骤7: 最终决策
    print(f"\n步骤7: 最终交易决策...")
    final_lot = adjusted_lot * risk_multiplier
    
    print(f"\n最终交易参数:")
    print(f"  品种: {symbol}")
    print(f"  手数: {final_lot:.2f}")
    print(f"  止损点数: {config.risk_params.stop_loss_pips}")
    print(f"  止盈点数: {config.risk_params.take_profit_pips}")
    
    if should_restrict:
        print(f"\n⚠️  由于事件限制，建议谨慎交易或等待事件结束")
    else:
        print(f"\n✓ 所有检查通过，可以执行交易")


def main():
    """主函数"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "品种配置和经济事件监控系统演示" + " " * 20 + "║")
    print("╚" + "=" * 78 + "╝")
    
    try:
        # 演示1: 品种配置管理器
        demo_symbol_config_manager()
        
        # 演示2: 经济事件监控器
        demo_economic_event_monitor()
        
        # 演示3: 集成工作流程
        demo_integrated_workflow()
        
        print("\n" + "=" * 80)
        print("演示完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
