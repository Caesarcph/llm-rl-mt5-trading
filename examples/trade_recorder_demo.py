"""
交易记录系统演示
展示如何使用TradeRecorder进行交易记录、分析和报告生成
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
from src.core.trade_recorder import TradeRecorder, TradeRecord, TradeStatus
from src.core.models import Position, PositionType
from src.data.database import DatabaseManager


def print_section(title: str):
    """打印分节标题"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def demo_basic_recording():
    """演示基本的交易记录功能"""
    print_section("1. 基本交易记录功能")
    
    # 创建数据库管理器和交易记录器
    db_manager = DatabaseManager("data/demo_trades.db")
    recorder = TradeRecorder(db_manager, config={'log_dir': 'logs'})
    
    print("\n✓ 交易记录器初始化完成")
    
    # 模拟开仓
    position = Position(
        position_id="DEMO_001",
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
        comment="Demo trade",
        magic_number=123
    )
    
    # 记录开仓
    trade_record = recorder.record_trade_open(position, "demo_strategy")
    print(f"\n✓ 记录开仓: {trade_record.trade_id}")
    print(f"  品种: {trade_record.symbol}")
    print(f"  方向: {trade_record.trade_type}")
    print(f"  手数: {trade_record.volume}")
    print(f"  开仓价: {trade_record.open_price}")
    
    # 模拟价格变动
    position.current_price = 1.1050
    position.profit = 50.0
    
    # 记录平仓
    close_record = recorder.record_trade_close(
        position=position,
        close_price=1.1050,
        profit=50.0,
        commission=2.0,
        swap=1.0
    )
    
    print(f"\n✓ 记录平仓: {close_record.trade_id}")
    print(f"  平仓价: {close_record.close_price}")
    print(f"  盈亏: ${close_record.profit:.2f}")
    print(f"  手续费: ${close_record.commission:.2f}")
    print(f"  净盈亏: ${close_record.profit - close_record.commission - close_record.swap:.2f}")
    print(f"  持仓时长: {close_record.duration_hours:.2f} 小时")
    print(f"  交易结果: {close_record.outcome.value}")
    
    return recorder


def demo_multiple_trades(recorder: TradeRecorder):
    """演示记录多笔交易"""
    print_section("2. 记录多笔交易")
    
    # 模拟多笔交易
    trades_data = [
        {"id": "DEMO_002", "symbol": "GBPUSD", "type": PositionType.LONG, "profit": 30.0},
        {"id": "DEMO_003", "symbol": "EURUSD", "type": PositionType.SHORT, "profit": -20.0},
        {"id": "DEMO_004", "symbol": "XAUUSD", "type": PositionType.LONG, "profit": 100.0},
        {"id": "DEMO_005", "symbol": "EURUSD", "type": PositionType.LONG, "profit": -15.0},
    ]
    
    for trade_data in trades_data:
        position = Position(
            position_id=trade_data["id"],
            symbol=trade_data["symbol"],
            type=trade_data["type"],
            volume=0.1,
            open_price=1.1000,
            current_price=1.1000,
            sl=1.0950,
            tp=1.1100,
            profit=0.0,
            swap=0.0,
            commission=0.0,
            open_time=datetime.now() - timedelta(hours=2),
            comment="Demo trade",
            magic_number=123
        )
        
        # 开仓
        recorder.record_trade_open(position, "demo_strategy")
        
        # 平仓
        position.profit = trade_data["profit"]
        recorder.record_trade_close(
            position=position,
            close_price=1.1050,
            profit=trade_data["profit"],
            commission=2.0,
            swap=0.5
        )
        
        print(f"✓ 记录交易: {trade_data['id']} - {trade_data['symbol']} - ${trade_data['profit']:.2f}")


def demo_statistics(recorder: TradeRecorder):
    """演示交易统计功能"""
    print_section("3. 交易统计分析")
    
    # 获取所有交易统计
    stats = recorder.get_trade_statistics()
    
    print("\n📊 整体统计:")
    print(f"  总交易数: {stats['closed_trades']}")
    print(f"  盈利交易: {stats['winning_trades']}")
    print(f"  亏损交易: {stats['losing_trades']}")
    print(f"  胜率: {stats['win_rate']:.2f}%")
    print(f"  总盈亏: ${stats['net_profit']:.2f}")
    print(f"  平均盈利: ${stats['avg_win']:.2f}")
    print(f"  平均亏损: ${stats['avg_loss']:.2f}")
    print(f"  盈亏比: {stats['profit_factor']:.2f}")
    print(f"  最大盈利: ${stats['max_win']:.2f}")
    print(f"  最大亏损: ${stats['max_loss']:.2f}")
    print(f"  最大连续盈利: {stats['max_consecutive_wins']}")
    print(f"  最大连续亏损: {stats['max_consecutive_losses']}")
    
    # 按品种统计
    print("\n📈 按品种统计:")
    for symbol, symbol_stats in stats['by_symbol'].items():
        print(f"  {symbol}:")
        print(f"    交易数: {symbol_stats['count']}")
        print(f"    盈亏: ${symbol_stats['profit']:.2f}")
        print(f"    胜率: {symbol_stats['win_rate']:.2f}%")
    
    # 按策略统计
    print("\n🎯 按策略统计:")
    for strategy, strategy_stats in stats['by_strategy'].items():
        print(f"  {strategy}:")
        print(f"    交易数: {strategy_stats['count']}")
        print(f"    盈亏: ${strategy_stats['profit']:.2f}")
        print(f"    胜率: {strategy_stats['win_rate']:.2f}%")


def demo_reports(recorder: TradeRecorder):
    """演示报告生成功能"""
    print_section("4. 报告生成")
    
    # 生成每日报告
    daily_report = recorder.generate_daily_report()
    print("\n📅 每日报告:")
    print(f"  日期: {daily_report['date']}")
    print(f"  交易数: {daily_report['summary'].get('closed_trades', 0)}")
    print(f"  净盈亏: ${daily_report['summary'].get('net_profit', 0):.2f}")
    
    # 生成每周报告
    weekly_report = recorder.generate_weekly_report()
    print("\n📆 每周报告:")
    print(f"  周期: {weekly_report['week_start']} 至 {weekly_report['week_end']}")
    print(f"  总交易数: {weekly_report['total_trades']}")
    print(f"  净盈亏: ${weekly_report['summary'].get('net_profit', 0):.2f}")
    
    # 生成每月报告
    monthly_report = recorder.generate_monthly_report()
    print("\n📊 每月报告:")
    print(f"  月份: {monthly_report['month']}")
    print(f"  总交易数: {monthly_report['total_trades']}")
    print(f"  净盈亏: ${monthly_report['summary'].get('net_profit', 0):.2f}")
    
    if monthly_report.get('best_day'):
        print(f"  最佳交易日: {monthly_report['best_day']['date']} (${monthly_report['best_day']['profit']:.2f})")
    
    if monthly_report.get('worst_day'):
        print(f"  最差交易日: {monthly_report['worst_day']['date']} (${monthly_report['worst_day']['profit']:.2f})")


def demo_audit_log(recorder: TradeRecorder):
    """演示审计日志功能"""
    print_section("5. 审计日志")
    
    # 获取审计日志
    audit_logs = recorder.get_audit_log()
    
    print(f"\n📝 审计日志 (最近 {min(5, len(audit_logs))} 条):")
    for log in audit_logs[:5]:
        print(f"  [{log['timestamp']}] {log['action']} - {log['trade_id']} - {log['symbol']}")
    
    # 按交易ID过滤
    if audit_logs:
        first_trade_id = audit_logs[0]['trade_id']
        filtered_logs = recorder.get_audit_log(trade_id=first_trade_id)
        print(f"\n🔍 交易 {first_trade_id} 的审计记录:")
        for log in filtered_logs:
            print(f"  [{log['timestamp']}] {log['action']}")


def demo_export(recorder: TradeRecorder):
    """演示导出功能"""
    print_section("6. 数据导出")
    
    # 导出到CSV
    csv_path = "data/demo_trades_export.csv"
    success = recorder.export_trades_to_csv(csv_path)
    
    if success:
        print(f"\n✓ 交易记录已导出到: {csv_path}")
    else:
        print("\n✗ 导出失败")


def demo_performance_summary(recorder: TradeRecorder):
    """演示性能摘要"""
    print_section("7. 性能摘要")
    
    summary = recorder.get_performance_summary()
    
    print("\n🎯 整体性能:")
    print(f"  总交易数: {summary['all_time'].get('closed_trades', 0)}")
    print(f"  总盈亏: ${summary['all_time'].get('net_profit', 0):.2f}")
    print(f"  胜率: {summary['all_time'].get('win_rate', 0):.2f}%")
    
    print("\n📅 最近30天:")
    print(f"  交易数: {summary['last_30_days'].get('closed_trades', 0)}")
    print(f"  盈亏: ${summary['last_30_days'].get('net_profit', 0):.2f}")
    print(f"  胜率: {summary['last_30_days'].get('win_rate', 0):.2f}%")
    
    print("\n📆 今日:")
    print(f"  交易数: {summary['today'].get('closed_trades', 0)}")
    print(f"  盈亏: ${summary['today'].get('net_profit', 0):.2f}")
    print(f"  胜率: {summary['today'].get('win_rate', 0):.2f}%")
    
    print(f"\n📊 当前状态:")
    print(f"  活跃交易: {summary['active_trades']}")
    print(f"  最近关闭交易: {summary['recent_closed_trades']}")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("  交易记录系统演示")
    print("=" * 60)
    
    try:
        # 1. 基本记录功能
        recorder = demo_basic_recording()
        
        # 2. 记录多笔交易
        demo_multiple_trades(recorder)
        
        # 3. 统计分析
        demo_statistics(recorder)
        
        # 4. 报告生成
        demo_reports(recorder)
        
        # 5. 审计日志
        demo_audit_log(recorder)
        
        # 6. 数据导出
        demo_export(recorder)
        
        # 7. 性能摘要
        demo_performance_summary(recorder)
        
        print("\n" + "=" * 60)
        print("  演示完成!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
