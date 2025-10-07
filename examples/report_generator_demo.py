"""
报告生成器演示
展示如何使用报告生成器生成各类交易报告
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
from src.utils.report_generator import ReportGenerator, ReportType, ReportFormat
from src.strategies.performance_tracker import PerformanceTracker, PerformancePeriod
from src.agents.risk_manager import RiskManagerAgent, RiskConfig
from src.core.models import Trade, Position, Account, TradeType, PositionType


def create_sample_data():
    """创建示例数据"""
    # 创建性能跟踪器
    performance_tracker = PerformanceTracker()
    
    # 添加示例交易记录
    now = datetime.now()
    
    # 策略1: 趋势跟踪策略
    for i in range(20):
        trade = Trade(
            trade_id=f"trend_{i}",
            symbol="EURUSD",
            type=TradeType.BUY if i % 2 == 0 else TradeType.SELL,
            volume=0.1,
            open_price=1.1000 + i * 0.0001,
            close_price=1.1010 + i * 0.0001 if i % 3 != 0 else 1.0995 + i * 0.0001,
            sl=1.0990,
            tp=1.1020,
            profit=10.0 if i % 3 != 0 else -5.0,
            commission=0.5,
            swap=0.1,
            open_time=now - timedelta(days=30-i),
            close_time=now - timedelta(days=30-i, hours=-2),
            strategy_id="trend_following"
        )
        performance_tracker.record_trade(trade)
    
    # 策略2: 震荡策略
    for i in range(15):
        trade = Trade(
            trade_id=f"range_{i}",
            symbol="XAUUSD",
            type=TradeType.BUY if i % 2 == 0 else TradeType.SELL,
            volume=0.01,
            open_price=1800.0 + i,
            close_price=1805.0 + i if i % 4 != 0 else 1797.0 + i,
            sl=1795.0,
            tp=1810.0,
            profit=5.0 if i % 4 != 0 else -3.0,
            commission=0.3,
            swap=-0.2,
            open_time=now - timedelta(days=20-i),
            close_time=now - timedelta(days=20-i, hours=-1),
            strategy_id="range_trading"
        )
        performance_tracker.record_trade(trade)
    
    # 策略3: 突破策略
    for i in range(10):
        trade = Trade(
            trade_id=f"breakout_{i}",
            symbol="USOIL",
            type=TradeType.BUY,
            volume=0.05,
            open_price=75.0 + i * 0.1,
            close_price=75.5 + i * 0.1 if i % 5 != 0 else 74.8 + i * 0.1,
            sl=74.5,
            tp=76.0,
            profit=25.0 if i % 5 != 0 else -10.0,
            commission=0.4,
            swap=0.0,
            open_time=now - timedelta(days=15-i),
            close_time=now - timedelta(days=15-i, hours=-3),
            strategy_id="breakout"
        )
        performance_tracker.record_trade(trade)
    
    return performance_tracker


def create_sample_account():
    """创建示例账户"""
    return Account(
        account_id="demo_account_12345",
        balance=10000.0,
        equity=10850.0,
        margin=500.0,
        free_margin=10350.0,
        margin_level=2170.0,
        currency="USD",
        leverage=100
    )


def create_sample_positions():
    """创建示例持仓"""
    now = datetime.now()
    return [
        Position(
            position_id="pos_1",
            symbol="EURUSD",
            type=PositionType.LONG,
            volume=0.1,
            open_price=1.1000,
            current_price=1.1015,
            sl=1.0990,
            tp=1.1030,
            profit=15.0,
            swap=0.2,
            commission=0.5,
            open_time=now - timedelta(hours=4)
        ),
        Position(
            position_id="pos_2",
            symbol="XAUUSD",
            type=PositionType.SHORT,
            volume=0.01,
            open_price=1800.0,
            current_price=1795.0,
            sl=1810.0,
            tp=1785.0,
            profit=5.0,
            swap=-0.3,
            commission=0.3,
            open_time=now - timedelta(hours=2)
        ),
        Position(
            position_id="pos_3",
            symbol="USOIL",
            type=PositionType.LONG,
            volume=0.05,
            open_price=75.0,
            current_price=75.3,
            sl=74.5,
            tp=76.0,
            profit=15.0,
            swap=0.0,
            commission=0.4,
            open_time=now - timedelta(hours=1)
        )
    ]


def demo_daily_report():
    """演示生成每日报告"""
    print("=" * 80)
    print("演示1: 生成每日交易报告")
    print("=" * 80)
    
    # 准备数据
    performance_tracker = create_sample_data()
    risk_manager = RiskManagerAgent(RiskConfig())
    account = create_sample_account()
    
    # 创建报告生成器
    report_generator = ReportGenerator(
        performance_tracker=performance_tracker,
        risk_manager=risk_manager,
        output_dir="logs/reports"
    )
    
    # 生成每日报告
    report = report_generator.generate_daily_report(
        date=datetime.now() - timedelta(days=1),
        account=account
    )
    
    print(f"\n报告ID: {report.report_id}")
    print(f"报告类型: {report.report_type.value}")
    print(f"统计周期: {report.period_start.date()} 至 {report.period_end.date()}")
    
    # 显示账户摘要
    print("\n账户摘要:")
    print(f"  余额: ${report.account_summary['balance']:.2f}")
    print(f"  净值: ${report.account_summary['equity']:.2f}")
    print(f"  盈亏: ${report.account_summary['profit_loss']:.2f} ({report.account_summary['profit_loss_pct']:.2f}%)")
    print(f"  保证金水平: {report.account_summary['margin_level']:.2f}%")
    
    # 显示交易统计
    print("\n交易统计:")
    print(f"  总交易次数: {report.trading_summary['total_trades']}")
    print(f"  盈利交易: {report.trading_summary['winning_trades']}")
    print(f"  亏损交易: {report.trading_summary['losing_trades']}")
    print(f"  胜率: {report.trading_summary['win_rate']:.2%}")
    print(f"  净利润: ${report.trading_summary['net_profit']:.2f}")
    print(f"  盈利因子: {report.trading_summary['profit_factor']:.2f}")
    
    # 显示风险指标
    print("\n风险指标:")
    print(f"  最大回撤: {report.risk_metrics['max_drawdown']:.2%}")
    print(f"  当前回撤: {report.risk_metrics['current_drawdown']:.2%}")
    print(f"  连续亏损: {report.risk_metrics['consecutive_losses']}")
    
    # 显示建议
    if report.recommendations:
        print("\n建议:")
        for i, rec in enumerate(report.recommendations[:5], 1):
            print(f"  {i}. {rec}")
    
    # 显示警告
    if report.warnings:
        print("\n警告:")
        for i, warn in enumerate(report.warnings, 1):
            print(f"  {i}. ⚠️ {warn}")
    
    print(f"\n报告已保存至: logs/reports/{report.report_id}.json")


def demo_strategy_analysis_report():
    """演示生成策略分析报告"""
    print("\n" + "=" * 80)
    print("演示2: 生成策略分析和性能对比报告")
    print("=" * 80)
    
    # 准备数据
    performance_tracker = create_sample_data()
    
    # 创建报告生成器
    report_generator = ReportGenerator(
        performance_tracker=performance_tracker,
        output_dir="logs/reports"
    )
    
    # 生成策略分析报告
    report = report_generator.generate_strategy_analysis_report(
        strategy_names=["trend_following", "range_trading", "breakout"],
        period=PerformancePeriod.MONTHLY
    )
    
    print(f"\n报告ID: {report.report_id}")
    print(f"分析周期: {report.period_start.date()} 至 {report.period_end.date()}")
    print(f"策略数量: {report.trading_summary['strategies_compared']}")
    
    # 显示策略表现
    print("\n策略表现:")
    for i, strategy in enumerate(report.strategy_performance, 1):
        print(f"\n  {i}. {strategy.strategy_name}")
        print(f"     总交易: {strategy.total_trades}")
        print(f"     胜率: {strategy.win_rate:.2%}")
        print(f"     净利润: ${strategy.net_profit:.2f}")
        print(f"     盈利因子: {strategy.profit_factor:.2f}")
        print(f"     夏普比率: {strategy.sharpe_ratio:.2f}")
        print(f"     最大回撤: {strategy.max_drawdown:.2f}")
    
    # 显示策略排名
    print("\n策略排名 (按净利润):")
    rankings = report.trading_summary['rankings']['by_net_profit']
    for i, rank in enumerate(rankings, 1):
        print(f"  {i}. {rank['strategy']}: ${rank['value']:.2f}")
    
    print("\n策略排名 (按胜率):")
    rankings = report.trading_summary['rankings']['by_win_rate']
    for i, rank in enumerate(rankings, 1):
        print(f"  {i}. {rank['strategy']}: {rank['value']:.2%}")
    
    # 显示建议
    if report.recommendations:
        print("\n策略优化建议:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. {rec}")
    
    print(f"\n报告已保存至: logs/reports/{report.report_id}.json")


def demo_risk_analysis_report():
    """演示生成风险分析报告"""
    print("\n" + "=" * 80)
    print("演示3: 生成风险分析报告")
    print("=" * 80)
    
    # 准备数据
    performance_tracker = create_sample_data()
    risk_manager = RiskManagerAgent(RiskConfig())
    account = create_sample_account()
    positions = create_sample_positions()
    
    # 更新风险管理器的权益历史
    now = datetime.now()
    for i in range(30):
        equity = 10000 + i * 20 - (i % 7) * 30
        risk_manager.drawdown_monitor.update_equity(
            now - timedelta(days=30-i),
            equity
        )
    
    # 创建报告生成器
    report_generator = ReportGenerator(
        performance_tracker=performance_tracker,
        risk_manager=risk_manager,
        output_dir="logs/reports"
    )
    
    # 生成风险分析报告
    report = report_generator.generate_risk_analysis_report(
        account=account,
        positions=positions,
        period=PerformancePeriod.MONTHLY
    )
    
    print(f"\n报告ID: {report.report_id}")
    print(f"分析周期: {report.period_start.date()} 至 {report.period_end.date()}")
    
    # 显示账户风险
    print("\n账户风险指标:")
    print(f"  账户净值: ${report.account_summary['equity']:.2f}")
    print(f"  总敞口: ${report.risk_metrics['total_exposure']:.2f}")
    print(f"  敞口比例: {report.risk_metrics['exposure_pct']:.2f}%")
    print(f"  保证金使用率: {report.risk_metrics['margin_usage_pct']:.2f}%")
    print(f"  保证金水平: {report.risk_metrics['margin_level']:.2f}%")
    
    # 显示回撤分析
    print("\n回撤分析:")
    print(f"  最大回撤: {report.risk_metrics['max_drawdown']:.2%}")
    print(f"  当前回撤: {report.risk_metrics['current_drawdown']:.2%}")
    print(f"  回撤持续天数: {report.risk_metrics['drawdown_duration_days']}")
    print(f"  恢复因子: {report.risk_metrics['recovery_factor']:.2f}")
    
    # 显示VaR指标（如果有）
    if 'var_1d' in report.risk_metrics:
        print("\nVaR风险指标:")
        print(f"  1日VaR: ${report.risk_metrics['var_1d']:.2f} ({report.risk_metrics['var_1d_pct']:.2f}%)")
        print(f"  5日VaR: ${report.risk_metrics['var_5d']:.2f}")
        print(f"  10日VaR: ${report.risk_metrics['var_10d']:.2f}")
    
    # 显示持仓分析
    print("\n持仓分析:")
    print(f"  总持仓数: {report.trading_summary['total_positions']}")
    print(f"  多头持仓: {report.trading_summary['long_positions']}")
    print(f"  空头持仓: {report.trading_summary['short_positions']}")
    
    print("\n按品种持仓:")
    for symbol, data in report.trading_summary['positions_by_symbol'].items():
        print(f"  {symbol}:")
        print(f"    持仓数: {data['count']}")
        print(f"    总手数: {data['total_volume']:.2f}")
        print(f"    敞口: ${data['total_exposure']:.2f}")
        print(f"    未实现盈亏: ${data['unrealized_pnl']:.2f}")
    
    # 显示风险建议
    if report.recommendations:
        print("\n风险管理建议:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. {rec}")
    
    # 显示风险警告
    if report.warnings:
        print("\n风险警告:")
        for i, warn in enumerate(report.warnings, 1):
            print(f"  {i}. ⚠️ {warn}")
    
    print(f"\n报告已保存至: logs/reports/{report.report_id}.json")


def demo_compliance_report():
    """演示生成合规检查报告"""
    print("\n" + "=" * 80)
    print("演示4: 生成合规检查报告")
    print("=" * 80)
    
    # 准备数据
    performance_tracker = create_sample_data()
    risk_manager = RiskManagerAgent(RiskConfig())
    account = create_sample_account()
    
    # 创建报告生成器
    report_generator = ReportGenerator(
        performance_tracker=performance_tracker,
        risk_manager=risk_manager,
        output_dir="logs/reports"
    )
    
    # 生成合规检查报告
    report = report_generator.generate_compliance_report(
        account=account,
        period=PerformancePeriod.MONTHLY
    )
    
    print(f"\n报告ID: {report.report_id}")
    print(f"检查周期: {report.period_start.date()} 至 {report.period_end.date()}")
    
    # 显示合规摘要
    summary = report.trading_summary['compliance_summary']
    print("\n合规检查摘要:")
    print(f"  总检查项: {summary['total_checks']}")
    print(f"  通过检查: {summary['passed_checks']}")
    print(f"  未通过检查: {summary['failed_checks']}")
    print(f"  合规率: {summary['compliance_rate']:.2f}%")
    print(f"  严重问题: {summary['critical_failures']}")
    print(f"  警告问题: {summary['warning_failures']}")
    
    # 显示合规检查详情
    print("\n合规检查详情:")
    for i, result in enumerate(report.compliance_results, 1):
        status = "✓ 通过" if result.passed else "✗ 未通过"
        print(f"\n  {i}. {result.rule_name} [{result.severity}] - {status}")
        print(f"     {result.message}")
        print(f"     实际值: {result.actual_value:.2%}, 限制: {result.threshold:.2%}")
    
    # 显示严重问题
    if summary['critical_issues']:
        print("\n严重合规问题:")
        for i, issue in enumerate(summary['critical_issues'], 1):
            print(f"  {i}. {issue}")
    
    # 显示合规建议
    if report.recommendations:
        print("\n合规改进建议:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. {rec}")
    
    print(f"\n报告已保存至: logs/reports/{report.report_id}.json")


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("报告生成器演示程序")
    print("=" * 80)
    
    # 演示1: 每日报告
    demo_daily_report()
    
    # 演示2: 策略分析报告
    demo_strategy_analysis_report()
    
    # 演示3: 风险分析报告
    demo_risk_analysis_report()
    
    # 演示4: 合规检查报告
    demo_compliance_report()
    
    print("\n" + "=" * 80)
    print("演示完成！所有报告已保存至 logs/reports/ 目录")
    print("=" * 80)


if __name__ == "__main__":
    main()
