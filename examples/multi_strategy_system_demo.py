"""
多策略管理系统集成演示
展示策略管理器、权重优化器和性能跟踪器的协同工作
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.strategies.strategy_manager import (
    StrategyManager, ConflictResolutionMethod, SignalAggregationMethod
)
from src.strategies.weight_optimizer import (
    StrategyWeightOptimizer, WeightOptimizationMethod, WeightOptimizationConfig
)
from src.strategies.performance_tracker import (
    PerformanceTracker, PerformancePeriod
)
from src.strategies.base_strategies import (
    TrendFollowingStrategy, MeanReversionStrategy, BreakoutStrategy,
    StrategyConfig, StrategyType
)
from src.core.models import MarketData, Trade, TradeType


def create_sample_market_data(symbol: str = "EURUSD") -> MarketData:
    """创建示例市场数据"""
    dates = pd.date_range(end=datetime.now(), periods=100, freq='h')
    
    np.random.seed(42)
    close_prices = 1.1000 + np.cumsum(np.random.randn(100) * 0.0001)
    
    ohlcv = pd.DataFrame({
        'time': dates,
        'open': close_prices + np.random.randn(100) * 0.0001,
        'high': close_prices + abs(np.random.randn(100) * 0.0002),
        'low': close_prices - abs(np.random.randn(100) * 0.0002),
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    return MarketData(
        symbol=symbol,
        timeframe="H1",
        timestamp=datetime.now(),
        ohlcv=ohlcv,
        indicators={},
        spread=0.00002
    )


def create_sample_trades(strategy_name: str, num_trades: int = 20) -> list:
    """创建示例交易记录"""
    trades = []
    
    for i in range(num_trades):
        # 模拟不同的胜率
        if strategy_name == "trend_strategy":
            profit = 100 if i < 14 else -50  # 70%胜率
        elif strategy_name == "mean_reversion":
            profit = 80 if i < 12 else -60  # 60%胜率
        else:  # breakout
            profit = 120 if i < 10 else -70  # 50%胜率
        
        trade = Trade(
            trade_id=f"{strategy_name}_{i}",
            symbol="EURUSD",
            type=TradeType.BUY if profit > 0 else TradeType.SELL,
            volume=0.1,
            open_price=1.1000,
            close_price=1.1000 + profit/10000,
            profit=profit,
            commission=2.0,
            swap=0.5,
            open_time=datetime.now() - timedelta(days=num_trades-i),
            close_time=datetime.now() - timedelta(days=num_trades-i, hours=-2),
            strategy_id=strategy_name
        )
        trades.append(trade)
    
    return trades


def main():
    """主演示函数"""
    print("=" * 80)
    print("多策略管理系统集成演示")
    print("=" * 80)
    
    # ========== 1. 初始化系统组件 ==========
    print("\n[1] 初始化系统组件...")
    
    # 创建策略管理器
    strategy_manager = StrategyManager(
        conflict_resolution=ConflictResolutionMethod.HIGHEST_STRENGTH,
        aggregation_method=SignalAggregationMethod.WEIGHTED_AVERAGE,
        max_signals_per_symbol=3
    )
    print("✓ 策略管理器已创建")
    
    # 创建权重优化器
    weight_config = WeightOptimizationConfig(
        method=WeightOptimizationMethod.PERFORMANCE_BASED,
        min_weight=0.1,
        max_weight=2.0,
        min_trades_required=10
    )
    weight_optimizer = StrategyWeightOptimizer(weight_config)
    print("✓ 权重优化器已创建")
    
    # 创建性能跟踪器
    performance_tracker = PerformanceTracker()
    print("✓ 性能跟踪器已创建")
    
    # ========== 2. 注册策略 ==========
    print("\n[2] 注册交易策略...")
    
    strategies_to_register = [
        ("trend_strategy", TrendFollowingStrategy, StrategyType.TREND_FOLLOWING),
        ("mean_reversion", MeanReversionStrategy, StrategyType.MEAN_REVERSION),
        ("breakout", BreakoutStrategy, StrategyType.BREAKOUT)
    ]
    
    for name, strategy_class, strategy_type in strategies_to_register:
        config = StrategyConfig(
            name=name,
            strategy_type=strategy_type,
            enabled=True,
            min_signal_strength=0.5
        )
        
        strategy_manager.register_strategy(
            name=name,
            strategy_class=strategy_class,
            config=config,
            weight=1.0,
            auto_load=True
        )
        print(f"✓ 已注册策略: {name}")
    
    # ========== 3. 模拟历史交易数据 ==========
    print("\n[3] 加载历史交易数据...")
    
    for name, _, _ in strategies_to_register:
        trades = create_sample_trades(name, num_trades=20)
        
        # 记录到性能跟踪器
        for trade in trades:
            performance_tracker.record_trade(trade)
        
        # 更新权重优化器
        weight_optimizer.update_strategy_metrics(name, trades)
        
        print(f"✓ 已加载 {name} 的 {len(trades)} 笔交易")
    
    # ========== 4. 计算和更新策略权重 ==========
    print("\n[4] 计算策略权重...")
    
    strategy_names = [name for name, _, _ in strategies_to_register]
    weights = weight_optimizer.calculate_weights(strategy_names, force_update=True)
    
    print("\n策略权重:")
    for name, weight in weights.items():
        print(f"  {name}: {weight:.3f}")
    
    # 更新策略管理器中的权重
    strategy_manager.update_strategy_weights(weights)
    print("\n✓ 策略权重已更新")
    
    # ========== 5. 生成交易信号 ==========
    print("\n[5] 生成交易信号...")
    
    market_data = create_sample_market_data("EURUSD")
    signals = strategy_manager.generate_signals(market_data)
    
    print(f"\n生成了 {len(signals)} 个信号:")
    for signal in signals:
        print(f"  策略: {signal.strategy_id}")
        print(f"    方向: {'买入' if signal.direction > 0 else '卖出' if signal.direction < 0 else '平仓'}")
        print(f"    强度: {signal.strength:.3f}")
        print(f"    置信度: {signal.confidence:.3f}")
        print(f"    入场价: {signal.entry_price:.5f}")
        print()
    
    # ========== 6. 信号聚合 ==========
    print("[6] 聚合信号...")
    
    if signals:
        aggregation_result = strategy_manager.aggregate_signals(signals, symbol="EURUSD")
        
        print(f"\n聚合结果:")
        print(f"  原始信号数: {len(aggregation_result.original_signals)}")
        print(f"  冲突检测: {'是' if aggregation_result.conflicts_detected else '否'}")
        print(f"  解决方法: {aggregation_result.resolution_method}")
        
        if aggregation_result.aggregated_signal:
            agg_signal = aggregation_result.aggregated_signal
            print(f"\n聚合信号:")
            print(f"  方向: {'买入' if agg_signal.direction > 0 else '卖出'}")
            print(f"  强度: {agg_signal.strength:.3f}")
            print(f"  置信度: {agg_signal.confidence:.3f}")
            print(f"  入场价: {agg_signal.entry_price:.5f}")
    
    # ========== 7. 生成性能报告 ==========
    print("\n[7] 生成性能报告...")
    
    report = performance_tracker.generate_report(
        period=PerformancePeriod.ALL_TIME,
        strategy_names=strategy_names
    )
    
    print(f"\n性能报告 (ID: {report.report_id}):")
    print(f"  统计周期: {report.period.value}")
    print(f"  策略数量: {len(report.strategies)}")
    
    # 显示摘要
    summary = report.summary
    print(f"\n摘要统计:")
    print(f"  总交易数: {summary['total_trades']}")
    print(f"  总净利润: {summary['total_net_profit']:.2f}")
    print(f"  平均胜率: {summary['avg_win_rate']:.2%}")
    print(f"  平均盈利因子: {summary['avg_profit_factor']:.2f}")
    
    # 显示排名
    print(f"\n策略排名 (按净利润):")
    for i, (name, profit) in enumerate(report.rankings['net_profit'], 1):
        print(f"  {i}. {name}: {profit:.2f}")
    
    # ========== 8. 详细性能指标 ==========
    print("\n[8] 详细性能指标...")
    
    for strategy_name in strategy_names:
        metrics = performance_tracker.calculate_metrics(strategy_name)
        
        print(f"\n{strategy_name}:")
        print(f"  总交易: {metrics.total_trades}")
        print(f"  胜率: {metrics.win_rate:.2%}")
        print(f"  净利润: {metrics.net_profit:.2f}")
        print(f"  盈利因子: {metrics.profit_factor:.2f}")
        print(f"  夏普比率: {metrics.sharpe_ratio:.2f}")
        print(f"  最大回撤: {metrics.max_drawdown:.2f}")
        print(f"  平均盈利/交易: {metrics.avg_profit_per_trade:.2f}")
    
    # ========== 9. 性能对比 ==========
    print("\n[9] 策略性能对比...")
    
    comparison = performance_tracker.get_performance_comparison(
        strategy_names=strategy_names,
        period=PerformancePeriod.ALL_TIME
    )
    
    print("\n" + comparison.to_string(index=False))
    
    # ========== 10. 导出报告 ==========
    print("\n[10] 导出报告...")
    
    report_file = "performance_report.json"
    if performance_tracker.export_report(report, report_file):
        print(f"✓ 报告已导出到: {report_file}")
    
    # ========== 总结 ==========
    print("\n" + "=" * 80)
    print("演示完成!")
    print("=" * 80)
    print("\n系统功能总结:")
    print("✓ 策略注册和管理")
    print("✓ 动态权重优化")
    print("✓ 信号生成和聚合")
    print("✓ 冲突检测和解决")
    print("✓ 性能实时监控")
    print("✓ 多维度排名")
    print("✓ 详细报告生成")
    print("\n系统已准备好进行实盘交易!")


if __name__ == "__main__":
    main()
