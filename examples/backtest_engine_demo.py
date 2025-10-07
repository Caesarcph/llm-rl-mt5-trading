"""
回测引擎演示脚本
展示BacktestEngine的完整功能，包括：
- 单策略回测
- 多策略并行回测
- 结果可视化
- 结果导出
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.models import MarketData
from src.strategies.base_strategies import (
    StrategyConfig, StrategyType,
    TrendFollowingStrategy, MeanReversionStrategy, BreakoutStrategy
)
from src.strategies.backtest import BacktestConfig, BacktestEngine


def create_market_data(periods=500, trend_type='uptrend'):
    """
    创建模拟市场数据
    
    Args:
        periods: 数据周期数
        trend_type: 趋势类型 ('uptrend', 'downtrend', 'sideways', 'volatile')
    """
    print(f"生成{periods}个周期的{trend_type}市场数据...")
    
    dates = pd.date_range(start='2023-01-01', periods=periods, freq='h')
    np.random.seed(42)
    
    base_price = 1.1000
    
    if trend_type == 'uptrend':
        # 上升趋势
        trend = np.linspace(0, 0.02, periods)
        noise = np.random.normal(0, 0.0003, periods)
    elif trend_type == 'downtrend':
        # 下降趋势
        trend = np.linspace(0, -0.02, periods)
        noise = np.random.normal(0, 0.0003, periods)
    elif trend_type == 'sideways':
        # 横盘震荡
        trend = np.sin(np.linspace(0, 4*np.pi, periods)) * 0.005
        noise = np.random.normal(0, 0.0002, periods)
    else:  # volatile
        # 高波动
        trend = np.cumsum(np.random.normal(0, 0.0005, periods))
        noise = np.random.normal(0, 0.0008, periods)
    
    prices = base_price + trend + noise
    
    market_data_list = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        high = price + abs(np.random.normal(0, 0.0003))
        low = price - abs(np.random.normal(0, 0.0003))
        open_price = prices[i-1] if i > 0 else price
        close_price = price
        volume = np.random.randint(5000, 15000)
        
        ohlcv_data = pd.DataFrame([{
            'timestamp': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        }])
        
        market_data = MarketData(
            symbol="EURUSD",
            timeframe="H1",
            timestamp=date,
            ohlcv=ohlcv_data
        )
        
        market_data_list.append(market_data)
    
    return market_data_list


def demo_single_strategy_backtest():
    """演示单策略回测"""
    print("\n" + "="*80)
    print("演示1: 单策略回测")
    print("="*80)
    
    # 创建回测配置
    config = BacktestConfig(
        initial_balance=10000.0,
        leverage=100,
        spread=0.0001,
        commission=0.0,
        slippage=0.0001
    )
    
    # 创建回测引擎
    engine = BacktestEngine(config)
    
    # 创建策略
    strategy_config = StrategyConfig(
        name="Trend Following Strategy",
        strategy_type=StrategyType.TREND_FOLLOWING,
        risk_per_trade=0.02,
        enabled=True
    )
    strategy = TrendFollowingStrategy(strategy_config)
    
    # 创建市场数据
    market_data = create_market_data(periods=500, trend_type='uptrend')
    
    # 运行回测
    print("\n运行回测...")
    result = engine.run_backtest(strategy, market_data)
    
    # 打印结果
    print("\n回测结果:")
    print(f"策略名称: {result.strategy_name}")
    print(f"品种: {result.symbol}")
    print(f"初始余额: ${result.initial_balance:.2f}")
    print(f"最终余额: ${result.final_balance:.2f}")
    print(f"总收益率: {result.total_return*100:.2f}%")
    print(f"总交易次数: {result.total_trades}")
    print(f"胜率: {result.win_rate*100:.2f}%")
    print(f"盈利因子: {result.profit_factor:.2f}")
    print(f"夏普比率: {result.sharpe_ratio:.2f}")
    print(f"最大回撤: {result.max_drawdown_percent*100:.2f}%")
    
    # 可视化结果
    print("\n生成可视化图表...")
    try:
        engine.visualize_results(result, save_path='logs/reports/backtest_single_strategy.png')
        print("图表已保存到: logs/reports/backtest_single_strategy.png")
    except Exception as e:
        print(f"可视化失败: {str(e)}")
    
    # 导出结果
    print("\n导出回测结果...")
    engine.export_results(result, 'logs/reports/backtest_result.json', format='json')
    engine.export_results(result, 'logs/reports/backtest_result.html', format='html')
    print("结果已导出到: logs/reports/")
    
    return result


def demo_multiple_strategies_backtest():
    """演示多策略回测"""
    print("\n" + "="*80)
    print("演示2: 多策略回测与对比")
    print("="*80)
    
    # 创建回测配置
    config = BacktestConfig(
        initial_balance=10000.0,
        leverage=100,
        spread=0.0001,
        commission=0.0,
        slippage=0.0001
    )
    
    # 创建回测引擎
    engine = BacktestEngine(config)
    
    # 创建多个策略
    strategies = []
    
    # 趋势跟踪策略
    trend_config = StrategyConfig(
        name="Trend Following",
        strategy_type=StrategyType.TREND_FOLLOWING,
        risk_per_trade=0.02,
        enabled=True
    )
    strategies.append(TrendFollowingStrategy(trend_config))
    
    # 均值回归策略
    mean_reversion_config = StrategyConfig(
        name="Mean Reversion",
        strategy_type=StrategyType.MEAN_REVERSION,
        risk_per_trade=0.015,
        enabled=True
    )
    strategies.append(MeanReversionStrategy(mean_reversion_config))
    
    # 突破策略
    breakout_config = StrategyConfig(
        name="Breakout",
        strategy_type=StrategyType.BREAKOUT,
        risk_per_trade=0.025,
        enabled=True
    )
    strategies.append(BreakoutStrategy(breakout_config))
    
    # 创建市场数据
    market_data = create_market_data(periods=500, trend_type='volatile')
    
    # 运行多策略回测
    print(f"\n运行{len(strategies)}个策略的回测...")
    results = engine.run_multiple_strategies(strategies, market_data, parallel=False)
    
    # 打印对比结果
    print("\n策略对比:")
    comparison_df = engine.compare_strategies(results)
    print(comparison_df.to_string(index=False))
    
    # 可视化对比
    print("\n生成策略对比图表...")
    try:
        engine.visualize_strategy_comparison(results, save_path='logs/reports/backtest_comparison.png')
        print("对比图表已保存到: logs/reports/backtest_comparison.png")
    except Exception as e:
        print(f"可视化失败: {str(e)}")
    
    return results


def demo_parallel_backtest():
    """演示并行回测"""
    print("\n" + "="*80)
    print("演示3: 并行回测（加速多策略测试）")
    print("="*80)
    
    # 创建回测配置
    config = BacktestConfig(
        initial_balance=10000.0,
        leverage=100,
        spread=0.0001,
        commission=0.0,
        slippage=0.0001
    )
    
    # 创建回测引擎
    engine = BacktestEngine(config)
    
    # 创建多个策略变体
    strategies = []
    for i, risk in enumerate([0.01, 0.015, 0.02, 0.025, 0.03]):
        config = StrategyConfig(
            name=f"Trend Strategy (Risk {risk*100:.1f}%)",
            strategy_type=StrategyType.TREND_FOLLOWING,
            risk_per_trade=risk,
            enabled=True
        )
        strategies.append(TrendFollowingStrategy(config))
    
    # 创建市场数据
    market_data = create_market_data(periods=500, trend_type='uptrend')
    
    # 顺序执行
    print(f"\n顺序执行{len(strategies)}个策略...")
    import time
    start_time = time.time()
    results_sequential = engine.run_multiple_strategies(strategies, market_data, parallel=False)
    sequential_time = time.time() - start_time
    print(f"顺序执行耗时: {sequential_time:.2f}秒")
    
    # 并行执行
    print(f"\n并行执行{len(strategies)}个策略...")
    start_time = time.time()
    results_parallel = engine.run_multiple_strategies(strategies, market_data, parallel=True)
    parallel_time = time.time() - start_time
    print(f"并行执行耗时: {parallel_time:.2f}秒")
    
    if sequential_time > 0:
        speedup = sequential_time / parallel_time
        print(f"\n加速比: {speedup:.2f}x")
    
    # 打印最佳策略
    print("\n策略排名（按总收益率）:")
    for i, (name, result) in enumerate(sorted(results_parallel.items(), 
                                              key=lambda x: x[1].total_return if x[1] else -float('inf'),
                                              reverse=True), 1):
        if result:
            print(f"{i}. {name}: {result.total_return*100:.2f}%")


def demo_detailed_analysis():
    """演示详细分析"""
    print("\n" + "="*80)
    print("演示4: 详细回测分析")
    print("="*80)
    
    # 创建回测配置
    config = BacktestConfig(
        initial_balance=10000.0,
        leverage=100,
        spread=0.0001,
        commission=0.0,
        slippage=0.0001
    )
    
    # 创建回测引擎
    engine = BacktestEngine(config)
    
    # 创建策略
    strategy_config = StrategyConfig(
        name="Comprehensive Test Strategy",
        strategy_type=StrategyType.TREND_FOLLOWING,
        risk_per_trade=0.02,
        enabled=True
    )
    strategy = TrendFollowingStrategy(strategy_config)
    
    # 创建市场数据
    market_data = create_market_data(periods=1000, trend_type='volatile')
    
    # 运行回测
    print("\n运行回测...")
    result = engine.run_backtest(strategy, market_data)
    
    # 生成详细分析图表
    print("\n生成详细分析图表...")
    try:
        engine.visualize_detailed_analysis(result, save_path='logs/reports/backtest_detailed_analysis.png')
        print("详细分析图表已保存到: logs/reports/backtest_detailed_analysis.png")
    except Exception as e:
        print(f"可视化失败: {str(e)}")
    
    # 导出多种格式
    print("\n导出多种格式...")
    engine.export_results(result, 'logs/reports/detailed_result.json', format='json')
    engine.export_results(result, 'logs/reports/detailed_result.csv', format='csv')
    engine.export_results(result, 'logs/reports/detailed_result.html', format='html')
    
    try:
        engine.export_results(result, 'logs/reports/detailed_result.xlsx', format='excel')
        print("Excel格式已导出")
    except Exception as e:
        print(f"Excel导出失败（可能缺少openpyxl库）: {str(e)}")
    
    print("所有格式已导出到: logs/reports/")


def main():
    """主函数"""
    print("="*80)
    print("回测引擎完整功能演示")
    print("="*80)
    
    # 确保输出目录存在
    os.makedirs('logs/reports', exist_ok=True)
    
    try:
        # 演示1: 单策略回测
        demo_single_strategy_backtest()
        
        # 演示2: 多策略回测
        demo_multiple_strategies_backtest()
        
        # 演示3: 并行回测
        demo_parallel_backtest()
        
        # 演示4: 详细分析
        demo_detailed_analysis()
        
        print("\n" + "="*80)
        print("所有演示完成！")
        print("="*80)
        print("\n查看生成的报告:")
        print("- logs/reports/backtest_single_strategy.png")
        print("- logs/reports/backtest_comparison.png")
        print("- logs/reports/backtest_detailed_analysis.png")
        print("- logs/reports/*.json, *.csv, *.html")
        
    except Exception as e:
        print(f"\n演示过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
