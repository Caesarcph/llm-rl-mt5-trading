#!/usr/bin/env python3
"""
回测示例
演示如何回测交易策略
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.backtest.backtest_engine import BacktestEngine
from src.strategies.trend_strategy import TrendStrategy
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_simple_backtest():
    """运行简单回测"""
    logger.info("=== 简单回测示例 ===")
    
    # 1. 设置回测参数
    initial_balance = 10000  # 初始资金
    symbol = "EURUSD"
    timeframe = "H1"
    
    # 回测时间范围（最近3个月）
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    logger.info(f"回测参数:")
    logger.info(f"  初始资金: ${initial_balance}")
    logger.info(f"  品种: {symbol}")
    logger.info(f"  时间周期: {timeframe}")
    logger.info(f"  开始日期: {start_date.strftime('%Y-%m-%d')}")
    logger.info(f"  结束日期: {end_date.strftime('%Y-%m-%d')}")
    
    # 2. 创建回测引擎
    logger.info("\n创建回测引擎...")
    engine = BacktestEngine(
        initial_balance=initial_balance,
        start_date=start_date,
        end_date=end_date,
        symbol=symbol,
        timeframe=timeframe
    )
    
    # 3. 添加策略
    logger.info("添加趋势策略...")
    strategy = TrendStrategy(
        fast_period=20,
        slow_period=50,
        signal_threshold=0.7
    )
    engine.add_strategy(strategy)
    
    # 4. 运行回测
    logger.info("\n开始回测...")
    results = engine.run()
    
    # 5. 显示结果
    logger.info("\n" + "=" * 60)
    logger.info("回测结果:")
    logger.info("=" * 60)
    logger.info(f"初始资金: ${results['initial_balance']:,.2f}")
    logger.info(f"最终资金: ${results['final_balance']:,.2f}")
    logger.info(f"总收益: ${results['total_profit']:,.2f} ({results['total_return']:.2%})")
    logger.info(f"最大回撤: {results['max_drawdown']:.2%}")
    logger.info(f"夏普比率: {results['sharpe_ratio']:.2f}")
    logger.info(f"索提诺比率: {results['sortino_ratio']:.2f}")
    logger.info(f"卡玛比率: {results['calmar_ratio']:.2f}")
    logger.info("-" * 60)
    logger.info(f"总交易次数: {results['total_trades']}")
    logger.info(f"盈利交易: {results['winning_trades']} ({results['win_rate']:.2%})")
    logger.info(f"亏损交易: {results['losing_trades']}")
    logger.info(f"平均盈利: ${results['avg_win']:,.2f}")
    logger.info(f"平均亏损: ${results['avg_loss']:,.2f}")
    logger.info(f"盈利因子: {results['profit_factor']:.2f}")
    logger.info(f"最大连续盈利: {results['max_consecutive_wins']}")
    logger.info(f"最大连续亏损: {results['max_consecutive_losses']}")
    logger.info("=" * 60)
    
    # 6. 生成报告
    logger.info("\n生成详细报告...")
    report_path = engine.generate_report()
    logger.info(f"报告已保存: {report_path}")
    
    return results


def run_parameter_optimization():
    """运行参数优化"""
    logger.info("\n=== 参数优化示例 ===")
    
    # 1. 定义参数范围
    param_ranges = {
        'fast_period': (10, 30, 5),      # (最小, 最大, 步长)
        'slow_period': (40, 80, 10),
        'signal_threshold': (0.5, 0.9, 0.1)
    }
    
    logger.info("参数优化范围:")
    for param, (min_val, max_val, step) in param_ranges.items():
        logger.info(f"  {param}: {min_val} - {max_val} (步长: {step})")
    
    # 2. 创建优化器
    from src.backtest.parameter_optimizer import ParameterOptimizer
    
    optimizer = ParameterOptimizer(
        strategy_class=TrendStrategy,
        symbol="EURUSD",
        timeframe="H1",
        start_date=datetime.now() - timedelta(days=90),
        end_date=datetime.now()
    )
    
    # 3. 运行优化
    logger.info("\n开始参数优化（这可能需要几分钟）...")
    best_params, best_score = optimizer.optimize(
        param_ranges=param_ranges,
        objective='sharpe_ratio',  # 优化目标
        n_trials=50  # 尝试次数
    )
    
    # 4. 显示结果
    logger.info("\n" + "=" * 60)
    logger.info("优化结果:")
    logger.info("=" * 60)
    logger.info(f"最佳参数:")
    for param, value in best_params.items():
        logger.info(f"  {param}: {value}")
    logger.info(f"\n最佳得分 (夏普比率): {best_score:.2f}")
    logger.info("=" * 60)
    
    # 5. 使用最优参数回测
    logger.info("\n使用最优参数进行回测...")
    strategy = TrendStrategy(**best_params)
    
    engine = BacktestEngine(
        initial_balance=10000,
        start_date=datetime.now() - timedelta(days=90),
        end_date=datetime.now(),
        symbol="EURUSD",
        timeframe="H1"
    )
    engine.add_strategy(strategy)
    
    results = engine.run()
    logger.info(f"优化后收益: {results['total_return']:.2%}")
    logger.info(f"优化后夏普比率: {results['sharpe_ratio']:.2f}")
    
    return best_params, best_score


def run_monte_carlo_simulation():
    """运行蒙特卡洛模拟"""
    logger.info("\n=== 蒙特卡洛模拟示例 ===")
    
    from src.backtest.monte_carlo_simulator import MonteCarloSimulator
    
    # 1. 创建模拟器
    simulator = MonteCarloSimulator(
        initial_balance=10000,
        n_simulations=1000
    )
    
    # 2. 添加策略
    strategy = TrendStrategy()
    simulator.add_strategy(strategy)
    
    # 3. 运行模拟
    logger.info("运行1000次蒙特卡洛模拟...")
    results = simulator.run(
        symbol="EURUSD",
        timeframe="H1",
        days=90
    )
    
    # 4. 显示结果
    logger.info("\n" + "=" * 60)
    logger.info("蒙特卡洛模拟结果:")
    logger.info("=" * 60)
    logger.info(f"平均收益: {results['mean_return']:.2%}")
    logger.info(f"收益标准差: {results['std_return']:.2%}")
    logger.info(f"最好情况: {results['best_case']:.2%}")
    logger.info(f"最坏情况: {results['worst_case']:.2%}")
    logger.info(f"95%置信区间: [{results['ci_lower']:.2%}, {results['ci_upper']:.2%}]")
    logger.info(f"盈利概率: {results['profit_probability']:.2%}")
    logger.info(f"破产概率: {results['ruin_probability']:.2%}")
    logger.info("=" * 60)
    
    # 5. 生成分布图
    logger.info("\n生成收益分布图...")
    simulator.plot_distribution()
    logger.info("分布图已保存")
    
    return results


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("回测示例程序")
    logger.info("=" * 60)
    
    # 选择要运行的示例
    print("\n请选择要运行的示例:")
    print("1. 简单回测")
    print("2. 参数优化")
    print("3. 蒙特卡洛模拟")
    print("4. 运行全部")
    
    choice = input("\n请输入选项 (1-4): ").strip()
    
    if choice == "1":
        run_simple_backtest()
    elif choice == "2":
        run_parameter_optimization()
    elif choice == "3":
        run_monte_carlo_simulation()
    elif choice == "4":
        run_simple_backtest()
        run_parameter_optimization()
        run_monte_carlo_simulation()
    else:
        logger.error("无效选项")
        return
    
    logger.info("\n=== 示例完成 ===")
    logger.info("\n提示:")
    logger.info("1. 回测结果仅供参考，不代表实盘表现")
    logger.info("2. 建议在多个时间段和品种上测试")
    logger.info("3. 注意过拟合风险")
    logger.info("4. 实盘前务必在模拟账户测试")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n程序被用户中断")
    except Exception as e:
        logger.error(f"程序异常: {e}", exc_info=True)
