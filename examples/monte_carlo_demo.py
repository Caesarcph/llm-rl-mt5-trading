"""
蒙特卡洛模拟器演示脚本
展示MonteCarloSimulator的完整功能，包括：
- 风险分析
- 压力测试
- 情景分析
- 风险指标计算
- 结果可视化和导出
"""

import sys
import os
import numpy as np
from datetime import datetime, timedelta

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.models import Trade, TradeType
from src.strategies.monte_carlo import (
    MonteCarloSimulator,
    MonteCarloConfig,
    StressTestScenario
)


def create_sample_trades(n_trades=100, win_rate=0.60, avg_win=100, avg_loss=-80):
    """
    创建示例交易数据
    
    Args:
        n_trades: 交易数量
        win_rate: 胜率
        avg_win: 平均盈利
        avg_loss: 平均亏损
    """
    print(f"生成{n_trades}笔交易数据 (胜率: {win_rate:.0%})...")
    
    trades = []
    np.random.seed(42)
    
    for i in range(n_trades):
        # 根据胜率决定盈亏
        if np.random.random() < win_rate:
            profit = np.random.normal(avg_win, avg_win * 0.3)
        else:
            profit = np.random.normal(avg_loss, abs(avg_loss) * 0.3)
        
        trade = Trade(
            trade_id=f"trade_{i:04d}",
            symbol="EURUSD",
            type=TradeType.BUY if i % 2 == 0 else TradeType.SELL,
            volume=0.1,
            open_price=1.1000 + np.random.normal(0, 0.001),
            close_price=1.1000 + np.random.normal(0, 0.001),
            profit=profit,
            commission=2.0,
            swap=0.5,
            open_time=datetime.now() - timedelta(hours=n_trades - i),
            close_time=datetime.now() - timedelta(hours=n_trades - i - 1),
            strategy_id="demo_strategy"
        )
        trades.append(trade)
    
    # 打印基本统计
    total_profit = sum(t.profit for t in trades)
    winning_trades = len([t for t in trades if t.profit > 0])
    actual_win_rate = winning_trades / n_trades
    
    print(f"总盈亏: ${total_profit:.2f}")
    print(f"实际胜率: {actual_win_rate:.2%}")
    print()
    
    return trades


def demo_basic_monte_carlo():
    """演示基本蒙特卡洛模拟"""
    print("="*80)
    print("演示1: 基本蒙特卡洛风险分析")
    print("="*80)
    
    # 创建交易数据
    trades = create_sample_trades(n_trades=100, win_rate=0.60)
    
    # 配置模拟器
    config = MonteCarloConfig(
        n_simulations=1000,
        confidence_level=0.95,
        random_seed=42,
        parallel=True,
        max_workers=4
    )
    
    simulator = MonteCarloSimulator(config)
    
    # 运行模拟
    print("运行1000次蒙特卡洛模拟...")
    result = simulator.simulate_from_trades(trades, initial_balance=10000.0)
    
    # 打印结果
    print("\n蒙特卡洛模拟结果:")
    print(f"模拟次数: {result.n_simulations}")
    print(f"置信水平: {result.confidence_level:.0%}")
    print()
    
    print("收益分布:")
    print(f"  平均收益率: {result.mean_return:.2%}")
    print(f"  中位数收益率: {result.median_return:.2%}")
    print(f"  标准差: {result.std_return:.2%}")
    print(f"  最小收益率: {result.min_return:.2%}")
    print(f"  最大收益率: {result.max_return:.2%}")
    print()
    
    print("风险指标:")
    print(f"  VaR (95%): {result.var:.2%}")
    print(f"  CVaR (95%): {result.cvar:.2%}")
    print(f"  平均最大回撤: {result.max_drawdown_mean:.2%}")
    print(f"  最坏最大回撤: {result.max_drawdown_worst:.2%}")
    print()
    
    print("概率分析:")
    print(f"  盈利概率: {result.prob_profit:.2%}")
    print(f"  亏损概率: {result.prob_loss:.2%}")
    print(f"  破产概率: {result.prob_ruin:.2%}")
    print()
    
    print("收益分位数:")
    for percentile, value in sorted(result.percentiles.items()):
        print(f"  {percentile:.0%}: {value:.2%}")
    
    # 导出结果
    print("\n导出结果...")
    os.makedirs('logs/reports', exist_ok=True)
    simulator.export_results(result, 'logs/reports/monte_carlo_basic.json', format='json')
    simulator.export_results(result, 'logs/reports/monte_carlo_basic.csv', format='csv')
    print("结果已导出到: logs/reports/")
    
    return result


def demo_stress_testing():
    """演示压力测试"""
    print("\n" + "="*80)
    print("演示2: 压力测试")
    print("="*80)
    
    # 创建交易数据
    trades = create_sample_trades(n_trades=100, win_rate=0.65)
    
    # 配置模拟器
    config = MonteCarloConfig(
        n_simulations=500,
        random_seed=42,
        parallel=True
    )
    
    simulator = MonteCarloSimulator(config)
    
    # 运行压力测试（使用默认场景）
    print("运行压力测试（5个默认场景）...")
    results = simulator.stress_test(trades, initial_balance=10000.0)
    
    # 打印结果
    print("\n压力测试结果:")
    print(f"{'场景名称':<15} {'原始收益':<12} {'压力收益':<12} {'收益影响':<12} {'生存概率':<12}")
    print("-" * 80)
    
    for result in results:
        print(f"{result.scenario_name:<15} "
              f"{result.original_return:>10.2%}  "
              f"{result.stressed_return:>10.2%}  "
              f"{result.return_impact:>10.2%}  "
              f"{result.survival_probability:>10.2%}")
    
    # 测试自定义场景
    print("\n\n运行自定义压力场景...")
    custom_scenarios = [
        StressTestScenario(
            name="黑天鹅事件",
            description="极端市场事件",
            return_shock=-0.50,
            volatility_multiplier=3.0,
            win_rate_adjustment=-0.25,
            drawdown_multiplier=3.0
        ),
        StressTestScenario(
            name="策略失效",
            description="策略完全失效",
            return_shock=-0.30,
            volatility_multiplier=1.0,
            win_rate_adjustment=-0.40,
            drawdown_multiplier=2.0
        )
    ]
    
    custom_results = simulator.stress_test(trades, scenarios=custom_scenarios, initial_balance=10000.0)
    
    print("\n自定义场景结果:")
    for result in custom_results:
        print(f"\n{result.scenario_name}:")
        print(f"  收益影响: {result.return_impact:.2%}")
        print(f"  回撤影响: {result.drawdown_impact:.2%}")
        print(f"  生存概率: {result.survival_probability:.2%}")


def demo_scenario_analysis():
    """演示情景分析"""
    print("\n" + "="*80)
    print("演示3: 情景分析")
    print("="*80)
    
    # 创建交易数据
    trades = create_sample_trades(n_trades=100, win_rate=0.60)
    
    # 配置模拟器
    config = MonteCarloConfig(
        n_simulations=500,
        random_seed=42,
        parallel=True
    )
    
    simulator = MonteCarloSimulator(config)
    
    # 运行情景分析（使用默认场景）
    print("运行情景分析（3个默认场景）...")
    results = simulator.scenario_analysis(trades, initial_balance=10000.0)
    
    # 打印结果对比
    print("\n情景分析结果对比:")
    print(f"{'场景':<12} {'平均收益':<12} {'VaR(95%)':<12} {'最大回撤':<12} {'盈利概率':<12}")
    print("-" * 80)
    
    for scenario_name, result in results.items():
        print(f"{scenario_name:<12} "
              f"{result.mean_return:>10.2%}  "
              f"{result.var:>10.2%}  "
              f"{result.max_drawdown_worst:>10.2%}  "
              f"{result.prob_profit:>10.2%}")
    
    # 自定义情景分析
    print("\n\n运行自定义情景分析...")
    custom_scenarios = {
        '牛市': {'return_multiplier': 1.5, 'volatility_multiplier': 0.8},
        '熊市': {'return_multiplier': 0.5, 'volatility_multiplier': 1.5},
        '震荡市': {'return_multiplier': 0.9, 'volatility_multiplier': 1.8},
        '极端波动': {'return_multiplier': 1.0, 'volatility_multiplier': 2.5}
    }
    
    custom_results = simulator.scenario_analysis(
        trades,
        initial_balance=10000.0,
        scenarios=custom_scenarios
    )
    
    print("\n自定义情景结果:")
    for scenario_name, result in custom_results.items():
        print(f"\n{scenario_name}:")
        print(f"  平均收益: {result.mean_return:.2%}")
        print(f"  标准差: {result.std_return:.2%}")
        print(f"  VaR: {result.var:.2%}")
        print(f"  盈利概率: {result.prob_profit:.2%}")


def demo_risk_metrics():
    """演示风险指标计算"""
    print("\n" + "="*80)
    print("演示4: 综合风险指标计算")
    print("="*80)
    
    # 创建不同风险特征的交易数据
    scenarios = {
        '保守策略': create_sample_trades(n_trades=100, win_rate=0.70, avg_win=50, avg_loss=-40),
        '激进策略': create_sample_trades(n_trades=100, win_rate=0.50, avg_win=200, avg_loss=-150),
        '平衡策略': create_sample_trades(n_trades=100, win_rate=0.60, avg_win=100, avg_loss=-80)
    }
    
    # 配置模拟器
    config = MonteCarloConfig(
        n_simulations=500,
        random_seed=42,
        parallel=True
    )
    
    simulator = MonteCarloSimulator(config)
    
    # 计算各策略的风险指标
    print("计算各策略风险指标...")
    print(f"\n{'策略':<12} {'VaR(1d)':<10} {'VaR(5d)':<10} {'夏普':<8} {'索提诺':<8} {'卡玛':<8} {'胜率':<8} {'盈利因子':<10}")
    print("-" * 100)
    
    for strategy_name, trades in scenarios.items():
        metrics = simulator.calculate_risk_metrics(trades, initial_balance=10000.0)
        
        print(f"{strategy_name:<12} "
              f"{metrics.var_1d:>8.2%}  "
              f"{metrics.var_5d:>8.2%}  "
              f"{metrics.sharpe_ratio:>6.2f}  "
              f"{metrics.sortino_ratio:>6.2f}  "
              f"{metrics.calmar_ratio:>6.2f}  "
              f"{metrics.win_rate:>6.2%}  "
              f"{metrics.profit_factor:>8.2f}")
        
        # 检查风险可接受性
        is_acceptable = metrics.is_risk_acceptable(max_var=0.05, max_dd=0.20)
        print(f"  风险可接受性: {'✓ 通过' if is_acceptable else '✗ 不通过'}")


def demo_comparison():
    """演示策略对比分析"""
    print("\n" + "="*80)
    print("演示5: 多策略蒙特卡洛对比")
    print("="*80)
    
    # 创建三个不同的策略
    strategies = {
        '趋势策略': create_sample_trades(n_trades=80, win_rate=0.55, avg_win=120, avg_loss=-90),
        '震荡策略': create_sample_trades(n_trades=120, win_rate=0.65, avg_win=60, avg_loss=-50),
        '突破策略': create_sample_trades(n_trades=60, win_rate=0.50, avg_win=180, avg_loss=-120)
    }
    
    # 配置模拟器
    config = MonteCarloConfig(
        n_simulations=1000,
        random_seed=42,
        parallel=True
    )
    
    simulator = MonteCarloSimulator(config)
    
    # 对每个策略运行蒙特卡洛模拟
    print("运行多策略蒙特卡洛模拟...")
    results = {}
    
    for strategy_name, trades in strategies.items():
        result = simulator.simulate_from_trades(trades, initial_balance=10000.0)
        results[strategy_name] = result
    
    # 打印对比结果
    print("\n策略对比结果:")
    print(f"{'策略':<12} {'平均收益':<12} {'中位收益':<12} {'标准差':<10} {'VaR':<10} {'盈利概率':<12}")
    print("-" * 90)
    
    for strategy_name, result in results.items():
        print(f"{strategy_name:<12} "
              f"{result.mean_return:>10.2%}  "
              f"{result.median_return:>10.2%}  "
              f"{result.std_return:>8.2%}  "
              f"{result.var:>8.2%}  "
              f"{result.prob_profit:>10.2%}")
    
    # 找出最佳策略
    print("\n策略排名:")
    
    # 按平均收益排名
    sorted_by_return = sorted(results.items(), key=lambda x: x[1].mean_return, reverse=True)
    print("\n按平均收益:")
    for i, (name, result) in enumerate(sorted_by_return, 1):
        print(f"  {i}. {name}: {result.mean_return:.2%}")
    
    # 按风险调整收益排名（夏普比率的简化版本）
    sorted_by_sharpe = sorted(
        results.items(),
        key=lambda x: x[1].mean_return / x[1].std_return if x[1].std_return > 0 else 0,
        reverse=True
    )
    print("\n按风险调整收益:")
    for i, (name, result) in enumerate(sorted_by_sharpe, 1):
        sharpe = result.mean_return / result.std_return if result.std_return > 0 else 0
        print(f"  {i}. {name}: {sharpe:.2f}")
    
    # 按盈利概率排名
    sorted_by_prob = sorted(results.items(), key=lambda x: x[1].prob_profit, reverse=True)
    print("\n按盈利概率:")
    for i, (name, result) in enumerate(sorted_by_prob, 1):
        print(f"  {i}. {name}: {result.prob_profit:.2%}")


def main():
    """主函数"""
    print("="*80)
    print("蒙特卡洛模拟器完整功能演示")
    print("="*80)
    print()
    
    # 确保输出目录存在
    os.makedirs('logs/reports', exist_ok=True)
    
    try:
        # 演示1: 基本蒙特卡洛模拟
        demo_basic_monte_carlo()
        
        # 演示2: 压力测试
        demo_stress_testing()
        
        # 演示3: 情景分析
        demo_scenario_analysis()
        
        # 演示4: 风险指标计算
        demo_risk_metrics()
        
        # 演示5: 多策略对比
        demo_comparison()
        
        print("\n" + "="*80)
        print("所有演示完成！")
        print("="*80)
        print("\n查看生成的报告:")
        print("- logs/reports/monte_carlo_basic.json")
        print("- logs/reports/monte_carlo_basic.csv")
        
    except Exception as e:
        print(f"\n演示过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
