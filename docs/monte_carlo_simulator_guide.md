# 蒙特卡洛模拟器使用指南

## 概述

蒙特卡洛模拟器是一个强大的风险分析工具，通过对历史交易数据进行大量随机重采样模拟，评估交易策略在不同市场条件下的表现和风险特征。

## 主要功能

### 1. 风险分析
- **VaR (Value at Risk)**: 在给定置信水平下的最大可能损失
- **CVaR (Conditional VaR)**: 超过VaR阈值的平均损失
- **最大回撤分析**: 评估策略的最坏情况
- **收益分布**: 完整的收益概率分布

### 2. 压力测试
- **预定义场景**: 温和衰退、严重衰退、市场崩盘等
- **自定义场景**: 根据需要定义特定的压力条件
- **生存概率**: 评估策略在极端条件下的存活能力

### 3. 情景分析
- **多场景对比**: 牛市、熊市、震荡市等不同市场环境
- **参数敏感性**: 评估收益和波动率变化的影响
- **概率评估**: 各种情景下的盈利概率

### 4. 风险指标计算
- 夏普比率 (Sharpe Ratio)
- 索提诺比率 (Sortino Ratio)
- 卡玛比率 (Calmar Ratio)
- 盈利因子 (Profit Factor)
- 胜率 (Win Rate)

## 快速开始

### 基本使用

```python
from src.strategies.monte_carlo import MonteCarloSimulator, MonteCarloConfig
from src.core.models import Trade

# 1. 配置模拟器
config = MonteCarloConfig(
    n_simulations=1000,      # 模拟次数
    confidence_level=0.95,   # 置信水平
    random_seed=42,          # 随机种子（可选）
    parallel=True,           # 并行计算
    max_workers=4            # 工作进程数
)

simulator = MonteCarloSimulator(config)

# 2. 准备交易数据
trades = [...]  # 历史交易列表

# 3. 运行模拟
result = simulator.simulate_from_trades(
    trades,
    initial_balance=10000.0
)

# 4. 查看结果
print(f"平均收益率: {result.mean_return:.2%}")
print(f"VaR (95%): {result.var:.2%}")
print(f"盈利概率: {result.prob_profit:.2%}")
```

## 详细功能说明

### 1. 蒙特卡洛风险分析

蒙特卡洛模拟通过Bootstrap重采样方法，从历史交易中随机抽取样本，生成大量可能的未来收益路径。

```python
# 运行基本模拟
result = simulator.simulate_from_trades(trades, initial_balance=10000.0)

# 访问结果
print("收益统计:")
print(f"  平均收益: {result.mean_return:.2%}")
print(f"  中位数收益: {result.median_return:.2%}")
print(f"  标准差: {result.std_return:.2%}")

print("\n风险指标:")
print(f"  VaR (95%): {result.var:.2%}")
print(f"  CVaR (95%): {result.cvar:.2%}")
print(f"  最大回撤: {result.max_drawdown_worst:.2%}")

print("\n概率分析:")
print(f"  盈利概率: {result.prob_profit:.2%}")
print(f"  破产概率: {result.prob_ruin:.2%}")

# 查看分位数
for percentile, value in result.percentiles.items():
    print(f"  {percentile:.0%}分位数: {value:.2%}")
```

**关键指标解释:**

- **VaR (Value at Risk)**: 在95%置信水平下，最多可能损失的金额。例如，VaR=-5%表示有95%的概率损失不会超过5%。

- **CVaR (Conditional VaR)**: 当损失超过VaR时的平均损失，也称为Expected Shortfall。它提供了尾部风险的更完整视图。

- **破产概率**: 账户余额跌至初始余额50%以下的概率。

### 2. 压力测试

压力测试评估策略在极端市场条件下的表现。

```python
# 使用默认压力场景
results = simulator.stress_test(trades, initial_balance=10000.0)

for result in results:
    print(f"\n{result.scenario_name}:")
    print(f"  收益影响: {result.return_impact:.2%}")
    print(f"  回撤影响: {result.drawdown_impact:.2%}")
    print(f"  生存概率: {result.survival_probability:.2%}")
```

**默认压力场景:**

1. **温和衰退**: 收益下降10%，波动率上升20%
2. **严重衰退**: 收益下降25%，波动率上升50%
3. **市场崩盘**: 收益下降40%，波动率上升100%
4. **高波动**: 收益不变，波动率上升100%
5. **胜率下降**: 策略有效性降低20%

**自定义压力场景:**

```python
from src.strategies.monte_carlo import StressTestScenario

custom_scenario = StressTestScenario(
    name="自定义场景",
    description="特定市场条件",
    return_shock=-0.30,              # 收益冲击-30%
    volatility_multiplier=1.5,       # 波动率增加50%
    win_rate_adjustment=-0.10,       # 胜率降低10%
    drawdown_multiplier=2.0          # 回撤增加100%
)

results = simulator.stress_test(
    trades,
    scenarios=[custom_scenario],
    initial_balance=10000.0
)
```

### 3. 情景分析

情景分析评估策略在不同市场环境下的表现。

```python
# 使用默认情景
results = simulator.scenario_analysis(trades, initial_balance=10000.0)

for scenario_name, result in results.items():
    print(f"\n{scenario_name}:")
    print(f"  平均收益: {result.mean_return:.2%}")
    print(f"  VaR: {result.var:.2%}")
    print(f"  盈利概率: {result.prob_profit:.2%}")
```

**默认情景:**

- **基准情景**: 历史表现的正常重现
- **乐观情景**: 收益增加20%，波动率降低20%
- **悲观情景**: 收益降低30%，波动率增加30%

**自定义情景:**

```python
custom_scenarios = {
    '牛市': {
        'return_multiplier': 1.5,      # 收益增加50%
        'volatility_multiplier': 0.8   # 波动率降低20%
    },
    '熊市': {
        'return_multiplier': 0.5,      # 收益降低50%
        'volatility_multiplier': 1.5   # 波动率增加50%
    },
    '震荡市': {
        'return_multiplier': 0.9,      # 收益略降
        'volatility_multiplier': 1.8   # 波动率大幅增加
    }
}

results = simulator.scenario_analysis(
    trades,
    initial_balance=10000.0,
    scenarios=custom_scenarios
)
```

### 4. 风险指标计算

计算综合风险指标，用于策略评估和对比。

```python
metrics = simulator.calculate_risk_metrics(trades, initial_balance=10000.0)

print("风险指标:")
print(f"  1日VaR: {metrics.var_1d:.2%}")
print(f"  5日VaR: {metrics.var_5d:.2%}")
print(f"  最大回撤: {metrics.max_drawdown:.2%}")
print(f"  夏普比率: {metrics.sharpe_ratio:.2f}")
print(f"  索提诺比率: {metrics.sortino_ratio:.2f}")
print(f"  卡玛比率: {metrics.calmar_ratio:.2f}")
print(f"  胜率: {metrics.win_rate:.2%}")
print(f"  盈利因子: {metrics.profit_factor:.2f}")

# 检查风险可接受性
is_acceptable = metrics.is_risk_acceptable(
    max_var=0.05,    # 最大VaR 5%
    max_dd=0.20      # 最大回撤 20%
)
print(f"风险可接受: {is_acceptable}")
```

**风险指标解释:**

- **夏普比率**: 每单位风险的超额收益，越高越好（>1为良好）
- **索提诺比率**: 类似夏普比率，但只考虑下行风险
- **卡玛比率**: 年化收益与最大回撤的比率
- **盈利因子**: 总盈利/总亏损，>1表示盈利

## 高级用法

### 并行计算优化

对于大量模拟，使用并行计算可以显著提高速度：

```python
config = MonteCarloConfig(
    n_simulations=10000,
    parallel=True,
    max_workers=8  # 根据CPU核心数调整
)

simulator = MonteCarloSimulator(config)
```

### 结果导出

```python
# 导出为JSON
simulator.export_results(
    result,
    'reports/monte_carlo_result.json',
    format='json'
)

# 导出为CSV
simulator.export_results(
    result,
    'reports/monte_carlo_result.csv',
    format='csv'
)
```

### 从回测结果模拟

```python
from src.strategies.backtest import BacktestResult

# 假设已有回测结果
backtest_result = BacktestResult(...)

# 直接从回测结果运行蒙特卡洛模拟
mc_result = simulator.simulate_from_backtest(backtest_result)
```

## 实际应用场景

### 场景1: 策略风险评估

在部署新策略前，评估其风险特征：

```python
# 1. 运行回测获取历史交易
trades = run_backtest(strategy)

# 2. 蒙特卡洛模拟
config = MonteCarloConfig(n_simulations=1000)
simulator = MonteCarloSimulator(config)
result = simulator.simulate_from_trades(trades, initial_balance=10000.0)

# 3. 评估风险
if result.prob_ruin > 0.01:  # 破产概率>1%
    print("警告: 策略风险过高")
elif result.var < -0.10:  # VaR<-10%
    print("警告: 潜在损失过大")
else:
    print("策略风险可接受")
```

### 场景2: 多策略对比

对比不同策略的风险收益特征：

```python
strategies = {
    '策略A': trades_a,
    '策略B': trades_b,
    '策略C': trades_c
}

results = {}
for name, trades in strategies.items():
    result = simulator.simulate_from_trades(trades, initial_balance=10000.0)
    results[name] = result

# 按风险调整收益排序
sorted_strategies = sorted(
    results.items(),
    key=lambda x: x[1].mean_return / x[1].std_return,
    reverse=True
)

print("策略排名（按风险调整收益）:")
for i, (name, result) in enumerate(sorted_strategies, 1):
    sharpe = result.mean_return / result.std_return
    print(f"{i}. {name}: {sharpe:.2f}")
```

### 场景3: 资金管理决策

根据风险分析决定资金分配：

```python
# 计算风险指标
metrics = simulator.calculate_risk_metrics(trades, initial_balance=10000.0)

# 根据VaR决定仓位
if metrics.var_1d > -0.02:  # VaR > -2%
    position_size = 1.0  # 满仓
elif metrics.var_1d > -0.05:  # VaR > -5%
    position_size = 0.5  # 半仓
else:
    position_size = 0.25  # 1/4仓

print(f"建议仓位: {position_size:.0%}")
```

### 场景4: 压力测试报告

生成完整的压力测试报告：

```python
# 运行压力测试
stress_results = simulator.stress_test(trades, initial_balance=10000.0)

# 生成报告
print("压力测试报告")
print("="*80)

for result in stress_results:
    print(f"\n场景: {result.scenario_name}")
    print(f"  原始收益: {result.original_return:.2%}")
    print(f"  压力收益: {result.stressed_return:.2%}")
    print(f"  收益影响: {result.return_impact:.2%}")
    print(f"  回撤影响: {result.drawdown_impact:.2%}")
    print(f"  生存概率: {result.survival_probability:.2%}")
    
    # 风险评级
    if result.survival_probability < 0.5:
        risk_level = "极高风险"
    elif result.survival_probability < 0.8:
        risk_level = "高风险"
    elif result.survival_probability < 0.95:
        risk_level = "中等风险"
    else:
        risk_level = "低风险"
    
    print(f"  风险评级: {risk_level}")
```

## 最佳实践

### 1. 模拟次数选择

- **快速评估**: 100-500次模拟
- **常规分析**: 1000-5000次模拟
- **精确分析**: 10000+次模拟

### 2. 数据要求

- 最少30笔交易（建议100+笔）
- 交易数据应覆盖不同市场条件
- 确保数据质量和完整性

### 3. 结果解读

- 关注中位数而非平均值（更稳健）
- 同时考虑收益和风险指标
- 重视尾部风险（VaR, CVaR）
- 考虑破产概率

### 4. 性能优化

```python
# 对于大量模拟，使用并行计算
config = MonteCarloConfig(
    n_simulations=10000,
    parallel=True,
    max_workers=min(8, os.cpu_count())
)

# 设置随机种子以确保可重复性
config.random_seed = 42
```

## 注意事项

1. **历史数据局限性**: 蒙特卡洛模拟基于历史数据，无法预测未来的新情况

2. **假设条件**: 模拟假设交易之间相互独立，实际市场可能存在相关性

3. **极端事件**: 历史数据可能未包含极端事件，压力测试可以弥补这一不足

4. **过度拟合**: 避免过度依赖模拟结果，应结合其他分析方法

5. **计算资源**: 大量模拟需要较多计算资源，合理设置模拟次数

## 故障排除

### 问题1: 模拟速度慢

**解决方案:**
```python
# 启用并行计算
config.parallel = True
config.max_workers = 4

# 减少模拟次数
config.n_simulations = 500
```

### 问题2: 内存不足

**解决方案:**
```python
# 不保存所有权益曲线
# 或分批处理大量交易数据
```

### 问题3: 结果不稳定

**解决方案:**
```python
# 设置随机种子
config.random_seed = 42

# 增加模拟次数
config.n_simulations = 5000
```

## 参考资料

- [蒙特卡洛方法](https://en.wikipedia.org/wiki/Monte_Carlo_method)
- [Value at Risk (VaR)](https://en.wikipedia.org/wiki/Value_at_risk)
- [Bootstrap方法](https://en.wikipedia.org/wiki/Bootstrapping_(statistics))
- [风险管理最佳实践](https://www.risk.net/)

## 示例代码

完整的示例代码请参考：
- `examples/monte_carlo_demo.py` - 完整功能演示
- `tests/test_monte_carlo.py` - 单元测试示例
