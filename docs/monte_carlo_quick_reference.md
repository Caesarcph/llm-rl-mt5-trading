# 蒙特卡洛模拟器快速参考

## 快速开始

```python
from src.strategies.monte_carlo import MonteCarloSimulator, MonteCarloConfig

# 1. 创建模拟器
config = MonteCarloConfig(n_simulations=1000, parallel=True)
simulator = MonteCarloSimulator(config)

# 2. 运行模拟
result = simulator.simulate_from_trades(trades, initial_balance=10000.0)

# 3. 查看结果
print(f"平均收益: {result.mean_return:.2%}")
print(f"VaR: {result.var:.2%}")
print(f"盈利概率: {result.prob_profit:.2%}")
```

## 配置选项

```python
MonteCarloConfig(
    n_simulations=1000,      # 模拟次数 (100-10000)
    confidence_level=0.95,   # 置信水平 (0.90-0.99)
    random_seed=42,          # 随机种子（可选）
    parallel=True,           # 并行计算
    max_workers=4            # 工作进程数
)
```

## 主要方法

### 1. 基本模拟
```python
# 从交易列表
result = simulator.simulate_from_trades(trades, initial_balance=10000.0)

# 从回测结果
result = simulator.simulate_from_backtest(backtest_result)
```

### 2. 压力测试
```python
# 默认场景
results = simulator.stress_test(trades, initial_balance=10000.0)

# 自定义场景
from src.strategies.monte_carlo import StressTestScenario

scenario = StressTestScenario(
    name="自定义",
    description="描述",
    return_shock=-0.30,
    volatility_multiplier=1.5,
    win_rate_adjustment=-0.10,
    drawdown_multiplier=2.0
)
results = simulator.stress_test(trades, scenarios=[scenario])
```

### 3. 情景分析
```python
# 默认情景
results = simulator.scenario_analysis(trades, initial_balance=10000.0)

# 自定义情景
scenarios = {
    '牛市': {'return_multiplier': 1.5, 'volatility_multiplier': 0.8},
    '熊市': {'return_multiplier': 0.5, 'volatility_multiplier': 1.5}
}
results = simulator.scenario_analysis(trades, scenarios=scenarios)
```

### 4. 风险指标
```python
metrics = simulator.calculate_risk_metrics(trades, initial_balance=10000.0)

print(f"VaR (1日): {metrics.var_1d:.2%}")
print(f"夏普比率: {metrics.sharpe_ratio:.2f}")
print(f"胜率: {metrics.win_rate:.2%}")
```

### 5. 结果导出
```python
# JSON格式
simulator.export_results(result, 'result.json', format='json')

# CSV格式
simulator.export_results(result, 'result.csv', format='csv')
```

## 结果对象

### MonteCarloResult
```python
result.mean_return          # 平均收益率
result.median_return        # 中位数收益率
result.std_return           # 标准差
result.var                  # VaR
result.cvar                 # CVaR
result.max_drawdown_worst   # 最大回撤
result.prob_profit          # 盈利概率
result.prob_ruin            # 破产概率
result.percentiles          # 分位数字典
```

### StressTestResult
```python
result.scenario_name        # 场景名称
result.original_return      # 原始收益
result.stressed_return      # 压力收益
result.return_impact        # 收益影响
result.survival_probability # 生存概率
```

### RiskMetrics
```python
metrics.var_1d              # 1日VaR
metrics.var_5d              # 5日VaR
metrics.max_drawdown        # 最大回撤
metrics.sharpe_ratio        # 夏普比率
metrics.sortino_ratio       # 索提诺比率
metrics.calmar_ratio        # 卡玛比率
metrics.win_rate            # 胜率
metrics.profit_factor       # 盈利因子
```

## 常用模式

### 策略风险评估
```python
result = simulator.simulate_from_trades(trades, initial_balance=10000.0)

if result.prob_ruin > 0.01:
    print("警告: 破产概率过高")
elif result.var < -0.10:
    print("警告: VaR过大")
else:
    print("风险可接受")
```

### 多策略对比
```python
strategies = {'策略A': trades_a, '策略B': trades_b}
results = {}

for name, trades in strategies.items():
    results[name] = simulator.simulate_from_trades(trades)

# 按风险调整收益排序
best = max(results.items(), key=lambda x: x[1].mean_return / x[1].std_return)
print(f"最佳策略: {best[0]}")
```

### 压力测试报告
```python
stress_results = simulator.stress_test(trades)

for result in stress_results:
    print(f"{result.scenario_name}:")
    print(f"  收益影响: {result.return_impact:.2%}")
    print(f"  生存概率: {result.survival_probability:.2%}")
```

## 性能提示

### 快速评估
```python
config = MonteCarloConfig(n_simulations=100, parallel=False)
```

### 精确分析
```python
config = MonteCarloConfig(n_simulations=10000, parallel=True, max_workers=8)
```

### 可重复结果
```python
config = MonteCarloConfig(random_seed=42)
```

## 关键指标解释

| 指标 | 含义 | 好的值 |
|------|------|--------|
| VaR | 最大可能损失 | > -5% |
| CVaR | 尾部平均损失 | > -10% |
| 夏普比率 | 风险调整收益 | > 1.0 |
| 盈利概率 | 盈利的可能性 | > 60% |
| 破产概率 | 账户破产风险 | < 1% |
| 最大回撤 | 最大损失幅度 | < 20% |

## 常见问题

**Q: 需要多少笔交易？**
A: 最少30笔，建议100+笔

**Q: 模拟多少次合适？**
A: 快速评估100-500次，精确分析1000-10000次

**Q: 如何加速模拟？**
A: 启用并行计算，增加max_workers

**Q: 结果不稳定怎么办？**
A: 设置random_seed或增加模拟次数

## 示例代码

完整示例: `examples/monte_carlo_demo.py`
单元测试: `tests/test_monte_carlo.py`
详细文档: `docs/monte_carlo_simulator_guide.md`
