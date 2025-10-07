# 回测引擎快速使用指南

## 快速开始

### 1. 基础回测（5分钟上手）

```python
from src.strategies.backtest import BacktestConfig, BacktestEngine
from src.strategies.base_strategies import TrendFollowingStrategy, StrategyConfig, StrategyType

# 创建配置
config = BacktestConfig(
    initial_balance=10000.0,  # 初始资金
    leverage=100,              # 杠杆
    spread=0.0001,            # 点差
    slippage=0.0001           # 滑点
)

# 创建引擎
engine = BacktestEngine(config)

# 创建策略
strategy = TrendFollowingStrategy(StrategyConfig(
    name="My Strategy",
    strategy_type=StrategyType.TREND_FOLLOWING,
    risk_per_trade=0.02  # 每笔交易风险2%
))

# 运行回测（假设你已有market_data_list）
result = engine.run_backtest(strategy, market_data_list)

# 查看结果
print(f"收益率: {result.total_return*100:.2f}%")
print(f"胜率: {result.win_rate*100:.2f}%")
```

### 2. 多策略对比

```python
# 创建多个策略
strategies = [
    TrendFollowingStrategy(config1),
    MeanReversionStrategy(config2),
    BreakoutStrategy(config3)
]

# 批量回测
results = engine.run_multiple_strategies(strategies, market_data_list)

# 生成对比表格
comparison = engine.compare_strategies(results)
print(comparison)
```

### 3. 并行加速

```python
# 开启并行模式（适合多个策略）
results = engine.run_multiple_strategies(
    strategies, 
    market_data_list,
    parallel=True  # 启用并行
)
```

## 可视化

### 基础图表

```python
# 生成标准图表（权益曲线、回撤、交易分布）
engine.visualize_results(result, save_path='backtest.png')
```

### 详细分析

```python
# 生成详细分析图表（6个子图）
engine.visualize_detailed_analysis(result, save_path='detailed.png')
```

### 策略对比

```python
# 生成策略对比图表
engine.visualize_strategy_comparison(results, save_path='comparison.png')
```

## 导出结果

### JSON格式

```python
engine.export_results(result, 'result.json', format='json')
```

### CSV格式

```python
# 生成两个文件：result_trades.csv 和 result_summary.csv
engine.export_results(result, 'result.csv', format='csv')
```

### Excel格式

```python
# 生成多工作表Excel文件
engine.export_results(result, 'result.xlsx', format='excel')
```

### HTML报告

```python
# 生成美观的HTML报告
engine.export_results(result, 'report.html', format='html')
```

## 常用配置

### 保守配置

```python
config = BacktestConfig(
    initial_balance=10000.0,
    leverage=50,           # 较低杠杆
    spread=0.0002,         # 较大点差
    commission=0.0001,     # 考虑手续费
    slippage=0.0002        # 较大滑点
)
```

### 激进配置

```python
config = BacktestConfig(
    initial_balance=10000.0,
    leverage=200,          # 高杠杆
    spread=0.00005,        # 小点差
    commission=0.0,        # 不考虑手续费
    slippage=0.00005       # 小滑点
)
```

## 关键指标说明

| 指标 | 说明 | 好的值 |
|------|------|--------|
| 总收益率 | 整体盈利百分比 | > 10% |
| 胜率 | 盈利交易占比 | > 50% |
| 盈利因子 | 总盈利/总亏损 | > 1.5 |
| 夏普比率 | 风险调整后收益 | > 1.0 |
| 索提诺比率 | 下行风险调整收益 | > 1.5 |
| 最大回撤 | 最大资金回撤 | < 20% |

## 常见问题

### Q: 回测没有交易怎么办？
A: 检查策略参数和市场数据，确保策略能够生成信号。

### Q: 并行回测反而更慢？
A: 在Windows上进程启动有开销，策略数量少时顺序执行可能更快。

### Q: 如何获取市场数据？
A: 使用DataPipeline从MT5获取历史数据：
```python
from src.data.pipeline import DataPipeline
pipeline = DataPipeline()
data = pipeline.get_historical_data("EURUSD", start_date, end_date)
```

### Q: 可以回测实盘数据吗？
A: 可以，只需提供实盘的MarketData列表即可。

## 最佳实践

1. **先小后大**: 先用少量数据测试，确认无误后再用完整数据
2. **保守估计**: 点差和滑点设置要保守，接近实盘情况
3. **多策略对比**: 同时测试多个策略，选择最优
4. **定期回测**: 策略参数变化后重新回测验证
5. **保存结果**: 导出结果便于后续分析和对比

## 完整示例

```python
# 完整的回测流程
from src.strategies.backtest import BacktestConfig, BacktestEngine
from src.strategies.base_strategies import *
from src.data.pipeline import DataPipeline

# 1. 获取数据
pipeline = DataPipeline()
data = pipeline.get_historical_data(
    symbol="EURUSD",
    start=datetime(2023, 1, 1),
    end=datetime(2023, 12, 31)
)

# 2. 配置回测
config = BacktestConfig(
    initial_balance=10000.0,
    leverage=100,
    spread=0.0001,
    slippage=0.0001
)

# 3. 创建引擎和策略
engine = BacktestEngine(config)
strategy = TrendFollowingStrategy(StrategyConfig(
    name="Trend Strategy",
    strategy_type=StrategyType.TREND_FOLLOWING,
    risk_per_trade=0.02
))

# 4. 运行回测
result = engine.run_backtest(strategy, data)

# 5. 分析结果
print(f"收益率: {result.total_return*100:.2f}%")
print(f"胜率: {result.win_rate*100:.2f}%")
print(f"夏普比率: {result.sharpe_ratio:.2f}")
print(f"最大回撤: {result.max_drawdown_percent*100:.2f}%")

# 6. 可视化
engine.visualize_detailed_analysis(result, 'analysis.png')

# 7. 导出
engine.export_results(result, 'result.json', 'json')
engine.export_results(result, 'report.html', 'html')
```

## 更多资源

- 完整演示: `examples/backtest_engine_demo.py`
- 测试用例: `tests/test_backtest.py`
- 详细文档: `TASK_10.1_BACKTEST_ENGINE_SUMMARY.md`
