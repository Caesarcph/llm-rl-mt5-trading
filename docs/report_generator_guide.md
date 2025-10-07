# 报告生成器使用指南

## 概述

报告生成器是一个综合性的交易报告系统，用于生成每日、周、月交易报告，策略分析和性能对比报告，以及风险分析和合规检查报告。

## 主要功能

### 1. 每日交易报告
- 账户摘要（余额、净值、盈亏）
- 交易统计（交易次数、胜率、盈利因子）
- 风险指标（回撤、连续亏损）
- 合规检查结果
- 建议和警告

### 2. 周/月交易报告
- 与每日报告类似，但统计周期更长
- 更全面的趋势分析
- 长期表现评估

### 3. 策略分析报告
- 多策略性能对比
- 策略排名（按净利润、胜率、夏普比率等）
- 策略优化建议
- 表现最佳/最差策略识别

### 4. 风险分析报告
- 账户风险指标（敞口、保证金使用率）
- 回撤分析（最大回撤、当前回撤、恢复因子）
- VaR风险指标（1日、5日、10日VaR）
- 持仓分析（按品种、多空分布）
- 风险建议和警告

### 5. 合规检查报告
- 合规规则检查（单笔风险、回撤限制、胜率要求等）
- 合规率统计
- 违规问题识别
- 合规改进建议

## 快速开始

### 基本使用

```python
from src.utils.report_generator import ReportGenerator
from src.strategies.performance_tracker import PerformanceTracker, PerformancePeriod
from src.agents.risk_manager import RiskManagerAgent

# 创建性能跟踪器和风险管理器
performance_tracker = PerformanceTracker()
risk_manager = RiskManagerAgent()

# 创建报告生成器
report_generator = ReportGenerator(
    performance_tracker=performance_tracker,
    risk_manager=risk_manager,
    output_dir="logs/reports"
)

# 生成每日报告
report = report_generator.generate_daily_report(account=account)
```

### 生成不同类型的报告

#### 1. 每日报告

```python
from datetime import datetime

report = report_generator.generate_daily_report(
    date=datetime.now(),
    account=account
)

print(f"报告ID: {report.report_id}")
print(f"净利润: ${report.trading_summary['net_profit']:.2f}")
print(f"胜率: {report.trading_summary['win_rate']:.2%}")
```

#### 2. 周报告

```python
from datetime import datetime, timedelta

# 生成本周报告
week_start = datetime.now() - timedelta(days=datetime.now().weekday())
report = report_generator.generate_weekly_report(
    week_start=week_start,
    account=account
)
```

#### 3. 月报告

```python
# 生成本月报告
month_start = datetime.now().replace(day=1)
report = report_generator.generate_monthly_report(
    month_start=month_start,
    account=account
)
```

#### 4. 策略分析报告

```python
report = report_generator.generate_strategy_analysis_report(
    strategy_names=["trend_following", "range_trading", "breakout"],
    period=PerformancePeriod.MONTHLY
)

# 查看策略排名
rankings = report.trading_summary['rankings']
print("按净利润排名:")
for rank in rankings['by_net_profit']:
    print(f"  {rank['strategy']}: ${rank['value']:.2f}")
```

#### 5. 风险分析报告

```python
report = report_generator.generate_risk_analysis_report(
    account=account,
    positions=positions,
    period=PerformancePeriod.MONTHLY
)

# 查看风险指标
print(f"总敞口: {report.risk_metrics['exposure_pct']:.2f}%")
print(f"最大回撤: {report.risk_metrics['max_drawdown']:.2%}")
print(f"VaR (1日): ${report.risk_metrics.get('var_1d', 0):.2f}")
```

#### 6. 合规检查报告

```python
report = report_generator.generate_compliance_report(
    account=account,
    period=PerformancePeriod.MONTHLY
)

# 查看合规摘要
summary = report.trading_summary['compliance_summary']
print(f"合规率: {summary['compliance_rate']:.2f}%")
print(f"未通过检查: {summary['failed_checks']}")

# 查看合规检查详情
for result in report.compliance_results:
    if not result.passed:
        print(f"未通过: {result.rule_name} - {result.message}")
```

## 报告结构

### TradingReport 对象

```python
@dataclass
class TradingReport:
    report_id: str                    # 报告ID
    report_type: ReportType           # 报告类型
    period_start: datetime            # 统计开始时间
    period_end: datetime              # 统计结束时间
    generated_at: datetime            # 生成时间
    
    account_summary: Dict             # 账户摘要
    trading_summary: Dict             # 交易统计
    strategy_performance: List        # 策略表现
    risk_metrics: Dict                # 风险指标
    compliance_results: List          # 合规检查结果
    recommendations: List[str]        # 建议
    warnings: List[str]               # 警告
```

### 账户摘要字段

```python
account_summary = {
    'account_id': str,           # 账户ID
    'balance': float,            # 余额
    'equity': float,             # 净值
    'margin': float,             # 已用保证金
    'free_margin': float,        # 可用保证金
    'margin_level': float,       # 保证金水平
    'currency': str,             # 货币
    'leverage': int,             # 杠杆
    'profit_loss': float,        # 盈亏金额
    'profit_loss_pct': float     # 盈亏百分比
}
```

### 交易统计字段

```python
trading_summary = {
    'total_trades': int,         # 总交易次数
    'winning_trades': int,       # 盈利交易次数
    'losing_trades': int,        # 亏损交易次数
    'total_profit': float,       # 总盈利
    'total_loss': float,         # 总亏损
    'net_profit': float,         # 净利润
    'win_rate': float,           # 胜率
    'profit_factor': float,      # 盈利因子
    'avg_profit_per_trade': float,  # 平均每笔盈利
    'largest_win': float,        # 最大盈利
    'largest_loss': float,       # 最大亏损
    'total_commission': float,   # 总佣金
    'total_swap': float          # 总隔夜利息
}
```

### 风险指标字段

```python
risk_metrics = {
    'max_drawdown': float,           # 最大回撤
    'current_drawdown': float,       # 当前回撤
    'drawdown_duration_days': int,   # 回撤持续天数
    'recovery_factor': float,        # 恢复因子
    'consecutive_losses': int,       # 连续亏损次数
    'circuit_breaker_active': bool,  # 熔断是否激活
    'trading_halted': bool,          # 交易是否暂停
    
    # 风险分析报告额外字段
    'total_exposure': float,         # 总敞口
    'exposure_pct': float,           # 敞口百分比
    'position_count': int,           # 持仓数量
    'unrealized_pnl': float,         # 未实现盈亏
    'margin_usage_pct': float,       # 保证金使用率
    'var_1d': float,                 # 1日VaR
    'var_5d': float,                 # 5日VaR
    'var_10d': float,                # 10日VaR
    'var_1d_pct': float              # 1日VaR百分比
}
```

## 合规规则

系统内置以下合规规则：

### 1. 单笔最大风险规则 (MaxRiskPerTradeRule)
- **限制**: 2%
- **严重程度**: CRITICAL
- **说明**: 单笔交易风险不得超过账户的2%

### 2. 日最大回撤规则 (MaxDailyDrawdownRule)
- **限制**: 5%
- **严重程度**: CRITICAL
- **说明**: 日回撤不得超过5%

### 3. 周最大回撤规则 (MaxWeeklyDrawdownRule)
- **限制**: 10%
- **严重程度**: WARNING
- **说明**: 周回撤不得超过10%

### 4. 月最大回撤规则 (MaxMonthlyDrawdownRule)
- **限制**: 15%
- **严重程度**: WARNING
- **说明**: 月回撤不得超过15%

### 5. 最低胜率规则 (MinWinRateRule)
- **限制**: 40%
- **严重程度**: WARNING
- **说明**: 胜率不得低于40%

### 6. 最大连续亏损规则 (MaxConsecutiveLossesRule)
- **限制**: 3次
- **严重程度**: WARNING
- **说明**: 连续亏损次数不得超过3次

## 报告格式

### 支持的格式

1. **JSON** (默认)
   - 结构化数据
   - 易于程序处理
   - 完整的数据保留

2. **Markdown**
   - 人类可读
   - 支持格式化
   - 适合文档展示

3. **Text**
   - 纯文本格式
   - 简单易读
   - 适合日志记录

### 保存报告

```python
from src.utils.report_generator import ReportFormat

# 保存为JSON（默认）
report_generator._save_report(report, ReportFormat.JSON)

# 保存为Markdown
report_generator._save_report(report, ReportFormat.MARKDOWN)

# 保存为文本
report_generator._save_report(report, ReportFormat.TEXT)
```

### 导出报告

```python
# 转换为字典
report_dict = report.to_dict()

# 转换为JSON字符串
json_str = report.to_json(indent=2)

# 保存到文件
with open('report.json', 'w', encoding='utf-8') as f:
    f.write(json_str)
```

## 高级用法

### 自定义合规规则

```python
from src.utils.report_generator import ComplianceRule

class CustomRule(ComplianceRule):
    def __init__(self):
        super().__init__(
            rule_id="custom_rule",
            rule_name="自定义规则",
            description="规则描述",
            threshold=0.05,
            severity="WARNING"
        )
    
    def check(self, value: float) -> Tuple[bool, str]:
        if value <= self.threshold:
            return True, f"检查通过: {value:.2%}"
        return False, f"检查失败: {value:.2%} 超过限制 {self.threshold:.2%}"

# 添加到报告生成器
report_generator.compliance_rules.append(CustomRule())
```

### 批量生成报告

```python
from datetime import datetime, timedelta

# 生成过去7天的每日报告
for i in range(7):
    date = datetime.now() - timedelta(days=i)
    report = report_generator.generate_daily_report(
        date=date,
        account=account
    )
    print(f"生成报告: {report.report_id}")
```

### 定时生成报告

```python
import schedule
import time

def generate_daily_report_job():
    """每日报告生成任务"""
    report = report_generator.generate_daily_report(account=account)
    print(f"每日报告已生成: {report.report_id}")

# 每天凌晨1点生成报告
schedule.every().day.at("01:00").do(generate_daily_report_job)

while True:
    schedule.run_pending()
    time.sleep(60)
```

## 最佳实践

### 1. 定期生成报告
- 每日报告：每天交易结束后生成
- 周报告：每周一生成上周报告
- 月报告：每月1日生成上月报告

### 2. 关注关键指标
- 胜率 > 40%
- 盈利因子 > 1.5
- 最大回撤 < 15%
- 夏普比率 > 1.0

### 3. 及时响应警告
- 连续亏损 >= 3次：暂停交易，检查策略
- 回撤 > 10%：降低仓位
- 保证金水平 < 200%：增加资金或减少持仓

### 4. 定期审查合规
- 每周检查合规报告
- 确保所有规则通过
- 及时调整违规策略

### 5. 策略优化
- 定期生成策略分析报告
- 识别表现不佳的策略
- 增加优秀策略权重
- 停用或优化亏损策略

## 故障排除

### 问题1: 报告生成失败

**原因**: 数据不足或数据格式错误

**解决方案**:
```python
# 检查性能跟踪器是否有数据
if not performance_tracker.trades_by_strategy:
    print("警告: 没有交易数据")

# 确保账户对象有效
if account.equity <= 0:
    print("错误: 账户净值无效")
```

### 问题2: VaR计算失败

**原因**: 历史数据不足（少于30笔交易）

**解决方案**:
- 积累更多交易数据
- VaR指标会在数据充足时自动计算

### 问题3: 合规检查全部失败

**原因**: 合规规则配置过于严格

**解决方案**:
```python
# 调整合规规则阈值
from src.agents.risk_manager import RiskConfig

config = RiskConfig(
    max_risk_per_trade=0.03,  # 放宽到3%
    max_daily_drawdown=0.08   # 放宽到8%
)
risk_manager = RiskManagerAgent(config)
```

## 示例代码

完整示例请参考 `examples/report_generator_demo.py`

## 相关文档

- [性能跟踪器指南](performance_tracker_guide.md)
- [风险管理器指南](risk_manager_guide.md)
- [交易系统架构](architecture.md)

## 更新日志

### v1.0.0 (2025-10-07)
- 初始版本发布
- 支持每日、周、月报告
- 支持策略分析报告
- 支持风险分析报告
- 支持合规检查报告
- 内置6个合规规则
- 支持JSON、Markdown、Text格式导出
