# 交易记录系统使用指南

## 概述

交易记录系统（Trade Recorder）是一个完整的交易历史记录、分析和报告生成模块。它提供了以下核心功能：

- 交易历史记录和存储
- 交易统计和分析
- 报告生成（日报、周报、月报）
- 审计日志
- 数据导出

## 核心组件

### 1. TradeRecord 数据类

交易记录的核心数据结构，包含所有交易相关信息。

```python
from src.core.trade_recorder import TradeRecord, TradeStatus, TradeOutcome

trade = TradeRecord(
    trade_id="12345",
    symbol="EURUSD",
    trade_type="BUY",
    volume=0.1,
    open_price=1.1000,
    open_time=datetime.now(),
    strategy_id="my_strategy",
    close_price=1.1050,
    close_time=datetime.now(),
    profit=50.0,
    commission=2.0,
    swap=1.0
)
```

### 2. TradeRecorder 类

主要的交易记录器类，负责记录、分析和报告生成。

```python
from src.core.trade_recorder import TradeRecorder
from src.data.database import DatabaseManager

# 初始化
db_manager = DatabaseManager("data/trading.db")
recorder = TradeRecorder(
    db_manager=db_manager,
    config={
        'log_dir': 'logs',
        'max_recent_trades': 100
    }
)
```

## 主要功能

### 1. 记录交易

#### 记录开仓

```python
from src.core.models import Position, PositionType

position = Position(
    position_id="12345",
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
    comment="My trade",
    magic_number=123
)

# 记录开仓
trade_record = recorder.record_trade_open(position, "my_strategy")
```

#### 记录平仓

```python
# 更新持仓信息
position.current_price = 1.1050
position.profit = 50.0

# 记录平仓
trade_record = recorder.record_trade_close(
    position=position,
    close_price=1.1050,
    profit=50.0,
    commission=2.0,
    swap=1.0
)
```

### 2. 获取交易历史

```python
# 获取所有交易
all_trades = recorder.get_trade_history()

# 按品种过滤
eurusd_trades = recorder.get_trade_history(symbol="EURUSD")

# 按策略过滤
strategy_trades = recorder.get_trade_history(strategy_id="my_strategy")

# 按时间范围过滤
from datetime import datetime, timedelta

start_time = datetime.now() - timedelta(days=7)
recent_trades = recorder.get_trade_history(start_time=start_time)
```

### 3. 交易统计分析

```python
# 获取统计信息
stats = recorder.get_trade_statistics()

print(f"总交易数: {stats['closed_trades']}")
print(f"盈利交易: {stats['winning_trades']}")
print(f"亏损交易: {stats['losing_trades']}")
print(f"胜率: {stats['win_rate']:.2f}%")
print(f"总盈亏: ${stats['net_profit']:.2f}")
print(f"平均盈利: ${stats['avg_win']:.2f}")
print(f"平均亏损: ${stats['avg_loss']:.2f}")
print(f"盈亏比: {stats['profit_factor']:.2f}")
print(f"最大连续盈利: {stats['max_consecutive_wins']}")
print(f"最大连续亏损: {stats['max_consecutive_losses']}")

# 按品种统计
for symbol, symbol_stats in stats['by_symbol'].items():
    print(f"{symbol}: {symbol_stats['count']} 笔, 盈亏 ${symbol_stats['profit']:.2f}")

# 按策略统计
for strategy, strategy_stats in stats['by_strategy'].items():
    print(f"{strategy}: {strategy_stats['count']} 笔, 胜率 {strategy_stats['win_rate']:.2f}%")
```

### 4. 报告生成

#### 每日报告

```python
# 生成今日报告
daily_report = recorder.generate_daily_report()

# 生成指定日期报告
from datetime import datetime
date = datetime(2024, 1, 15)
daily_report = recorder.generate_daily_report(date)

print(f"日期: {daily_report['date']}")
print(f"交易数: {daily_report['summary']['closed_trades']}")
print(f"净盈亏: ${daily_report['summary']['net_profit']:.2f}")
```

#### 每周报告

```python
# 生成本周报告
weekly_report = recorder.generate_weekly_report()

print(f"周期: {weekly_report['week_start']} 至 {weekly_report['week_end']}")
print(f"总交易数: {weekly_report['total_trades']}")

# 每日明细
for date, day_stats in weekly_report['daily_breakdown'].items():
    print(f"{date}: {day_stats['trades']} 笔, ${day_stats['profit']:.2f}")
```

#### 每月报告

```python
# 生成本月报告
monthly_report = recorder.generate_monthly_report()

print(f"月份: {monthly_report['month']}")
print(f"总交易数: {monthly_report['total_trades']}")
print(f"最佳交易日: {monthly_report['best_day']}")
print(f"最差交易日: {monthly_report['worst_day']}")

# 每周明细
for week, week_stats in monthly_report['weekly_breakdown'].items():
    print(f"{week}: {week_stats['trades']} 笔, ${week_stats['profit']:.2f}")
```

### 5. 审计日志

```python
# 获取所有审计日志
audit_logs = recorder.get_audit_log()

# 按时间范围过滤
start_time = datetime.now() - timedelta(days=1)
recent_logs = recorder.get_audit_log(start_time=start_time)

# 按交易ID过滤
trade_logs = recorder.get_audit_log(trade_id="12345")

# 显示日志
for log in audit_logs:
    print(f"[{log['timestamp']}] {log['action']} - {log['trade_id']}")
```

### 6. 数据导出

```python
# 导出所有交易到CSV
recorder.export_trades_to_csv("exports/all_trades.csv")

# 导出特定品种
recorder.export_trades_to_csv(
    "exports/eurusd_trades.csv",
    symbol="EURUSD"
)

# 导出特定时间范围
recorder.export_trades_to_csv(
    "exports/last_month.csv",
    start_time=datetime.now() - timedelta(days=30)
)
```

### 7. 性能摘要

```python
# 获取整体性能摘要
summary = recorder.get_performance_summary()

print("整体性能:")
print(f"  总交易数: {summary['all_time']['closed_trades']}")
print(f"  总盈亏: ${summary['all_time']['net_profit']:.2f}")

print("\n最近30天:")
print(f"  交易数: {summary['last_30_days']['closed_trades']}")
print(f"  盈亏: ${summary['last_30_days']['net_profit']:.2f}")

print("\n今日:")
print(f"  交易数: {summary['today']['closed_trades']}")
print(f"  盈亏: ${summary['today']['net_profit']:.2f}")

print(f"\n活跃交易: {summary['active_trades']}")
```

### 8. 日志清理

```python
# 清理90天前的旧日志
recorder.cleanup_old_logs(days_to_keep=90)
```

## 统计指标说明

### 基本指标

- **total_trades**: 总交易数
- **closed_trades**: 已关闭交易数
- **open_trades**: 未平仓交易数
- **winning_trades**: 盈利交易数
- **losing_trades**: 亏损交易数

### 盈亏指标

- **win_rate**: 胜率（盈利交易占比）
- **total_profit**: 总盈亏
- **net_profit**: 净盈亏（扣除手续费和隔夜利息）
- **avg_win**: 平均盈利
- **avg_loss**: 平均亏损
- **max_win**: 最大单笔盈利
- **max_loss**: 最大单笔亏损

### 风险指标

- **profit_factor**: 盈亏比（总盈利/总亏损）
- **max_consecutive_wins**: 最大连续盈利次数
- **max_consecutive_losses**: 最大连续亏损次数
- **avg_duration_hours**: 平均持仓时长

### 交易指标

- **pips**: 点数
- **duration_hours**: 持仓时长
- **risk_reward_ratio**: 风险回报比
- **outcome**: 交易结果（WIN/LOSS/BREAKEVEN）

## 配置选项

```python
config = {
    'log_dir': 'logs',              # 日志目录
    'max_recent_trades': 100,       # 最大缓存的最近交易数
}

recorder = TradeRecorder(db_manager, config)
```

## 文件结构

交易记录系统会创建以下文件结构：

```
logs/
├── trades.log                      # 交易日志
└── reports/
    ├── daily/
    │   └── report_20240115.json   # 每日报告
    ├── weekly/
    │   └── report_week_20240115.json  # 每周报告
    └── monthly/
        └── report_202401.json     # 每月报告
```

## 最佳实践

### 1. 及时记录

在每次开仓和平仓时立即记录，确保数据完整性：

```python
# 开仓后立即记录
position = execute_order(signal)
recorder.record_trade_open(position, strategy_id)

# 平仓后立即记录
close_result = close_position(position)
recorder.record_trade_close(position, close_result.price, close_result.profit)
```

### 2. 定期生成报告

建议每日生成报告，便于跟踪交易表现：

```python
# 每日定时任务
def daily_report_task():
    report = recorder.generate_daily_report()
    # 发送报告通知
    send_notification(report)
```

### 3. 定期清理日志

避免日志文件过大，定期清理旧数据：

```python
# 每周清理一次
recorder.cleanup_old_logs(days_to_keep=90)
```

### 4. 备份数据

定期备份交易数据：

```python
# 定期备份数据库
db_manager.backup_database("backups/trading_backup.db")
```

### 5. 监控关键指标

持续监控关键性能指标：

```python
def check_performance():
    stats = recorder.get_trade_statistics()
    
    # 检查胜率
    if stats['win_rate'] < 40:
        logger.warning("胜率过低，需要调整策略")
    
    # 检查连续亏损
    if stats['max_consecutive_losses'] > 5:
        logger.warning("连续亏损过多，建议暂停交易")
    
    # 检查盈亏比
    if stats['profit_factor'] < 1.5:
        logger.warning("盈亏比不理想")
```

## 集成示例

### 与交易系统集成

```python
from src.core.trade_recorder import TradeRecorder
from src.core.order_executor import OrderExecutor
from src.core.position_manager import PositionManager
from src.data.database import DatabaseManager

class TradingSystem:
    def __init__(self):
        self.db_manager = DatabaseManager("data/trading.db")
        self.recorder = TradeRecorder(self.db_manager)
        self.order_executor = OrderExecutor()
        self.position_manager = PositionManager()
    
    def execute_signal(self, signal):
        # 执行订单
        result = self.order_executor.send_order(signal)
        
        if result['success']:
            # 记录开仓
            position = self.position_manager.get_position(result['order_id'])
            self.recorder.record_trade_open(position, signal.strategy_id)
    
    def close_position(self, position_id):
        # 平仓
        position = self.position_manager.get_position(position_id)
        success = self.position_manager.close_position(position_id)
        
        if success:
            # 记录平仓
            self.recorder.record_trade_close(
                position=position,
                close_price=position.current_price,
                profit=position.profit,
                commission=position.commission,
                swap=position.swap
            )
    
    def generate_daily_summary(self):
        # 生成每日摘要
        report = self.recorder.generate_daily_report()
        stats = self.recorder.get_trade_statistics()
        
        return {
            'report': report,
            'stats': stats
        }
```

## 故障排除

### 问题：交易记录未保存

**解决方案**：
1. 检查数据库连接是否正常
2. 确认数据库文件有写入权限
3. 查看日志文件中的错误信息

### 问题：统计数据不准确

**解决方案**：
1. 确保所有交易都已正确记录
2. 检查时间范围过滤条件
3. 验证数据库中的数据完整性

### 问题：报告生成失败

**解决方案**：
1. 确认日志目录存在且有写入权限
2. 检查是否有足够的磁盘空间
3. 查看错误日志获取详细信息

## 参考资料

- [数据库管理器文档](database_guide.md)
- [订单执行器文档](order_executor_guide.md)
- [仓位管理器文档](position_manager_guide.md)
