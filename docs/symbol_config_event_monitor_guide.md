# 品种配置和经济事件监控系统指南

## 概述

品种配置和经济事件监控系统是交易系统的核心组件，负责管理不同交易品种的特定配置和监控影响市场的重要经济事件。

## 主要功能

### 1. 品种配置管理 (SymbolConfigManager)

管理各交易品种的专门配置，包括：
- 交易时段管理
- 风险参数配置
- 策略选择
- 手数验证
- 点差检查

### 2. 经济事件监控 (EconomicEventMonitor)

监控影响市场的重要事件：
- 经济日历事件（NFP、FOMC、CPI等）
- EIA原油库存数据
- OPEC会议决议
- 事件驱动的策略调整

## 快速开始

### 基本使用

```python
from src.core.symbol_config_manager import SymbolConfigManager
from src.core.economic_event_monitor import EconomicEventMonitor

# 初始化管理器
symbol_manager = SymbolConfigManager("config/symbols")
event_monitor = EconomicEventMonitor("config/economic_calendar.json")

# 获取品种配置
config = symbol_manager.get_config("EURUSD")

# 检查交易限制
should_restrict, reason = event_monitor.check_trading_restrictions("EURUSD")
```

## 品种配置详解

### 配置文件结构

每个品种的配置文件位于 `config/symbols/` 目录下，使用YAML格式：

```yaml
symbol: "EURUSD"
spread_limit: 2.0
min_lot: 0.01
max_lot: 10.0
lot_step: 0.01
risk_multiplier: 1.0

strategies:
  - "ma_cross"
  - "macd"

timeframes:
  - "M5"
  - "H1"

trading_hours:
  monday: "00:00-23:59"
  tuesday: "00:00-23:59"
  # ...

optimize_params:
  ma_fast_period: [5, 50, 5]
  ma_slow_period: [20, 200, 10]

risk_params:
  max_spread: 3.0
  min_equity: 1000.0
  max_slippage: 3
  stop_loss_pips: 20
  take_profit_pips: 40
```

### 支持的品种

#### 1. EURUSD (欧元/美元)
- **特点**: 流动性最高，点差最小
- **风险倍数**: 1.0 (标准)
- **策略**: MA交叉、MACD、趋势跟踪
- **交易时段**: 全天候（周一至周五）

#### 2. XAUUSD (黄金)
- **特点**: 避险资产，波动较大
- **风险倍数**: 0.8 (降低风险)
- **策略**: 突破、支撑阻力
- **事件监控**: FOMC、NFP、CPI、央行决议
- **交易时段**: 避开亚洲盘低波动时段

#### 3. USOIL (原油)
- **特点**: 波动极大，受供需影响显著
- **风险倍数**: 0.7 (最低风险)
- **策略**: 新闻交易、库存交易、动量
- **特殊监控**: 
  - EIA库存数据（每周三15:30 UTC）
  - OPEC会议决议
- **交易时段**: 重点关注美国交易时段

#### 4. US30 (道琼斯指数)
- **特点**: 美股指数，受财报和经济数据影响
- **风险倍数**: 0.9 (略微降低)
- **策略**: 趋势跟踪、动量、突破
- **事件监控**: FOMC、NFP、CPI、财报季、VIX
- **交易时段**: 美股交易时段

### 关键功能

#### 1. 交易时间检查

```python
# 检查当前是否在交易时间内
config = symbol_manager.get_config("EURUSD")
if config.is_trading_time():
    print("可以交易")
```

#### 2. 手数验证

```python
# 验证并调整手数
valid, adjusted_lot = symbol_manager.validate_lot_size("EURUSD", 0.123)
print(f"调整后手数: {adjusted_lot}")  # 输出: 0.12
```

#### 3. 点差检查

```python
# 检查点差是否可接受
if symbol_manager.check_spread("EURUSD", 1.8):
    print("点差可接受")
```

#### 4. 获取活跃品种

```python
# 获取当前可交易的品种
active_symbols = symbol_manager.get_active_symbols()
print(f"可交易品种: {active_symbols}")
```

## 经济事件监控详解

### 事件类型

#### 1. 经济日历事件

支持的事件类型：
- **就业数据**: NFP (非农就业)
- **通胀数据**: CPI (消费者物价指数)
- **央行决议**: FOMC、ECB、BOE利率决议
- **GDP**: 经济增长数据
- **制造业**: PMI指数
- **消费者**: 零售销售

#### 2. EIA库存数据

- **发布时间**: 每周三 15:30 UTC
- **影响品种**: USOIL、UKOIL
- **监控内容**: 原油库存变化
- **交易限制**: 发布前30分钟至发布后15分钟

#### 3. OPEC会议

- **会议类型**: 定期会议、特别会议
- **影响品种**: USOIL、UKOIL
- **监控内容**: 产量决议
- **交易限制**: 会议前2小时至会议后1小时

### 事件影响级别

```python
from src.core.economic_event_monitor import EventImpact

# LOW: 低影响 - 不影响交易
# MEDIUM: 中等影响 - 建议关注
# HIGH: 高影响 - 建议减少仓位
# CRITICAL: 关键影响 - 建议暂停交易
```

### 使用示例

#### 1. 检查交易限制

```python
# 检查是否应该限制交易
should_restrict, reason = event_monitor.check_trading_restrictions("EURUSD")

if should_restrict:
    print(f"限制交易: {reason}")
else:
    print("可以正常交易")
```

#### 2. 获取即将发生的事件

```python
# 获取未来24小时内的高影响事件
upcoming_events = event_monitor.calendar.get_upcoming_events(
    hours_ahead=24,
    min_impact=EventImpact.HIGH
)

for event in upcoming_events:
    print(f"{event.name} - {event.scheduled_time}")
```

#### 3. 获取事件驱动的调整建议

```python
# 获取基于事件的策略调整
adjustments = event_monitor.get_event_driven_adjustments("EURUSD")

if adjustments['reduce_position']:
    print("建议减少仓位")
    
if adjustments['avoid_new_trades']:
    print("建议避免新交易")
    
print(f"风险倍数调整为: {adjustments['risk_multiplier']}")
```

#### 4. EIA库存监控

```python
# 检查是否临近EIA发布
is_near, release_time = event_monitor.eia_monitor.is_near_release()

if is_near:
    print(f"临近EIA发布: {release_time}")
    print("建议暂停USOIL交易")

# 分析库存趋势
trend = event_monitor.eia_monitor.analyze_trend(weeks=4)
print(f"库存趋势: {trend['trend']}")  # 'building' 或 'drawing'
```

#### 5. OPEC会议监控

```python
# 获取下次会议
next_meeting = event_monitor.opec_monitor.get_next_meeting()

if next_meeting:
    print(f"下次OPEC会议: {next_meeting.meeting_date}")
    
# 检查是否临近会议
is_near, meeting = event_monitor.opec_monitor.is_near_meeting()

if is_near:
    print("临近OPEC会议，建议暂停USOIL交易")
```

## 集成工作流程

### 完整的交易决策流程

```python
def make_trading_decision(symbol, requested_lot, current_spread):
    """完整的交易决策流程"""
    
    # 1. 检查品种配置
    if not symbol_manager.is_symbol_configured(symbol):
        return False, "品种未配置"
    
    config = symbol_manager.get_config(symbol)
    
    # 2. 检查交易时间
    if not config.is_trading_time():
        return False, "不在交易时间内"
    
    # 3. 检查点差
    if not symbol_manager.check_spread(symbol, current_spread):
        return False, "点差超出限制"
    
    # 4. 验证手数
    valid, adjusted_lot = symbol_manager.validate_lot_size(symbol, requested_lot)
    if not valid:
        return False, "手数无效"
    
    # 5. 检查经济事件
    should_restrict, reason = event_monitor.check_trading_restrictions(symbol)
    if should_restrict:
        return False, f"事件限制: {reason}"
    
    # 6. 获取风险调整
    adjustments = event_monitor.get_event_driven_adjustments(symbol)
    risk_multiplier = config.risk_multiplier * adjustments['risk_multiplier']
    
    # 7. 计算最终参数
    final_lot = adjusted_lot * risk_multiplier
    
    return True, {
        'lot': final_lot,
        'stop_loss': config.risk_params.stop_loss_pips,
        'take_profit': config.risk_params.take_profit_pips,
        'risk_multiplier': risk_multiplier
    }
```

## 配置文件管理

### 添加新品种

1. 创建配置文件 `config/symbols/newpair.yaml`
2. 填写必要的配置项
3. 重启系统或重新加载配置

```python
# 创建默认配置
default_config = symbol_manager.create_default_config("NEWPAIR")

# 修改配置
default_config.spread_limit = 2.5
default_config.strategies = ["ma_cross", "rsi"]

# 保存配置
symbol_manager.save_config("NEWPAIR", default_config)
```

### 更新经济日历

经济日历文件位于 `config/economic_calendar.json`：

```json
{
  "events": [
    {
      "event_id": "nfp_2024_06",
      "name": "Non-Farm Payrolls",
      "country": "US",
      "event_type": "employment",
      "impact": "critical",
      "scheduled_time": "2024-06-07T12:30:00",
      "forecast_value": 180000.0,
      "affected_symbols": ["EURUSD", "XAUUSD"]
    }
  ]
}
```

## 最佳实践

### 1. 品种配置

- **风险倍数**: 根据品种波动性调整
  - 低波动 (EURUSD): 1.0
  - 中波动 (XAUUSD): 0.8
  - 高波动 (USOIL): 0.7

- **交易时段**: 选择流动性最好的时段
  - 外汇: 伦敦+纽约重叠时段
  - 黄金: 避开亚洲盘
  - 原油: 美国交易时段
  - 股指: 对应市场开盘时间

### 2. 事件监控

- **高影响事件**: 提前30分钟停止交易
- **关键事件**: 提前1-2小时减少仓位
- **定期更新**: 每周更新经济日历
- **历史分析**: 记录事件对市场的实际影响

### 3. 风险管理

- **事件前**: 减少仓位50%
- **事件中**: 避免新交易
- **事件后**: 等待15-30分钟观察
- **意外数据**: 立即评估影响并调整策略

## 故障排除

### 常见问题

1. **品种配置未加载**
   - 检查配置文件路径
   - 验证YAML格式
   - 查看日志错误信息

2. **交易时间判断错误**
   - 确认使用UTC时间
   - 检查交易时段配置
   - 考虑夏令时影响

3. **事件监控不生效**
   - 更新经济日历文件
   - 检查事件时间格式
   - 验证品种关联配置

## 性能优化

- **配置缓存**: 配置加载后缓存在内存中
- **事件索引**: 按时间和品种建立索引
- **定期清理**: 清理过期的历史事件
- **异步处理**: 事件检查使用异步方式

## 扩展开发

### 添加新的事件类型

```python
from src.core.economic_event_monitor import EventType

# 在EventType枚举中添加新类型
class EventType(Enum):
    # ... 现有类型
    CUSTOM_EVENT = "custom_event"
```

### 自定义事件监控器

```python
class CustomEventMonitor:
    """自定义事件监控器"""
    
    def check_custom_event(self, symbol: str) -> Tuple[bool, str]:
        """检查自定义事件"""
        # 实现自定义逻辑
        pass
```

## 参考资料

- [Requirements 3.1-3.4](../requirements.md#requirement-3): 多品种交易配置
- [Requirements 5.3](../requirements.md#requirement-5): 经济事件监控
- [Design Document](../design.md): 系统架构设计

## 更新日志

- **2024-10**: 初始版本发布
  - 实现品种配置管理器
  - 实现经济事件监控器
  - 支持EURUSD、XAUUSD、USOIL、US30配置
  - 集成EIA和OPEC监控
