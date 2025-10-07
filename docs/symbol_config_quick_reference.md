# 品种配置和事件监控快速参考

## 快速开始

```python
from src.core.symbol_config_manager import SymbolConfigManager
from src.core.economic_event_monitor import EconomicEventMonitor

# 初始化
symbol_mgr = SymbolConfigManager("config/symbols")
event_mon = EconomicEventMonitor("config/economic_calendar.json")
```

## 常用操作

### 品种配置

```python
# 获取配置
config = symbol_mgr.get_config("EURUSD")

# 检查交易时间
if config.is_trading_time():
    print("可以交易")

# 验证手数
valid, lot = symbol_mgr.validate_lot_size("EURUSD", 0.5)

# 检查点差
if symbol_mgr.check_spread("EURUSD", 1.8):
    print("点差可接受")

# 获取活跃品种
active = symbol_mgr.get_active_symbols()
```

### 事件监控

```python
# 检查交易限制
should_restrict, reason = event_mon.check_trading_restrictions("EURUSD")

# 获取调整建议
adj = event_mon.get_event_driven_adjustments("EURUSD")
risk_mult = adj['risk_multiplier']

# 获取即将发生的事件
events = event_mon.calendar.get_upcoming_events(hours_ahead=24)

# EIA监控
is_near, time = event_mon.eia_monitor.is_near_release()

# OPEC监控
meeting = event_mon.opec_monitor.get_next_meeting()
```

## 品种特性

| 品种 | 风险倍数 | 止损点数 | 特殊监控 |
|------|----------|----------|----------|
| EURUSD | 1.0 | 20 | - |
| XAUUSD | 0.8 | 100 | 央行事件 |
| USOIL | 0.7 | 150 | EIA, OPEC |
| US30 | 0.9 | 80 | 财报, VIX |

## 事件影响级别

- **LOW**: 不影响交易
- **MEDIUM**: 建议关注
- **HIGH**: 建议减少仓位
- **CRITICAL**: 建议暂停交易

## 完整交易检查

```python
def can_trade(symbol, lot, spread):
    # 1. 品种配置
    if not symbol_mgr.is_symbol_configured(symbol):
        return False, "品种未配置"
    
    config = symbol_mgr.get_config(symbol)
    
    # 2. 交易时间
    if not config.is_trading_time():
        return False, "不在交易时间"
    
    # 3. 点差
    if not symbol_mgr.check_spread(symbol, spread):
        return False, "点差超限"
    
    # 4. 手数
    valid, adj_lot = symbol_mgr.validate_lot_size(symbol, lot)
    if not valid:
        return False, "手数无效"
    
    # 5. 事件
    restrict, reason = event_mon.check_trading_restrictions(symbol)
    if restrict:
        return False, f"事件限制: {reason}"
    
    # 6. 风险调整
    adj = event_mon.get_event_driven_adjustments(symbol)
    final_lot = adj_lot * config.risk_multiplier * adj['risk_multiplier']
    
    return True, {
        'lot': final_lot,
        'sl': config.risk_params.stop_loss_pips,
        'tp': config.risk_params.take_profit_pips
    }
```

## 配置文件位置

- 品种配置: `config/symbols/*.yaml`
- 经济日历: `config/economic_calendar.json`

## 运行演示

```bash
python examples/symbol_config_event_monitor_demo.py
```

## 运行测试

```bash
pytest tests/test_symbol_config_manager.py -v
pytest tests/test_economic_event_monitor.py -v
```
