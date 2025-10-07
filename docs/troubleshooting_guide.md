# 故障排除和维护指南

## 概述

本指南提供LLM-RL MT5交易系统常见问题的诊断和解决方案，以及日常维护建议。

## 目录

- [快速诊断](#快速诊断)
- [连接问题](#连接问题)
- [数据问题](#数据问题)
- [订单执行问题](#订单执行问题)
- [性能问题](#性能问题)
- [LLM相关问题](#llm相关问题)
- [配置问题](#配置问题)
- [日志分析](#日志分析)
- [维护建议](#维护建议)

---

## 快速诊断

### 系统诊断脚本

运行诊断脚本快速检查系统状态：

```bash
python diagnose.py
```

**输出示例：**
```
=== 系统诊断报告 ===
✓ MT5连接: 正常
✓ Redis连接: 正常
✓ 数据库: 正常
✗ EA31337桥接: 未连接
✓ 配置文件: 有效
⚠ LLM模型: 未找到

建议:
1. 检查EA31337是否运行
2. 下载LLM模型到models/目录
```

### 手动健康检查

```python
from src.core.config import load_config
from src.data.mt5_connection import MT5Connection
from src.bridge.ea31337_bridge import EA31337Bridge

# 检查MT5连接
mt5 = MT5Connection()
print(f"MT5连接: {mt5.is_connected()}")

# 检查EA31337桥接
bridge = EA31337Bridge()
health = bridge.health_check()
print(f"桥接状态: {health}")

# 检查配置
config = load_config()
print(f"配置有效: {config.validate()}")
```

---

## 连接问题

### 问题1: MT5连接失败

**症状：**
```
ERROR: 无法连接到MT5平台
ConnectionError: MT5 initialization failed
```

**可能原因：**
1. MT5未运行
2. 账户信息错误
3. 服务器不可达
4. 防火墙阻止

**解决方案：**

#### 步骤1: 检查MT5是否运行
```bash
# Windows
tasklist | findstr terminal64.exe

# 如果未运行，启动MT5
start "" "C:\Program Files\MetaTrader 5\terminal64.exe"
```

#### 步骤2: 验证账户信息
```python
# 检查config.yaml
mt5:
  server: "YourBroker-Server"
  login: 12345678
  password: "your_password"
  path: "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
```

#### 步骤3: 测试连接
```python
import MetaTrader5 as mt5

# 初始化MT5
if not mt5.initialize():
    print(f"初始化失败: {mt5.last_error()}")
else:
    print("连接成功")
    print(f"账户信息: {mt5.account_info()}")
    mt5.shutdown()
```

#### 步骤4: 检查防火墙
```bash
# 添加MT5到防火墙例外
netsh advfirewall firewall add rule name="MT5" dir=in action=allow program="C:\Program Files\MetaTrader 5\terminal64.exe"
```

### 问题2: EA31337桥接未响应

**症状：**
```
WARNING: EA31337桥接未连接
BridgeError: 状态文件不存在或过期
```

**解决方案：**

#### 步骤1: 检查EA31337是否加载
1. 打开MT5
2. 查看"导航器" → "Expert Advisors"
3. 确认EA31337已加载到图表

#### 步骤2: 检查通信文件
```bash
# 检查文件是否存在
dir ea31337\*.json

# 应该看到:
# signals.json
# status.json
```

#### 步骤3: 检查文件权限
```python
from pathlib import Path

ea_path = Path("ea31337")
test_file = ea_path / "test.txt"

try:
    test_file.write_text("test")
    test_file.unlink()
    print("文件权限正常")
except Exception as e:
    print(f"文件权限问题: {e}")
```

#### 步骤4: 重启EA31337
1. 在MT5中移除EA
2. 等待5秒
3. 重新添加EA到图表
4. 检查日志输出

### 问题3: Redis连接失败

**症状：**
```
ERROR: Redis连接失败
redis.exceptions.ConnectionError: Error connecting to Redis
```

**解决方案：**

#### 步骤1: 检查Redis服务
```bash
# Windows
sc query Redis

# 如果未运行，启动Redis
net start Redis
```

#### 步骤2: 测试连接
```python
import redis

try:
    r = redis.Redis(host='localhost', port=6379, db=0)
    r.ping()
    print("Redis连接正常")
except Exception as e:
    print(f"Redis连接失败: {e}")
```

#### 步骤3: 检查配置
```yaml
# config.yaml
database:
  redis_host: "localhost"
  redis_port: 6379
  redis_db: 0
```

---

## 数据问题

### 问题4: 数据获取失败

**症状：**
```
ERROR: 获取市场数据失败
DataException: No data available for EURUSD
```

**解决方案：**

#### 步骤1: 检查品种是否可用
```python
import MetaTrader5 as mt5

mt5.initialize()
symbols = mt5.symbols_get()
symbol_names = [s.name for s in symbols]

if "EURUSD" in symbol_names:
    print("品种可用")
else:
    print("品种不可用，请在MT5中添加")

mt5.shutdown()
```

#### 步骤2: 检查市场是否开放
```python
from datetime import datetime

def is_market_open(symbol: str) -> bool:
    now = datetime.now()
    weekday = now.weekday()
    
    # 周末市场关闭
    if weekday >= 5:  # 周六、周日
        return False
    
    # 检查交易时段
    hour = now.hour
    if symbol == "EURUSD":
        return 0 <= hour <= 23  # 24小时交易
    
    return True

print(f"市场开放: {is_market_open('EURUSD')}")
```

#### 步骤3: 检查数据时间范围
```python
import MetaTrader5 as mt5
from datetime import datetime, timedelta

mt5.initialize()

# 获取最近100根K线
rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H1, 0, 100)

if rates is None:
    print(f"数据获取失败: {mt5.last_error()}")
else:
    print(f"获取到 {len(rates)} 根K线")
    print(f"最新时间: {datetime.fromtimestamp(rates[-1]['time'])}")

mt5.shutdown()
```

### 问题5: 数据不一致

**症状：**
```
WARNING: 数据验证失败
ValidationError: OHLC数据不一致
```

**解决方案：**

#### 步骤1: 清理缓存
```python
import redis

r = redis.Redis(host='localhost', port=6379, db=0)
# 清理所有市场数据缓存
keys = r.keys("market_data:*")
if keys:
    r.delete(*keys)
    print(f"清理了 {len(keys)} 个缓存键")
```

#### 步骤2: 重新同步数据
```python
from src.data.data_pipeline import DataPipeline

pipeline = DataPipeline()
# 强制从MT5重新获取数据
market_data = pipeline.get_realtime_data("EURUSD", "H1", use_cache=False)
```

#### 步骤3: 验证数据完整性
```python
def validate_ohlc(df):
    """验证OHLC数据"""
    checks = {
        'high_ge_low': (df['high'] >= df['low']).all(),
        'high_ge_open': (df['high'] >= df['open']).all(),
        'high_ge_close': (df['high'] >= df['close']).all(),
        'low_le_open': (df['low'] <= df['open']).all(),
        'low_le_close': (df['low'] <= df['close']).all(),
        'no_nulls': df.notnull().all().all()
    }
    
    for check, result in checks.items():
        print(f"{check}: {'✓' if result else '✗'}")
    
    return all(checks.values())
```

---

## 订单执行问题

### 问题6: 订单被拒绝

**症状：**
```
ERROR: 订单执行失败
OrderException: Order rejected - Invalid volume
```

**常见错误代码：**

| 错误代码 | 含义 | 解决方案 |
|---------|------|---------|
| 10004 | 资金不足 | 减少手数或增加资金 |
| 10006 | 无效请求 | 检查订单参数 |
| 10013 | 无效手数 | 调整到允许的手数范围 |
| 10015 | 无效价格 | 使用当前市场价格 |
| 10016 | 无效止损/止盈 | 检查止损止盈距离 |
| 10018 | 市场关闭 | 等待市场开放 |

**解决方案：**

#### 步骤1: 检查账户余额
```python
import MetaTrader5 as mt5

mt5.initialize()
account_info = mt5.account_info()

print(f"余额: {account_info.balance}")
print(f"净值: {account_info.equity}")
print(f"可用保证金: {account_info.margin_free}")
print(f"保证金水平: {account_info.margin_level}%")

mt5.shutdown()
```

#### 步骤2: 验证手数
```python
def validate_lot_size(symbol: str, lot_size: float) -> float:
    """验证并调整手数"""
    symbol_info = mt5.symbol_info(symbol)
    
    if symbol_info is None:
        raise ValueError(f"品种不存在: {symbol}")
    
    # 调整到最小手数
    if lot_size < symbol_info.volume_min:
        lot_size = symbol_info.volume_min
    
    # 调整到最大手数
    if lot_size > symbol_info.volume_max:
        lot_size = symbol_info.volume_max
    
    # 调整到步长
    steps = round((lot_size - symbol_info.volume_min) / symbol_info.volume_step)
    lot_size = symbol_info.volume_min + steps * symbol_info.volume_step
    
    return lot_size

# 使用
adjusted_lot = validate_lot_size("EURUSD", 0.15)
print(f"调整后手数: {adjusted_lot}")
```

#### 步骤3: 检查止损止盈距离
```python
def validate_stops(symbol: str, order_type: int, price: float, sl: float, tp: float):
    """验证止损止盈"""
    symbol_info = mt5.symbol_info(symbol)
    stops_level = symbol_info.trade_stops_level * symbol_info.point
    
    if order_type == mt5.ORDER_TYPE_BUY:
        # 买单：止损必须低于价格，止盈必须高于价格
        if sl > 0 and price - sl < stops_level:
            print(f"止损距离不足: {price - sl} < {stops_level}")
            return False
        if tp > 0 and tp - price < stops_level:
            print(f"止盈距离不足: {tp - price} < {stops_level}")
            return False
    else:
        # 卖单：止损必须高于价格，止盈必须低于价格
        if sl > 0 and sl - price < stops_level:
            print(f"止损距离不足: {sl - price} < {stops_level}")
            return False
        if tp > 0 and price - tp < stops_level:
            print(f"止盈距离不足: {price - tp} < {stops_level}")
            return False
    
    return True
```

### 问题7: 订单延迟过高

**症状：**
```
WARNING: 订单执行延迟: 2.5秒
```

**解决方案：**

#### 步骤1: 测量延迟
```python
import time

def measure_order_latency():
    """测量订单延迟"""
    start = time.time()
    
    # 发送订单
    result = send_order(signal)
    
    latency = time.time() - start
    print(f"订单延迟: {latency:.3f}秒")
    
    return latency
```

#### 步骤2: 优化网络
- 使用VPS靠近经纪商服务器
- 检查网络延迟: `ping broker-server.com`
- 使用有线连接而非WiFi

#### 步骤3: 优化代码
```python
# 使用异步订单执行
import asyncio

async def send_order_async(signal: Signal):
    """异步发送订单"""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, send_order, signal)
    return result
```

---

## 性能问题

### 问题8: 内存使用过高

**症状：**
```
WARNING: 内存使用: 85%
```

**解决方案：**

#### 步骤1: 监控内存使用
```python
import psutil

def check_memory():
    """检查内存使用"""
    memory = psutil.virtual_memory()
    print(f"总内存: {memory.total / 1024**3:.2f} GB")
    print(f"已用: {memory.used / 1024**3:.2f} GB ({memory.percent}%)")
    print(f"可用: {memory.available / 1024**3:.2f} GB")
```

#### 步骤2: 清理缓存
```python
# 清理Redis缓存
import redis

r = redis.Redis()
r.flushdb()  # 清理当前数据库

# 清理Python对象缓存
import gc
gc.collect()
```

#### 步骤3: 优化数据加载
```python
# 使用分块加载大数据集
def load_large_dataset(file_path, chunk_size=10000):
    """分块加载数据"""
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        process_chunk(chunk)
        # 处理完立即释放
        del chunk
```

### 问题9: CPU使用率过高

**症状：**
```
WARNING: CPU使用率: 95%
```

**解决方案：**

#### 步骤1: 识别瓶颈
```python
import cProfile
import pstats

# 性能分析
profiler = cProfile.Profile()
profiler.enable()

# 运行代码
trading_system.run()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # 显示前10个最耗时的函数
```

#### 步骤2: 优化计算
```python
# 使用向量化操作
import numpy as np

# 慢速循环
result = []
for i in range(len(data)):
    result.append(data[i] * 2)

# 快速向量化
result = data * 2  # NumPy数组操作
```

#### 步骤3: 减少LLM调用频率
```python
# config.yaml
llm:
  call_interval: 300  # 每5分钟调用一次
  cache_results: true  # 缓存结果
```

---

## LLM相关问题

### 问题10: LLM模型加载失败

**症状：**
```
ERROR: 无法加载LLM模型
FileNotFoundError: models/llama-3.2-1b not found
```

**解决方案：**

#### 步骤1: 下载模型
```bash
# 使用Hugging Face CLI
pip install huggingface-hub

# 下载模型
huggingface-cli download meta-llama/Llama-3.2-1B --local-dir models/llama-3.2-1b
```

#### 步骤2: 验证模型文件
```python
from pathlib import Path

model_path = Path("models/llama-3.2-1b")
required_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]

for file in required_files:
    file_path = model_path / file
    if file_path.exists():
        print(f"✓ {file}")
    else:
        print(f"✗ {file} 缺失")
```

#### 步骤3: 检查内存
```python
# LLM需要足够内存
# 1B模型: ~4GB
# 3B模型: ~12GB

import psutil
memory = psutil.virtual_memory()
required_memory = 4 * 1024**3  # 4GB

if memory.available < required_memory:
    print("内存不足，无法加载模型")
else:
    print("内存充足")
```

### 问题11: LLM推理速度慢

**症状：**
```
WARNING: LLM推理时间: 15秒
```

**解决方案：**

#### 步骤1: 使用GPU加速
```python
# config.yaml
llm:
  use_gpu: true
  device: "cuda"  # 或 "mps" for Mac
```

#### 步骤2: 减少token数量
```python
# config.yaml
llm:
  max_tokens: 256  # 从512减少到256
  temperature: 0.7
```

#### 步骤3: 使用更小的模型
```python
# 使用1B而非3B模型
llm:
  model_path: "models/llama-3.2-1b"  # 更快
```

---

## 配置问题

### 问题12: 配置文件无效

**症状：**
```
ERROR: 配置验证失败
ValueError: 单笔风险必须在0-10%之间
```

**解决方案：**

#### 步骤1: 验证配置
```python
from src.core.config import load_config

try:
    config = load_config("config.yaml")
    if config.validate():
        print("配置有效")
    else:
        print("配置无效")
except Exception as e:
    print(f"配置错误: {e}")
```

#### 步骤2: 使用默认配置
```bash
# 备份当前配置
cp config/config.yaml config/config.yaml.bak

# 生成默认配置
python -c "from src.core.config import ConfigManager; ConfigManager().save_config()"
```

#### 步骤3: 逐项检查
```yaml
# 检查风险参数
risk:
  max_risk_per_trade: 0.02  # 必须 0 < x <= 0.1
  max_daily_drawdown: 0.05  # 必须 0 < x <= 0.2

# 检查交易参数
trading:
  default_lot_size: 0.01  # 必须 > 0
  symbols: ["EURUSD"]  # 不能为空
```

---

## 日志分析

### 查看日志

```bash
# 查看最新日志
tail -f logs/trading.log

# 查看错误日志
tail -f logs/error.log

# 搜索特定错误
grep "ERROR" logs/trading.log

# 查看今天的日志
grep "2025-01-07" logs/trading.log
```

### 日志级别

```python
# 调整日志级别
# config.yaml
logging:
  level: "DEBUG"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### 常见日志模式

**正常运行：**
```
INFO: MT5连接成功
INFO: 获取到 5 个有效信号
INFO: 订单执行成功: EURUSD BUY 0.1
```

**警告信号：**
```
WARNING: 信号已过期: trend_following
WARNING: 保证金水平低: 150%
WARNING: 连续亏损: 3笔
```

**错误信号：**
```
ERROR: MT5连接失败
ERROR: 订单执行失败: Invalid volume
ERROR: 数据获取失败: No data available
```

---

## 维护建议

### 日常维护

**每日任务：**
1. 检查系统日志
2. 验证MT5连接
3. 检查账户状态
4. 查看交易报告

```bash
# 每日检查脚本
python scripts/daily_check.py
```

**每周任务：**
1. 清理旧日志文件
2. 备份数据库
3. 更新策略参数
4. 性能分析

```bash
# 每周维护脚本
python scripts/weekly_maintenance.py
```

**每月任务：**
1. 系统性能评估
2. 策略回测
3. 参数优化
4. 系统更新

### 数据备份

```bash
# 备份数据库
python scripts/backup_database.py

# 备份配置
cp -r config config_backup_$(date +%Y%m%d)

# 备份日志
tar -czf logs_backup_$(date +%Y%m%d).tar.gz logs/
```

### 系统更新

```bash
# 更新Python依赖
pip install -r requirements.txt --upgrade

# 更新EA31337
# 从GitHub下载最新版本

# 重启系统
python main.py
```

### 性能优化

```python
# 定期优化数据库
from src.data.database import DatabaseManager

db = DatabaseManager()
db.vacuum()  # 清理和优化数据库
db.analyze()  # 更新统计信息
```

---

## 紧急情况处理

### 紧急停止

```python
# 方法1: 使用桥接器
from src.bridge.ea31337_bridge import EA31337Bridge

bridge = EA31337Bridge()
bridge.emergency_stop()

# 方法2: 直接关闭MT5
import MetaTrader5 as mt5
mt5.initialize()
# 关闭所有持仓
positions = mt5.positions_get()
for pos in positions:
    mt5.Close(pos.ticket)
mt5.shutdown()
```

### 系统恢复

```bash
# 1. 停止系统
Ctrl+C

# 2. 检查状态
python diagnose.py

# 3. 清理临时文件
rm ea31337/commands.json
rm ea31337/responses.json

# 4. 重启系统
python main.py
```

---

## 获取帮助

### 日志收集

```bash
# 收集诊断信息
python scripts/collect_diagnostics.py

# 生成诊断报告
# 输出: diagnostics_20250107.zip
```

### 联系支持

提交Issue时请包含：
1. 错误信息和日志
2. 系统配置
3. 复现步骤
4. 诊断报告

---

## 总结

本指南涵盖了常见问题的诊断和解决方案。如遇到未列出的问题：

1. 查看详细日志
2. 运行诊断脚本
3. 查阅API文档
4. 提交GitHub Issue

定期维护和监控可以预防大多数问题的发生。
