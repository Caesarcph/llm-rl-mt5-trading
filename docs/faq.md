# 常见问题解答 (FAQ)

## 目录

- [一般问题](#一般问题)
- [安装和配置](#安装和配置)
- [交易相关](#交易相关)
- [技术问题](#技术问题)
- [性能优化](#性能优化)
- [安全和风险](#安全和风险)

---

## 一般问题

### Q1: 这个系统适合我吗？

**A:** 本系统适合：
- ✓ 有一定编程基础的交易者
- ✓ 希望使用自动化交易的投资者
- ✓ 对AI和机器学习感兴趣的用户
- ✓ 愿意学习和测试的人

不适合：
- ✗ 完全没有交易经验的新手
- ✗ 期望快速致富的人
- ✗ 不愿意承担风险的投资者

### Q2: 需要多少初始资金？

**A:** 建议资金规模：
- **最低**: $1000（仅用于测试）
- **推荐**: $5000-$10000（实盘交易）
- **理想**: $10000以上（更好的风险管理）

**原因**：
- 更大的资金允许更好的风险分散
- 可以承受正常的市场波动
- 有足够的保证金应对回撤

### Q3: 系统能保证盈利吗？

**A:** **不能**。重要事实：
- 没有任何交易系统能保证盈利
- 过去的表现不代表未来结果
- 外汇交易存在重大风险
- 可能会损失全部投资

本系统提供：
- 系统化的交易方法
- 风险管理工具
- 回测和优化功能
- 但最终结果取决于市场条件和参数设置

### Q4: 需要多少时间维护？

**A:** 时间投入：
- **每日**: 10-15分钟（检查系统状态）
- **每周**: 30-60分钟（性能分析和调整）
- **每月**: 2-3小时（全面评估和优化）

系统可以24/7自动运行，但建议定期监控。

### Q5: 支持哪些交易品种？

**A:** 理论上支持所有MT5品种：
- **外汇**: EURUSD, GBPUSD, USDJPY等
- **贵金属**: XAUUSD (黄金), XAGUSD (白银)
- **能源**: USOIL (原油), UKOIL
- **股指**: US30, NAS100, SPX500等
- **加密货币**: BTCUSD (如果经纪商支持)

实际支持取决于您的经纪商。

---

## 安装和配置

### Q6: 必须使用Windows吗？

**A:** 不是必须，但推荐：
- **Windows**: 最佳支持，MT5原生运行
- **Linux**: 需要Wine运行MT5，Python部分完全支持
- **Mac**: 需要Wine或虚拟机，或使用VPS

**推荐方案**：
- 本地开发: Windows/Mac
- 生产运行: Windows VPS

### Q7: 可以使用模拟账户吗？

**A:** 强烈推荐！
```yaml
# config.yaml
simulation_mode: true
```

**模拟账户优势**：
- 零风险测试
- 验证系统稳定性
- 学习系统操作
- 优化参数设置

**建议流程**：
1. 模拟账户测试2-4周
2. 验证盈利能力
3. 小资金实盘测试
4. 逐步增加资金

### Q8: LLM模型是必需的吗？

**A:** 不是必需的。

**不使用LLM**：
- 系统仍可正常运行
- 使用技术分析策略
- 节省计算资源

**使用LLM**：
- 增加市场情绪分析
- 新闻事件解读
- 更全面的决策支持

```yaml
# 禁用LLM
llm:
  enabled: false
```

### Q9: 如何更新系统？

**A:** 更新步骤：
```bash
# 1. 备份配置和数据
cp -r config config_backup
cp -r data data_backup

# 2. 拉取最新代码
git pull origin main

# 3. 更新依赖
pip install -r requirements.txt --upgrade

# 4. 运行迁移脚本（如果有）
python scripts/migrate.py

# 5. 重启系统
python main.py
```

### Q10: 可以同时运行多个实例吗？

**A:** 可以，但需要注意：

**不同账户**：
```bash
# 实例1
python main.py --config config/account1.yaml

# 实例2
python main.py --config config/account2.yaml
```

**相同账户**：
- ⚠️ 不推荐，可能导致冲突
- 如果必须，确保：
  - 使用不同的品种
  - 使用不同的magic_number
  - 协调风险管理

---

## 交易相关

### Q11: 如何设置止损止盈？

**A:** 三种方式：

**1. 全局默认**：
```yaml
# config.yaml
risk:
  stop_loss_pct: 0.02    # 2%止损
  take_profit_pct: 0.04  # 4%止盈
```

**2. 品种特定**：
```yaml
# config/symbols/eurusd.yaml
risk_params:
  stop_loss_pct: 0.015   # EURUSD使用1.5%
  take_profit_pct: 0.03
```

**3. 策略特定**：
```python
# 在策略代码中
signal.sl = entry_price * (1 - 0.02)  # 2%止损
signal.tp = entry_price * (1 + 0.04)  # 4%止盈
```

### Q12: 如何控制交易频率？

**A:** 多种控制方法：

**1. 信号强度阈值**：
```yaml
strategies:
  trend:
    min_signal_strength: 0.7  # 只接受强度>0.7的信号
```

**2. 时间间隔**：
```python
# 限制同一品种的交易间隔
min_trade_interval: 3600  # 1小时
```

**3. 最大持仓数**：
```yaml
risk:
  max_positions: 5  # 最多5个仓位
  max_positions_per_symbol: 1  # 每个品种最多1个
```

### Q13: 如何处理隔夜持仓？

**A:** 配置选项：

**允许隔夜**：
```yaml
trading:
  allow_overnight: true
  overnight_risk_multiplier: 0.5  # 降低隔夜仓位风险
```

**禁止隔夜**：
```yaml
trading:
  allow_overnight: false
  close_before_market_close: true
  close_time: "22:00"  # 每天22:00平仓
```

**周末处理**：
```yaml
trading:
  close_before_weekend: true
  friday_close_time: "20:00"  # 周五20:00平仓
```

### Q14: 如何添加自定义策略？

**A:** 步骤：

**1. 创建策略文件**：
```python
# src/strategies/my_strategy.py
from src.core.models import Strategy, Signal, MarketData
from typing import Optional

class MyStrategy(Strategy):
    def __init__(self, period: int = 20):
        self.period = period
        self.name = "my_strategy"
    
    def generate_signal(self, market_data: MarketData) -> Optional[Signal]:
        # 实现您的策略逻辑
        df = market_data.ohlcv
        
        # 示例：简单移动平均线交叉
        df['sma'] = df['close'].rolling(self.period).mean()
        
        if df['close'].iloc[-1] > df['sma'].iloc[-1]:
            # 买入信号
            return Signal(
                strategy_id=self.name,
                symbol=market_data.symbol,
                direction=1,
                strength=0.8,
                entry_price=df['close'].iloc[-1],
                sl=df['close'].iloc[-1] * 0.98,
                tp=df['close'].iloc[-1] * 1.04,
                size=0.1,
                confidence=0.75,
                timestamp=market_data.timestamp
            )
        
        return None
    
    def update_parameters(self, params: dict) -> None:
        if 'period' in params:
            self.period = params['period']
```

**2. 注册策略**：
```python
# main.py
from src.strategies.my_strategy import MyStrategy

strategy_manager.register_strategy("my_strategy", MyStrategy())
```

**3. 启用策略**：
```yaml
# config.yaml
trading:
  strategies_enabled:
    - my_strategy
```

### Q15: 如何回测策略？

**A:** 使用回测引擎：

```python
# scripts/backtest_strategy.py
from src.backtest.backtest_engine import BacktestEngine
from src.strategies.my_strategy import MyStrategy
from datetime import datetime, timedelta

# 创建回测引擎
engine = BacktestEngine(
    initial_balance=10000,
    start_date=datetime.now() - timedelta(days=90),
    end_date=datetime.now(),
    symbol="EURUSD",
    timeframe="H1"
)

# 添加策略
strategy = MyStrategy(period=20)
engine.add_strategy(strategy)

# 运行回测
results = engine.run()

# 打印结果
print(f"总收益: {results['total_return']:.2%}")
print(f"夏普比率: {results['sharpe_ratio']:.2f}")
print(f"最大回撤: {results['max_drawdown']:.2%}")
print(f"胜率: {results['win_rate']:.2%}")
print(f"盈利因子: {results['profit_factor']:.2f}")

# 生成报告
engine.generate_report("backtest_report.html")
```

---

## 技术问题

### Q16: 系统占用多少资源？

**A:** 资源使用：

**CPU**：
- 空闲: 5-10%
- 正常运行: 15-30%
- LLM分析时: 50-80%

**内存**：
- 基础系统: 500MB-1GB
- 加载LLM 1B: +4GB
- 加载LLM 3B: +12GB

**硬盘**：
- 系统文件: ~500MB
- LLM模型: 2-6GB
- 数据和日志: 随时间增长

**网络**：
- 实时数据: ~10KB/s
- 新闻抓取: ~100KB/s

### Q17: 如何优化性能？

**A:** 优化建议：

**1. 使用SSD**：
```yaml
database:
  sqlite_path: "D:/SSD/trading.db"  # SSD路径
```

**2. 调整缓存**：
```yaml
database:
  redis_cache_ttl: 300  # 5分钟缓存
  enable_query_cache: true
```

**3. 减少日志**：
```yaml
logging:
  level: "INFO"  # 从DEBUG改为INFO
  console_output: false  # 禁用控制台输出
```

**4. 优化LLM**：
```yaml
llm:
  model_path: "models/llama-3.2-1b"  # 使用1B而非3B
  call_interval: 600  # 增加调用间隔
  use_gpu: true  # 启用GPU加速
```

**5. 限制历史数据**：
```python
# 只保留最近3个月数据
data_retention_days: 90
```

### Q18: 如何备份数据？

**A:** 备份策略：

**自动备份**：
```yaml
# config.yaml
backup:
  enabled: true
  interval: "daily"  # daily, weekly
  retention: 30  # 保留30天
  path: "backups/"
```

**手动备份**：
```bash
# 备份脚本
python scripts/backup_database.py

# 或手动复制
cp data/trading.db backups/trading_$(date +%Y%m%d).db
cp -r config backups/config_$(date +%Y%m%d)
```

**恢复数据**：
```bash
# 恢复数据库
cp backups/trading_20250107.db data/trading.db

# 恢复配置
cp -r backups/config_20250107 config/
```

### Q19: 日志文件太大怎么办？

**A:** 日志管理：

**1. 配置日志轮转**：
```yaml
logging:
  max_file_size: 10485760  # 10MB
  backup_count: 5  # 保留5个备份
```

**2. 定期清理**：
```bash
# 清理30天前的日志
find logs/ -name "*.log" -mtime +30 -delete

# 压缩旧日志
gzip logs/*.log.1
```

**3. 使用日志管理工具**：
```bash
# 安装logrotate (Linux)
sudo apt install logrotate

# 配置
# /etc/logrotate.d/trading-system
/path/to/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
}
```

### Q20: 如何监控系统健康？

**A:** 监控方法：

**1. 内置健康检查**：
```python
from src.core.monitoring import SystemMonitor

monitor = SystemMonitor()
health = monitor.check_health()

print(f"MT5连接: {health['mt5_connected']}")
print(f"Redis可用: {health['redis_available']}")
print(f"内存使用: {health['memory_usage']:.1f}%")
print(f"CPU使用: {health['cpu_usage']:.1f}%")
```

**2. 定时检查脚本**：
```bash
# crontab -e
*/5 * * * * /path/to/venv/bin/python /path/to/scripts/health_check.py
```

**3. 告警通知**：
```yaml
# config/alert_config.yaml
monitoring:
  health_check_interval: 300  # 5分钟
  alerts:
    - condition: "cpu_usage > 90"
      action: "send_alert"
    - condition: "memory_usage > 85"
      action: "send_alert"
    - condition: "mt5_disconnected"
      action: "emergency_stop"
```

---

## 性能优化

### Q21: 如何提高订单执行速度？

**A:** 优化方法：

**1. 使用VPS**：
- 选择靠近经纪商服务器的VPS
- 推荐: AWS, Vultr, DigitalOcean

**2. 优化网络**：
```yaml
mt5:
  timeout: 5000  # 减少超时时间
  max_retries: 2  # 减少重试次数
```

**3. 异步处理**：
```python
# 使用异步订单执行
async def execute_orders_async(signals):
    tasks = [execute_order_async(signal) for signal in signals]
    results = await asyncio.gather(*tasks)
    return results
```

**4. 预计算指标**：
```python
# 缓存指标计算结果
@lru_cache(maxsize=100)
def calculate_indicators(symbol, timeframe):
    # 计算指标
    pass
```

### Q22: 如何减少LLM推理时间？

**A:** 优化LLM性能：

**1. 使用GPU**：
```yaml
llm:
  use_gpu: true
  device: "cuda"  # 或 "mps" for Mac
```

**2. 量化模型**：
```python
# 使用4位量化
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "models/llama-3.2-1b",
    load_in_4bit=True,
    device_map="auto"
)
```

**3. 批处理**：
```python
# 批量处理多个分析请求
results = llm_analyst.analyze_batch(symbols)
```

**4. 缓存结果**：
```python
# 缓存LLM分析结果
@cache_result(ttl=3600)  # 缓存1小时
def analyze_news(symbol):
    return llm_analyst.analyze_news_sentiment(symbol)
```

---

## 安全和风险

### Q23: 如何保护账户安全？

**A:** 安全措施：

**1. 配置文件加密**：
```python
# 加密敏感信息
from cryptography.fernet import Fernet

key = Fernet.generate_key()
cipher = Fernet(key)

encrypted_password = cipher.encrypt(b"your_password")
```

**2. 使用环境变量**：
```bash
# .env
MT5_PASSWORD=your_password
TELEGRAM_TOKEN=your_token

# 在代码中
import os
password = os.getenv('MT5_PASSWORD')
```

**3. 限制API访问**：
```yaml
security:
  allowed_ips:
    - "192.168.1.100"
  require_authentication: true
```

**4. 定期更改密码**：
- 每3个月更改MT5密码
- 使用强密码
- 启用两步验证（如果支持）

### Q24: 如何防止过度交易？

**A:** 控制措施：

**1. 设置交易限制**：
```yaml
risk:
  max_trades_per_day: 10
  max_trades_per_week: 30
  min_trade_interval: 3600  # 1小时
```

**2. 冷却期**：
```yaml
risk:
  consecutive_loss_limit: 3
  cooldown_period: 86400  # 24小时
```

**3. 资金使用限制**：
```yaml
risk:
  max_capital_usage: 0.5  # 最多使用50%资金
  reserve_capital: 0.5  # 保留50%储备
```

### Q25: 如何应对黑天鹅事件？

**A:** 应急措施：

**1. 紧急停止**：
```python
# 一键停止所有交易
from src.bridge.ea31337_bridge import EA31337Bridge

bridge = EA31337Bridge()
bridge.emergency_stop()
```

**2. 自动熔断**：
```yaml
risk:
  circuit_breaker:
    enabled: true
    daily_loss_threshold: 0.05  # 日亏损5%触发
    volatility_threshold: 3.0  # 波动率超过3倍触发
```

**3. 新闻事件过滤**：
```yaml
trading:
  avoid_high_impact_news: true
  news_buffer_minutes: 30  # 新闻前后30分钟不交易
```

**4. 最大回撤保护**：
```yaml
risk:
  max_drawdown_stop: 0.20  # 回撤20%停止交易
  require_manual_restart: true
```

---

## 获取更多帮助

### 文档资源

- [用户手册](user_manual.md)
- [API参考](api_reference.md)
- [架构指南](architecture_guide.md)
- [故障排除](troubleshooting_guide.md)

### 社区支持

- GitHub Issues
- Discord社区
- Telegram群组

### 专业支持

- 技术支持邮箱
- 一对一咨询
- 定制开发服务

---

**还有其他问题？** 欢迎提交Issue或加入社区讨论！
