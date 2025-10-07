# 用户手册

## 欢迎使用LLM-RL MT5交易系统

本手册将指导您完成系统的安装、配置和使用。无论您是初次使用还是有经验的用户，都能在这里找到所需的信息。

## 目录

- [系统要求](#系统要求)
- [安装指南](#安装指南)
- [快速开始](#快速开始)
- [配置指南](#配置指南)
- [使用教程](#使用教程)
- [最佳实践](#最佳实践)
- [常见问题](#常见问题)

---

## 系统要求

### 硬件要求

**最低配置：**
- CPU: 4核心
- 内存: 8GB RAM
- 硬盘: 20GB可用空间
- 网络: 稳定的互联网连接

**推荐配置：**
- CPU: 8核心或更高
- 内存: 16GB RAM或更高
- 硬盘: 50GB SSD
- 网络: 低延迟宽带连接

### 软件要求

**必需软件：**
- Windows 10/11 (64位) 或 Ubuntu 20.04+
- Python 3.9 或更高版本
- MetaTrader 5 平台
- Redis 服务器

**可选软件：**
- CUDA (用于GPU加速LLM)
- Docker (用于容器化部署)

---

## 安装指南

### 步骤1: 安装Python

#### Windows
1. 访问 [python.org](https://www.python.org/downloads/)
2. 下载Python 3.9+安装程序
3. 运行安装程序，**勾选"Add Python to PATH"**
4. 验证安装：
```bash
python --version
```

#### Linux
```bash
sudo apt update
sudo apt install python3.9 python3-pip
python3 --version
```

### 步骤2: 安装MetaTrader 5

1. 访问您的经纪商网站
2. 下载MT5平台
3. 安装并登录您的交易账户
4. 记录以下信息：
   - 服务器名称
   - 账户号码
   - 密码

### 步骤3: 安装Redis

#### Windows
1. 下载Redis for Windows: [GitHub](https://github.com/microsoftarchive/redis/releases)
2. 解压并运行 `redis-server.exe`
3. 或使用Windows服务：
```bash
redis-server --service-install
redis-server --service-start
```

#### Linux
```bash
sudo apt install redis-server
sudo systemctl start redis
sudo systemctl enable redis
```

验证Redis运行：
```bash
redis-cli ping
# 应返回: PONG
```

### 步骤4: 下载交易系统

```bash
# 克隆仓库
git clone https://github.com/your-repo/llm-rl-mt5-trading.git
cd llm-rl-mt5-trading

# 或下载ZIP文件并解压
```

### 步骤5: 安装Python依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 步骤6: 下载LLM模型（可选）

如果您想使用LLM分析功能：

```bash
# 安装Hugging Face CLI
pip install huggingface-hub

# 下载Llama 3.2 1B模型
huggingface-cli download meta-llama/Llama-3.2-1B --local-dir models/llama-3.2-1b

# 或下载3B模型（需要更多内存）
huggingface-cli download meta-llama/Llama-3.2-3B --local-dir models/llama-3.2-3b
```

### 步骤7: 验证安装

```bash
# 运行安装验证脚本
python validate_setup.py
```

**预期输出：**
```
✓ Python版本: 3.9.x
✓ 所有依赖已安装
✓ MT5已安装
✓ Redis运行中
✓ 目录结构正确
⚠ LLM模型未找到（可选）

安装验证完成！
```

---

## 快速开始

### 第一次运行

#### 1. 配置MT5连接

编辑 `config/config.yaml`：

```yaml
mt5:
  server: "YourBroker-Server"  # 您的经纪商服务器
  login: 12345678              # 您的账户号码
  password: "your_password"    # 您的密码
  path: "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
```

#### 2. 配置风险参数

```yaml
risk:
  max_risk_per_trade: 0.02     # 单笔最大风险2%
  max_daily_drawdown: 0.05     # 日最大回撤5%
  max_positions: 10            # 最大持仓数
```

#### 3. 选择交易品种

```yaml
trading:
  symbols:
    - EURUSD
    - XAUUSD
    - USOIL
  timeframes:
    - M5
    - M15
    - H1
    - H4
```

#### 4. 启动系统

```bash
# 模拟模式（推荐首次使用）
python main.py --simulation

# 实盘模式（谨慎使用）
python main.py
```

#### 5. 监控系统

打开浏览器访问：
- 日志: `logs/trading.log`
- 报告: `logs/reports/`

---

## 配置指南

### 基础配置

#### MT5连接配置

```yaml
mt5:
  server: "Broker-Server"      # 经纪商服务器
  login: 12345678              # 账户号码
  password: "password"         # 密码
  path: "path/to/terminal64.exe"  # MT5路径
  timeout: 60000               # 超时时间(毫秒)
```

**如何找到服务器名称：**
1. 打开MT5
2. 点击"工具" → "选项"
3. 查看"服务器"标签

#### 数据库配置

```yaml
database:
  sqlite_path: "data/trading.db"  # SQLite数据库路径
  redis_host: "localhost"         # Redis主机
  redis_port: 6379                # Redis端口
  redis_db: 0                     # Redis数据库编号
```

### 风险管理配置

#### 基础风险参数

```yaml
risk:
  max_risk_per_trade: 0.02     # 单笔最大风险2%
  max_daily_drawdown: 0.05     # 日最大回撤5%
  max_weekly_drawdown: 0.10    # 周最大回撤10%
  max_monthly_drawdown: 0.15   # 月最大回撤15%
  max_positions: 10            # 最大持仓数
  max_lot_per_symbol: 1.0      # 单品种最大手数
```

**风险参数说明：**

- **max_risk_per_trade**: 每笔交易最多可以亏损账户的百分比
  - 保守: 0.01 (1%)
  - 适中: 0.02 (2%)
  - 激进: 0.03 (3%)

- **max_daily_drawdown**: 单日最大亏损限制
  - 触发后暂停交易24小时

- **max_positions**: 同时持有的最大仓位数
  - 分散风险，避免过度集中

#### 止损止盈配置

```yaml
risk:
  stop_loss_pct: 0.02          # 默认止损2%
  take_profit_pct: 0.04        # 默认止盈4%
```

### 交易配置

#### 品种配置

```yaml
trading:
  symbols:
    - EURUSD
    - GBPUSD
    - XAUUSD
    - USOIL
  default_lot_size: 0.01       # 默认手数
  slippage: 3                  # 允许滑点(点)
  magic_number: 123456         # EA魔术号
```

#### 策略配置

```yaml
trading:
  strategies_enabled:
    - trend                    # 趋势策略
    - scalp                    # 剥头皮策略
    - breakout                 # 突破策略
```

### LLM配置

```yaml
llm:
  model_path: "models/llama-3.2-1b"  # 模型路径
  model_type: "llama"                # 模型类型
  max_tokens: 512                    # 最大token数
  temperature: 0.7                   # 温度参数
  use_gpu: true                      # 使用GPU
```

**LLM使用建议：**

- **1B模型**: 快速分析，适合实时使用
- **3B模型**: 深度分析，适合离线分析
- **GPU加速**: 显著提升推理速度

### 强化学习配置

```yaml
rl:
  algorithm: "PPO"             # 算法: PPO, SAC, A2C
  learning_rate: 0.0003        # 学习率
  n_steps: 2048                # 步数
  batch_size: 64               # 批次大小
```

### 日志配置

```yaml
logging:
  level: "INFO"                # 日志级别
  file_path: "logs/trading.log"  # 日志文件
  console_output: true         # 控制台输出
```

**日志级别：**
- **DEBUG**: 详细调试信息
- **INFO**: 一般信息（推荐）
- **WARNING**: 警告信息
- **ERROR**: 错误信息

---

## 使用教程

### 教程1: 运行第一个策略

#### 步骤1: 选择策略

编辑 `config/config.yaml`：

```yaml
trading:
  strategies_enabled:
    - trend  # 启用趋势策略
```

#### 步骤2: 配置品种

```yaml
trading:
  symbols:
    - EURUSD  # 从单一品种开始
```

#### 步骤3: 设置保守风险

```yaml
risk:
  max_risk_per_trade: 0.01  # 1%风险
  max_positions: 3          # 最多3个仓位
```

#### 步骤4: 启动模拟模式

```bash
python main.py --simulation
```

#### 步骤5: 观察日志

```bash
tail -f logs/trading.log
```

**预期输出：**
```
INFO: 系统启动
INFO: MT5连接成功
INFO: 加载策略: trend
INFO: 开始监控: EURUSD
INFO: 生成信号: EURUSD BUY 强度:0.75
INFO: 风险验证通过
INFO: 订单执行成功
```

### 教程2: 配置多品种交易

#### 步骤1: 创建品种配置

创建 `config/symbols/eurusd.yaml`：

```yaml
symbol: EURUSD
spread_limit: 2.0            # 点差限制
min_lot: 0.01                # 最小手数
max_lot: 1.0                 # 最大手数
lot_step: 0.01               # 手数步长
risk_multiplier: 1.0         # 风险倍数

strategies:
  - trend
  - scalp

timeframes:
  - M5
  - M15
  - H1

trading_hours:
  monday: "00:00-23:59"
  tuesday: "00:00-23:59"
  wednesday: "00:00-23:59"
  thursday: "00:00-23:59"
  friday: "00:00-22:00"
```

创建 `config/symbols/xauusd.yaml`：

```yaml
symbol: XAUUSD
spread_limit: 5.0            # 黄金点差较大
min_lot: 0.01
max_lot: 0.5                 # 黄金波动大，降低最大手数
lot_step: 0.01
risk_multiplier: 0.8         # 降低风险倍数

strategies:
  - breakout                 # 黄金适合突破策略

timeframes:
  - M15
  - H1
  - H4
```

#### 步骤2: 启用多品种

```yaml
# config.yaml
trading:
  symbols:
    - EURUSD
    - XAUUSD
```

#### 步骤3: 设置相关性控制

```yaml
risk:
  correlation_threshold: 0.7  # 相关性阈值
  max_correlated_positions: 2  # 最多2个相关仓位
```

### 教程3: 使用LLM分析

#### 步骤1: 确认模型已下载

```bash
ls models/llama-3.2-1b/
# 应看到: config.json, pytorch_model.bin等
```

#### 步骤2: 启用LLM

```yaml
# config.yaml
llm:
  enabled: true
  model_path: "models/llama-3.2-1b"
  call_interval: 300  # 每5分钟分析一次
```

#### 步骤3: 配置新闻源

```yaml
llm:
  news_sources:
    - "https://www.forexfactory.com/calendar"
    - "https://www.investing.com/economic-calendar/"
```

#### 步骤4: 查看LLM分析

```bash
# 查看LLM分析日志
grep "LLM" logs/trading.log

# 查看生成的市场评论
cat logs/reports/market_commentary_$(date +%Y%m%d).txt
```

### 教程4: 回测策略

#### 步骤1: 准备历史数据

```python
# scripts/download_historical_data.py
from src.data.data_pipeline import DataPipeline
from datetime import datetime, timedelta

pipeline = DataPipeline()

# 下载3个月历史数据
end_date = datetime.now()
start_date = end_date - timedelta(days=90)

data = pipeline.get_historical_data("EURUSD", start_date, end_date)
data.to_csv("data/eurusd_h1_3months.csv")
```

#### 步骤2: 运行回测

```python
# scripts/run_backtest.py
from src.backtest.backtest_engine import BacktestEngine
from src.strategies.trend_strategy import TrendStrategy

# 创建回测引擎
engine = BacktestEngine(
    initial_balance=10000,
    data_file="data/eurusd_h1_3months.csv"
)

# 添加策略
strategy = TrendStrategy()
engine.add_strategy(strategy)

# 运行回测
results = engine.run()

# 查看结果
print(f"总收益: {results['total_return']:.2%}")
print(f"夏普比率: {results['sharpe_ratio']:.2f}")
print(f"最大回撤: {results['max_drawdown']:.2%}")
print(f"胜率: {results['win_rate']:.2%}")
```

#### 步骤3: 查看回测报告

```bash
# 生成详细报告
python scripts/generate_backtest_report.py

# 查看报告
open logs/reports/backtest_report_$(date +%Y%m%d).html
```

### 教程5: 参数优化

#### 步骤1: 定义优化范围

```python
# scripts/optimize_parameters.py
from src.backtest.parameter_optimizer import ParameterOptimizer

optimizer = ParameterOptimizer()

# 定义参数范围
param_ranges = {
    'ma_period': (10, 50, 5),      # (最小, 最大, 步长)
    'threshold': (0.3, 0.7, 0.1),
    'stop_loss_pct': (0.01, 0.03, 0.005)
}

# 运行优化
best_params = optimizer.optimize(
    strategy='trend',
    symbol='EURUSD',
    param_ranges=param_ranges,
    objective='sharpe_ratio'  # 优化目标
)

print(f"最优参数: {best_params}")
```

#### 步骤2: 应用优化结果

```yaml
# config/symbols/eurusd.yaml
optimize_params:
  ma_period: 25
  threshold: 0.5
  stop_loss_pct: 0.02
```

---

## 最佳实践

### 1. 渐进式资金管理

**阶段1: 测试阶段（10%资金）**
- 运行时间: 2-4周
- 目标: 验证系统稳定性
- 风险: 最低

```yaml
risk:
  max_risk_per_trade: 0.01
  max_positions: 3
  capital_usage: 0.10  # 仅使用10%资金
```

**阶段2: 验证阶段（30%资金）**
- 运行时间: 1-2个月
- 目标: 验证盈利能力
- 风险: 适中

```yaml
risk:
  max_risk_per_trade: 0.02
  max_positions: 5
  capital_usage: 0.30
```

**阶段3: 正式运行（50%资金）**
- 运行时间: 持续
- 目标: 稳定盈利
- 风险: 可控

```yaml
risk:
  max_risk_per_trade: 0.02
  max_positions: 10
  capital_usage: 0.50  # 最多50%
```

### 2. 风险控制原则

**永远不要：**
- ❌ 使用全部资金交易
- ❌ 忽略止损
- ❌ 在亏损后加大仓位
- ❌ 同时持有过多相关品种
- ❌ 在重大新闻前开仓

**始终要：**
- ✓ 设置止损止盈
- ✓ 控制单笔风险在2%以内
- ✓ 分散投资多个品种
- ✓ 定期检查系统状态
- ✓ 保留至少50%资金作为储备

### 3. 策略选择建议

**趋势市场：**
```yaml
strategies_enabled:
  - trend
  - breakout
```

**震荡市场：**
```yaml
strategies_enabled:
  - scalp
  - mean_reversion
```

**不确定市场：**
```yaml
strategies_enabled:
  - trend
  - scalp
  - breakout
# 使用多策略分散风险
```

### 4. 监控和维护

**每日检查：**
```bash
# 运行每日检查脚本
python scripts/daily_check.py
```

检查内容：
- MT5连接状态
- 账户余额和净值
- 持仓情况
- 今日盈亏
- 系统日志

**每周检查：**
- 策略性能评估
- 参数调整
- 数据库备份
- 日志清理

**每月检查：**
- 全面性能分析
- 策略回测
- 系统更新
- 风险评估

### 5. 告警设置

```yaml
# config/alert_config.yaml
alerts:
  telegram:
    enabled: true
    bot_token: "your_bot_token"
    chat_id: "your_chat_id"
  
  email:
    enabled: true
    smtp_server: "smtp.gmail.com"
    smtp_port: 587
    from_email: "your_email@gmail.com"
    to_email: "your_email@gmail.com"
  
  triggers:
    - type: "daily_loss"
      threshold: 0.03  # 日亏损3%
      severity: "high"
    
    - type: "margin_level"
      threshold: 200   # 保证金水平低于200%
      severity: "critical"
    
    - type: "consecutive_losses"
      threshold: 3     # 连续3笔亏损
      severity: "medium"
```

---

## 常见问题

### Q1: 系统可以24小时运行吗？

**A:** 可以，但建议：
- 使用VPS确保稳定性
- 设置自动重启机制
- 配置告警通知
- 定期检查系统状态

### Q2: 最低需要多少资金？

**A:** 建议：
- 最低: $1000（模拟测试）
- 推荐: $5000-$10000（实盘交易）
- 理想: $10000以上

### Q3: 可以同时运行多个策略吗？

**A:** 可以，系统支持多策略并行：
```yaml
strategies_enabled:
  - trend
  - scalp
  - breakout
```

系统会自动进行信号聚合和冲突解决。

### Q4: 如何处理连续亏损？

**A:** 系统内置熔断机制：
```yaml
risk:
  consecutive_loss_limit: 3  # 连续3笔亏损
  pause_duration: 24         # 暂停24小时
```

手动处理：
1. 停止系统
2. 分析亏损原因
3. 调整参数或策略
4. 小仓位重新测试

### Q5: LLM分析准确吗？

**A:** LLM分析作为辅助参考：
- 提供市场情绪分析
- 解释异常行情
- 生成市场评论

不应完全依赖LLM做交易决策，应结合技术分析和风险管理。

### Q6: 如何备份数据？

**A:** 
```bash
# 自动备份
python scripts/backup_database.py

# 手动备份
cp data/trading.db data/trading_backup_$(date +%Y%m%d).db
```

建议每周备份一次。

### Q7: 系统支持哪些经纪商？

**A:** 支持所有提供MT5平台的经纪商，包括：
- IC Markets
- Pepperstone
- FXCM
- Exness
- 等等

### Q8: 可以在Mac上运行吗？

**A:** 可以，但需要：
1. 使用Wine运行MT5
2. 或使用VPS运行Windows
3. Python部分完全支持Mac

### Q9: 如何优化系统性能？

**A:** 
- 使用SSD硬盘
- 增加内存
- 使用GPU加速LLM
- 优化Redis配置
- 减少日志级别

### Q10: 出现问题如何获取帮助？

**A:** 
1. 查看[故障排除指南](troubleshooting_guide.md)
2. 运行诊断脚本: `python diagnose.py`
3. 查看日志: `logs/error.log`
4. 提交GitHub Issue
5. 联系技术支持

---

## 下一步

现在您已经了解了系统的基本使用，建议：

1. **阅读技术文档**: [API参考](api_reference.md)
2. **学习架构设计**: [架构指南](architecture_guide.md)
3. **查看示例代码**: `examples/` 目录
4. **加入社区讨论**: GitHub Discussions

---

## 免责声明

⚠️ **重要提示**:

- 外汇交易存在重大风险，可能导致资金损失
- 本系统仅供学习和研究使用
- 过去的表现不代表未来的结果
- 请在充分了解风险的情况下使用
- 建议先在模拟账户中测试
- 不要投资超过您能承受损失的资金

使用本系统即表示您理解并接受这些风险。

---

**祝您交易顺利！** 🚀
