# 系统架构指南

## 概述

LLM-RL MT5交易系统采用三层混合架构设计，结合了传统交易执行、现代Python控制和人工智能决策能力。本文档详细介绍系统的架构设计、模块组织和技术选型。

## 目录

- [架构概览](#架构概览)
- [分层设计](#分层设计)
- [核心模块](#核心模块)
- [数据流](#数据流)
- [技术栈](#技术栈)
- [设计模式](#设计模式)
- [扩展性](#扩展性)

---

## 架构概览

### 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      智能决策层 (Python)                      │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  LLM分析模块  │  │  强化学习模块 │  │  智能Agent   │      │
│  │  Llama 3.2   │  │  PPO/SAC     │  │  系统        │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         │                  │                  │             │
│         └──────────────────┴──────────────────┘             │
│                            │                                │
├────────────────────────────┼────────────────────────────────┤
│                   Python Bridge层                           │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  策略管理器   │  │  数据管理器   │  │  参数优化器   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  订单管理器   │  │  风险控制器   │  │  监控系统     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                            │                                │
├────────────────────────────┼────────────────────────────────┤
│                   EA31337执行层 (MQL5)                      │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  EA31337框架  │  │  技术策略集   │  │  指标管理器   │      │
│  │              │  │  30+策略     │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                            │                                │
├────────────────────────────┼────────────────────────────────┤
│                      MT5平台                                │
└─────────────────────────────────────────────────────────────┘
         │                   │                   │
         ▼                   ▼                   ▼
    ┌────────┐          ┌────────┐         ┌────────┐
    │ SQLite │          │ Redis  │         │ 日志系统│
    └────────┘          └────────┘         └────────┘
```

### 架构特点

1. **分层解耦**: 各层职责明确，降低耦合度
2. **双向通信**: Python层与EA31337层通过文件系统通信
3. **异步处理**: 支持异步数据获取和订单执行
4. **模块化设计**: 每个模块可独立开发和测试
5. **可扩展性**: 易于添加新策略、Agent和功能

---

## 分层设计

### 1. EA31337执行层

**职责：**
- 与MT5平台直接交互
- 执行订单和管理持仓
- 提供30+内置技术策略
- 计算技术指标
- 实时数据采集

**技术：**
- MQL5语言
- EA31337框架
- MT5 API

**优势：**
- 成熟稳定的交易执行
- 低延迟订单处理
- 丰富的技术指标库
- 完善的回测支持

**文件结构：**
```
ea31337/
├── EA31337.mq5           # 主EA文件
├── strategies/           # 策略文件
│   ├── trend.set
│   ├── scalp.set
│   └── breakout.set
├── signals.json          # 信号输出
├── status.json           # 状态输出
└── commands.json         # 命令输入
```

### 2. Python Bridge层

**职责：**
- 策略管理和协调
- 参数优化和调整
- 数据处理和缓存
- 订单路由和管理
- 风险控制和监控

**核心组件：**

#### 策略管理器 (StrategyManager)
```python
class StrategyManager:
    """管理多个策略的协调和信号聚合"""
    - 策略注册和加载
    - 信号聚合和冲突解决
    - 策略权重动态调整
    - 性能跟踪和评估
```

#### 数据管理器 (DataPipeline)
```python
class DataPipeline:
    """统一数据获取和处理"""
    - MT5数据获取
    - 数据验证和清洗
    - Redis缓存管理
    - SQLite持久化
```

#### 订单管理器 (OrderManager)
```python
class OrderManager:
    """订单执行和管理"""
    - 订单发送和确认
    - 滑点控制
    - 订单状态跟踪
    - 错误处理和重试
```

#### 风险控制器 (RiskController)
```python
class RiskController:
    """多层级风险控制"""
    - 单笔风险限制
    - 日/周/月回撤控制
    - 相关性管理
    - 熔断机制
```

### 3. 智能决策层

**职责：**
- 市场分析和预测
- 策略优化和学习
- 新闻情绪分析
- 风险评估和建议

**核心组件：**

#### LLM分析模块
```python
class LLMAnalystAgent:
    """本地LLM分析"""
    - 新闻情绪分析
    - 市场评论生成
    - 事件影响评估
    - 异常行情解释
```

**模型选择：**
- Llama 3.2 1B: 快速分析，低资源消耗
- Llama 3.2 3B: 深度分析，更高准确度

#### 强化学习模块
```python
class RLOptimizer:
    """策略权重优化"""
    - PPO/SAC算法
    - 在线学习
    - 策略权重调整
    - 风险收益平衡
```

**环境设计：**
- 状态空间: 价格、指标、持仓、市场状态
- 动作空间: 买入、卖出、持有、调整仓位
- 奖励函数: 收益 - 风险惩罚 - 交易成本

#### 智能Agent系统
```python
# 市场分析Agent
class MarketAnalystAgent:
    - 技术面分析
    - 基本面分析
    - 市场状态检测
    - 相关性分析

# 风险管理Agent
class RiskManagerAgent:
    - VaR计算
    - 最大回撤监控
    - 仓位管理
    - 风险预警

# 执行优化Agent
class ExecutionOptimizerAgent:
    - 滑点预测
    - 最佳入场时机
    - 订单拆分
    - 流动性管理
```

---

## 核心模块

### 1. 核心模块 (src/core/)

```
src/core/
├── models.py          # 数据模型定义
├── config.py          # 配置管理
├── logging.py         # 日志系统
└── exceptions.py      # 异常定义
```

**职责：**
- 定义系统数据结构
- 管理配置文件
- 提供日志功能
- 定义异常类型

### 2. 数据模块 (src/data/)

```
src/data/
├── mt5_connection.py  # MT5连接管理
├── data_pipeline.py   # 数据管道
├── database.py        # 数据库管理
└── cache.py           # 缓存管理
```

**数据流：**
```
MT5平台 → MT5Connection → DataPipeline → Redis缓存
                                      ↓
                                  SQLite数据库
```

### 3. 策略模块 (src/strategies/)

```
src/strategies/
├── base_strategy.py   # 策略基类
├── trend_strategy.py  # 趋势策略
├── scalp_strategy.py  # 剥头皮策略
├── breakout_strategy.py  # 突破策略
└── strategy_manager.py   # 策略管理器
```

**策略生命周期：**
```
初始化 → 参数配置 → 信号生成 → 性能评估 → 参数优化
   ↑                                          ↓
   └──────────────────────────────────────────┘
```

### 4. Agent模块 (src/agents/)

```
src/agents/
├── market_analyst_agent.py    # 市场分析
├── risk_manager_agent.py      # 风险管理
└── execution_optimizer_agent.py  # 执行优化
```

### 5. 桥接模块 (src/bridge/)

```
src/bridge/
├── ea31337_bridge.py  # EA31337桥接器
└── order_executor.py  # 订单执行器
```

**通信机制：**
```
Python → commands.json → EA31337
Python ← signals.json  ← EA31337
Python ← status.json   ← EA31337
```

### 6. LLM模块 (src/llm/)

```
src/llm/
├── llm_model.py       # LLM模型封装
├── llm_analyst.py     # LLM分析器
└── llm_optimizer.py   # LLM调用优化
```

### 7. 强化学习模块 (src/rl/)

```
src/rl/
├── trading_env.py     # 交易环境
├── rl_trainer.py      # RL训练器
└── rl_optimizer.py    # RL优化器
```

### 8. 工具模块 (src/utils/)

```
src/utils/
├── file_utils.py      # 文件工具
├── time_utils.py      # 时间工具
├── calc_utils.py      # 计算工具
└── indicators.py      # 技术指标
```

---

## 数据流

### 1. 实时数据流

```
MT5平台 (实时行情)
    ↓
MT5Connection (连接管理)
    ↓
DataPipeline (数据处理)
    ├→ Redis (缓存, TTL=5分钟)
    └→ SQLite (持久化)
    ↓
StrategyManager (策略处理)
    ↓
Agents (智能分析)
    ↓
OrderManager (订单执行)
    ↓
EA31337 (执行)
    ↓
MT5平台 (订单执行)
```

### 2. 信号处理流

```
EA31337策略 → signals.json
                ↓
Python策略 → StrategyManager (信号聚合)
                ↓
         RiskManagerAgent (风险验证)
                ↓
         ExecutionOptimizerAgent (执行优化)
                ↓
         OrderManager (订单发送)
                ↓
         EA31337 → MT5平台
```

### 3. 学习反馈流

```
交易执行 → 性能数据
            ↓
    PerformanceTracker (性能跟踪)
            ↓
    RLOptimizer (强化学习)
            ↓
    策略权重调整
            ↓
    StrategyManager (应用新权重)
```

---

## 技术栈

### 编程语言

| 层级 | 语言 | 用途 |
|------|------|------|
| 执行层 | MQL5 | EA31337框架，订单执行 |
| 控制层 | Python 3.9+ | 策略管理，数据处理 |
| 配置层 | YAML/JSON | 配置文件 |

### 核心库

#### Python依赖
```python
# 数据处理
pandas>=1.5.0
numpy>=1.23.0
ta-lib>=0.4.0

# MT5集成
MetaTrader5>=5.0.0

# 机器学习
stable-baselines3>=2.0.0
gymnasium>=0.28.0
scikit-learn>=1.2.0
optuna>=3.0.0

# LLM
transformers>=4.30.0
llama-cpp-python>=0.2.0
langchain>=0.1.0

# 数据存储
redis>=4.5.0
sqlalchemy>=2.0.0

# 工具
pyyaml>=6.0
python-telegram-bot>=20.0
```

### 数据存储

| 类型 | 技术 | 用途 |
|------|------|------|
| 关系数据库 | SQLite | 交易记录，历史数据 |
| 缓存 | Redis | 实时数据，临时状态 |
| 文件系统 | JSON/YAML | 配置，通信文件 |

### 部署环境

**本地部署：**
- Windows 10/11
- 4核8GB (最低)
- 8核16GB (推荐)

**云端部署：**
- VPS 2核4GB (最低)
- VPS 4核8GB (推荐)
- Ubuntu 20.04/22.04

---

## 设计模式

### 1. 策略模式 (Strategy Pattern)

用于策略系统，允许动态切换不同的交易策略。

```python
class Strategy(ABC):
    @abstractmethod
    def generate_signal(self, market_data: MarketData) -> Optional[Signal]:
        pass

class TrendStrategy(Strategy):
    def generate_signal(self, market_data: MarketData) -> Optional[Signal]:
        # 趋势策略实现
        pass

class ScalpStrategy(Strategy):
    def generate_signal(self, market_data: MarketData) -> Optional[Signal]:
        # 剥头皮策略实现
        pass
```

### 2. 观察者模式 (Observer Pattern)

用于事件监控和告警系统。

```python
class EventObserver(ABC):
    @abstractmethod
    def update(self, event: Event) -> None:
        pass

class TelegramNotifier(EventObserver):
    def update(self, event: Event) -> None:
        # 发送Telegram通知
        pass

class EmailNotifier(EventObserver):
    def update(self, event: Event) -> None:
        # 发送邮件通知
        pass
```

### 3. 单例模式 (Singleton Pattern)

用于配置管理器和连接管理器。

```python
class ConfigManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

### 4. 工厂模式 (Factory Pattern)

用于创建不同类型的策略和Agent。

```python
class StrategyFactory:
    @staticmethod
    def create_strategy(strategy_type: str) -> Strategy:
        if strategy_type == "trend":
            return TrendStrategy()
        elif strategy_type == "scalp":
            return ScalpStrategy()
        elif strategy_type == "breakout":
            return BreakoutStrategy()
        else:
            raise ValueError(f"未知策略类型: {strategy_type}")
```

### 5. 适配器模式 (Adapter Pattern)

用于EA31337桥接器，适配不同的通信接口。

```python
class EA31337Bridge:
    """适配EA31337的文件通信接口"""
    def get_signals(self) -> List[Signal]:
        # 从文件读取并转换为Signal对象
        pass
```

---

## 扩展性

### 1. 添加新策略

```python
# 1. 创建策略类
class MyNewStrategy(Strategy):
    def generate_signal(self, market_data: MarketData) -> Optional[Signal]:
        # 实现策略逻辑
        pass

# 2. 注册策略
strategy_manager.register_strategy("my_new_strategy", MyNewStrategy())

# 3. 配置启用
# config.yaml
trading:
  strategies_enabled:
    - my_new_strategy
```

### 2. 添加新Agent

```python
# 1. 创建Agent类
class MyNewAgent:
    def analyze(self, data: Any) -> Any:
        # 实现分析逻辑
        pass

# 2. 集成到系统
trading_system.add_agent("my_new_agent", MyNewAgent())
```

### 3. 添加新数据源

```python
# 1. 实现DataProvider接口
class MyDataProvider(DataProvider):
    def get_market_data(self, symbol: str, timeframe: str) -> MarketData:
        # 实现数据获取
        pass

# 2. 注册数据源
data_pipeline.register_provider("my_provider", MyDataProvider())
```

### 4. 添加新指标

```python
# 1. 在indicators.py中添加
def my_custom_indicator(data: pd.DataFrame, period: int) -> pd.Series:
    # 实现指标计算
    pass

# 2. 在策略中使用
indicator_value = my_custom_indicator(market_data.ohlcv, 20)
```

---

## 性能优化

### 1. 数据缓存

```python
# Redis缓存热数据
cache_key = f"market_data:{symbol}:{timeframe}"
cached_data = redis_client.get(cache_key)
if cached_data:
    return pickle.loads(cached_data)
```

### 2. 异步处理

```python
# 异步获取多个品种数据
async def get_multiple_symbols_data(symbols: List[str]):
    tasks = [get_market_data_async(symbol) for symbol in symbols]
    return await asyncio.gather(*tasks)
```

### 3. 批量处理

```python
# 批量计算指标
indicators = calculate_indicators_batch(market_data_list)
```

### 4. 连接池

```python
# 数据库连接池
engine = create_engine(
    'sqlite:///data/trading.db',
    pool_size=10,
    max_overflow=20
)
```

---

## 安全性

### 1. 配置加密

```python
# 敏感信息加密存储
from cryptography.fernet import Fernet

def encrypt_password(password: str, key: bytes) -> str:
    f = Fernet(key)
    return f.encrypt(password.encode()).decode()
```

### 2. API密钥管理

```python
# 使用环境变量
import os
mt5_password = os.getenv('MT5_PASSWORD')
```

### 3. 日志脱敏

```python
# 敏感信息脱敏
logger.info(f"账户: {account_id[:4]}****")
```

---

## 监控和维护

### 1. 健康检查

```python
def system_health_check() -> Dict[str, bool]:
    return {
        'mt5_connected': mt5_connection.is_connected(),
        'redis_available': redis_client.ping(),
        'database_accessible': database.test_connection(),
        'ea31337_responsive': bridge.is_connected()
    }
```

### 2. 性能监控

```python
# 监控关键指标
metrics = {
    'signal_latency': measure_signal_latency(),
    'order_execution_time': measure_execution_time(),
    'memory_usage': get_memory_usage(),
    'cpu_usage': get_cpu_usage()
}
```

### 3. 日志管理

```python
# 结构化日志
logger.info("订单执行", extra={
    'symbol': 'EURUSD',
    'direction': 'BUY',
    'volume': 0.1,
    'execution_time': 0.05
})
```

---

## 总结

本系统采用分层架构设计，具有以下优势：

1. **模块化**: 各模块职责清晰，易于维护
2. **可扩展**: 支持添加新策略、Agent和功能
3. **高性能**: 通过缓存、异步处理优化性能
4. **可靠性**: 多层错误处理和健康检查
5. **智能化**: 集成LLM和RL技术

系统设计遵循SOLID原则，使用成熟的设计模式，确保代码质量和可维护性。
