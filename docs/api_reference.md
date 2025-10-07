# API参考文档

## 概述

本文档提供LLM-RL MT5交易系统的完整API参考，包括所有核心模块、类和方法的详细说明。

## 目录

- [核心数据模型](#核心数据模型)
- [配置管理](#配置管理)
- [EA31337桥接](#ea31337桥接)
- [数据管理](#数据管理)
- [策略系统](#策略系统)
- [智能Agent](#智能agent)
- [风险管理](#风险管理)
- [强化学习](#强化学习)

---

## 核心数据模型

### MarketData

市场数据模型，包含OHLCV数据和技术指标。

```python
@dataclass
class MarketData:
    symbol: str              # 交易品种
    timeframe: str           # 时间周期
    timestamp: datetime      # 时间戳
    ohlcv: pd.DataFrame     # OHLCV数据
    indicators: Dict[str, float]  # 技术指标
    spread: float           # 点差
    liquidity: float        # 流动性
    volume_profile: Dict[str, float]  # 成交量分布
```

**方法：**
- `__post_init__()`: 数据验证

**使用示例：**
```python
from src.core.models import MarketData
import pandas as pd
from datetime import datetime

market_data = MarketData(
    symbol="EURUSD",
    timeframe="H1",
    timestamp=datetime.now(),
    ohlcv=df,
    indicators={"RSI": 65.5, "MACD": 0.0012},
    spread=1.5
)
```

### Signal

交易信号模型，表示策略生成的交易建议。

```python
@dataclass
class Signal:
    strategy_id: str        # 策略ID
    symbol: str             # 交易品种
    direction: int          # 方向: 1=买入, -1=卖出, 0=平仓
    strength: float         # 信号强度 (0-1)
    entry_price: float      # 入场价格
    sl: float              # 止损价格
    tp: float              # 止盈价格
    size: float            # 交易手数
    confidence: float      # 置信度 (0-1)
    timestamp: datetime    # 时间戳
    metadata: Dict[str, Any]  # 元数据
```

**验证规则：**
- `direction` 必须为 -1, 0, 或 1
- `strength` 和 `confidence` 必须在 0-1 之间
- `size` 必须大于 0

**使用示例：**
```python
from src.core.models import Signal
from datetime import datetime

signal = Signal(
    strategy_id="trend_following",
    symbol="EURUSD",
    direction=1,
    strength=0.8,
    entry_price=1.1050,
    sl=1.1000,
    tp=1.1150,
    size=0.1,
    confidence=0.75,
    timestamp=datetime.now(),
    metadata={"indicator": "MA_crossover"}
)
```

### Account

账户信息模型。

```python
@dataclass
class Account:
    account_id: str
    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level: float
    currency: str
    leverage: int
```

**方法：**
- `get_available_margin() -> float`: 获取可用保证金
- `can_open_position(required_margin: float) -> bool`: 检查是否可以开仓
- `get_margin_level_percent() -> float`: 获取保证金水平百分比

**使用示例：**
```python
account = Account(
    account_id="12345",
    balance=10000.0,
    equity=10500.0,
    margin=1000.0,
    free_margin=9500.0,
    margin_level=1050.0,
    currency="USD",
    leverage=100
)

if account.can_open_position(500.0):
    print("可以开仓")
```

### Position

持仓模型。

```python
@dataclass
class Position:
    position_id: str
    symbol: str
    type: PositionType      # LONG or SHORT
    volume: float
    open_price: float
    current_price: float
    sl: float
    tp: float
    profit: float
    swap: float
    commission: float
    open_time: datetime
```

**方法：**
- `calculate_unrealized_pnl(current_price: float = None) -> float`: 计算未实现盈亏
- `update_current_price(price: float) -> None`: 更新当前价格
- `is_profitable() -> bool`: 检查是否盈利

### Trade

交易记录模型。

```python
@dataclass
class Trade:
    trade_id: str
    symbol: str
    type: TradeType
    volume: float
    open_price: float
    close_price: float
    sl: float
    tp: float
    profit: float
    commission: float
    swap: float
    open_time: datetime
    close_time: Optional[datetime]
    strategy_id: str
```

**方法：**
- `calculate_pnl() -> float`: 计算已实现盈亏
- `get_duration() -> Optional[float]`: 获取交易持续时间(小时)
- `is_closed() -> bool`: 检查交易是否已关闭

---

## 配置管理

### SystemConfig

系统总配置类，包含所有子配置。

```python
@dataclass
class SystemConfig:
    database: DatabaseConfig
    mt5: MT5Config
    risk: RiskConfig
    llm: LLMConfig
    rl: RLConfig
    logging: LoggingConfig
    trading: TradingConfig
    debug_mode: bool
    simulation_mode: bool
```

**方法：**
- `validate() -> bool`: 验证配置有效性

### ConfigManager

配置管理器，负责加载、保存和管理配置。

```python
class ConfigManager:
    def __init__(self, config_dir: str = "config")
```

**方法：**

#### load_config
```python
def load_config(self, config_file: str = "config.yaml") -> SystemConfig
```
加载配置文件。

**参数：**
- `config_file`: 配置文件名

**返回：**
- `SystemConfig`: 系统配置对象

**示例：**
```python
from src.core.config import ConfigManager

config_manager = ConfigManager()
config = config_manager.load_config("config.yaml")
```

#### save_config
```python
def save_config(self, config_file: str = "config.yaml") -> bool
```
保存配置文件。

#### get_symbol_config
```python
def get_symbol_config(self, symbol: str) -> Optional[Dict[str, Any]]
```
获取特定品种的配置。

**参数：**
- `symbol`: 品种名称（如 "EURUSD"）

**返回：**
- 品种配置字典或 None

**示例：**
```python
eurusd_config = config_manager.get_symbol_config("EURUSD")
if eurusd_config:
    print(f"点差限制: {eurusd_config['spread_limit']}")
```

---

## EA31337桥接

### EA31337Bridge

EA31337框架集成桥接器，提供与EA31337的文件通信接口。

```python
class EA31337Bridge:
    def __init__(self, config_path: str = "ea31337", signal_timeout: int = 30)
```

**参数：**
- `config_path`: EA31337配置文件路径
- `signal_timeout`: 信号超时时间(秒)

**方法：**

#### is_connected
```python
def is_connected(self) -> bool
```
检查与EA31337的连接状态。

**返回：**
- `bool`: 连接状态

#### get_signals
```python
def get_signals(self) -> List[Signal]
```
获取EA生成的交易信号。

**返回：**
- `List[Signal]`: 交易信号列表

**异常：**
- `BridgeError`: 获取信号失败

**示例：**
```python
from src.bridge.ea31337_bridge import EA31337Bridge

bridge = EA31337Bridge("ea31337")
if bridge.is_connected():
    signals = bridge.get_signals()
    for signal in signals:
        print(f"信号: {signal.symbol} {signal.direction}")
```

#### get_status
```python
def get_status(self) -> Optional[Dict[str, Any]]
```
获取EA运行状态。

**返回：**
- EA状态信息字典或 None

#### send_command
```python
def send_command(self, command: str, params: Dict[str, Any] = None) -> bool
```
向EA发送控制命令。

**参数：**
- `command`: 命令名称
- `params`: 命令参数

**返回：**
- `bool`: 命令发送是否成功

**支持的命令：**
- `update_parameters`: 更新策略参数
- `start_strategy`: 启动策略
- `stop_strategy`: 停止策略
- `emergency_stop`: 紧急停止

**示例：**
```python
# 更新策略参数
bridge.send_command('update_parameters', {
    'strategy': 'trend_following',
    'parameters': {'period': 20, 'threshold': 0.5}
})

# 紧急停止
bridge.emergency_stop()
```

#### update_parameters
```python
def update_parameters(self, strategy: str, params: Dict[str, Any]) -> bool
```
动态更新EA策略参数。

**参数：**
- `strategy`: 策略名称
- `params`: 参数字典

**返回：**
- `bool`: 更新是否成功

#### health_check
```python
def health_check(self) -> Dict[str, Any]
```
执行健康检查。

**返回：**
- 健康检查结果字典

**示例：**
```python
health = bridge.health_check()
print(f"桥接连接: {health['bridge_connected']}")
print(f"EA状态: {health.get('ea_status', 'unknown')}")
```

---

## 数据管理

### DataPipeline

统一数据获取和处理管道。

```python
class DataPipeline:
    def __init__(self)
```

**方法：**

#### get_realtime_data
```python
def get_realtime_data(self, symbol: str, timeframe: str) -> MarketData
```
获取实时市场数据。

**参数：**
- `symbol`: 交易品种
- `timeframe`: 时间周期

**返回：**
- `MarketData`: 市场数据对象

#### get_historical_data
```python
def get_historical_data(self, symbol: str, start: datetime, end: datetime) -> DataFrame
```
获取历史数据。

**参数：**
- `symbol`: 交易品种
- `start`: 开始时间
- `end`: 结束时间

**返回：**
- `DataFrame`: 历史数据

#### cache_data
```python
def cache_data(self, key: str, data: any, ttl: int = 300) -> bool
```
缓存数据到Redis。

**参数：**
- `key`: 缓存键
- `data`: 数据
- `ttl`: 生存时间(秒)

**返回：**
- `bool`: 缓存是否成功

---

## 策略系统

### Strategy

策略基类接口。

```python
class Strategy:
    def generate_signal(self, market_data: MarketData) -> Optional[Signal]
    def update_parameters(self, params: Dict[str, Any]) -> None
    def get_performance_metrics(self) -> RiskMetrics
```

**方法说明：**

#### generate_signal
生成交易信号。

**参数：**
- `market_data`: 市场数据

**返回：**
- `Signal` 或 None

#### update_parameters
更新策略参数。

**参数：**
- `params`: 参数字典

#### get_performance_metrics
获取策略性能指标。

**返回：**
- `RiskMetrics`: 风险指标

**实现示例：**
```python
from src.core.models import Strategy, Signal, MarketData

class MyStrategy(Strategy):
    def __init__(self):
        self.period = 20
        self.threshold = 0.5
    
    def generate_signal(self, market_data: MarketData) -> Optional[Signal]:
        # 实现信号生成逻辑
        if self._check_conditions(market_data):
            return Signal(
                strategy_id="my_strategy",
                symbol=market_data.symbol,
                direction=1,
                strength=0.8,
                entry_price=market_data.ohlcv['close'].iloc[-1],
                sl=self._calculate_sl(market_data),
                tp=self._calculate_tp(market_data),
                size=0.1,
                confidence=0.75,
                timestamp=market_data.timestamp
            )
        return None
    
    def update_parameters(self, params: Dict[str, Any]) -> None:
        if 'period' in params:
            self.period = params['period']
        if 'threshold' in params:
            self.threshold = params['threshold']
```

---

## 智能Agent

### MarketAnalystAgent

市场分析智能体。

```python
class MarketAnalystAgent:
    def __init__(self)
    def analyze_market_state(self, symbol: str) -> MarketState
    def generate_market_outlook(self, symbols: List[str]) -> MarketOutlook
```

### RiskManagerAgent

风险管理智能体。

```python
class RiskManagerAgent:
    def __init__(self, config: RiskConfig)
    def validate_trade(self, signal: Signal, portfolio: Portfolio) -> ValidationResult
    def calculate_position_size(self, signal: Signal, account: Account) -> float
    def monitor_portfolio_risk(self, portfolio: Portfolio) -> RiskMetrics
    def trigger_risk_controls(self, portfolio: Portfolio) -> List[RiskAction]
```

**方法说明：**

#### validate_trade
验证交易信号是否符合风险要求。

**参数：**
- `signal`: 交易信号
- `portfolio`: 投资组合

**返回：**
- `ValidationResult`: 验证结果

#### calculate_position_size
根据风险参数计算仓位大小。

**参数：**
- `signal`: 交易信号
- `account`: 账户信息

**返回：**
- `float`: 建议仓位大小

**示例：**
```python
from src.agents.risk_manager_agent import RiskManagerAgent
from src.core.config import RiskConfig

risk_config = RiskConfig(
    max_risk_per_trade=0.02,
    max_daily_drawdown=0.05
)

risk_agent = RiskManagerAgent(risk_config)
position_size = risk_agent.calculate_position_size(signal, account)
```

### LLMAnalystAgent

LLM分析智能体。

```python
class LLMAnalystAgent:
    def __init__(self, model_path: str)
    def analyze_news_sentiment(self, symbol: str) -> SentimentAnalysis
    def generate_market_commentary(self, market_data: MarketData) -> str
    def analyze_economic_events(self, events: List[EconomicEvent]) -> EventImpact
```

---

## 风险管理

### RiskMetrics

风险指标模型。

```python
@dataclass
class RiskMetrics:
    var_1d: float           # 1日VaR
    var_5d: float           # 5日VaR
    max_drawdown: float     # 最大回撤
    sharpe_ratio: float     # 夏普比率
    sortino_ratio: float    # 索提诺比率
    calmar_ratio: float     # 卡玛比率
    win_rate: float         # 胜率
    profit_factor: float    # 盈利因子
```

**方法：**
- `is_risk_acceptable(max_var: float = 0.05, max_dd: float = 0.20) -> bool`: 检查风险是否可接受

---

## 强化学习

### TradingEnvironment

强化学习交易环境。

```python
class TradingEnvironment(gymnasium.Env):
    def __init__(self, symbol: str, data_provider: DataProvider)
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]
    def reset(self) -> np.ndarray
    def calculate_reward(self, action: int, prev_value: float, current_value: float) -> float
```

### RLTrainer

强化学习训练器。

```python
class RLTrainer:
    def __init__(self, env: TradingEnvironment)
    def train(self, total_timesteps: int) -> None
    def evaluate(self, n_episodes: int = 100) -> Dict[str, float]
    def save_model(self, path: str) -> None
    def load_model(self, path: str) -> None
```

**使用示例：**
```python
from src.rl.trading_env import TradingEnvironment
from src.rl.rl_trainer import RLTrainer

# 创建环境
env = TradingEnvironment("EURUSD", data_provider)

# 创建训练器
trainer = RLTrainer(env)

# 训练模型
trainer.train(total_timesteps=100000)

# 评估模型
metrics = trainer.evaluate(n_episodes=100)
print(f"平均收益: {metrics['mean_reward']}")

# 保存模型
trainer.save_model("models/rl_eurusd.zip")
```

---

## 异常处理

### 自定义异常

```python
class TradingSystemException(Exception):
    """交易系统基础异常"""
    pass

class BridgeError(TradingSystemException):
    """桥接器异常"""
    pass

class ConfigurationException(TradingSystemException):
    """配置异常"""
    pass

class DataException(TradingSystemException):
    """数据异常"""
    pass

class OrderException(TradingSystemException):
    """订单异常"""
    pass
```

**使用示例：**
```python
from src.core.exceptions import BridgeError

try:
    signals = bridge.get_signals()
except BridgeError as e:
    logger.error(f"获取信号失败: {e}")
    # 处理异常
```

---

## 工具函数

### 文件工具

```python
def safe_read_file(file_path: Path) -> Optional[str]
def safe_write_file(file_path: Path, content: str) -> bool
```

### 时间工具

```python
def get_trading_hours(symbol: str) -> Dict[str, str]
def is_market_open(symbol: str, current_time: datetime) -> bool
```

### 计算工具

```python
def calculate_position_size(account_balance: float, risk_percent: float, 
                          stop_loss_pips: float, pip_value: float) -> float
def calculate_pip_value(symbol: str, lot_size: float) -> float
```

---

## 最佳实践

### 1. 配置管理

```python
# 推荐：使用配置管理器
from src.core.config import ConfigManager

config_manager = ConfigManager()
config = config_manager.load_config()

# 访问配置
max_risk = config.risk.max_risk_per_trade
```

### 2. 错误处理

```python
# 推荐：使用try-except处理异常
try:
    signals = bridge.get_signals()
except BridgeError as e:
    logger.error(f"桥接错误: {e}")
    # 实现重试逻辑
except Exception as e:
    logger.critical(f"未知错误: {e}")
    # 紧急停止
```

### 3. 资源管理

```python
# 推荐：使用上下文管理器
with EA31337Bridge("ea31337") as bridge:
    signals = bridge.get_signals()
    # 自动清理资源
```

### 4. 日志记录

```python
# 推荐：使用结构化日志
import logging

logger = logging.getLogger(__name__)
logger.info(f"处理信号: {signal.symbol}", extra={
    'strategy': signal.strategy_id,
    'direction': signal.direction,
    'confidence': signal.confidence
})
```

---

## 版本历史

- **v1.0.0** (2025-01-07): 初始版本
  - 核心数据模型
  - 配置管理系统
  - EA31337桥接器
  - 基础策略系统

---

## 支持

如有问题或建议，请：
- 查看[故障排除指南](troubleshooting_guide.md)
- 提交GitHub Issue
- 查阅[用户手册](user_manual.md)
