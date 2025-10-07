"""
核心数据模型定义
定义系统中使用的所有数据结构和接口
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import pandas as pd


class PositionType(Enum):
    """持仓类型"""
    LONG = 1
    SHORT = -1


class TradeType(Enum):
    """交易类型"""
    BUY = 1
    SELL = -1
    CLOSE = 0


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class MarketData:
    """市场数据模型"""
    symbol: str
    timeframe: str
    timestamp: datetime
    ohlcv: pd.DataFrame
    indicators: Dict[str, float] = field(default_factory=dict)
    spread: float = 0.0
    liquidity: float = 0.0
    volume_profile: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """数据验证"""
        if self.ohlcv.empty:
            raise ValueError("OHLCV数据不能为空")
        if self.spread < 0:
            raise ValueError("点差不能为负数")


@dataclass
class Signal:
    """交易信号模型"""
    strategy_id: str
    symbol: str
    direction: int  # 1=买入, -1=卖出, 0=平仓
    strength: float  # 信号强度 0-1
    entry_price: float
    sl: float  # 止损价格
    tp: float  # 止盈价格
    size: float  # 交易手数
    confidence: float  # 信号置信度 0-1
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """信号验证"""
        if not -1 <= self.direction <= 1:
            raise ValueError("交易方向必须为-1, 0, 1")
        if not 0 <= self.strength <= 1:
            raise ValueError("信号强度必须在0-1之间")
        if not 0 <= self.confidence <= 1:
            raise ValueError("信号置信度必须在0-1之间")
        if self.size <= 0:
            raise ValueError("交易手数必须大于0")


@dataclass
class Account:
    """账户信息模型"""
    account_id: str
    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level: float
    currency: str
    leverage: int
    server: str = ""
    company: str = ""
    
    def get_available_margin(self) -> float:
        """获取可用保证金"""
        return self.free_margin
    
    def can_open_position(self, required_margin: float) -> bool:
        """检查是否可以开仓"""
        return self.free_margin >= required_margin
    
    def get_margin_level_percent(self) -> float:
        """获取保证金水平百分比"""
        if self.margin == 0:
            return float('inf')
        return (self.equity / self.margin) * 100


@dataclass
class Position:
    """持仓模型"""
    position_id: str
    symbol: str
    type: PositionType
    volume: float
    open_price: float
    current_price: float
    sl: float = 0.0
    tp: float = 0.0
    profit: float = 0.0
    swap: float = 0.0
    commission: float = 0.0
    open_time: datetime = field(default_factory=datetime.now)
    comment: str = ""
    magic_number: int = 0
    
    def calculate_unrealized_pnl(self, current_price: float = None) -> float:
        """计算未实现盈亏"""
        if current_price is None:
            current_price = self.current_price
            
        if self.type == PositionType.LONG:
            return (current_price - self.open_price) * self.volume
        else:
            return (self.open_price - current_price) * self.volume
    
    def update_current_price(self, price: float) -> None:
        """更新当前价格"""
        self.current_price = price
        self.profit = self.calculate_unrealized_pnl(price)
    
    def is_profitable(self) -> bool:
        """检查是否盈利"""
        return self.profit > 0


@dataclass
class Trade:
    """交易记录模型"""
    trade_id: str
    symbol: str
    type: TradeType
    volume: float
    open_price: float
    close_price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    profit: float = 0.0
    commission: float = 0.0
    swap: float = 0.0
    open_time: datetime = field(default_factory=datetime.now)
    close_time: Optional[datetime] = None
    strategy_id: str = ""
    comment: str = ""
    magic_number: int = 0
    
    def calculate_pnl(self) -> float:
        """计算已实现盈亏"""
        if self.close_price == 0:
            return 0.0
            
        if self.type == TradeType.BUY:
            return (self.close_price - self.open_price) * self.volume
        else:
            return (self.open_price - self.close_price) * self.volume
    
    def get_duration(self) -> Optional[float]:
        """获取交易持续时间(小时)"""
        if self.close_time is None:
            return None
        return (self.close_time - self.open_time).total_seconds() / 3600
    
    def is_closed(self) -> bool:
        """检查交易是否已关闭"""
        return self.close_time is not None


@dataclass
class RiskMetrics:
    """风险指标模型"""
    var_1d: float  # 1日VaR
    var_5d: float  # 5日VaR
    max_drawdown: float  # 最大回撤
    sharpe_ratio: float  # 夏普比率
    sortino_ratio: float  # 索提诺比率
    calmar_ratio: float  # 卡玛比率
    win_rate: float  # 胜率
    profit_factor: float  # 盈利因子
    
    def is_risk_acceptable(self, max_var: float = 0.05, max_dd: float = 0.20) -> bool:
        """检查风险是否可接受"""
        return self.var_1d <= max_var and abs(self.max_drawdown) <= max_dd


@dataclass
class MarketState:
    """市场状态模型"""
    trend: str  # "uptrend", "downtrend", "sideways"
    volatility: float  # 波动率
    regime: str  # "trending", "ranging", "breakout"
    support_resistance: Dict[str, List[float]]  # 支撑阻力位
    correlation_matrix: Optional[pd.DataFrame] = None
    
    def is_trending(self) -> bool:
        """检查是否处于趋势状态"""
        return self.regime == "trending"
    
    def is_high_volatility(self, threshold: float = 0.02) -> bool:
        """检查是否高波动"""
        return self.volatility > threshold


@dataclass
class SymbolConfig:
    """品种配置模型"""
    symbol: str
    spread_limit: float
    min_lot: float
    max_lot: float
    lot_step: float
    strategies: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=list)
    risk_multiplier: float = 1.0
    trading_hours: Dict[str, str] = field(default_factory=dict)
    optimize_params: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    
    def is_trading_time(self, current_time: datetime) -> bool:
        """检查是否在交易时间内"""
        if not self.trading_hours:
            return True  # 如果没有设置交易时间，默认24小时交易
        
        weekday = current_time.strftime("%A").lower()
        if weekday in self.trading_hours:
            start_time, end_time = self.trading_hours[weekday].split("-")
            current_hour = current_time.hour
            start_hour = int(start_time.split(":")[0])
            end_hour = int(end_time.split(":")[0])
            
            if start_hour <= end_hour:
                return start_hour <= current_hour <= end_hour
            else:  # 跨日交易
                return current_hour >= start_hour or current_hour <= end_hour
        
        return False
    
    def validate_lot_size(self, lot_size: float) -> float:
        """验证并调整手数"""
        if lot_size < self.min_lot:
            return self.min_lot
        if lot_size > self.max_lot:
            return self.max_lot
        
        # 调整到最近的步长
        steps = round((lot_size - self.min_lot) / self.lot_step)
        return self.min_lot + steps * self.lot_step


# 接口定义
class DataProvider:
    """数据提供者接口"""
    
    def get_market_data(self, symbol: str, timeframe: str, count: int = 100) -> MarketData:
        """获取市场数据"""
        raise NotImplementedError
    
    def get_account_info(self) -> Account:
        """获取账户信息"""
        raise NotImplementedError
    
    def get_positions(self) -> List[Position]:
        """获取持仓列表"""
        raise NotImplementedError
    
    def get_trades_history(self, start_date: datetime, end_date: datetime) -> List[Trade]:
        """获取交易历史"""
        raise NotImplementedError


class OrderExecutor:
    """订单执行器接口"""
    
    def send_order(self, signal: Signal) -> bool:
        """发送订单"""
        raise NotImplementedError
    
    def close_position(self, position_id: str) -> bool:
        """关闭持仓"""
        raise NotImplementedError
    
    def modify_position(self, position_id: str, sl: float = None, tp: float = None) -> bool:
        """修改持仓"""
        raise NotImplementedError


class Strategy:
    """策略接口"""
    
    def generate_signal(self, market_data: MarketData) -> Optional[Signal]:
        """生成交易信号"""
        raise NotImplementedError
    
    def update_parameters(self, params: Dict[str, Any]) -> None:
        """更新策略参数"""
        raise NotImplementedError
    
    def get_performance_metrics(self) -> RiskMetrics:
        """获取策略性能指标"""
        raise NotImplementedError