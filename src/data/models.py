#!/usr/bin/env python3
"""
数据模型定义
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import pandas as pd


class TimeFrame(Enum):
    """时间周期枚举"""
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"
    W1 = "W1"
    MN1 = "MN1"


@dataclass
class MarketData:
    """市场数据模型"""
    symbol: str
    timeframe: TimeFrame
    timestamp: datetime
    ohlcv: pd.DataFrame  # OHLCV数据
    indicators: Dict[str, float] = field(default_factory=dict)
    spread: float = 0.0
    liquidity: float = 0.0
    volume_profile: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """数据验证"""
        if self.ohlcv is not None and not self.ohlcv.empty:
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in self.ohlcv.columns]
            if missing_columns:
                raise ValueError(f"OHLCV数据缺少必要列: {missing_columns}")


@dataclass
class Tick:
    """Tick数据模型"""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last: float
    volume: int
    flags: int = 0
    
    @property
    def spread(self) -> float:
        """计算点差"""
        return self.ask - self.bid
    
    @property
    def mid_price(self) -> float:
        """计算中间价"""
        return (self.bid + self.ask) / 2


@dataclass
class ValidationResult:
    """数据验证结果"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, error: str):
        """添加错误"""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """添加警告"""
        self.warnings.append(warning)


@dataclass
class DataRequest:
    """数据请求模型"""
    symbol: str
    timeframe: TimeFrame
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    count: Optional[int] = None
    include_indicators: bool = True
    cache_ttl: int = 300  # 缓存时间(秒)


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    data: Any
    timestamp: datetime
    ttl: int  # 生存时间(秒)
    
    @property
    def is_expired(self) -> bool:
        """检查是否过期"""
        return (datetime.now() - self.timestamp).total_seconds() > self.ttl


@dataclass
class DataStats:
    """数据统计信息"""
    symbol: str
    timeframe: TimeFrame
    total_bars: int
    start_time: datetime
    end_time: datetime
    missing_bars: int = 0
    data_quality_score: float = 1.0
    last_update: datetime = field(default_factory=datetime.now)