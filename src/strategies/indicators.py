"""
技术指标计算模块
集成ta-lib库，实现常用技术指标计算，支持多周期分析和缓存机制
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import logging
from functools import wraps
import time

from src.core.models import MarketData
from src.core.exceptions import DataValidationError, IndicatorCalculationError


class IndicatorType(Enum):
    """指标类型枚举"""
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    SUPPORT_RESISTANCE = "support_resistance"


@dataclass
class IndicatorConfig:
    """指标配置"""
    name: str
    type: IndicatorType
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeframes: List[str] = field(default_factory=lambda: ["M15", "H1", "H4", "D1"])
    cache_ttl: int = 300  # 缓存时间(秒)
    enabled: bool = True


@dataclass
class IndicatorResult:
    """指标计算结果"""
    name: str
    symbol: str
    timeframe: str
    timestamp: datetime
    values: Dict[str, Union[float, np.ndarray]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_latest_value(self, key: str = "value") -> Optional[float]:
        """获取最新值"""
        if key in self.values:
            value = self.values[key]
            if isinstance(value, np.ndarray):
                return float(value[-1]) if len(value) > 0 and not np.isnan(value[-1]) else None
            return float(value) if not np.isnan(value) else None
        return None


class IndicatorCache:
    """指标缓存管理器"""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Tuple[IndicatorResult, datetime]] = {}
        self.max_size = max_size
        self.logger = logging.getLogger(__name__)
    
    def _generate_key(self, symbol: str, timeframe: str, indicator_name: str, 
                     params: Dict[str, Any]) -> str:
        """生成缓存键"""
        key_data = {
            "symbol": symbol,
            "timeframe": timeframe,
            "indicator": indicator_name,
            "params": params
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, symbol: str, timeframe: str, indicator_name: str, 
            params: Dict[str, Any], ttl: int = 300) -> Optional[IndicatorResult]:
        """获取缓存的指标结果"""
        key = self._generate_key(symbol, timeframe, indicator_name, params)
        
        if key in self.cache:
            result, timestamp = self.cache[key]
            if (datetime.now() - timestamp).total_seconds() < ttl:
                return result
            else:
                # 缓存过期，删除
                del self.cache[key]
        
        return None
    
    def set(self, symbol: str, timeframe: str, indicator_name: str, 
            params: Dict[str, Any], result: IndicatorResult) -> None:
        """设置缓存"""
        key = self._generate_key(symbol, timeframe, indicator_name, params)
        
        # 如果缓存已满，删除最旧的条目
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        self.cache[key] = (result, datetime.now())
    
    def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "usage_percent": (len(self.cache) / self.max_size) * 100
        }


def cache_indicator(func):
    """指标缓存装饰器"""
    @wraps(func)
    def wrapper(self, market_data: MarketData, **kwargs):
        if not hasattr(self, 'cache') or not self.use_cache:
            return func(self, market_data, **kwargs)
        
        # 获取指标名称和参数
        indicator_name = func.__name__.replace('calculate_', '')
        params = kwargs.copy()
        
        # 尝试从缓存获取
        cached_result = self.cache.get(
            market_data.symbol, 
            market_data.timeframe, 
            indicator_name, 
            params,
            self.cache_ttl
        )
        
        if cached_result is not None:
            return cached_result
        
        # 计算指标
        result = func(self, market_data, **kwargs)
        
        # 缓存结果
        if result is not None:
            self.cache.set(
                market_data.symbol, 
                market_data.timeframe, 
                indicator_name, 
                params, 
                result
            )
        
        return result
    
    return wrapper


class TechnicalIndicators:
    """技术指标计算器"""
    
    def __init__(self, use_cache: bool = True, cache_ttl: int = 300):
        self.use_cache = use_cache
        self.cache_ttl = cache_ttl
        self.cache = IndicatorCache() if use_cache else None
        self.logger = logging.getLogger(__name__)
        
        # 预定义指标配置
        self.indicator_configs = self._initialize_indicator_configs()
    
    def _initialize_indicator_configs(self) -> Dict[str, IndicatorConfig]:
        """初始化指标配置"""
        configs = {
            # 趋势指标
            "sma": IndicatorConfig("SMA", IndicatorType.TREND, {"period": 20}),
            "ema": IndicatorConfig("EMA", IndicatorType.TREND, {"period": 20}),
            "macd": IndicatorConfig("MACD", IndicatorType.TREND, 
                                  {"fast": 12, "slow": 26, "signal": 9}),
            "adx": IndicatorConfig("ADX", IndicatorType.TREND, {"period": 14}),
            "parabolic_sar": IndicatorConfig("Parabolic SAR", IndicatorType.TREND,
                                           {"acceleration": 0.02, "maximum": 0.2}),
            
            # 动量指标
            "rsi": IndicatorConfig("RSI", IndicatorType.MOMENTUM, {"period": 14}),
            "stochastic": IndicatorConfig("Stochastic", IndicatorType.MOMENTUM,
                                        {"k_period": 14, "d_period": 3}),
            "cci": IndicatorConfig("CCI", IndicatorType.MOMENTUM, {"period": 20}),
            "williams_r": IndicatorConfig("Williams %R", IndicatorType.MOMENTUM, 
                                        {"period": 14}),
            "momentum": IndicatorConfig("Momentum", IndicatorType.MOMENTUM, 
                                      {"period": 10}),
            
            # 波动率指标
            "bollinger_bands": IndicatorConfig("Bollinger Bands", IndicatorType.VOLATILITY,
                                             {"period": 20, "std_dev": 2}),
            "atr": IndicatorConfig("ATR", IndicatorType.VOLATILITY, {"period": 14}),
            "envelopes": IndicatorConfig("Envelopes", IndicatorType.VOLATILITY,
                                       {"period": 20, "deviation": 0.1}),
            
            # 成交量指标
            "obv": IndicatorConfig("OBV", IndicatorType.VOLUME),
            "ad_line": IndicatorConfig("A/D Line", IndicatorType.VOLUME),
            "mfi": IndicatorConfig("MFI", IndicatorType.VOLUME, {"period": 14}),
        }
        
        return configs
    
    def _validate_market_data(self, market_data: MarketData) -> None:
        """验证市场数据"""
        if market_data.ohlcv.empty:
            raise DataValidationError("市场数据为空")
        
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns 
                          if col not in market_data.ohlcv.columns]
        
        if missing_columns:
            raise DataValidationError(f"缺少必要的数据列: {missing_columns}")
        
        # 检查数据质量
        ohlcv = market_data.ohlcv
        if (ohlcv['high'] < ohlcv['low']).any():
            raise DataValidationError("数据质量错误: 最高价小于最低价")
        
        if (ohlcv['high'] < ohlcv['close']).any() or (ohlcv['low'] > ohlcv['close']).any():
            raise DataValidationError("数据质量错误: 收盘价超出高低价范围")
    
    @cache_indicator
    def calculate_sma(self, market_data: MarketData, period: int = 20) -> IndicatorResult:
        """计算简单移动平均线"""
        self._validate_market_data(market_data)
        
        try:
            close_prices = market_data.ohlcv['close'].values
            sma_values = talib.SMA(close_prices, timeperiod=period)
            
            return IndicatorResult(
                name="SMA",
                symbol=market_data.symbol,
                timeframe=market_data.timeframe,
                timestamp=market_data.timestamp,
                values={"value": sma_values, "period": period},
                metadata={"period": period}
            )
        except Exception as e:
            raise IndicatorCalculationError(f"SMA计算失败: {str(e)}")
    
    @cache_indicator
    def calculate_ema(self, market_data: MarketData, period: int = 20) -> IndicatorResult:
        """计算指数移动平均线"""
        self._validate_market_data(market_data)
        
        try:
            close_prices = market_data.ohlcv['close'].values
            ema_values = talib.EMA(close_prices, timeperiod=period)
            
            return IndicatorResult(
                name="EMA",
                symbol=market_data.symbol,
                timeframe=market_data.timeframe,
                timestamp=market_data.timestamp,
                values={"value": ema_values, "period": period},
                metadata={"period": period}
            )
        except Exception as e:
            raise IndicatorCalculationError(f"EMA计算失败: {str(e)}")
    
    @cache_indicator
    def calculate_macd(self, market_data: MarketData, fast: int = 12, 
                      slow: int = 26, signal: int = 9) -> IndicatorResult:
        """计算MACD指标"""
        self._validate_market_data(market_data)
        
        try:
            close_prices = market_data.ohlcv['close'].values
            macd, macd_signal, macd_hist = talib.MACD(
                close_prices, fastperiod=fast, slowperiod=slow, signalperiod=signal
            )
            
            return IndicatorResult(
                name="MACD",
                symbol=market_data.symbol,
                timeframe=market_data.timeframe,
                timestamp=market_data.timestamp,
                values={
                    "macd": macd,
                    "signal": macd_signal,
                    "histogram": macd_hist
                },
                metadata={"fast": fast, "slow": slow, "signal": signal}
            )
        except Exception as e:
            raise IndicatorCalculationError(f"MACD计算失败: {str(e)}")
    
    @cache_indicator
    def calculate_rsi(self, market_data: MarketData, period: int = 14) -> IndicatorResult:
        """计算RSI指标"""
        self._validate_market_data(market_data)
        
        try:
            close_prices = market_data.ohlcv['close'].values
            rsi_values = talib.RSI(close_prices, timeperiod=period)
            
            return IndicatorResult(
                name="RSI",
                symbol=market_data.symbol,
                timeframe=market_data.timeframe,
                timestamp=market_data.timestamp,
                values={"value": rsi_values, "period": period},
                metadata={"period": period}
            )
        except Exception as e:
            raise IndicatorCalculationError(f"RSI计算失败: {str(e)}")
    
    @cache_indicator
    def calculate_bollinger_bands(self, market_data: MarketData, period: int = 20, 
                                 std_dev: float = 2.0) -> IndicatorResult:
        """计算布林带指标"""
        self._validate_market_data(market_data)
        
        try:
            close_prices = market_data.ohlcv['close'].values
            upper, middle, lower = talib.BBANDS(
                close_prices, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev
            )
            
            return IndicatorResult(
                name="Bollinger Bands",
                symbol=market_data.symbol,
                timeframe=market_data.timeframe,
                timestamp=market_data.timestamp,
                values={
                    "upper": upper,
                    "middle": middle,
                    "lower": lower,
                    "bandwidth": (upper - lower) / middle * 100
                },
                metadata={"period": period, "std_dev": std_dev}
            )
        except Exception as e:
            raise IndicatorCalculationError(f"布林带计算失败: {str(e)}")
    
    @cache_indicator
    def calculate_adx(self, market_data: MarketData, period: int = 14) -> IndicatorResult:
        """计算ADX指标"""
        self._validate_market_data(market_data)
        
        try:
            high_prices = market_data.ohlcv['high'].values
            low_prices = market_data.ohlcv['low'].values
            close_prices = market_data.ohlcv['close'].values
            
            adx_values = talib.ADX(high_prices, low_prices, close_prices, timeperiod=period)
            plus_di = talib.PLUS_DI(high_prices, low_prices, close_prices, timeperiod=period)
            minus_di = talib.MINUS_DI(high_prices, low_prices, close_prices, timeperiod=period)
            
            return IndicatorResult(
                name="ADX",
                symbol=market_data.symbol,
                timeframe=market_data.timeframe,
                timestamp=market_data.timestamp,
                values={
                    "adx": adx_values,
                    "plus_di": plus_di,
                    "minus_di": minus_di
                },
                metadata={"period": period}
            )
        except Exception as e:
            raise IndicatorCalculationError(f"ADX计算失败: {str(e)}")
    
    @cache_indicator
    def calculate_stochastic(self, market_data: MarketData, k_period: int = 14, 
                           d_period: int = 3) -> IndicatorResult:
        """计算随机指标"""
        self._validate_market_data(market_data)
        
        try:
            high_prices = market_data.ohlcv['high'].values
            low_prices = market_data.ohlcv['low'].values
            close_prices = market_data.ohlcv['close'].values
            
            slowk, slowd = talib.STOCH(
                high_prices, low_prices, close_prices,
                fastk_period=k_period, slowk_period=d_period, slowd_period=d_period
            )
            
            return IndicatorResult(
                name="Stochastic",
                symbol=market_data.symbol,
                timeframe=market_data.timeframe,
                timestamp=market_data.timestamp,
                values={
                    "k": slowk,
                    "d": slowd
                },
                metadata={"k_period": k_period, "d_period": d_period}
            )
        except Exception as e:
            raise IndicatorCalculationError(f"随机指标计算失败: {str(e)}")
    
    @cache_indicator
    def calculate_atr(self, market_data: MarketData, period: int = 14) -> IndicatorResult:
        """计算ATR指标"""
        self._validate_market_data(market_data)
        
        try:
            high_prices = market_data.ohlcv['high'].values
            low_prices = market_data.ohlcv['low'].values
            close_prices = market_data.ohlcv['close'].values
            
            atr_values = talib.ATR(high_prices, low_prices, close_prices, timeperiod=period)
            
            return IndicatorResult(
                name="ATR",
                symbol=market_data.symbol,
                timeframe=market_data.timeframe,
                timestamp=market_data.timestamp,
                values={"value": atr_values, "period": period},
                metadata={"period": period}
            )
        except Exception as e:
            raise IndicatorCalculationError(f"ATR计算失败: {str(e)}")
    
    @cache_indicator
    def calculate_cci(self, market_data: MarketData, period: int = 20) -> IndicatorResult:
        """计算CCI指标"""
        self._validate_market_data(market_data)
        
        try:
            high_prices = market_data.ohlcv['high'].values
            low_prices = market_data.ohlcv['low'].values
            close_prices = market_data.ohlcv['close'].values
            
            cci_values = talib.CCI(high_prices, low_prices, close_prices, timeperiod=period)
            
            return IndicatorResult(
                name="CCI",
                symbol=market_data.symbol,
                timeframe=market_data.timeframe,
                timestamp=market_data.timestamp,
                values={"value": cci_values, "period": period},
                metadata={"period": period}
            )
        except Exception as e:
            raise IndicatorCalculationError(f"CCI计算失败: {str(e)}")
    
    def calculate_multiple_timeframes(self, market_data_dict: Dict[str, MarketData], 
                                    indicator_name: str, **kwargs) -> Dict[str, IndicatorResult]:
        """计算多周期指标"""
        results = {}
        
        for timeframe, market_data in market_data_dict.items():
            try:
                method_name = f"calculate_{indicator_name.lower()}"
                if hasattr(self, method_name):
                    method = getattr(self, method_name)
                    result = method(market_data, **kwargs)
                    results[timeframe] = result
                else:
                    self.logger.warning(f"未找到指标计算方法: {method_name}")
            except Exception as e:
                self.logger.error(f"计算{timeframe}周期{indicator_name}指标失败: {str(e)}")
        
        return results
    
    def get_indicator_summary(self, market_data: MarketData) -> Dict[str, Any]:
        """获取指标汇总"""
        summary = {
            "symbol": market_data.symbol,
            "timeframe": market_data.timeframe,
            "timestamp": market_data.timestamp,
            "indicators": {}
        }
        
        # 计算主要指标
        try:
            # 趋势指标
            sma_20 = self.calculate_sma(market_data, period=20)
            ema_20 = self.calculate_ema(market_data, period=20)
            macd = self.calculate_macd(market_data)
            adx = self.calculate_adx(market_data)
            
            # 动量指标
            rsi = self.calculate_rsi(market_data)
            stoch = self.calculate_stochastic(market_data)
            cci = self.calculate_cci(market_data)
            
            # 波动率指标
            bb = self.calculate_bollinger_bands(market_data)
            atr = self.calculate_atr(market_data)
            
            # 汇总结果
            current_price = market_data.ohlcv['close'].iloc[-1]
            
            summary["indicators"] = {
                "trend": {
                    "sma_20": sma_20.get_latest_value(),
                    "ema_20": ema_20.get_latest_value(),
                    "macd": macd.get_latest_value("macd"),
                    "macd_signal": macd.get_latest_value("signal"),
                    "adx": adx.get_latest_value("adx"),
                    "trend_strength": self._analyze_trend_strength(adx, macd)
                },
                "momentum": {
                    "rsi": rsi.get_latest_value(),
                    "stoch_k": stoch.get_latest_value("k"),
                    "stoch_d": stoch.get_latest_value("d"),
                    "cci": cci.get_latest_value(),
                    "momentum_signal": self._analyze_momentum(rsi, stoch, cci)
                },
                "volatility": {
                    "bb_upper": bb.get_latest_value("upper"),
                    "bb_middle": bb.get_latest_value("middle"),
                    "bb_lower": bb.get_latest_value("lower"),
                    "atr": atr.get_latest_value(),
                    "volatility_level": self._analyze_volatility(bb, atr, current_price)
                }
            }
            
        except Exception as e:
            self.logger.error(f"生成指标汇总失败: {str(e)}")
            summary["error"] = str(e)
        
        return summary
    
    def _analyze_trend_strength(self, adx: IndicatorResult, macd: IndicatorResult) -> str:
        """分析趋势强度"""
        adx_value = adx.get_latest_value("adx")
        macd_value = macd.get_latest_value("macd")
        macd_signal = macd.get_latest_value("signal")
        
        if adx_value is None or macd_value is None or macd_signal is None:
            return "unknown"
        
        if adx_value > 25:
            if macd_value > macd_signal:
                return "strong_uptrend"
            else:
                return "strong_downtrend"
        elif adx_value > 15:
            if macd_value > macd_signal:
                return "weak_uptrend"
            else:
                return "weak_downtrend"
        else:
            return "sideways"
    
    def _analyze_momentum(self, rsi: IndicatorResult, stoch: IndicatorResult, 
                         cci: IndicatorResult) -> str:
        """分析动量信号"""
        rsi_value = rsi.get_latest_value()
        stoch_k = stoch.get_latest_value("k")
        cci_value = cci.get_latest_value()
        
        if None in [rsi_value, stoch_k, cci_value]:
            return "unknown"
        
        # 超买超卖分析
        overbought_signals = 0
        oversold_signals = 0
        
        if rsi_value > 70:
            overbought_signals += 1
        elif rsi_value < 30:
            oversold_signals += 1
        
        if stoch_k > 80:
            overbought_signals += 1
        elif stoch_k < 20:
            oversold_signals += 1
        
        if cci_value > 100:
            overbought_signals += 1
        elif cci_value < -100:
            oversold_signals += 1
        
        if overbought_signals >= 2:
            return "overbought"
        elif oversold_signals >= 2:
            return "oversold"
        else:
            return "neutral"
    
    def _analyze_volatility(self, bb: IndicatorResult, atr: IndicatorResult, 
                           current_price: float) -> str:
        """分析波动率水平"""
        bb_upper = bb.get_latest_value("upper")
        bb_lower = bb.get_latest_value("lower")
        bb_middle = bb.get_latest_value("middle")
        atr_value = atr.get_latest_value()
        
        if None in [bb_upper, bb_lower, bb_middle, atr_value]:
            return "unknown"
        
        # 布林带位置分析
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
        
        # ATR相对分析（需要历史ATR数据进行比较，这里简化处理）
        atr_ratio = atr_value / current_price
        
        if bb_position > 0.8 or bb_position < 0.2:
            if atr_ratio > 0.02:  # 假设2%为高波动阈值
                return "high_volatility"
            else:
                return "medium_volatility"
        else:
            if atr_ratio > 0.02:
                return "medium_volatility"
            else:
                return "low_volatility"
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        if self.cache:
            return self.cache.get_cache_stats()
        return {"cache_disabled": True}
    
    def clear_cache(self) -> None:
        """清空缓存"""
        if self.cache:
            self.cache.clear()