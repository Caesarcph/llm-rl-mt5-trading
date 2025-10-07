"""
基础交易策略模块
实现趋势跟踪、震荡和突破策略
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import logging

from src.core.models import MarketData, Signal, Strategy
from src.core.exceptions import StrategyException, DataValidationError
from src.strategies.indicators import TechnicalIndicators, IndicatorResult


class StrategyType(Enum):
    """策略类型"""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    SCALPING = "scalping"


class SignalStrength(Enum):
    """信号强度"""
    WEAK = 0.3
    MEDIUM = 0.6
    STRONG = 0.9


@dataclass
class StrategyConfig:
    """策略配置"""
    name: str
    strategy_type: StrategyType
    enabled: bool = True
    risk_per_trade: float = 0.02  # 每笔交易风险百分比
    max_positions: int = 3
    min_signal_strength: float = 0.5
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """验证配置"""
        if self.risk_per_trade <= 0 or self.risk_per_trade > 0.1:
            raise ValueError("每笔交易风险必须在0-10%之间")
        if self.max_positions <= 0:
            raise ValueError("最大持仓数必须大于0")
        if not 0 <= self.min_signal_strength <= 1:
            raise ValueError("最小信号强度必须在0-1之间")
        return True


@dataclass
class StrategyPerformance:
    """策略性能指标"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_profit: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update_performance(self, trade_profit: float) -> None:
        """更新性能指标"""
        self.total_trades += 1
        self.total_profit += trade_profit
        
        if trade_profit > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        self.win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        self.last_updated = datetime.now()
    
    def get_performance_score(self) -> float:
        """获取综合性能评分"""
        if self.total_trades < 10:
            return 0.5  # 交易次数不足，返回中性评分
        
        # 综合考虑胜率、盈利因子和夏普比率
        score = (self.win_rate * 0.4 + 
                min(self.profit_factor / 2, 1) * 0.4 + 
                min(max(self.sharpe_ratio, 0) / 2, 1) * 0.2)
        
        return min(max(score, 0), 1)


class BaseStrategy(Strategy, ABC):
    """基础策略抽象类"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.indicators = TechnicalIndicators()
        self.performance = StrategyPerformance()
        self.logger = logging.getLogger(f"{__name__}.{config.name}")
        
        # 验证配置
        self.config.validate()
        
        # 策略状态
        self.is_enabled = config.enabled
        self.last_signal_time = None
        self.current_positions = 0
    
    @abstractmethod
    def _calculate_entry_signal(self, market_data: MarketData) -> Optional[Signal]:
        """计算入场信号"""
        pass
    
    @abstractmethod
    def _calculate_exit_signal(self, market_data: MarketData, position_info: Dict[str, Any]) -> Optional[Signal]:
        """计算出场信号"""
        pass
    
    def generate_signal(self, market_data: MarketData) -> Optional[Signal]:
        """生成交易信号"""
        if not self.is_enabled:
            return None
        
        try:
            # 检查是否可以开新仓
            if self.current_positions >= self.config.max_positions:
                self.logger.debug("已达到最大持仓数限制")
                return None
            
            # 计算入场信号
            signal = self._calculate_entry_signal(market_data)
            
            if signal and signal.strength >= self.config.min_signal_strength:
                # 记录信号时间
                self.last_signal_time = market_data.timestamp
                
                # 设置策略ID
                signal.strategy_id = self.config.name
                
                self.logger.info(f"生成{signal.direction}信号，强度: {signal.strength:.2f}")
                return signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"生成信号时发生错误: {str(e)}")
            raise StrategyException(f"策略{self.config.name}信号生成失败: {str(e)}")
    
    def update_parameters(self, params: Dict[str, Any]) -> None:
        """更新策略参数"""
        self.config.parameters.update(params)
        self.logger.info(f"更新策略参数: {params}")
    
    def get_performance_metrics(self) -> StrategyPerformance:
        """获取策略性能指标"""
        return self.performance
    
    def _calculate_position_size(self, market_data: MarketData, entry_price: float, 
                               stop_loss: float, account_balance: float = 10000) -> float:
        """计算仓位大小"""
        if stop_loss == 0 or entry_price == stop_loss:
            return 0.01  # 默认最小手数
        
        # 计算风险金额
        risk_amount = account_balance * self.config.risk_per_trade
        
        # 计算每点价值（简化计算）
        pip_value = 1.0  # 假设每点价值为1美元
        
        # 计算止损点数
        stop_loss_pips = abs(entry_price - stop_loss) * 10000  # 转换为点数
        
        if stop_loss_pips == 0:
            return 0.01
        
        # 计算手数
        lot_size = risk_amount / (stop_loss_pips * pip_value)
        
        # 限制在合理范围内
        return max(0.01, min(lot_size, 1.0))
    
    def _calculate_stop_loss(self, market_data: MarketData, direction: int, 
                           entry_price: float, atr_multiplier: float = 2.0) -> float:
        """计算止损价格"""
        try:
            # 使用ATR计算动态止损
            atr_result = self.indicators.calculate_atr(market_data, period=14)
            atr_value = atr_result.get_latest_value()
            
            if atr_value is None:
                # 如果ATR不可用，使用固定百分比
                return entry_price * (1 - 0.01 * direction)
            
            # 基于ATR的止损
            stop_distance = atr_value * atr_multiplier
            
            if direction > 0:  # 买入
                return entry_price - stop_distance
            else:  # 卖出
                return entry_price + stop_distance
                
        except Exception as e:
            self.logger.warning(f"计算止损失败，使用默认值: {str(e)}")
            return entry_price * (1 - 0.01 * direction)
    
    def _calculate_take_profit(self, entry_price: float, stop_loss: float, 
                             direction: int, risk_reward_ratio: float = 2.0) -> float:
        """计算止盈价格"""
        stop_distance = abs(entry_price - stop_loss)
        profit_distance = stop_distance * risk_reward_ratio
        
        if direction > 0:  # 买入
            return entry_price + profit_distance
        else:  # 卖出
            return entry_price - profit_distance


class TrendFollowingStrategy(BaseStrategy):
    """趋势跟踪策略"""
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        
        # 默认参数
        self.default_params = {
            "ma_fast_period": 12,
            "ma_slow_period": 26,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "adx_period": 14,
            "adx_threshold": 25,
            "risk_reward_ratio": 2.0
        }
        
        # 合并用户参数
        for key, value in self.default_params.items():
            if key not in self.config.parameters:
                self.config.parameters[key] = value
    
    def _calculate_entry_signal(self, market_data: MarketData) -> Optional[Signal]:
        """计算趋势跟踪入场信号"""
        try:
            params = self.config.parameters
            
            # 计算移动平均线
            ma_fast = self.indicators.calculate_ema(market_data, period=params["ma_fast_period"])
            ma_slow = self.indicators.calculate_ema(market_data, period=params["ma_slow_period"])
            
            # 计算MACD
            macd = self.indicators.calculate_macd(
                market_data, 
                fast=params["macd_fast"],
                slow=params["macd_slow"],
                signal=params["macd_signal"]
            )
            
            # 计算ADX
            adx = self.indicators.calculate_adx(market_data, period=params["adx_period"])
            
            # 获取最新值
            current_price = market_data.ohlcv['close'].iloc[-1]
            ma_fast_value = ma_fast.get_latest_value()
            ma_slow_value = ma_slow.get_latest_value()
            macd_value = macd.get_latest_value("macd")
            macd_signal_value = macd.get_latest_value("signal")
            adx_value = adx.get_latest_value("adx")
            
            # 检查数据有效性
            if None in [ma_fast_value, ma_slow_value, macd_value, macd_signal_value, adx_value]:
                return None
            
            # 趋势强度检查
            if adx_value < params["adx_threshold"]:
                return None  # 趋势不够强
            
            # 信号计算
            direction = 0
            strength = 0.0
            confidence = 0.0
            
            # MA交叉信号
            ma_signal = 0
            if ma_fast_value > ma_slow_value:
                ma_signal = 1
            elif ma_fast_value < ma_slow_value:
                ma_signal = -1
            
            # MACD信号
            macd_signal_dir = 0
            if macd_value > macd_signal_value and macd_value > 0:
                macd_signal_dir = 1
            elif macd_value < macd_signal_value and macd_value < 0:
                macd_signal_dir = -1
            
            # 价格与MA关系
            price_ma_signal = 0
            if current_price > ma_fast_value > ma_slow_value:
                price_ma_signal = 1
            elif current_price < ma_fast_value < ma_slow_value:
                price_ma_signal = -1
            
            # 综合信号
            signal_count = 0
            if ma_signal == 1:
                signal_count += 1
            elif ma_signal == -1:
                signal_count -= 1
            
            if macd_signal_dir == 1:
                signal_count += 1
            elif macd_signal_dir == -1:
                signal_count -= 1
            
            if price_ma_signal == 1:
                signal_count += 1
            elif price_ma_signal == -1:
                signal_count -= 1
            
            # 确定方向和强度
            if signal_count >= 2:
                direction = 1
                strength = min(0.9, 0.3 + abs(signal_count) * 0.2)
            elif signal_count <= -2:
                direction = -1
                strength = min(0.9, 0.3 + abs(signal_count) * 0.2)
            else:
                return None  # 信号不够强
            
            # 计算置信度（基于ADX强度）
            confidence = min(1.0, adx_value / 50.0)
            
            # 计算入场价格、止损和止盈
            entry_price = current_price
            stop_loss = self._calculate_stop_loss(market_data, direction, entry_price)
            take_profit = self._calculate_take_profit(
                entry_price, stop_loss, direction, params["risk_reward_ratio"]
            )
            
            # 计算仓位大小
            position_size = self._calculate_position_size(market_data, entry_price, stop_loss)
            
            return Signal(
                strategy_id=self.config.name,
                symbol=market_data.symbol,
                direction=direction,
                strength=strength,
                entry_price=entry_price,
                sl=stop_loss,
                tp=take_profit,
                size=position_size,
                confidence=confidence,
                timestamp=market_data.timestamp,
                metadata={
                    "ma_fast": ma_fast_value,
                    "ma_slow": ma_slow_value,
                    "macd": macd_value,
                    "macd_signal": macd_signal_value,
                    "adx": adx_value,
                    "signal_count": signal_count
                }
            )
            
        except Exception as e:
            self.logger.error(f"计算趋势跟踪信号失败: {str(e)}")
            return None
    
    def _calculate_exit_signal(self, market_data: MarketData, position_info: Dict[str, Any]) -> Optional[Signal]:
        """计算趋势跟踪出场信号"""
        # 简化实现，主要依赖止损止盈
        return None


class MeanReversionStrategy(BaseStrategy):
    """均值回归策略"""
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        
        # 默认参数
        self.default_params = {
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "bb_period": 20,
            "bb_std_dev": 2.0,
            "stoch_k_period": 14,
            "stoch_d_period": 3,
            "stoch_overbought": 80,
            "stoch_oversold": 20,
            "risk_reward_ratio": 1.5
        }
        
        # 合并用户参数
        for key, value in self.default_params.items():
            if key not in self.config.parameters:
                self.config.parameters[key] = value
    
    def _calculate_entry_signal(self, market_data: MarketData) -> Optional[Signal]:
        """计算均值回归入场信号"""
        try:
            params = self.config.parameters
            
            # 计算RSI
            rsi = self.indicators.calculate_rsi(market_data, period=params["rsi_period"])
            
            # 计算布林带
            bb = self.indicators.calculate_bollinger_bands(
                market_data, 
                period=params["bb_period"],
                std_dev=params["bb_std_dev"]
            )
            
            # 计算随机指标
            stoch = self.indicators.calculate_stochastic(
                market_data,
                k_period=params["stoch_k_period"],
                d_period=params["stoch_d_period"]
            )
            
            # 获取最新值
            current_price = market_data.ohlcv['close'].iloc[-1]
            rsi_value = rsi.get_latest_value()
            bb_upper = bb.get_latest_value("upper")
            bb_lower = bb.get_latest_value("lower")
            bb_middle = bb.get_latest_value("middle")
            stoch_k = stoch.get_latest_value("k")
            stoch_d = stoch.get_latest_value("d")
            
            # 检查数据有效性
            if None in [rsi_value, bb_upper, bb_lower, bb_middle, stoch_k, stoch_d]:
                return None
            
            # 信号计算
            direction = 0
            strength = 0.0
            confidence = 0.0
            oversold_signals = 0
            overbought_signals = 0
            
            # RSI信号
            if rsi_value <= params["rsi_oversold"]:
                oversold_signals += 1
            elif rsi_value >= params["rsi_overbought"]:
                overbought_signals += 1
            
            # 布林带信号
            if current_price <= bb_lower:
                oversold_signals += 1
            elif current_price >= bb_upper:
                overbought_signals += 1
            
            # 随机指标信号
            if stoch_k <= params["stoch_oversold"] and stoch_d <= params["stoch_oversold"]:
                oversold_signals += 1
            elif stoch_k >= params["stoch_overbought"] and stoch_d >= params["stoch_overbought"]:
                overbought_signals += 1
            
            # 确定方向和强度
            if oversold_signals >= 2:
                direction = 1  # 买入
                strength = min(0.9, 0.4 + oversold_signals * 0.15)
            elif overbought_signals >= 2:
                direction = -1  # 卖出
                strength = min(0.9, 0.4 + overbought_signals * 0.15)
            else:
                return None  # 信号不够强
            
            # 计算置信度
            if direction == 1:
                # 买入信号置信度
                rsi_confidence = max(0, (params["rsi_oversold"] - rsi_value) / params["rsi_oversold"])
                bb_confidence = max(0, (bb_lower - current_price) / (bb_middle - bb_lower))
            else:
                # 卖出信号置信度
                rsi_confidence = max(0, (rsi_value - params["rsi_overbought"]) / (100 - params["rsi_overbought"]))
                bb_confidence = max(0, (current_price - bb_upper) / (bb_upper - bb_middle))
            
            confidence = min(1.0, (rsi_confidence + bb_confidence) / 2)
            
            # 计算入场价格、止损和止盈
            entry_price = current_price
            
            # 均值回归策略的止损设置
            if direction == 1:
                stop_loss = bb_lower * 0.999  # 稍微低于布林带下轨
                take_profit = bb_middle + (bb_middle - stop_loss) * params["risk_reward_ratio"]
            else:
                stop_loss = bb_upper * 1.001  # 稍微高于布林带上轨
                take_profit = bb_middle - (stop_loss - bb_middle) * params["risk_reward_ratio"]
            
            # 计算仓位大小
            position_size = self._calculate_position_size(market_data, entry_price, stop_loss)
            
            return Signal(
                strategy_id=self.config.name,
                symbol=market_data.symbol,
                direction=direction,
                strength=strength,
                entry_price=entry_price,
                sl=stop_loss,
                tp=take_profit,
                size=position_size,
                confidence=confidence,
                timestamp=market_data.timestamp,
                metadata={
                    "rsi": rsi_value,
                    "bb_upper": bb_upper,
                    "bb_lower": bb_lower,
                    "bb_middle": bb_middle,
                    "stoch_k": stoch_k,
                    "stoch_d": stoch_d,
                    "oversold_signals": oversold_signals,
                    "overbought_signals": overbought_signals
                }
            )
            
        except Exception as e:
            self.logger.error(f"计算均值回归信号失败: {str(e)}")
            return None
    
    def _calculate_exit_signal(self, market_data: MarketData, position_info: Dict[str, Any]) -> Optional[Signal]:
        """计算均值回归出场信号"""
        # 简化实现，主要依赖止损止盈
        return None


class BreakoutStrategy(BaseStrategy):
    """突破策略"""
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        
        # 默认参数
        self.default_params = {
            "bb_period": 20,
            "bb_std_dev": 2.0,
            "volume_threshold": 1.5,  # 成交量倍数阈值
            "atr_period": 14,
            "atr_multiplier": 1.5,
            "min_breakout_strength": 0.001,  # 最小突破强度
            "risk_reward_ratio": 2.5
        }
        
        # 合并用户参数
        for key, value in self.default_params.items():
            if key not in self.config.parameters:
                self.config.parameters[key] = value
    
    def _calculate_entry_signal(self, market_data: MarketData) -> Optional[Signal]:
        """计算突破入场信号"""
        try:
            params = self.config.parameters
            
            # 计算布林带
            bb = self.indicators.calculate_bollinger_bands(
                market_data,
                period=params["bb_period"],
                std_dev=params["bb_std_dev"]
            )
            
            # 计算ATR
            atr = self.indicators.calculate_atr(market_data, period=params["atr_period"])
            
            # 获取最新值
            ohlcv = market_data.ohlcv
            current_price = ohlcv['close'].iloc[-1]
            current_volume = ohlcv['volume'].iloc[-1] if 'volume' in ohlcv.columns else 0
            
            bb_upper = bb.get_latest_value("upper")
            bb_lower = bb.get_latest_value("lower")
            bb_middle = bb.get_latest_value("middle")
            atr_value = atr.get_latest_value()
            
            # 检查数据有效性
            if None in [bb_upper, bb_lower, bb_middle, atr_value]:
                return None
            
            # 计算平均成交量
            avg_volume = ohlcv['volume'].rolling(window=20).mean().iloc[-1] if 'volume' in ohlcv.columns else 1
            
            # 检查突破条件
            direction = 0
            strength = 0.0
            confidence = 0.0
            breakout_type = ""
            
            # 上突破检查
            if current_price > bb_upper:
                breakout_strength = (current_price - bb_upper) / (bb_upper - bb_middle)
                if breakout_strength >= params["min_breakout_strength"]:
                    direction = 1
                    strength = min(0.9, 0.5 + breakout_strength * 0.4)
                    breakout_type = "upward_breakout"
            
            # 下突破检查
            elif current_price < bb_lower:
                breakout_strength = (bb_lower - current_price) / (bb_middle - bb_lower)
                if breakout_strength >= params["min_breakout_strength"]:
                    direction = -1
                    strength = min(0.9, 0.5 + breakout_strength * 0.4)
                    breakout_type = "downward_breakout"
            
            if direction == 0:
                return None  # 没有突破
            
            # 成交量确认
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            if volume_ratio < params["volume_threshold"]:
                return None  # 成交量不足，可能是假突破
            
            # 计算置信度
            volume_confidence = min(1.0, volume_ratio / (params["volume_threshold"] * 2))
            atr_confidence = min(1.0, atr_value / (current_price * 0.01))  # ATR相对强度
            confidence = (volume_confidence + atr_confidence) / 2
            
            # 计算入场价格、止损和止盈
            entry_price = current_price
            
            # 基于ATR的止损
            stop_distance = atr_value * params["atr_multiplier"]
            if direction == 1:
                stop_loss = entry_price - stop_distance
                take_profit = entry_price + stop_distance * params["risk_reward_ratio"]
            else:
                stop_loss = entry_price + stop_distance
                take_profit = entry_price - stop_distance * params["risk_reward_ratio"]
            
            # 计算仓位大小
            position_size = self._calculate_position_size(market_data, entry_price, stop_loss)
            
            return Signal(
                strategy_id=self.config.name,
                symbol=market_data.symbol,
                direction=direction,
                strength=strength,
                entry_price=entry_price,
                sl=stop_loss,
                tp=take_profit,
                size=position_size,
                confidence=confidence,
                timestamp=market_data.timestamp,
                metadata={
                    "breakout_type": breakout_type,
                    "bb_upper": bb_upper,
                    "bb_lower": bb_lower,
                    "bb_middle": bb_middle,
                    "atr": atr_value,
                    "volume_ratio": volume_ratio,
                    "breakout_strength": (current_price - bb_upper) / (bb_upper - bb_middle) if direction == 1 else (bb_lower - current_price) / (bb_middle - bb_lower)
                }
            )
            
        except Exception as e:
            self.logger.error(f"计算突破信号失败: {str(e)}")
            return None
    
    def _calculate_exit_signal(self, market_data: MarketData, position_info: Dict[str, Any]) -> Optional[Signal]:
        """计算突破出场信号"""
        # 简化实现，主要依赖止损止盈
        return None


class StrategyManager:
    """策略管理器"""
    
    def __init__(self):
        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_weights: Dict[str, float] = {}
        self.logger = logging.getLogger(__name__)
    
    def add_strategy(self, strategy: BaseStrategy, weight: float = 1.0) -> None:
        """添加策略"""
        self.strategies[strategy.config.name] = strategy
        self.strategy_weights[strategy.config.name] = weight
        self.logger.info(f"添加策略: {strategy.config.name}, 权重: {weight}")
    
    def remove_strategy(self, strategy_name: str) -> None:
        """移除策略"""
        if strategy_name in self.strategies:
            del self.strategies[strategy_name]
            del self.strategy_weights[strategy_name]
            self.logger.info(f"移除策略: {strategy_name}")
    
    def generate_signals(self, market_data: MarketData) -> List[Signal]:
        """生成所有策略的信号"""
        signals = []
        
        for strategy_name, strategy in self.strategies.items():
            try:
                signal = strategy.generate_signal(market_data)
                if signal:
                    # 应用策略权重
                    weight = self.strategy_weights.get(strategy_name, 1.0)
                    signal.strength *= weight
                    signal.confidence *= weight
                    
                    signals.append(signal)
                    
            except Exception as e:
                self.logger.error(f"策略{strategy_name}生成信号失败: {str(e)}")
        
        return signals
    
    def update_strategy_weights(self, performance_data: Dict[str, float]) -> None:
        """根据性能数据更新策略权重"""
        for strategy_name, performance_score in performance_data.items():
            if strategy_name in self.strategy_weights:
                # 基于性能调整权重
                new_weight = max(0.1, min(2.0, performance_score))
                self.strategy_weights[strategy_name] = new_weight
                self.logger.info(f"更新策略{strategy_name}权重: {new_weight:.2f}")
    
    def get_strategy_performance(self) -> Dict[str, StrategyPerformance]:
        """获取所有策略的性能指标"""
        return {name: strategy.get_performance_metrics() 
                for name, strategy in self.strategies.items()}
    
    def enable_strategy(self, strategy_name: str) -> None:
        """启用策略"""
        if strategy_name in self.strategies:
            self.strategies[strategy_name].is_enabled = True
            self.logger.info(f"启用策略: {strategy_name}")
    
    def disable_strategy(self, strategy_name: str) -> None:
        """禁用策略"""
        if strategy_name in self.strategies:
            self.strategies[strategy_name].is_enabled = False
            self.logger.info(f"禁用策略: {strategy_name}")
    
    def get_active_strategies(self) -> List[str]:
        """获取活跃策略列表"""
        return [name for name, strategy in self.strategies.items() if strategy.is_enabled]