"""
执行优化Agent
优化订单执行，实现滑点预测、最佳入场时机选择、订单拆分和流动性管理
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from src.core.models import (
    MarketData, Signal, Account, Position, Trade, OrderStatus
)
from src.core.exceptions import OrderException, DataValidationException


class ExecutionMethod(Enum):
    """执行方法"""
    MARKET = "market"           # 市价单
    LIMIT = "limit"            # 限价单
    STOP = "stop"              # 止损单
    TWAP = "twap"              # 时间加权平均价格
    VWAP = "vwap"              # 成交量加权平均价格
    ICEBERG = "iceberg"        # 冰山订单


class LiquidityLevel(Enum):
    """流动性水平"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


class MarketCondition(Enum):
    """市场状况"""
    NORMAL = "normal"
    VOLATILE = "volatile"
    ILLIQUID = "illiquid"
    NEWS_EVENT = "news_event"


@dataclass
class SlippageAnalysis:
    """滑点分析结果"""
    expected_slippage: float  # 预期滑点(点数)
    slippage_probability: float  # 滑点概率
    worst_case_slippage: float  # 最坏情况滑点
    confidence_interval: Tuple[float, float]  # 置信区间
    historical_average: float  # 历史平均滑点
    
    def get_slippage_cost(self, volume: float, point_value: float) -> float:
        """计算滑点成本"""
        return self.expected_slippage * volume * point_value


@dataclass
class LiquidityAnalysis:
    """流动性分析结果"""
    liquidity_level: LiquidityLevel
    bid_ask_spread: float
    market_depth: Dict[str, float]  # {"bid_depth": 100, "ask_depth": 120}
    volume_profile: Dict[str, float]  # 成交量分布
    impact_cost: float  # 市场冲击成本
    
    def can_execute_size(self, volume: float, max_impact: float = 0.001) -> bool:
        """检查是否可以执行指定手数"""
        return self.impact_cost <= max_impact


@dataclass
class TimingAnalysis:
    """时机分析结果"""
    optimal_entry_time: datetime
    entry_score: float  # 0-100
    market_momentum: float  # -1 to 1
    volatility_forecast: float
    execution_urgency: float  # 0-1
    
    def should_delay_execution(self, threshold: float = 0.3) -> bool:
        """是否应该延迟执行"""
        return self.execution_urgency < threshold


@dataclass
class OrderSplit:
    """订单拆分结果"""
    child_orders: List[Dict[str, Any]]
    execution_schedule: List[datetime]
    total_volume: float
    average_size: float
    
    def get_next_order(self) -> Optional[Dict[str, Any]]:
        """获取下一个子订单"""
        if self.child_orders:
            return self.child_orders.pop(0)
        return None


@dataclass
class ExecutionPlan:
    """执行计划"""
    signal: Signal
    method: ExecutionMethod
    order_splits: List[OrderSplit]
    timing_analysis: TimingAnalysis
    slippage_analysis: SlippageAnalysis
    liquidity_analysis: LiquidityAnalysis
    estimated_cost: float
    execution_priority: int  # 1-10
    
    def get_total_estimated_cost(self) -> float:
        """获取总预估成本"""
        return self.estimated_cost + self.slippage_analysis.expected_slippage


class SlippagePredictor:
    """滑点预测器"""
    
    def __init__(self, lookback_period: int = 100):
        self.lookback_period = lookback_period
        self.logger = logging.getLogger(__name__)
        self.slippage_history: Dict[str, List[Tuple[datetime, float, float]]] = {}
    
    def predict_slippage(self, symbol: str, market_data: MarketData, 
                        volume: float, direction: int) -> SlippageAnalysis:
        """预测滑点"""
        try:
            # 获取历史滑点数据
            historical_slippage = self._get_historical_slippage(symbol)
            
            # 计算基础滑点
            base_slippage = self._calculate_base_slippage(market_data, volume)
            
            # 考虑市场状况调整
            market_adjustment = self._calculate_market_adjustment(market_data)
            
            # 考虑交易方向调整
            direction_adjustment = self._calculate_direction_adjustment(market_data, direction)
            
            # 预期滑点
            expected_slippage = base_slippage * market_adjustment * direction_adjustment
            
            # 计算置信区间
            if historical_slippage:
                std_dev = np.std(historical_slippage)
                confidence_interval = (
                    expected_slippage - 1.96 * std_dev,
                    expected_slippage + 1.96 * std_dev
                )
                historical_average = np.mean(historical_slippage)
            else:
                confidence_interval = (expected_slippage * 0.5, expected_slippage * 2.0)
                historical_average = expected_slippage
            
            # 最坏情况滑点
            worst_case_slippage = confidence_interval[1]
            
            # 滑点概率
            slippage_probability = min(1.0, volume / 100.0)  # 简化计算
            
            return SlippageAnalysis(
                expected_slippage=expected_slippage,
                slippage_probability=slippage_probability,
                worst_case_slippage=worst_case_slippage,
                confidence_interval=confidence_interval,
                historical_average=historical_average
            )
            
        except Exception as e:
            self.logger.error(f"滑点预测失败 {symbol}: {str(e)}")
            return self._get_default_slippage_analysis()
    
    def record_actual_slippage(self, symbol: str, expected_price: float, 
                              actual_price: float, timestamp: datetime) -> None:
        """记录实际滑点"""
        try:
            slippage = abs(actual_price - expected_price)
            
            if symbol not in self.slippage_history:
                self.slippage_history[symbol] = []
            
            self.slippage_history[symbol].append((timestamp, expected_price, slippage))
            
            # 保持历史数据在合理范围内
            if len(self.slippage_history[symbol]) > self.lookback_period:
                self.slippage_history[symbol] = self.slippage_history[symbol][-self.lookback_period:]
                
        except Exception as e:
            self.logger.error(f"滑点记录失败 {symbol}: {str(e)}")
    
    def _get_historical_slippage(self, symbol: str) -> List[float]:
        """获取历史滑点数据"""
        if symbol in self.slippage_history:
            return [record[2] for record in self.slippage_history[symbol]]
        return []
    
    def _calculate_base_slippage(self, market_data: MarketData, volume: float) -> float:
        """计算基础滑点"""
        # 基于点差和成交量的简化计算
        base_slippage = market_data.spread * 0.5  # 半个点差作为基础
        volume_impact = min(volume / 10.0, 2.0)  # 成交量影响，最大2倍
        return base_slippage * (1 + volume_impact)
    
    def _calculate_market_adjustment(self, market_data: MarketData) -> float:
        """计算市场状况调整因子"""
        try:
            # 基于波动率调整
            if 'atr' in market_data.indicators:
                atr = market_data.indicators['atr']
                current_price = market_data.ohlcv['close'].iloc[-1]
                volatility_ratio = atr / current_price if current_price > 0 else 0.01
                
                # 高波动率增加滑点
                if volatility_ratio > 0.02:
                    return 2.0
                elif volatility_ratio > 0.01:
                    return 1.5
                else:
                    return 1.0
            
            return 1.0
            
        except Exception:
            return 1.0
    
    def _calculate_direction_adjustment(self, market_data: MarketData, direction: int) -> float:
        """计算交易方向调整因子"""
        try:
            # 基于市场趋势调整
            if len(market_data.ohlcv) >= 20:
                recent_prices = market_data.ohlcv['close'].tail(20)
                trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
                
                # 顺势交易滑点较小，逆势交易滑点较大
                if (direction > 0 and trend > 0) or (direction < 0 and trend < 0):
                    return 0.8  # 顺势减少滑点
                elif (direction > 0 and trend < -0.01) or (direction < 0 and trend > 0.01):
                    return 1.3  # 逆势增加滑点
            
            return 1.0
            
        except Exception:
            return 1.0
    
    def _get_default_slippage_analysis(self) -> SlippageAnalysis:
        """获取默认滑点分析"""
        return SlippageAnalysis(
            expected_slippage=2.0,  # 2点默认滑点
            slippage_probability=0.5,
            worst_case_slippage=5.0,
            confidence_interval=(1.0, 4.0),
            historical_average=2.0
        )


class LiquidityAnalyzer:
    """流动性分析器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_liquidity(self, symbol: str, market_data: MarketData) -> LiquidityAnalysis:
        """分析市场流动性"""
        try:
            # 计算买卖价差
            bid_ask_spread = market_data.spread
            
            # 分析成交量分布
            volume_profile = self._analyze_volume_profile(market_data)
            
            # 估算市场深度
            market_depth = self._estimate_market_depth(market_data)
            
            # 计算市场冲击成本
            impact_cost = self._calculate_impact_cost(market_data, volume_profile)
            
            # 确定流动性水平
            liquidity_level = self._determine_liquidity_level(
                bid_ask_spread, volume_profile, market_depth
            )
            
            return LiquidityAnalysis(
                liquidity_level=liquidity_level,
                bid_ask_spread=bid_ask_spread,
                market_depth=market_depth,
                volume_profile=volume_profile,
                impact_cost=impact_cost
            )
            
        except Exception as e:
            self.logger.error(f"流动性分析失败 {symbol}: {str(e)}")
            return self._get_default_liquidity_analysis()
    
    def _analyze_volume_profile(self, market_data: MarketData) -> Dict[str, float]:
        """分析成交量分布"""
        try:
            if 'volume' in market_data.ohlcv.columns:
                volumes = market_data.ohlcv['volume'].tail(50)
                
                return {
                    'average_volume': volumes.mean(),
                    'volume_std': volumes.std(),
                    'current_volume': volumes.iloc[-1],
                    'volume_ratio': volumes.iloc[-1] / volumes.mean() if volumes.mean() > 0 else 1.0
                }
            else:
                return {
                    'average_volume': 1000.0,
                    'volume_std': 200.0,
                    'current_volume': 1000.0,
                    'volume_ratio': 1.0
                }
                
        except Exception:
            return {
                'average_volume': 1000.0,
                'volume_std': 200.0,
                'current_volume': 1000.0,
                'volume_ratio': 1.0
            }
    
    def _estimate_market_depth(self, market_data: MarketData) -> Dict[str, float]:
        """估算市场深度"""
        try:
            # 基于历史数据估算
            if 'volume' in market_data.ohlcv.columns:
                avg_volume = market_data.ohlcv['volume'].tail(20).mean()
                
                # 简化估算：假设市场深度与平均成交量相关
                bid_depth = avg_volume * 0.3
                ask_depth = avg_volume * 0.3
            else:
                bid_depth = 500.0
                ask_depth = 500.0
            
            return {
                'bid_depth': bid_depth,
                'ask_depth': ask_depth,
                'total_depth': bid_depth + ask_depth
            }
            
        except Exception:
            return {
                'bid_depth': 500.0,
                'ask_depth': 500.0,
                'total_depth': 1000.0
            }
    
    def _calculate_impact_cost(self, market_data: MarketData, 
                              volume_profile: Dict[str, float]) -> float:
        """计算市场冲击成本"""
        try:
            # 基于点差和成交量比率的简化计算
            spread_cost = market_data.spread * 0.5
            volume_ratio = volume_profile.get('volume_ratio', 1.0)
            
            # 成交量比率越高，冲击成本越大
            impact_multiplier = 1.0 + max(0, volume_ratio - 1.0) * 0.5
            
            return spread_cost * impact_multiplier
            
        except Exception:
            return market_data.spread * 0.5
    
    def _determine_liquidity_level(self, spread: float, volume_profile: Dict[str, float],
                                  market_depth: Dict[str, float]) -> LiquidityLevel:
        """确定流动性水平"""
        try:
            # 综合评分
            score = 0
            
            # 点差评分 (40%)
            if spread <= 0.0001:  # 1点以内
                score += 40
            elif spread <= 0.0002:  # 2点以内
                score += 30
            elif spread <= 0.0005:  # 5点以内
                score += 20
            else:
                score += 10
            
            # 成交量评分 (30%)
            volume_ratio = volume_profile.get('volume_ratio', 1.0)
            if volume_ratio >= 1.5:
                score += 30
            elif volume_ratio >= 1.0:
                score += 20
            elif volume_ratio >= 0.5:
                score += 15
            else:
                score += 5
            
            # 市场深度评分 (30%)
            total_depth = market_depth.get('total_depth', 1000)
            if total_depth >= 2000:
                score += 30
            elif total_depth >= 1000:
                score += 20
            elif total_depth >= 500:
                score += 15
            else:
                score += 5
            
            # 根据评分确定流动性水平
            if score >= 80:
                return LiquidityLevel.HIGH
            elif score >= 60:
                return LiquidityLevel.MEDIUM
            elif score >= 40:
                return LiquidityLevel.LOW
            else:
                return LiquidityLevel.VERY_LOW
                
        except Exception:
            return LiquidityLevel.MEDIUM
    
    def _get_default_liquidity_analysis(self) -> LiquidityAnalysis:
        """获取默认流动性分析"""
        return LiquidityAnalysis(
            liquidity_level=LiquidityLevel.MEDIUM,
            bid_ask_spread=0.0002,
            market_depth={'bid_depth': 500, 'ask_depth': 500, 'total_depth': 1000},
            volume_profile={'average_volume': 1000, 'volume_std': 200, 
                          'current_volume': 1000, 'volume_ratio': 1.0},
            impact_cost=0.0001
        )


class TimingOptimizer:
    """时机优化器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_timing(self, signal: Signal, market_data: MarketData) -> TimingAnalysis:
        """分析最佳执行时机"""
        try:
            # 计算市场动量
            momentum = self._calculate_market_momentum(market_data)
            
            # 预测波动率
            volatility_forecast = self._forecast_volatility(market_data)
            
            # 计算执行紧急程度
            urgency = self._calculate_execution_urgency(signal, market_data)
            
            # 计算入场评分
            entry_score = self._calculate_entry_score(signal, market_data, momentum, volatility_forecast)
            
            # 确定最佳入场时间
            optimal_time = self._determine_optimal_entry_time(signal, entry_score, urgency)
            
            return TimingAnalysis(
                optimal_entry_time=optimal_time,
                entry_score=entry_score,
                market_momentum=momentum,
                volatility_forecast=volatility_forecast,
                execution_urgency=urgency
            )
            
        except Exception as e:
            self.logger.error(f"时机分析失败: {str(e)}")
            return self._get_default_timing_analysis()
    
    def _calculate_market_momentum(self, market_data: MarketData) -> float:
        """计算市场动量"""
        try:
            if len(market_data.ohlcv) >= 20:
                prices = market_data.ohlcv['close'].tail(20)
                
                # 计算短期和长期移动平均
                short_ma = prices.tail(5).mean()
                long_ma = prices.tail(20).mean()
                
                # 动量 = (短期MA - 长期MA) / 长期MA
                momentum = (short_ma - long_ma) / long_ma if long_ma > 0 else 0
                
                # 限制在-1到1之间
                return max(-1.0, min(1.0, momentum * 10))
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _forecast_volatility(self, market_data: MarketData) -> float:
        """预测波动率"""
        try:
            if len(market_data.ohlcv) >= 20:
                prices = market_data.ohlcv['close'].tail(20)
                returns = prices.pct_change().dropna()
                
                # 简单的GARCH(1,1)近似
                current_vol = returns.std()
                long_term_vol = returns.tail(20).std()
                
                # 预测下一期波动率
                forecast_vol = 0.1 * long_term_vol + 0.9 * current_vol
                
                return forecast_vol
            
            return 0.02  # 默认2%波动率
            
        except Exception:
            return 0.02
    
    def _calculate_execution_urgency(self, signal: Signal, market_data: MarketData) -> float:
        """计算执行紧急程度"""
        try:
            urgency = 0.5  # 基础紧急程度
            
            # 基于信号强度调整
            urgency += signal.strength * 0.3
            
            # 基于信号置信度调整
            urgency += signal.confidence * 0.2
            
            # 基于市场波动率调整
            if 'atr' in market_data.indicators:
                atr = market_data.indicators['atr']
                current_price = market_data.ohlcv['close'].iloc[-1]
                volatility_ratio = atr / current_price if current_price > 0 else 0.01
                
                # 高波动率增加紧急程度
                if volatility_ratio > 0.02:
                    urgency += 0.2
                elif volatility_ratio > 0.01:
                    urgency += 0.1
            
            return max(0.0, min(1.0, urgency))
            
        except Exception:
            return 0.5
    
    def _calculate_entry_score(self, signal: Signal, market_data: MarketData,
                              momentum: float, volatility_forecast: float) -> float:
        """计算入场评分"""
        try:
            score = 50.0  # 基础分数
            
            # 信号方向与动量一致性 (30%)
            if (signal.direction > 0 and momentum > 0) or (signal.direction < 0 and momentum < 0):
                score += 30 * abs(momentum)
            else:
                score -= 20 * abs(momentum)
            
            # 信号强度 (25%)
            score += signal.strength * 25
            
            # 信号置信度 (25%)
            score += signal.confidence * 25
            
            # 波动率适宜性 (20%)
            if 0.01 <= volatility_forecast <= 0.03:  # 适中波动率
                score += 20
            elif volatility_forecast > 0.05:  # 过高波动率
                score -= 15
            elif volatility_forecast < 0.005:  # 过低波动率
                score -= 10
            
            return max(0.0, min(100.0, score))
            
        except Exception:
            return 50.0
    
    def _determine_optimal_entry_time(self, signal: Signal, entry_score: float, 
                                    urgency: float) -> datetime:
        """确定最佳入场时间"""
        try:
            base_time = signal.timestamp
            
            # 如果评分很高且紧急程度高，立即执行
            if entry_score >= 80 and urgency >= 0.8:
                return base_time
            
            # 如果评分中等，延迟1-5分钟
            elif entry_score >= 60:
                delay_minutes = int((1 - urgency) * 5)
                return base_time + timedelta(minutes=delay_minutes)
            
            # 如果评分较低，延迟5-15分钟
            else:
                delay_minutes = int(5 + (1 - urgency) * 10)
                return base_time + timedelta(minutes=delay_minutes)
                
        except Exception:
            return signal.timestamp
    
    def _get_default_timing_analysis(self) -> TimingAnalysis:
        """获取默认时机分析"""
        return TimingAnalysis(
            optimal_entry_time=datetime.now(),
            entry_score=50.0,
            market_momentum=0.0,
            volatility_forecast=0.02,
            execution_urgency=0.5
        )


class OrderSplitter:
    """订单拆分器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def split_order(self, signal: Signal, liquidity_analysis: LiquidityAnalysis,
                   max_chunk_size: float = 10.0) -> OrderSplit:
        """拆分大订单"""
        try:
            total_volume = signal.size
            
            # 如果订单较小，不需要拆分
            if total_volume <= max_chunk_size:
                return OrderSplit(
                    child_orders=[self._create_child_order(signal, total_volume, 0)],
                    execution_schedule=[signal.timestamp],
                    total_volume=total_volume,
                    average_size=total_volume
                )
            
            # 计算拆分策略
            chunk_size = self._calculate_optimal_chunk_size(
                total_volume, liquidity_analysis, max_chunk_size
            )
            
            # 生成子订单
            child_orders = []
            execution_schedule = []
            remaining_volume = total_volume
            order_index = 0
            
            while remaining_volume > 0:
                current_size = min(chunk_size, remaining_volume)
                
                child_orders.append(self._create_child_order(signal, current_size, order_index))
                
                # 计算执行时间间隔
                delay_seconds = self._calculate_execution_delay(order_index, liquidity_analysis)
                execution_time = signal.timestamp + timedelta(seconds=delay_seconds)
                execution_schedule.append(execution_time)
                
                remaining_volume -= current_size
                order_index += 1
            
            return OrderSplit(
                child_orders=child_orders,
                execution_schedule=execution_schedule,
                total_volume=total_volume,
                average_size=total_volume / len(child_orders)
            )
            
        except Exception as e:
            self.logger.error(f"订单拆分失败: {str(e)}")
            # 返回单个订单作为备选
            return OrderSplit(
                child_orders=[self._create_child_order(signal, signal.size, 0)],
                execution_schedule=[signal.timestamp],
                total_volume=signal.size,
                average_size=signal.size
            )
    
    def _calculate_optimal_chunk_size(self, total_volume: float, 
                                    liquidity_analysis: LiquidityAnalysis,
                                    max_chunk_size: float) -> float:
        """计算最优拆分大小"""
        try:
            # 基于流动性水平调整拆分大小
            liquidity_multiplier = {
                LiquidityLevel.HIGH: 1.0,
                LiquidityLevel.MEDIUM: 0.7,
                LiquidityLevel.LOW: 0.5,
                LiquidityLevel.VERY_LOW: 0.3
            }.get(liquidity_analysis.liquidity_level, 0.7)
            
            # 基于市场深度调整
            market_depth = liquidity_analysis.market_depth.get('total_depth', 1000)
            depth_factor = min(1.0, market_depth / 1000.0)
            
            # 计算最优拆分大小
            optimal_size = max_chunk_size * liquidity_multiplier * depth_factor
            
            # 确保不超过总量的50%
            optimal_size = min(optimal_size, total_volume * 0.5)
            
            # 最小拆分大小
            return max(1.0, optimal_size)
            
        except Exception:
            return min(max_chunk_size, total_volume * 0.3)
    
    def _create_child_order(self, parent_signal: Signal, volume: float, index: int) -> Dict[str, Any]:
        """创建子订单"""
        return {
            'parent_signal_id': f"{parent_signal.strategy_id}_{parent_signal.timestamp.strftime('%Y%m%d_%H%M%S')}",
            'child_index': index,
            'symbol': parent_signal.symbol,
            'direction': parent_signal.direction,
            'volume': volume,
            'entry_price': parent_signal.entry_price,
            'sl': parent_signal.sl,
            'tp': parent_signal.tp,
            'strategy_id': parent_signal.strategy_id,
            'metadata': {
                **parent_signal.metadata,
                'is_child_order': True,
                'parent_volume': parent_signal.size,
                'child_index': index
            }
        }
    
    def _calculate_execution_delay(self, order_index: int, 
                                 liquidity_analysis: LiquidityAnalysis) -> int:
        """计算执行延迟(秒)"""
        try:
            # 基础延迟
            base_delay = order_index * 30  # 每个子订单间隔30秒
            
            # 基于流动性调整延迟
            liquidity_multiplier = {
                LiquidityLevel.HIGH: 0.5,
                LiquidityLevel.MEDIUM: 1.0,
                LiquidityLevel.LOW: 1.5,
                LiquidityLevel.VERY_LOW: 2.0
            }.get(liquidity_analysis.liquidity_level, 1.0)
            
            return int(base_delay * liquidity_multiplier)
            
        except Exception:
            return order_index * 60  # 默认1分钟间隔


class ExecutionOptimizerAgent:
    """执行优化Agent"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        self.slippage_predictor = SlippagePredictor(
            lookback_period=self.config.get('slippage_lookback', 100)
        )
        self.liquidity_analyzer = LiquidityAnalyzer()
        self.timing_optimizer = TimingOptimizer()
        self.order_splitter = OrderSplitter()
        
        # 执行历史
        self.execution_history: List[Dict[str, Any]] = []
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'slippage_lookback': 100,
            'max_chunk_size': 10.0,
            'min_liquidity_score': 40,
            'max_impact_cost': 0.001,
            'timing_threshold': 60.0,
            'enable_order_splitting': True,
            'enable_timing_optimization': True
        }
    
    def optimize_execution(self, signal: Signal, market_data: MarketData,
                          account: Account) -> ExecutionPlan:
        """优化订单执行"""
        try:
            # 滑点预测
            slippage_analysis = self.slippage_predictor.predict_slippage(
                signal.symbol, market_data, signal.size, signal.direction
            )
            
            # 流动性分析
            liquidity_analysis = self.liquidity_analyzer.analyze_liquidity(
                signal.symbol, market_data
            )
            
            # 时机分析
            timing_analysis = self.timing_optimizer.analyze_timing(signal, market_data)
            
            # 选择执行方法
            execution_method = self._select_execution_method(
                signal, slippage_analysis, liquidity_analysis, timing_analysis
            )
            
            # 订单拆分
            order_splits = []
            if (self.config.get('enable_order_splitting', True) and 
                signal.size > self.config.get('max_chunk_size', 10.0)):
                
                order_split = self.order_splitter.split_order(
                    signal, liquidity_analysis, self.config.get('max_chunk_size', 10.0)
                )
                order_splits.append(order_split)
            
            # 计算预估成本
            estimated_cost = self._calculate_estimated_cost(
                signal, slippage_analysis, liquidity_analysis
            )
            
            # 确定执行优先级
            execution_priority = self._calculate_execution_priority(
                signal, timing_analysis, liquidity_analysis
            )
            
            # 创建执行计划
            execution_plan = ExecutionPlan(
                signal=signal,
                method=execution_method,
                order_splits=order_splits,
                timing_analysis=timing_analysis,
                slippage_analysis=slippage_analysis,
                liquidity_analysis=liquidity_analysis,
                estimated_cost=estimated_cost,
                execution_priority=execution_priority
            )
            
            return execution_plan
            
        except Exception as e:
            self.logger.error(f"执行优化失败: {str(e)}")
            raise OrderException(f"执行优化失败: {str(e)}")
    
    def should_delay_execution(self, execution_plan: ExecutionPlan) -> bool:
        """判断是否应该延迟执行"""
        try:
            # 检查时机分析
            if execution_plan.timing_analysis.should_delay_execution():
                return True
            
            # 检查流动性
            if execution_plan.liquidity_analysis.liquidity_level == LiquidityLevel.VERY_LOW:
                return True
            
            # 检查预估成本
            if execution_plan.get_total_estimated_cost() > self.config.get('max_impact_cost', 0.001):
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"延迟判断失败: {str(e)}")
            return False
    
    def record_execution_result(self, signal: Signal, executed_price: float,
                              execution_time: datetime, slippage: float) -> None:
        """记录执行结果"""
        try:
            # 记录到滑点预测器
            self.slippage_predictor.record_actual_slippage(
                signal.symbol, signal.entry_price, executed_price, execution_time
            )
            
            # 记录到执行历史
            execution_record = {
                'signal_id': f"{signal.strategy_id}_{signal.timestamp.strftime('%Y%m%d_%H%M%S')}",
                'symbol': signal.symbol,
                'expected_price': signal.entry_price,
                'executed_price': executed_price,
                'slippage': slippage,
                'execution_time': execution_time,
                'volume': signal.size,
                'direction': signal.direction
            }
            
            self.execution_history.append(execution_record)
            
            # 保持历史记录在合理范围内
            if len(self.execution_history) > 1000:
                self.execution_history = self.execution_history[-500:]
                
        except Exception as e:
            self.logger.error(f"执行结果记录失败: {str(e)}")
    
    def get_execution_statistics(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """获取执行统计"""
        try:
            if not self.execution_history:
                return {"status": "no_data"}
            
            # 过滤数据
            data = self.execution_history
            if symbol:
                data = [record for record in data if record['symbol'] == symbol]
            
            if not data:
                return {"status": "no_data", "symbol": symbol}
            
            # 计算统计指标
            slippages = [record['slippage'] for record in data]
            
            return {
                "status": "success",
                "symbol": symbol,
                "total_executions": len(data),
                "average_slippage": np.mean(slippages),
                "median_slippage": np.median(slippages),
                "max_slippage": max(slippages),
                "min_slippage": min(slippages),
                "slippage_std": np.std(slippages),
                "recent_executions": len([r for r in data if 
                                        (datetime.now() - r['execution_time']).days <= 7])
            }
            
        except Exception as e:
            self.logger.error(f"执行统计计算失败: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _select_execution_method(self, signal: Signal, slippage_analysis: SlippageAnalysis,
                               liquidity_analysis: LiquidityAnalysis,
                               timing_analysis: TimingAnalysis) -> ExecutionMethod:
        """选择执行方法"""
        try:
            # 高流动性且低滑点风险 -> 市价单
            if (liquidity_analysis.liquidity_level in [LiquidityLevel.HIGH, LiquidityLevel.MEDIUM] and
                slippage_analysis.expected_slippage <= 2.0):
                return ExecutionMethod.MARKET
            
            # 大订单 -> TWAP或冰山订单
            if signal.size > self.config.get('max_chunk_size', 10.0):
                if liquidity_analysis.liquidity_level == LiquidityLevel.HIGH:
                    return ExecutionMethod.TWAP
                else:
                    return ExecutionMethod.ICEBERG
            
            # 时机不佳 -> 限价单
            if timing_analysis.entry_score < self.config.get('timing_threshold', 60.0):
                return ExecutionMethod.LIMIT
            
            # 默认市价单
            return ExecutionMethod.MARKET
            
        except Exception:
            return ExecutionMethod.MARKET
    
    def _calculate_estimated_cost(self, signal: Signal, slippage_analysis: SlippageAnalysis,
                                liquidity_analysis: LiquidityAnalysis) -> float:
        """计算预估成本"""
        try:
            # 滑点成本
            slippage_cost = slippage_analysis.expected_slippage * signal.size * 0.1  # 假设每点0.1成本
            
            # 市场冲击成本
            impact_cost = liquidity_analysis.impact_cost * signal.size
            
            # 点差成本
            spread_cost = liquidity_analysis.bid_ask_spread * signal.size * 0.5
            
            return slippage_cost + impact_cost + spread_cost
            
        except Exception:
            return signal.size * 0.001  # 默认成本
    
    def _calculate_execution_priority(self, signal: Signal, timing_analysis: TimingAnalysis,
                                    liquidity_analysis: LiquidityAnalysis) -> int:
        """计算执行优先级"""
        try:
            priority = 5  # 基础优先级
            
            # 基于信号强度调整
            priority += int(signal.strength * 3)
            
            # 基于紧急程度调整
            priority += int(timing_analysis.execution_urgency * 2)
            
            # 基于流动性调整
            if liquidity_analysis.liquidity_level == LiquidityLevel.HIGH:
                priority += 1
            elif liquidity_analysis.liquidity_level == LiquidityLevel.VERY_LOW:
                priority -= 2
            
            return max(1, min(10, priority))
            
        except Exception:
            return 5
    
    def clear_history(self) -> None:
        """清空历史记录"""
        self.execution_history.clear()
        self.slippage_predictor.slippage_history.clear()