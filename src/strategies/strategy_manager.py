"""
多策略管理系统
统一管理所有交易策略，包括策略注册、动态加载、信号聚合和冲突解决
"""

import logging
from typing import Dict, List, Optional, Any, Callable, Type
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import importlib
import inspect

from src.core.models import MarketData, Signal, Strategy
from src.core.exceptions import StrategyException
from src.strategies.base_strategies import (
    BaseStrategy, StrategyConfig, StrategyPerformance,
    TrendFollowingStrategy, MeanReversionStrategy, BreakoutStrategy
)


class ConflictResolutionMethod(Enum):
    """信号冲突解决方法"""
    HIGHEST_STRENGTH = "highest_strength"  # 选择强度最高的信号
    HIGHEST_CONFIDENCE = "highest_confidence"  # 选择置信度最高的信号
    WEIGHTED_AVERAGE = "weighted_average"  # 加权平均
    MAJORITY_VOTE = "majority_vote"  # 多数投票
    FIRST_SIGNAL = "first_signal"  # 第一个信号
    CANCEL_ALL = "cancel_all"  # 取消所有冲突信号


class SignalAggregationMethod(Enum):
    """信号聚合方法"""
    SIMPLE_AVERAGE = "simple_average"  # 简单平均
    WEIGHTED_AVERAGE = "weighted_average"  # 加权平均
    ENSEMBLE = "ensemble"  # 集成方法
    FILTER_TOP_N = "filter_top_n"  # 筛选前N个


@dataclass
class StrategyRegistration:
    """策略注册信息"""
    name: str
    strategy_class: Type[BaseStrategy]
    config: StrategyConfig
    weight: float = 1.0
    enabled: bool = True
    auto_load: bool = True
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SignalAggregationResult:
    """信号聚合结果"""
    aggregated_signal: Optional[Signal]
    original_signals: List[Signal]
    conflicts_detected: bool
    resolution_method: str
    aggregation_metadata: Dict[str, Any] = field(default_factory=dict)


class StrategyManager:
    """
    多策略管理器核心类
    负责策略的注册、加载、信号生成、聚合和冲突解决
    """
    
    def __init__(
        self,
        conflict_resolution: ConflictResolutionMethod = ConflictResolutionMethod.HIGHEST_STRENGTH,
        aggregation_method: SignalAggregationMethod = SignalAggregationMethod.WEIGHTED_AVERAGE,
        max_signals_per_symbol: int = 3
    ):
        """
        初始化策略管理器
        
        Args:
            conflict_resolution: 冲突解决方法
            aggregation_method: 信号聚合方法
            max_signals_per_symbol: 每个品种最大信号数
        """
        self.logger = logging.getLogger(__name__)
        
        # 策略存储
        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_registrations: Dict[str, StrategyRegistration] = {}
        self.strategy_weights: Dict[str, float] = {}
        
        # 配置
        self.conflict_resolution = conflict_resolution
        self.aggregation_method = aggregation_method
        self.max_signals_per_symbol = max_signals_per_symbol
        
        # 性能跟踪
        self.performance_history: Dict[str, List[StrategyPerformance]] = {}
        
        # 信号历史
        self.signal_history: List[Signal] = []
        self.max_signal_history = 1000
        
        self.logger.info("策略管理器初始化完成")
    
    def register_strategy(
        self,
        name: str,
        strategy_class: Type[BaseStrategy],
        config: StrategyConfig,
        weight: float = 1.0,
        enabled: bool = True,
        auto_load: bool = True,
        dependencies: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        注册策略
        
        Args:
            name: 策略名称
            strategy_class: 策略类
            config: 策略配置
            weight: 策略权重
            enabled: 是否启用
            auto_load: 是否自动加载
            dependencies: 依赖的其他策略
            metadata: 元数据
            
        Returns:
            注册是否成功
        """
        try:
            # 验证策略类
            if not issubclass(strategy_class, BaseStrategy):
                raise ValueError(f"策略类必须继承自BaseStrategy: {strategy_class}")
            
            # 检查是否已注册
            if name in self.strategy_registrations:
                self.logger.warning(f"策略{name}已注册，将覆盖")
            
            # 创建注册信息
            registration = StrategyRegistration(
                name=name,
                strategy_class=strategy_class,
                config=config,
                weight=weight,
                enabled=enabled,
                auto_load=auto_load,
                dependencies=dependencies or [],
                metadata=metadata or {}
            )
            
            self.strategy_registrations[name] = registration
            self.strategy_weights[name] = weight
            
            # 如果设置了自动加载，立即加载策略
            if auto_load:
                self.load_strategy(name)
            
            self.logger.info(f"策略{name}注册成功，权重: {weight}")
            return True
            
        except Exception as e:
            self.logger.error(f"注册策略{name}失败: {str(e)}")
            raise StrategyException(f"策略注册失败: {str(e)}")
    
    def load_strategy(self, name: str) -> bool:
        """
        加载策略实例
        
        Args:
            name: 策略名称
            
        Returns:
            加载是否成功
        """
        try:
            if name not in self.strategy_registrations:
                raise ValueError(f"策略{name}未注册")
            
            registration = self.strategy_registrations[name]
            
            # 检查依赖
            for dep in registration.dependencies:
                if dep not in self.strategies:
                    self.logger.warning(f"策略{name}依赖{dep}未加载，尝试加载")
                    self.load_strategy(dep)
            
            # 创建策略实例
            strategy = registration.strategy_class(registration.config)
            # 使用注册信息中的enabled状态，而不是config中的
            strategy.is_enabled = registration.enabled
            strategy.config.enabled = registration.enabled
            
            self.strategies[name] = strategy
            
            # 初始化性能历史
            if name not in self.performance_history:
                self.performance_history[name] = []
            
            self.logger.info(f"策略{name}加载成功")
            return True
            
        except Exception as e:
            self.logger.error(f"加载策略{name}失败: {str(e)}")
            raise StrategyException(f"策略加载失败: {str(e)}")
    
    def unload_strategy(self, name: str) -> bool:
        """
        卸载策略实例
        
        Args:
            name: 策略名称
            
        Returns:
            卸载是否成功
        """
        try:
            if name in self.strategies:
                del self.strategies[name]
                self.logger.info(f"策略{name}卸载成功")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"卸载策略{name}失败: {str(e)}")
            return False
    
    def unregister_strategy(self, name: str) -> bool:
        """
        注销策略
        
        Args:
            name: 策略名称
            
        Returns:
            注销是否成功
        """
        try:
            # 先卸载实例
            self.unload_strategy(name)
            
            # 删除注册信息
            if name in self.strategy_registrations:
                del self.strategy_registrations[name]
            if name in self.strategy_weights:
                del self.strategy_weights[name]
            
            self.logger.info(f"策略{name}注销成功")
            return True
            
        except Exception as e:
            self.logger.error(f"注销策略{name}失败: {str(e)}")
            return False
    
    def generate_signals(self, market_data: MarketData) -> List[Signal]:
        """
        生成所有活跃策略的信号
        
        Args:
            market_data: 市场数据
            
        Returns:
            信号列表
        """
        signals = []
        
        for strategy_name, strategy in self.strategies.items():
            if not strategy.is_enabled:
                continue
            
            try:
                signal = strategy.generate_signal(market_data)
                
                if signal:
                    # 应用策略权重
                    weight = self.strategy_weights.get(strategy_name, 1.0)
                    signal.strength *= weight
                    signal.confidence *= weight
                    
                    # 添加策略信息到元数据
                    signal.metadata['strategy_weight'] = weight
                    signal.metadata['original_strength'] = signal.strength / weight
                    
                    signals.append(signal)
                    
                    # 记录信号历史
                    self._add_to_signal_history(signal)
                    
            except Exception as e:
                self.logger.error(f"策略{strategy_name}生成信号失败: {str(e)}")
        
        self.logger.debug(f"生成{len(signals)}个信号，品种: {market_data.symbol}")
        return signals
    
    def aggregate_signals(
        self,
        signals: List[Signal],
        symbol: str = None
    ) -> SignalAggregationResult:
        """
        聚合多个信号
        
        Args:
            signals: 信号列表
            symbol: 品种（可选，用于过滤）
            
        Returns:
            信号聚合结果
        """
        if not signals:
            return SignalAggregationResult(
                aggregated_signal=None,
                original_signals=[],
                conflicts_detected=False,
                resolution_method="none"
            )
        
        # 过滤品种
        if symbol:
            signals = [s for s in signals if s.symbol == symbol]
        
        # 检测冲突
        conflicts_detected = self._detect_conflicts(signals)
        
        # 如果有冲突，使用冲突解决方法
        if conflicts_detected:
            aggregated_signal = self._resolve_conflicts(signals)
            resolution_method = self.conflict_resolution.value
        else:
            # 没有冲突，使用聚合方法
            aggregated_signal = self._aggregate_signals(signals)
            resolution_method = self.aggregation_method.value
        
        return SignalAggregationResult(
            aggregated_signal=aggregated_signal,
            original_signals=signals,
            conflicts_detected=conflicts_detected,
            resolution_method=resolution_method,
            aggregation_metadata={
                'num_signals': len(signals),
                'avg_strength': sum(s.strength for s in signals) / len(signals),
                'avg_confidence': sum(s.confidence for s in signals) / len(signals)
            }
        )
    
    def _detect_conflicts(self, signals: List[Signal]) -> bool:
        """
        检测信号冲突
        
        Args:
            signals: 信号列表
            
        Returns:
            是否存在冲突
        """
        if len(signals) <= 1:
            return False
        
        # 检查方向冲突
        directions = set(s.direction for s in signals)
        
        # 如果有相反方向的信号，视为冲突
        if 1 in directions and -1 in directions:
            return True
        
        # 如果有平仓信号和开仓信号，视为冲突
        if 0 in directions and len(directions) > 1:
            return True
        
        return False
    
    def _resolve_conflicts(self, signals: List[Signal]) -> Optional[Signal]:
        """
        解决信号冲突
        
        Args:
            signals: 冲突的信号列表
            
        Returns:
            解决后的信号
        """
        if not signals:
            return None
        
        method = self.conflict_resolution
        
        if method == ConflictResolutionMethod.HIGHEST_STRENGTH:
            return max(signals, key=lambda s: s.strength)
        
        elif method == ConflictResolutionMethod.HIGHEST_CONFIDENCE:
            return max(signals, key=lambda s: s.confidence)
        
        elif method == ConflictResolutionMethod.WEIGHTED_AVERAGE:
            return self._weighted_average_signal(signals)
        
        elif method == ConflictResolutionMethod.MAJORITY_VOTE:
            return self._majority_vote_signal(signals)
        
        elif method == ConflictResolutionMethod.FIRST_SIGNAL:
            return signals[0]
        
        elif method == ConflictResolutionMethod.CANCEL_ALL:
            self.logger.warning("检测到信号冲突，取消所有信号")
            return None
        
        return None
    
    def _aggregate_signals(self, signals: List[Signal]) -> Optional[Signal]:
        """
        聚合无冲突的信号
        
        Args:
            signals: 信号列表
            
        Returns:
            聚合后的信号
        """
        if not signals:
            return None
        
        method = self.aggregation_method
        
        if method == SignalAggregationMethod.SIMPLE_AVERAGE:
            return self._simple_average_signal(signals)
        
        elif method == SignalAggregationMethod.WEIGHTED_AVERAGE:
            return self._weighted_average_signal(signals)
        
        elif method == SignalAggregationMethod.ENSEMBLE:
            return self._ensemble_signal(signals)
        
        elif method == SignalAggregationMethod.FILTER_TOP_N:
            # 选择强度最高的N个信号
            top_signals = sorted(signals, key=lambda s: s.strength, reverse=True)[:self.max_signals_per_symbol]
            return self._weighted_average_signal(top_signals)
        
        return signals[0]
    
    def _simple_average_signal(self, signals: List[Signal]) -> Signal:
        """简单平均信号"""
        if not signals:
            return None
        
        # 使用第一个信号作为模板
        base_signal = signals[0]
        
        # 计算平均值
        avg_strength = sum(s.strength for s in signals) / len(signals)
        avg_confidence = sum(s.confidence for s in signals) / len(signals)
        avg_entry = sum(s.entry_price for s in signals) / len(signals)
        avg_sl = sum(s.sl for s in signals) / len(signals)
        avg_tp = sum(s.tp for s in signals) / len(signals)
        avg_size = sum(s.size for s in signals) / len(signals)
        
        # 方向取多数
        direction = self._get_majority_direction(signals)
        
        return Signal(
            strategy_id="aggregated",
            symbol=base_signal.symbol,
            direction=direction,
            strength=avg_strength,
            entry_price=avg_entry,
            sl=avg_sl,
            tp=avg_tp,
            size=avg_size,
            confidence=avg_confidence,
            timestamp=datetime.now(),
            metadata={
                'aggregation_method': 'simple_average',
                'num_signals': len(signals),
                'source_strategies': [s.strategy_id for s in signals]
            }
        )
    
    def _weighted_average_signal(self, signals: List[Signal]) -> Signal:
        """加权平均信号"""
        if not signals:
            return None
        
        # 使用置信度作为权重
        total_weight = sum(s.confidence for s in signals)
        
        if total_weight == 0:
            return self._simple_average_signal(signals)
        
        # 计算加权平均
        weighted_strength = sum(s.strength * s.confidence for s in signals) / total_weight
        weighted_entry = sum(s.entry_price * s.confidence for s in signals) / total_weight
        weighted_sl = sum(s.sl * s.confidence for s in signals) / total_weight
        weighted_tp = sum(s.tp * s.confidence for s in signals) / total_weight
        weighted_size = sum(s.size * s.confidence for s in signals) / total_weight
        avg_confidence = sum(s.confidence for s in signals) / len(signals)
        
        # 方向取多数
        direction = self._get_majority_direction(signals)
        
        base_signal = signals[0]
        
        return Signal(
            strategy_id="aggregated",
            symbol=base_signal.symbol,
            direction=direction,
            strength=weighted_strength,
            entry_price=weighted_entry,
            sl=weighted_sl,
            tp=weighted_tp,
            size=weighted_size,
            confidence=avg_confidence,
            timestamp=datetime.now(),
            metadata={
                'aggregation_method': 'weighted_average',
                'num_signals': len(signals),
                'source_strategies': [s.strategy_id for s in signals],
                'total_weight': total_weight
            }
        )
    
    def _ensemble_signal(self, signals: List[Signal]) -> Signal:
        """集成方法信号"""
        # 集成方法：结合强度和置信度
        if not signals:
            return None
        
        # 计算综合得分
        scored_signals = [
            (s, s.strength * 0.6 + s.confidence * 0.4)
            for s in signals
        ]
        
        # 选择得分最高的信号
        best_signal = max(scored_signals, key=lambda x: x[1])[0]
        
        # 但使用所有信号的平均置信度
        avg_confidence = sum(s.confidence for s in signals) / len(signals)
        
        result = Signal(
            strategy_id="aggregated",
            symbol=best_signal.symbol,
            direction=best_signal.direction,
            strength=best_signal.strength,
            entry_price=best_signal.entry_price,
            sl=best_signal.sl,
            tp=best_signal.tp,
            size=best_signal.size,
            confidence=avg_confidence,
            timestamp=datetime.now(),
            metadata={
                'aggregation_method': 'ensemble',
                'num_signals': len(signals),
                'source_strategies': [s.strategy_id for s in signals],
                'best_strategy': best_signal.strategy_id
            }
        )
        
        return result
    
    def _majority_vote_signal(self, signals: List[Signal]) -> Optional[Signal]:
        """多数投票信号"""
        if not signals:
            return None
        
        # 统计方向投票
        direction_votes = {}
        for signal in signals:
            direction = signal.direction
            if direction not in direction_votes:
                direction_votes[direction] = []
            direction_votes[direction].append(signal)
        
        # 找出票数最多的方向
        majority_direction = max(direction_votes.keys(), key=lambda d: len(direction_votes[d]))
        majority_signals = direction_votes[majority_direction]
        
        # 对多数方向的信号进行加权平均
        return self._weighted_average_signal(majority_signals)
    
    def _get_majority_direction(self, signals: List[Signal]) -> int:
        """获取多数方向"""
        if not signals:
            return 0
        
        direction_counts = {}
        for signal in signals:
            direction = signal.direction
            direction_counts[direction] = direction_counts.get(direction, 0) + 1
        
        return max(direction_counts.keys(), key=lambda d: direction_counts[d])
    
    def _add_to_signal_history(self, signal: Signal) -> None:
        """添加信号到历史记录"""
        self.signal_history.append(signal)
        
        # 限制历史记录大小
        if len(self.signal_history) > self.max_signal_history:
            self.signal_history = self.signal_history[-self.max_signal_history:]
    
    def update_strategy_weights(self, performance_data: Dict[str, float]) -> None:
        """
        根据性能数据更新策略权重
        
        Args:
            performance_data: 策略名称到性能评分的映射
        """
        for strategy_name, performance_score in performance_data.items():
            if strategy_name in self.strategy_weights:
                # 基于性能调整权重，限制在0.1-2.0之间
                new_weight = max(0.1, min(2.0, performance_score))
                old_weight = self.strategy_weights[strategy_name]
                
                self.strategy_weights[strategy_name] = new_weight
                
                # 更新注册信息中的权重
                if strategy_name in self.strategy_registrations:
                    self.strategy_registrations[strategy_name].weight = new_weight
                
                self.logger.info(
                    f"更新策略{strategy_name}权重: {old_weight:.2f} -> {new_weight:.2f}"
                )
    
    def get_strategy_performance(self) -> Dict[str, StrategyPerformance]:
        """
        获取所有策略的性能指标
        
        Returns:
            策略名称到性能指标的映射
        """
        return {
            name: strategy.get_performance_metrics()
            for name, strategy in self.strategies.items()
        }
    
    def enable_strategy(self, strategy_name: str) -> bool:
        """
        启用策略
        
        Args:
            strategy_name: 策略名称
            
        Returns:
            操作是否成功
        """
        if strategy_name in self.strategies:
            self.strategies[strategy_name].is_enabled = True
            if strategy_name in self.strategy_registrations:
                self.strategy_registrations[strategy_name].enabled = True
            self.logger.info(f"启用策略: {strategy_name}")
            return True
        return False
    
    def disable_strategy(self, strategy_name: str) -> bool:
        """
        禁用策略
        
        Args:
            strategy_name: 策略名称
            
        Returns:
            操作是否成功
        """
        if strategy_name in self.strategies:
            self.strategies[strategy_name].is_enabled = False
            if strategy_name in self.strategy_registrations:
                self.strategy_registrations[strategy_name].enabled = False
            self.logger.info(f"禁用策略: {strategy_name}")
            return True
        return False
    
    def get_active_strategies(self) -> List[str]:
        """
        获取活跃策略列表
        
        Returns:
            活跃策略名称列表
        """
        return [
            name for name, strategy in self.strategies.items()
            if strategy.is_enabled
        ]
    
    def get_strategy_info(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """
        获取策略信息
        
        Args:
            strategy_name: 策略名称
            
        Returns:
            策略信息字典
        """
        if strategy_name not in self.strategies:
            return None
        
        strategy = self.strategies[strategy_name]
        registration = self.strategy_registrations.get(strategy_name)
        
        info = {
            'name': strategy_name,
            'enabled': strategy.is_enabled,
            'weight': self.strategy_weights.get(strategy_name, 1.0),
            'config': strategy.config.__dict__,
            'performance': strategy.get_performance_metrics().__dict__,
        }
        
        if registration:
            info.update({
                'dependencies': registration.dependencies,
                'metadata': registration.metadata
            })
        
        return info
    
    def get_all_strategies_info(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有策略信息
        
        Returns:
            所有策略信息字典
        """
        return {
            name: self.get_strategy_info(name)
            for name in self.strategies.keys()
        }
    
    def clear_signal_history(self) -> None:
        """清空信号历史"""
        self.signal_history.clear()
        self.logger.info("信号历史已清空")
    
    def get_signal_history(
        self,
        symbol: str = None,
        strategy_id: str = None,
        limit: int = 100
    ) -> List[Signal]:
        """
        获取信号历史
        
        Args:
            symbol: 品种过滤
            strategy_id: 策略ID过滤
            limit: 返回数量限制
            
        Returns:
            信号列表
        """
        signals = self.signal_history
        
        # 过滤
        if symbol:
            signals = [s for s in signals if s.symbol == symbol]
        if strategy_id:
            signals = [s for s in signals if s.strategy_id == strategy_id]
        
        # 限制数量
        return signals[-limit:]
