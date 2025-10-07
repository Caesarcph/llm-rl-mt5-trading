"""
RL策略优化器
将RL模型集成到策略权重调整系统，实现在线学习和模型更新
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

from ..core.models import Signal, MarketData, Account, Position
from .trading_env import TradingEnvironment
from .rl_trainer import RLTrainer, TrainingConfig


logger = logging.getLogger(__name__)


@dataclass
class RLOptimizerConfig:
    """RL优化器配置"""
    # 在线学习配置
    online_learning_enabled: bool = True
    update_frequency: int = 1000  # 每N步更新一次模型
    min_samples_for_update: int = 100  # 最小样本数
    
    # 策略权重配置
    initial_rl_weight: float = 0.3  # RL策略初始权重
    max_rl_weight: float = 0.7  # RL策略最大权重
    min_rl_weight: float = 0.1  # RL策略最小权重
    weight_adjustment_rate: float = 0.05  # 权重调整速率
    
    # 性能评估配置
    performance_window: int = 50  # 性能评估窗口
    performance_threshold: float = 0.6  # 性能阈值
    
    # 模型保存配置
    model_save_path: str = "models/rl_optimizer"
    checkpoint_frequency: int = 5000  # 检查点频率


class RLStrategyOptimizer:
    """
    RL策略优化器
    
    功能:
    1. 将RL模型集成到策略权重调整系统
    2. 实现在线学习和模型更新
    3. 动态调整RL策略权重
    4. 融合RL决策与传统策略
    """
    
    def __init__(
        self,
        trainer: RLTrainer,
        config: Optional[RLOptimizerConfig] = None
    ):
        """
        初始化RL策略优化器
        
        Args:
            trainer: RL训练器
            config: 优化器配置
        """
        self.trainer = trainer
        self.config = config or RLOptimizerConfig()
        
        # 策略权重
        self.rl_weight = self.config.initial_rl_weight
        self.traditional_weight = 1.0 - self.rl_weight
        
        # 性能跟踪
        self.rl_performance_history: List[float] = []
        self.traditional_performance_history: List[float] = []
        self.combined_performance_history: List[float] = []
        
        # 在线学习状态
        self.steps_since_update = 0
        self.online_samples: List[Tuple[np.ndarray, int, float, np.ndarray]] = []
        
        # 统计信息
        self.total_predictions = 0
        self.rl_predictions = 0
        self.traditional_predictions = 0
        self.weight_adjustments = 0
        
        os.makedirs(self.config.model_save_path, exist_ok=True)
        
        logger.info(f"初始化RL策略优化器: RL权重={self.rl_weight:.2f}")
    
    def get_combined_signal(
        self,
        market_data: MarketData,
        traditional_signals: List[Signal],
        observation: np.ndarray
    ) -> Optional[Signal]:
        """
        获取组合信号 - 融合RL决策与传统策略
        
        Args:
            market_data: 市场数据
            traditional_signals: 传统策略信号列表
            observation: RL模型观察
            
        Returns:
            组合后的信号
        """
        self.total_predictions += 1
        
        # 获取RL预测
        rl_action = self.trainer.predict(observation, deterministic=True)
        rl_signal = self._action_to_signal(rl_action, market_data)
        
        # 如果没有传统信号，直接使用RL信号
        if not traditional_signals:
            if rl_signal is not None:
                self.rl_predictions += 1
                return rl_signal
            return None
        
        # 聚合传统信号
        aggregated_traditional = self._aggregate_traditional_signals(traditional_signals)
        
        # 根据权重融合信号
        combined_signal = self._fuse_signals(
            rl_signal,
            aggregated_traditional,
            market_data
        )
        
        # 记录使用的策略类型
        if combined_signal is not None:
            if combined_signal.metadata.get('source') == 'rl':
                self.rl_predictions += 1
            else:
                self.traditional_predictions += 1
        
        return combined_signal
    
    def _action_to_signal(self, action: int, market_data: MarketData) -> Optional[Signal]:
        """将RL动作转换为交易信号"""
        from .trading_env import ActionType
        
        action_type = ActionType(action)
        current_price = market_data.ohlcv.iloc[-1]['close']
        
        if action_type == ActionType.BUY:
            return Signal(
                strategy_id='rl_model',
                symbol=market_data.symbol,
                direction=1,
                strength=0.8,
                entry_price=current_price,
                sl=current_price * 0.98,  # 2% 止损
                tp=current_price * 1.04,  # 4% 止盈
                size=0.1,
                confidence=0.8,
                timestamp=datetime.now(),
                metadata={'source': 'rl', 'action': action_type.name}
            )
        elif action_type == ActionType.SELL:
            return Signal(
                strategy_id='rl_model',
                symbol=market_data.symbol,
                direction=-1,
                strength=0.8,
                entry_price=current_price,
                sl=current_price * 1.02,  # 2% 止损
                tp=current_price * 0.96,  # 4% 止盈
                size=0.1,
                confidence=0.8,
                timestamp=datetime.now(),
                metadata={'source': 'rl', 'action': action_type.name}
            )
        
        return None  # HOLD, CLOSE, ADJUST不生成新信号
    
    def _aggregate_traditional_signals(self, signals: List[Signal]) -> Optional[Signal]:
        """聚合传统策略信号"""
        if not signals:
            return None
        
        # 计算加权平均方向
        total_weight = sum(s.strength * s.confidence for s in signals)
        if total_weight == 0:
            return None
        
        weighted_direction = sum(
            s.direction * s.strength * s.confidence for s in signals
        ) / total_weight
        
        # 选择最强信号作为基础
        strongest_signal = max(signals, key=lambda s: s.strength * s.confidence)
        
        # 创建聚合信号
        aggregated = Signal(
            strategy_id='traditional_aggregated',
            symbol=strongest_signal.symbol,
            direction=1 if weighted_direction > 0 else -1,
            strength=min(abs(weighted_direction), 1.0),
            entry_price=strongest_signal.entry_price,
            sl=strongest_signal.sl,
            tp=strongest_signal.tp,
            size=strongest_signal.size,
            confidence=total_weight / len(signals),
            timestamp=datetime.now(),
            metadata={'source': 'traditional', 'num_signals': len(signals)}
        )
        
        return aggregated
    
    def _fuse_signals(
        self,
        rl_signal: Optional[Signal],
        traditional_signal: Optional[Signal],
        market_data: MarketData
    ) -> Optional[Signal]:
        """融合RL信号和传统信号"""
        # 如果只有一个信号，直接返回
        if rl_signal is None:
            return traditional_signal
        if traditional_signal is None:
            return rl_signal
        
        # 计算信号得分
        rl_score = rl_signal.strength * rl_signal.confidence * self.rl_weight
        traditional_score = traditional_signal.strength * traditional_signal.confidence * self.traditional_weight
        
        # 如果方向一致，增强信号
        if rl_signal.direction == traditional_signal.direction:
            combined_strength = min((rl_score + traditional_score) / 2, 1.0)
            combined_confidence = min(
                (rl_signal.confidence * self.rl_weight + 
                 traditional_signal.confidence * self.traditional_weight),
                1.0
            )
            
            # 使用传统信号的价格参数（通常更保守）
            return Signal(
                strategy_id='rl_traditional_combined',
                symbol=market_data.symbol,
                direction=rl_signal.direction,
                strength=combined_strength,
                entry_price=traditional_signal.entry_price,
                sl=traditional_signal.sl,
                tp=traditional_signal.tp,
                size=min(rl_signal.size, traditional_signal.size),
                confidence=combined_confidence,
                timestamp=datetime.now(),
                metadata={
                    'source': 'combined',
                    'rl_score': rl_score,
                    'traditional_score': traditional_score
                }
            )
        else:
            # 方向冲突，选择得分更高的
            if rl_score > traditional_score:
                rl_signal.metadata['source'] = 'rl'
                return rl_signal
            else:
                traditional_signal.metadata['source'] = 'traditional'
                return traditional_signal
    
    def update_performance(
        self,
        rl_performance: float,
        traditional_performance: float,
        combined_performance: float
    ) -> None:
        """
        更新性能指标并调整权重
        
        Args:
            rl_performance: RL策略性能
            traditional_performance: 传统策略性能
            combined_performance: 组合策略性能
        """
        self.rl_performance_history.append(rl_performance)
        self.traditional_performance_history.append(traditional_performance)
        self.combined_performance_history.append(combined_performance)
        
        # 保持窗口大小
        if len(self.rl_performance_history) > self.config.performance_window:
            self.rl_performance_history.pop(0)
            self.traditional_performance_history.pop(0)
            self.combined_performance_history.pop(0)
        
        # 调整权重
        if len(self.rl_performance_history) >= self.config.performance_window:
            self._adjust_weights()
    
    def _adjust_weights(self) -> None:
        """根据性能调整策略权重"""
        # 计算平均性能
        avg_rl = np.mean(self.rl_performance_history)
        avg_traditional = np.mean(self.traditional_performance_history)
        
        # 计算性能差异
        performance_diff = avg_rl - avg_traditional
        
        # 调整权重
        old_weight = self.rl_weight
        
        if performance_diff > 0.1:  # RL明显更好
            self.rl_weight = min(
                self.rl_weight + self.config.weight_adjustment_rate,
                self.config.max_rl_weight
            )
        elif performance_diff < -0.1:  # 传统策略明显更好
            self.rl_weight = max(
                self.rl_weight - self.config.weight_adjustment_rate,
                self.config.min_rl_weight
            )
        
        self.traditional_weight = 1.0 - self.rl_weight
        
        if abs(old_weight - self.rl_weight) > 0.001:
            self.weight_adjustments += 1
            logger.info(
                f"调整策略权重: RL={self.rl_weight:.2f}, "
                f"传统={self.traditional_weight:.2f}, "
                f"性能差异={performance_diff:.3f}"
            )
    
    def add_online_sample(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray
    ) -> None:
        """
        添加在线学习样本
        
        Args:
            observation: 观察
            action: 动作
            reward: 奖励
            next_observation: 下一个观察
        """
        if not self.config.online_learning_enabled:
            return
        
        self.online_samples.append((observation, action, reward, next_observation))
        self.steps_since_update += 1
        
        # 检查是否需要更新模型
        if (self.steps_since_update >= self.config.update_frequency and
            len(self.online_samples) >= self.config.min_samples_for_update):
            self._update_model_online()
    
    def _update_model_online(self) -> None:
        """在线更新RL模型"""
        logger.info(f"开始在线学习: {len(self.online_samples)} 个样本")
        
        try:
            # 使用收集的样本继续训练
            # 注意: 这是简化实现，实际应该使用replay buffer
            self.trainer.continue_training(
                additional_timesteps=len(self.online_samples)
            )
            
            # 清空样本
            self.online_samples.clear()
            self.steps_since_update = 0
            
            # 保存检查点
            if self.total_predictions % self.config.checkpoint_frequency == 0:
                self.save_checkpoint()
            
            logger.info("在线学习完成")
            
        except Exception as e:
            logger.error(f"在线学习失败: {e}")
    
    def save_checkpoint(self) -> str:
        """保存检查点"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"rl_optimizer_{timestamp}"
        
        # 保存模型
        model_path = self.trainer.save_model(checkpoint_name)
        
        # 保存优化器状态
        state = {
            'rl_weight': self.rl_weight,
            'traditional_weight': self.traditional_weight,
            'rl_performance_history': self.rl_performance_history,
            'traditional_performance_history': self.traditional_performance_history,
            'combined_performance_history': self.combined_performance_history,
            'total_predictions': self.total_predictions,
            'rl_predictions': self.rl_predictions,
            'traditional_predictions': self.traditional_predictions,
            'weight_adjustments': self.weight_adjustments
        }
        
        state_path = os.path.join(self.config.model_save_path, f"{checkpoint_name}_state.npy")
        np.save(state_path, state)
        
        logger.info(f"检查点已保存: {checkpoint_name}")
        return checkpoint_name
    
    def load_checkpoint(self, checkpoint_name: str) -> None:
        """加载检查点"""
        # 加载模型
        model_path = os.path.join(self.config.model_save_path, checkpoint_name)
        self.trainer.load_model(model_path)
        
        # 加载优化器状态
        state_path = os.path.join(self.config.model_save_path, f"{checkpoint_name}_state.npy")
        state = np.load(state_path, allow_pickle=True).item()
        
        self.rl_weight = state['rl_weight']
        self.traditional_weight = state['traditional_weight']
        self.rl_performance_history = state['rl_performance_history']
        self.traditional_performance_history = state['traditional_performance_history']
        self.combined_performance_history = state['combined_performance_history']
        self.total_predictions = state['total_predictions']
        self.rl_predictions = state['rl_predictions']
        self.traditional_predictions = state['traditional_predictions']
        self.weight_adjustments = state['weight_adjustments']
        
        logger.info(f"检查点已加载: {checkpoint_name}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'rl_weight': self.rl_weight,
            'traditional_weight': self.traditional_weight,
            'total_predictions': self.total_predictions,
            'rl_predictions': self.rl_predictions,
            'traditional_predictions': self.traditional_predictions,
            'rl_prediction_rate': self.rl_predictions / max(self.total_predictions, 1),
            'weight_adjustments': self.weight_adjustments,
            'avg_rl_performance': np.mean(self.rl_performance_history) if self.rl_performance_history else 0.0,
            'avg_traditional_performance': np.mean(self.traditional_performance_history) if self.traditional_performance_history else 0.0,
            'avg_combined_performance': np.mean(self.combined_performance_history) if self.combined_performance_history else 0.0,
            'online_samples_collected': len(self.online_samples),
            'steps_since_update': self.steps_since_update
        }
    
    def reset_statistics(self) -> None:
        """重置统计信息"""
        self.total_predictions = 0
        self.rl_predictions = 0
        self.traditional_predictions = 0
        self.weight_adjustments = 0
        self.rl_performance_history.clear()
        self.traditional_performance_history.clear()
        self.combined_performance_history.clear()
        
        logger.info("统计信息已重置")
