"""
策略权重优化系统
基于历史表现和RL驱动的动态权重调整
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import pickle
import os

from src.core.models import Signal, Trade, RiskMetrics
from src.strategies.base_strategies import StrategyPerformance


class WeightOptimizationMethod(Enum):
    """权重优化方法"""
    EQUAL_WEIGHT = "equal_weight"  # 等权重
    PERFORMANCE_BASED = "performance_based"  # 基于性能
    SHARPE_RATIO = "sharpe_ratio"  # 基于夏普比率
    WIN_RATE = "win_rate"  # 基于胜率
    PROFIT_FACTOR = "profit_factor"  # 基于盈利因子
    RL_OPTIMIZED = "rl_optimized"  # RL优化
    ENSEMBLE = "ensemble"  # 集成方法


@dataclass
class StrategyMetrics:
    """策略指标"""
    strategy_name: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_profit: float = 0.0
    total_loss: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    avg_profit_per_trade: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def calculate_metrics(self, trades: List[Trade]) -> None:
        """从交易记录计算指标"""
        if not trades:
            return
        
        self.total_trades = len(trades)
        profits = [t.profit for t in trades]
        
        self.winning_trades = sum(1 for p in profits if p > 0)
        self.losing_trades = sum(1 for p in profits if p < 0)
        
        self.total_profit = sum(p for p in profits if p > 0)
        self.total_loss = abs(sum(p for p in profits if p < 0))
        
        self.win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        self.profit_factor = self.total_profit / self.total_loss if self.total_loss > 0 else 0
        
        self.avg_profit_per_trade = sum(profits) / self.total_trades if self.total_trades > 0 else 0
        self.avg_win = self.total_profit / self.winning_trades if self.winning_trades > 0 else 0
        self.avg_loss = self.total_loss / self.losing_trades if self.losing_trades > 0 else 0
        
        # 计算夏普比率
        if len(profits) > 1:
            returns = np.array(profits)
            self.sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            
            # 计算索提诺比率（只考虑下行波动）
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 0:
                downside_std = np.std(negative_returns)
                self.sortino_ratio = np.mean(returns) / downside_std if downside_std > 0 else 0
        
        # 计算最大回撤
        cumulative = np.cumsum(profits)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        self.max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        self.last_updated = datetime.now()
    
    def get_performance_score(self) -> float:
        """获取综合性能评分 (0-1)"""
        if self.total_trades < 5:
            return 0.5  # 交易次数不足，返回中性评分
        
        # 归一化各指标
        win_rate_score = self.win_rate
        profit_factor_score = min(self.profit_factor / 2.0, 1.0) if self.profit_factor > 0 else 0
        sharpe_score = min(max(self.sharpe_ratio, 0) / 2.0, 1.0)
        
        # 惩罚大回撤
        drawdown_penalty = max(0, 1 + self.max_drawdown / 1000)  # 假设1000为最大可接受回撤
        
        # 综合评分
        score = (
            win_rate_score * 0.3 +
            profit_factor_score * 0.3 +
            sharpe_score * 0.2 +
            (self.avg_profit_per_trade / 100) * 0.2  # 假设100为良好的平均利润
        ) * drawdown_penalty
        
        return min(max(score, 0), 1)


@dataclass
class WeightOptimizationConfig:
    """权重优化配置"""
    method: WeightOptimizationMethod = WeightOptimizationMethod.PERFORMANCE_BASED
    min_weight: float = 0.1
    max_weight: float = 2.0
    min_trades_required: int = 10  # 最少交易次数要求
    lookback_period_days: int = 30  # 回看期（天）
    update_frequency_hours: int = 24  # 更新频率（小时）
    rl_learning_rate: float = 0.001
    rl_discount_factor: float = 0.95
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        'win_rate': 0.25,
        'profit_factor': 0.25,
        'sharpe_ratio': 0.25,
        'avg_profit': 0.25
    })


class StrategyWeightOptimizer:
    """
    策略权重优化器
    基于历史表现和RL动态调整策略权重
    """
    
    def __init__(self, config: WeightOptimizationConfig = None):
        """
        初始化权重优化器
        
        Args:
            config: 优化配置
        """
        self.config = config or WeightOptimizationConfig()
        self.logger = logging.getLogger(__name__)
        
        # 策略指标存储
        self.strategy_metrics: Dict[str, StrategyMetrics] = {}
        
        # 当前权重
        self.current_weights: Dict[str, float] = {}
        
        # 权重历史
        self.weight_history: List[Dict[str, Any]] = []
        
        # RL状态
        self.rl_state: Dict[str, Any] = {}
        self.rl_model_path = "models/weight_optimizer_rl.pkl"
        
        # 上次更新时间
        self.last_update_time = datetime.now()
        
        self.logger.info(f"权重优化器初始化完成，方法: {self.config.method.value}")
    
    def update_strategy_metrics(
        self,
        strategy_name: str,
        trades: List[Trade]
    ) -> None:
        """
        更新策略指标
        
        Args:
            strategy_name: 策略名称
            trades: 交易记录列表
        """
        if strategy_name not in self.strategy_metrics:
            self.strategy_metrics[strategy_name] = StrategyMetrics(strategy_name=strategy_name)
        
        # 过滤回看期内的交易
        cutoff_date = datetime.now() - timedelta(days=self.config.lookback_period_days)
        recent_trades = [t for t in trades if t.open_time >= cutoff_date]
        
        # 计算指标
        self.strategy_metrics[strategy_name].calculate_metrics(recent_trades)
        
        self.logger.debug(
            f"更新策略{strategy_name}指标: "
            f"交易数={len(recent_trades)}, "
            f"胜率={self.strategy_metrics[strategy_name].win_rate:.2%}"
        )
    
    def calculate_weights(
        self,
        strategy_names: List[str],
        force_update: bool = False
    ) -> Dict[str, float]:
        """
        计算策略权重
        
        Args:
            strategy_names: 策略名称列表
            force_update: 是否强制更新
            
        Returns:
            策略权重字典
        """
        # 检查是否需要更新
        time_since_update = (datetime.now() - self.last_update_time).total_seconds() / 3600
        if not force_update and time_since_update < self.config.update_frequency_hours:
            return self.current_weights
        
        method = self.config.method
        
        if method == WeightOptimizationMethod.EQUAL_WEIGHT:
            weights = self._calculate_equal_weights(strategy_names)
        
        elif method == WeightOptimizationMethod.PERFORMANCE_BASED:
            weights = self._calculate_performance_based_weights(strategy_names)
        
        elif method == WeightOptimizationMethod.SHARPE_RATIO:
            weights = self._calculate_sharpe_based_weights(strategy_names)
        
        elif method == WeightOptimizationMethod.WIN_RATE:
            weights = self._calculate_win_rate_based_weights(strategy_names)
        
        elif method == WeightOptimizationMethod.PROFIT_FACTOR:
            weights = self._calculate_profit_factor_based_weights(strategy_names)
        
        elif method == WeightOptimizationMethod.RL_OPTIMIZED:
            weights = self._calculate_rl_optimized_weights(strategy_names)
        
        elif method == WeightOptimizationMethod.ENSEMBLE:
            weights = self._calculate_ensemble_weights(strategy_names)
        
        else:
            weights = self._calculate_equal_weights(strategy_names)
        
        # 应用权重限制
        weights = self._apply_weight_constraints(weights)
        
        # 更新当前权重
        self.current_weights = weights
        self.last_update_time = datetime.now()
        
        # 记录权重历史
        self._record_weight_history(weights)
        
        self.logger.info(f"权重更新完成: {weights}")
        
        return weights
    
    def _calculate_equal_weights(self, strategy_names: List[str]) -> Dict[str, float]:
        """计算等权重"""
        return {name: 1.0 for name in strategy_names}
    
    def _calculate_performance_based_weights(
        self,
        strategy_names: List[str]
    ) -> Dict[str, float]:
        """基于综合性能评分计算权重"""
        weights = {}
        
        for name in strategy_names:
            if name in self.strategy_metrics:
                metrics = self.strategy_metrics[name]
                
                # 检查最少交易次数
                if metrics.total_trades < self.config.min_trades_required:
                    weights[name] = 0.5  # 数据不足，给予中性权重
                else:
                    weights[name] = metrics.get_performance_score()
            else:
                weights[name] = 0.5  # 无数据，给予中性权重
        
        # 归一化权重（保持平均值为1.0）
        total = sum(weights.values())
        if total > 0 and len(strategy_names) > 0:
            avg_weight = total / len(strategy_names)
            weights = {k: v / avg_weight for k, v in weights.items()}
        
        return weights
    
    def _calculate_sharpe_based_weights(
        self,
        strategy_names: List[str]
    ) -> Dict[str, float]:
        """基于夏普比率计算权重"""
        weights = {}
        sharpe_ratios = []
        
        for name in strategy_names:
            if name in self.strategy_metrics:
                metrics = self.strategy_metrics[name]
                if metrics.total_trades >= self.config.min_trades_required:
                    sharpe = max(metrics.sharpe_ratio, 0)  # 负夏普比率设为0
                    weights[name] = sharpe
                    sharpe_ratios.append(sharpe)
                else:
                    weights[name] = 0.5
            else:
                weights[name] = 0.5
        
        # 归一化
        if sharpe_ratios and sum(sharpe_ratios) > 0:
            avg_sharpe = np.mean(sharpe_ratios)
            weights = {k: v / avg_sharpe if avg_sharpe > 0 else 1.0 for k, v in weights.items()}
        
        return weights
    
    def _calculate_win_rate_based_weights(
        self,
        strategy_names: List[str]
    ) -> Dict[str, float]:
        """基于胜率计算权重"""
        weights = {}
        
        for name in strategy_names:
            if name in self.strategy_metrics:
                metrics = self.strategy_metrics[name]
                if metrics.total_trades >= self.config.min_trades_required:
                    # 胜率转换为权重（0.5胜率对应1.0权重）
                    weights[name] = metrics.win_rate / 0.5
                else:
                    weights[name] = 1.0
            else:
                weights[name] = 1.0
        
        return weights
    
    def _calculate_profit_factor_based_weights(
        self,
        strategy_names: List[str]
    ) -> Dict[str, float]:
        """基于盈利因子计算权重"""
        weights = {}
        
        for name in strategy_names:
            if name in self.strategy_metrics:
                metrics = self.strategy_metrics[name]
                if metrics.total_trades >= self.config.min_trades_required:
                    # 盈利因子转换为权重（1.0对应0.5权重，2.0对应1.0权重）
                    weights[name] = min(metrics.profit_factor / 2.0, 2.0) if metrics.profit_factor > 0 else 0.1
                else:
                    weights[name] = 1.0
            else:
                weights[name] = 1.0
        
        return weights
    
    def _calculate_rl_optimized_weights(
        self,
        strategy_names: List[str]
    ) -> Dict[str, float]:
        """
        使用RL优化权重
        简化的Q-learning实现
        """
        # 如果没有足够的历史数据，使用性能基础权重
        if not self.weight_history or len(self.weight_history) < 10:
            return self._calculate_performance_based_weights(strategy_names)
        
        # 初始化RL状态
        if not self.rl_state:
            self.rl_state = {
                'q_table': {},
                'learning_rate': self.config.rl_learning_rate,
                'discount_factor': self.config.rl_discount_factor
            }
        
        # 获取当前状态
        current_state = self._get_rl_state(strategy_names)
        
        # 选择动作（权重调整）
        action = self._select_rl_action(current_state, strategy_names)
        
        # 应用动作获得新权重
        weights = self._apply_rl_action(action, strategy_names)
        
        return weights
    
    def _get_rl_state(self, strategy_names: List[str]) -> str:
        """获取RL状态表示"""
        state_features = []
        
        for name in strategy_names:
            if name in self.strategy_metrics:
                metrics = self.strategy_metrics[name]
                # 离散化性能指标
                win_rate_bin = int(metrics.win_rate * 10)
                profit_factor_bin = min(int(metrics.profit_factor), 5)
                state_features.append(f"{win_rate_bin}_{profit_factor_bin}")
            else:
                state_features.append("0_0")
        
        return "_".join(state_features)
    
    def _select_rl_action(self, state: str, strategy_names: List[str]) -> Dict[str, float]:
        """选择RL动作（epsilon-greedy）"""
        epsilon = 0.1  # 探索率
        
        if np.random.random() < epsilon:
            # 探索：随机调整
            return {name: np.random.uniform(0.5, 1.5) for name in strategy_names}
        else:
            # 利用：使用Q表
            q_table = self.rl_state.get('q_table', {})
            if state in q_table:
                return q_table[state]
            else:
                # 如果状态未见过，使用性能基础权重
                return self._calculate_performance_based_weights(strategy_names)
    
    def _apply_rl_action(
        self,
        action: Dict[str, float],
        strategy_names: List[str]
    ) -> Dict[str, float]:
        """应用RL动作"""
        return action
    
    def update_rl_model(
        self,
        previous_weights: Dict[str, float],
        reward: float,
        strategy_names: List[str]
    ) -> None:
        """
        更新RL模型
        
        Args:
            previous_weights: 之前的权重
            reward: 奖励值（例如，组合收益）
            strategy_names: 策略名称列表
        """
        if not self.rl_state:
            return
        
        # 获取之前的状态
        prev_state = self._get_rl_state(strategy_names)
        
        # 获取当前状态
        current_state = self._get_rl_state(strategy_names)
        
        # Q-learning更新
        q_table = self.rl_state.get('q_table', {})
        
        if prev_state not in q_table:
            q_table[prev_state] = previous_weights
        
        # 计算Q值更新
        learning_rate = self.rl_state['learning_rate']
        discount_factor = self.rl_state['discount_factor']
        
        # 简化的Q值更新：基于奖励调整权重
        for name in strategy_names:
            if name in previous_weights:
                old_weight = previous_weights[name]
                # 如果奖励为正，增加权重；否则减少
                adjustment = learning_rate * reward
                new_weight = old_weight * (1 + adjustment)
                q_table[prev_state][name] = new_weight
        
        self.rl_state['q_table'] = q_table
        
        # 保存模型
        self._save_rl_model()
    
    def _save_rl_model(self) -> None:
        """保存RL模型"""
        try:
            os.makedirs(os.path.dirname(self.rl_model_path), exist_ok=True)
            with open(self.rl_model_path, 'wb') as f:
                pickle.dump(self.rl_state, f)
            self.logger.debug("RL模型已保存")
        except Exception as e:
            self.logger.error(f"保存RL模型失败: {str(e)}")
    
    def _load_rl_model(self) -> bool:
        """加载RL模型"""
        try:
            if os.path.exists(self.rl_model_path):
                with open(self.rl_model_path, 'rb') as f:
                    self.rl_state = pickle.load(f)
                self.logger.info("RL模型已加载")
                return True
        except Exception as e:
            self.logger.error(f"加载RL模型失败: {str(e)}")
        return False
    
    def _calculate_ensemble_weights(
        self,
        strategy_names: List[str]
    ) -> Dict[str, float]:
        """集成多种方法计算权重"""
        ensemble_weights = self.config.ensemble_weights
        
        # 计算各方法的权重
        win_rate_weights = self._calculate_win_rate_based_weights(strategy_names)
        profit_factor_weights = self._calculate_profit_factor_based_weights(strategy_names)
        sharpe_weights = self._calculate_sharpe_based_weights(strategy_names)
        performance_weights = self._calculate_performance_based_weights(strategy_names)
        
        # 加权组合
        final_weights = {}
        for name in strategy_names:
            final_weights[name] = (
                win_rate_weights.get(name, 1.0) * ensemble_weights.get('win_rate', 0.25) +
                profit_factor_weights.get(name, 1.0) * ensemble_weights.get('profit_factor', 0.25) +
                sharpe_weights.get(name, 1.0) * ensemble_weights.get('sharpe_ratio', 0.25) +
                performance_weights.get(name, 1.0) * ensemble_weights.get('avg_profit', 0.25)
            )
        
        return final_weights
    
    def _apply_weight_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """应用权重约束"""
        constrained_weights = {}
        
        for name, weight in weights.items():
            # 限制在最小和最大权重之间
            constrained_weight = max(self.config.min_weight, min(weight, self.config.max_weight))
            constrained_weights[name] = constrained_weight
        
        return constrained_weights
    
    def _record_weight_history(self, weights: Dict[str, float]) -> None:
        """记录权重历史"""
        self.weight_history.append({
            'timestamp': datetime.now(),
            'weights': weights.copy(),
            'method': self.config.method.value
        })
        
        # 限制历史记录大小
        max_history = 1000
        if len(self.weight_history) > max_history:
            self.weight_history = self.weight_history[-max_history:]
    
    def get_weight_history(
        self,
        strategy_name: str = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        获取权重历史
        
        Args:
            strategy_name: 策略名称（可选）
            limit: 返回数量限制
            
        Returns:
            权重历史列表
        """
        history = self.weight_history[-limit:]
        
        if strategy_name:
            # 只返回特定策略的权重历史
            filtered_history = []
            for record in history:
                if strategy_name in record['weights']:
                    filtered_history.append({
                        'timestamp': record['timestamp'],
                        'weight': record['weights'][strategy_name],
                        'method': record['method']
                    })
            return filtered_history
        
        return history
    
    def get_strategy_metrics(self, strategy_name: str) -> Optional[StrategyMetrics]:
        """
        获取策略指标
        
        Args:
            strategy_name: 策略名称
            
        Returns:
            策略指标
        """
        return self.strategy_metrics.get(strategy_name)
    
    def get_all_metrics(self) -> Dict[str, StrategyMetrics]:
        """获取所有策略指标"""
        return self.strategy_metrics.copy()
