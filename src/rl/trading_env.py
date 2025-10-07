"""
RL交易环境实现
基于gymnasium.Env的交易环境，用于强化学习训练
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import logging

from ..core.models import MarketData, Position, PositionType, Account


logger = logging.getLogger(__name__)


class ActionType(Enum):
    """动作类型"""
    HOLD = 0      # 持有
    BUY = 1       # 买入
    SELL = 2      # 卖出
    CLOSE = 3     # 平仓
    ADJUST = 4    # 调整仓位


@dataclass
class ObservationSpace:
    """观察空间配置"""
    price_window: int = 20          # 价格窗口大小
    indicator_count: int = 15       # 技术指标数量
    position_features: int = 5      # 持仓特征数量
    market_state_features: int = 10 # 市场状态特征数量
    
    def get_total_size(self) -> int:
        """获取观察空间总大小"""
        return (
            self.price_window * 4 +  # OHLC
            self.indicator_count +
            self.position_features +
            self.market_state_features
        )


class TradingEnvironment(gym.Env):
    """
    强化学习交易环境
    
    状态空间包括:
    - 价格数据 (OHLC)
    - 技术指标
    - 持仓信息
    - 市场状态
    
    动作空间:
    - 0: 持有
    - 1: 买入
    - 2: 卖出
    - 3: 平仓
    - 4: 调整仓位
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        symbol: str,
        data: pd.DataFrame,
        initial_balance: float = 10000.0,
        max_position_size: float = 1.0,
        transaction_cost: float = 0.0001,
        obs_config: Optional[ObservationSpace] = None,
        reward_scaling: float = 1.0
    ):
        """
        初始化交易环境
        
        Args:
            symbol: 交易品种
            data: 历史数据 (包含OHLC和指标)
            initial_balance: 初始资金
            max_position_size: 最大持仓手数
            transaction_cost: 交易成本 (点差+佣金)
            obs_config: 观察空间配置
            reward_scaling: 奖励缩放因子
        """
        super().__init__()
        
        self.symbol = symbol
        self.data = data
        self.initial_balance = initial_balance
        self.max_position_size = max_position_size
        self.transaction_cost = transaction_cost
        self.reward_scaling = reward_scaling
        
        # 观察空间配置
        self.obs_config = obs_config or ObservationSpace()
        obs_size = self.obs_config.get_total_size()
        
        # 定义观察空间和动作空间
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(5)  # 5种动作
        
        # 环境状态
        self.current_step = 0
        self.balance = initial_balance
        self.equity = initial_balance
        self.position: Optional[Position] = None
        self.trades_history: List[Dict] = []
        
        # 性能跟踪
        self.total_reward = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.max_equity = initial_balance
        self.max_drawdown = 0.0
        
        logger.info(f"初始化交易环境: {symbol}, 初始资金: {initial_balance}")
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        重置环境
        
        Returns:
            observation: 初始观察
            info: 额外信息
        """
        super().reset(seed=seed)
        
        # 重置状态
        self.current_step = self.obs_config.price_window
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position = None
        self.trades_history = []
        
        # 重置性能指标
        self.total_reward = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.max_equity = self.initial_balance
        self.max_drawdown = 0.0
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        执行一步动作
        
        Args:
            action: 动作索引 (0-4)
            
        Returns:
            observation: 新的观察
            reward: 奖励
            terminated: 是否终止
            truncated: 是否截断
            info: 额外信息
        """
        # 保存前一步的权益
        prev_equity = self.equity
        
        # 执行动作
        self._execute_action(action)
        
        # 更新当前价格和权益
        current_price = self._get_current_price()
        self._update_equity(current_price)
        
        # 计算奖励
        reward = self._calculate_reward(prev_equity, self.equity, action)
        self.total_reward += reward
        
        # 更新最大权益和回撤
        if self.equity > self.max_equity:
            self.max_equity = self.equity
        
        current_drawdown = (self.max_equity - self.equity) / self.max_equity
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
        
        # 移动到下一步
        self.current_step += 1
        
        # 检查是否终止
        terminated = self._is_terminated()
        truncated = self.current_step >= len(self.data) - 1
        
        # 获取新观察和信息
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _execute_action(self, action: int) -> None:
        """执行交易动作"""
        action_type = ActionType(action)
        current_price = self._get_current_price()
        
        if action_type == ActionType.BUY and self.position is None:
            # 买入
            position_size = self._calculate_position_size()
            cost = position_size * current_price * (1 + self.transaction_cost)
            
            if cost <= self.balance:
                self.position = Position(
                    position_id=f"pos_{self.current_step}",
                    symbol=self.symbol,
                    type=PositionType.LONG,
                    volume=position_size,
                    open_price=current_price,
                    current_price=current_price,
                    open_time=datetime.now()
                )
                self.balance -= cost
                self.total_trades += 1
                logger.debug(f"买入: 价格={current_price:.5f}, 手数={position_size:.2f}")
        
        elif action_type == ActionType.SELL and self.position is None:
            # 卖出
            position_size = self._calculate_position_size()
            cost = position_size * current_price * (1 + self.transaction_cost)
            
            if cost <= self.balance:
                self.position = Position(
                    position_id=f"pos_{self.current_step}",
                    symbol=self.symbol,
                    type=PositionType.SHORT,
                    volume=position_size,
                    open_price=current_price,
                    current_price=current_price,
                    open_time=datetime.now()
                )
                self.balance -= cost
                self.total_trades += 1
                logger.debug(f"卖出: 价格={current_price:.5f}, 手数={position_size:.2f}")
        
        elif action_type == ActionType.CLOSE and self.position is not None:
            # 平仓
            self._close_position(current_price)
        
        elif action_type == ActionType.ADJUST and self.position is not None:
            # 调整仓位 (简化实现: 减半仓位)
            self.position.volume *= 0.5
            logger.debug(f"调整仓位: 新手数={self.position.volume:.2f}")
    
    def _close_position(self, current_price: float) -> None:
        """平仓"""
        if self.position is None:
            return
        
        # 计算盈亏
        pnl = self.position.calculate_unrealized_pnl(current_price)
        pnl -= self.position.volume * current_price * self.transaction_cost
        
        # 更新余额
        self.balance += self.position.volume * self.position.open_price + pnl
        
        # 记录交易
        if pnl > 0:
            self.winning_trades += 1
        
        self.trades_history.append({
            'open_price': self.position.open_price,
            'close_price': current_price,
            'type': self.position.type.name,
            'volume': self.position.volume,
            'pnl': pnl,
            'step': self.current_step
        })
        
        logger.debug(f"平仓: 盈亏={pnl:.2f}, 价格={current_price:.5f}")
        self.position = None
    
    def _calculate_position_size(self) -> float:
        """计算仓位大小"""
        # 简化实现: 使用固定比例的资金
        risk_per_trade = 0.02  # 每笔交易风险2%
        position_size = (self.balance * risk_per_trade) / self._get_current_price()
        return min(position_size, self.max_position_size)
    
    def _update_equity(self, current_price: float) -> None:
        """更新权益"""
        self.equity = self.balance
        
        if self.position is not None:
            self.position.update_current_price(current_price)
            unrealized_pnl = self.position.calculate_unrealized_pnl(current_price)
            self.equity += self.position.volume * self.position.open_price + unrealized_pnl
    
    def _calculate_reward(self, prev_equity: float, current_equity: float, action: int) -> float:
        """
        计算奖励函数
        
        奖励设计:
        1. 收益奖励: 权益变化
        2. 风险惩罚: 回撤惩罚
        3. 交易成本惩罚
        """
        # 收益奖励
        return_reward = (current_equity - prev_equity) / prev_equity
        
        # 风险惩罚
        risk_penalty = 0.0
        if self.max_drawdown > 0.1:  # 回撤超过10%
            risk_penalty = self.max_drawdown * 0.5
        
        # 交易成本惩罚
        cost_penalty = 0.0
        if action in [ActionType.BUY.value, ActionType.SELL.value, ActionType.CLOSE.value]:
            cost_penalty = self.transaction_cost
        
        # 持仓时间奖励 (鼓励持有盈利仓位)
        holding_reward = 0.0
        if self.position is not None and self.position.profit > 0:
            holding_reward = 0.001
        
        # 综合奖励
        total_reward = (return_reward - risk_penalty - cost_penalty + holding_reward) * self.reward_scaling
        
        return total_reward
    
    def _get_observation(self) -> np.ndarray:
        """
        获取当前观察
        
        观察包括:
        1. 价格数据 (归一化的OHLC)
        2. 技术指标
        3. 持仓信息
        4. 市场状态
        """
        obs_list = []
        
        # 1. 价格数据
        start_idx = max(0, self.current_step - self.obs_config.price_window)
        end_idx = self.current_step
        
        price_data = self.data.iloc[start_idx:end_idx]
        
        # 归一化价格数据
        if len(price_data) > 0:
            for col in ['open', 'high', 'low', 'close']:
                if col in price_data.columns:
                    values = price_data[col].values
                    normalized = (values - values.mean()) / (values.std() + 1e-8)
                    # 填充到固定长度
                    if len(normalized) < self.obs_config.price_window:
                        normalized = np.pad(
                            normalized,
                            (self.obs_config.price_window - len(normalized), 0),
                            mode='edge'
                        )
                    obs_list.extend(normalized)
        else:
            obs_list.extend([0.0] * (self.obs_config.price_window * 4))
        
        # 2. 技术指标
        current_data = self.data.iloc[self.current_step]
        indicator_cols = [col for col in self.data.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        indicators = []
        for col in indicator_cols[:self.obs_config.indicator_count]:
            value = current_data.get(col, 0.0)
            indicators.append(float(value) if not pd.isna(value) else 0.0)
        
        # 填充到固定数量
        while len(indicators) < self.obs_config.indicator_count:
            indicators.append(0.0)
        
        obs_list.extend(indicators)
        
        # 3. 持仓信息
        if self.position is not None:
            current_price = self._get_current_price()
            position_features = [
                1.0 if self.position.type == PositionType.LONG else -1.0,  # 持仓方向
                self.position.volume / self.max_position_size,  # 归一化手数
                (current_price - self.position.open_price) / self.position.open_price,  # 收益率
                self.position.profit / self.initial_balance,  # 归一化盈亏
                (self.current_step - self.trades_history[-1]['step']) / 100.0 if self.trades_history else 0.0  # 持仓时间
            ]
        else:
            position_features = [0.0] * self.obs_config.position_features
        
        obs_list.extend(position_features)
        
        # 4. 市场状态特征
        market_features = [
            self.balance / self.initial_balance,  # 归一化余额
            self.equity / self.initial_balance,  # 归一化权益
            self.max_drawdown,  # 最大回撤
            self.winning_trades / max(self.total_trades, 1),  # 胜率
            len(self.trades_history) / 100.0,  # 交易次数
            self.total_reward,  # 累计奖励
            self.current_step / len(self.data),  # 进度
            0.0, 0.0, 0.0  # 预留特征
        ]
        
        obs_list.extend(market_features)
        
        return np.array(obs_list, dtype=np.float32)
    
    def _get_current_price(self) -> float:
        """获取当前价格"""
        if self.current_step >= len(self.data):
            return self.data.iloc[-1]['close']
        return self.data.iloc[self.current_step]['close']
    
    def _is_terminated(self) -> bool:
        """检查是否终止"""
        # 资金耗尽
        if self.equity <= self.initial_balance * 0.5:
            logger.warning(f"资金耗尽: 权益={self.equity:.2f}")
            return True
        
        # 回撤过大
        if self.max_drawdown > 0.3:
            logger.warning(f"回撤过大: {self.max_drawdown:.2%}")
            return True
        
        return False
    
    def _get_info(self) -> Dict[str, Any]:
        """获取额外信息"""
        return {
            'step': self.current_step,
            'balance': self.balance,
            'equity': self.equity,
            'position': self.position is not None,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.winning_trades / max(self.total_trades, 1),
            'max_drawdown': self.max_drawdown,
            'total_reward': self.total_reward
        }
    
    def render(self, mode: str = 'human') -> None:
        """渲染环境状态"""
        if mode == 'human':
            print(f"\n=== Step {self.current_step} ===")
            print(f"Balance: ${self.balance:.2f}")
            print(f"Equity: ${self.equity:.2f}")
            print(f"Position: {self.position.type.name if self.position else 'None'}")
            print(f"Total Trades: {self.total_trades}")
            print(f"Win Rate: {self.winning_trades / max(self.total_trades, 1):.2%}")
            print(f"Max Drawdown: {self.max_drawdown:.2%}")
            print(f"Total Reward: {self.total_reward:.4f}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        total_return = (self.equity - self.initial_balance) / self.initial_balance
        win_rate = self.winning_trades / max(self.total_trades, 1)
        
        # 计算夏普比率
        if len(self.trades_history) > 1:
            returns = [t['pnl'] / self.initial_balance for t in self.trades_history]
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        return {
            'total_return': total_return,
            'final_equity': self.equity,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_reward': self.total_reward
        }
