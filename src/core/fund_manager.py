"""
资金管理器模块
实现分阶段资金投入、10%->30%->50%的渐进式资金管理和自动资金调整
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

from src.core.models import Account, Trade
from src.core.exceptions import RiskException


logger = logging.getLogger(__name__)


class FundStage(Enum):
    """资金阶段"""
    TESTING = "testing"  # 测试阶段 10%
    STABLE = "stable"    # 稳定阶段 30%
    FULL = "full"        # 完全阶段 50%
    RESERVED = "reserved"  # 储备 50%


@dataclass
class StageConfig:
    """阶段配置"""
    stage: FundStage
    allocation_percentage: float  # 资金分配百分比
    min_trades: int  # 最少交易次数
    min_win_rate: float  # 最低胜率
    min_profit_factor: float  # 最低盈利因子
    min_duration_days: int  # 最短持续天数
    max_drawdown_threshold: float  # 最大回撤阈值


@dataclass
class StagePerformance:
    """阶段表现"""
    stage: FundStage
    start_date: datetime
    end_date: Optional[datetime] = None
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_profit: float = 0.0
    total_loss: float = 0.0
    max_drawdown: float = 0.0
    duration_days: int = 0
    
    def get_win_rate(self) -> float:
        """获取胜率"""
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades
    
    def get_profit_factor(self) -> float:
        """获取盈利因子"""
        if abs(self.total_loss) < 0.01:
            return float('inf') if self.total_profit > 0 else 0.0
        return self.total_profit / abs(self.total_loss)
    
    def meets_requirements(self, config: StageConfig) -> Tuple[bool, List[str]]:
        """检查是否满足阶段要求"""
        reasons = []
        
        if self.total_trades < config.min_trades:
            reasons.append(f"交易次数不足: {self.total_trades}/{config.min_trades}")
        
        win_rate = self.get_win_rate()
        if win_rate < config.min_win_rate:
            reasons.append(f"胜率不足: {win_rate:.2%}/{config.min_win_rate:.2%}")
        
        profit_factor = self.get_profit_factor()
        if profit_factor < config.min_profit_factor:
            reasons.append(f"盈利因子不足: {profit_factor:.2f}/{config.min_profit_factor:.2f}")
        
        if self.duration_days < config.min_duration_days:
            reasons.append(f"运行时间不足: {self.duration_days}/{config.min_duration_days}天")
        
        if abs(self.max_drawdown) > config.max_drawdown_threshold:
            reasons.append(f"回撤过大: {abs(self.max_drawdown):.2%}/{config.max_drawdown_threshold:.2%}")
        
        return len(reasons) == 0, reasons



class FundManager:
    """资金管理器"""
    
    def __init__(self, total_capital: float, config_path: Optional[str] = None):
        self.total_capital = total_capital
        self.reserved_capital = total_capital * 0.5
        self.available_capital = total_capital * 0.5
        self.current_stage = FundStage.TESTING
        self.stage_start_date = datetime.now()
        self.stage_configs = self._initialize_stage_configs()
        self.current_performance = StagePerformance(
            stage=self.current_stage,
            start_date=self.stage_start_date
        )
        self.performance_history: List[StagePerformance] = []
        self.trade_history: List[Trade] = []
        self.adjustment_history: List[Dict[str, Any]] = []
        self.config_path = config_path
        logger.info(f"资金管理器初始化: 总资金={total_capital}, 当前阶段={self.current_stage.value}")
    
    def _initialize_stage_configs(self) -> Dict[FundStage, StageConfig]:
        return {
            FundStage.TESTING: StageConfig(
                stage=FundStage.TESTING, allocation_percentage=0.10,
                min_trades=20, min_win_rate=0.50, min_profit_factor=1.2,
                min_duration_days=14, max_drawdown_threshold=0.10
            ),
            FundStage.STABLE: StageConfig(
                stage=FundStage.STABLE, allocation_percentage=0.30,
                min_trades=50, min_win_rate=0.55, min_profit_factor=1.5,
                min_duration_days=30, max_drawdown_threshold=0.12
            ),
            FundStage.FULL: StageConfig(
                stage=FundStage.FULL, allocation_percentage=0.50,
                min_trades=100, min_win_rate=0.55, min_profit_factor=1.5,
                min_duration_days=60, max_drawdown_threshold=0.15
            )
        }
    
    def get_allocated_capital(self) -> float:
        config = self.stage_configs[self.current_stage]
        return self.available_capital * config.allocation_percentage
    
    def get_max_position_size(self, risk_per_trade: float = 0.02) -> float:
        return self.get_allocated_capital() * risk_per_trade

    
    def record_trade(self, trade: Trade) -> None:
        self.trade_history.append(trade)
        self.current_performance.total_trades += 1
        if trade.profit > 0:
            self.current_performance.winning_trades += 1
            self.current_performance.total_profit += trade.profit
        else:
            self.current_performance.losing_trades += 1
            self.current_performance.total_loss += abs(trade.profit)
        self.current_performance.duration_days = (datetime.now() - self.stage_start_date).days
        logger.debug(f"记录交易: {trade.trade_id}, 盈亏: {trade.profit}")
    
    def update_drawdown(self, current_drawdown: float) -> None:
        if abs(current_drawdown) > abs(self.current_performance.max_drawdown):
            self.current_performance.max_drawdown = current_drawdown
    
    def evaluate_stage_progression(self) -> Tuple[bool, Optional[FundStage], List[str]]:
        next_stage_map = {
            FundStage.TESTING: FundStage.STABLE,
            FundStage.STABLE: FundStage.FULL,
            FundStage.FULL: None
        }
        next_stage = next_stage_map.get(self.current_stage)
        if next_stage is None:
            return False, None, ["已经是最高阶段"]
        current_config = self.stage_configs[self.current_stage]
        meets_requirements, reasons = self.current_performance.meets_requirements(current_config)
        return (True, next_stage, []) if meets_requirements else (False, None, reasons)
    
    def progress_to_next_stage(self) -> bool:
        can_progress, next_stage, reasons = self.evaluate_stage_progression()
        if not can_progress:
            logger.info(f"无法晋级: {', '.join(reasons)}")
            return False
        self.current_performance.end_date = datetime.now()
        self.performance_history.append(self.current_performance)
        old_stage = self.current_stage
        self.current_stage = next_stage
        self.stage_start_date = datetime.now()
        self.current_performance = StagePerformance(stage=self.current_stage, start_date=self.stage_start_date)
        self._record_adjustment(
            f"阶段晋级: {old_stage.value} -> {next_stage.value}",
            old_allocation=self.stage_configs[old_stage].allocation_percentage,
            new_allocation=self.stage_configs[next_stage].allocation_percentage
        )
        logger.info(f"成功晋级: {old_stage.value} -> {next_stage.value}")
        return True

    
    def demote_stage(self, reason: str) -> bool:
        demote_map = {
            FundStage.FULL: FundStage.STABLE,
            FundStage.STABLE: FundStage.TESTING,
            FundStage.TESTING: None
        }
        target_stage = demote_map.get(self.current_stage)
        if target_stage is None:
            logger.warning("已经是最低阶段，无法降级")
            return False
        self.current_performance.end_date = datetime.now()
        self.performance_history.append(self.current_performance)
        old_stage = self.current_stage
        self.current_stage = target_stage
        self.stage_start_date = datetime.now()
        self.current_performance = StagePerformance(stage=self.current_stage, start_date=self.stage_start_date)
        self._record_adjustment(
            f"阶段降级: {old_stage.value} -> {target_stage.value}, 原因: {reason}",
            old_allocation=self.stage_configs[old_stage].allocation_percentage,
            new_allocation=self.stage_configs[target_stage].allocation_percentage
        )
        logger.warning(f"阶段降级: {old_stage.value} -> {target_stage.value}, 原因: {reason}")
        return True
    
    def auto_adjust_allocation(self, account: Account, current_drawdown: float) -> bool:
        adjusted = False
        self.update_drawdown(current_drawdown)
        current_config = self.stage_configs[self.current_stage]
        if abs(current_drawdown) > current_config.max_drawdown_threshold:
            if self.demote_stage(f"回撤过大: {abs(current_drawdown):.2%}"):
                adjusted = True
        elif self.current_stage != FundStage.FULL:
            can_progress, _, _ = self.evaluate_stage_progression()
            if can_progress:
                if self.progress_to_next_stage():
                    adjusted = True
        return adjusted
    
    def get_stage_status(self) -> Dict[str, Any]:
        config = self.stage_configs[self.current_stage]
        can_progress, next_stage, reasons = self.evaluate_stage_progression()
        return {
            'current_stage': self.current_stage.value,
            'allocated_capital': self.get_allocated_capital(),
            'allocation_percentage': config.allocation_percentage,
            'stage_start_date': self.stage_start_date,
            'duration_days': self.current_performance.duration_days,
            'total_trades': self.current_performance.total_trades,
            'win_rate': self.current_performance.get_win_rate(),
            'profit_factor': self.current_performance.get_profit_factor(),
            'max_drawdown': self.current_performance.max_drawdown,
            'can_progress': can_progress,
            'next_stage': next_stage.value if next_stage else None,
            'progress_blockers': reasons
        }

    
    def get_risk_assessment(self) -> Dict[str, Any]:
        allocated = self.get_allocated_capital()
        reserved = self.reserved_capital
        return {
            'total_capital': self.total_capital,
            'allocated_capital': allocated,
            'reserved_capital': reserved,
            'allocation_ratio': allocated / self.total_capital if self.total_capital > 0 else 0,
            'reserve_ratio': reserved / self.total_capital if self.total_capital > 0 else 0,
            'current_stage': self.current_stage.value,
            'stage_performance': {
                'win_rate': self.current_performance.get_win_rate(),
                'profit_factor': self.current_performance.get_profit_factor(),
                'max_drawdown': self.current_performance.max_drawdown
            }
        }
    
    def _record_adjustment(self, reason: str, old_allocation: float, new_allocation: float) -> None:
        adjustment = {
            'timestamp': datetime.now(),
            'reason': reason,
            'old_allocation': old_allocation,
            'new_allocation': new_allocation,
            'old_stage': self.current_stage.value,
            'allocated_capital': self.get_allocated_capital()
        }
        self.adjustment_history.append(adjustment)
        if len(self.adjustment_history) > 100:
            self.adjustment_history = self.adjustment_history[-100:]
    
    def save_state(self, filepath: Optional[str] = None) -> bool:
        try:
            filepath = filepath or self.config_path or "fund_manager_state.json"
            state = {
                'total_capital': self.total_capital,
                'current_stage': self.current_stage.value,
                'stage_start_date': self.stage_start_date.isoformat(),
                'current_performance': {
                    'stage': self.current_performance.stage.value,
                    'start_date': self.current_performance.start_date.isoformat(),
                    'total_trades': self.current_performance.total_trades,
                    'winning_trades': self.current_performance.winning_trades,
                    'losing_trades': self.current_performance.losing_trades,
                    'total_profit': self.current_performance.total_profit,
                    'total_loss': self.current_performance.total_loss,
                    'max_drawdown': self.current_performance.max_drawdown,
                    'duration_days': self.current_performance.duration_days
                },
                'adjustment_history': self.adjustment_history
            }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"资金管理器状态已保存: {filepath}")
            return True
        except Exception as e:
            logger.error(f"保存状态失败: {str(e)}")
            return False

    
    def load_state(self, filepath: Optional[str] = None) -> bool:
        try:
            filepath = filepath or self.config_path or "fund_manager_state.json"
            with open(filepath, 'r', encoding='utf-8') as f:
                state = json.load(f)
            self.total_capital = state['total_capital']
            self.current_stage = FundStage(state['current_stage'])
            self.stage_start_date = datetime.fromisoformat(state['stage_start_date'])
            perf = state['current_performance']
            self.current_performance = StagePerformance(
                stage=FundStage(perf['stage']),
                start_date=datetime.fromisoformat(perf['start_date']),
                total_trades=perf['total_trades'],
                winning_trades=perf['winning_trades'],
                losing_trades=perf['losing_trades'],
                total_profit=perf['total_profit'],
                total_loss=perf['total_loss'],
                max_drawdown=perf['max_drawdown'],
                duration_days=perf['duration_days']
            )
            self.adjustment_history = state.get('adjustment_history', [])
            logger.info(f"资金管理器状态已加载: {filepath}")
            return True
        except FileNotFoundError:
            logger.warning(f"状态文件不存在: {filepath}")
            return False
        except Exception as e:
            logger.error(f"加载状态失败: {str(e)}")
            return False
