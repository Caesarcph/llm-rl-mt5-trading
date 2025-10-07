"""
风险控制系统模块
实现多层级止损机制（单笔、日、周、月）、熔断和暂停交易功能、风险预警和自动调整
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

from src.core.models import Account, Position, Trade
from src.core.exceptions import RiskException


logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """风险等级"""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class CircuitBreakerStatus(Enum):
    """熔断状态"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    COOLING_DOWN = "cooling_down"


@dataclass
class StopLossConfig:
    """止损配置"""
    # 单笔止损
    max_loss_per_trade: float = 0.02  # 2%
    
    # 日止损
    max_daily_loss: float = 0.05  # 5%
    daily_loss_action: str = "halt_24h"  # 暂停24小时
    
    # 周止损
    max_weekly_loss: float = 0.10  # 10%
    weekly_loss_action: str = "reduce_50"  # 减少50%仓位
    
    # 月止损
    max_monthly_loss: float = 0.15  # 15%
    monthly_loss_action: str = "stop_trading"  # 停止交易
    
    # 连续亏损
    max_consecutive_losses: int = 3
    consecutive_loss_halt_hours: int = 24



@dataclass
class CircuitBreakerConfig:
    """熔断配置"""
    threshold: float = 0.08  # 8%回撤触发熔断
    duration_hours: int = 4  # 熔断持续4小时
    cooldown_hours: int = 2  # 冷却期2小时
    max_triggers_per_day: int = 3  # 每日最多触发3次


@dataclass
class LossRecord:
    """亏损记录"""
    timestamp: datetime
    amount: float
    percentage: float
    trade_id: Optional[str] = None
    reason: str = ""


@dataclass
class RiskAlert:
    """风险告警"""
    timestamp: datetime
    level: RiskLevel
    category: str  # "single_trade", "daily", "weekly", "monthly", "consecutive"
    message: str
    current_value: float
    threshold: float
    action_taken: str = ""



class RiskControlSystem:
    """风险控制系统"""
    
    def __init__(self, 
                 initial_balance: float,
                 stop_loss_config: Optional[StopLossConfig] = None,
                 circuit_breaker_config: Optional[CircuitBreakerConfig] = None):
        """
        初始化风险控制系统
        
        Args:
            initial_balance: 初始余额
            stop_loss_config: 止损配置
            circuit_breaker_config: 熔断配置
        """
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        
        # 配置
        self.stop_loss_config = stop_loss_config or StopLossConfig()
        self.circuit_breaker_config = circuit_breaker_config or CircuitBreakerConfig()
        
        # 亏损跟踪
        self.daily_losses: List[LossRecord] = []
        self.weekly_losses: List[LossRecord] = []
        self.monthly_losses: List[LossRecord] = []
        self.consecutive_losses = 0
        self.last_loss_time: Optional[datetime] = None
        
        # 熔断状态
        self.circuit_breaker_status = CircuitBreakerStatus.INACTIVE
        self.circuit_breaker_end_time: Optional[datetime] = None
        self.circuit_breaker_triggers_today = 0
        self.last_circuit_breaker_date: Optional[datetime] = None
        
        # 交易暂停状态
        self.trading_halted = False
        self.halt_end_time: Optional[datetime] = None
        self.halt_reason: str = ""
        
        # 仓位缩减状态
        self.position_reduction_active = False
        self.position_reduction_percentage = 1.0  # 100% = 正常
        
        # 告警历史
        self.alerts: List[RiskAlert] = []
        
        logger.info(f"风险控制系统初始化: 初始余额={initial_balance}")

    
    def update_balance(self, new_balance: float) -> None:
        """更新余额"""
        self.current_balance = new_balance
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance
    
    def record_trade_result(self, trade: Trade) -> None:
        """记录交易结果"""
        if trade.profit < 0:
            loss_percentage = abs(trade.profit) / self.current_balance if self.current_balance > 0 else 0
            
            loss_record = LossRecord(
                timestamp=trade.close_time or datetime.now(),
                amount=abs(trade.profit),
                percentage=loss_percentage,
                trade_id=trade.trade_id,
                reason=f"交易亏损: {trade.symbol}"
            )
            
            # 添加到各时间段记录
            self.daily_losses.append(loss_record)
            self.weekly_losses.append(loss_record)
            self.monthly_losses.append(loss_record)
            
            # 更新连续亏损
            self.consecutive_losses += 1
            self.last_loss_time = loss_record.timestamp
            
            # 检查单笔止损
            self._check_single_trade_stop_loss(loss_record)
            
            logger.debug(f"记录亏损: {trade.trade_id}, 金额: {trade.profit}, 连续亏损: {self.consecutive_losses}")
        else:
            # 盈利交易重置连续亏损
            self.consecutive_losses = 0
    
    def check_risk_limits(self, account: Account) -> Tuple[bool, List[RiskAlert]]:
        """
        检查风险限制
        
        Returns:
            (是否可以交易, 告警列表)
        """
        alerts = []
        can_trade = True
        
        # 清理过期记录
        self._cleanup_old_records()
        
        # 检查熔断状态
        if self._is_circuit_breaker_active():
            can_trade = False
            alerts.append(self._create_alert(
                RiskLevel.EMERGENCY,
                "circuit_breaker",
                "系统熔断中",
                0, 0, "禁止交易"
            ))
        
        # 检查交易暂停
        if self._is_trading_halted():
            can_trade = False
            alerts.append(self._create_alert(
                RiskLevel.CRITICAL,
                "trading_halt",
                f"交易已暂停: {self.halt_reason}",
                0, 0, "禁止交易"
            ))
        
        # 检查日止损
        daily_alert = self._check_daily_stop_loss()
        if daily_alert:
            alerts.append(daily_alert)
            if daily_alert.level in [RiskLevel.CRITICAL, RiskLevel.EMERGENCY]:
                can_trade = False
        
        # 检查周止损
        weekly_alert = self._check_weekly_stop_loss()
        if weekly_alert:
            alerts.append(weekly_alert)
        
        # 检查月止损
        monthly_alert = self._check_monthly_stop_loss()
        if monthly_alert:
            alerts.append(monthly_alert)
            if monthly_alert.level == RiskLevel.EMERGENCY:
                can_trade = False
        
        # 检查连续亏损
        consecutive_alert = self._check_consecutive_losses()
        if consecutive_alert:
            alerts.append(consecutive_alert)
            if consecutive_alert.level == RiskLevel.CRITICAL:
                can_trade = False
        
        # 检查回撤熔断
        current_drawdown = self._calculate_current_drawdown()
        if abs(current_drawdown) >= self.circuit_breaker_config.threshold:
            self._trigger_circuit_breaker(current_drawdown)
            can_trade = False
            alerts.append(self._create_alert(
                RiskLevel.EMERGENCY,
                "circuit_breaker",
                f"触发熔断: 回撤{abs(current_drawdown):.2%}",
                abs(current_drawdown),
                self.circuit_breaker_config.threshold,
                "系统熔断"
            ))
        
        # 保存告警
        self.alerts.extend(alerts)
        
        return can_trade, alerts

    
    def _check_single_trade_stop_loss(self, loss_record: LossRecord) -> None:
        """检查单笔止损"""
        if loss_record.percentage > self.stop_loss_config.max_loss_per_trade:
            alert = self._create_alert(
                RiskLevel.WARNING,
                "single_trade",
                f"单笔亏损超限: {loss_record.percentage:.2%}",
                loss_record.percentage,
                self.stop_loss_config.max_loss_per_trade,
                "记录告警"
            )
            self.alerts.append(alert)
            logger.warning(f"单笔止损告警: {loss_record.trade_id}, 亏损: {loss_record.percentage:.2%}")
    
    def _check_daily_stop_loss(self) -> Optional[RiskAlert]:
        """检查日止损"""
        daily_loss = sum(loss.amount for loss in self.daily_losses)
        daily_loss_pct = daily_loss / self.initial_balance if self.initial_balance > 0 else 0
        
        if daily_loss_pct >= self.stop_loss_config.max_daily_loss:
            # 触发日止损
            if self.stop_loss_config.daily_loss_action == "halt_24h":
                self._halt_trading("日亏损超限", 24)
            
            return self._create_alert(
                RiskLevel.CRITICAL,
                "daily",
                f"日亏损超限: {daily_loss_pct:.2%}",
                daily_loss_pct,
                self.stop_loss_config.max_daily_loss,
                self.stop_loss_config.daily_loss_action
            )
        elif daily_loss_pct >= self.stop_loss_config.max_daily_loss * 0.8:
            # 接近日止损
            return self._create_alert(
                RiskLevel.WARNING,
                "daily",
                f"日亏损接近限制: {daily_loss_pct:.2%}",
                daily_loss_pct,
                self.stop_loss_config.max_daily_loss,
                "预警"
            )
        
        return None
    
    def _check_weekly_stop_loss(self) -> Optional[RiskAlert]:
        """检查周止损"""
        weekly_loss = sum(loss.amount for loss in self.weekly_losses)
        weekly_loss_pct = weekly_loss / self.initial_balance if self.initial_balance > 0 else 0
        
        if weekly_loss_pct >= self.stop_loss_config.max_weekly_loss:
            # 触发周止损
            if self.stop_loss_config.weekly_loss_action == "reduce_50":
                self._reduce_position_size(0.5)
            
            return self._create_alert(
                RiskLevel.CRITICAL,
                "weekly",
                f"周亏损超限: {weekly_loss_pct:.2%}",
                weekly_loss_pct,
                self.stop_loss_config.max_weekly_loss,
                self.stop_loss_config.weekly_loss_action
            )
        elif weekly_loss_pct >= self.stop_loss_config.max_weekly_loss * 0.8:
            return self._create_alert(
                RiskLevel.WARNING,
                "weekly",
                f"周亏损接近限制: {weekly_loss_pct:.2%}",
                weekly_loss_pct,
                self.stop_loss_config.max_weekly_loss,
                "预警"
            )
        
        return None

    
    def _check_monthly_stop_loss(self) -> Optional[RiskAlert]:
        """检查月止损"""
        monthly_loss = sum(loss.amount for loss in self.monthly_losses)
        monthly_loss_pct = monthly_loss / self.initial_balance if self.initial_balance > 0 else 0
        
        if monthly_loss_pct >= self.stop_loss_config.max_monthly_loss:
            # 触发月止损
            if self.stop_loss_config.monthly_loss_action == "stop_trading":
                self._halt_trading("月亏损超限", 720)  # 30天
            
            return self._create_alert(
                RiskLevel.EMERGENCY,
                "monthly",
                f"月亏损超限: {monthly_loss_pct:.2%}",
                monthly_loss_pct,
                self.stop_loss_config.max_monthly_loss,
                self.stop_loss_config.monthly_loss_action
            )
        elif monthly_loss_pct >= self.stop_loss_config.max_monthly_loss * 0.8:
            return self._create_alert(
                RiskLevel.CRITICAL,
                "monthly",
                f"月亏损接近限制: {monthly_loss_pct:.2%}",
                monthly_loss_pct,
                self.stop_loss_config.max_monthly_loss,
                "严重预警"
            )
        
        return None
    
    def _check_consecutive_losses(self) -> Optional[RiskAlert]:
        """检查连续亏损"""
        if self.consecutive_losses >= self.stop_loss_config.max_consecutive_losses:
            # 触发连续亏损限制
            self._halt_trading(
                f"连续亏损{self.consecutive_losses}次",
                self.stop_loss_config.consecutive_loss_halt_hours
            )
            
            return self._create_alert(
                RiskLevel.CRITICAL,
                "consecutive",
                f"连续亏损{self.consecutive_losses}次",
                self.consecutive_losses,
                self.stop_loss_config.max_consecutive_losses,
                f"暂停交易{self.stop_loss_config.consecutive_loss_halt_hours}小时"
            )
        
        return None
    
    def _halt_trading(self, reason: str, hours: int) -> None:
        """暂停交易"""
        self.trading_halted = True
        self.halt_reason = reason
        self.halt_end_time = datetime.now() + timedelta(hours=hours)
        logger.warning(f"交易已暂停: {reason}, 持续{hours}小时")
    
    def _reduce_position_size(self, reduction_factor: float) -> None:
        """缩减仓位"""
        self.position_reduction_active = True
        self.position_reduction_percentage = reduction_factor
        logger.warning(f"仓位已缩减至{reduction_factor * 100}%")
    
    def _trigger_circuit_breaker(self, drawdown: float) -> None:
        """触发熔断"""
        now = datetime.now()
        
        # 检查是否是新的一天
        if self.last_circuit_breaker_date is None or \
           self.last_circuit_breaker_date.date() != now.date():
            self.circuit_breaker_triggers_today = 0
            self.last_circuit_breaker_date = now
        
        # 检查今日触发次数
        if self.circuit_breaker_triggers_today >= self.circuit_breaker_config.max_triggers_per_day:
            logger.error("今日熔断触发次数已达上限")
            return
        
        self.circuit_breaker_status = CircuitBreakerStatus.ACTIVE
        self.circuit_breaker_end_time = now + timedelta(hours=self.circuit_breaker_config.duration_hours)
        self.circuit_breaker_triggers_today += 1
        
        logger.critical(f"系统熔断触发: 回撤{abs(drawdown):.2%}, 持续{self.circuit_breaker_config.duration_hours}小时")

    
    def _is_circuit_breaker_active(self) -> bool:
        """检查熔断是否激活"""
        if self.circuit_breaker_status == CircuitBreakerStatus.INACTIVE:
            return False
        
        now = datetime.now()
        
        if self.circuit_breaker_end_time and now > self.circuit_breaker_end_time:
            # 熔断结束，进入冷却期
            self.circuit_breaker_status = CircuitBreakerStatus.COOLING_DOWN
            self.circuit_breaker_end_time = now + timedelta(hours=self.circuit_breaker_config.cooldown_hours)
            logger.info("熔断结束，进入冷却期")
            return False
        
        return self.circuit_breaker_status == CircuitBreakerStatus.ACTIVE
    
    def _is_trading_halted(self) -> bool:
        """检查交易是否暂停"""
        if not self.trading_halted:
            return False
        
        if self.halt_end_time and datetime.now() > self.halt_end_time:
            self.trading_halted = False
            self.halt_end_time = None
            self.halt_reason = ""
            logger.info("交易暂停解除")
            return False
        
        return True
    
    def _calculate_current_drawdown(self) -> float:
        """计算当前回撤"""
        if self.peak_balance == 0:
            return 0.0
        return (self.current_balance - self.peak_balance) / self.peak_balance
    
    def _cleanup_old_records(self) -> None:
        """清理过期记录"""
        now = datetime.now()
        
        # 清理日记录（保留24小时）
        self.daily_losses = [
            loss for loss in self.daily_losses
            if (now - loss.timestamp).total_seconds() < 86400
        ]
        
        # 清理周记录（保留7天）
        self.weekly_losses = [
            loss for loss in self.weekly_losses
            if (now - loss.timestamp).days < 7
        ]
        
        # 清理月记录（保留30天）
        self.monthly_losses = [
            loss for loss in self.monthly_losses
            if (now - loss.timestamp).days < 30
        ]
        
        # 清理告警（保留最近1000条）
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]
    
    def _create_alert(self, level: RiskLevel, category: str, message: str,
                     current_value: float, threshold: float, action: str) -> RiskAlert:
        """创建告警"""
        return RiskAlert(
            timestamp=datetime.now(),
            level=level,
            category=category,
            message=message,
            current_value=current_value,
            threshold=threshold,
            action_taken=action
        )

    
    def get_position_size_multiplier(self) -> float:
        """获取仓位大小乘数"""
        if self.position_reduction_active:
            return self.position_reduction_percentage
        return 1.0
    
    def reset_position_reduction(self) -> None:
        """重置仓位缩减"""
        self.position_reduction_active = False
        self.position_reduction_percentage = 1.0
        logger.info("仓位缩减已重置")
    
    def force_resume_trading(self) -> None:
        """强制恢复交易（人工干预）"""
        self.trading_halted = False
        self.halt_end_time = None
        self.halt_reason = ""
        self.circuit_breaker_status = CircuitBreakerStatus.INACTIVE
        self.circuit_breaker_end_time = None
        logger.warning("交易已强制恢复（人工干预）")
    
    def get_risk_status(self) -> Dict[str, Any]:
        """获取风险状态"""
        now = datetime.now()
        
        # 计算各时间段亏损
        daily_loss = sum(loss.amount for loss in self.daily_losses)
        weekly_loss = sum(loss.amount for loss in self.weekly_losses)
        monthly_loss = sum(loss.amount for loss in self.monthly_losses)
        
        daily_loss_pct = daily_loss / self.initial_balance if self.initial_balance > 0 else 0
        weekly_loss_pct = weekly_loss / self.initial_balance if self.initial_balance > 0 else 0
        monthly_loss_pct = monthly_loss / self.initial_balance if self.initial_balance > 0 else 0
        
        current_drawdown = self._calculate_current_drawdown()
        
        return {
            'timestamp': now,
            'current_balance': self.current_balance,
            'peak_balance': self.peak_balance,
            'current_drawdown': current_drawdown,
            'daily_loss': {
                'amount': daily_loss,
                'percentage': daily_loss_pct,
                'limit': self.stop_loss_config.max_daily_loss,
                'utilization': daily_loss_pct / self.stop_loss_config.max_daily_loss if self.stop_loss_config.max_daily_loss > 0 else 0
            },
            'weekly_loss': {
                'amount': weekly_loss,
                'percentage': weekly_loss_pct,
                'limit': self.stop_loss_config.max_weekly_loss,
                'utilization': weekly_loss_pct / self.stop_loss_config.max_weekly_loss if self.stop_loss_config.max_weekly_loss > 0 else 0
            },
            'monthly_loss': {
                'amount': monthly_loss,
                'percentage': monthly_loss_pct,
                'limit': self.stop_loss_config.max_monthly_loss,
                'utilization': monthly_loss_pct / self.stop_loss_config.max_monthly_loss if self.stop_loss_config.max_monthly_loss > 0 else 0
            },
            'consecutive_losses': self.consecutive_losses,
            'circuit_breaker': {
                'status': self.circuit_breaker_status.value,
                'end_time': self.circuit_breaker_end_time,
                'triggers_today': self.circuit_breaker_triggers_today
            },
            'trading_halted': self.trading_halted,
            'halt_reason': self.halt_reason,
            'halt_end_time': self.halt_end_time,
            'position_reduction_active': self.position_reduction_active,
            'position_size_multiplier': self.get_position_size_multiplier(),
            'recent_alerts': len([a for a in self.alerts if (now - a.timestamp).total_seconds() < 3600])
        }
    
    def get_recent_alerts(self, hours: int = 24) -> List[RiskAlert]:
        """获取最近的告警"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alerts if alert.timestamp > cutoff_time]
    
    def save_state(self, filepath: str = "risk_control_state.json") -> bool:
        """保存状态"""
        try:
            state = {
                'initial_balance': self.initial_balance,
                'current_balance': self.current_balance,
                'peak_balance': self.peak_balance,
                'consecutive_losses': self.consecutive_losses,
                'last_loss_time': self.last_loss_time.isoformat() if self.last_loss_time else None,
                'circuit_breaker_status': self.circuit_breaker_status.value,
                'circuit_breaker_end_time': self.circuit_breaker_end_time.isoformat() if self.circuit_breaker_end_time else None,
                'circuit_breaker_triggers_today': self.circuit_breaker_triggers_today,
                'trading_halted': self.trading_halted,
                'halt_reason': self.halt_reason,
                'halt_end_time': self.halt_end_time.isoformat() if self.halt_end_time else None,
                'position_reduction_active': self.position_reduction_active,
                'position_reduction_percentage': self.position_reduction_percentage
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"风险控制状态已保存: {filepath}")
            return True
        except Exception as e:
            logger.error(f"保存状态失败: {str(e)}")
            return False
    
    def load_state(self, filepath: str = "risk_control_state.json") -> bool:
        """加载状态"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            self.initial_balance = state['initial_balance']
            self.current_balance = state['current_balance']
            self.peak_balance = state['peak_balance']
            self.consecutive_losses = state['consecutive_losses']
            self.last_loss_time = datetime.fromisoformat(state['last_loss_time']) if state['last_loss_time'] else None
            self.circuit_breaker_status = CircuitBreakerStatus(state['circuit_breaker_status'])
            self.circuit_breaker_end_time = datetime.fromisoformat(state['circuit_breaker_end_time']) if state['circuit_breaker_end_time'] else None
            self.circuit_breaker_triggers_today = state['circuit_breaker_triggers_today']
            self.trading_halted = state['trading_halted']
            self.halt_reason = state['halt_reason']
            self.halt_end_time = datetime.fromisoformat(state['halt_end_time']) if state['halt_end_time'] else None
            self.position_reduction_active = state['position_reduction_active']
            self.position_reduction_percentage = state['position_reduction_percentage']
            
            logger.info(f"风险控制状态已加载: {filepath}")
            return True
        except FileNotFoundError:
            logger.warning(f"状态文件不存在: {filepath}")
            return False
        except Exception as e:
            logger.error(f"加载状态失败: {str(e)}")
            return False
