"""
风险管理Agent
实现仓位管理、风险控制、VaR计算、最大回撤监控和动态止损止盈调整
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
    Account, Position, Trade, Signal, MarketData, RiskMetrics,
    PositionType, TradeType
)
from src.core.exceptions import RiskException, DataValidationError


class RiskLevel(Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskAction(Enum):
    """风险控制动作"""
    NONE = "none"
    REDUCE_POSITION = "reduce_position"
    CLOSE_POSITION = "close_position"
    HALT_TRADING = "halt_trading"
    EMERGENCY_STOP = "emergency_stop"


class AlertType(Enum):
    """告警类型"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class RiskConfig:
    """风险管理配置"""
    # 基础风险参数
    max_risk_per_trade: float = 0.02  # 单笔最大风险2%
    max_daily_drawdown: float = 0.05  # 日最大回撤5%
    max_weekly_drawdown: float = 0.10  # 周最大回撤10%
    max_monthly_drawdown: float = 0.15  # 月最大回撤15%
    
    # 仓位管理
    max_position_size: float = 0.10  # 单个品种最大仓位10%
    max_total_exposure: float = 0.50  # 总敞口50%
    correlation_threshold: float = 0.7  # 相关性阈值
    max_correlated_exposure: float = 0.40  # 相关品种最大敞口40%
    
    # VaR参数
    var_confidence_level: float = 0.95  # VaR置信水平95%
    var_lookback_days: int = 252  # VaR回望期252天
    max_var_limit: float = 0.03  # VaR限额3%
    
    # 连续亏损控制
    max_consecutive_losses: int = 3  # 最大连续亏损次数
    consecutive_loss_halt_hours: int = 24  # 连续亏损后暂停交易小时数
    
    # 动态调整参数
    trailing_stop_activation: float = 0.02  # 追踪止损激活阈值2%
    trailing_stop_distance: float = 0.01  # 追踪止损距离1%
    profit_target_adjustment: bool = True  # 是否动态调整止盈
    
    # 熔断机制
    circuit_breaker_threshold: float = 0.08  # 熔断阈值8%
    circuit_breaker_duration_hours: int = 4  # 熔断持续时间


@dataclass
class VaRResult:
    """VaR计算结果"""
    var_1d: float  # 1日VaR
    var_5d: float  # 5日VaR
    var_10d: float  # 10日VaR
    confidence_level: float
    method: str  # 计算方法
    timestamp: datetime
    portfolio_value: float
    
    def get_var_percentage(self, days: int = 1) -> float:
        """获取VaR百分比"""
        if days == 1:
            return abs(self.var_1d / self.portfolio_value) if self.portfolio_value > 0 else 0
        elif days == 5:
            return abs(self.var_5d / self.portfolio_value) if self.portfolio_value > 0 else 0
        elif days == 10:
            return abs(self.var_10d / self.portfolio_value) if self.portfolio_value > 0 else 0
        return 0


@dataclass
class DrawdownAnalysis:
    """回撤分析结果"""
    current_drawdown: float  # 当前回撤
    max_drawdown: float  # 最大回撤
    drawdown_duration: int  # 回撤持续天数
    recovery_factor: float  # 恢复因子
    underwater_curve: pd.Series  # 水下曲线
    peak_date: datetime  # 峰值日期
    trough_date: Optional[datetime] = None  # 谷底日期
    
    def is_in_drawdown(self) -> bool:
        """是否处于回撤状态"""
        return self.current_drawdown < -0.001  # 0.1%以上认为是回撤
    
    def get_drawdown_severity(self) -> RiskLevel:
        """获取回撤严重程度"""
        dd = abs(self.current_drawdown)
        if dd >= 0.15:
            return RiskLevel.CRITICAL
        elif dd >= 0.10:
            return RiskLevel.HIGH
        elif dd >= 0.05:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW


@dataclass
class ValidationResult:
    """交易验证结果"""
    is_valid: bool
    risk_score: float  # 0-100
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommended_size: Optional[float] = None
    
    def add_reason(self, reason: str):
        """添加拒绝原因"""
        self.reasons.append(reason)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """添加警告"""
        self.warnings.append(warning)


class VaRCalculator:
    """VaR计算器"""
    
    def __init__(self, confidence_level: float = 0.95, lookback_days: int = 252):
        self.confidence_level = confidence_level
        self.lookback_days = lookback_days
        self.logger = logging.getLogger(__name__)
    
    def calculate_historical_var(self, returns: pd.Series, portfolio_value: float) -> VaRResult:
        """计算历史模拟法VaR"""
        try:
            if len(returns) < 30:
                return self._get_default_var_result(portfolio_value)
            
            # 计算历史VaR
            var_1d_pct = np.percentile(returns, (1 - self.confidence_level) * 100)
            var_1d = portfolio_value * var_1d_pct
            var_5d = var_1d * np.sqrt(5)
            var_10d = var_1d * np.sqrt(10)
            
            return VaRResult(
                var_1d=var_1d,
                var_5d=var_5d,
                var_10d=var_10d,
                confidence_level=self.confidence_level,
                method="historical",
                timestamp=datetime.now(),
                portfolio_value=portfolio_value
            )
            
        except Exception as e:
            self.logger.error(f"历史模拟法VaR计算失败: {str(e)}")
            return self._get_default_var_result(portfolio_value)
    
    def _get_default_var_result(self, portfolio_value: float) -> VaRResult:
        """获取默认VaR结果"""
        # 使用保守估计：2%的日VaR
        var_1d = portfolio_value * -0.02
        return VaRResult(
            var_1d=var_1d,
            var_5d=var_1d * np.sqrt(5),
            var_10d=var_1d * np.sqrt(10),
            confidence_level=self.confidence_level,
            method="default",
            timestamp=datetime.now(),
            portfolio_value=portfolio_value
        )


class DrawdownMonitor:
    """回撤监控器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.equity_history: List[Tuple[datetime, float]] = []
    
    def update_equity(self, timestamp: datetime, equity: float) -> None:
        """更新权益数据"""
        self.equity_history.append((timestamp, equity))
        
        # 保持历史数据在合理范围内
        if len(self.equity_history) > 10000:
            self.equity_history = self.equity_history[-5000:]
    
    def calculate_drawdown(self) -> DrawdownAnalysis:
        """计算回撤分析"""
        try:
            if len(self.equity_history) < 2:
                return self._get_default_drawdown_analysis()
            
            # 转换为DataFrame
            df = pd.DataFrame(self.equity_history, columns=['timestamp', 'equity'])
            df.set_index('timestamp', inplace=True)
            
            # 计算累计最高点
            df['peak'] = df['equity'].expanding().max()
            
            # 计算回撤
            df['drawdown'] = (df['equity'] - df['peak']) / df['peak']
            
            # 当前回撤
            current_drawdown = df['drawdown'].iloc[-1]
            
            # 最大回撤
            max_drawdown = df['drawdown'].min()
            
            # 找到最大回撤的峰值和谷底
            max_dd_idx = df['drawdown'].idxmin()
            peak_idx = df.loc[:max_dd_idx, 'peak'].idxmax()
            
            # 回撤持续时间
            if current_drawdown < -0.001:  # 当前处于回撤
                current_peak_idx = df.loc[:df.index[-1], 'peak'].idxmax()
                drawdown_duration = (df.index[-1] - current_peak_idx).days
            else:
                drawdown_duration = 0
            
            # 恢复因子
            if max_drawdown != 0:
                total_return = (df['equity'].iloc[-1] - df['equity'].iloc[0]) / df['equity'].iloc[0]
                recovery_factor = total_return / abs(max_drawdown)
            else:
                recovery_factor = float('inf')
            
            return DrawdownAnalysis(
                current_drawdown=current_drawdown,
                max_drawdown=max_drawdown,
                drawdown_duration=drawdown_duration,
                recovery_factor=recovery_factor,
                underwater_curve=df['drawdown'],
                peak_date=peak_idx,
                trough_date=max_dd_idx if max_drawdown < -0.001 else None
            )
            
        except Exception as e:
            self.logger.error(f"回撤计算失败: {str(e)}")
            return self._get_default_drawdown_analysis()
    
    def _get_default_drawdown_analysis(self) -> DrawdownAnalysis:
        """获取默认回撤分析"""
        return DrawdownAnalysis(
            current_drawdown=0.0,
            max_drawdown=0.0,
            drawdown_duration=0,
            recovery_factor=1.0,
            underwater_curve=pd.Series([0.0]),
            peak_date=datetime.now()
        )


class RiskManagerAgent:
    """风险管理Agent"""
    
    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        self.var_calculator = VaRCalculator(
            confidence_level=self.config.var_confidence_level,
            lookback_days=self.config.var_lookback_days
        )
        self.drawdown_monitor = DrawdownMonitor()
        
        # 状态跟踪
        self.consecutive_losses = 0
        self.last_loss_time: Optional[datetime] = None
        self.circuit_breaker_active = False
        self.circuit_breaker_end_time: Optional[datetime] = None
        self.trading_halted = False
        self.halt_end_time: Optional[datetime] = None
        
        # 历史记录
        self.risk_alerts: List[Dict[str, Any]] = []
    
    def validate_trade(self, signal: Signal, account: Account, 
                      positions: List[Position], 
                      correlation_matrix: Optional[pd.DataFrame] = None) -> ValidationResult:
        """验证交易信号"""
        try:
            result = ValidationResult(is_valid=True, risk_score=0.0)
            
            # 检查熔断状态
            if self._is_circuit_breaker_active():
                result.add_reason("系统熔断中，禁止交易")
                return result
            
            # 检查交易暂停状态
            if self._is_trading_halted():
                result.add_reason("交易已暂停")
                return result
            
            # 检查账户状态
            if not self._validate_account_status(account, result):
                return result
            
            # 检查单笔风险
            if not self._validate_single_trade_risk(signal, account, result):
                return result
            
            # 检查仓位限制
            if not self._validate_position_limits(signal, positions, account, result):
                return result
            
            # 检查保证金充足性
            if not self._validate_margin_requirements(signal, account, result):
                return result
            
            # 计算推荐仓位大小
            result.recommended_size = self.calculate_position_size(signal, account, positions)
            
            # 计算风险评分
            result.risk_score = self._calculate_trade_risk_score(signal, account, positions)
            
            return result
            
        except Exception as e:
            self.logger.error(f"交易验证失败: {str(e)}")
            result = ValidationResult(is_valid=False, risk_score=100.0)
            result.add_reason(f"验证过程出错: {str(e)}")
            return result
    
    def calculate_position_size(self, signal: Signal, account: Account, 
                              positions: List[Position]) -> float:
        """计算仓位大小"""
        try:
            # 基于风险的仓位计算
            risk_amount = account.equity * self.config.max_risk_per_trade
            
            # 计算止损距离
            if signal.direction > 0:  # 买入
                stop_distance = abs(signal.entry_price - signal.sl)
            else:  # 卖出
                stop_distance = abs(signal.sl - signal.entry_price)
            
            if stop_distance <= 0:
                self.logger.warning("止损距离无效，使用默认仓位")
                return self.config.max_position_size * account.equity / signal.entry_price
            
            # 基础仓位大小
            base_position_size = risk_amount / stop_distance
            
            # 考虑现有仓位的影响
            existing_exposure = self._calculate_symbol_exposure(signal.symbol, positions)
            max_symbol_exposure = account.equity * self.config.max_position_size
            available_exposure = max_symbol_exposure - existing_exposure
            
            # 限制仓位大小
            position_size = min(base_position_size, available_exposure / signal.entry_price)
            
            # 考虑账户杠杆和保证金
            max_leverage_size = account.free_margin / (signal.entry_price * 0.01)  # 假设1%保证金
            position_size = min(position_size, max_leverage_size)
            
            # 确保仓位为正数
            return max(0.0, position_size)
            
        except Exception as e:
            self.logger.error(f"仓位计算失败: {str(e)}")
            return 0.0
    
    def record_trade_result(self, trade: Trade) -> None:
        """记录交易结果"""
        try:
            if trade.profit < 0:
                self.consecutive_losses += 1
                self.last_loss_time = trade.close_time or datetime.now()
                
                # 检查连续亏损限制
                if self.consecutive_losses >= self.config.max_consecutive_losses:
                    self._halt_trading_for_consecutive_losses()
                    self._create_alert(
                        "WARNING",
                        f"连续亏损 {self.consecutive_losses} 次，暂停交易"
                    )
            else:
                self.consecutive_losses = 0  # 重置连续亏损计数
                
        except Exception as e:
            self.logger.error(f"交易结果记录失败: {str(e)}")
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """获取风险摘要"""
        try:
            return {
                "timestamp": datetime.now(),
                "consecutive_losses": self.consecutive_losses,
                "circuit_breaker_active": self.circuit_breaker_active,
                "trading_halted": self.trading_halted,
                "recent_alerts": len(self.risk_alerts[-10:])
            }
            
        except Exception as e:
            self.logger.error(f"风险摘要生成失败: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    # 私有方法实现
    def _is_circuit_breaker_active(self) -> bool:
        """检查熔断是否激活"""
        if not self.circuit_breaker_active:
            return False
        
        if self.circuit_breaker_end_time and datetime.now() > self.circuit_breaker_end_time:
            self.circuit_breaker_active = False
            self.circuit_breaker_end_time = None
            self.logger.info("熔断解除")
            return False
        
        return True
    
    def _is_trading_halted(self) -> bool:
        """检查交易是否暂停"""
        if not self.trading_halted:
            return False
        
        if self.halt_end_time and datetime.now() > self.halt_end_time:
            self.trading_halted = False
            self.halt_end_time = None
            self.logger.info("交易暂停解除")
            return False
        
        return True
    
    def _validate_account_status(self, account: Account, result: ValidationResult) -> bool:
        """验证账户状态"""
        if account.margin_level < 100:
            result.add_reason("保证金水平不足")
            return False
        
        if account.free_margin <= 0:
            result.add_reason("可用保证金不足")
            return False
        
        return True
    
    def _validate_single_trade_risk(self, signal: Signal, account: Account, 
                                  result: ValidationResult) -> bool:
        """验证单笔交易风险"""
        # 计算风险金额
        if signal.direction > 0:  # 买入
            risk_distance = abs(signal.entry_price - signal.sl)
        else:  # 卖出
            risk_distance = abs(signal.sl - signal.entry_price)
        
        risk_amount = signal.size * risk_distance
        risk_pct = risk_amount / account.equity if account.equity > 0 else 1.0
        
        if risk_pct > self.config.max_risk_per_trade:
            result.add_reason(f"单笔风险过高: {risk_pct:.2%} > {self.config.max_risk_per_trade:.2%}")
            return False
        
        return True
    
    def _validate_position_limits(self, signal: Signal, positions: List[Position],
                                account: Account, result: ValidationResult) -> bool:
        """验证仓位限制"""
        # 计算品种现有敞口
        existing_exposure = self._calculate_symbol_exposure(signal.symbol, positions)
        new_exposure = signal.size * signal.entry_price
        total_symbol_exposure = existing_exposure + new_exposure
        
        max_symbol_exposure = account.equity * self.config.max_position_size
        
        if total_symbol_exposure > max_symbol_exposure:
            result.add_reason(f"品种敞口超限: {signal.symbol}")
            return False
        
        # 计算总敞口
        total_exposure = sum(abs(pos.volume * pos.current_price) for pos in positions)
        total_exposure += new_exposure
        
        max_total_exposure = account.equity * self.config.max_total_exposure
        
        if total_exposure > max_total_exposure:
            result.add_reason("总敞口超限")
            return False
        
        return True
    
    def _validate_margin_requirements(self, signal: Signal, account: Account,
                                    result: ValidationResult) -> bool:
        """验证保证金要求"""
        # 估算所需保证金（简化计算）
        required_margin = signal.size * signal.entry_price * 0.01  # 假设1%保证金要求
        
        if required_margin > account.free_margin:
            result.add_reason("保证金不足")
            return False
        
        return True
    
    def _calculate_trade_risk_score(self, signal: Signal, account: Account,
                                  positions: List[Position]) -> float:
        """计算交易风险评分"""
        try:
            score = 0.0
            
            # 基于风险百分比的评分
            risk_pct = self._calculate_trade_risk_percentage(signal, account)
            score += min(50.0, risk_pct * 2500)  # 2%风险 = 50分
            
            # 基于账户状态的评分
            if account.margin_level < 200:
                score += 30.0
            elif account.margin_level < 300:
                score += 15.0
            
            # 基于现有仓位的评分
            total_positions = len(positions)
            if total_positions > 10:
                score += 20.0
            elif total_positions > 5:
                score += 10.0
            
            return min(100.0, score)
            
        except Exception as e:
            self.logger.error(f"风险评分计算失败: {str(e)}")
            return 50.0
    
    def _calculate_trade_risk_percentage(self, signal: Signal, account: Account) -> float:
        """计算交易风险百分比"""
        try:
            if signal.direction > 0:  # 买入
                risk_distance = abs(signal.entry_price - signal.sl)
            else:  # 卖出
                risk_distance = abs(signal.sl - signal.entry_price)
            
            risk_amount = signal.size * risk_distance
            return risk_amount / account.equity if account.equity > 0 else 1.0
            
        except Exception:
            return 0.02  # 默认2%风险
    
    def _calculate_symbol_exposure(self, symbol: str, positions: List[Position]) -> float:
        """计算品种敞口"""
        return sum(abs(pos.volume * pos.current_price) 
                  for pos in positions if pos.symbol == symbol)
    
    def _halt_trading_for_consecutive_losses(self) -> None:
        """因连续亏损暂停交易"""
        self.trading_halted = True
        self.halt_end_time = datetime.now() + timedelta(
            hours=self.config.consecutive_loss_halt_hours
        )
        self.logger.warning(f"连续亏损暂停交易 {self.config.consecutive_loss_halt_hours} 小时")
    
    def _create_alert(self, alert_type: str, message: str) -> None:
        """创建风险告警"""
        alert = {
            "alert_type": alert_type,
            "message": message,
            "timestamp": datetime.now()
        }
        
        self.risk_alerts.append(alert)
        
        # 保持告警历史在合理范围内
        if len(self.risk_alerts) > 1000:
            self.risk_alerts = self.risk_alerts[-500:]
        
        # 记录日志
        log_level = {
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "CRITICAL": logging.CRITICAL,
            "EMERGENCY": logging.CRITICAL
        }.get(alert_type, logging.INFO)
        
        self.logger.log(log_level, f"风险告警: {message}")