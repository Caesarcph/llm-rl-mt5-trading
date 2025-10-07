"""
报告生成器
实现每日、周、月交易报告生成，策略分析和性能对比报告，风险分析和合规检查报告
"""

import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import pandas as pd
import numpy as np

from src.core.models import Trade, Position, Account
from src.strategies.performance_tracker import (
    PerformanceTracker, PerformancePeriod, StrategyPerformanceMetrics
)
from src.agents.risk_manager import RiskManagerAgent, RiskConfig, VaRResult, DrawdownAnalysis


class ReportType(Enum):
    """报告类型"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    STRATEGY_ANALYSIS = "strategy_analysis"
    RISK_ANALYSIS = "risk_analysis"
    COMPLIANCE = "compliance"


class ReportFormat(Enum):
    """报告格式"""
    JSON = "json"
    HTML = "html"
    TEXT = "text"
    MARKDOWN = "markdown"


@dataclass
class ComplianceRule:
    """合规规则"""
    rule_id: str
    rule_name: str
    description: str
    threshold: float
    severity: str  # INFO, WARNING, CRITICAL
    
    def check(self, value: float) -> Tuple[bool, str]:
        """检查规则是否通过"""
        raise NotImplementedError


@dataclass
class ComplianceCheckResult:
    """合规检查结果"""
    rule_id: str
    rule_name: str
    passed: bool
    actual_value: float
    threshold: float
    severity: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TradingReport:
    """交易报告"""
    report_id: str
    report_type: ReportType
    period_start: datetime
    period_end: datetime
    generated_at: datetime
    
    # 账户信息
    account_summary: Dict[str, Any] = field(default_factory=dict)
    
    # 交易统计
    trading_summary: Dict[str, Any] = field(default_factory=dict)
    
    # 策略表现
    strategy_performance: List[StrategyPerformanceMetrics] = field(default_factory=list)
    
    # 风险指标
    risk_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # 合规检查
    compliance_results: List[ComplianceCheckResult] = field(default_factory=list)
    
    # 建议和警告
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'report_id': self.report_id,
            'report_type': self.report_type.value,
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'generated_at': self.generated_at.isoformat(),
            'account_summary': self.account_summary,
            'trading_summary': self.trading_summary,
            'strategy_performance': [s.to_dict() for s in self.strategy_performance],
            'risk_metrics': self.risk_metrics,
            'compliance_results': [asdict(c) for c in self.compliance_results],
            'recommendations': self.recommendations,
            'warnings': self.warnings
        }
    
    def to_json(self, indent: int = 2) -> str:
        """转换为JSON"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, default=str)


class ReportGenerator:
    """
    报告生成器
    生成各类交易报告、策略分析和风险报告
    """
    
    def __init__(
        self,
        performance_tracker: PerformanceTracker,
        risk_manager: Optional[RiskManagerAgent] = None,
        output_dir: str = "logs/reports"
    ):
        """
        初始化报告生成器
        
        Args:
            performance_tracker: 性能跟踪器
            risk_manager: 风险管理器
            output_dir: 报告输出目录
        """
        self.logger = logging.getLogger(__name__)
        self.performance_tracker = performance_tracker
        self.risk_manager = risk_manager or RiskManagerAgent()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 合规规则
        self.compliance_rules = self._initialize_compliance_rules()
        
        self.logger.info("报告生成器初始化完成")
    
    def _initialize_compliance_rules(self) -> List[ComplianceRule]:
        """初始化合规规则"""
        return [
            MaxRiskPerTradeRule("max_risk_per_trade", "单笔最大风险", 0.02),
            MaxDailyDrawdownRule("max_daily_drawdown", "日最大回撤", 0.05),
            MaxWeeklyDrawdownRule("max_weekly_drawdown", "周最大回撤", 0.10),
            MaxMonthlyDrawdownRule("max_monthly_drawdown", "月最大回撤", 0.15),
            MinWinRateRule("min_win_rate", "最低胜率", 0.40),
            MaxConsecutiveLossesRule("max_consecutive_losses", "最大连续亏损", 3),
        ]
    
    def generate_daily_report(
        self,
        date: Optional[datetime] = None,
        account: Optional[Account] = None
    ) -> TradingReport:
        """
        生成每日交易报告
        
        Args:
            date: 报告日期（默认为今天）
            account: 账户信息
            
        Returns:
            交易报告
        """
        if date is None:
            date = datetime.now()
        
        period_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        period_end = period_start + timedelta(days=1)
        
        self.logger.info(f"生成每日报告: {period_start.date()}")
        
        # 创建报告对象
        report = TradingReport(
            report_id=f"daily_{period_start.strftime('%Y%m%d')}",
            report_type=ReportType.DAILY,
            period_start=period_start,
            period_end=period_end,
            generated_at=datetime.now()
        )
        
        # 生成账户摘要
        if account:
            report.account_summary = self._generate_account_summary(account)
        
        # 生成交易统计
        report.trading_summary = self._generate_trading_summary(period_start, period_end)
        
        # 生成策略表现
        report.strategy_performance = self._generate_strategy_performance(
            PerformancePeriod.DAILY, period_start, period_end
        )
        
        # 生成风险指标
        report.risk_metrics = self._generate_risk_metrics(period_start, period_end)
        
        # 执行合规检查
        report.compliance_results = self._perform_compliance_checks(report)
        
        # 生成建议和警告
        report.recommendations, report.warnings = self._generate_recommendations(report)
        
        # 保存报告
        self._save_report(report, ReportFormat.JSON)
        
        self.logger.info(f"每日报告生成完成: {report.report_id}")
        return report

    
    def generate_weekly_report(
        self,
        week_start: Optional[datetime] = None,
        account: Optional[Account] = None
    ) -> TradingReport:
        """
        生成周交易报告
        
        Args:
            week_start: 周开始日期（默认为本周一）
            account: 账户信息
            
        Returns:
            交易报告
        """
        if week_start is None:
            today = datetime.now()
            week_start = today - timedelta(days=today.weekday())
        
        period_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        period_end = period_start + timedelta(days=7)
        
        self.logger.info(f"生成周报告: {period_start.date()} - {period_end.date()}")
        
        report = TradingReport(
            report_id=f"weekly_{period_start.strftime('%Y%m%d')}",
            report_type=ReportType.WEEKLY,
            period_start=period_start,
            period_end=period_end,
            generated_at=datetime.now()
        )
        
        if account:
            report.account_summary = self._generate_account_summary(account)
        
        report.trading_summary = self._generate_trading_summary(period_start, period_end)
        report.strategy_performance = self._generate_strategy_performance(
            PerformancePeriod.WEEKLY, period_start, period_end
        )
        report.risk_metrics = self._generate_risk_metrics(period_start, period_end)
        report.compliance_results = self._perform_compliance_checks(report)
        report.recommendations, report.warnings = self._generate_recommendations(report)
        
        self._save_report(report, ReportFormat.JSON)
        
        self.logger.info(f"周报告生成完成: {report.report_id}")
        return report
    
    def generate_monthly_report(
        self,
        month_start: Optional[datetime] = None,
        account: Optional[Account] = None
    ) -> TradingReport:
        """
        生成月交易报告
        
        Args:
            month_start: 月开始日期（默认为本月1日）
            account: 账户信息
            
        Returns:
            交易报告
        """
        if month_start is None:
            today = datetime.now()
            month_start = today.replace(day=1)
        
        period_start = month_start.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # 计算下个月的第一天
        if period_start.month == 12:
            period_end = period_start.replace(year=period_start.year + 1, month=1)
        else:
            period_end = period_start.replace(month=period_start.month + 1)
        
        self.logger.info(f"生成月报告: {period_start.strftime('%Y-%m')}")
        
        report = TradingReport(
            report_id=f"monthly_{period_start.strftime('%Y%m')}",
            report_type=ReportType.MONTHLY,
            period_start=period_start,
            period_end=period_end,
            generated_at=datetime.now()
        )
        
        if account:
            report.account_summary = self._generate_account_summary(account)
        
        report.trading_summary = self._generate_trading_summary(period_start, period_end)
        report.strategy_performance = self._generate_strategy_performance(
            PerformancePeriod.MONTHLY, period_start, period_end
        )
        report.risk_metrics = self._generate_risk_metrics(period_start, period_end)
        report.compliance_results = self._perform_compliance_checks(report)
        report.recommendations, report.warnings = self._generate_recommendations(report)
        
        self._save_report(report, ReportFormat.JSON)
        
        self.logger.info(f"月报告生成完成: {report.report_id}")
        return report
    
    def generate_strategy_analysis_report(
        self,
        strategy_names: Optional[List[str]] = None,
        period: PerformancePeriod = PerformancePeriod.MONTHLY
    ) -> TradingReport:
        """
        生成策略分析和性能对比报告
        
        Args:
            strategy_names: 策略名称列表
            period: 分析周期
            
        Returns:
            交易报告
        """
        end_date = datetime.now()
        start_date = self._get_period_start_date(period, end_date)
        
        self.logger.info(f"生成策略分析报告: {period.value}")
        
        report = TradingReport(
            report_id=f"strategy_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            report_type=ReportType.STRATEGY_ANALYSIS,
            period_start=start_date,
            period_end=end_date,
            generated_at=datetime.now()
        )
        
        # 获取策略列表
        if strategy_names is None:
            strategy_names = list(self.performance_tracker.trades_by_strategy.keys())
        
        # 生成策略表现
        report.strategy_performance = self._generate_strategy_performance(
            period, start_date, end_date, strategy_names
        )
        
        # 生成策略对比分析
        report.trading_summary = self._generate_strategy_comparison(
            strategy_names, start_date, end_date
        )
        
        # 生成策略排名
        report.trading_summary['rankings'] = self._generate_strategy_rankings(
            report.strategy_performance
        )
        
        # 生成策略建议
        report.recommendations = self._generate_strategy_recommendations(
            report.strategy_performance
        )
        
        self._save_report(report, ReportFormat.JSON)
        
        self.logger.info(f"策略分析报告生成完成: {report.report_id}")
        return report
    
    def generate_risk_analysis_report(
        self,
        account: Account,
        positions: List[Position],
        period: PerformancePeriod = PerformancePeriod.MONTHLY
    ) -> TradingReport:
        """
        生成风险分析报告
        
        Args:
            account: 账户信息
            positions: 持仓列表
            period: 分析周期
            
        Returns:
            交易报告
        """
        end_date = datetime.now()
        start_date = self._get_period_start_date(period, end_date)
        
        self.logger.info(f"生成风险分析报告: {period.value}")
        
        report = TradingReport(
            report_id=f"risk_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            report_type=ReportType.RISK_ANALYSIS,
            period_start=start_date,
            period_end=end_date,
            generated_at=datetime.now()
        )
        
        # 账户摘要
        report.account_summary = self._generate_account_summary(account)
        
        # 风险指标
        report.risk_metrics = self._generate_comprehensive_risk_metrics(
            account, positions, start_date, end_date
        )
        
        # 持仓分析
        report.trading_summary = self._generate_position_analysis(positions, account)
        
        # 风险建议
        report.recommendations = self._generate_risk_recommendations(
            report.risk_metrics, positions
        )
        
        # 风险警告
        report.warnings = self._generate_risk_warnings(report.risk_metrics)
        
        self._save_report(report, ReportFormat.JSON)
        
        self.logger.info(f"风险分析报告生成完成: {report.report_id}")
        return report
    
    def generate_compliance_report(
        self,
        account: Account,
        period: PerformancePeriod = PerformancePeriod.MONTHLY
    ) -> TradingReport:
        """
        生成合规检查报告
        
        Args:
            account: 账户信息
            period: 检查周期
            
        Returns:
            交易报告
        """
        end_date = datetime.now()
        start_date = self._get_period_start_date(period, end_date)
        
        self.logger.info(f"生成合规检查报告: {period.value}")
        
        report = TradingReport(
            report_id=f"compliance_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            report_type=ReportType.COMPLIANCE,
            period_start=start_date,
            period_end=end_date,
            generated_at=datetime.now()
        )
        
        # 账户摘要
        report.account_summary = self._generate_account_summary(account)
        
        # 交易统计
        report.trading_summary = self._generate_trading_summary(start_date, end_date)
        
        # 策略表现
        report.strategy_performance = self._generate_strategy_performance(
            period, start_date, end_date
        )
        
        # 风险指标
        report.risk_metrics = self._generate_risk_metrics(start_date, end_date)
        
        # 执行合规检查
        report.compliance_results = self._perform_compliance_checks(report)
        
        # 生成合规摘要
        report.trading_summary['compliance_summary'] = self._generate_compliance_summary(
            report.compliance_results
        )
        
        # 生成合规建议
        report.recommendations = self._generate_compliance_recommendations(
            report.compliance_results
        )
        
        self._save_report(report, ReportFormat.JSON)
        
        self.logger.info(f"合规检查报告生成完成: {report.report_id}")
        return report
    
    # 私有辅助方法
    
    def _generate_account_summary(self, account: Account) -> Dict[str, Any]:
        """生成账户摘要"""
        return {
            'account_id': account.account_id,
            'balance': account.balance,
            'equity': account.equity,
            'margin': account.margin,
            'free_margin': account.free_margin,
            'margin_level': account.margin_level,
            'currency': account.currency,
            'leverage': account.leverage,
            'profit_loss': account.equity - account.balance,
            'profit_loss_pct': ((account.equity - account.balance) / account.balance * 100) 
                              if account.balance > 0 else 0
        }
    
    def _generate_trading_summary(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """生成交易统计摘要"""
        summary = {
            'period_start': start_date.isoformat(),
            'period_end': end_date.isoformat(),
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'net_profit': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_profit_per_trade': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'total_commission': 0.0,
            'total_swap': 0.0
        }
        
        # 收集所有策略的交易
        all_trades = []
        for trades in self.performance_tracker.trades_by_strategy.values():
            filtered_trades = [
                t for t in trades 
                if start_date <= t.open_time <= end_date
            ]
            all_trades.extend(filtered_trades)
        
        if not all_trades:
            return summary
        
        # 计算统计
        summary['total_trades'] = len(all_trades)
        
        profits = [t.profit for t in all_trades]
        winning_trades = [t for t in all_trades if t.profit > 0]
        losing_trades = [t for t in all_trades if t.profit < 0]
        
        summary['winning_trades'] = len(winning_trades)
        summary['losing_trades'] = len(losing_trades)
        summary['total_profit'] = sum(t.profit for t in winning_trades)
        summary['total_loss'] = abs(sum(t.profit for t in losing_trades))
        summary['net_profit'] = sum(profits)
        summary['win_rate'] = len(winning_trades) / len(all_trades) if all_trades else 0
        summary['profit_factor'] = (summary['total_profit'] / summary['total_loss'] 
                                   if summary['total_loss'] > 0 else 0)
        summary['avg_profit_per_trade'] = summary['net_profit'] / len(all_trades)
        summary['largest_win'] = max(profits) if profits else 0
        summary['largest_loss'] = min(profits) if profits else 0
        summary['total_commission'] = sum(t.commission for t in all_trades)
        summary['total_swap'] = sum(t.swap for t in all_trades)
        
        return summary

    
    def _generate_strategy_performance(
        self,
        period: PerformancePeriod,
        start_date: datetime,
        end_date: datetime,
        strategy_names: Optional[List[str]] = None
    ) -> List[StrategyPerformanceMetrics]:
        """生成策略表现"""
        if strategy_names is None:
            strategy_names = list(self.performance_tracker.trades_by_strategy.keys())
        
        performance_list = []
        for name in strategy_names:
            metrics = self.performance_tracker.calculate_metrics(
                name, start_date, end_date, use_cache=False
            )
            performance_list.append(metrics)
        
        return performance_list
    
    def _generate_risk_metrics(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """生成风险指标"""
        metrics = {
            'max_drawdown': 0.0,
            'current_drawdown': 0.0,
            'drawdown_duration_days': 0,
            'recovery_factor': 0.0,
            'consecutive_losses': self.risk_manager.consecutive_losses,
            'circuit_breaker_active': self.risk_manager.circuit_breaker_active,
            'trading_halted': self.risk_manager.trading_halted
        }
        
        # 计算回撤
        drawdown_analysis = self.risk_manager.drawdown_monitor.calculate_drawdown()
        metrics['max_drawdown'] = drawdown_analysis.max_drawdown
        metrics['current_drawdown'] = drawdown_analysis.current_drawdown
        metrics['drawdown_duration_days'] = drawdown_analysis.drawdown_duration
        metrics['recovery_factor'] = drawdown_analysis.recovery_factor
        metrics['drawdown_severity'] = drawdown_analysis.get_drawdown_severity().value
        
        return metrics
    
    def _generate_comprehensive_risk_metrics(
        self,
        account: Account,
        positions: List[Position],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """生成综合风险指标"""
        metrics = self._generate_risk_metrics(start_date, end_date)
        
        # 添加持仓风险
        total_exposure = sum(abs(pos.volume * pos.current_price) for pos in positions)
        metrics['total_exposure'] = total_exposure
        metrics['exposure_pct'] = (total_exposure / account.equity * 100) if account.equity > 0 else 0
        metrics['position_count'] = len(positions)
        
        # 计算未实现盈亏
        unrealized_pnl = sum(pos.profit for pos in positions)
        metrics['unrealized_pnl'] = unrealized_pnl
        metrics['unrealized_pnl_pct'] = (unrealized_pnl / account.equity * 100) if account.equity > 0 else 0
        
        # 保证金使用率
        metrics['margin_usage_pct'] = (account.margin / account.equity * 100) if account.equity > 0 else 0
        metrics['margin_level'] = account.margin_level
        
        # VaR计算（如果有足够的历史数据）
        try:
            # 收集历史收益率
            all_trades = []
            for trades in self.performance_tracker.trades_by_strategy.values():
                all_trades.extend(trades)
            
            if len(all_trades) >= 30:
                returns = pd.Series([t.profit / account.equity for t in all_trades])
                var_result = self.risk_manager.var_calculator.calculate_historical_var(
                    returns, account.equity
                )
                metrics['var_1d'] = var_result.var_1d
                metrics['var_5d'] = var_result.var_5d
                metrics['var_10d'] = var_result.var_10d
                metrics['var_1d_pct'] = var_result.get_var_percentage(1) * 100
        except Exception as e:
            self.logger.warning(f"VaR计算失败: {str(e)}")
        
        return metrics
    
    def _generate_position_analysis(
        self,
        positions: List[Position],
        account: Account
    ) -> Dict[str, Any]:
        """生成持仓分析"""
        analysis = {
            'total_positions': len(positions),
            'long_positions': 0,
            'short_positions': 0,
            'positions_by_symbol': {},
            'largest_position': None,
            'smallest_position': None,
            'avg_position_size': 0.0
        }
        
        if not positions:
            return analysis
        
        # 统计多空仓位
        for pos in positions:
            if pos.type.value == 'LONG':
                analysis['long_positions'] += 1
            else:
                analysis['short_positions'] += 1
            
            # 按品种统计
            symbol = pos.symbol
            if symbol not in analysis['positions_by_symbol']:
                analysis['positions_by_symbol'][symbol] = {
                    'count': 0,
                    'total_volume': 0.0,
                    'total_exposure': 0.0,
                    'unrealized_pnl': 0.0
                }
            
            analysis['positions_by_symbol'][symbol]['count'] += 1
            analysis['positions_by_symbol'][symbol]['total_volume'] += pos.volume
            analysis['positions_by_symbol'][symbol]['total_exposure'] += abs(pos.volume * pos.current_price)
            analysis['positions_by_symbol'][symbol]['unrealized_pnl'] += pos.profit
        
        # 找出最大和最小持仓
        position_sizes = [abs(pos.volume * pos.current_price) for pos in positions]
        if position_sizes:
            max_idx = position_sizes.index(max(position_sizes))
            min_idx = position_sizes.index(min(position_sizes))
            
            analysis['largest_position'] = {
                'symbol': positions[max_idx].symbol,
                'size': position_sizes[max_idx],
                'pct_of_equity': (position_sizes[max_idx] / account.equity * 100) if account.equity > 0 else 0
            }
            
            analysis['smallest_position'] = {
                'symbol': positions[min_idx].symbol,
                'size': position_sizes[min_idx],
                'pct_of_equity': (position_sizes[min_idx] / account.equity * 100) if account.equity > 0 else 0
            }
            
            analysis['avg_position_size'] = np.mean(position_sizes)
        
        return analysis
    
    def _generate_strategy_comparison(
        self,
        strategy_names: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """生成策略对比"""
        comparison = {
            'strategies_compared': len(strategy_names),
            'comparison_data': []
        }
        
        for name in strategy_names:
            metrics = self.performance_tracker.calculate_metrics(
                name, start_date, end_date, use_cache=False
            )
            
            comparison['comparison_data'].append({
                'strategy_name': name,
                'total_trades': metrics.total_trades,
                'win_rate': metrics.win_rate,
                'net_profit': metrics.net_profit,
                'profit_factor': metrics.profit_factor,
                'sharpe_ratio': metrics.sharpe_ratio,
                'max_drawdown': metrics.max_drawdown,
                'avg_profit_per_trade': metrics.avg_profit_per_trade
            })
        
        return comparison
    
    def _generate_strategy_rankings(
        self,
        performance_list: List[StrategyPerformanceMetrics]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """生成策略排名"""
        rankings = {}
        
        # 按净利润排名
        rankings['by_net_profit'] = sorted(
            [{'strategy': m.strategy_name, 'value': m.net_profit} for m in performance_list],
            key=lambda x: x['value'],
            reverse=True
        )
        
        # 按胜率排名
        rankings['by_win_rate'] = sorted(
            [{'strategy': m.strategy_name, 'value': m.win_rate} for m in performance_list],
            key=lambda x: x['value'],
            reverse=True
        )
        
        # 按夏普比率排名
        rankings['by_sharpe_ratio'] = sorted(
            [{'strategy': m.strategy_name, 'value': m.sharpe_ratio} for m in performance_list],
            key=lambda x: x['value'],
            reverse=True
        )
        
        # 按盈利因子排名
        rankings['by_profit_factor'] = sorted(
            [{'strategy': m.strategy_name, 'value': m.profit_factor} for m in performance_list],
            key=lambda x: x['value'],
            reverse=True
        )
        
        return rankings
    
    def _perform_compliance_checks(self, report: TradingReport) -> List[ComplianceCheckResult]:
        """执行合规检查"""
        results = []
        
        for rule in self.compliance_rules:
            try:
                result = self._check_compliance_rule(rule, report)
                results.append(result)
            except Exception as e:
                self.logger.error(f"合规检查失败 {rule.rule_id}: {str(e)}")
        
        return results
    
    def _check_compliance_rule(
        self,
        rule: ComplianceRule,
        report: TradingReport
    ) -> ComplianceCheckResult:
        """检查单个合规规则"""
        # 根据规则类型获取实际值
        actual_value = self._get_rule_actual_value(rule, report)
        
        # 执行检查
        passed, message = rule.check(actual_value)
        
        return ComplianceCheckResult(
            rule_id=rule.rule_id,
            rule_name=rule.rule_name,
            passed=passed,
            actual_value=actual_value,
            threshold=rule.threshold,
            severity=rule.severity,
            message=message
        )
    
    def _get_rule_actual_value(self, rule: ComplianceRule, report: TradingReport) -> float:
        """获取规则的实际值"""
        rule_value_map = {
            'max_risk_per_trade': lambda: self._calculate_max_risk_per_trade(report),
            'max_daily_drawdown': lambda: abs(report.risk_metrics.get('current_drawdown', 0)),
            'max_weekly_drawdown': lambda: abs(report.risk_metrics.get('max_drawdown', 0)),
            'max_monthly_drawdown': lambda: abs(report.risk_metrics.get('max_drawdown', 0)),
            'min_win_rate': lambda: report.trading_summary.get('win_rate', 0),
            'max_consecutive_losses': lambda: report.risk_metrics.get('consecutive_losses', 0)
        }
        
        getter = rule_value_map.get(rule.rule_id)
        if getter:
            return getter()
        
        return 0.0
    
    def _calculate_max_risk_per_trade(self, report: TradingReport) -> float:
        """计算单笔最大风险"""
        # 从策略表现中找出最大单笔亏损
        max_loss = 0.0
        for strategy in report.strategy_performance:
            if strategy.largest_loss < max_loss:
                max_loss = strategy.largest_loss
        
        # 转换为百分比（假设账户权益）
        if report.account_summary and report.account_summary.get('equity', 0) > 0:
            return abs(max_loss / report.account_summary['equity'])
        
        return 0.0
    
    def _generate_compliance_summary(
        self,
        compliance_results: List[ComplianceCheckResult]
    ) -> Dict[str, Any]:
        """生成合规摘要"""
        total_checks = len(compliance_results)
        passed_checks = sum(1 for r in compliance_results if r.passed)
        failed_checks = total_checks - passed_checks
        
        critical_failures = [r for r in compliance_results if not r.passed and r.severity == 'CRITICAL']
        warning_failures = [r for r in compliance_results if not r.passed and r.severity == 'WARNING']
        
        return {
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'failed_checks': failed_checks,
            'compliance_rate': (passed_checks / total_checks * 100) if total_checks > 0 else 0,
            'critical_failures': len(critical_failures),
            'warning_failures': len(warning_failures),
            'critical_issues': [r.message for r in critical_failures],
            'warnings': [r.message for r in warning_failures]
        }

    
    def _generate_recommendations(self, report: TradingReport) -> Tuple[List[str], List[str]]:
        """生成建议和警告"""
        recommendations = []
        warnings = []
        
        # 基于交易统计的建议
        if report.trading_summary:
            win_rate = report.trading_summary.get('win_rate', 0)
            profit_factor = report.trading_summary.get('profit_factor', 0)
            
            if win_rate < 0.40:
                warnings.append(f"胜率较低 ({win_rate:.1%})，建议优化策略或调整参数")
            
            if profit_factor < 1.0:
                warnings.append(f"盈利因子小于1 ({profit_factor:.2f})，系统处于亏损状态")
            elif profit_factor < 1.5:
                recommendations.append(f"盈利因子偏低 ({profit_factor:.2f})，建议提高盈亏比")
            
            if report.trading_summary.get('total_trades', 0) < 10:
                recommendations.append("交易次数较少，样本量不足以评估策略有效性")
        
        # 基于风险指标的警告
        if report.risk_metrics:
            current_dd = abs(report.risk_metrics.get('current_drawdown', 0))
            max_dd = abs(report.risk_metrics.get('max_drawdown', 0))
            
            if current_dd > 0.10:
                warnings.append(f"当前回撤较大 ({current_dd:.1%})，建议降低仓位或暂停交易")
            elif current_dd > 0.05:
                recommendations.append(f"当前回撤 {current_dd:.1%}，注意风险控制")
            
            if max_dd > 0.20:
                warnings.append(f"最大回撤过大 ({max_dd:.1%})，系统风险较高")
            
            if report.risk_metrics.get('consecutive_losses', 0) >= 3:
                warnings.append("连续亏损次数达到限制，建议检查策略有效性")
            
            if report.risk_metrics.get('circuit_breaker_active', False):
                warnings.append("系统熔断已激活，交易已暂停")
        
        # 基于策略表现的建议
        if report.strategy_performance:
            losing_strategies = [
                s for s in report.strategy_performance 
                if s.net_profit < 0
            ]
            
            if losing_strategies:
                strategy_names = ', '.join([s.strategy_name for s in losing_strategies])
                recommendations.append(f"以下策略表现不佳，建议优化或停用: {strategy_names}")
            
            # 找出表现最好的策略
            if len(report.strategy_performance) > 1:
                best_strategy = max(report.strategy_performance, key=lambda s: s.net_profit)
                recommendations.append(
                    f"策略 {best_strategy.strategy_name} 表现最佳，"
                    f"净利润 {best_strategy.net_profit:.2f}，可考虑增加权重"
                )
        
        # 基于合规检查的警告
        if report.compliance_results:
            failed_checks = [r for r in report.compliance_results if not r.passed]
            for check in failed_checks:
                if check.severity == 'CRITICAL':
                    warnings.append(f"严重合规问题: {check.message}")
                else:
                    recommendations.append(f"合规建议: {check.message}")
        
        return recommendations, warnings
    
    def _generate_strategy_recommendations(
        self,
        performance_list: List[StrategyPerformanceMetrics]
    ) -> List[str]:
        """生成策略建议"""
        recommendations = []
        
        if not performance_list:
            return recommendations
        
        # 按净利润排序
        sorted_strategies = sorted(performance_list, key=lambda s: s.net_profit, reverse=True)
        
        # 推荐表现最好的策略
        best = sorted_strategies[0]
        if best.net_profit > 0:
            recommendations.append(
                f"最佳策略: {best.strategy_name}, "
                f"净利润 {best.net_profit:.2f}, "
                f"胜率 {best.win_rate:.1%}, "
                f"盈利因子 {best.profit_factor:.2f}"
            )
        
        # 警告表现最差的策略
        worst = sorted_strategies[-1]
        if worst.net_profit < 0:
            recommendations.append(
                f"最差策略: {worst.strategy_name}, "
                f"净利润 {worst.net_profit:.2f}, "
                f"建议停用或优化"
            )
        
        # 识别高风险策略
        high_risk_strategies = [
            s for s in performance_list 
            if abs(s.max_drawdown) > 0.15
        ]
        if high_risk_strategies:
            names = ', '.join([s.strategy_name for s in high_risk_strategies])
            recommendations.append(f"高风险策略 (回撤>15%): {names}")
        
        # 识别低效策略
        low_efficiency_strategies = [
            s for s in performance_list 
            if s.profit_factor < 1.2 and s.total_trades > 10
        ]
        if low_efficiency_strategies:
            names = ', '.join([s.strategy_name for s in low_efficiency_strategies])
            recommendations.append(f"低效策略 (盈利因子<1.2): {names}")
        
        return recommendations
    
    def _generate_risk_recommendations(
        self,
        risk_metrics: Dict[str, Any],
        positions: List[Position]
    ) -> List[str]:
        """生成风险建议"""
        recommendations = []
        
        # 敞口建议
        exposure_pct = risk_metrics.get('exposure_pct', 0)
        if exposure_pct > 50:
            recommendations.append(f"总敞口过高 ({exposure_pct:.1f}%)，建议降低仓位")
        elif exposure_pct > 40:
            recommendations.append(f"总敞口较高 ({exposure_pct:.1f}%)，注意风险控制")
        
        # 保证金建议
        margin_level = risk_metrics.get('margin_level', 0)
        if margin_level < 200:
            recommendations.append(f"保证金水平较低 ({margin_level:.0f}%)，建议增加资金或减少仓位")
        
        # 持仓数量建议
        position_count = risk_metrics.get('position_count', 0)
        if position_count > 10:
            recommendations.append(f"持仓数量较多 ({position_count})，建议精简持仓")
        
        # VaR建议
        var_1d_pct = risk_metrics.get('var_1d_pct', 0)
        if var_1d_pct > 3:
            recommendations.append(f"1日VaR较高 ({var_1d_pct:.2f}%)，建议降低风险敞口")
        
        # 回撤建议
        current_dd = abs(risk_metrics.get('current_drawdown', 0))
        if current_dd > 0.05:
            recommendations.append(f"当前回撤 {current_dd:.1%}，建议谨慎交易")
        
        return recommendations
    
    def _generate_risk_warnings(self, risk_metrics: Dict[str, Any]) -> List[str]:
        """生成风险警告"""
        warnings = []
        
        # 严重回撤警告
        current_dd = abs(risk_metrics.get('current_drawdown', 0))
        if current_dd > 0.15:
            warnings.append(f"严重回撤警告: 当前回撤 {current_dd:.1%}")
        
        # 保证金警告
        margin_level = risk_metrics.get('margin_level', 0)
        if margin_level < 150:
            warnings.append(f"保证金水平危险: {margin_level:.0f}%")
        
        # 熔断警告
        if risk_metrics.get('circuit_breaker_active', False):
            warnings.append("系统熔断已激活")
        
        if risk_metrics.get('trading_halted', False):
            warnings.append("交易已暂停")
        
        # 连续亏损警告
        consecutive_losses = risk_metrics.get('consecutive_losses', 0)
        if consecutive_losses >= 3:
            warnings.append(f"连续亏损 {consecutive_losses} 次")
        
        return warnings
    
    def _generate_compliance_recommendations(
        self,
        compliance_results: List[ComplianceCheckResult]
    ) -> List[str]:
        """生成合规建议"""
        recommendations = []
        
        failed_checks = [r for r in compliance_results if not r.passed]
        
        for check in failed_checks:
            if check.severity == 'CRITICAL':
                recommendations.append(
                    f"[严重] {check.rule_name}: {check.message} "
                    f"(实际: {check.actual_value:.2%}, 限制: {check.threshold:.2%})"
                )
            elif check.severity == 'WARNING':
                recommendations.append(
                    f"[警告] {check.rule_name}: {check.message} "
                    f"(实际: {check.actual_value:.2%}, 限制: {check.threshold:.2%})"
                )
            else:
                recommendations.append(
                    f"[提示] {check.rule_name}: {check.message}"
                )
        
        # 如果全部通过
        if not failed_checks:
            recommendations.append("所有合规检查均已通过")
        
        return recommendations
    
    def _get_period_start_date(self, period: PerformancePeriod, end_date: datetime) -> datetime:
        """获取周期开始日期"""
        if period == PerformancePeriod.DAILY:
            return end_date - timedelta(days=1)
        elif period == PerformancePeriod.WEEKLY:
            return end_date - timedelta(weeks=1)
        elif period == PerformancePeriod.MONTHLY:
            return end_date - timedelta(days=30)
        elif period == PerformancePeriod.QUARTERLY:
            return end_date - timedelta(days=90)
        elif period == PerformancePeriod.YEARLY:
            return end_date - timedelta(days=365)
        else:  # ALL_TIME
            return datetime(2000, 1, 1)
    
    def _save_report(self, report: TradingReport, format: ReportFormat = ReportFormat.JSON) -> bool:
        """保存报告到文件"""
        try:
            filename = f"{report.report_id}.{format.value}"
            filepath = self.output_dir / filename
            
            if format == ReportFormat.JSON:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(report.to_json())
            elif format == ReportFormat.MARKDOWN:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(self._format_report_as_markdown(report))
            elif format == ReportFormat.TEXT:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(self._format_report_as_text(report))
            
            self.logger.info(f"报告已保存: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存报告失败: {str(e)}")
            return False
    
    def _format_report_as_markdown(self, report: TradingReport) -> str:
        """将报告格式化为Markdown"""
        lines = []
        lines.append(f"# {report.report_type.value.upper()} 交易报告")
        lines.append(f"\n**报告ID**: {report.report_id}")
        lines.append(f"**生成时间**: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**统计周期**: {report.period_start.date()} 至 {report.period_end.date()}")
        
        # 账户摘要
        if report.account_summary:
            lines.append("\n## 账户摘要")
            lines.append(f"- 账户余额: {report.account_summary.get('balance', 0):.2f}")
            lines.append(f"- 账户净值: {report.account_summary.get('equity', 0):.2f}")
            lines.append(f"- 盈亏: {report.account_summary.get('profit_loss', 0):.2f} "
                        f"({report.account_summary.get('profit_loss_pct', 0):.2f}%)")
            lines.append(f"- 保证金水平: {report.account_summary.get('margin_level', 0):.2f}%")
        
        # 交易统计
        if report.trading_summary:
            lines.append("\n## 交易统计")
            lines.append(f"- 总交易次数: {report.trading_summary.get('total_trades', 0)}")
            lines.append(f"- 盈利交易: {report.trading_summary.get('winning_trades', 0)}")
            lines.append(f"- 亏损交易: {report.trading_summary.get('losing_trades', 0)}")
            lines.append(f"- 胜率: {report.trading_summary.get('win_rate', 0):.2%}")
            lines.append(f"- 净利润: {report.trading_summary.get('net_profit', 0):.2f}")
            lines.append(f"- 盈利因子: {report.trading_summary.get('profit_factor', 0):.2f}")
        
        # 风险指标
        if report.risk_metrics:
            lines.append("\n## 风险指标")
            lines.append(f"- 最大回撤: {report.risk_metrics.get('max_drawdown', 0):.2%}")
            lines.append(f"- 当前回撤: {report.risk_metrics.get('current_drawdown', 0):.2%}")
            lines.append(f"- 连续亏损: {report.risk_metrics.get('consecutive_losses', 0)}")
        
        # 建议
        if report.recommendations:
            lines.append("\n## 建议")
            for rec in report.recommendations:
                lines.append(f"- {rec}")
        
        # 警告
        if report.warnings:
            lines.append("\n## 警告")
            for warn in report.warnings:
                lines.append(f"- ⚠️ {warn}")
        
        return '\n'.join(lines)
    
    def _format_report_as_text(self, report: TradingReport) -> str:
        """将报告格式化为纯文本"""
        return self._format_report_as_markdown(report).replace('#', '').replace('**', '')


# 合规规则实现类

class MaxRiskPerTradeRule(ComplianceRule):
    """单笔最大风险规则"""
    
    def __init__(self, rule_id: str, rule_name: str, threshold: float):
        super().__init__(rule_id, rule_name, "单笔交易风险不得超过账户的指定百分比", threshold, "CRITICAL")
    
    def check(self, value: float) -> Tuple[bool, str]:
        if value <= self.threshold:
            return True, f"单笔风险 {value:.2%} 符合要求"
        return False, f"单笔风险 {value:.2%} 超过限制 {self.threshold:.2%}"


class MaxDailyDrawdownRule(ComplianceRule):
    """日最大回撤规则"""
    
    def __init__(self, rule_id: str, rule_name: str, threshold: float):
        super().__init__(rule_id, rule_name, "日回撤不得超过指定百分比", threshold, "CRITICAL")
    
    def check(self, value: float) -> Tuple[bool, str]:
        if value <= self.threshold:
            return True, f"日回撤 {value:.2%} 符合要求"
        return False, f"日回撤 {value:.2%} 超过限制 {self.threshold:.2%}"


class MaxWeeklyDrawdownRule(ComplianceRule):
    """周最大回撤规则"""
    
    def __init__(self, rule_id: str, rule_name: str, threshold: float):
        super().__init__(rule_id, rule_name, "周回撤不得超过指定百分比", threshold, "WARNING")
    
    def check(self, value: float) -> Tuple[bool, str]:
        if value <= self.threshold:
            return True, f"周回撤 {value:.2%} 符合要求"
        return False, f"周回撤 {value:.2%} 超过限制 {self.threshold:.2%}"


class MaxMonthlyDrawdownRule(ComplianceRule):
    """月最大回撤规则"""
    
    def __init__(self, rule_id: str, rule_name: str, threshold: float):
        super().__init__(rule_id, rule_name, "月回撤不得超过指定百分比", threshold, "WARNING")
    
    def check(self, value: float) -> Tuple[bool, str]:
        if value <= self.threshold:
            return True, f"月回撤 {value:.2%} 符合要求"
        return False, f"月回撤 {value:.2%} 超过限制 {self.threshold:.2%}"


class MinWinRateRule(ComplianceRule):
    """最低胜率规则"""
    
    def __init__(self, rule_id: str, rule_name: str, threshold: float):
        super().__init__(rule_id, rule_name, "胜率不得低于指定百分比", threshold, "WARNING")
    
    def check(self, value: float) -> Tuple[bool, str]:
        if value >= self.threshold:
            return True, f"胜率 {value:.2%} 符合要求"
        return False, f"胜率 {value:.2%} 低于要求 {self.threshold:.2%}"


class MaxConsecutiveLossesRule(ComplianceRule):
    """最大连续亏损规则"""
    
    def __init__(self, rule_id: str, rule_name: str, threshold: float):
        super().__init__(rule_id, rule_name, "连续亏损次数不得超过指定次数", threshold, "WARNING")
    
    def check(self, value: float) -> Tuple[bool, str]:
        if value <= self.threshold:
            return True, f"连续亏损 {int(value)} 次符合要求"
        return False, f"连续亏损 {int(value)} 次超过限制 {int(self.threshold)} 次"
