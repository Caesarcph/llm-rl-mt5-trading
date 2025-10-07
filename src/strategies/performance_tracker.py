"""
策略性能跟踪器
实时监控策略表现，生成性能报告和策略排名
"""

import logging
import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import pandas as pd
import numpy as np

from src.core.models import Trade, Signal, RiskMetrics


class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理numpy类型"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class PerformancePeriod(Enum):
    """性能统计周期"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    ALL_TIME = "all_time"


@dataclass
class StrategyPerformanceMetrics:
    """策略性能指标"""
    strategy_name: str
    period: str
    start_date: datetime
    end_date: datetime
    
    # 交易统计
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    # 盈亏统计
    total_profit: float = 0.0
    total_loss: float = 0.0
    net_profit: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    
    # 比率指标
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # 风险指标
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    avg_drawdown: float = 0.0
    recovery_factor: float = 0.0
    
    # 交易质量
    avg_profit_per_trade: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_trade_duration: float = 0.0  # 小时
    
    # 连续性指标
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    current_streak: int = 0
    
    # 其他
    total_commission: float = 0.0
    total_swap: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        # 转换datetime为字符串
        result['start_date'] = self.start_date.isoformat()
        result['end_date'] = self.end_date.isoformat()
        
        # 转换numpy类型为Python原生类型
        for key, value in result.items():
            if hasattr(value, 'item'):  # numpy类型
                result[key] = value.item()
        
        return result


@dataclass
class PerformanceReport:
    """性能报告"""
    report_id: str
    generated_at: datetime
    period: PerformancePeriod
    strategies: List[StrategyPerformanceMetrics]
    summary: Dict[str, Any] = field(default_factory=dict)
    rankings: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'report_id': self.report_id,
            'generated_at': self.generated_at.isoformat(),
            'period': self.period.value,
            'strategies': [s.to_dict() for s in self.strategies],
            'summary': self.summary,
            'rankings': self.rankings
        }
    
    def to_json(self) -> str:
        """转换为JSON"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, cls=NumpyEncoder)


class PerformanceTracker:
    """
    策略性能跟踪器
    实时监控和记录各策略的表现
    """
    
    def __init__(self, db_path: str = "data/performance.db"):
        """
        初始化性能跟踪器
        
        Args:
            db_path: 数据库路径
        """
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        
        # 内存中的交易记录
        self.trades_by_strategy: Dict[str, List[Trade]] = {}
        
        # 信号记录
        self.signals_by_strategy: Dict[str, List[Signal]] = {}
        
        # 性能指标缓存
        self.metrics_cache: Dict[str, StrategyPerformanceMetrics] = {}
        
        # 报告历史
        self.report_history: List[PerformanceReport] = []
        
        self.logger.info("性能跟踪器初始化完成")
    
    def record_trade(self, trade: Trade) -> None:
        """
        记录交易
        
        Args:
            trade: 交易记录
        """
        strategy_id = trade.strategy_id
        
        if strategy_id not in self.trades_by_strategy:
            self.trades_by_strategy[strategy_id] = []
        
        self.trades_by_strategy[strategy_id].append(trade)
        
        # 清除缓存
        if strategy_id in self.metrics_cache:
            del self.metrics_cache[strategy_id]
        
        self.logger.debug(f"记录交易: {strategy_id}, 盈亏: {trade.profit}")
    
    def record_signal(self, signal: Signal) -> None:
        """
        记录信号
        
        Args:
            signal: 交易信号
        """
        strategy_id = signal.strategy_id
        
        if strategy_id not in self.signals_by_strategy:
            self.signals_by_strategy[strategy_id] = []
        
        self.signals_by_strategy[strategy_id].append(signal)
        
        self.logger.debug(f"记录信号: {strategy_id}, 方向: {signal.direction}")
    
    def calculate_metrics(
        self,
        strategy_name: str,
        start_date: datetime = None,
        end_date: datetime = None,
        use_cache: bool = True
    ) -> StrategyPerformanceMetrics:
        """
        计算策略性能指标
        
        Args:
            strategy_name: 策略名称
            start_date: 开始日期
            end_date: 结束日期
            use_cache: 是否使用缓存
            
        Returns:
            性能指标
        """
        # 检查缓存
        cache_key = f"{strategy_name}_{start_date}_{end_date}"
        if use_cache and cache_key in self.metrics_cache:
            return self.metrics_cache[cache_key]
        
        # 获取交易记录
        trades = self.trades_by_strategy.get(strategy_name, [])
        
        # 过滤日期范围
        if start_date or end_date:
            trades = self._filter_trades_by_date(trades, start_date, end_date)
        
        # 计算指标
        metrics = self._calculate_metrics_from_trades(
            strategy_name, trades, start_date, end_date
        )
        
        # 缓存结果
        self.metrics_cache[cache_key] = metrics
        
        return metrics
    
    def _filter_trades_by_date(
        self,
        trades: List[Trade],
        start_date: datetime = None,
        end_date: datetime = None
    ) -> List[Trade]:
        """按日期过滤交易"""
        filtered = trades
        
        if start_date:
            filtered = [t for t in filtered if t.open_time >= start_date]
        
        if end_date:
            filtered = [t for t in filtered if t.open_time <= end_date]
        
        return filtered
    
    def _calculate_metrics_from_trades(
        self,
        strategy_name: str,
        trades: List[Trade],
        start_date: datetime = None,
        end_date: datetime = None
    ) -> StrategyPerformanceMetrics:
        """从交易记录计算指标"""
        if not start_date:
            start_date = trades[0].open_time if trades else datetime.now()
        if not end_date:
            end_date = trades[-1].close_time if trades and trades[-1].close_time else datetime.now()
        
        metrics = StrategyPerformanceMetrics(
            strategy_name=strategy_name,
            period="custom",
            start_date=start_date,
            end_date=end_date
        )
        
        if not trades:
            return metrics
        
        # 基本统计
        metrics.total_trades = len(trades)
        
        profits = [t.profit for t in trades]
        winning_trades = [t for t in trades if t.profit > 0]
        losing_trades = [t for t in trades if t.profit < 0]
        
        metrics.winning_trades = len(winning_trades)
        metrics.losing_trades = len(losing_trades)
        
        # 盈亏统计
        metrics.gross_profit = sum(t.profit for t in winning_trades)
        metrics.gross_loss = abs(sum(t.profit for t in losing_trades))
        metrics.net_profit = sum(profits)
        metrics.total_profit = metrics.gross_profit
        metrics.total_loss = metrics.gross_loss
        
        # 佣金和隔夜利息
        metrics.total_commission = sum(t.commission for t in trades)
        metrics.total_swap = sum(t.swap for t in trades)
        
        # 比率指标
        metrics.win_rate = metrics.winning_trades / metrics.total_trades if metrics.total_trades > 0 else 0
        metrics.profit_factor = metrics.gross_profit / metrics.gross_loss if metrics.gross_loss > 0 else 0
        
        # 夏普比率
        if len(profits) > 1:
            returns = np.array(profits)
            metrics.sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            
            # 索提诺比率
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 0:
                downside_std = np.std(negative_returns)
                metrics.sortino_ratio = np.mean(returns) / downside_std if downside_std > 0 else 0
        
        # 回撤分析
        cumulative = np.cumsum(profits)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        
        metrics.max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0
        
        if running_max[-1] > 0:
            metrics.max_drawdown_percent = (metrics.max_drawdown / running_max[-1]) * 100
        
        # 卡玛比率
        if metrics.max_drawdown > 0:
            metrics.calmar_ratio = metrics.net_profit / metrics.max_drawdown
            metrics.recovery_factor = metrics.net_profit / metrics.max_drawdown
        
        # 交易质量
        metrics.avg_profit_per_trade = metrics.net_profit / metrics.total_trades if metrics.total_trades > 0 else 0
        metrics.avg_win = metrics.gross_profit / metrics.winning_trades if metrics.winning_trades > 0 else 0
        metrics.avg_loss = metrics.gross_loss / metrics.losing_trades if metrics.losing_trades > 0 else 0
        
        if profits:
            metrics.largest_win = max(profits)
            metrics.largest_loss = min(profits)
        
        # 平均交易持续时间
        durations = [t.get_duration() for t in trades if t.get_duration() is not None]
        metrics.avg_trade_duration = np.mean(durations) if durations else 0
        
        # 连续性指标
        metrics.max_consecutive_wins = self._calculate_max_consecutive(profits, positive=True)
        metrics.max_consecutive_losses = self._calculate_max_consecutive(profits, positive=False)
        metrics.current_streak = self._calculate_current_streak(profits)
        
        return metrics
    
    def _calculate_max_consecutive(self, profits: List[float], positive: bool = True) -> int:
        """计算最大连续盈利/亏损次数"""
        if not profits:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for profit in profits:
            if (positive and profit > 0) or (not positive and profit < 0):
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_current_streak(self, profits: List[float]) -> int:
        """计算当前连续盈利/亏损次数（正数表示盈利，负数表示亏损）"""
        if not profits:
            return 0
        
        streak = 0
        for profit in reversed(profits):
            if profit > 0:
                if streak >= 0:
                    streak += 1
                else:
                    break
            elif profit < 0:
                if streak <= 0:
                    streak -= 1
                else:
                    break
            else:
                break
        
        return streak
    
    def generate_report(
        self,
        period: PerformancePeriod = PerformancePeriod.DAILY,
        strategy_names: List[str] = None
    ) -> PerformanceReport:
        """
        生成性能报告
        
        Args:
            period: 统计周期
            strategy_names: 策略名称列表（None表示所有策略）
            
        Returns:
            性能报告
        """
        # 确定日期范围
        end_date = datetime.now()
        start_date = self._get_period_start_date(period, end_date)
        
        # 确定要报告的策略
        if strategy_names is None:
            strategy_names = list(self.trades_by_strategy.keys())
        
        # 计算各策略指标
        strategies_metrics = []
        for name in strategy_names:
            metrics = self.calculate_metrics(name, start_date, end_date, use_cache=False)
            strategies_metrics.append(metrics)
        
        # 生成排名
        rankings = self._generate_rankings(strategies_metrics)
        
        # 生成摘要
        summary = self._generate_summary(strategies_metrics)
        
        # 创建报告
        report = PerformanceReport(
            report_id=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            generated_at=datetime.now(),
            period=period,
            strategies=strategies_metrics,
            summary=summary,
            rankings=rankings
        )
        
        # 保存到历史
        self.report_history.append(report)
        
        self.logger.info(f"生成性能报告: {report.report_id}, 策略数: {len(strategies_metrics)}")
        
        return report
    
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
    
    def _generate_rankings(
        self,
        strategies_metrics: List[StrategyPerformanceMetrics]
    ) -> Dict[str, List[Tuple[str, float]]]:
        """生成策略排名"""
        rankings = {}
        
        # 按净利润排名
        rankings['net_profit'] = sorted(
            [(m.strategy_name, m.net_profit) for m in strategies_metrics],
            key=lambda x: x[1],
            reverse=True
        )
        
        # 按胜率排名
        rankings['win_rate'] = sorted(
            [(m.strategy_name, m.win_rate) for m in strategies_metrics],
            key=lambda x: x[1],
            reverse=True
        )
        
        # 按盈利因子排名
        rankings['profit_factor'] = sorted(
            [(m.strategy_name, m.profit_factor) for m in strategies_metrics],
            key=lambda x: x[1],
            reverse=True
        )
        
        # 按夏普比率排名
        rankings['sharpe_ratio'] = sorted(
            [(m.strategy_name, m.sharpe_ratio) for m in strategies_metrics],
            key=lambda x: x[1],
            reverse=True
        )
        
        # 按最大回撤排名（越小越好）
        rankings['max_drawdown'] = sorted(
            [(m.strategy_name, m.max_drawdown) for m in strategies_metrics],
            key=lambda x: x[1]
        )
        
        return rankings
    
    def _generate_summary(
        self,
        strategies_metrics: List[StrategyPerformanceMetrics]
    ) -> Dict[str, Any]:
        """生成摘要统计"""
        if not strategies_metrics:
            return {}
        
        total_trades = sum(m.total_trades for m in strategies_metrics)
        total_net_profit = sum(m.net_profit for m in strategies_metrics)
        avg_win_rate = np.mean([m.win_rate for m in strategies_metrics])
        avg_profit_factor = np.mean([m.profit_factor for m in strategies_metrics if m.profit_factor > 0])
        
        return {
            'total_strategies': len(strategies_metrics),
            'total_trades': total_trades,
            'total_net_profit': total_net_profit,
            'avg_win_rate': avg_win_rate,
            'avg_profit_factor': avg_profit_factor,
            'best_strategy': strategies_metrics[0].strategy_name if strategies_metrics else None,
            'worst_strategy': strategies_metrics[-1].strategy_name if strategies_metrics else None
        }
    
    def get_strategy_ranking(
        self,
        metric: str = 'net_profit',
        period: PerformancePeriod = PerformancePeriod.ALL_TIME
    ) -> List[Tuple[str, float]]:
        """
        获取策略排名
        
        Args:
            metric: 排名指标
            period: 统计周期
            
        Returns:
            排名列表 [(策略名, 指标值)]
        """
        end_date = datetime.now()
        start_date = self._get_period_start_date(period, end_date)
        
        rankings = []
        for strategy_name in self.trades_by_strategy.keys():
            metrics = self.calculate_metrics(strategy_name, start_date, end_date)
            
            if hasattr(metrics, metric):
                value = getattr(metrics, metric)
                rankings.append((strategy_name, value))
        
        # 排序（大多数指标越大越好，除了回撤）
        reverse = metric not in ['max_drawdown', 'max_drawdown_percent', 'avg_loss']
        rankings.sort(key=lambda x: x[1], reverse=reverse)
        
        return rankings
    
    def get_performance_comparison(
        self,
        strategy_names: List[str],
        period: PerformancePeriod = PerformancePeriod.MONTHLY
    ) -> pd.DataFrame:
        """
        获取策略性能对比
        
        Args:
            strategy_names: 策略名称列表
            period: 统计周期
            
        Returns:
            对比数据框
        """
        end_date = datetime.now()
        start_date = self._get_period_start_date(period, end_date)
        
        comparison_data = []
        
        for name in strategy_names:
            metrics = self.calculate_metrics(name, start_date, end_date)
            
            comparison_data.append({
                'Strategy': name,
                'Total Trades': metrics.total_trades,
                'Win Rate': f"{metrics.win_rate:.2%}",
                'Net Profit': f"{metrics.net_profit:.2f}",
                'Profit Factor': f"{metrics.profit_factor:.2f}",
                'Sharpe Ratio': f"{metrics.sharpe_ratio:.2f}",
                'Max Drawdown': f"{metrics.max_drawdown:.2f}",
                'Avg Profit/Trade': f"{metrics.avg_profit_per_trade:.2f}"
            })
        
        return pd.DataFrame(comparison_data)
    
    def get_equity_curve(
        self,
        strategy_name: str,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> pd.DataFrame:
        """
        获取权益曲线
        
        Args:
            strategy_name: 策略名称
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            权益曲线数据框
        """
        trades = self.trades_by_strategy.get(strategy_name, [])
        trades = self._filter_trades_by_date(trades, start_date, end_date)
        
        if not trades:
            return pd.DataFrame()
        
        equity_data = []
        cumulative_profit = 0
        
        for trade in trades:
            cumulative_profit += trade.profit
            equity_data.append({
                'timestamp': trade.close_time or trade.open_time,
                'profit': trade.profit,
                'cumulative_profit': cumulative_profit,
                'trade_id': trade.trade_id
            })
        
        return pd.DataFrame(equity_data)
    
    def clear_cache(self) -> None:
        """清除指标缓存"""
        self.metrics_cache.clear()
        self.logger.info("性能指标缓存已清除")
    
    def get_report_history(self, limit: int = 10) -> List[PerformanceReport]:
        """
        获取报告历史
        
        Args:
            limit: 返回数量限制
            
        Returns:
            报告列表
        """
        return self.report_history[-limit:]
    
    def export_report(self, report: PerformanceReport, filepath: str) -> bool:
        """
        导出报告到文件
        
        Args:
            report: 性能报告
            filepath: 文件路径
            
        Returns:
            是否成功
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report.to_json())
            self.logger.info(f"报告已导出: {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"导出报告失败: {str(e)}")
            return False
