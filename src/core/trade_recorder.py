"""
交易记录系统模块
实现交易历史记录、分析、统计和报告生成
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import pandas as pd
from pathlib import Path

from src.core.models import Position, PositionType
from src.data.database import DatabaseManager


logger = logging.getLogger(__name__)


class TradeStatus(Enum):
    """交易状态"""
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class TradeOutcome(Enum):
    """交易结果"""
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"


@dataclass
class TradeRecord:
    """交易记录数据类"""
    trade_id: str
    symbol: str
    trade_type: str  # BUY/SELL
    volume: float
    open_price: float
    open_time: datetime
    strategy_id: str
    
    # 可选字段
    close_price: Optional[float] = None
    close_time: Optional[datetime] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    profit: float = 0.0
    commission: float = 0.0
    swap: float = 0.0
    comment: Optional[str] = None
    status: TradeStatus = TradeStatus.OPEN
    
    # 分析字段
    pips: Optional[float] = None
    duration_hours: Optional[float] = None
    outcome: Optional[TradeOutcome] = None
    risk_reward_ratio: Optional[float] = None
    mae: Optional[float] = None  # Maximum Adverse Excursion
    mfe: Optional[float] = None  # Maximum Favorable Excursion
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        # 转换枚举为字符串
        data['status'] = self.status.value
        if self.outcome:
            data['outcome'] = self.outcome.value
        # 转换datetime为字符串
        data['open_time'] = self.open_time.isoformat()
        if self.close_time:
            data['close_time'] = self.close_time.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradeRecord':
        """从字典创建"""
        # 转换字符串为枚举
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = TradeStatus(data['status'])
        if 'outcome' in data and isinstance(data['outcome'], str):
            data['outcome'] = TradeOutcome(data['outcome'])
        # 转换字符串为datetime
        if 'open_time' in data and isinstance(data['open_time'], str):
            data['open_time'] = datetime.fromisoformat(data['open_time'])
        if 'close_time' in data and isinstance(data['close_time'], str):
            data['close_time'] = datetime.fromisoformat(data['close_time'])
        return cls(**data)


class TradeRecorder:
    """交易记录器类"""
    
    def __init__(self, db_manager: DatabaseManager, config: Optional[Dict[str, Any]] = None):
        """
        初始化交易记录器
        
        Args:
            db_manager: 数据库管理器
            config: 配置字典
        """
        self.db_manager = db_manager
        self.config = config or {}
        
        # 日志配置
        self.log_dir = Path(self.config.get('log_dir', 'logs'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 内存缓存
        self.active_trades: Dict[str, TradeRecord] = {}
        self.recent_closed_trades: List[TradeRecord] = []
        self.max_recent_trades = self.config.get('max_recent_trades', 100)
        
        logger.info("交易记录器初始化完成")
    
    def record_trade_open(self, position: Position, strategy_id: str) -> TradeRecord:
        """
        记录开仓交易
        
        Args:
            position: 持仓对象
            strategy_id: 策略ID
            
        Returns:
            交易记录对象
        """
        trade_record = TradeRecord(
            trade_id=position.position_id,
            symbol=position.symbol,
            trade_type=position.type.name,
            volume=position.volume,
            open_price=position.open_price,
            open_time=position.open_time,
            sl=position.sl,
            tp=position.tp,
            strategy_id=strategy_id,
            comment=position.comment,
            status=TradeStatus.OPEN
        )
        
        # 保存到内存
        self.active_trades[trade_record.trade_id] = trade_record
        
        # 保存到数据库
        self._save_to_database(trade_record)
        
        # 写入日志
        self._write_trade_log(trade_record, "OPEN")
        
        logger.info(f"记录开仓: {trade_record.trade_id} {trade_record.symbol} {trade_record.trade_type}")
        
        return trade_record
    
    def record_trade_close(self, position: Position, close_price: float, 
                          profit: float, commission: float = 0.0, swap: float = 0.0) -> TradeRecord:
        """
        记录平仓交易
        
        Args:
            position: 持仓对象
            close_price: 平仓价格
            profit: 盈亏
            commission: 手续费
            swap: 隔夜利息
            
        Returns:
            交易记录对象
        """
        trade_id = position.position_id
        
        # 从活跃交易中获取或创建新记录
        if trade_id in self.active_trades:
            trade_record = self.active_trades[trade_id]
        else:
            # 如果不在活跃交易中，创建新记录
            trade_record = TradeRecord(
                trade_id=trade_id,
                symbol=position.symbol,
                trade_type=position.type.name,
                volume=position.volume,
                open_price=position.open_price,
                open_time=position.open_time,
                sl=position.sl,
                tp=position.tp,
                strategy_id="unknown",
                comment=position.comment
            )
        
        # 更新平仓信息
        trade_record.close_price = close_price
        trade_record.close_time = datetime.now()
        trade_record.profit = profit
        trade_record.commission = commission
        trade_record.swap = swap
        trade_record.status = TradeStatus.CLOSED
        
        # 计算分析指标
        self._calculate_trade_metrics(trade_record)
        
        # 从活跃交易中移除
        if trade_id in self.active_trades:
            del self.active_trades[trade_id]
        
        # 添加到最近关闭交易
        self.recent_closed_trades.append(trade_record)
        if len(self.recent_closed_trades) > self.max_recent_trades:
            self.recent_closed_trades.pop(0)
        
        # 更新数据库
        self._save_to_database(trade_record)
        
        # 写入日志
        self._write_trade_log(trade_record, "CLOSE")
        
        logger.info(f"记录平仓: {trade_record.trade_id} {trade_record.symbol} 盈亏: {profit:.2f}")
        
        return trade_record
    
    def _calculate_trade_metrics(self, trade_record: TradeRecord) -> None:
        """
        计算交易指标
        
        Args:
            trade_record: 交易记录
        """
        if not trade_record.close_price or not trade_record.close_time:
            return
        
        # 计算点数
        price_diff = trade_record.close_price - trade_record.open_price
        if trade_record.trade_type == "SHORT":
            price_diff = -price_diff
        
        # 简化的点数计算（实际应根据品种调整）
        trade_record.pips = price_diff * 10000  # 假设是外汇对
        
        # 计算持仓时长
        duration = trade_record.close_time - trade_record.open_time
        trade_record.duration_hours = duration.total_seconds() / 3600
        
        # 确定交易结果
        net_profit = trade_record.profit - trade_record.commission - trade_record.swap
        if net_profit > 0.01:
            trade_record.outcome = TradeOutcome.WIN
        elif net_profit < -0.01:
            trade_record.outcome = TradeOutcome.LOSS
        else:
            trade_record.outcome = TradeOutcome.BREAKEVEN
        
        # 计算风险回报比
        if trade_record.sl and trade_record.tp:
            if trade_record.trade_type == "BUY":
                risk = trade_record.open_price - trade_record.sl
                reward = trade_record.tp - trade_record.open_price
            else:
                risk = trade_record.sl - trade_record.open_price
                reward = trade_record.open_price - trade_record.tp
            
            if risk > 0:
                trade_record.risk_reward_ratio = reward / risk
    
    def _save_to_database(self, trade_record: TradeRecord) -> bool:
        """
        保存交易记录到数据库
        
        Args:
            trade_record: 交易记录
            
        Returns:
            是否成功
        """
        try:
            trade_data = {
                'trade_id': trade_record.trade_id,
                'symbol': trade_record.symbol,
                'trade_type': trade_record.trade_type,
                'volume': trade_record.volume,
                'open_price': trade_record.open_price,
                'close_price': trade_record.close_price,
                'sl': trade_record.sl,
                'tp': trade_record.tp,
                'profit': trade_record.profit,
                'commission': trade_record.commission,
                'swap': trade_record.swap,
                'open_time': trade_record.open_time,
                'close_time': trade_record.close_time,
                'strategy_id': trade_record.strategy_id,
                'comment': trade_record.comment
            }
            
            return self.db_manager.save_trade_record(trade_data)
        except Exception as e:
            logger.error(f"保存交易记录到数据库失败: {str(e)}")
            return False
    
    def _write_trade_log(self, trade_record: TradeRecord, action: str) -> None:
        """
        写入交易日志
        
        Args:
            trade_record: 交易记录
            action: 操作类型 (OPEN/CLOSE)
        """
        try:
            log_file = self.log_dir / "trades.log"
            
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'action': action,
                'trade_id': trade_record.trade_id,
                'symbol': trade_record.symbol,
                'type': trade_record.trade_type,
                'volume': trade_record.volume,
                'open_price': trade_record.open_price,
                'close_price': trade_record.close_price,
                'profit': trade_record.profit,
                'strategy_id': trade_record.strategy_id
            }
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                
        except Exception as e:
            logger.error(f"写入交易日志失败: {str(e)}")
    
    def get_trade_history(self, symbol: Optional[str] = None,
                         strategy_id: Optional[str] = None,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> List[TradeRecord]:
        """
        获取交易历史
        
        Args:
            symbol: 交易品种
            strategy_id: 策略ID
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            交易记录列表
        """
        try:
            df = self.db_manager.get_trade_records(
                symbol=symbol,
                strategy_id=strategy_id,
                start_time=start_time,
                end_time=end_time
            )
            
            if df is None or df.empty:
                return []
            
            # 转换为TradeRecord对象列表
            trades = []
            for _, row in df.iterrows():
                trade = TradeRecord(
                    trade_id=row['trade_id'],
                    symbol=row['symbol'],
                    trade_type=row['trade_type'],
                    volume=row['volume'],
                    open_price=row['open_price'],
                    open_time=row['open_time'],
                    close_price=row['close_price'],
                    close_time=row['close_time'],
                    sl=row['sl'],
                    tp=row['tp'],
                    profit=row['profit'],
                    commission=row['commission'],
                    swap=row['swap'],
                    strategy_id=row['strategy_id'],
                    comment=row['comment'],
                    status=TradeStatus.CLOSED if row['close_time'] else TradeStatus.OPEN
                )
                
                # 计算指标
                if trade.close_time:
                    self._calculate_trade_metrics(trade)
                
                trades.append(trade)
            
            return trades
            
        except Exception as e:
            logger.error(f"获取交易历史失败: {str(e)}")
            return []
    
    def get_trade_statistics(self, symbol: Optional[str] = None,
                           strategy_id: Optional[str] = None,
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        获取交易统计信息
        
        Args:
            symbol: 交易品种
            strategy_id: 策略ID
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            统计信息字典
        """
        trades = self.get_trade_history(symbol, strategy_id, start_time, end_time)
        
        if not trades:
            return {'total_trades': 0}
        
        # 只统计已关闭的交易
        closed_trades = [t for t in trades if t.status == TradeStatus.CLOSED]
        
        if not closed_trades:
            return {'total_trades': len(trades), 'closed_trades': 0}
        
        # 基本统计
        total_profit = sum(t.profit for t in closed_trades)
        total_commission = sum(t.commission for t in closed_trades)
        total_swap = sum(t.swap for t in closed_trades)
        net_profit = total_profit - total_commission - total_swap
        
        # 盈亏交易
        winning_trades = [t for t in closed_trades if t.outcome == TradeOutcome.WIN]
        losing_trades = [t for t in closed_trades if t.outcome == TradeOutcome.LOSS]
        
        # 胜率
        win_rate = (len(winning_trades) / len(closed_trades) * 100) if closed_trades else 0
        
        # 平均盈亏
        avg_win = (sum(t.profit for t in winning_trades) / len(winning_trades)) if winning_trades else 0
        avg_loss = (sum(t.profit for t in losing_trades) / len(losing_trades)) if losing_trades else 0
        
        # 最大盈亏
        max_win = max((t.profit for t in closed_trades), default=0)
        max_loss = min((t.profit for t in closed_trades), default=0)
        
        # 平均持仓时长
        avg_duration = (sum(t.duration_hours for t in closed_trades if t.duration_hours) / 
                       len(closed_trades)) if closed_trades else 0
        
        # 盈亏比
        profit_factor = (abs(sum(t.profit for t in winning_trades)) / 
                        abs(sum(t.profit for t in losing_trades))) if losing_trades and sum(t.profit for t in losing_trades) != 0 else 0
        
        # 连续盈亏
        max_consecutive_wins, max_consecutive_losses = self._calculate_consecutive_trades(closed_trades)
        
        # 按品种统计
        symbols_stats = {}
        for symbol_name in set(t.symbol for t in closed_trades):
            symbol_trades = [t for t in closed_trades if t.symbol == symbol_name]
            symbols_stats[symbol_name] = {
                'count': len(symbol_trades),
                'profit': sum(t.profit for t in symbol_trades),
                'win_rate': len([t for t in symbol_trades if t.outcome == TradeOutcome.WIN]) / len(symbol_trades) * 100
            }
        
        # 按策略统计
        strategies_stats = {}
        for strat_id in set(t.strategy_id for t in closed_trades):
            strat_trades = [t for t in closed_trades if t.strategy_id == strat_id]
            strategies_stats[strat_id] = {
                'count': len(strat_trades),
                'profit': sum(t.profit for t in strat_trades),
                'win_rate': len([t for t in strat_trades if t.outcome == TradeOutcome.WIN]) / len(strat_trades) * 100
            }
        
        return {
            'total_trades': len(trades),
            'closed_trades': len(closed_trades),
            'open_trades': len(trades) - len(closed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': round(win_rate, 2),
            'total_profit': round(total_profit, 2),
            'total_commission': round(total_commission, 2),
            'total_swap': round(total_swap, 2),
            'net_profit': round(net_profit, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'max_win': round(max_win, 2),
            'max_loss': round(max_loss, 2),
            'avg_duration_hours': round(avg_duration, 2),
            'profit_factor': round(profit_factor, 2),
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'by_symbol': symbols_stats,
            'by_strategy': strategies_stats
        }
    
    def _calculate_consecutive_trades(self, trades: List[TradeRecord]) -> Tuple[int, int]:
        """
        计算最大连续盈亏次数
        
        Args:
            trades: 交易记录列表
            
        Returns:
            (最大连续盈利次数, 最大连续亏损次数)
        """
        if not trades:
            return 0, 0
        
        # 按时间排序
        sorted_trades = sorted(trades, key=lambda t: t.close_time or t.open_time)
        
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in sorted_trades:
            if trade.outcome == TradeOutcome.WIN:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif trade.outcome == TradeOutcome.LOSS:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
            else:
                current_wins = 0
                current_losses = 0
        
        return max_wins, max_losses

    def generate_daily_report(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        生成每日交易报告
        
        Args:
            date: 日期（默认为今天）
            
        Returns:
            报告字典
        """
        if date is None:
            date = datetime.now()
        
        start_time = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = start_time + timedelta(days=1)
        
        # 获取当日交易
        trades = self.get_trade_history(start_time=start_time, end_time=end_time)
        
        # 获取统计信息
        stats = self.get_trade_statistics(start_time=start_time, end_time=end_time)
        
        report = {
            'date': date.strftime('%Y-%m-%d'),
            'summary': stats,
            'trades': [t.to_dict() for t in trades if t.status == TradeStatus.CLOSED],
            'open_positions': [t.to_dict() for t in trades if t.status == TradeStatus.OPEN]
        }
        
        # 保存报告
        self._save_report(report, 'daily', date)
        
        logger.info(f"生成每日报告: {date.strftime('%Y-%m-%d')}")
        
        return report
    
    def generate_weekly_report(self, week_start: Optional[datetime] = None) -> Dict[str, Any]:
        """
        生成每周交易报告
        
        Args:
            week_start: 周开始日期（默认为本周一）
            
        Returns:
            报告字典
        """
        if week_start is None:
            today = datetime.now()
            week_start = today - timedelta(days=today.weekday())
        
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        week_end = week_start + timedelta(days=7)
        
        # 获取本周交易
        trades = self.get_trade_history(start_time=week_start, end_time=week_end)
        
        # 获取统计信息
        stats = self.get_trade_statistics(start_time=week_start, end_time=week_end)
        
        # 按日统计
        daily_stats = {}
        for i in range(7):
            day = week_start + timedelta(days=i)
            day_end = day + timedelta(days=1)
            day_stats = self.get_trade_statistics(start_time=day, end_time=day_end)
            daily_stats[day.strftime('%Y-%m-%d')] = {
                'trades': day_stats.get('closed_trades', 0),
                'profit': day_stats.get('net_profit', 0)
            }
        
        report = {
            'week_start': week_start.strftime('%Y-%m-%d'),
            'week_end': week_end.strftime('%Y-%m-%d'),
            'summary': stats,
            'daily_breakdown': daily_stats,
            'total_trades': len([t for t in trades if t.status == TradeStatus.CLOSED])
        }
        
        # 保存报告
        self._save_report(report, 'weekly', week_start)
        
        logger.info(f"生成每周报告: {week_start.strftime('%Y-%m-%d')}")
        
        return report
    
    def generate_monthly_report(self, month: Optional[datetime] = None) -> Dict[str, Any]:
        """
        生成每月交易报告
        
        Args:
            month: 月份（默认为本月）
            
        Returns:
            报告字典
        """
        if month is None:
            month = datetime.now()
        
        # 月初和月末
        month_start = month.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if month_start.month == 12:
            month_end = month_start.replace(year=month_start.year + 1, month=1)
        else:
            month_end = month_start.replace(month=month_start.month + 1)
        
        # 获取本月交易
        trades = self.get_trade_history(start_time=month_start, end_time=month_end)
        
        # 获取统计信息
        stats = self.get_trade_statistics(start_time=month_start, end_time=month_end)
        
        # 按周统计
        weekly_stats = {}
        current_week_start = month_start
        week_num = 1
        while current_week_start < month_end:
            current_week_end = min(current_week_start + timedelta(days=7), month_end)
            week_stats = self.get_trade_statistics(start_time=current_week_start, end_time=current_week_end)
            weekly_stats[f'week_{week_num}'] = {
                'start': current_week_start.strftime('%Y-%m-%d'),
                'end': current_week_end.strftime('%Y-%m-%d'),
                'trades': week_stats.get('closed_trades', 0),
                'profit': week_stats.get('net_profit', 0)
            }
            current_week_start = current_week_end
            week_num += 1
        
        report = {
            'month': month_start.strftime('%Y-%m'),
            'summary': stats,
            'weekly_breakdown': weekly_stats,
            'total_trades': len([t for t in trades if t.status == TradeStatus.CLOSED]),
            'best_day': self._find_best_day(trades),
            'worst_day': self._find_worst_day(trades)
        }
        
        # 保存报告
        self._save_report(report, 'monthly', month_start)
        
        logger.info(f"生成每月报告: {month_start.strftime('%Y-%m')}")
        
        return report
    
    def _find_best_day(self, trades: List[TradeRecord]) -> Dict[str, Any]:
        """
        找出最佳交易日
        
        Args:
            trades: 交易记录列表
            
        Returns:
            最佳交易日信息
        """
        if not trades:
            return {}
        
        # 按日期分组
        daily_profits = {}
        for trade in trades:
            if trade.status == TradeStatus.CLOSED and trade.close_time:
                date_key = trade.close_time.strftime('%Y-%m-%d')
                if date_key not in daily_profits:
                    daily_profits[date_key] = 0
                daily_profits[date_key] += trade.profit
        
        if not daily_profits:
            return {}
        
        best_date = max(daily_profits, key=daily_profits.get)
        return {
            'date': best_date,
            'profit': round(daily_profits[best_date], 2)
        }
    
    def _find_worst_day(self, trades: List[TradeRecord]) -> Dict[str, Any]:
        """
        找出最差交易日
        
        Args:
            trades: 交易记录列表
            
        Returns:
            最差交易日信息
        """
        if not trades:
            return {}
        
        # 按日期分组
        daily_profits = {}
        for trade in trades:
            if trade.status == TradeStatus.CLOSED and trade.close_time:
                date_key = trade.close_time.strftime('%Y-%m-%d')
                if date_key not in daily_profits:
                    daily_profits[date_key] = 0
                daily_profits[date_key] += trade.profit
        
        if not daily_profits:
            return {}
        
        worst_date = min(daily_profits, key=daily_profits.get)
        return {
            'date': worst_date,
            'profit': round(daily_profits[worst_date], 2)
        }
    
    def _save_report(self, report: Dict[str, Any], report_type: str, date: datetime) -> None:
        """
        保存报告到文件
        
        Args:
            report: 报告字典
            report_type: 报告类型 (daily/weekly/monthly)
            date: 日期
        """
        try:
            report_dir = self.log_dir / 'reports' / report_type
            report_dir.mkdir(parents=True, exist_ok=True)
            
            if report_type == 'daily':
                filename = f"report_{date.strftime('%Y%m%d')}.json"
            elif report_type == 'weekly':
                filename = f"report_week_{date.strftime('%Y%m%d')}.json"
            else:  # monthly
                filename = f"report_{date.strftime('%Y%m')}.json"
            
            report_file = report_dir / filename
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            logger.debug(f"报告已保存: {report_file}")
            
        except Exception as e:
            logger.error(f"保存报告失败: {str(e)}")
    
    def get_audit_log(self, start_time: Optional[datetime] = None,
                     end_time: Optional[datetime] = None,
                     trade_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取审计日志
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            trade_id: 交易ID
            
        Returns:
            审计日志列表
        """
        try:
            log_file = self.log_dir / "trades.log"
            
            if not log_file.exists():
                return []
            
            audit_logs = []
            
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        
                        # 过滤条件
                        if trade_id and log_entry.get('trade_id') != trade_id:
                            continue
                        
                        entry_time = datetime.fromisoformat(log_entry['timestamp'])
                        
                        if start_time and entry_time < start_time:
                            continue
                        
                        if end_time and entry_time > end_time:
                            continue
                        
                        audit_logs.append(log_entry)
                        
                    except json.JSONDecodeError:
                        continue
            
            return audit_logs
            
        except Exception as e:
            logger.error(f"获取审计日志失败: {str(e)}")
            return []
    
    def export_trades_to_csv(self, filepath: str,
                           symbol: Optional[str] = None,
                           strategy_id: Optional[str] = None,
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None) -> bool:
        """
        导出交易记录到CSV文件
        
        Args:
            filepath: 文件路径
            symbol: 交易品种
            strategy_id: 策略ID
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            是否成功
        """
        try:
            trades = self.get_trade_history(symbol, strategy_id, start_time, end_time)
            
            if not trades:
                logger.warning("没有交易记录可导出")
                return False
            
            # 转换为DataFrame
            data = [t.to_dict() for t in trades]
            df = pd.DataFrame(data)
            
            # 导出到CSV
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            
            logger.info(f"交易记录已导出到: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"导出交易记录失败: {str(e)}")
            return False
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        获取整体性能摘要
        
        Returns:
            性能摘要字典
        """
        # 获取所有交易统计
        all_stats = self.get_trade_statistics()
        
        # 获取最近30天统计
        thirty_days_ago = datetime.now() - timedelta(days=30)
        recent_stats = self.get_trade_statistics(start_time=thirty_days_ago)
        
        # 获取今日统计
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_stats = self.get_trade_statistics(start_time=today_start)
        
        return {
            'all_time': all_stats,
            'last_30_days': recent_stats,
            'today': today_stats,
            'active_trades': len(self.active_trades),
            'recent_closed_trades': len(self.recent_closed_trades)
        }
    
    def cleanup_old_logs(self, days_to_keep: int = 90) -> bool:
        """
        清理旧日志文件
        
        Args:
            days_to_keep: 保留天数
            
        Returns:
            是否成功
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # 清理交易日志
            log_file = self.log_dir / "trades.log"
            if log_file.exists():
                temp_file = self.log_dir / "trades_temp.log"
                
                with open(log_file, 'r', encoding='utf-8') as f_in:
                    with open(temp_file, 'w', encoding='utf-8') as f_out:
                        for line in f_in:
                            try:
                                log_entry = json.loads(line.strip())
                                entry_time = datetime.fromisoformat(log_entry['timestamp'])
                                
                                if entry_time >= cutoff_date:
                                    f_out.write(line)
                            except:
                                continue
                
                # 替换原文件
                temp_file.replace(log_file)
            
            logger.info(f"清理旧日志完成，保留 {days_to_keep} 天")
            return True
            
        except Exception as e:
            logger.error(f"清理旧日志失败: {str(e)}")
            return False
