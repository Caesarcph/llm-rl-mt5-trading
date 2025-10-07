"""
回测框架模块
用于测试和验证策略性能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
from collections import defaultdict

from src.core.models import MarketData, Signal, Trade, Account, Position, PositionType, TradeType
from src.core.exceptions import BacktestException
from src.strategies.base_strategies import BaseStrategy, StrategyManager


@dataclass
class BacktestConfig:
    """回测配置"""
    initial_balance: float = 10000.0
    leverage: int = 100
    spread: float = 0.0001  # 点差
    commission: float = 0.0  # 手续费
    slippage: float = 0.0001  # 滑点
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    def validate(self) -> bool:
        """验证配置"""
        if self.initial_balance <= 0:
            raise ValueError("初始余额必须大于0")
        if self.leverage <= 0:
            raise ValueError("杠杆必须大于0")
        if self.spread < 0:
            raise ValueError("点差不能为负数")
        return True


@dataclass
class BacktestResult:
    """回测结果"""
    strategy_name: str
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_balance: float
    final_balance: float
    total_return: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown: float
    max_drawdown_percent: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    average_trade: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    consecutive_wins: int
    consecutive_losses: int
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)
    drawdown_curve: List[Tuple[datetime, float]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'strategy_name': self.strategy_name,
            'symbol': self.symbol,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'initial_balance': self.initial_balance,
            'final_balance': self.final_balance,
            'total_return': self.total_return,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_percent': self.max_drawdown_percent,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'average_trade': self.average_trade,
            'average_win': self.average_win,
            'average_loss': self.average_loss,
            'largest_win': self.largest_win,
            'largest_loss': self.largest_loss,
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses
        }


class BacktestEngine:
    """回测引擎 - 支持历史数据回测和多策略并行回测"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.config.validate()
        
        self.logger = logging.getLogger(__name__)
        
        # 回测状态
        self.current_balance = config.initial_balance
        self.current_equity = config.initial_balance
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.drawdown_curve: List[Tuple[datetime, float]] = []
        
        # 统计信息
        self.peak_equity = config.initial_balance
        self.max_drawdown = 0.0
        self.trade_counter = 0
        
        # 并行回测支持
        self.parallel_results: Dict[str, BacktestResult] = {}
    
    def run_backtest(self, strategy: BaseStrategy, market_data_list: List[MarketData]) -> BacktestResult:
        """运行回测"""
        try:
            self.logger.info(f"开始回测策略: {strategy.config.name}")
            
            # 重置状态
            self._reset_state()
            
            # 按时间排序数据
            market_data_list.sort(key=lambda x: x.timestamp)
            
            for i, market_data in enumerate(market_data_list):
                # 更新持仓价格
                self._update_positions(market_data)
                
                # 生成信号
                signal = strategy.generate_signal(market_data)
                
                if signal:
                    # 执行交易
                    self._execute_signal(signal, market_data)
                
                # 记录权益曲线
                self._update_equity_curve(market_data.timestamp)
                
                # 检查止损止盈
                self._check_stop_loss_take_profit(market_data)
            
            # 平仓所有持仓
            final_market_data = market_data_list[-1] if market_data_list else None
            if final_market_data:
                self._close_all_positions(final_market_data)
            
            # 计算回测结果
            result = self._calculate_backtest_result(strategy, market_data_list)
            
            self.logger.info(f"回测完成: {strategy.config.name}, 总收益: {result.total_return:.2%}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"回测失败: {str(e)}")
            raise BacktestException(f"回测执行失败: {str(e)}")
    
    def run_multiple_strategies(self, strategies: List[BaseStrategy], 
                              market_data_list: List[MarketData],
                              parallel: bool = False) -> Dict[str, BacktestResult]:
        """
        运行多个策略的回测
        
        Args:
            strategies: 策略列表
            market_data_list: 市场数据列表
            parallel: 是否并行执行
            
        Returns:
            策略名称到回测结果的映射
        """
        results = {}
        
        if parallel and len(strategies) > 1:
            # 并行执行
            self.logger.info(f"并行回测{len(strategies)}个策略")
            results = self._run_parallel_backtest(strategies, market_data_list)
        else:
            # 顺序执行
            for strategy in strategies:
                try:
                    self.logger.info(f"回测策略: {strategy.config.name}")
                    result = self.run_backtest(strategy, market_data_list)
                    results[strategy.config.name] = result
                except Exception as e:
                    self.logger.error(f"策略{strategy.config.name}回测失败: {str(e)}")
                    results[strategy.config.name] = None
        
        self.parallel_results = results
        return results
    
    def _run_parallel_backtest(self, strategies: List[BaseStrategy], 
                              market_data_list: List[MarketData]) -> Dict[str, BacktestResult]:
        """
        并行运行多个策略的回测
        
        Args:
            strategies: 策略列表
            market_data_list: 市场数据列表
            
        Returns:
            策略名称到回测结果的映射
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing
        
        results = {}
        max_workers = min(len(strategies), multiprocessing.cpu_count())
        
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                future_to_strategy = {
                    executor.submit(self._run_single_backtest_worker, strategy, market_data_list, self.config): strategy
                    for strategy in strategies
                }
                
                # 收集结果
                for future in as_completed(future_to_strategy):
                    strategy = future_to_strategy[future]
                    try:
                        result = future.result()
                        results[strategy.config.name] = result
                        self.logger.info(f"策略{strategy.config.name}回测完成")
                    except Exception as e:
                        self.logger.error(f"策略{strategy.config.name}回测失败: {str(e)}")
                        results[strategy.config.name] = None
        
        except Exception as e:
            self.logger.error(f"并行回测失败: {str(e)}, 回退到顺序执行")
            # 回退到顺序执行
            for strategy in strategies:
                try:
                    result = self.run_backtest(strategy, market_data_list)
                    results[strategy.config.name] = result
                except Exception as e:
                    self.logger.error(f"策略{strategy.config.name}回测失败: {str(e)}")
                    results[strategy.config.name] = None
        
        return results
    
    @staticmethod
    def _run_single_backtest_worker(strategy: BaseStrategy, 
                                   market_data_list: List[MarketData],
                                   config: BacktestConfig) -> BacktestResult:
        """
        单个策略回测的工作函数（用于并行执行）
        
        Args:
            strategy: 策略实例
            market_data_list: 市场数据列表
            config: 回测配置
            
        Returns:
            回测结果
        """
        # 创建新的回测引擎实例（避免共享状态）
        engine = BacktestEngine(config)
        return engine.run_backtest(strategy, market_data_list)
    
    def compare_strategies(self, results: Dict[str, BacktestResult]) -> pd.DataFrame:
        """
        比较多个策略的回测结果
        
        Args:
            results: 策略回测结果字典
            
        Returns:
            对比数据框
        """
        comparison_data = []
        
        for strategy_name, result in results.items():
            if result is None:
                continue
                
            comparison_data.append({
                'Strategy': strategy_name,
                'Total Return': f"{result.total_return:.2%}",
                'Total Trades': result.total_trades,
                'Win Rate': f"{result.win_rate:.2%}",
                'Profit Factor': f"{result.profit_factor:.2f}",
                'Sharpe Ratio': f"{result.sharpe_ratio:.2f}",
                'Max Drawdown': f"{result.max_drawdown_percent:.2%}",
                'Avg Trade': f"{result.average_trade:.2f}",
                'Final Balance': f"{result.final_balance:.2f}"
            })
        
        return pd.DataFrame(comparison_data)
    
    def visualize_results(self, result: BacktestResult, save_path: str = None) -> None:
        """
        可视化回测结果
        
        Args:
            result: 回测结果
            save_path: 保存路径（可选）
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            fig.suptitle(f'Backtest Results: {result.strategy_name} - {result.symbol}', fontsize=16)
            
            # 权益曲线
            if result.equity_curve:
                dates, equity = zip(*result.equity_curve)
                axes[0].plot(dates, equity, label='Equity', linewidth=2)
                axes[0].axhline(y=result.initial_balance, color='r', linestyle='--', label='Initial Balance')
                axes[0].set_title('Equity Curve')
                axes[0].set_ylabel('Equity ($)')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            
            # 回撤曲线
            if result.drawdown_curve:
                dates, drawdown = zip(*result.drawdown_curve)
                axes[1].fill_between(dates, 0, [-d*100 for d in drawdown], alpha=0.3, color='red')
                axes[1].plot(dates, [-d*100 for d in drawdown], color='red', linewidth=2)
                axes[1].set_title('Drawdown')
                axes[1].set_ylabel('Drawdown (%)')
                axes[1].grid(True, alpha=0.3)
                axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            
            # 交易分布
            if result.trades:
                profits = [t.profit for t in result.trades]
                axes[2].bar(range(len(profits)), profits, 
                           color=['green' if p > 0 else 'red' for p in profits],
                           alpha=0.6)
                axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                axes[2].set_title('Trade P&L Distribution')
                axes[2].set_xlabel('Trade Number')
                axes[2].set_ylabel('Profit/Loss ($)')
                axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"回测结果图表已保存: {save_path}")
            else:
                plt.show()
                
        except ImportError:
            self.logger.warning("matplotlib未安装，无法生成可视化图表")
        except Exception as e:
            self.logger.error(f"可视化失败: {str(e)}")
    
    def visualize_detailed_analysis(self, result: BacktestResult, save_path: str = None) -> None:
        """
        生成详细的回测分析图表
        
        Args:
            result: 回测结果
            save_path: 保存路径（可选）
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from matplotlib.gridspec import GridSpec
            
            fig = plt.figure(figsize=(16, 12))
            gs = GridSpec(3, 3, figure=fig)
            fig.suptitle(f'Detailed Backtest Analysis: {result.strategy_name} - {result.symbol}', fontsize=16)
            
            # 1. 权益曲线和回撤（大图）
            ax1 = fig.add_subplot(gs[0, :])
            if result.equity_curve:
                dates, equity = zip(*result.equity_curve)
                ax1.plot(dates, equity, label='Equity', linewidth=2, color='blue')
                ax1.axhline(y=result.initial_balance, color='gray', linestyle='--', label='Initial Balance', alpha=0.7)
                ax1.fill_between(dates, result.initial_balance, equity, 
                               where=[e >= result.initial_balance for e in equity],
                               alpha=0.2, color='green', label='Profit Zone')
                ax1.fill_between(dates, result.initial_balance, equity,
                               where=[e < result.initial_balance for e in equity],
                               alpha=0.2, color='red', label='Loss Zone')
                ax1.set_title('Equity Curve with Profit/Loss Zones')
                ax1.set_ylabel('Equity ($)')
                ax1.legend(loc='best')
                ax1.grid(True, alpha=0.3)
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            
            # 2. 累计收益率
            ax2 = fig.add_subplot(gs[1, 0])
            if result.equity_curve:
                dates, equity = zip(*result.equity_curve)
                returns = [(e - result.initial_balance) / result.initial_balance * 100 for e in equity]
                ax2.plot(dates, returns, linewidth=2, color='green')
                ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                ax2.set_title('Cumulative Return (%)')
                ax2.set_ylabel('Return (%)')
                ax2.grid(True, alpha=0.3)
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            
            # 3. 回撤曲线
            ax3 = fig.add_subplot(gs[1, 1])
            if result.drawdown_curve:
                dates, drawdown = zip(*result.drawdown_curve)
                ax3.fill_between(dates, 0, [-d*100 for d in drawdown], alpha=0.5, color='red')
                ax3.plot(dates, [-d*100 for d in drawdown], color='darkred', linewidth=2)
                ax3.set_title(f'Drawdown (Max: {result.max_drawdown_percent*100:.2f}%)')
                ax3.set_ylabel('Drawdown (%)')
                ax3.grid(True, alpha=0.3)
                ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            
            # 4. 盈亏分布直方图
            ax4 = fig.add_subplot(gs[1, 2])
            if result.trades:
                profits = [t.profit for t in result.trades]
                ax4.hist(profits, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
                ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
                ax4.set_title('P&L Distribution')
                ax4.set_xlabel('Profit/Loss ($)')
                ax4.set_ylabel('Frequency')
                ax4.grid(True, alpha=0.3, axis='y')
            
            # 5. 交易序列
            ax5 = fig.add_subplot(gs[2, :2])
            if result.trades:
                profits = [t.profit for t in result.trades]
                colors = ['green' if p > 0 else 'red' for p in profits]
                ax5.bar(range(len(profits)), profits, color=colors, alpha=0.6, edgecolor='black')
                ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                ax5.set_title(f'Trade Sequence (Total: {len(profits)}, Win Rate: {result.win_rate*100:.1f}%)')
                ax5.set_xlabel('Trade Number')
                ax5.set_ylabel('Profit/Loss ($)')
                ax5.grid(True, alpha=0.3, axis='y')
            
            # 6. 关键指标表格
            ax6 = fig.add_subplot(gs[2, 2])
            ax6.axis('off')
            
            metrics_text = f"""
            Key Metrics:
            
            Total Return: {result.total_return*100:.2f}%
            Total Trades: {result.total_trades}
            Win Rate: {result.win_rate*100:.2f}%
            Profit Factor: {result.profit_factor:.2f}
            
            Sharpe Ratio: {result.sharpe_ratio:.2f}
            Sortino Ratio: {result.sortino_ratio:.2f}
            Calmar Ratio: {result.calmar_ratio:.2f}
            
            Max Drawdown: ${result.max_drawdown:.2f}
            Max DD %: {result.max_drawdown_percent*100:.2f}%
            
            Avg Trade: ${result.average_trade:.2f}
            Avg Win: ${result.average_win:.2f}
            Avg Loss: ${result.average_loss:.2f}
            
            Largest Win: ${result.largest_win:.2f}
            Largest Loss: ${result.largest_loss:.2f}
            
            Consecutive Wins: {result.consecutive_wins}
            Consecutive Losses: {result.consecutive_losses}
            """
            
            ax6.text(0.1, 0.95, metrics_text, transform=ax6.transAxes,
                    fontsize=9, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"详细分析图表已保存: {save_path}")
            else:
                plt.show()
                
        except ImportError:
            self.logger.warning("matplotlib未安装，无法生成可视化图表")
        except Exception as e:
            self.logger.error(f"详细分析可视化失败: {str(e)}")
    
    def visualize_strategy_comparison(self, results: Dict[str, BacktestResult], save_path: str = None) -> None:
        """
        可视化多个策略的对比
        
        Args:
            results: 策略回测结果字典
            save_path: 保存路径（可选）
        """
        try:
            import matplotlib.pyplot as plt
            
            # 过滤掉None结果
            valid_results = {k: v for k, v in results.items() if v is not None}
            
            if not valid_results:
                self.logger.warning("没有有效的回测结果可供对比")
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Strategy Comparison', fontsize=16)
            
            strategy_names = list(valid_results.keys())
            
            # 1. 权益曲线对比
            ax1 = axes[0, 0]
            for name, result in valid_results.items():
                if result.equity_curve:
                    dates, equity = zip(*result.equity_curve)
                    # 归一化到初始余额
                    normalized_equity = [(e / result.initial_balance - 1) * 100 for e in equity]
                    ax1.plot(dates, normalized_equity, label=name, linewidth=2)
            ax1.set_title('Normalized Equity Curves')
            ax1.set_ylabel('Return (%)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. 收益率对比
            ax2 = axes[0, 1]
            returns = [valid_results[name].total_return * 100 for name in strategy_names]
            colors = ['green' if r > 0 else 'red' for r in returns]
            ax2.bar(strategy_names, returns, color=colors, alpha=0.7, edgecolor='black')
            ax2.set_title('Total Return Comparison')
            ax2.set_ylabel('Return (%)')
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax2.grid(True, alpha=0.3, axis='y')
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # 3. 风险指标对比
            ax3 = axes[1, 0]
            x = np.arange(len(strategy_names))
            width = 0.25
            
            sharpe_ratios = [valid_results[name].sharpe_ratio for name in strategy_names]
            sortino_ratios = [valid_results[name].sortino_ratio for name in strategy_names]
            calmar_ratios = [valid_results[name].calmar_ratio for name in strategy_names]
            
            ax3.bar(x - width, sharpe_ratios, width, label='Sharpe', alpha=0.8)
            ax3.bar(x, sortino_ratios, width, label='Sortino', alpha=0.8)
            ax3.bar(x + width, calmar_ratios, width, label='Calmar', alpha=0.8)
            
            ax3.set_title('Risk-Adjusted Return Metrics')
            ax3.set_ylabel('Ratio')
            ax3.set_xticks(x)
            ax3.set_xticklabels(strategy_names)
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # 4. 胜率和盈利因子对比
            ax4 = axes[1, 1]
            win_rates = [valid_results[name].win_rate * 100 for name in strategy_names]
            profit_factors = [valid_results[name].profit_factor for name in strategy_names]
            
            ax4_twin = ax4.twinx()
            
            bars1 = ax4.bar(x - width/2, win_rates, width, label='Win Rate (%)', 
                           color='steelblue', alpha=0.7, edgecolor='black')
            bars2 = ax4_twin.bar(x + width/2, profit_factors, width, label='Profit Factor',
                                color='orange', alpha=0.7, edgecolor='black')
            
            ax4.set_title('Win Rate & Profit Factor')
            ax4.set_ylabel('Win Rate (%)', color='steelblue')
            ax4_twin.set_ylabel('Profit Factor', color='orange')
            ax4.set_xticks(x)
            ax4.set_xticklabels(strategy_names)
            ax4.tick_params(axis='y', labelcolor='steelblue')
            ax4_twin.tick_params(axis='y', labelcolor='orange')
            ax4.grid(True, alpha=0.3, axis='y')
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # 添加图例
            lines1, labels1 = ax4.get_legend_handles_labels()
            lines2, labels2 = ax4_twin.get_legend_handles_labels()
            ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"策略对比图表已保存: {save_path}")
            else:
                plt.show()
                
        except ImportError:
            self.logger.warning("matplotlib未安装，无法生成可视化图表")
        except Exception as e:
            self.logger.error(f"策略对比可视化失败: {str(e)}")
    
    def _reset_state(self) -> None:
        """重置回测状态"""
        self.current_balance = self.config.initial_balance
        self.current_equity = self.config.initial_balance
        self.positions.clear()
        self.trades.clear()
        self.equity_curve.clear()
        self.drawdown_curve.clear()
        self.peak_equity = self.config.initial_balance
        self.max_drawdown = 0.0
        self.trade_counter = 0
    
    def _execute_signal(self, signal: Signal, market_data: MarketData) -> None:
        """执行交易信号"""
        try:
            # 检查是否有足够资金
            required_margin = self._calculate_required_margin(signal)
            if required_margin > self.current_balance:
                self.logger.warning(f"资金不足，无法执行交易: 需要{required_margin}, 可用{self.current_balance}")
                return
            
            # 应用滑点
            execution_price = self._apply_slippage(signal.entry_price, signal.direction)
            
            # 创建持仓
            position_id = f"{signal.symbol}_{self.trade_counter}"
            self.trade_counter += 1
            
            position = Position(
                position_id=position_id,
                symbol=signal.symbol,
                type=PositionType.LONG if signal.direction > 0 else PositionType.SHORT,
                volume=signal.size,
                open_price=execution_price,
                current_price=execution_price,
                sl=signal.sl,
                tp=signal.tp,
                open_time=market_data.timestamp,
                comment=f"Strategy: {signal.strategy_id}"
            )
            
            self.positions[position_id] = position
            
            # 扣除保证金
            self.current_balance -= required_margin
            
            self.logger.debug(f"开仓: {position.symbol} {position.type.name} {position.volume}手 @ {execution_price}")
            
        except Exception as e:
            self.logger.error(f"执行信号失败: {str(e)}")
    
    def _update_positions(self, market_data: MarketData) -> None:
        """更新持仓价格"""
        current_price = market_data.ohlcv['close'].iloc[-1]
        
        for position in self.positions.values():
            if position.symbol == market_data.symbol:
                position.update_current_price(current_price)
    
    def _check_stop_loss_take_profit(self, market_data: MarketData) -> None:
        """检查止损止盈"""
        current_price = market_data.ohlcv['close'].iloc[-1]
        positions_to_close = []
        
        for position_id, position in self.positions.items():
            if position.symbol != market_data.symbol:
                continue
            
            should_close = False
            close_reason = ""
            
            # 检查止损
            if position.sl > 0:
                if position.type == PositionType.LONG and current_price <= position.sl:
                    should_close = True
                    close_reason = "Stop Loss"
                elif position.type == PositionType.SHORT and current_price >= position.sl:
                    should_close = True
                    close_reason = "Stop Loss"
            
            # 检查止盈
            if not should_close and position.tp > 0:
                if position.type == PositionType.LONG and current_price >= position.tp:
                    should_close = True
                    close_reason = "Take Profit"
                elif position.type == PositionType.SHORT and current_price <= position.tp:
                    should_close = True
                    close_reason = "Take Profit"
            
            if should_close:
                positions_to_close.append((position_id, close_reason))
        
        # 关闭触发条件的持仓
        for position_id, close_reason in positions_to_close:
            self._close_position(position_id, market_data, close_reason)
    
    def _close_position(self, position_id: str, market_data: MarketData, reason: str = "") -> None:
        """关闭持仓"""
        if position_id not in self.positions:
            return
        
        position = self.positions[position_id]
        current_price = market_data.ohlcv['close'].iloc[-1]
        
        # 应用滑点
        close_price = self._apply_slippage(current_price, -position.type.value)
        
        # 计算盈亏
        if position.type == PositionType.LONG:
            profit = (close_price - position.open_price) * position.volume * 100000  # 假设每手100000基础货币
        else:
            profit = (position.open_price - close_price) * position.volume * 100000
        
        # 扣除手续费
        commission = position.volume * self.config.commission
        profit -= commission
        
        # 创建交易记录
        trade = Trade(
            trade_id=position_id,
            symbol=position.symbol,
            type=TradeType.BUY if position.type == PositionType.LONG else TradeType.SELL,
            volume=position.volume,
            open_price=position.open_price,
            close_price=close_price,
            sl=position.sl,
            tp=position.tp,
            profit=profit,
            commission=commission,
            open_time=position.open_time,
            close_time=market_data.timestamp,
            comment=f"{position.comment} | {reason}"
        )
        
        self.trades.append(trade)
        
        # 更新余额
        self.current_balance += profit
        
        # 释放保证金
        required_margin = self._calculate_required_margin_for_position(position)
        self.current_balance += required_margin
        
        # 移除持仓
        del self.positions[position_id]
        
        self.logger.debug(f"平仓: {position.symbol} {position.type.name} {position.volume}手 @ {close_price}, 盈亏: {profit:.2f}, 原因: {reason}")
    
    def _close_all_positions(self, market_data: MarketData) -> None:
        """平仓所有持仓"""
        position_ids = list(self.positions.keys())
        for position_id in position_ids:
            self._close_position(position_id, market_data, "End of Backtest")
    
    def _calculate_required_margin(self, signal: Signal) -> float:
        """计算所需保证金"""
        # 简化计算：假设每手需要1000美元保证金
        return signal.size * 1000 / self.config.leverage
    
    def _calculate_required_margin_for_position(self, position: Position) -> float:
        """计算持仓所需保证金"""
        return position.volume * 1000 / self.config.leverage
    
    def _apply_slippage(self, price: float, direction: int) -> float:
        """应用滑点"""
        if direction > 0:  # 买入
            return price + self.config.slippage
        else:  # 卖出
            return price - self.config.slippage
    
    def _update_equity_curve(self, timestamp: datetime) -> None:
        """更新权益曲线"""
        # 计算当前权益
        unrealized_pnl = sum(position.profit for position in self.positions.values())
        self.current_equity = self.current_balance + unrealized_pnl
        
        # 记录权益曲线
        self.equity_curve.append((timestamp, self.current_equity))
        
        # 更新最大回撤
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
        
        drawdown = self.peak_equity - self.current_equity
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
        
        # 记录回撤曲线
        drawdown_percent = drawdown / self.peak_equity if self.peak_equity > 0 else 0
        self.drawdown_curve.append((timestamp, drawdown_percent))
    
    def _calculate_backtest_result(self, strategy: BaseStrategy, 
                                 market_data_list: List[MarketData]) -> BacktestResult:
        """计算回测结果"""
        if not market_data_list:
            raise BacktestException("没有市场数据")
        
        start_date = market_data_list[0].timestamp
        end_date = market_data_list[-1].timestamp
        symbol = market_data_list[0].symbol
        
        # 基本统计
        total_trades = len(self.trades)
        winning_trades = sum(1 for trade in self.trades if trade.profit > 0)
        losing_trades = sum(1 for trade in self.trades if trade.profit < 0)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # 盈亏统计
        profits = [trade.profit for trade in self.trades if trade.profit > 0]
        losses = [trade.profit for trade in self.trades if trade.profit < 0]
        
        total_profit = sum(profits)
        total_loss = abs(sum(losses))
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        average_trade = sum(trade.profit for trade in self.trades) / total_trades if total_trades > 0 else 0
        average_win = sum(profits) / len(profits) if profits else 0
        average_loss = sum(losses) / len(losses) if losses else 0
        
        largest_win = max(profits) if profits else 0
        largest_loss = min(losses) if losses else 0
        
        # 连续盈亏统计
        consecutive_wins = self._calculate_consecutive_wins()
        consecutive_losses = self._calculate_consecutive_losses()
        
        # 风险指标
        total_return = (self.current_equity - self.config.initial_balance) / self.config.initial_balance
        max_drawdown_percent = self.max_drawdown / self.peak_equity if self.peak_equity > 0 else 0
        
        # 夏普比率和索提诺比率
        returns = self._calculate_returns()
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        
        # 卡玛比率
        calmar_ratio = total_return / max_drawdown_percent if max_drawdown_percent > 0 else 0
        
        return BacktestResult(
            strategy_name=strategy.config.name,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            initial_balance=self.config.initial_balance,
            final_balance=self.current_equity,
            total_return=total_return,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown=self.max_drawdown,
            max_drawdown_percent=max_drawdown_percent,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            average_trade=average_trade,
            average_win=average_win,
            average_loss=average_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses,
            trades=self.trades.copy(),
            equity_curve=self.equity_curve.copy(),
            drawdown_curve=self.drawdown_curve.copy()
        )
    
    def _calculate_consecutive_wins(self) -> int:
        """计算最大连续盈利次数"""
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in self.trades:
            if trade.profit > 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_consecutive_losses(self) -> int:
        """计算最大连续亏损次数"""
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in self.trades:
            if trade.profit < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_returns(self) -> List[float]:
        """计算收益率序列"""
        if len(self.equity_curve) < 2:
            return []
        
        returns = []
        for i in range(1, len(self.equity_curve)):
            prev_equity = self.equity_curve[i-1][1]
            curr_equity = self.equity_curve[i][1]
            
            if prev_equity > 0:
                return_rate = (curr_equity - prev_equity) / prev_equity
                returns.append(return_rate)
        
        return returns
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        if not returns:
            return 0.0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate / 252  # 假设252个交易日
        
        if len(excess_returns) == 0 or np.std(excess_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def _calculate_sortino_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """计算索提诺比率"""
        if not returns:
            return 0.0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate / 252
        
        # 只考虑负收益的标准差
        negative_returns = excess_returns[excess_returns < 0]
        
        if len(negative_returns) == 0:
            return float('inf')
        
        downside_deviation = np.std(negative_returns)
        
        if downside_deviation == 0:
            return 0.0
        
        return np.mean(excess_returns) / downside_deviation * np.sqrt(252)
    
    def export_results(self, result: BacktestResult, filepath: str, format: str = 'json') -> bool:
        """
        导出回测结果到文件
        
        Args:
            result: 回测结果
            filepath: 文件路径
            format: 导出格式 ('json', 'csv', 'excel', 'html')
            
        Returns:
            是否成功
        """
        try:
            if format == 'json':
                return self._export_to_json(result, filepath)
            elif format == 'csv':
                return self._export_to_csv(result, filepath)
            elif format == 'excel':
                return self._export_to_excel(result, filepath)
            elif format == 'html':
                return self._export_to_html(result, filepath)
            else:
                self.logger.error(f"不支持的导出格式: {format}")
                return False
                
        except Exception as e:
            self.logger.error(f"导出回测结果失败: {str(e)}")
            return False
    
    def _export_to_json(self, result: BacktestResult, filepath: str) -> bool:
        """导出为JSON格式"""
        import json
        
        data = result.to_dict()
        
        # 转换交易记录
        data['trades'] = [
            {
                'trade_id': t.trade_id,
                'symbol': t.symbol,
                'type': t.type.name,
                'volume': t.volume,
                'open_price': t.open_price,
                'close_price': t.close_price,
                'profit': t.profit,
                'open_time': t.open_time.isoformat(),
                'close_time': t.close_time.isoformat() if t.close_time else None
            }
            for t in result.trades
        ]
        
        # 转换权益曲线
        data['equity_curve'] = [
            {'timestamp': ts.isoformat(), 'equity': eq}
            for ts, eq in result.equity_curve
        ]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"回测结果已导出为JSON: {filepath}")
        return True
    
    def _export_to_csv(self, result: BacktestResult, filepath: str) -> bool:
        """导出为CSV格式"""
        # 导出交易记录
        trades_data = []
        for t in result.trades:
            trades_data.append({
                'trade_id': t.trade_id,
                'symbol': t.symbol,
                'type': t.type.name,
                'volume': t.volume,
                'open_price': t.open_price,
                'close_price': t.close_price,
                'profit': t.profit,
                'commission': t.commission,
                'swap': t.swap,
                'open_time': t.open_time,
                'close_time': t.close_time
            })
        
        df_trades = pd.DataFrame(trades_data)
        
        # 导出汇总指标
        summary_data = {
            'Metric': [
                'Strategy Name', 'Symbol', 'Initial Balance', 'Final Balance',
                'Total Return', 'Total Trades', 'Win Rate', 'Profit Factor',
                'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown', 'Max Drawdown %'
            ],
            'Value': [
                result.strategy_name, result.symbol, result.initial_balance, result.final_balance,
                f"{result.total_return*100:.2f}%", result.total_trades, f"{result.win_rate*100:.2f}%",
                f"{result.profit_factor:.2f}", f"{result.sharpe_ratio:.2f}", f"{result.sortino_ratio:.2f}",
                f"${result.max_drawdown:.2f}", f"{result.max_drawdown_percent*100:.2f}%"
            ]
        }
        df_summary = pd.DataFrame(summary_data)
        
        # 保存到CSV
        base_path = filepath.rsplit('.', 1)[0]
        df_trades.to_csv(f"{base_path}_trades.csv", index=False)
        df_summary.to_csv(f"{base_path}_summary.csv", index=False)
        
        self.logger.info(f"回测结果已导出为CSV: {base_path}_*.csv")
        return True
    
    def _export_to_excel(self, result: BacktestResult, filepath: str) -> bool:
        """导出为Excel格式"""
        try:
            # 创建Excel写入器
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # 汇总页
                summary_data = {
                    'Metric': [
                        'Strategy Name', 'Symbol', 'Start Date', 'End Date',
                        'Initial Balance', 'Final Balance', 'Total Return',
                        'Total Trades', 'Winning Trades', 'Losing Trades', 'Win Rate',
                        'Profit Factor', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio',
                        'Max Drawdown', 'Max Drawdown %', 'Average Trade', 'Average Win',
                        'Average Loss', 'Largest Win', 'Largest Loss',
                        'Consecutive Wins', 'Consecutive Losses'
                    ],
                    'Value': [
                        result.strategy_name, result.symbol,
                        result.start_date.strftime('%Y-%m-%d'), result.end_date.strftime('%Y-%m-%d'),
                        f"${result.initial_balance:.2f}", f"${result.final_balance:.2f}",
                        f"{result.total_return*100:.2f}%",
                        result.total_trades, result.winning_trades, result.losing_trades,
                        f"{result.win_rate*100:.2f}%", f"{result.profit_factor:.2f}",
                        f"{result.sharpe_ratio:.2f}", f"{result.sortino_ratio:.2f}",
                        f"{result.calmar_ratio:.2f}", f"${result.max_drawdown:.2f}",
                        f"{result.max_drawdown_percent*100:.2f}%", f"${result.average_trade:.2f}",
                        f"${result.average_win:.2f}", f"${result.average_loss:.2f}",
                        f"${result.largest_win:.2f}", f"${result.largest_loss:.2f}",
                        result.consecutive_wins, result.consecutive_losses
                    ]
                }
                df_summary = pd.DataFrame(summary_data)
                df_summary.to_excel(writer, sheet_name='Summary', index=False)
                
                # 交易记录页
                trades_data = []
                for t in result.trades:
                    trades_data.append({
                        'Trade ID': t.trade_id,
                        'Symbol': t.symbol,
                        'Type': t.type.name,
                        'Volume': t.volume,
                        'Open Price': t.open_price,
                        'Close Price': t.close_price,
                        'Profit': t.profit,
                        'Commission': t.commission,
                        'Swap': t.swap,
                        'Open Time': t.open_time,
                        'Close Time': t.close_time,
                        'Duration (h)': t.get_duration() if t.get_duration() else 0
                    })
                df_trades = pd.DataFrame(trades_data)
                df_trades.to_excel(writer, sheet_name='Trades', index=False)
                
                # 权益曲线页
                equity_data = []
                for ts, eq in result.equity_curve:
                    equity_data.append({
                        'Timestamp': ts,
                        'Equity': eq,
                        'Return': (eq - result.initial_balance) / result.initial_balance * 100
                    })
                df_equity = pd.DataFrame(equity_data)
                df_equity.to_excel(writer, sheet_name='Equity Curve', index=False)
                
                # 回撤曲线页
                drawdown_data = []
                for ts, dd in result.drawdown_curve:
                    drawdown_data.append({
                        'Timestamp': ts,
                        'Drawdown': dd,
                        'Drawdown %': dd * 100
                    })
                df_drawdown = pd.DataFrame(drawdown_data)
                df_drawdown.to_excel(writer, sheet_name='Drawdown', index=False)
            
            self.logger.info(f"回测结果已导出为Excel: {filepath}")
            return True
            
        except ImportError:
            self.logger.error("openpyxl未安装，无法导出Excel格式")
            return False
    
    def _export_to_html(self, result: BacktestResult, filepath: str) -> bool:
        """导出为HTML格式"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Results - {result.strategy_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
                h2 {{ color: #666; margin-top: 30px; }}
                .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0; }}
                .metric-card {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; border-left: 4px solid #4CAF50; }}
                .metric-label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #333; margin-top: 5px; }}
                .positive {{ color: #4CAF50; }}
                .negative {{ color: #f44336; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Backtest Results: {result.strategy_name}</h1>
                <p><strong>Symbol:</strong> {result.symbol} | <strong>Period:</strong> {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}</p>
                
                <h2>Performance Summary</h2>
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-label">Total Return</div>
                        <div class="metric-value {'positive' if result.total_return > 0 else 'negative'}">{result.total_return*100:.2f}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Final Balance</div>
                        <div class="metric-value">${result.final_balance:.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Total Trades</div>
                        <div class="metric-value">{result.total_trades}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Win Rate</div>
                        <div class="metric-value">{result.win_rate*100:.2f}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Profit Factor</div>
                        <div class="metric-value">{result.profit_factor:.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Sharpe Ratio</div>
                        <div class="metric-value">{result.sharpe_ratio:.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Max Drawdown</div>
                        <div class="metric-value negative">{result.max_drawdown_percent*100:.2f}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Sortino Ratio</div>
                        <div class="metric-value">{result.sortino_ratio:.2f}</div>
                    </div>
                </div>
                
                <h2>Trade Statistics</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Winning Trades</td>
                        <td class="positive">{result.winning_trades}</td>
                    </tr>
                    <tr>
                        <td>Losing Trades</td>
                        <td class="negative">{result.losing_trades}</td>
                    </tr>
                    <tr>
                        <td>Average Trade</td>
                        <td>${result.average_trade:.2f}</td>
                    </tr>
                    <tr>
                        <td>Average Win</td>
                        <td class="positive">${result.average_win:.2f}</td>
                    </tr>
                    <tr>
                        <td>Average Loss</td>
                        <td class="negative">${result.average_loss:.2f}</td>
                    </tr>
                    <tr>
                        <td>Largest Win</td>
                        <td class="positive">${result.largest_win:.2f}</td>
                    </tr>
                    <tr>
                        <td>Largest Loss</td>
                        <td class="negative">${result.largest_loss:.2f}</td>
                    </tr>
                    <tr>
                        <td>Max Consecutive Wins</td>
                        <td>{result.consecutive_wins}</td>
                    </tr>
                    <tr>
                        <td>Max Consecutive Losses</td>
                        <td>{result.consecutive_losses}</td>
                    </tr>
                </table>
                
                <div class="footer">
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"回测结果已导出为HTML: {filepath}")
        return True