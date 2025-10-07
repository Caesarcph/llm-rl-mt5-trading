"""
蒙特卡洛模拟器
用于风险分析、压力测试和情景分析
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

from src.core.models import Trade, RiskMetrics
from src.strategies.backtest import BacktestResult


@dataclass
class MonteCarloConfig:
    """蒙特卡洛配置"""
    n_simulations: int = 1000  # 模拟次数
    confidence_level: float = 0.95  # 置信水平
    random_seed: Optional[int] = None  # 随机种子
    parallel: bool = True  # 是否并行计算
    max_workers: int = 4  # 最大工作进程数
    
    def validate(self) -> bool:
        """验证配置"""
        if self.n_simulations <= 0:
            raise ValueError("模拟次数必须大于0")
        if not 0 < self.confidence_level < 1:
            raise ValueError("置信水平必须在0-1之间")
        if self.max_workers <= 0:
            raise ValueError("工作进程数必须大于0")
        return True


@dataclass
class MonteCarloResult:
    """蒙特卡洛模拟结果"""
    simulation_id: str
    n_simulations: int
    confidence_level: float
    
    # 收益分布
    mean_return: float
    median_return: float
    std_return: float
    min_return: float
    max_return: float
    
    # 风险指标
    var: float  # Value at Risk
    cvar: float  # Conditional VaR (Expected Shortfall)
    max_drawdown_mean: float
    max_drawdown_worst: float
    
    # 概率分析
    prob_profit: float  # 盈利概率
    prob_loss: float  # 亏损概率
    prob_ruin: float  # 破产概率
    
    # 分位数
    percentiles: Dict[float, float] = field(default_factory=dict)
    
    # 详细数据
    all_returns: List[float] = field(default_factory=list)
    all_drawdowns: List[float] = field(default_factory=list)
    equity_curves: List[List[float]] = field(default_factory=list)
    
    # 元数据
    generated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'simulation_id': self.simulation_id,
            'n_simulations': self.n_simulations,
            'confidence_level': self.confidence_level,
            'mean_return': float(self.mean_return),
            'median_return': float(self.median_return),
            'std_return': float(self.std_return),
            'min_return': float(self.min_return),
            'max_return': float(self.max_return),
            'var': float(self.var),
            'cvar': float(self.cvar),
            'max_drawdown_mean': float(self.max_drawdown_mean),
            'max_drawdown_worst': float(self.max_drawdown_worst),
            'prob_profit': float(self.prob_profit),
            'prob_loss': float(self.prob_loss),
            'prob_ruin': float(self.prob_ruin),
            'percentiles': {str(k): float(v) for k, v in self.percentiles.items()},
            'generated_at': self.generated_at.isoformat()
        }


@dataclass
class StressTestScenario:
    """压力测试场景"""
    name: str
    description: str
    return_shock: float  # 收益冲击 (例如 -0.20 表示-20%)
    volatility_multiplier: float  # 波动率倍数
    win_rate_adjustment: float  # 胜率调整
    drawdown_multiplier: float  # 回撤倍数


@dataclass
class StressTestResult:
    """压力测试结果"""
    scenario_name: str
    original_return: float
    stressed_return: float
    return_impact: float
    original_drawdown: float
    stressed_drawdown: float
    drawdown_impact: float
    survival_probability: float
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'scenario_name': self.scenario_name,
            'original_return': float(self.original_return),
            'stressed_return': float(self.stressed_return),
            'return_impact': float(self.return_impact),
            'original_drawdown': float(self.original_drawdown),
            'stressed_drawdown': float(self.stressed_drawdown),
            'drawdown_impact': float(self.drawdown_impact),
            'survival_probability': float(self.survival_probability)
        }


class MonteCarloSimulator:
    """
    蒙特卡洛模拟器
    基于历史交易数据进行风险分析和压力测试
    """
    
    def __init__(self, config: MonteCarloConfig = None):
        """
        初始化模拟器
        
        Args:
            config: 蒙特卡洛配置
        """
        self.config = config or MonteCarloConfig()
        self.config.validate()
        
        self.logger = logging.getLogger(__name__)
        
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
        
        self.logger.info(f"蒙特卡洛模拟器初始化完成: {self.config.n_simulations}次模拟")
    
    def simulate_from_trades(
        self,
        trades: List[Trade],
        initial_balance: float = 10000.0
    ) -> MonteCarloResult:
        """
        基于历史交易进行蒙特卡洛模拟
        
        Args:
            trades: 历史交易列表
            initial_balance: 初始余额
            
        Returns:
            蒙特卡洛结果
        """
        if not trades:
            raise ValueError("交易列表不能为空")
        
        self.logger.info(f"开始蒙特卡洛模拟: {len(trades)}笔交易, {self.config.n_simulations}次模拟")
        
        # 提取交易收益
        returns = np.array([t.profit for t in trades])
        
        # 运行模拟
        if self.config.parallel:
            simulation_results = self._run_parallel_simulations(returns, initial_balance)
        else:
            simulation_results = self._run_sequential_simulations(returns, initial_balance)
        
        # 分析结果
        result = self._analyze_simulation_results(simulation_results, initial_balance)
        
        self.logger.info(f"模拟完成: 平均收益={result.mean_return:.2f}, VaR={result.var:.2f}")
        
        return result
    
    def simulate_from_backtest(
        self,
        backtest_result: BacktestResult
    ) -> MonteCarloResult:
        """
        基于回测结果进行蒙特卡洛模拟
        
        Args:
            backtest_result: 回测结果
            
        Returns:
            蒙特卡洛结果
        """
        return self.simulate_from_trades(
            backtest_result.trades,
            backtest_result.initial_balance
        )
    
    def _run_sequential_simulations(
        self,
        returns: np.ndarray,
        initial_balance: float
    ) -> List[Dict[str, Any]]:
        """顺序运行模拟"""
        results = []
        
        for i in range(self.config.n_simulations):
            result = self._run_single_simulation(returns, initial_balance, i)
            results.append(result)
            
            if (i + 1) % 100 == 0:
                self.logger.debug(f"完成 {i + 1}/{self.config.n_simulations} 次模拟")
        
        return results
    
    def _run_parallel_simulations(
        self,
        returns: np.ndarray,
        initial_balance: float
    ) -> List[Dict[str, Any]]:
        """并行运行模拟"""
        results = []
        
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(
                    self._run_single_simulation,
                    returns,
                    initial_balance,
                    i
                ): i for i in range(self.config.n_simulations)
            }
            
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                completed += 1
                
                if completed % 100 == 0:
                    self.logger.debug(f"完成 {completed}/{self.config.n_simulations} 次模拟")
        
        return results
    
    def _run_single_simulation(
        self,
        returns: np.ndarray,
        initial_balance: float,
        simulation_id: int
    ) -> Dict[str, Any]:
        """
        运行单次模拟
        
        使用bootstrap重采样方法
        """
        n_trades = len(returns)
        
        # Bootstrap重采样
        sampled_returns = np.random.choice(returns, size=n_trades, replace=True)
        
        # 计算权益曲线
        equity_curve = [initial_balance]
        balance = initial_balance
        
        for ret in sampled_returns:
            balance += ret
            equity_curve.append(balance)
        
        # 计算指标
        final_balance = equity_curve[-1]
        total_return = (final_balance - initial_balance) / initial_balance
        
        # 计算回撤
        equity_array = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        return {
            'simulation_id': simulation_id,
            'final_balance': final_balance,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'equity_curve': equity_curve
        }
    
    def _analyze_simulation_results(
        self,
        simulation_results: List[Dict[str, Any]],
        initial_balance: float
    ) -> MonteCarloResult:
        """分析模拟结果"""
        # 提取数据
        returns = np.array([r['total_return'] for r in simulation_results])
        drawdowns = np.array([r['max_drawdown'] for r in simulation_results])
        equity_curves = [r['equity_curve'] for r in simulation_results]
        
        # 基本统计
        mean_return = np.mean(returns)
        median_return = np.median(returns)
        std_return = np.std(returns)
        min_return = np.min(returns)
        max_return = np.max(returns)
        
        # 风险指标
        var = self._calculate_var(returns, self.config.confidence_level)
        cvar = self._calculate_cvar(returns, self.config.confidence_level)
        
        max_drawdown_mean = np.mean(drawdowns)
        max_drawdown_worst = np.min(drawdowns)
        
        # 概率分析
        prob_profit = np.sum(returns > 0) / len(returns)
        prob_loss = np.sum(returns < 0) / len(returns)
        
        # 破产概率 (余额低于初始余额的50%)
        ruin_threshold = -0.5
        prob_ruin = np.sum(returns < ruin_threshold) / len(returns)
        
        # 分位数
        percentiles = {
            0.01: np.percentile(returns, 1),
            0.05: np.percentile(returns, 5),
            0.10: np.percentile(returns, 10),
            0.25: np.percentile(returns, 25),
            0.50: np.percentile(returns, 50),
            0.75: np.percentile(returns, 75),
            0.90: np.percentile(returns, 90),
            0.95: np.percentile(returns, 95),
            0.99: np.percentile(returns, 99)
        }
        
        return MonteCarloResult(
            simulation_id=f"mc_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            n_simulations=self.config.n_simulations,
            confidence_level=self.config.confidence_level,
            mean_return=mean_return,
            median_return=median_return,
            std_return=std_return,
            min_return=min_return,
            max_return=max_return,
            var=var,
            cvar=cvar,
            max_drawdown_mean=max_drawdown_mean,
            max_drawdown_worst=max_drawdown_worst,
            prob_profit=prob_profit,
            prob_loss=prob_loss,
            prob_ruin=prob_ruin,
            percentiles=percentiles,
            all_returns=returns.tolist(),
            all_drawdowns=drawdowns.tolist(),
            equity_curves=equity_curves
        )
    
    def _calculate_var(self, returns: np.ndarray, confidence_level: float) -> float:
        """
        计算VaR (Value at Risk)
        
        Args:
            returns: 收益数组
            confidence_level: 置信水平
            
        Returns:
            VaR值
        """
        percentile = (1 - confidence_level) * 100
        return np.percentile(returns, percentile)
    
    def _calculate_cvar(self, returns: np.ndarray, confidence_level: float) -> float:
        """
        计算CVaR (Conditional VaR / Expected Shortfall)
        
        Args:
            returns: 收益数组
            confidence_level: 置信水平
            
        Returns:
            CVaR值
        """
        var = self._calculate_var(returns, confidence_level)
        # CVaR是所有低于VaR的收益的平均值
        tail_returns = returns[returns <= var]
        return np.mean(tail_returns) if len(tail_returns) > 0 else var
    
    def stress_test(
        self,
        trades: List[Trade],
        scenarios: List[StressTestScenario] = None,
        initial_balance: float = 10000.0
    ) -> List[StressTestResult]:
        """
        压力测试
        
        Args:
            trades: 历史交易列表
            scenarios: 压力测试场景列表
            initial_balance: 初始余额
            
        Returns:
            压力测试结果列表
        """
        if not trades:
            raise ValueError("交易列表不能为空")
        
        # 使用默认场景
        if scenarios is None:
            scenarios = self._get_default_stress_scenarios()
        
        self.logger.info(f"开始压力测试: {len(scenarios)}个场景")
        
        # 计算原始指标
        returns = np.array([t.profit for t in trades])
        original_return = np.sum(returns) / initial_balance
        original_drawdown = self._calculate_max_drawdown_from_returns(returns, initial_balance)
        
        results = []
        
        for scenario in scenarios:
            self.logger.debug(f"测试场景: {scenario.name}")
            
            # 应用压力场景
            stressed_returns = self._apply_stress_scenario(returns, scenario)
            
            # 计算压力后指标
            stressed_return = np.sum(stressed_returns) / initial_balance
            stressed_drawdown = self._calculate_max_drawdown_from_returns(
                stressed_returns, initial_balance
            )
            
            # 计算生存概率
            survival_prob = self._calculate_survival_probability(
                stressed_returns, initial_balance
            )
            
            result = StressTestResult(
                scenario_name=scenario.name,
                original_return=original_return,
                stressed_return=stressed_return,
                return_impact=stressed_return - original_return,
                original_drawdown=original_drawdown,
                stressed_drawdown=stressed_drawdown,
                drawdown_impact=stressed_drawdown - original_drawdown,
                survival_probability=survival_prob
            )
            
            results.append(result)
            
            self.logger.debug(
                f"场景 {scenario.name}: 收益影响={result.return_impact:.2%}, "
                f"生存概率={result.survival_probability:.2%}"
            )
        
        self.logger.info("压力测试完成")
        
        return results
    
    def _get_default_stress_scenarios(self) -> List[StressTestScenario]:
        """获取默认压力测试场景"""
        return [
            StressTestScenario(
                name="温和衰退",
                description="市场温和下跌，波动率小幅上升",
                return_shock=-0.10,
                volatility_multiplier=1.2,
                win_rate_adjustment=-0.05,
                drawdown_multiplier=1.3
            ),
            StressTestScenario(
                name="严重衰退",
                description="市场大幅下跌，波动率显著上升",
                return_shock=-0.25,
                volatility_multiplier=1.5,
                win_rate_adjustment=-0.10,
                drawdown_multiplier=1.8
            ),
            StressTestScenario(
                name="市场崩盘",
                description="极端市场崩盘情景",
                return_shock=-0.40,
                volatility_multiplier=2.0,
                win_rate_adjustment=-0.15,
                drawdown_multiplier=2.5
            ),
            StressTestScenario(
                name="高波动",
                description="收益不变但波动率大幅上升",
                return_shock=0.0,
                volatility_multiplier=2.0,
                win_rate_adjustment=0.0,
                drawdown_multiplier=1.5
            ),
            StressTestScenario(
                name="胜率下降",
                description="策略有效性降低",
                return_shock=-0.15,
                volatility_multiplier=1.0,
                win_rate_adjustment=-0.20,
                drawdown_multiplier=1.4
            )
        ]
    
    def _apply_stress_scenario(
        self,
        returns: np.ndarray,
        scenario: StressTestScenario
    ) -> np.ndarray:
        """应用压力场景到收益序列"""
        stressed_returns = returns.copy()
        
        # 应用收益冲击
        stressed_returns = stressed_returns * (1 + scenario.return_shock)
        
        # 应用波动率倍数
        mean_return = np.mean(stressed_returns)
        stressed_returns = mean_return + (stressed_returns - mean_return) * scenario.volatility_multiplier
        
        # 应用胜率调整
        if scenario.win_rate_adjustment != 0:
            # 将一些盈利交易转为亏损
            winning_indices = np.where(stressed_returns > 0)[0]
            n_to_flip = int(len(winning_indices) * abs(scenario.win_rate_adjustment))
            
            if n_to_flip > 0:
                flip_indices = np.random.choice(winning_indices, size=n_to_flip, replace=False)
                stressed_returns[flip_indices] *= -1
        
        return stressed_returns
    
    def _calculate_max_drawdown_from_returns(
        self,
        returns: np.ndarray,
        initial_balance: float
    ) -> float:
        """从收益序列计算最大回撤"""
        equity_curve = np.concatenate([[initial_balance], initial_balance + np.cumsum(returns)])
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        return np.min(drawdown)
    
    def _calculate_survival_probability(
        self,
        returns: np.ndarray,
        initial_balance: float,
        ruin_threshold: float = 0.5
    ) -> float:
        """
        计算生存概率
        
        Args:
            returns: 收益序列
            initial_balance: 初始余额
            ruin_threshold: 破产阈值（余额低于初始余额的比例）
            
        Returns:
            生存概率
        """
        # 运行多次模拟
        n_simulations = 1000
        survival_count = 0
        
        for _ in range(n_simulations):
            # 随机打乱收益顺序
            shuffled_returns = np.random.permutation(returns)
            
            # 计算权益曲线
            balance = initial_balance
            survived = True
            
            for ret in shuffled_returns:
                balance += ret
                if balance < initial_balance * ruin_threshold:
                    survived = False
                    break
            
            if survived:
                survival_count += 1
        
        return survival_count / n_simulations
    
    def scenario_analysis(
        self,
        trades: List[Trade],
        initial_balance: float = 10000.0,
        scenarios: Dict[str, Dict[str, float]] = None
    ) -> Dict[str, MonteCarloResult]:
        """
        情景分析
        
        Args:
            trades: 历史交易列表
            initial_balance: 初始余额
            scenarios: 自定义场景字典
            
        Returns:
            场景分析结果字典
        """
        if not trades:
            raise ValueError("交易列表不能为空")
        
        # 使用默认场景
        if scenarios is None:
            scenarios = {
                '基准情景': {'return_multiplier': 1.0, 'volatility_multiplier': 1.0},
                '乐观情景': {'return_multiplier': 1.2, 'volatility_multiplier': 0.8},
                '悲观情景': {'return_multiplier': 0.7, 'volatility_multiplier': 1.3}
            }
        
        self.logger.info(f"开始情景分析: {len(scenarios)}个场景")
        
        results = {}
        returns = np.array([t.profit for t in trades])
        
        for scenario_name, params in scenarios.items():
            self.logger.debug(f"分析场景: {scenario_name}")
            
            # 调整收益
            adjusted_returns = returns * params.get('return_multiplier', 1.0)
            
            # 调整波动率
            mean_return = np.mean(adjusted_returns)
            volatility_mult = params.get('volatility_multiplier', 1.0)
            adjusted_returns = mean_return + (adjusted_returns - mean_return) * volatility_mult
            
            # 运行模拟
            if self.config.parallel:
                simulation_results = self._run_parallel_simulations(adjusted_returns, initial_balance)
            else:
                simulation_results = self._run_sequential_simulations(adjusted_returns, initial_balance)
            
            # 分析结果
            result = self._analyze_simulation_results(simulation_results, initial_balance)
            result.simulation_id = f"{scenario_name}_{result.simulation_id}"
            
            results[scenario_name] = result
            
            self.logger.debug(
                f"场景 {scenario_name}: 平均收益={result.mean_return:.2%}, "
                f"VaR={result.var:.2%}"
            )
        
        self.logger.info("情景分析完成")
        
        return results
    
    def calculate_risk_metrics(
        self,
        trades: List[Trade],
        initial_balance: float = 10000.0
    ) -> RiskMetrics:
        """
        计算风险指标
        
        Args:
            trades: 历史交易列表
            initial_balance: 初始余额
            
        Returns:
            风险指标
        """
        if not trades:
            raise ValueError("交易列表不能为空")
        
        returns = np.array([t.profit for t in trades])
        
        # 运行蒙特卡洛模拟
        mc_result = self.simulate_from_trades(trades, initial_balance)
        
        # 计算其他指标
        winning_trades = [t for t in trades if t.profit > 0]
        losing_trades = [t for t in trades if t.profit < 0]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        gross_profit = sum(t.profit for t in winning_trades)
        gross_loss = abs(sum(t.profit for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # 夏普比率
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # 索提诺比率
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            downside_std = np.std(negative_returns)
            sortino_ratio = np.mean(returns) / downside_std if downside_std > 0 else 0
        else:
            sortino_ratio = 0
        
        # 卡玛比率
        max_dd = abs(mc_result.max_drawdown_mean)
        calmar_ratio = mc_result.mean_return / max_dd if max_dd > 0 else 0
        
        return RiskMetrics(
            var_1d=mc_result.var,
            var_5d=mc_result.var * np.sqrt(5),  # 简化的5日VaR
            max_drawdown=mc_result.max_drawdown_worst,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor
        )
    
    def export_results(
        self,
        result: MonteCarloResult,
        filepath: str,
        format: str = 'json'
    ) -> bool:
        """
        导出结果
        
        Args:
            result: 蒙特卡洛结果
            filepath: 文件路径
            format: 导出格式 ('json', 'csv')
            
        Returns:
            是否成功
        """
        try:
            if format == 'json':
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
            
            elif format == 'csv':
                # 导出收益分布
                df = pd.DataFrame({
                    'simulation': range(len(result.all_returns)),
                    'return': result.all_returns,
                    'drawdown': result.all_drawdowns
                })
                df.to_csv(filepath, index=False)
            
            else:
                raise ValueError(f"不支持的格式: {format}")
            
            self.logger.info(f"结果已导出: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"导出结果失败: {str(e)}")
            return False
