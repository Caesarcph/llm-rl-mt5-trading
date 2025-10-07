"""
参数优化器模块
使用Optuna进行贝叶斯优化和遗传算法优化策略参数
"""

import logging
import optuna
from optuna.samplers import TPESampler, CmaEsSampler, NSGAIISampler
from optuna.pruners import MedianPruner
from typing import Dict, List, Tuple, Callable, Any, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
from datetime import datetime

from src.core.models import MarketData
from src.strategies.base_strategies import BaseStrategy
from src.strategies.backtest import BacktestEngine, BacktestConfig, BacktestResult


@dataclass
class OptimizationConfig:
    """优化配置"""
    n_trials: int = 100  # 优化试验次数
    timeout: int = 3600  # 超时时间（秒）
    n_jobs: int = 1  # 并行任务数
    sampler: str = "tpe"  # 采样器类型: tpe, cmaes, random
    pruner: bool = True  # 是否使用剪枝
    direction: str = "maximize"  # 优化方向: maximize, minimize
    
    # 多目标优化
    multi_objective: bool = False
    objectives: List[str] = None  # 例如: ['total_return', 'sharpe_ratio']
    
    # 早停
    early_stopping: bool = False
    early_stopping_rounds: int = 20
    
    def __post_init__(self):
        if self.objectives is None:
            self.objectives = ['total_return']


@dataclass
class ParameterSpace:
    """参数空间定义"""
    name: str
    type: str  # int, float, categorical
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    step: Optional[float] = None
    log: bool = False  # 是否使用对数尺度
    
    def suggest(self, trial: optuna.Trial) -> Any:
        """从试验中建议参数值"""
        if self.type == "int":
            if self.log:
                return trial.suggest_int(self.name, int(self.low), int(self.high), log=True)
            elif self.step:
                return trial.suggest_int(self.name, int(self.low), int(self.high), step=int(self.step))
            else:
                return trial.suggest_int(self.name, int(self.low), int(self.high))
        
        elif self.type == "float":
            if self.log:
                return trial.suggest_float(self.name, self.low, self.high, log=True)
            elif self.step:
                return trial.suggest_float(self.name, self.low, self.high, step=self.step)
            else:
                return trial.suggest_float(self.name, self.low, self.high)
        
        elif self.type == "categorical":
            return trial.suggest_categorical(self.name, self.choices)
        
        else:
            raise ValueError(f"不支持的参数类型: {self.type}")


@dataclass
class OptimizationResult:
    """优化结果"""
    best_params: Dict[str, Any]
    best_value: float
    best_trial: int
    n_trials: int
    optimization_history: List[Tuple[int, float]]
    param_importances: Dict[str, float]
    study: optuna.Study
    
    # 多目标优化结果
    pareto_front: Optional[List[Dict[str, Any]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'best_params': self.best_params,
            'best_value': self.best_value,
            'best_trial': self.best_trial,
            'n_trials': self.n_trials,
            'param_importances': self.param_importances,
            'pareto_front': self.pareto_front
        }


class ParameterOptimizer:
    """
    参数优化器
    集成Optuna库，支持贝叶斯优化、遗传算法和多目标优化
    """
    
    def __init__(self, config: OptimizationConfig = None):
        """
        初始化参数优化器
        
        Args:
            config: 优化配置
        """
        self.config = config or OptimizationConfig()
        self.logger = logging.getLogger(__name__)
        
        # 创建采样器
        self.sampler = self._create_sampler()
        
        # 创建剪枝器
        self.pruner = MedianPruner() if self.config.pruner else None
        
        # 优化历史
        self.studies: Dict[str, optuna.Study] = {}
        
        self.logger.info(f"参数优化器初始化完成: {self.config.sampler} sampler")
    
    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """创建采样器"""
        if self.config.sampler == "tpe":
            return TPESampler(seed=42)
        elif self.config.sampler == "cmaes":
            return CmaEsSampler(seed=42)
        elif self.config.sampler == "random":
            return optuna.samplers.RandomSampler(seed=42)
        else:
            self.logger.warning(f"未知采样器类型: {self.config.sampler}, 使用TPE")
            return TPESampler(seed=42)
    
    def optimize_strategy(
        self,
        strategy_class: type,
        param_space: List[ParameterSpace],
        market_data_list: List[MarketData],
        backtest_config: BacktestConfig,
        objective_metric: str = "total_return",
        study_name: str = None
    ) -> OptimizationResult:
        """
        优化策略参数
        
        Args:
            strategy_class: 策略类
            param_space: 参数空间定义
            market_data_list: 市场数据列表
            backtest_config: 回测配置
            objective_metric: 优化目标指标
            study_name: 研究名称
            
        Returns:
            优化结果
        """
        if study_name is None:
            study_name = f"optimize_{strategy_class.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"开始优化策略: {study_name}")
        self.logger.info(f"参数空间: {[p.name for p in param_space]}")
        self.logger.info(f"优化目标: {objective_metric}")
        
        # 创建目标函数
        def objective(trial: optuna.Trial) -> float:
            # 从参数空间中采样参数
            params = {}
            for param in param_space:
                params[param.name] = param.suggest(trial)
            
            try:
                # 创建策略实例
                strategy = strategy_class(params)
                
                # 运行回测
                engine = BacktestEngine(backtest_config)
                result = engine.run_backtest(strategy, market_data_list)
                
                # 获取目标指标值
                if hasattr(result, objective_metric):
                    value = getattr(result, objective_metric)
                else:
                    raise ValueError(f"未知的目标指标: {objective_metric}")
                
                # 记录中间结果用于剪枝
                trial.set_user_attr("backtest_result", result.to_dict())
                
                return value
                
            except Exception as e:
                self.logger.error(f"试验{trial.number}失败: {str(e)}")
                return float('-inf') if self.config.direction == "maximize" else float('inf')
        
        # 创建研究
        study = optuna.create_study(
            study_name=study_name,
            direction=self.config.direction,
            sampler=self.sampler,
            pruner=self.pruner
        )
        
        # 运行优化
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs,
            show_progress_bar=True
        )
        
        # 保存研究
        self.studies[study_name] = study
        
        # 计算参数重要性
        try:
            param_importances = optuna.importance.get_param_importances(study)
        except Exception as e:
            self.logger.warning(f"无法计算参数重要性: {str(e)}")
            param_importances = {}
        
        # 构建优化历史
        optimization_history = [
            (trial.number, trial.value) 
            for trial in study.trials 
            if trial.value is not None
        ]
        
        # 创建结果
        result = OptimizationResult(
            best_params=study.best_params,
            best_value=study.best_value,
            best_trial=study.best_trial.number,
            n_trials=len(study.trials),
            optimization_history=optimization_history,
            param_importances=param_importances,
            study=study
        )
        
        self.logger.info(f"优化完成: 最佳{objective_metric}={result.best_value:.4f}")
        self.logger.info(f"最佳参数: {result.best_params}")
        
        return result
    
    def multi_objective_optimize(
        self,
        strategy_class: type,
        param_space: List[ParameterSpace],
        market_data_list: List[MarketData],
        backtest_config: BacktestConfig,
        objectives: List[str] = None,
        study_name: str = None
    ) -> OptimizationResult:
        """
        多目标优化（例如：同时优化收益和风险）
        
        Args:
            strategy_class: 策略类
            param_space: 参数空间定义
            market_data_list: 市场数据列表
            backtest_config: 回测配置
            objectives: 优化目标列表，例如 ['total_return', 'sharpe_ratio']
            study_name: 研究名称
            
        Returns:
            优化结果
        """
        if objectives is None:
            objectives = ['total_return', 'sharpe_ratio']
        
        if study_name is None:
            study_name = f"multi_opt_{strategy_class.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"开始多目标优化: {study_name}")
        self.logger.info(f"优化目标: {objectives}")
        
        # 创建多目标函数
        def objective(trial: optuna.Trial) -> Tuple[float, ...]:
            # 从参数空间中采样参数
            params = {}
            for param in param_space:
                params[param.name] = param.suggest(trial)
            
            try:
                # 创建策略实例
                strategy = strategy_class(params)
                
                # 运行回测
                engine = BacktestEngine(backtest_config)
                result = engine.run_backtest(strategy, market_data_list)
                
                # 获取所有目标指标值
                values = []
                for obj in objectives:
                    if hasattr(result, obj):
                        values.append(getattr(result, obj))
                    else:
                        raise ValueError(f"未知的目标指标: {obj}")
                
                return tuple(values)
                
            except Exception as e:
                self.logger.error(f"试验{trial.number}失败: {str(e)}")
                return tuple([float('-inf')] * len(objectives))
        
        # 创建多目标研究（使用NSGA-II算法）
        study = optuna.create_study(
            study_name=study_name,
            directions=["maximize"] * len(objectives),  # 假设都是最大化
            sampler=NSGAIISampler(seed=42)
        )
        
        # 运行优化
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs,
            show_progress_bar=True
        )
        
        # 保存研究
        self.studies[study_name] = study
        
        # 获取帕累托前沿
        pareto_front = []
        for trial in study.best_trials:
            pareto_point = {
                'params': trial.params,
                'values': {obj: val for obj, val in zip(objectives, trial.values)}
            }
            pareto_front.append(pareto_point)
        
        # 选择一个"最佳"试验（例如，第一个帕累托最优解）
        best_trial = study.best_trials[0] if study.best_trials else None
        
        result = OptimizationResult(
            best_params=best_trial.params if best_trial else {},
            best_value=best_trial.values[0] if best_trial else 0.0,
            best_trial=best_trial.number if best_trial else 0,
            n_trials=len(study.trials),
            optimization_history=[],
            param_importances={},
            study=study,
            pareto_front=pareto_front
        )
        
        self.logger.info(f"多目标优化完成: 找到{len(pareto_front)}个帕累托最优解")
        
        return result
    
    def visualize_optimization(
        self,
        study_name: str,
        save_path: str = None
    ) -> None:
        """
        可视化优化过程
        
        Args:
            study_name: 研究名称
            save_path: 保存路径（可选）
        """
        if study_name not in self.studies:
            self.logger.error(f"未找到研究: {study_name}")
            return
        
        study = self.studies[study_name]
        
        try:
            import matplotlib.pyplot as plt
            from optuna.visualization.matplotlib import (
                plot_optimization_history,
                plot_param_importances,
                plot_parallel_coordinate,
                plot_slice
            )
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Optimization Results: {study_name}', fontsize=16)
            
            # 优化历史
            plot_optimization_history(study, ax=axes[0, 0])
            axes[0, 0].set_title('Optimization History')
            
            # 参数重要性
            try:
                plot_param_importances(study, ax=axes[0, 1])
                axes[0, 1].set_title('Parameter Importances')
            except Exception as e:
                self.logger.warning(f"无法绘制参数重要性: {str(e)}")
            
            # 平行坐标图
            try:
                plot_parallel_coordinate(study, ax=axes[1, 0])
                axes[1, 0].set_title('Parallel Coordinate Plot')
            except Exception as e:
                self.logger.warning(f"无法绘制平行坐标图: {str(e)}")
            
            # 切片图
            try:
                plot_slice(study, ax=axes[1, 1])
                axes[1, 1].set_title('Slice Plot')
            except Exception as e:
                self.logger.warning(f"无法绘制切片图: {str(e)}")
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"优化可视化已保存: {save_path}")
            else:
                plt.show()
                
        except ImportError:
            self.logger.warning("matplotlib或optuna.visualization未安装，无法生成可视化图表")
        except Exception as e:
            self.logger.error(f"可视化失败: {str(e)}")
    
    def get_best_params(self, study_name: str) -> Dict[str, Any]:
        """
        获取最佳参数
        
        Args:
            study_name: 研究名称
            
        Returns:
            最佳参数字典
        """
        if study_name not in self.studies:
            raise ValueError(f"未找到研究: {study_name}")
        
        return self.studies[study_name].best_params
    
    def export_results(self, study_name: str, filepath: str) -> bool:
        """
        导出优化结果
        
        Args:
            study_name: 研究名称
            filepath: 文件路径
            
        Returns:
            是否成功
        """
        if study_name not in self.studies:
            self.logger.error(f"未找到研究: {study_name}")
            return False
        
        try:
            study = self.studies[study_name]
            df = study.trials_dataframe()
            df.to_csv(filepath, index=False)
            self.logger.info(f"优化结果已导出: {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"导出失败: {str(e)}")
            return False


class GeneticAlgorithmOptimizer:
    """
    遗传算法优化器
    提供传统遗传算法优化作为Optuna的补充
    """
    
    def __init__(
        self,
        population_size: int = 50,
        generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8
    ):
        """
        初始化遗传算法优化器
        
        Args:
            population_size: 种群大小
            generations: 代数
            mutation_rate: 变异率
            crossover_rate: 交叉率
        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"遗传算法优化器初始化: 种群={population_size}, 代数={generations}")
    
    def optimize(
        self,
        fitness_function: Callable,
        param_space: List[ParameterSpace],
        maximize: bool = True
    ) -> Dict[str, Any]:
        """
        使用遗传算法优化
        
        Args:
            fitness_function: 适应度函数
            param_space: 参数空间
            maximize: 是否最大化
            
        Returns:
            最佳参数
        """
        # 初始化种群
        population = self._initialize_population(param_space)
        
        best_individual = None
        best_fitness = float('-inf') if maximize else float('inf')
        
        for generation in range(self.generations):
            # 评估适应度
            fitness_scores = [fitness_function(ind) for ind in population]
            
            # 更新最佳个体
            for ind, fitness in zip(population, fitness_scores):
                if maximize and fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = ind.copy()
                elif not maximize and fitness < best_fitness:
                    best_fitness = fitness
                    best_individual = ind.copy()
            
            # 选择
            selected = self._selection(population, fitness_scores, maximize)
            
            # 交叉
            offspring = self._crossover(selected)
            
            # 变异
            offspring = self._mutation(offspring, param_space)
            
            # 更新种群
            population = offspring
            
            if generation % 10 == 0:
                self.logger.info(f"代数 {generation}: 最佳适应度 = {best_fitness:.4f}")
        
        self.logger.info(f"遗传算法优化完成: 最佳适应度 = {best_fitness:.4f}")
        return best_individual
    
    def _initialize_population(self, param_space: List[ParameterSpace]) -> List[Dict[str, Any]]:
        """初始化种群"""
        population = []
        for _ in range(self.population_size):
            individual = {}
            for param in param_space:
                if param.type == "int":
                    individual[param.name] = np.random.randint(param.low, param.high + 1)
                elif param.type == "float":
                    individual[param.name] = np.random.uniform(param.low, param.high)
                elif param.type == "categorical":
                    individual[param.name] = np.random.choice(param.choices)
            population.append(individual)
        return population
    
    def _selection(
        self,
        population: List[Dict[str, Any]],
        fitness_scores: List[float],
        maximize: bool
    ) -> List[Dict[str, Any]]:
        """选择操作（锦标赛选择）"""
        selected = []
        for _ in range(len(population)):
            # 随机选择两个个体
            idx1, idx2 = np.random.choice(len(population), 2, replace=False)
            
            # 选择适应度更好的
            if maximize:
                winner = idx1 if fitness_scores[idx1] > fitness_scores[idx2] else idx2
            else:
                winner = idx1 if fitness_scores[idx1] < fitness_scores[idx2] else idx2
            
            selected.append(population[winner].copy())
        
        return selected
    
    def _crossover(self, population: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """交叉操作"""
        offspring = []
        for i in range(0, len(population), 2):
            parent1 = population[i]
            parent2 = population[i + 1] if i + 1 < len(population) else population[0]
            
            if np.random.random() < self.crossover_rate:
                # 单点交叉
                keys = list(parent1.keys())
                crossover_point = np.random.randint(1, len(keys))
                
                child1 = {}
                child2 = {}
                for j, key in enumerate(keys):
                    if j < crossover_point:
                        child1[key] = parent1[key]
                        child2[key] = parent2[key]
                    else:
                        child1[key] = parent2[key]
                        child2[key] = parent1[key]
                
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1.copy(), parent2.copy()])
        
        return offspring[:len(population)]
    
    def _mutation(
        self,
        population: List[Dict[str, Any]],
        param_space: List[ParameterSpace]
    ) -> List[Dict[str, Any]]:
        """变异操作"""
        for individual in population:
            for param in param_space:
                if np.random.random() < self.mutation_rate:
                    if param.type == "int":
                        individual[param.name] = np.random.randint(param.low, param.high + 1)
                    elif param.type == "float":
                        individual[param.name] = np.random.uniform(param.low, param.high)
                    elif param.type == "categorical":
                        individual[param.name] = np.random.choice(param.choices)
        
        return population
