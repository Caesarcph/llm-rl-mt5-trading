"""
RL训练器实现
集成stable-baselines3进行强化学习训练
"""

import os
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging
import numpy as np
import pandas as pd

from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from .trading_env import TradingEnvironment


logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """训练配置"""
    algorithm: str = "PPO"  # PPO, SAC, A2C
    total_timesteps: int = 100000
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_steps: int = 2048  # For PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # 训练参数
    n_envs: int = 1  # 并行环境数量
    eval_freq: int = 10000  # 评估频率
    save_freq: int = 10000  # 保存频率
    n_eval_episodes: int = 10  # 评估回合数
    
    # 模型保存
    model_save_path: str = "models/rl"
    log_path: str = "logs/rl"
    
    # 早停
    early_stopping_patience: int = 10
    early_stopping_threshold: float = 0.01
    
    def __post_init__(self):
        """创建必要的目录"""
        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)


class TrainingCallback(BaseCallback):
    """
    自定义训练回调
    用于监控训练过程和实现早停
    """
    
    def __init__(
        self,
        check_freq: int = 1000,
        log_dir: str = "logs/rl",
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "training_metrics.csv")
        
        # 训练指标
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.training_metrics: List[Dict[str, Any]] = []
        
        # 早停相关
        self.best_mean_reward = -np.inf
        self.no_improvement_count = 0
        
        os.makedirs(log_dir, exist_ok=True)
    
    def _on_step(self) -> bool:
        """每步调用"""
        # 收集回合信息
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                self.episode_rewards.append(info['r'])
                self.episode_lengths.append(info['l'])
        
        # 定期记录
        if self.n_calls % self.check_freq == 0:
            self._log_metrics()
        
        return True
    
    def _log_metrics(self) -> None:
        """记录训练指标"""
        if len(self.episode_rewards) == 0:
            return
        
        mean_reward = np.mean(self.episode_rewards[-100:])
        mean_length = np.mean(self.episode_lengths[-100:])
        
        metrics = {
            'timestep': self.num_timesteps,
            'mean_reward': mean_reward,
            'mean_length': mean_length,
            'n_episodes': len(self.episode_rewards),
            'timestamp': datetime.now().isoformat()
        }
        
        self.training_metrics.append(metrics)
        
        if self.verbose > 0:
            logger.info(
                f"Timestep: {self.num_timesteps}, "
                f"Mean Reward: {mean_reward:.2f}, "
                f"Mean Length: {mean_length:.1f}"
            )
        
        # 保存指标
        self._save_metrics()
    
    def _save_metrics(self) -> None:
        """保存训练指标到CSV"""
        if len(self.training_metrics) > 0:
            df = pd.DataFrame(self.training_metrics)
            df.to_csv(self.save_path, index=False)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取训练指标摘要"""
        if len(self.episode_rewards) == 0:
            return {}
        
        return {
            'total_episodes': len(self.episode_rewards),
            'mean_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'min_reward': np.min(self.episode_rewards),
            'max_reward': np.max(self.episode_rewards),
            'mean_length': np.mean(self.episode_lengths),
            'total_timesteps': self.num_timesteps
        }


class RLTrainer:
    """
    强化学习训练器
    集成stable-baselines3进行模型训练
    """
    
    def __init__(
        self,
        env: TradingEnvironment,
        config: Optional[TrainingConfig] = None
    ):
        """
        初始化训练器
        
        Args:
            env: 交易环境
            config: 训练配置
        """
        self.env = env
        self.config = config or TrainingConfig()
        
        # 包装环境
        self.train_env = self._make_env()
        self.eval_env = self._make_env()
        
        # 创建模型
        self.model = self._create_model()
        
        # 回调
        self.callbacks: List[BaseCallback] = []
        self._setup_callbacks()
        
        logger.info(f"初始化RL训练器: 算法={self.config.algorithm}")
    
    def _make_env(self) -> Monitor:
        """创建并包装环境"""
        env = Monitor(self.env, self.config.log_path)
        return env
    
    def _create_model(self):
        """创建RL模型"""
        algorithm = self.config.algorithm.upper()
        
        if algorithm == "PPO":
            model = PPO(
                "MlpPolicy",
                self.train_env,
                learning_rate=self.config.learning_rate,
                n_steps=self.config.n_steps,
                batch_size=self.config.batch_size,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
                ent_coef=self.config.ent_coef,
                vf_coef=self.config.vf_coef,
                max_grad_norm=self.config.max_grad_norm,
                verbose=1,
                tensorboard_log=None  # Disable tensorboard to avoid dependency
            )
        elif algorithm == "SAC":
            # SAC only supports continuous action spaces (Box)
            # For discrete action spaces, use PPO or A2C instead
            model = SAC(
                "MlpPolicy",
                self.train_env,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                gamma=self.config.gamma,
                verbose=1,
                tensorboard_log=None  # Disable tensorboard to avoid dependency
            )
        elif algorithm == "A2C":
            model = A2C(
                "MlpPolicy",
                self.train_env,
                learning_rate=self.config.learning_rate,
                n_steps=self.config.n_steps,
                gamma=self.config.gamma,
                ent_coef=self.config.ent_coef,
                vf_coef=self.config.vf_coef,
                max_grad_norm=self.config.max_grad_norm,
                verbose=1,
                tensorboard_log=None  # Disable tensorboard to avoid dependency
            )
        else:
            raise ValueError(f"不支持的算法: {algorithm}")
        
        logger.info(f"创建{algorithm}模型")
        return model
    
    def _setup_callbacks(self) -> None:
        """设置训练回调"""
        # 训练监控回调
        training_callback = TrainingCallback(
            check_freq=1000,
            log_dir=self.config.log_path,
            verbose=1
        )
        self.callbacks.append(training_callback)
        
        # 评估回调
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=os.path.join(self.config.model_save_path, "best_model"),
            log_path=self.config.log_path,
            eval_freq=self.config.eval_freq,
            n_eval_episodes=self.config.n_eval_episodes,
            deterministic=True,
            render=False
        )
        self.callbacks.append(eval_callback)
        
        # 检查点回调
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config.save_freq,
            save_path=self.config.model_save_path,
            name_prefix="rl_model"
        )
        self.callbacks.append(checkpoint_callback)
    
    def train(self, total_timesteps: Optional[int] = None) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            total_timesteps: 训练步数 (如果为None则使用配置中的值)
            
        Returns:
            训练结果摘要
        """
        timesteps = total_timesteps or self.config.total_timesteps
        
        logger.info(f"开始训练: {timesteps} 步")
        start_time = datetime.now()
        
        try:
            self.model.learn(
                total_timesteps=timesteps,
                callback=self.callbacks,
                progress_bar=True
            )
            
            training_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"训练完成: 用时 {training_time:.2f} 秒")
            
            # 保存最终模型
            self.save_model("final_model")
            
            # 获取训练摘要
            summary = self._get_training_summary(training_time)
            
            return summary
            
        except Exception as e:
            logger.error(f"训练失败: {e}")
            raise
    
    def evaluate(self, n_episodes: int = 10) -> Dict[str, Any]:
        """
        评估模型
        
        Args:
            n_episodes: 评估回合数
            
        Returns:
            评估结果
        """
        logger.info(f"评估模型: {n_episodes} 回合")
        
        mean_reward, std_reward = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=n_episodes,
            deterministic=True
        )
        
        # 详细评估
        episode_rewards = []
        episode_lengths = []
        
        for _ in range(n_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        results = {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths)
        }
        
        logger.info(f"评估结果: 平均奖励={mean_reward:.2f} ± {std_reward:.2f}")
        
        return results
    
    def save_model(self, name: str = "model") -> str:
        """
        保存模型
        
        Args:
            name: 模型名称
            
        Returns:
            保存路径
        """
        save_path = os.path.join(self.config.model_save_path, name)
        self.model.save(save_path)
        logger.info(f"模型已保存: {save_path}")
        return save_path
    
    def load_model(self, path: str) -> None:
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        algorithm = self.config.algorithm.upper()
        
        if algorithm == "PPO":
            self.model = PPO.load(path, env=self.train_env)
        elif algorithm == "SAC":
            self.model = SAC.load(path, env=self.train_env)
        elif algorithm == "A2C":
            self.model = A2C.load(path, env=self.train_env)
        else:
            raise ValueError(f"不支持的算法: {algorithm}")
        
        logger.info(f"模型已加载: {path}")
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """
        预测动作
        
        Args:
            observation: 观察
            deterministic: 是否使用确定性策略
            
        Returns:
            动作
        """
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return int(action)
    
    def _get_training_summary(self, training_time: float) -> Dict[str, Any]:
        """获取训练摘要"""
        summary = {
            'algorithm': self.config.algorithm,
            'total_timesteps': self.config.total_timesteps,
            'training_time': training_time,
            'model_path': self.config.model_save_path
        }
        
        # 从回调中获取指标
        for callback in self.callbacks:
            if isinstance(callback, TrainingCallback):
                summary.update(callback.get_metrics_summary())
                break
        
        return summary
    
    def continue_training(self, additional_timesteps: int) -> Dict[str, Any]:
        """
        继续训练
        
        Args:
            additional_timesteps: 额外训练步数
            
        Returns:
            训练结果摘要
        """
        logger.info(f"继续训练: {additional_timesteps} 步")
        return self.train(total_timesteps=additional_timesteps)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'algorithm': self.config.algorithm,
            'policy': str(self.model.policy),
            'learning_rate': self.config.learning_rate,
            'gamma': self.config.gamma,
            'observation_space': str(self.model.observation_space),
            'action_space': str(self.model.action_space)
        }


class MultiEnvTrainer(RLTrainer):
    """
    多环境并行训练器
    使用多个环境并行训练以提高效率
    """
    
    def __init__(
        self,
        env_fn: Callable[[], TradingEnvironment],
        n_envs: int = 4,
        config: Optional[TrainingConfig] = None
    ):
        """
        初始化多环境训练器
        
        Args:
            env_fn: 环境创建函数
            n_envs: 并行环境数量
            config: 训练配置
        """
        self.env_fn = env_fn
        self.n_envs = n_envs
        
        # 不调用父类__init__，自己实现
        self.config = config or TrainingConfig()
        self.config.n_envs = n_envs
        
        # 创建向量化环境
        self.train_env = self._make_vec_env()
        self.eval_env = Monitor(env_fn(), self.config.log_path)
        
        # 创建模型
        self.model = self._create_model()
        
        # 回调
        self.callbacks: List[BaseCallback] = []
        self._setup_callbacks()
        
        logger.info(f"初始化多环境训练器: {n_envs} 个并行环境")
    
    def _make_vec_env(self):
        """创建向量化环境"""
        def make_env():
            env = self.env_fn()
            env = Monitor(env, self.config.log_path)
            return env
        
        # 使用SubprocVecEnv进行真正的并行
        vec_env = SubprocVecEnv([make_env for _ in range(self.n_envs)])
        
        return vec_env
