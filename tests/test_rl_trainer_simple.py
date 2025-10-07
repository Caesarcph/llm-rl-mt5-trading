"""
RL训练器简化测试 - 避免文件锁定问题
"""

import unittest
import numpy as np
import pandas as pd

from src.rl.trading_env import TradingEnvironment
from src.rl.rl_trainer import RLTrainer, TrainingConfig, TrainingCallback


class TestRLTrainerSimple(unittest.TestCase):
    """简化的RL训练器测试"""
    
    @classmethod
    def setUpClass(cls):
        """类级别设置 - 创建共享测试数据"""
        dates = pd.date_range(start='2024-01-01', periods=500, freq='h')
        np.random.seed(42)
        
        close_prices = 100 + np.cumsum(np.random.randn(500) * 0.5)
        
        cls.test_data = pd.DataFrame({
            'open': close_prices + np.random.randn(500) * 0.1,
            'high': close_prices + np.abs(np.random.randn(500) * 0.2),
            'low': close_prices - np.abs(np.random.randn(500) * 0.2),
            'close': close_prices,
            'volume': np.random.randint(1000, 10000, 500),
            'sma_20': close_prices,
            'rsi': np.random.uniform(30, 70, 500),
            'macd': np.random.randn(500) * 0.5
        }, index=dates)
    
    def test_trainer_creation(self):
        """测试训练器创建"""
        env = TradingEnvironment(
            symbol='EURUSD',
            data=self.test_data,
            initial_balance=10000.0
        )
        
        config = TrainingConfig(
            algorithm="PPO",
            total_timesteps=1000
        )
        
        trainer = RLTrainer(env, config)
        
        self.assertIsNotNone(trainer.model)
        self.assertIsNotNone(trainer.train_env)
        self.assertEqual(trainer.config.algorithm, "PPO")
        
        # 清理
        trainer.train_env.close()
        trainer.eval_env.close()
    
    def test_prediction(self):
        """测试预测功能"""
        env = TradingEnvironment(
            symbol='EURUSD',
            data=self.test_data,
            initial_balance=10000.0
        )
        
        config = TrainingConfig(algorithm="PPO")
        trainer = RLTrainer(env, config)
        
        obs, _ = env.reset()
        action = trainer.predict(obs)
        
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, env.action_space.n)
        
        # 清理
        trainer.train_env.close()
        trainer.eval_env.close()
        env.close()
    
    def test_model_info(self):
        """测试获取模型信息"""
        env = TradingEnvironment(
            symbol='EURUSD',
            data=self.test_data,
            initial_balance=10000.0
        )
        
        config = TrainingConfig(algorithm="A2C")
        trainer = RLTrainer(env, config)
        
        info = trainer.get_model_info()
        
        self.assertIn('algorithm', info)
        self.assertEqual(info['algorithm'], "A2C")
        self.assertIn('policy', info)
        self.assertIn('learning_rate', info)
        
        # 清理
        trainer.train_env.close()
        trainer.eval_env.close()
        env.close()


if __name__ == '__main__':
    unittest.main()
