"""
RL交易环境测试
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import gymnasium as gym

from src.rl.trading_env import TradingEnvironment, ActionType, ObservationSpace
from src.core.models import PositionType


class TestTradingEnvironment(unittest.TestCase):
    """测试RL交易环境"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建模拟数据
        dates = pd.date_range(start='2024-01-01', periods=1000, freq='1H')
        np.random.seed(42)
        
        # 生成价格数据
        close_prices = 100 + np.cumsum(np.random.randn(1000) * 0.5)
        
        self.test_data = pd.DataFrame({
            'open': close_prices + np.random.randn(1000) * 0.1,
            'high': close_prices + np.abs(np.random.randn(1000) * 0.2),
            'low': close_prices - np.abs(np.random.randn(1000) * 0.2),
            'close': close_prices,
            'volume': np.random.randint(1000, 10000, 1000),
            'sma_20': close_prices,  # 简化的SMA
            'rsi': np.random.uniform(30, 70, 1000),
            'macd': np.random.randn(1000) * 0.5
        }, index=dates)
        
        self.env = TradingEnvironment(
            symbol='EURUSD',
            data=self.test_data,
            initial_balance=10000.0,
            max_position_size=1.0,
            transaction_cost=0.0001
        )
    
    def test_environment_initialization(self):
        """测试环境初始化"""
        self.assertEqual(self.env.symbol, 'EURUSD')
        self.assertEqual(self.env.initial_balance, 10000.0)
        self.assertEqual(self.env.max_position_size, 1.0)
        self.assertIsNotNone(self.env.observation_space)
        self.assertIsNotNone(self.env.action_space)
        self.assertEqual(self.env.action_space.n, 5)
    
    def test_observation_space_size(self):
        """测试观察空间大小"""
        obs_config = ObservationSpace(
            price_window=20,
            indicator_count=15,
            position_features=5,
            market_state_features=10
        )
        
        expected_size = 20 * 4 + 15 + 5 + 10  # 110
        self.assertEqual(obs_config.get_total_size(), expected_size)
        
        env = TradingEnvironment(
            symbol='EURUSD',
            data=self.test_data,
            obs_config=obs_config
        )
        
        self.assertEqual(env.observation_space.shape[0], expected_size)
    
    def test_reset(self):
        """测试环境重置"""
        observation, info = self.env.reset()
        
        # 检查观察形状
        self.assertEqual(observation.shape, self.env.observation_space.shape)
        
        # 检查初始状态
        self.assertEqual(self.env.balance, 10000.0)
        self.assertEqual(self.env.equity, 10000.0)
        self.assertIsNone(self.env.position)
        self.assertEqual(self.env.total_trades, 0)
        self.assertEqual(len(self.env.trades_history), 0)
        
        # 检查info
        self.assertIn('balance', info)
        self.assertIn('equity', info)
        self.assertIn('total_trades', info)
    
    def test_buy_action(self):
        """测试买入动作"""
        self.env.reset()
        
        # 执行买入动作
        observation, reward, terminated, truncated, info = self.env.step(ActionType.BUY.value)
        
        # 检查持仓已创建
        self.assertIsNotNone(self.env.position)
        self.assertEqual(self.env.position.type, PositionType.LONG)
        self.assertGreater(self.env.position.volume, 0)
        self.assertEqual(self.env.total_trades, 1)
        
        # 检查余额减少
        self.assertLess(self.env.balance, 10000.0)
    
    def test_sell_action(self):
        """测试卖出动作"""
        self.env.reset()
        
        # 执行卖出动作
        observation, reward, terminated, truncated, info = self.env.step(ActionType.SELL.value)
        
        # 检查持仓已创建
        self.assertIsNotNone(self.env.position)
        self.assertEqual(self.env.position.type, PositionType.SHORT)
        self.assertGreater(self.env.position.volume, 0)
        self.assertEqual(self.env.total_trades, 1)
    
    def test_close_action(self):
        """测试平仓动作"""
        self.env.reset()
        
        # 先买入
        self.env.step(ActionType.BUY.value)
        initial_trades = self.env.total_trades
        
        # 再平仓
        self.env.step(ActionType.CLOSE.value)
        
        # 检查持仓已关闭
        self.assertIsNone(self.env.position)
        self.assertEqual(len(self.env.trades_history), 1)
        
        # 检查交易记录
        trade = self.env.trades_history[0]
        self.assertIn('open_price', trade)
        self.assertIn('close_price', trade)
        self.assertIn('pnl', trade)
    
    def test_hold_action(self):
        """测试持有动作"""
        self.env.reset()
        
        # 执行持有动作
        observation, reward, terminated, truncated, info = self.env.step(ActionType.HOLD.value)
        
        # 检查没有持仓
        self.assertIsNone(self.env.position)
        self.assertEqual(self.env.total_trades, 0)
    
    def test_adjust_action(self):
        """测试调整仓位动作"""
        self.env.reset()
        
        # 先买入
        self.env.step(ActionType.BUY.value)
        initial_volume = self.env.position.volume
        
        # 调整仓位
        self.env.step(ActionType.ADJUST.value)
        
        # 检查仓位已调整
        self.assertIsNotNone(self.env.position)
        self.assertLess(self.env.position.volume, initial_volume)
    
    def test_reward_calculation(self):
        """测试奖励计算"""
        self.env.reset()
        
        # 执行一系列动作
        _, reward1, _, _, _ = self.env.step(ActionType.BUY.value)
        _, reward2, _, _, _ = self.env.step(ActionType.HOLD.value)
        _, reward3, _, _, _ = self.env.step(ActionType.CLOSE.value)
        
        # 检查奖励是数值
        self.assertIsInstance(reward1, (int, float))
        self.assertIsInstance(reward2, (int, float))
        self.assertIsInstance(reward3, (int, float))
    
    def test_termination_conditions(self):
        """测试终止条件"""
        self.env.reset()
        
        # 模拟资金耗尽
        self.env.equity = self.env.initial_balance * 0.4
        terminated = self.env._is_terminated()
        self.assertTrue(terminated)
        
        # 重置并模拟回撤过大
        self.env.reset()
        self.env.max_drawdown = 0.35
        terminated = self.env._is_terminated()
        self.assertTrue(terminated)
    
    def test_observation_shape(self):
        """测试观察形状"""
        observation, _ = self.env.reset()
        
        # 检查观察是numpy数组
        self.assertIsInstance(observation, np.ndarray)
        
        # 检查形状正确
        self.assertEqual(observation.shape, self.env.observation_space.shape)
        
        # 检查数据类型
        self.assertEqual(observation.dtype, np.float32)
    
    def test_multiple_episodes(self):
        """测试多个回合"""
        for episode in range(3):
            observation, info = self.env.reset()
            
            done = False
            steps = 0
            max_steps = 100
            
            while not done and steps < max_steps:
                action = self.env.action_space.sample()
                observation, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                steps += 1
            
            # 检查每个回合都能正常完成
            self.assertGreater(steps, 0)
    
    def test_performance_summary(self):
        """测试性能摘要"""
        self.env.reset()
        
        # 执行一些交易
        self.env.step(ActionType.BUY.value)
        for _ in range(10):
            self.env.step(ActionType.HOLD.value)
        self.env.step(ActionType.CLOSE.value)
        
        # 获取性能摘要
        summary = self.env.get_performance_summary()
        
        # 检查摘要包含所有必要字段
        self.assertIn('total_return', summary)
        self.assertIn('final_equity', summary)
        self.assertIn('total_trades', summary)
        self.assertIn('win_rate', summary)
        self.assertIn('max_drawdown', summary)
        self.assertIn('sharpe_ratio', summary)
    
    def test_transaction_costs(self):
        """测试交易成本"""
        self.env.reset()
        initial_balance = self.env.balance
        
        # 买入并立即平仓
        self.env.step(ActionType.BUY.value)
        balance_after_buy = self.env.balance
        
        self.env.step(ActionType.CLOSE.value)
        balance_after_close = self.env.balance
        
        # 检查交易成本已扣除
        self.assertLess(balance_after_buy, initial_balance)
    
    def test_position_size_calculation(self):
        """测试仓位大小计算"""
        self.env.reset()
        
        position_size = self.env._calculate_position_size()
        
        # 检查仓位大小合理
        self.assertGreater(position_size, 0)
        self.assertLessEqual(position_size, self.env.max_position_size)
    
    def test_equity_update(self):
        """测试权益更新"""
        self.env.reset()
        
        # 买入
        self.env.step(ActionType.BUY.value)
        
        # 模拟价格变化
        for _ in range(5):
            prev_equity = self.env.equity
            self.env.step(ActionType.HOLD.value)
            
            # 权益应该随价格变化
            # (可能增加或减少)
            self.assertIsInstance(self.env.equity, float)
    
    def test_gymnasium_compatibility(self):
        """测试与gymnasium的兼容性"""
        # 检查环境是gymnasium.Env的实例
        self.assertIsInstance(self.env, gym.Env)
        
        # 检查必要的方法存在
        self.assertTrue(hasattr(self.env, 'reset'))
        self.assertTrue(hasattr(self.env, 'step'))
        self.assertTrue(hasattr(self.env, 'render'))
        
        # 检查空间定义
        self.assertIsInstance(self.env.observation_space, gym.spaces.Box)
        self.assertIsInstance(self.env.action_space, gym.spaces.Discrete)


class TestObservationSpace(unittest.TestCase):
    """测试观察空间配置"""
    
    def test_default_configuration(self):
        """测试默认配置"""
        obs_config = ObservationSpace()
        
        self.assertEqual(obs_config.price_window, 20)
        self.assertEqual(obs_config.indicator_count, 15)
        self.assertEqual(obs_config.position_features, 5)
        self.assertEqual(obs_config.market_state_features, 10)
    
    def test_custom_configuration(self):
        """测试自定义配置"""
        obs_config = ObservationSpace(
            price_window=30,
            indicator_count=20,
            position_features=8,
            market_state_features=12
        )
        
        self.assertEqual(obs_config.price_window, 30)
        self.assertEqual(obs_config.indicator_count, 20)
        self.assertEqual(obs_config.position_features, 8)
        self.assertEqual(obs_config.market_state_features, 12)
    
    def test_total_size_calculation(self):
        """测试总大小计算"""
        obs_config = ObservationSpace(
            price_window=10,
            indicator_count=5,
            position_features=3,
            market_state_features=7
        )
        
        expected_size = 10 * 4 + 5 + 3 + 7  # 55
        self.assertEqual(obs_config.get_total_size(), expected_size)


if __name__ == '__main__':
    unittest.main()
