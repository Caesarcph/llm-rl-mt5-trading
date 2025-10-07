"""
RL系统演示
展示如何使用强化学习优化系统进行交易决策
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.rl import (
    TradingEnvironment,
    RLTrainer,
    TrainingConfig,
    RLStrategyOptimizer,
    RLOptimizerConfig
)
from src.core.models import Signal, MarketData


def create_sample_data(n_samples=1000):
    """创建示例市场数据"""
    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='h')
    np.random.seed(42)
    
    # 生成价格数据
    close_prices = 100 + np.cumsum(np.random.randn(n_samples) * 0.5)
    
    data = pd.DataFrame({
        'open': close_prices + np.random.randn(n_samples) * 0.1,
        'high': close_prices + np.abs(np.random.randn(n_samples) * 0.2),
        'low': close_prices - np.abs(np.random.randn(n_samples) * 0.2),
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, n_samples),
        # 技术指标
        'sma_20': close_prices,
        'sma_50': close_prices * 0.99,
        'rsi': np.random.uniform(30, 70, n_samples),
        'macd': np.random.randn(n_samples) * 0.5,
        'bb_upper': close_prices * 1.02,
        'bb_lower': close_prices * 0.98,
    }, index=dates)
    
    return data


def create_traditional_signal(market_data: MarketData) -> Signal:
    """创建传统策略信号（示例）"""
    current_price = market_data.ohlcv.iloc[-1]['close']
    sma_20 = market_data.ohlcv.iloc[-1]['sma_20']
    rsi = market_data.ohlcv.iloc[-1]['rsi']
    
    # 简单的MA交叉 + RSI策略
    if current_price > sma_20 and rsi < 70:
        # 买入信号
        return Signal(
            strategy_id='ma_rsi_strategy',
            symbol=market_data.symbol,
            direction=1,
            strength=0.7,
            entry_price=current_price,
            sl=current_price * 0.98,
            tp=current_price * 1.04,
            size=0.1,
            confidence=0.7,
            timestamp=datetime.now(),
            metadata={'strategy': 'MA_RSI', 'rsi': rsi}
        )
    elif current_price < sma_20 and rsi > 30:
        # 卖出信号
        return Signal(
            strategy_id='ma_rsi_strategy',
            symbol=market_data.symbol,
            direction=-1,
            strength=0.6,
            entry_price=current_price,
            sl=current_price * 1.02,
            tp=current_price * 0.96,
            size=0.1,
            confidence=0.6,
            timestamp=datetime.now(),
            metadata={'strategy': 'MA_RSI', 'rsi': rsi}
        )
    
    return None


def main():
    """主演示函数"""
    print("=" * 60)
    print("RL交易系统演示")
    print("=" * 60)
    
    # 1. 创建数据
    print("\n1. 创建示例数据...")
    data = create_sample_data(1000)
    print(f"   数据点数: {len(data)}")
    print(f"   时间范围: {data.index[0]} 到 {data.index[-1]}")
    
    # 2. 创建交易环境
    print("\n2. 创建RL交易环境...")
    env = TradingEnvironment(
        symbol='EURUSD',
        data=data,
        initial_balance=10000.0,
        max_position_size=1.0,
        transaction_cost=0.0001
    )
    print(f"   观察空间维度: {env.observation_space.shape[0]}")
    print(f"   动作空间大小: {env.action_space.n}")
    
    # 3. 创建训练配置
    print("\n3. 配置RL训练器...")
    training_config = TrainingConfig(
        algorithm="PPO",
        total_timesteps=5000,  # 演示用较少步数
        learning_rate=3e-4,
        batch_size=64,
        n_steps=2048
    )
    print(f"   算法: {training_config.algorithm}")
    print(f"   训练步数: {training_config.total_timesteps}")
    
    # 4. 创建训练器
    print("\n4. 创建RL训练器...")
    trainer = RLTrainer(env, training_config)
    print("   训练器已创建")
    
    # 5. 训练模型（可选，演示中跳过）
    print("\n5. 训练RL模型...")
    print("   [演示模式] 跳过训练，使用未训练模型")
    # summary = trainer.train()
    # print(f"   训练完成: {summary}")
    
    # 6. 创建策略优化器
    print("\n6. 创建RL策略优化器...")
    optimizer_config = RLOptimizerConfig(
        initial_rl_weight=0.3,
        max_rl_weight=0.7,
        min_rl_weight=0.1,
        online_learning_enabled=True
    )
    optimizer = RLStrategyOptimizer(trainer, optimizer_config)
    print(f"   RL权重: {optimizer.rl_weight:.2f}")
    print(f"   传统权重: {optimizer.traditional_weight:.2f}")
    
    # 7. 模拟交易决策
    print("\n7. 模拟交易决策...")
    print("-" * 60)
    
    # 重置环境
    observation, _ = env.reset()
    
    # 模拟10个交易决策
    for i in range(10):
        # 创建市场数据
        current_idx = env.current_step
        market_data = MarketData(
            symbol='EURUSD',
            timeframe='H1',
            timestamp=datetime.now(),
            ohlcv=data.iloc[:current_idx+1]
        )
        
        # 获取传统策略信号
        traditional_signal = create_traditional_signal(market_data)
        traditional_signals = [traditional_signal] if traditional_signal else []
        
        # 获取组合信号
        combined_signal = optimizer.get_combined_signal(
            market_data,
            traditional_signals,
            observation
        )
        
        # 打印决策
        print(f"\n步骤 {i+1}:")
        print(f"  价格: {market_data.ohlcv.iloc[-1]['close']:.2f}")
        
        if traditional_signal:
            print(f"  传统信号: {traditional_signal.direction} "
                  f"(强度={traditional_signal.strength:.2f})")
        else:
            print(f"  传统信号: 无")
        
        if combined_signal:
            print(f"  组合信号: {combined_signal.direction} "
                  f"(强度={combined_signal.strength:.2f}, "
                  f"来源={combined_signal.metadata.get('source', 'unknown')})")
        else:
            print(f"  组合信号: 无")
        
        # 执行动作
        action = 0  # HOLD
        if combined_signal:
            if combined_signal.direction == 1:
                action = 1  # BUY
            elif combined_signal.direction == -1:
                action = 2  # SELL
        
        # 环境步进
        next_observation, reward, terminated, truncated, info = env.step(action)
        
        print(f"  动作: {['HOLD', 'BUY', 'SELL', 'CLOSE', 'ADJUST'][action]}")
        print(f"  奖励: {reward:.4f}")
        print(f"  权益: ${info['equity']:.2f}")
        
        # 添加在线学习样本
        optimizer.add_online_sample(observation, action, reward, next_observation)
        
        observation = next_observation
        
        if terminated or truncated:
            print("\n  [回合结束]")
            break
    
    # 8. 显示统计信息
    print("\n" + "=" * 60)
    print("8. 统计信息")
    print("=" * 60)
    
    stats = optimizer.get_statistics()
    print(f"\n策略权重:")
    print(f"  RL权重: {stats['rl_weight']:.2%}")
    print(f"  传统权重: {stats['traditional_weight']:.2%}")
    
    print(f"\n预测统计:")
    print(f"  总预测数: {stats['total_predictions']}")
    print(f"  RL预测数: {stats['rl_predictions']}")
    print(f"  传统预测数: {stats['traditional_predictions']}")
    print(f"  RL使用率: {stats['rl_prediction_rate']:.2%}")
    
    print(f"\n在线学习:")
    print(f"  收集样本数: {stats['online_samples_collected']}")
    print(f"  距离更新步数: {stats['steps_since_update']}")
    
    # 9. 环境性能摘要
    print("\n" + "=" * 60)
    print("9. 环境性能摘要")
    print("=" * 60)
    
    performance = env.get_performance_summary()
    print(f"\n最终权益: ${performance['final_equity']:.2f}")
    print(f"总收益率: {performance['total_return']:.2%}")
    print(f"总交易数: {performance['total_trades']}")
    print(f"胜率: {performance['win_rate']:.2%}")
    print(f"最大回撤: {performance['max_drawdown']:.2%}")
    print(f"夏普比率: {performance['sharpe_ratio']:.2f}")
    
    # 10. 清理
    print("\n" + "=" * 60)
    print("10. 清理资源...")
    trainer.train_env.close()
    trainer.eval_env.close()
    env.close()
    print("   完成!")
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
