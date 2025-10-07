#!/usr/bin/env python3
"""
自定义策略示例
演示如何创建和使用自定义交易策略
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.models import Strategy, Signal, MarketData
from src.data.data_pipeline import DataPipeline
from datetime import datetime
from typing import Optional
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleMAStrategy(Strategy):
    """
    简单移动平均线策略示例
    
    策略逻辑:
    - 快速MA上穿慢速MA: 买入信号
    - 快速MA下穿慢速MA: 卖出信号
    - RSI过滤: 避免超买超卖区域
    """
    
    def __init__(self, fast_period: int = 10, slow_period: int = 30, rsi_period: int = 14):
        """
        初始化策略
        
        Args:
            fast_period: 快速MA周期
            slow_period: 慢速MA周期
            rsi_period: RSI周期
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.rsi_period = rsi_period
        self.name = "simple_ma_strategy"
        
        logger.info(f"初始化 {self.name}")
        logger.info(f"参数: 快速MA={fast_period}, 慢速MA={slow_period}, RSI={rsi_period}")
    
    def generate_signal(self, market_data: MarketData) -> Optional[Signal]:
        """
        生成交易信号
        
        Args:
            market_data: 市场数据
            
        Returns:
            Signal对象或None
        """
        try:
            df = market_data.ohlcv.copy()
            
            # 计算移动平均线
            df['ma_fast'] = df['close'].rolling(window=self.fast_period).mean()
            df['ma_slow'] = df['close'].rolling(window=self.slow_period).mean()
            
            # 计算RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # 检查是否有足够的数据
            if len(df) < self.slow_period + self.rsi_period:
                logger.debug("数据不足，无法生成信号")
                return None
            
            # 获取最新数据
            current_price = df['close'].iloc[-1]
            ma_fast_current = df['ma_fast'].iloc[-1]
            ma_slow_current = df['ma_slow'].iloc[-1]
            ma_fast_prev = df['ma_fast'].iloc[-2]
            ma_slow_prev = df['ma_slow'].iloc[-2]
            rsi_current = df['rsi'].iloc[-1]
            
            # 检测交叉
            bullish_cross = (ma_fast_prev <= ma_slow_prev) and (ma_fast_current > ma_slow_current)
            bearish_cross = (ma_fast_prev >= ma_slow_prev) and (ma_fast_current < ma_slow_current)
            
            # 生成信号
            signal = None
            
            if bullish_cross and rsi_current < 70:  # 买入信号
                # 计算止损止盈
                atr = self._calculate_atr(df, period=14)
                sl = current_price - (2 * atr)
                tp = current_price + (3 * atr)
                
                signal = Signal(
                    strategy_id=self.name,
                    symbol=market_data.symbol,
                    direction=1,  # 买入
                    strength=0.8,
                    entry_price=current_price,
                    sl=sl,
                    tp=tp,
                    size=0.1,
                    confidence=0.75,
                    timestamp=market_data.timestamp,
                    metadata={
                        'ma_fast': ma_fast_current,
                        'ma_slow': ma_slow_current,
                        'rsi': rsi_current,
                        'signal_type': 'bullish_cross'
                    }
                )
                
                logger.info(f"生成买入信号: {market_data.symbol} @ {current_price:.5f}")
                
            elif bearish_cross and rsi_current > 30:  # 卖出信号
                # 计算止损止盈
                atr = self._calculate_atr(df, period=14)
                sl = current_price + (2 * atr)
                tp = current_price - (3 * atr)
                
                signal = Signal(
                    strategy_id=self.name,
                    symbol=market_data.symbol,
                    direction=-1,  # 卖出
                    strength=0.8,
                    entry_price=current_price,
                    sl=sl,
                    tp=tp,
                    size=0.1,
                    confidence=0.75,
                    timestamp=market_data.timestamp,
                    metadata={
                        'ma_fast': ma_fast_current,
                        'ma_slow': ma_slow_current,
                        'rsi': rsi_current,
                        'signal_type': 'bearish_cross'
                    }
                )
                
                logger.info(f"生成卖出信号: {market_data.symbol} @ {current_price:.5f}")
            
            return signal
            
        except Exception as e:
            logger.error(f"生成信号失败: {e}")
            return None
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """计算ATR (Average True Range)"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        return atr
    
    def update_parameters(self, params: dict) -> None:
        """更新策略参数"""
        if 'fast_period' in params:
            self.fast_period = params['fast_period']
            logger.info(f"更新快速MA周期: {self.fast_period}")
        
        if 'slow_period' in params:
            self.slow_period = params['slow_period']
            logger.info(f"更新慢速MA周期: {self.slow_period}")
        
        if 'rsi_period' in params:
            self.rsi_period = params['rsi_period']
            logger.info(f"更新RSI周期: {self.rsi_period}")
    
    def get_performance_metrics(self):
        """获取策略性能指标"""
        # 这里简化处理，实际应该从数据库获取历史交易记录
        logger.info("获取性能指标（示例）")
        return None


def main():
    """主函数"""
    logger.info("=== 自定义策略示例 ===")
    
    # 1. 创建策略实例
    logger.info("创建策略实例")
    strategy = SimpleMAStrategy(
        fast_period=10,
        slow_period=30,
        rsi_period=14
    )
    
    # 2. 获取市场数据
    logger.info("获取市场数据")
    data_pipeline = DataPipeline()
    
    try:
        market_data = data_pipeline.get_realtime_data("EURUSD", "H1")
        logger.info(f"获取数据成功: {len(market_data.ohlcv)} 根K线")
    except Exception as e:
        logger.error(f"获取数据失败: {e}")
        return
    
    # 3. 生成信号
    logger.info("生成交易信号")
    signal = strategy.generate_signal(market_data)
    
    if signal:
        logger.info("=" * 50)
        logger.info("交易信号详情:")
        logger.info(f"  策略: {signal.strategy_id}")
        logger.info(f"  品种: {signal.symbol}")
        logger.info(f"  方向: {'买入' if signal.direction == 1 else '卖出'}")
        logger.info(f"  入场价: {signal.entry_price:.5f}")
        logger.info(f"  止损: {signal.sl:.5f} (风险: {abs(signal.entry_price - signal.sl):.5f})")
        logger.info(f"  止盈: {signal.tp:.5f} (收益: {abs(signal.tp - signal.entry_price):.5f})")
        logger.info(f"  风险收益比: {abs(signal.tp - signal.entry_price) / abs(signal.entry_price - signal.sl):.2f}")
        logger.info(f"  手数: {signal.size}")
        logger.info(f"  信号强度: {signal.strength:.2f}")
        logger.info(f"  置信度: {signal.confidence:.2f}")
        logger.info(f"  元数据: {signal.metadata}")
        logger.info("=" * 50)
    else:
        logger.info("当前无交易信号")
    
    # 4. 演示参数更新
    logger.info("\n演示参数更新")
    strategy.update_parameters({
        'fast_period': 15,
        'slow_period': 40
    })
    
    # 5. 再次生成信号
    logger.info("使用新参数生成信号")
    signal2 = strategy.generate_signal(market_data)
    
    if signal2:
        logger.info(f"新信号: {signal2.symbol} {'买入' if signal2.direction == 1 else '卖出'}")
    else:
        logger.info("使用新参数后无交易信号")
    
    logger.info("\n=== 示例完成 ===")
    logger.info("\n如何使用自定义策略:")
    logger.info("1. 继承Strategy基类")
    logger.info("2. 实现generate_signal()方法")
    logger.info("3. 实现update_parameters()方法")
    logger.info("4. 在策略管理器中注册")
    logger.info("5. 在配置文件中启用")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n程序被用户中断")
    except Exception as e:
        logger.error(f"程序异常: {e}", exc_info=True)
