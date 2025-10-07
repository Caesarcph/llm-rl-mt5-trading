#!/usr/bin/env python3
"""
生成测试数据脚本
用于生成模拟的市场数据，方便测试和演示
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_ohlc_data(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    timeframe: str = "H1",
    initial_price: float = 1.1000,
    volatility: float = 0.001
) -> pd.DataFrame:
    """
    生成OHLC数据
    
    Args:
        symbol: 品种名称
        start_date: 开始日期
        end_date: 结束日期
        timeframe: 时间周期
        initial_price: 初始价格
        volatility: 波动率
        
    Returns:
        DataFrame包含OHLC数据
    """
    logger.info(f"生成 {symbol} {timeframe} 数据...")
    
    # 计算时间间隔
    timeframe_minutes = {
        "M1": 1,
        "M5": 5,
        "M15": 15,
        "M30": 30,
        "H1": 60,
        "H4": 240,
        "D1": 1440
    }
    
    minutes = timeframe_minutes.get(timeframe, 60)
    
    # 生成时间序列
    time_range = pd.date_range(start=start_date, end=end_date, freq=f"{minutes}min")
    n_bars = len(time_range)
    
    logger.info(f"生成 {n_bars} 根K线")
    
    # 生成价格数据（随机游走）
    returns = np.random.normal(0, volatility, n_bars)
    prices = initial_price * np.exp(np.cumsum(returns))
    
    # 生成OHLC
    data = []
    for i, (timestamp, close) in enumerate(zip(time_range, prices)):
        # 生成高低价
        high_offset = abs(np.random.normal(0, volatility * initial_price))
        low_offset = abs(np.random.normal(0, volatility * initial_price))
        
        open_price = prices[i-1] if i > 0 else initial_price
        high = max(open_price, close) + high_offset
        low = min(open_price, close) - low_offset
        
        # 生成成交量
        volume = int(np.random.uniform(100, 1000))
        
        data.append({
            'time': timestamp,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    logger.info(f"数据生成完成: {len(df)} 行")
    
    return df


def generate_symbol_data(symbol: str, days: int = 90):
    """生成单个品种的测试数据"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # 根据品种设置初始价格和波动率
    symbol_params = {
        "EURUSD": {"initial_price": 1.1000, "volatility": 0.0005},
        "GBPUSD": {"initial_price": 1.2500, "volatility": 0.0006},
        "USDJPY": {"initial_price": 110.00, "volatility": 0.0008},
        "XAUUSD": {"initial_price": 1800.0, "volatility": 0.01},
        "USOIL": {"initial_price": 75.00, "volatility": 0.02},
        "BTCUSD": {"initial_price": 40000.0, "volatility": 0.03}
    }
    
    params = symbol_params.get(symbol, {"initial_price": 1.0, "volatility": 0.001})
    
    # 生成不同时间周期的数据
    timeframes = ["M5", "M15", "H1", "H4", "D1"]
    
    for timeframe in timeframes:
        df = generate_ohlc_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            **params
        )
        
        # 保存数据
        output_dir = "data/test"
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"{output_dir}/{symbol}_{timeframe}_{days}days.csv"
        df.to_csv(filename, index=False)
        logger.info(f"保存数据: {filename}")


def generate_all_test_data():
    """生成所有品种的测试数据"""
    logger.info("=" * 60)
    logger.info("生成测试数据")
    logger.info("=" * 60)
    
    symbols = ["EURUSD", "GBPUSD", "XAUUSD", "USOIL"]
    days = 90
    
    for symbol in symbols:
        logger.info(f"\n处理 {symbol}...")
        generate_symbol_data(symbol, days)
    
    logger.info("\n" + "=" * 60)
    logger.info("测试数据生成完成")
    logger.info("=" * 60)
    logger.info(f"\n数据保存在: data/test/")
    logger.info(f"品种: {', '.join(symbols)}")
    logger.info(f"时间范围: 最近{days}天")
    logger.info(f"时间周期: M5, M15, H1, H4, D1")


def generate_sample_trades():
    """生成示例交易记录"""
    logger.info("\n生成示例交易记录...")
    
    trades = []
    start_date = datetime.now() - timedelta(days=30)
    
    for i in range(50):
        # 随机生成交易
        symbol = np.random.choice(["EURUSD", "GBPUSD", "XAUUSD"])
        trade_type = np.random.choice(["BUY", "SELL"])
        
        open_time = start_date + timedelta(hours=np.random.randint(0, 720))
        close_time = open_time + timedelta(hours=np.random.randint(1, 48))
        
        open_price = np.random.uniform(1.0, 2.0)
        close_price = open_price * (1 + np.random.normal(0, 0.01))
        
        volume = round(np.random.uniform(0.01, 0.5), 2)
        
        if trade_type == "BUY":
            profit = (close_price - open_price) * volume * 100000
        else:
            profit = (open_price - close_price) * volume * 100000
        
        trades.append({
            'trade_id': f"T{i+1:04d}",
            'symbol': symbol,
            'type': trade_type,
            'volume': volume,
            'open_price': round(open_price, 5),
            'close_price': round(close_price, 5),
            'open_time': open_time,
            'close_time': close_time,
            'profit': round(profit, 2),
            'strategy': np.random.choice(["trend", "scalp", "breakout"])
        })
    
    df = pd.DataFrame(trades)
    
    # 保存
    output_file = "data/test/sample_trades.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"示例交易记录已保存: {output_file}")
    
    # 显示统计
    logger.info(f"\n交易统计:")
    logger.info(f"  总交易数: {len(df)}")
    logger.info(f"  盈利交易: {len(df[df['profit'] > 0])}")
    logger.info(f"  亏损交易: {len(df[df['profit'] < 0])}")
    logger.info(f"  总盈亏: ${df['profit'].sum():.2f}")
    logger.info(f"  平均盈亏: ${df['profit'].mean():.2f}")


def main():
    """主函数"""
    print("\n请选择要生成的测试数据:")
    print("1. 市场数据 (OHLC)")
    print("2. 交易记录")
    print("3. 全部生成")
    
    choice = input("\n请输入选项 (1-3): ").strip()
    
    if choice == "1":
        generate_all_test_data()
    elif choice == "2":
        generate_sample_trades()
    elif choice == "3":
        generate_all_test_data()
        generate_sample_trades()
    else:
        logger.error("无效选项")
        return
    
    logger.info("\n完成！")
    logger.info("\n使用方法:")
    logger.info("1. 测试数据位于 data/test/ 目录")
    logger.info("2. 可以在回测中使用这些数据")
    logger.info("3. 可以用于演示和测试功能")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n程序被用户中断")
    except Exception as e:
        logger.error(f"程序异常: {e}", exc_info=True)
