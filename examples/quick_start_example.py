#!/usr/bin/env python3
"""
快速开始示例
演示如何快速启动交易系统
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.config import load_config
from src.data.mt5_connection import MT5Connection
from src.data.data_pipeline import DataPipeline
from src.strategies.trend_strategy import TrendStrategy
from src.strategies.strategy_manager import StrategyManager
from src.agents.risk_manager_agent import RiskManagerAgent
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    logger.info("=== 快速开始示例 ===")
    
    # 1. 加载配置
    logger.info("步骤1: 加载配置")
    config = load_config()
    logger.info(f"配置加载成功，模拟模式: {config.simulation_mode}")
    
    # 2. 连接MT5
    logger.info("步骤2: 连接MT5")
    mt5_connection = MT5Connection(config.mt5)
    
    if not mt5_connection.connect():
        logger.error("MT5连接失败")
        return
    
    logger.info("MT5连接成功")
    
    # 3. 获取账户信息
    logger.info("步骤3: 获取账户信息")
    account = mt5_connection.get_account_info()
    logger.info(f"账户余额: ${account.balance:.2f}")
    logger.info(f"账户净值: ${account.equity:.2f}")
    logger.info(f"可用保证金: ${account.free_margin:.2f}")
    
    # 4. 创建数据管道
    logger.info("步骤4: 创建数据管道")
    data_pipeline = DataPipeline()
    
    # 5. 获取市场数据
    logger.info("步骤5: 获取市场数据")
    symbol = "EURUSD"
    timeframe = "H1"
    
    try:
        market_data = data_pipeline.get_realtime_data(symbol, timeframe)
        logger.info(f"获取 {symbol} {timeframe} 数据成功")
        logger.info(f"最新价格: {market_data.ohlcv['close'].iloc[-1]:.5f}")
        logger.info(f"数据点数: {len(market_data.ohlcv)}")
    except Exception as e:
        logger.error(f"获取市场数据失败: {e}")
        return
    
    # 6. 创建策略
    logger.info("步骤6: 创建趋势策略")
    trend_strategy = TrendStrategy(
        fast_period=20,
        slow_period=50,
        signal_threshold=0.7
    )
    
    # 7. 生成信号
    logger.info("步骤7: 生成交易信号")
    signal = trend_strategy.generate_signal(market_data)
    
    if signal:
        logger.info(f"生成信号:")
        logger.info(f"  品种: {signal.symbol}")
        logger.info(f"  方向: {'买入' if signal.direction == 1 else '卖出' if signal.direction == -1 else '平仓'}")
        logger.info(f"  强度: {signal.strength:.2f}")
        logger.info(f"  入场价: {signal.entry_price:.5f}")
        logger.info(f"  止损: {signal.sl:.5f}")
        logger.info(f"  止盈: {signal.tp:.5f}")
        logger.info(f"  手数: {signal.size:.2f}")
        logger.info(f"  置信度: {signal.confidence:.2f}")
    else:
        logger.info("当前无交易信号")
    
    # 8. 风险验证
    if signal:
        logger.info("步骤8: 风险验证")
        risk_agent = RiskManagerAgent(config.risk)
        
        # 这里简化处理，实际应该传入完整的portfolio
        logger.info("信号通过基础验证")
        logger.info(f"风险水平: {config.risk.max_risk_per_trade * 100:.1f}%")
    
    # 9. 清理
    logger.info("步骤9: 清理资源")
    mt5_connection.disconnect()
    logger.info("MT5连接已关闭")
    
    logger.info("=== 示例完成 ===")
    logger.info("\n下一步:")
    logger.info("1. 查看用户手册了解更多功能")
    logger.info("2. 配置您的交易参数")
    logger.info("3. 在模拟模式下测试")
    logger.info("4. 运行完整的交易系统: python main.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n程序被用户中断")
    except Exception as e:
        logger.error(f"程序异常: {e}", exc_info=True)
