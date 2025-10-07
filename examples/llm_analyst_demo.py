"""
LLM分析Agent演示
展示如何使用LLM分析Agent进行新闻情绪分析、市场评论生成和事件影响分析
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.llm_analyst import (
    LLMAnalystAgent,
    NewsItem,
    EconomicEvent,
    SentimentType,
    ImpactLevel
)
from src.core.models import MarketData


def create_sample_market_data(symbol: str = 'EURUSD') -> MarketData:
    """创建示例市场数据"""
    dates = pd.date_range(end=datetime.now(), periods=100, freq='1h')
    
    # 生成模拟价格数据
    np.random.seed(42)
    close_prices = 1.1000 + np.cumsum(np.random.randn(100) * 0.0001)
    
    ohlcv = pd.DataFrame({
        'open': close_prices + np.random.randn(100) * 0.0001,
        'high': close_prices + abs(np.random.randn(100) * 0.0002),
        'low': close_prices - abs(np.random.randn(100) * 0.0002),
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    return MarketData(
        symbol=symbol,
        timeframe='1H',
        timestamp=datetime.now(),
        ohlcv=ohlcv,
        indicators={'volatility': 0.015, 'atr': 0.0012},
        spread=0.00002
    )


def demo_news_sentiment_analysis():
    """演示新闻情绪分析"""
    print("=" * 80)
    print("新闻情绪分析演示")
    print("=" * 80)
    
    # 初始化Agent（不使用LLM模型，仅使用基础分析）
    config = {
        'model_path': None,
        'use_llm_for_sentiment': False,
        'min_call_interval': 0.1
    }
    agent = LLMAnalystAgent(config=config)
    
    # 创建测试新闻
    news_items = [
        NewsItem(
            headline="Fed signals potential rate cut as inflation cools",
            content="The Federal Reserve indicated it may cut interest rates...",
            source="Reuters",
            timestamp=datetime.now(),
            symbols=['EURUSD', 'XAUUSD']
        ),
        NewsItem(
            headline="Gold prices surge on safe-haven demand",
            content="Gold rallied to new highs as investors seek safety...",
            source="Bloomberg",
            timestamp=datetime.now() - timedelta(hours=1),
            symbols=['XAUUSD']
        ),
        NewsItem(
            headline="EUR/USD drops on weak European economic data",
            content="The euro fell against the dollar after disappointing data...",
            source="FX Street",
            timestamp=datetime.now() - timedelta(hours=2),
            symbols=['EURUSD']
        )
    ]
    
    # 分析每条新闻
    for i, news in enumerate(news_items, 1):
        print(f"\n新闻 {i}:")
        print(f"标题: {news.headline}")
        print(f"来源: {news.source}")
        print(f"时间: {news.timestamp.strftime('%Y-%m-%d %H:%M')}")
        print(f"相关品种: {', '.join(news.symbols)}")
        
        # 进行情绪分析
        sentiment = agent.analyze_news_sentiment(news, symbol='EURUSD')
        
        print(f"\n情绪分析结果:")
        print(f"  情绪类型: {sentiment.sentiment.value}")
        print(f"  情绪分数: {sentiment.score:.2f} (-1到1)")
        print(f"  置信度: {sentiment.confidence:.2%}")
        print(f"  影响级别: {sentiment.impact_level.value}")
        print(f"  关键词: {', '.join(sentiment.keywords[:5])}")
        print(f"  解释: {sentiment.explanation}")
        print(f"  可操作: {'是' if sentiment.is_actionable() else '否'}")
    
    print("\n" + "=" * 80)


def demo_market_commentary():
    """演示市场评论生成"""
    print("\n" + "=" * 80)
    print("市场评论生成演示")
    print("=" * 80)
    
    # 初始化Agent
    config = {
        'model_path': None,
        'use_llm_for_sentiment': False
    }
    agent = LLMAnalystAgent(config=config)
    
    # 创建市场数据
    market_data = create_sample_market_data('EURUSD')
    
    # 技术信号
    technical_signals = {
        'rsi_signal': 0.3,      # 超卖
        'macd_signal': 1.0,     # 看涨
        'stoch_signal': 0.5,    # 中性
        'trend_strength': 0.8   # 强趋势
    }
    
    print(f"\n品种: {market_data.symbol}")
    print(f"时间框架: {market_data.timeframe}")
    print(f"当前价格: {market_data.ohlcv['close'].iloc[-1]:.5f}")
    print(f"波动率: {market_data.indicators.get('volatility', 0):.4f}")
    
    # 生成市场评论
    commentary = agent.generate_market_commentary(
        symbol='EURUSD',
        market_data=market_data,
        technical_signals=technical_signals
    )
    
    print(f"\n市场评论:")
    print(f"  摘要: {commentary.summary}")
    print(f"\n  关键点:")
    for point in commentary.key_points:
        print(f"    - {point}")
    print(f"\n  市场状态: {commentary.market_regime}")
    print(f"  风险评估: {commentary.risk_assessment}")
    print(f"  交易建议: {commentary.trading_recommendation}")
    print(f"  置信度: {commentary.confidence:.2%}")
    
    print("\n" + "=" * 80)


def demo_economic_events_analysis():
    """演示经济事件分析"""
    print("\n" + "=" * 80)
    print("经济事件影响分析演示")
    print("=" * 80)
    
    # 初始化Agent
    config = {
        'model_path': None,
        'use_llm_for_sentiment': False
    }
    agent = LLMAnalystAgent(config=config)
    
    # 创建经济事件
    events = [
        EconomicEvent(
            name="Non-Farm Payrolls",
            country="US",
            timestamp=datetime.now(),
            importance="high",
            actual=250000,
            forecast=200000,
            previous=180000,
            currency="USD"
        ),
        EconomicEvent(
            name="CPI (Consumer Price Index)",
            country="US",
            timestamp=datetime.now() - timedelta(hours=2),
            importance="high",
            actual=3.2,
            forecast=3.5,
            previous=3.7,
            currency="USD"
        ),
        EconomicEvent(
            name="ECB Interest Rate Decision",
            country="EU",
            timestamp=datetime.now() - timedelta(days=1),
            importance="high",
            actual=4.5,
            forecast=4.5,
            previous=4.25,
            currency="EUR"
        )
    ]
    
    # 分析事件影响
    impacts = agent.analyze_economic_events(events, symbol='EURUSD')
    
    for i, impact in enumerate(impacts, 1):
        event = impact.event
        print(f"\n事件 {i}: {event.name}")
        print(f"  国家: {event.country}")
        print(f"  重要性: {event.importance}")
        print(f"  时间: {event.timestamp.strftime('%Y-%m-%d %H:%M')}")
        print(f"  实际值: {event.actual}")
        print(f"  预期值: {event.forecast}")
        print(f"  前值: {event.previous}")
        
        surprise = event.get_surprise_factor()
        if surprise is not None:
            print(f"  意外因子: {surprise:+.2%}")
        
        print(f"\n  影响分析:")
        print(f"    受影响品种: {', '.join(impact.affected_symbols)}")
        print(f"    影响评估: {impact.impact_assessment}")
        print(f"    市场情绪: {impact.sentiment.value}")
        print(f"    置信度: {impact.confidence:.2%}")
        print(f"    风险级别: {impact.risk_level}")
        
        print(f"\n    建议操作:")
        for action in impact.recommended_actions:
            print(f"      - {action}")
    
    print("\n" + "=" * 80)


def demo_fetch_and_analyze_news():
    """演示获取并分析新闻"""
    print("\n" + "=" * 80)
    print("获取并分析新闻演示")
    print("=" * 80)
    
    # 初始化Agent
    config = {
        'model_path': None,
        'use_llm_for_sentiment': False
    }
    agent = LLMAnalystAgent(config=config)
    
    # 获取并分析新闻
    symbols = ['EURUSD', 'XAUUSD', 'USOIL']
    results = agent.fetch_and_analyze_news(symbols=symbols, max_items=5)
    
    print(f"\n获取到 {len(results)} 条新闻\n")
    
    # 统计情绪分布
    sentiment_counts = {
        SentimentType.BULLISH: 0,
        SentimentType.BEARISH: 0,
        SentimentType.NEUTRAL: 0
    }
    
    high_impact_count = 0
    actionable_count = 0
    
    for news, sentiment in results:
        sentiment_counts[sentiment.sentiment] += 1
        if sentiment.impact_level == ImpactLevel.HIGH:
            high_impact_count += 1
        if sentiment.is_actionable():
            actionable_count += 1
        
        print(f"标题: {news.headline}")
        print(f"  情绪: {sentiment.sentiment.value} (分数: {sentiment.score:+.2f})")
        print(f"  影响: {sentiment.impact_level.value}")
        print(f"  置信度: {sentiment.confidence:.2%}")
        print()
    
    print("情绪统计:")
    print(f"  看涨: {sentiment_counts[SentimentType.BULLISH]}")
    print(f"  看跌: {sentiment_counts[SentimentType.BEARISH]}")
    print(f"  中性: {sentiment_counts[SentimentType.NEUTRAL]}")
    print(f"\n高影响新闻: {high_impact_count}")
    print(f"可操作信号: {actionable_count}")
    
    print("\n" + "=" * 80)


def demo_daily_summary():
    """演示每日总结生成"""
    print("\n" + "=" * 80)
    print("每日市场总结演示")
    print("=" * 80)
    
    # 初始化Agent
    config = {
        'model_path': None,
        'use_llm_for_sentiment': False
    }
    agent = LLMAnalystAgent(config=config)
    
    # 准备性能数据
    date = datetime.now()
    performance_data = {
        'total_trades': 15,
        'winning_trades': 10,
        'losing_trades': 5,
        'win_rate': '66.7%',
        'total_profit': '$2,450.00',
        'max_drawdown': '-$320.00',
        'sharpe_ratio': 1.85,
        'best_symbol': 'EURUSD (+$850)',
        'worst_symbol': 'USOIL (-$180)'
    }
    
    events = [
        "Fed kept interest rates unchanged at 5.25-5.50%",
        "Strong US employment data exceeded expectations",
        "EUR/USD reached 3-month high on ECB hawkish comments",
        "Gold prices consolidated after recent rally",
        "Oil prices declined on oversupply concerns"
    ]
    
    # 生成每日总结
    summary = agent.generate_daily_summary(date, performance_data, events)
    
    print(f"\n{summary}")
    
    print("\n" + "=" * 80)


def demo_agent_statistics():
    """演示Agent统计信息"""
    print("\n" + "=" * 80)
    print("Agent统计信息演示")
    print("=" * 80)
    
    # 初始化Agent
    config = {
        'model_path': None,
        'use_llm_for_sentiment': False
    }
    agent = LLMAnalystAgent(config=config)
    
    # 执行一些操作以填充缓存
    news = NewsItem(
        headline="Test news",
        content="Test content",
        source="Test",
        timestamp=datetime.now()
    )
    agent.analyze_news_sentiment(news)
    
    market_data = create_sample_market_data()
    agent.generate_market_commentary('EURUSD', market_data)
    
    # 获取统计信息
    stats = agent.get_statistics()
    
    print("\nAgent统计信息:")
    print(f"  情绪分析缓存大小: {stats['sentiment_cache_size']}")
    print(f"  市场评论缓存大小: {stats['commentary_cache_size']}")
    print(f"  LLM模型已加载: {stats['model_loaded']}")
    print(f"  最后调用时间: {datetime.fromtimestamp(stats['last_call_time']).strftime('%Y-%m-%d %H:%M:%S') if stats['last_call_time'] > 0 else 'N/A'}")
    
    # 清空缓存
    print("\n清空缓存...")
    agent.clear_cache()
    
    stats_after = agent.get_statistics()
    print(f"  情绪分析缓存大小: {stats_after['sentiment_cache_size']}")
    print(f"  市场评论缓存大小: {stats_after['commentary_cache_size']}")
    
    print("\n" + "=" * 80)


def main():
    """主函数"""
    print("\n")
    print("*" * 80)
    print("LLM分析Agent演示程序")
    print("*" * 80)
    
    try:
        # 运行各个演示
        demo_news_sentiment_analysis()
        demo_market_commentary()
        demo_economic_events_analysis()
        demo_fetch_and_analyze_news()
        demo_daily_summary()
        demo_agent_statistics()
        
        print("\n" + "*" * 80)
        print("演示完成！")
        print("*" * 80)
        print("\n注意: 本演示使用基础关键词分析。")
        print("要启用完整的LLM功能，请:")
        print("  1. 下载Llama 3.2模型文件")
        print("  2. 在配置中设置model_path")
        print("  3. 设置use_llm_for_sentiment=True")
        print("\n")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
