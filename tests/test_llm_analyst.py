"""
LLM分析Agent测试
测试新闻情绪分析、市场评论生成和事件影响分析功能
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.agents.llm_analyst import (
    LLMAnalystAgent,
    NewsItem,
    SentimentAnalysis,
    SentimentType,
    ImpactLevel,
    EconomicEvent,
    EventImpact,
    MarketCommentary,
    NewsPreprocessor,
    NewsScraper
)
from src.core.models import MarketData


class TestNewsPreprocessor(unittest.TestCase):
    """测试新闻预处理器"""
    
    def setUp(self):
        self.preprocessor = NewsPreprocessor()
    
    def test_preprocess_text(self):
        """测试文本预处理"""
        text = "Gold PRICES Surge 5%! Amazing Rally!!!"
        processed = self.preprocessor.preprocess_text(text)
        
        self.assertIsInstance(processed, str)
        self.assertEqual(processed.lower(), processed)
        self.assertNotIn('!', processed)
    
    def test_extract_keywords(self):
        """测试关键词提取"""
        text = "Federal Reserve raises interest rates amid inflation concerns"
        keywords = self.preprocessor.extract_keywords(text, top_n=5)
        
        self.assertIsInstance(keywords, list)
        self.assertGreater(len(keywords), 0)
        self.assertTrue(all(isinstance(k, str) for k in keywords))
    
    def test_calculate_sentiment_score_bullish(self):
        """测试看涨情绪分数计算"""
        text = "Market rally continues with strong gains and positive outlook"
        score = self.preprocessor.calculate_sentiment_score(text)
        
        self.assertGreater(score, 0)
        self.assertLessEqual(score, 1.0)
    
    def test_calculate_sentiment_score_bearish(self):
        """测试看跌情绪分数计算"""
        text = "Market crash continues with heavy losses and negative outlook"
        score = self.preprocessor.calculate_sentiment_score(text)
        
        self.assertLess(score, 0)
        self.assertGreaterEqual(score, -1.0)
    
    def test_calculate_sentiment_score_neutral(self):
        """测试中性情绪分数计算"""
        text = "Market remains stable with no significant changes"
        score = self.preprocessor.calculate_sentiment_score(text)
        
        self.assertAlmostEqual(score, 0.0, delta=0.3)
    
    def test_assess_impact_level_high(self):
        """测试高影响级别评估"""
        text = "Fed announces emergency interest rate cut amid inflation crisis"
        impact = self.preprocessor.assess_impact_level(text)
        
        self.assertIn(impact, [ImpactLevel.HIGH, ImpactLevel.MEDIUM])
    
    def test_assess_impact_level_low(self):
        """测试低影响级别评估"""
        text = "Minor technical adjustment in trading hours"
        impact = self.preprocessor.assess_impact_level(text)
        
        self.assertEqual(impact, ImpactLevel.LOW)
    
    def test_extract_symbols(self):
        """测试品种提取"""
        text = "EUR/USD rallies while Gold (XAU/USD) drops on strong dollar"
        symbols = self.preprocessor.extract_symbols(text)
        
        self.assertIsInstance(symbols, list)
        self.assertGreater(len(symbols), 0)


class TestNewsScraper(unittest.TestCase):
    """测试新闻抓取器"""
    
    def setUp(self):
        self.scraper = NewsScraper()
    
    def test_fetch_latest_news(self):
        """测试获取最新新闻"""
        news_items = self.scraper.fetch_latest_news(max_items=5)
        
        self.assertIsInstance(news_items, list)
        self.assertLessEqual(len(news_items), 5)
        
        for item in news_items:
            self.assertIsInstance(item, NewsItem)
            self.assertIsInstance(item.headline, str)
            self.assertIsInstance(item.timestamp, datetime)
    
    def test_fetch_news_with_symbols(self):
        """测试按品种获取新闻"""
        symbols = ['EURUSD', 'XAUUSD']
        news_items = self.scraper.fetch_latest_news(symbols=symbols, max_items=10)
        
        self.assertIsInstance(news_items, list)
        self.assertGreater(len(news_items), 0)


class TestLLMAnalystAgent(unittest.TestCase):
    """测试LLM分析Agent"""
    
    def setUp(self):
        """设置测试环境"""
        self.config = {
            'model_path': None,  # 不加载实际模型
            'auto_load_model': False,
            'use_llm_for_sentiment': False,  # 使用基础分析
            'min_call_interval': 0.1,
            'cache_ttl': 60
        }
        self.agent = LLMAnalystAgent(config=self.config)
        
        # 创建测试数据
        self.test_news = NewsItem(
            headline="Fed signals rate cut as inflation cools",
            content="The Federal Reserve indicated potential interest rate cuts...",
            source="TestSource",
            timestamp=datetime.now(),
            symbols=['EURUSD', 'XAUUSD'],
            category="central_bank"
        )
        
        self.test_market_data = self._create_test_market_data()
    
    def _create_test_market_data(self) -> MarketData:
        """创建测试市场数据"""
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
            symbol='EURUSD',
            timeframe='1H',
            timestamp=datetime.now(),
            ohlcv=ohlcv,
            indicators={'volatility': 0.015},
            spread=0.00002
        )
    
    def test_initialization(self):
        """测试Agent初始化"""
        self.assertIsNotNone(self.agent)
        self.assertIsNotNone(self.agent.news_scraper)
        self.assertIsNotNone(self.agent.preprocessor)
        self.assertIsInstance(self.agent.sentiment_cache, dict)
        self.assertIsInstance(self.agent.commentary_cache, dict)
    
    def test_analyze_news_sentiment_bullish(self):
        """测试看涨新闻情绪分析"""
        news = NewsItem(
            headline="Market rallies on strong economic data and positive outlook",
            content="Markets surged today with gains across all sectors...",
            source="TestSource",
            timestamp=datetime.now()
        )
        
        result = self.agent.analyze_news_sentiment(news, symbol='EURUSD')
        
        self.assertIsInstance(result, SentimentAnalysis)
        self.assertIn(result.sentiment, [SentimentType.BULLISH, SentimentType.NEUTRAL])
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        self.assertIsInstance(result.impact_level, ImpactLevel)
        self.assertIsInstance(result.explanation, str)
        self.assertGreater(len(result.explanation), 0)
    
    def test_analyze_news_sentiment_bearish(self):
        """测试看跌新闻情绪分析"""
        news = NewsItem(
            headline="Market crashes on recession fears and negative data",
            content="Markets plunged today with losses across all sectors...",
            source="TestSource",
            timestamp=datetime.now()
        )
        
        result = self.agent.analyze_news_sentiment(news, symbol='EURUSD')
        
        self.assertIsInstance(result, SentimentAnalysis)
        self.assertIn(result.sentiment, [SentimentType.BEARISH, SentimentType.NEUTRAL])
        self.assertLess(result.score, 0.5)
    
    def test_analyze_news_sentiment_neutral(self):
        """测试中性新闻情绪分析"""
        news = NewsItem(
            headline="Market remains stable with mixed signals",
            content="Markets showed little movement today...",
            source="TestSource",
            timestamp=datetime.now()
        )
        
        result = self.agent.analyze_news_sentiment(news, symbol='EURUSD')
        
        self.assertIsInstance(result, SentimentAnalysis)
        # 中性新闻可能被判断为任何情绪，但分数应该接近0
        self.assertAlmostEqual(result.score, 0.0, delta=0.5)
    
    def test_sentiment_analysis_caching(self):
        """测试情绪分析缓存"""
        # 第一次分析
        result1 = self.agent.analyze_news_sentiment(self.test_news, symbol='EURUSD')
        
        # 第二次分析（应该从缓存获取）
        result2 = self.agent.analyze_news_sentiment(self.test_news, symbol='EURUSD')
        
        # 结果应该相同
        self.assertEqual(result1.sentiment, result2.sentiment)
        self.assertEqual(result1.confidence, result2.confidence)
        self.assertEqual(result1.score, result2.score)
    
    def test_generate_market_commentary(self):
        """测试市场评论生成"""
        technical_signals = {
            'rsi_signal': 0.3,
            'macd_signal': 1.0,
            'trend_strength': 0.8
        }
        
        result = self.agent.generate_market_commentary(
            symbol='EURUSD',
            market_data=self.test_market_data,
            technical_signals=technical_signals
        )
        
        self.assertIsInstance(result, MarketCommentary)
        self.assertEqual(result.symbol, 'EURUSD')
        self.assertIsInstance(result.summary, str)
        self.assertGreater(len(result.summary), 0)
        self.assertIsInstance(result.key_points, list)
        self.assertGreater(len(result.key_points), 0)
        self.assertIsInstance(result.market_regime, str)
        self.assertIsInstance(result.risk_assessment, str)
        self.assertIsInstance(result.trading_recommendation, str)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
    
    def test_commentary_caching(self):
        """测试评论缓存"""
        # 第一次生成
        result1 = self.agent.generate_market_commentary(
            symbol='EURUSD',
            market_data=self.test_market_data
        )
        
        # 第二次生成（应该从缓存获取）
        result2 = self.agent.generate_market_commentary(
            symbol='EURUSD',
            market_data=self.test_market_data
        )
        
        # 结果应该相同
        self.assertEqual(result1.summary, result2.summary)
        self.assertEqual(result1.confidence, result2.confidence)
    
    def test_analyze_economic_events(self):
        """测试经济事件分析"""
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
                name="CPI",
                country="US",
                timestamp=datetime.now(),
                importance="high",
                actual=3.5,
                forecast=3.2,
                previous=3.0,
                currency="USD"
            )
        ]
        
        results = self.agent.analyze_economic_events(events, symbol='EURUSD')
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        
        for result in results:
            self.assertIsInstance(result, EventImpact)
            self.assertIsInstance(result.event, EconomicEvent)
            self.assertIsInstance(result.affected_symbols, list)
            self.assertIsInstance(result.impact_assessment, str)
            self.assertIsInstance(result.sentiment, SentimentType)
            self.assertGreaterEqual(result.confidence, 0.0)
            self.assertLessEqual(result.confidence, 1.0)
            self.assertIsInstance(result.recommended_actions, list)
    
    def test_economic_event_surprise_factor(self):
        """测试经济事件意外因子"""
        # 正面意外
        positive_event = EconomicEvent(
            name="GDP",
            country="US",
            timestamp=datetime.now(),
            importance="high",
            actual=3.0,
            forecast=2.0,
            previous=1.5,
            currency="USD"
        )
        
        surprise = positive_event.get_surprise_factor()
        self.assertIsNotNone(surprise)
        self.assertGreater(surprise, 0)
        
        # 负面意外
        negative_event = EconomicEvent(
            name="Unemployment",
            country="US",
            timestamp=datetime.now(),
            importance="high",
            actual=5.0,
            forecast=4.0,
            previous=4.2,
            currency="USD"
        )
        
        surprise = negative_event.get_surprise_factor()
        self.assertIsNotNone(surprise)
        self.assertGreater(surprise, 0)
    
    def test_fetch_and_analyze_news(self):
        """测试获取并分析新闻"""
        symbols = ['EURUSD', 'XAUUSD']
        results = self.agent.fetch_and_analyze_news(symbols=symbols, max_items=5)
        
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        for news_item, sentiment in results:
            self.assertIsInstance(news_item, NewsItem)
            self.assertIsInstance(sentiment, SentimentAnalysis)
    
    def test_generate_daily_summary(self):
        """测试每日总结生成"""
        date = datetime.now()
        performance_data = {
            'total_trades': 10,
            'win_rate': 0.6,
            'profit': 1500.0,
            'max_drawdown': -200.0
        }
        events = [
            "Fed kept rates unchanged",
            "Strong employment data released",
            "EUR/USD reached new high"
        ]
        
        summary = self.agent.generate_daily_summary(date, performance_data, events)
        
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)
        self.assertIn(date.strftime('%Y-%m-%d'), summary)
        self.assertIn('Performance', summary)
    
    def test_sentiment_is_actionable(self):
        """测试情绪可操作性判断"""
        # 高置信度高影响
        actionable_sentiment = SentimentAnalysis(
            sentiment=SentimentType.BULLISH,
            confidence=0.9,
            impact_level=ImpactLevel.HIGH,
            explanation="Strong bullish signal",
            score=0.8
        )
        self.assertTrue(actionable_sentiment.is_actionable())
        
        # 低置信度
        non_actionable_sentiment = SentimentAnalysis(
            sentiment=SentimentType.BULLISH,
            confidence=0.3,
            impact_level=ImpactLevel.HIGH,
            explanation="Weak signal",
            score=0.2
        )
        self.assertFalse(non_actionable_sentiment.is_actionable())
        
        # 低影响
        low_impact_sentiment = SentimentAnalysis(
            sentiment=SentimentType.BULLISH,
            confidence=0.9,
            impact_level=ImpactLevel.LOW,
            explanation="High confidence but low impact",
            score=0.7
        )
        self.assertFalse(low_impact_sentiment.is_actionable())
    
    def test_clear_cache(self):
        """测试清空缓存"""
        # 添加一些缓存数据
        self.agent.analyze_news_sentiment(self.test_news, symbol='EURUSD')
        self.agent.generate_market_commentary('EURUSD', self.test_market_data)
        
        self.assertGreater(len(self.agent.sentiment_cache), 0)
        self.assertGreater(len(self.agent.commentary_cache), 0)
        
        # 清空缓存
        self.agent.clear_cache()
        
        self.assertEqual(len(self.agent.sentiment_cache), 0)
        self.assertEqual(len(self.agent.commentary_cache), 0)
    
    def test_get_statistics(self):
        """测试获取统计信息"""
        stats = self.agent.get_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('sentiment_cache_size', stats)
        self.assertIn('commentary_cache_size', stats)
        self.assertIn('model_loaded', stats)
        self.assertIn('last_call_time', stats)
    
    def test_rate_limiting(self):
        """测试调用频率限制"""
        import time
        
        # 设置较短的间隔用于测试
        self.agent.min_call_interval = 0.5
        
        start_time = time.time()
        
        # 第一次调用
        self.agent._check_rate_limit()
        first_call_time = time.time()
        
        # 第二次调用（应该被限制）
        self.agent._check_rate_limit()
        second_call_time = time.time()
        
        # 验证时间间隔
        time_diff = second_call_time - first_call_time
        self.assertGreaterEqual(time_diff, self.agent.min_call_interval * 0.9)  # 允许10%误差


class TestSentimentAnalysisAccuracy(unittest.TestCase):
    """测试情绪分析准确性"""
    
    def setUp(self):
        self.config = {
            'model_path': None,
            'auto_load_model': False,
            'use_llm_for_sentiment': False,
            'min_call_interval': 0.0
        }
        self.agent = LLMAnalystAgent(config=self.config)
    
    def test_bullish_news_accuracy(self):
        """测试看涨新闻识别准确性"""
        bullish_headlines = [
            "Market rallies on strong earnings",
            "Gold surges to new highs",
            "Positive economic data boosts sentiment",
            "Strong gains across all sectors",
            "Bullish outlook drives market higher"
        ]
        
        bullish_count = 0
        for headline in bullish_headlines:
            news = NewsItem(
                headline=headline,
                content=headline,
                source="Test",
                timestamp=datetime.now()
            )
            result = self.agent.analyze_news_sentiment(news)
            if result.sentiment == SentimentType.BULLISH or result.score > 0:
                bullish_count += 1
        
        # 至少60%的看涨新闻应该被正确识别
        accuracy = bullish_count / len(bullish_headlines)
        self.assertGreaterEqual(accuracy, 0.6, 
                               f"Bullish accuracy: {accuracy:.2%}")
    
    def test_bearish_news_accuracy(self):
        """测试看跌新闻识别准确性"""
        bearish_headlines = [
            "Market crashes on recession fears",
            "Gold plunges on strong dollar",
            "Negative economic data weighs on sentiment",
            "Heavy losses across all sectors",
            "Bearish outlook drives market lower"
        ]
        
        bearish_count = 0
        for headline in bearish_headlines:
            news = NewsItem(
                headline=headline,
                content=headline,
                source="Test",
                timestamp=datetime.now()
            )
            result = self.agent.analyze_news_sentiment(news)
            if result.sentiment == SentimentType.BEARISH or result.score < 0:
                bearish_count += 1
        
        # 至少60%的看跌新闻应该被正确识别
        accuracy = bearish_count / len(bearish_headlines)
        self.assertGreaterEqual(accuracy, 0.6,
                               f"Bearish accuracy: {accuracy:.2%}")
    
    def test_impact_level_accuracy(self):
        """测试影响级别评估准确性"""
        high_impact_news = [
            "Fed announces emergency rate cut",
            "Central bank policy shift impacts markets",
            "Major inflation data surprises markets"
        ]
        
        low_impact_news = [
            "Minor technical adjustment announced",
            "Routine market update",
            "Small company reports earnings"
        ]
        
        # 测试高影响新闻
        high_impact_correct = 0
        for headline in high_impact_news:
            news = NewsItem(
                headline=headline,
                content=headline,
                source="Test",
                timestamp=datetime.now()
            )
            result = self.agent.analyze_news_sentiment(news)
            if result.impact_level in [ImpactLevel.HIGH, ImpactLevel.MEDIUM]:
                high_impact_correct += 1
        
        high_accuracy = high_impact_correct / len(high_impact_news)
        self.assertGreaterEqual(high_accuracy, 0.6,
                               f"High impact accuracy: {high_accuracy:.2%}")
        
        # 测试低影响新闻
        low_impact_correct = 0
        for headline in low_impact_news:
            news = NewsItem(
                headline=headline,
                content=headline,
                source="Test",
                timestamp=datetime.now()
            )
            result = self.agent.analyze_news_sentiment(news)
            if result.impact_level == ImpactLevel.LOW:
                low_impact_correct += 1
        
        low_accuracy = low_impact_correct / len(low_impact_news)
        self.assertGreaterEqual(low_accuracy, 0.5,
                               f"Low impact accuracy: {low_accuracy:.2%}")


if __name__ == '__main__':
    unittest.main()
