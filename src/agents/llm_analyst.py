"""
LLM分析Agent
提供新闻情绪分析、市场评论生成和事件影响分析
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import re
import json

from src.llm.model_manager import ModelManager
from src.llm.llama_model import ModelConfig
from src.llm.call_optimizer import LLMCallOptimizer, CallPriority, LLMResultCache
from src.core.models import MarketData, Signal
from src.core.exceptions import AnalysisError


class SentimentType(Enum):
    """情绪类型"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class ImpactLevel(Enum):
    """影响级别"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


@dataclass
class NewsItem:
    """新闻条目"""
    headline: str
    content: str
    source: str
    timestamp: datetime
    symbols: List[str] = field(default_factory=list)
    category: str = ""
    url: str = ""
    
    def get_age_hours(self) -> float:
        """获取新闻年龄（小时）"""
        return (datetime.now() - self.timestamp).total_seconds() / 3600


@dataclass
class SentimentAnalysis:
    """情绪分析结果"""
    sentiment: SentimentType
    confidence: float  # 0-1
    impact_level: ImpactLevel
    explanation: str
    keywords: List[str] = field(default_factory=list)
    score: float = 0.0  # -1到1，负数看跌，正数看涨
    
    def is_actionable(self, min_confidence: float = 0.7, 
                     min_impact: ImpactLevel = ImpactLevel.MEDIUM) -> bool:
        """判断是否可操作"""
        impact_order = {ImpactLevel.NONE: 0, ImpactLevel.LOW: 1, 
                       ImpactLevel.MEDIUM: 2, ImpactLevel.HIGH: 3}
        return (self.confidence >= min_confidence and 
                impact_order[self.impact_level] >= impact_order[min_impact])


@dataclass
class EconomicEvent:
    """经济事件"""
    name: str
    country: str
    timestamp: datetime
    importance: str  # "high", "medium", "low"
    actual: Optional[float] = None
    forecast: Optional[float] = None
    previous: Optional[float] = None
    currency: str = ""
    
    def get_surprise_factor(self) -> Optional[float]:
        """计算意外因子"""
        if self.actual is not None and self.forecast is not None:
            if self.forecast != 0:
                return (self.actual - self.forecast) / abs(self.forecast)
        return None


@dataclass
class EventImpact:
    """事件影响分析"""
    event: EconomicEvent
    affected_symbols: List[str]
    impact_assessment: str
    sentiment: SentimentType
    confidence: float
    recommended_actions: List[str] = field(default_factory=list)
    risk_level: str = "medium"


@dataclass
class MarketCommentary:
    """市场评论"""
    symbol: str
    timestamp: datetime
    summary: str
    key_points: List[str]
    market_regime: str
    risk_assessment: str
    trading_recommendation: str
    confidence: float


class NewsPreprocessor:
    """新闻预处理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 关键词字典
        self.bullish_keywords = [
            'rally', 'surge', 'gain', 'rise', 'up', 'bullish', 'positive',
            'growth', 'increase', 'strong', 'boost', 'advance', 'climb',
            'recovery', 'optimistic', 'upgrade', 'beat', 'exceed'
        ]
        
        self.bearish_keywords = [
            'fall', 'drop', 'decline', 'down', 'bearish', 'negative',
            'loss', 'decrease', 'weak', 'plunge', 'crash', 'slump',
            'recession', 'pessimistic', 'downgrade', 'miss', 'below'
        ]
        
        self.high_impact_keywords = [
            'fed', 'central bank', 'interest rate', 'inflation', 'gdp',
            'employment', 'crisis', 'war', 'sanctions', 'policy',
            'earnings', 'bankruptcy', 'merger', 'acquisition'
        ]
    
    def preprocess_text(self, text: str) -> str:
        """预处理文本"""
        # 转小写
        text = text.lower()
        
        # 移除特殊字符（包括标点符号）
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """提取关键词"""
        text = self.preprocess_text(text)
        words = text.split()
        
        # 简单的关键词提取（基于词频）
        word_freq = {}
        for word in words:
            if len(word) > 3:  # 忽略短词
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # 排序并返回top N
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_n]]
    
    def calculate_sentiment_score(self, text: str) -> float:
        """计算情绪分数（-1到1）"""
        text = self.preprocess_text(text)
        
        bullish_count = sum(1 for keyword in self.bullish_keywords if keyword in text)
        bearish_count = sum(1 for keyword in self.bearish_keywords if keyword in text)
        
        total = bullish_count + bearish_count
        if total == 0:
            return 0.0
        
        return (bullish_count - bearish_count) / total
    
    def assess_impact_level(self, text: str) -> ImpactLevel:
        """评估影响级别"""
        text = self.preprocess_text(text)
        
        high_impact_count = sum(1 for keyword in self.high_impact_keywords if keyword in text)
        
        if high_impact_count >= 3:
            return ImpactLevel.HIGH
        elif high_impact_count >= 1:
            return ImpactLevel.MEDIUM
        else:
            return ImpactLevel.LOW
    
    def extract_symbols(self, text: str) -> List[str]:
        """从文本中提取交易品种"""
        text = text.upper()
        
        # 常见交易品种模式
        symbol_patterns = [
            r'\b(EUR/USD|EURUSD)\b',
            r'\b(GBP/USD|GBPUSD)\b',
            r'\b(USD/JPY|USDJPY)\b',
            r'\b(XAU/USD|XAUUSD|GOLD)\b',
            r'\b(XAG/USD|XAGUSD|SILVER)\b',
            r'\b(WTI|USOIL|CRUDE)\b',
            r'\b(BTC/USD|BTCUSD|BITCOIN)\b',
            r'\b(SPX|SP500|S&P 500)\b',
            r'\b(NAS100|NASDAQ)\b',
        ]
        
        symbols = []
        for pattern in symbol_patterns:
            matches = re.findall(pattern, text)
            symbols.extend(matches)
        
        return list(set(symbols))


class NewsScraper:
    """新闻数据抓取器（模拟实现）"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.preprocessor = NewsPreprocessor()
    
    def fetch_latest_news(self, symbols: Optional[List[str]] = None, 
                         max_items: int = 10) -> List[NewsItem]:
        """
        获取最新新闻
        
        注意：这是一个模拟实现。实际应用中应该集成真实的新闻API
        如：Alpha Vantage, NewsAPI, Bloomberg API等
        """
        self.logger.info(f"Fetching latest news for symbols: {symbols}")
        
        # 模拟新闻数据
        mock_news = self._generate_mock_news(symbols, max_items)
        
        return mock_news
    
    def _generate_mock_news(self, symbols: Optional[List[str]], 
                           max_items: int) -> List[NewsItem]:
        """生成模拟新闻数据"""
        mock_headlines = [
            "Fed signals potential rate cut amid economic slowdown",
            "Gold prices surge on safe-haven demand",
            "EUR/USD rallies as ECB maintains hawkish stance",
            "Oil prices drop on oversupply concerns",
            "US employment data beats expectations",
            "Central banks coordinate to stabilize markets",
            "Tech stocks lead market recovery",
            "Inflation data shows signs of cooling",
            "Geopolitical tensions impact currency markets",
            "Strong earnings boost investor confidence"
        ]
        
        news_items = []
        for i, headline in enumerate(mock_headlines[:max_items]):
            news_item = NewsItem(
                headline=headline,
                content=f"Full content of: {headline}",
                source="MockNewsSource",
                timestamp=datetime.now() - timedelta(hours=i),
                symbols=self.preprocessor.extract_symbols(headline),
                category="market_news"
            )
            news_items.append(news_item)
        
        return news_items


class LLMAnalystAgent:
    """LLM分析Agent"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        self.model_manager: Optional[ModelManager] = None
        self.news_scraper = NewsScraper()
        self.preprocessor = NewsPreprocessor()
        
        # 初始化调用优化器
        optimizer_config = {
            'base_interval': self.config.get('min_call_interval', 5.0),
            'min_interval': self.config.get('min_call_interval', 1.0),
            'max_interval': 30.0,
            'cache_ttl': self.config.get('cache_ttl', 3600),
            'cache_max_size': 1000,
            'max_concurrent': 3,
            'enable_cache': True,
            'enable_rate_limit': True
        }
        self.call_optimizer = LLMCallOptimizer(optimizer_config)
        
        # 保留旧缓存以兼容（将逐步迁移到call_optimizer）
        self.sentiment_cache: Dict[str, Tuple[SentimentAnalysis, float]] = {}
        self.commentary_cache: Dict[str, Tuple[MarketCommentary, float]] = {}
        
        # 调用频率控制（已由call_optimizer管理）
        self.last_call_time = 0.0
        self.min_call_interval = self.config.get('min_call_interval', 5.0)
        
        # 初始化模型管理器
        self._initialize_model_manager()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'model_path': 'models/llama-3.2-1b.gguf',
            'auto_load_model': False,
            'min_call_interval': 5.0,
            'cache_ttl': 3600,
            'sentiment_confidence_threshold': 0.6,
            'use_llm_for_sentiment': True,  # 是否使用LLM进行情绪分析
            'temperature': 0.5,
            'max_tokens': 200
        }
    
    def _initialize_model_manager(self) -> None:
        """初始化模型管理器"""
        try:
            model_path = self.config.get('model_path')
            if model_path:
                self.model_manager = ModelManager(
                    model_path=model_path,
                    auto_load=self.config.get('auto_load_model', False)
                )
                self.logger.info("Model manager initialized")
            else:
                self.logger.warning("No model path configured, LLM features will be limited")
        except Exception as e:
            self.logger.error(f"Failed to initialize model manager: {e}")
            self.model_manager = None
    
    def _check_rate_limit(self) -> bool:
        """检查调用频率限制"""
        current_time = time.time()
        if current_time - self.last_call_time < self.min_call_interval:
            wait_time = self.min_call_interval - (current_time - self.last_call_time)
            self.logger.debug(f"Rate limit: waiting {wait_time:.2f}s")
            time.sleep(wait_time)
        
        self.last_call_time = time.time()
        return True
    
    def _get_from_cache(self, cache_dict: Dict, key: str) -> Optional[Any]:
        """从缓存获取数据"""
        if key in cache_dict:
            data, timestamp = cache_dict[key]
            if time.time() - timestamp < self.config.get('cache_ttl', 3600):
                self.logger.debug(f"Cache hit for key: {key}")
                return data
            else:
                del cache_dict[key]
        return None
    
    def _save_to_cache(self, cache_dict: Dict, key: str, data: Any) -> None:
        """保存数据到缓存"""
        cache_dict[key] = (data, time.time())
    
    def analyze_news_sentiment(self, news_item: NewsItem, 
                              symbol: Optional[str] = None) -> SentimentAnalysis:
        """
        分析新闻情绪
        
        Args:
            news_item: 新闻条目
            symbol: 目标交易品种
            
        Returns:
            SentimentAnalysis: 情绪分析结果
        """
        try:
            # 检查缓存
            cache_key = f"{news_item.headline}_{symbol}"
            cached_result = self._get_from_cache(self.sentiment_cache, cache_key)
            if cached_result:
                return cached_result
            
            # 基础情绪分析（使用关键词）
            sentiment_score = self.preprocessor.calculate_sentiment_score(
                news_item.headline + " " + news_item.content
            )
            impact_level = self.preprocessor.assess_impact_level(
                news_item.headline + " " + news_item.content
            )
            keywords = self.preprocessor.extract_keywords(news_item.headline)
            
            # 如果启用LLM且模型可用，使用LLM增强分析
            if self.config.get('use_llm_for_sentiment') and self._is_model_available():
                llm_result = self._llm_sentiment_analysis(news_item, symbol)
                if llm_result:
                    # 合并LLM结果和基础分析
                    sentiment_score = (sentiment_score + llm_result['score']) / 2
                    explanation = llm_result['explanation']
                    confidence = llm_result['confidence']
                else:
                    explanation = self._generate_basic_explanation(sentiment_score, keywords)
                    confidence = 0.6
            else:
                explanation = self._generate_basic_explanation(sentiment_score, keywords)
                confidence = 0.6
            
            # 确定情绪类型
            if sentiment_score > 0.2:
                sentiment = SentimentType.BULLISH
            elif sentiment_score < -0.2:
                sentiment = SentimentType.BEARISH
            else:
                sentiment = SentimentType.NEUTRAL
            
            result = SentimentAnalysis(
                sentiment=sentiment,
                confidence=confidence,
                impact_level=impact_level,
                explanation=explanation,
                keywords=keywords,
                score=sentiment_score
            )
            
            # 保存到缓存
            self._save_to_cache(self.sentiment_cache, cache_key, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"News sentiment analysis failed: {e}")
            # 返回中性结果
            return SentimentAnalysis(
                sentiment=SentimentType.NEUTRAL,
                confidence=0.0,
                impact_level=ImpactLevel.LOW,
                explanation="Analysis failed",
                keywords=[],
                score=0.0
            )
    
    def _llm_sentiment_analysis(self, news_item: NewsItem, 
                               symbol: Optional[str]) -> Optional[Dict[str, Any]]:
        """使用LLM进行情绪分析"""
        try:
            # 构建提示词
            prompt = self._build_sentiment_prompt(news_item, symbol)
            
            # 生成缓存键
            cache_key = LLMResultCache.generate_key(
                prompt,
                {'temperature': self.config.get('temperature', 0.5)}
            )
            
            # 使用优化器调用LLM
            def _call_llm():
                model = self.model_manager.get_model()
                if not model or not model.is_loaded:
                    if self.model_manager:
                        self.model_manager.load_model()
                        model = self.model_manager.get_model()
                    else:
                        return None
                
                return model.generate(
                    prompt=prompt,
                    temperature=self.config.get('temperature', 0.5),
                    max_tokens=self.config.get('max_tokens', 200)
                )
            
            response = self.call_optimizer.call(
                _call_llm,
                priority=CallPriority.MEDIUM,
                cache_key=cache_key,
                cache_ttl=self.config.get('cache_ttl', 3600)
            )
            
            if response is None:
                return None
            
            # 解析LLM响应
            parsed_result = self._parse_sentiment_response(response)
            return parsed_result
            
        except Exception as e:
            self.logger.error(f"LLM sentiment analysis failed: {e}")
            return None
    
    def _build_sentiment_prompt(self, news_item: NewsItem, 
                               symbol: Optional[str]) -> str:
        """构建情绪分析提示词"""
        symbol_text = f" on {symbol}" if symbol else ""
        
        prompt = f"""Analyze the sentiment of this news headline{symbol_text}:
"{news_item.headline}"

Provide:
1. Sentiment: Bullish/Bearish/Neutral
2. Confidence: 0-100%
3. Impact: High/Medium/Low
4. Brief explanation (1-2 sentences)

Keep response concise and structured."""
        
        return prompt
    
    def _parse_sentiment_response(self, response: str) -> Dict[str, Any]:
        """解析LLM情绪分析响应"""
        try:
            # 简单的响应解析
            response_lower = response.lower()
            
            # 提取情绪
            if 'bullish' in response_lower:
                score = 0.7
            elif 'bearish' in response_lower:
                score = -0.7
            else:
                score = 0.0
            
            # 提取置信度
            confidence_match = re.search(r'(\d+)%', response)
            confidence = float(confidence_match.group(1)) / 100 if confidence_match else 0.7
            
            # 提取解释
            explanation = response.strip()
            
            return {
                'score': score,
                'confidence': confidence,
                'explanation': explanation
            }
            
        except Exception as e:
            self.logger.error(f"Failed to parse sentiment response: {e}")
            return {'score': 0.0, 'confidence': 0.5, 'explanation': response}
    
    def _generate_basic_explanation(self, sentiment_score: float, 
                                   keywords: List[str]) -> str:
        """生成基础解释"""
        if sentiment_score > 0.2:
            sentiment_text = "bullish"
        elif sentiment_score < -0.2:
            sentiment_text = "bearish"
        else:
            sentiment_text = "neutral"
        
        keywords_text = ", ".join(keywords[:3]) if keywords else "general market factors"
        
        return f"Analysis shows {sentiment_text} sentiment based on {keywords_text}."
    
    def generate_market_commentary(self, symbol: str, 
                                  market_data: MarketData,
                                  technical_signals: Optional[Dict[str, float]] = None) -> MarketCommentary:
        """
        生成市场评论
        
        Args:
            symbol: 交易品种
            market_data: 市场数据
            technical_signals: 技术信号
            
        Returns:
            MarketCommentary: 市场评论
        """
        try:
            # 检查缓存
            cache_key = f"{symbol}_{market_data.timestamp.isoformat()}"
            cached_result = self._get_from_cache(self.commentary_cache, cache_key)
            if cached_result:
                return cached_result
            
            # 基础分析
            basic_analysis = self._generate_basic_commentary(symbol, market_data, technical_signals)
            
            # 如果LLM可用，使用LLM增强评论
            if self._is_model_available():
                llm_commentary = self._llm_market_commentary(symbol, market_data, technical_signals)
                if llm_commentary:
                    # 合并基础分析和LLM评论
                    result = self._merge_commentary(basic_analysis, llm_commentary)
                else:
                    result = basic_analysis
            else:
                result = basic_analysis
            
            # 保存到缓存
            self._save_to_cache(self.commentary_cache, cache_key, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Market commentary generation failed: {e}")
            raise AnalysisError(f"Failed to generate market commentary: {e}")
    
    def _generate_basic_commentary(self, symbol: str, market_data: MarketData,
                                  technical_signals: Optional[Dict[str, float]]) -> MarketCommentary:
        """生成基础市场评论"""
        # 计算基本指标
        close_prices = market_data.ohlcv['close']
        current_price = close_prices.iloc[-1]
        price_change = (current_price - close_prices.iloc[-2]) / close_prices.iloc[-2] * 100
        
        # 生成摘要
        direction = "up" if price_change > 0 else "down"
        summary = f"{symbol} is trading {direction} {abs(price_change):.2f}% at {current_price:.5f}"
        
        # 关键点
        key_points = [
            f"Current price: {current_price:.5f}",
            f"24h change: {price_change:+.2f}%",
            f"Volatility: {market_data.indicators.get('volatility', 'N/A')}"
        ]
        
        # 市场状态
        if technical_signals:
            rsi = technical_signals.get('rsi_signal', 0)
            if rsi > 0.5:
                market_regime = "Oversold conditions"
            elif rsi < -0.5:
                market_regime = "Overbought conditions"
            else:
                market_regime = "Neutral conditions"
        else:
            market_regime = "Normal trading conditions"
        
        # 风险评估
        volatility = market_data.indicators.get('volatility', 0.02)
        if volatility > 0.03:
            risk_assessment = "High volatility - exercise caution"
        elif volatility > 0.02:
            risk_assessment = "Moderate volatility"
        else:
            risk_assessment = "Low volatility - stable conditions"
        
        # 交易建议
        if price_change > 1:
            trading_recommendation = "Consider taking profits on long positions"
        elif price_change < -1:
            trading_recommendation = "Potential buying opportunity if support holds"
        else:
            trading_recommendation = "Wait for clearer signals"
        
        return MarketCommentary(
            symbol=symbol,
            timestamp=market_data.timestamp,
            summary=summary,
            key_points=key_points,
            market_regime=market_regime,
            risk_assessment=risk_assessment,
            trading_recommendation=trading_recommendation,
            confidence=0.7
        )
    
    def _llm_market_commentary(self, symbol: str, market_data: MarketData,
                              technical_signals: Optional[Dict[str, float]]) -> Optional[Dict[str, Any]]:
        """使用LLM生成市场评论"""
        try:
            # 构建提示词
            prompt = self._build_commentary_prompt(symbol, market_data, technical_signals)
            
            # 生成缓存键
            cache_key = LLMResultCache.generate_key(
                prompt,
                {'temperature': 0.6, 'symbol': symbol}
            )
            
            # 使用优化器调用LLM
            def _call_llm():
                model = self.model_manager.get_model()
                if not model or not model.is_loaded:
                    if self.model_manager:
                        self.model_manager.load_model()
                        model = self.model_manager.get_model()
                    else:
                        return None
                
                return model.generate(
                    prompt=prompt,
                    temperature=0.6,
                    max_tokens=300
                )
            
            response = self.call_optimizer.call(
                _call_llm,
                priority=CallPriority.LOW,  # 市场评论优先级较低
                cache_key=cache_key,
                cache_ttl=1800  # 30分钟缓存
            )
            
            if response is None:
                return None
            
            return {'commentary': response}
            
        except Exception as e:
            self.logger.error(f"LLM market commentary failed: {e}")
            return None
    
    def _build_commentary_prompt(self, symbol: str, market_data: MarketData,
                                technical_signals: Optional[Dict[str, float]]) -> str:
        """构建市场评论提示词"""
        close_prices = market_data.ohlcv['close']
        current_price = close_prices.iloc[-1]
        
        signals_text = ""
        if technical_signals:
            signals_text = "\n".join([f"{k}: {v:.2f}" for k, v in technical_signals.items()])
        
        prompt = f"""Market Analysis Request:

Symbol: {symbol}
Current Price: {current_price:.5f}
Volatility: {market_data.indicators.get('volatility', 'N/A')}

Technical Signals:
{signals_text if signals_text else 'No signals available'}

Provide:
1. Market regime assessment
2. Risk level evaluation
3. Key observations (2-3 points)
4. Trading recommendation

Keep response concise and actionable."""
        
        return prompt
    
    def _merge_commentary(self, basic: MarketCommentary, 
                         llm_result: Dict[str, Any]) -> MarketCommentary:
        """合并基础评论和LLM评论"""
        # 使用基础评论作为基础，用LLM结果增强
        llm_text = llm_result.get('commentary', '')
        
        # 更新摘要和关键点
        enhanced_summary = f"{basic.summary}. {llm_text[:100]}"
        
        return MarketCommentary(
            symbol=basic.symbol,
            timestamp=basic.timestamp,
            summary=enhanced_summary,
            key_points=basic.key_points,
            market_regime=basic.market_regime,
            risk_assessment=basic.risk_assessment,
            trading_recommendation=basic.trading_recommendation,
            confidence=0.85  # 提高置信度
        )
    
    def analyze_economic_events(self, events: List[EconomicEvent],
                               symbol: Optional[str] = None) -> List[EventImpact]:
        """
        分析经济事件影响
        
        Args:
            events: 经济事件列表
            symbol: 目标交易品种
            
        Returns:
            List[EventImpact]: 事件影响分析列表
        """
        try:
            impacts = []
            
            for event in events:
                impact = self._analyze_single_event(event, symbol)
                impacts.append(impact)
            
            # 按重要性排序
            impacts.sort(key=lambda x: x.confidence, reverse=True)
            
            return impacts
            
        except Exception as e:
            self.logger.error(f"Economic events analysis failed: {e}")
            return []
    
    def _analyze_single_event(self, event: EconomicEvent,
                             symbol: Optional[str]) -> EventImpact:
        """分析单个经济事件"""
        # 确定受影响的品种
        affected_symbols = self._determine_affected_symbols(event, symbol)
        
        # 评估影响
        surprise_factor = event.get_surprise_factor()
        
        if surprise_factor is not None:
            if abs(surprise_factor) > 0.5:
                impact_assessment = "Significant surprise - high market impact expected"
                confidence = 0.9
            elif abs(surprise_factor) > 0.2:
                impact_assessment = "Moderate surprise - some market reaction expected"
                confidence = 0.7
            else:
                impact_assessment = "In line with expectations - limited impact"
                confidence = 0.5
            
            # 确定情绪
            if surprise_factor > 0:
                sentiment = SentimentType.BULLISH
            elif surprise_factor < 0:
                sentiment = SentimentType.BEARISH
            else:
                sentiment = SentimentType.NEUTRAL
        else:
            impact_assessment = "Event pending - monitor for actual data"
            sentiment = SentimentType.NEUTRAL
            confidence = 0.3
        
        # 生成建议
        recommended_actions = self._generate_event_recommendations(
            event, surprise_factor, affected_symbols
        )
        
        # 评估风险级别
        if event.importance == "high":
            risk_level = "high"
        elif event.importance == "medium":
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return EventImpact(
            event=event,
            affected_symbols=affected_symbols,
            impact_assessment=impact_assessment,
            sentiment=sentiment,
            confidence=confidence,
            recommended_actions=recommended_actions,
            risk_level=risk_level
        )
    
    def _determine_affected_symbols(self, event: EconomicEvent,
                                   target_symbol: Optional[str]) -> List[str]:
        """确定受事件影响的交易品种"""
        affected = []
        
        # 基于货币的影响
        currency_symbol_map = {
            'USD': ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD', 'USOIL'],
            'EUR': ['EURUSD', 'EURGBP', 'EURJPY'],
            'GBP': ['GBPUSD', 'EURGBP', 'GBPJPY'],
            'JPY': ['USDJPY', 'EURJPY', 'GBPJPY'],
        }
        
        if event.currency in currency_symbol_map:
            affected.extend(currency_symbol_map[event.currency])
        
        # 如果指定了目标品种，检查是否受影响
        if target_symbol and target_symbol in affected:
            return [target_symbol]
        
        return affected if affected else ['GENERAL']
    
    def _generate_event_recommendations(self, event: EconomicEvent,
                                       surprise_factor: Optional[float],
                                       affected_symbols: List[str]) -> List[str]:
        """生成事件相关建议"""
        recommendations = []
        
        if event.importance == "high":
            recommendations.append("Monitor positions closely during event")
            recommendations.append("Consider reducing position size before release")
        
        if surprise_factor is not None and abs(surprise_factor) > 0.3:
            if surprise_factor > 0:
                recommendations.append(f"Positive surprise - consider long positions in {', '.join(affected_symbols[:2])}")
            else:
                recommendations.append(f"Negative surprise - consider short positions or exit longs")
        
        if not recommendations:
            recommendations.append("Wait for event outcome before taking action")
        
        return recommendations
    
    def get_optimizer_stats(self) -> Dict[str, Any]:
        """
        获取调用优化器统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        return self.call_optimizer.get_stats()
    
    def reset_optimizer_metrics(self) -> None:
        """重置优化器指标"""
        self.call_optimizer.reset_metrics()
        self.logger.info("Optimizer metrics reset")
    
    def clear_optimizer_cache(self) -> None:
        """清空优化器缓存"""
        self.call_optimizer.clear_cache()
        self.logger.info("Optimizer cache cleared")
    
    async def analyze_news_sentiment_async(self, news_item: NewsItem,
                                          symbol: Optional[str] = None) -> SentimentAnalysis:
        """
        异步分析新闻情绪
        
        Args:
            news_item: 新闻条目
            symbol: 目标交易品种
            
        Returns:
            SentimentAnalysis: 情绪分析结果
        """
        # 使用同步方法，但通过异步执行器运行
        return await self.call_optimizer.async_executor.execute_async(
            self.analyze_news_sentiment,
            news_item,
            symbol,
            priority=CallPriority.MEDIUM
        )
    
    async def generate_market_commentary_async(self, symbol: str,
                                              market_data: MarketData,
                                              technical_signals: Optional[Dict[str, float]] = None) -> MarketCommentary:
        """
        异步生成市场评论
        
        Args:
            symbol: 交易品种
            market_data: 市场数据
            technical_signals: 技术信号
            
        Returns:
            MarketCommentary: 市场评论
        """
        return await self.call_optimizer.async_executor.execute_async(
            self.generate_market_commentary,
            symbol,
            market_data,
            technical_signals,
            priority=CallPriority.LOW
        )
    
    def fetch_and_analyze_news(self, symbols: Optional[List[str]] = None,
                              max_items: int = 10) -> List[Tuple[NewsItem, SentimentAnalysis]]:
        """
        获取并分析最新新闻
        
        Args:
            symbols: 目标交易品种列表
            max_items: 最大新闻数量
            
        Returns:
            List[Tuple[NewsItem, SentimentAnalysis]]: 新闻和情绪分析对
        """
        try:
            # 获取新闻
            news_items = self.news_scraper.fetch_latest_news(symbols, max_items)
            
            # 分析每条新闻
            results = []
            for news_item in news_items:
                # 确定相关品种
                target_symbol = None
                if symbols and news_item.symbols:
                    # 找到第一个匹配的品种
                    for symbol in symbols:
                        if symbol in news_item.symbols:
                            target_symbol = symbol
                            break
                
                # 分析情绪
                sentiment = self.analyze_news_sentiment(news_item, target_symbol)
                results.append((news_item, sentiment))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to fetch and analyze news: {e}")
            return []
    
    def generate_daily_summary(self, date: datetime,
                              performance_data: Dict[str, Any],
                              events: List[str]) -> str:
        """
        生成每日市场总结
        
        Args:
            date: 日期
            performance_data: 性能数据
            events: 关键事件列表
            
        Returns:
            str: 每日总结
        """
        try:
            # 基础总结
            summary_parts = [
                f"Daily Market Summary - {date.strftime('%Y-%m-%d')}",
                "",
                "Performance Overview:",
            ]
            
            for key, value in performance_data.items():
                summary_parts.append(f"  {key}: {value}")
            
            summary_parts.append("")
            summary_parts.append("Key Events:")
            for event in events:
                summary_parts.append(f"  - {event}")
            
            basic_summary = "\n".join(summary_parts)
            
            # 如果LLM可用，增强总结
            if self._is_model_available():
                llm_summary = self._llm_daily_summary(date, performance_data, events)
                if llm_summary:
                    return f"{basic_summary}\n\nAnalysis:\n{llm_summary}"
            
            return basic_summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate daily summary: {e}")
            return f"Daily summary generation failed: {e}"
    
    def _llm_daily_summary(self, date: datetime, performance_data: Dict[str, Any],
                          events: List[str]) -> Optional[str]:
        """使用LLM生成每日总结"""
        try:
            self._check_rate_limit()
            
            prompt = f"""Daily Market Summary:

Date: {date.strftime('%Y-%m-%d')}

Performance:
{json.dumps(performance_data, indent=2)}

Key Events:
{chr(10).join(f'- {event}' for event in events)}

Provide:
1. Overall market sentiment
2. Best performing assets
3. Key risks identified
4. Outlook for tomorrow

Keep response concise (3-4 paragraphs)."""
            
            model = self.model_manager.get_model()
            if not model or not model.is_loaded:
                if self.model_manager:
                    self.model_manager.load_model()
                    model = self.model_manager.get_model()
                else:
                    return None
            
            response = model.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=500
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"LLM daily summary failed: {e}")
            return None
    
    def _is_model_available(self) -> bool:
        """检查LLM模型是否可用"""
        return (self.model_manager is not None and 
                (self.model_manager.is_model_loaded() or 
                 self.model_manager.load_model()))
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self.sentiment_cache.clear()
        self.commentary_cache.clear()
        self.logger.info("LLM analyst cache cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'sentiment_cache_size': len(self.sentiment_cache),
            'commentary_cache_size': len(self.commentary_cache),
            'model_loaded': self._is_model_available(),
            'last_call_time': self.last_call_time
        }
