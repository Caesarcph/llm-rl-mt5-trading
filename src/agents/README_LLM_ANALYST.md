# LLM分析Agent文档

## 概述

LLM分析Agent (`LLMAnalystAgent`) 是一个智能分析代理，提供新闻情绪分析、市场评论生成和经济事件影响分析功能。它结合了基于关键词的传统分析方法和可选的大语言模型(LLM)增强分析。

## 主要功能

### 1. 新闻情绪分析
- 分析新闻标题和内容的市场情绪（看涨/看跌/中性）
- 评估新闻影响级别（高/中/低）
- 提取关键词和相关交易品种
- 计算情绪分数和置信度

### 2. 市场评论生成
- 基于市场数据生成综合评论
- 分析技术指标和市场状态
- 提供风险评估和交易建议
- 支持多时间框架分析

### 3. 经济事件影响分析
- 分析经济数据发布的市场影响
- 计算实际值与预期值的意外因子
- 识别受影响的交易品种
- 生成具体的操作建议

### 4. 新闻数据获取
- 自动抓取最新市场新闻（可扩展）
- 按交易品种过滤新闻
- 批量分析多条新闻

### 5. 每日市场总结
- 生成每日交易总结报告
- 汇总关键市场事件
- 分析整体市场情绪

## 核心组件

### LLMAnalystAgent
主要的分析代理类，协调所有分析功能。

```python
from src.agents.llm_analyst import LLMAnalystAgent

# 初始化Agent
config = {
    'model_path': 'models/llama-3.2-1b.gguf',  # LLM模型路径
    'use_llm_for_sentiment': True,              # 启用LLM分析
    'min_call_interval': 5.0,                   # 最小调用间隔（秒）
    'cache_ttl': 3600                           # 缓存过期时间（秒）
}
agent = LLMAnalystAgent(config=config)
```

### NewsPreprocessor
新闻文本预处理器，提供文本清洗、关键词提取等功能。

```python
from src.agents.llm_analyst import NewsPreprocessor

preprocessor = NewsPreprocessor()

# 预处理文本
clean_text = preprocessor.preprocess_text("Gold PRICES Surge 5%!")

# 提取关键词
keywords = preprocessor.extract_keywords(text, top_n=10)

# 计算情绪分数
score = preprocessor.calculate_sentiment_score(text)

# 评估影响级别
impact = preprocessor.assess_impact_level(text)
```

### NewsScraper
新闻数据抓取器（当前为模拟实现，可扩展对接真实新闻API）。

```python
from src.agents.llm_analyst import NewsScraper

scraper = NewsScraper()

# 获取最新新闻
news_items = scraper.fetch_latest_news(
    symbols=['EURUSD', 'XAUUSD'],
    max_items=10
)
```

## 数据模型

### NewsItem
新闻条目数据结构。

```python
from src.agents.llm_analyst import NewsItem
from datetime import datetime

news = NewsItem(
    headline="Fed signals rate cut",
    content="Full article content...",
    source="Reuters",
    timestamp=datetime.now(),
    symbols=['EURUSD', 'XAUUSD'],
    category="central_bank"
)
```

### SentimentAnalysis
情绪分析结果。

```python
from src.agents.llm_analyst import SentimentAnalysis, SentimentType, ImpactLevel

sentiment = SentimentAnalysis(
    sentiment=SentimentType.BULLISH,
    confidence=0.85,
    impact_level=ImpactLevel.HIGH,
    explanation="Strong bullish indicators detected",
    keywords=['rally', 'surge', 'gain'],
    score=0.75
)

# 检查是否可操作
if sentiment.is_actionable(min_confidence=0.7, min_impact=ImpactLevel.MEDIUM):
    print("This is an actionable signal")
```

### EconomicEvent
经济事件数据。

```python
from src.agents.llm_analyst import EconomicEvent
from datetime import datetime

event = EconomicEvent(
    name="Non-Farm Payrolls",
    country="US",
    timestamp=datetime.now(),
    importance="high",
    actual=250000,
    forecast=200000,
    previous=180000,
    currency="USD"
)

# 计算意外因子
surprise = event.get_surprise_factor()  # +25%
```

### MarketCommentary
市场评论结果。

```python
commentary = MarketCommentary(
    symbol="EURUSD",
    timestamp=datetime.now(),
    summary="EUR/USD trading higher on ECB comments",
    key_points=["Price at 1.1000", "Strong momentum"],
    market_regime="Trending",
    risk_assessment="Moderate volatility",
    trading_recommendation="Consider long positions",
    confidence=0.8
)
```

## 使用示例

### 示例1: 新闻情绪分析

```python
from src.agents.llm_analyst import LLMAnalystAgent, NewsItem
from datetime import datetime

# 初始化Agent
agent = LLMAnalystAgent()

# 创建新闻
news = NewsItem(
    headline="Fed signals potential rate cut",
    content="The Federal Reserve indicated...",
    source="Reuters",
    timestamp=datetime.now(),
    symbols=['EURUSD']
)

# 分析情绪
sentiment = agent.analyze_news_sentiment(news, symbol='EURUSD')

print(f"情绪: {sentiment.sentiment.value}")
print(f"分数: {sentiment.score}")
print(f"置信度: {sentiment.confidence:.2%}")
print(f"影响: {sentiment.impact_level.value}")
```

### 示例2: 市场评论生成

```python
from src.agents.llm_analyst import LLMAnalystAgent
from src.core.models import MarketData

# 初始化Agent
agent = LLMAnalystAgent()

# 准备市场数据和技术信号
market_data = get_market_data('EURUSD')  # 你的数据获取函数
technical_signals = {
    'rsi_signal': 0.3,
    'macd_signal': 1.0,
    'trend_strength': 0.8
}

# 生成评论
commentary = agent.generate_market_commentary(
    symbol='EURUSD',
    market_data=market_data,
    technical_signals=technical_signals
)

print(f"摘要: {commentary.summary}")
print(f"建议: {commentary.trading_recommendation}")
```

### 示例3: 经济事件分析

```python
from src.agents.llm_analyst import LLMAnalystAgent, EconomicEvent
from datetime import datetime

# 初始化Agent
agent = LLMAnalystAgent()

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
    )
]

# 分析影响
impacts = agent.analyze_economic_events(events, symbol='EURUSD')

for impact in impacts:
    print(f"事件: {impact.event.name}")
    print(f"影响: {impact.impact_assessment}")
    print(f"建议: {', '.join(impact.recommended_actions)}")
```

### 示例4: 批量新闻分析

```python
from src.agents.llm_analyst import LLMAnalystAgent

# 初始化Agent
agent = LLMAnalystAgent()

# 获取并分析新闻
symbols = ['EURUSD', 'XAUUSD', 'USOIL']
results = agent.fetch_and_analyze_news(symbols=symbols, max_items=10)

# 统计情绪
bullish_count = sum(1 for _, s in results if s.sentiment.value == 'bullish')
bearish_count = sum(1 for _, s in results if s.sentiment.value == 'bearish')

print(f"看涨新闻: {bullish_count}")
print(f"看跌新闻: {bearish_count}")
```

### 示例5: 每日总结

```python
from src.agents.llm_analyst import LLMAnalystAgent
from datetime import datetime

# 初始化Agent
agent = LLMAnalystAgent()

# 准备数据
performance_data = {
    'total_trades': 15,
    'win_rate': '66.7%',
    'total_profit': '$2,450.00'
}

events = [
    "Fed kept rates unchanged",
    "Strong employment data",
    "EUR/USD reached new high"
]

# 生成总结
summary = agent.generate_daily_summary(
    date=datetime.now(),
    performance_data=performance_data,
    events=events
)

print(summary)
```

## 配置选项

### 基础配置

```python
config = {
    # LLM模型配置
    'model_path': 'models/llama-3.2-1b.gguf',  # 模型文件路径
    'auto_load_model': False,                   # 是否自动加载模型
    'use_llm_for_sentiment': True,              # 是否使用LLM进行情绪分析
    
    # 调用控制
    'min_call_interval': 5.0,                   # 最小调用间隔（秒）
    'cache_ttl': 3600,                          # 缓存过期时间（秒）
    
    # 分析参数
    'sentiment_confidence_threshold': 0.6,      # 情绪置信度阈值
    'temperature': 0.5,                         # LLM生成温度
    'max_tokens': 200                           # 最大生成token数
}
```

### LLM模型配置

如果要使用完整的LLM功能：

1. 下载Llama 3.2模型（1B或3B版本）
2. 安装llama-cpp-python: `pip install llama-cpp-python`
3. 配置模型路径

```python
config = {
    'model_path': 'models/llama-3.2-1b.gguf',
    'use_llm_for_sentiment': True,
    'auto_load_model': True
}
```

## 性能优化

### 1. 缓存机制
Agent自动缓存分析结果，避免重复计算：

```python
# 第一次分析（计算）
sentiment1 = agent.analyze_news_sentiment(news)

# 第二次分析（从缓存获取）
sentiment2 = agent.analyze_news_sentiment(news)
```

### 2. 调用频率控制
自动限制LLM调用频率，避免过度使用：

```python
config = {
    'min_call_interval': 5.0  # 最少5秒间隔
}
```

### 3. 清空缓存
定期清空缓存释放内存：

```python
agent.clear_cache()
```

### 4. 统计信息
监控Agent使用情况：

```python
stats = agent.get_statistics()
print(f"缓存大小: {stats['sentiment_cache_size']}")
print(f"模型状态: {stats['model_loaded']}")
```

## 扩展开发

### 对接真实新闻API

修改`NewsScraper`类以对接真实新闻源：

```python
class NewsScraper:
    def fetch_latest_news(self, symbols, max_items):
        # 对接NewsAPI、Alpha Vantage等
        response = requests.get(
            'https://newsapi.org/v2/everything',
            params={
                'q': ' OR '.join(symbols),
                'apiKey': 'your_api_key'
            }
        )
        # 解析并返回NewsItem列表
        return parse_news_response(response.json())
```

### 自定义情绪分析

继承并扩展情绪分析功能：

```python
class CustomLLMAnalyst(LLMAnalystAgent):
    def analyze_news_sentiment(self, news_item, symbol=None):
        # 调用父类方法
        base_sentiment = super().analyze_news_sentiment(news_item, symbol)
        
        # 添加自定义逻辑
        if 'bitcoin' in news_item.headline.lower():
            base_sentiment.impact_level = ImpactLevel.HIGH
        
        return base_sentiment
```

## 测试

运行完整测试套件：

```bash
python -m pytest tests/test_llm_analyst.py -v
```

运行特定测试：

```bash
# 测试情绪分析
python -m pytest tests/test_llm_analyst.py::TestLLMAnalystAgent::test_analyze_news_sentiment_bullish -v

# 测试准确性
python -m pytest tests/test_llm_analyst.py::TestSentimentAnalysisAccuracy -v
```

## 注意事项

1. **LLM模型**: 完整功能需要下载并配置Llama模型
2. **内存使用**: LLM模型会占用1-3GB内存
3. **调用频率**: 建议设置合理的调用间隔避免过度使用
4. **新闻源**: 当前使用模拟数据，生产环境需对接真实API
5. **缓存管理**: 定期清空缓存避免内存泄漏

## 相关文档

- [LLM模块指南](../../docs/llm_module_guide.md)
- [LLM快速参考](../../docs/llm_quick_reference.md)
- [配置文件说明](../../config/llm_config.yaml)

## 更新日志

### v1.0.0 (2025-10-03)
- 初始版本发布
- 实现新闻情绪分析
- 实现市场评论生成
- 实现经济事件影响分析
- 添加完整测试套件
- 添加演示程序
