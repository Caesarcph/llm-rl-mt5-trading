# LLM模块使用指南

## 概述

LLM模块提供了本地Llama 3.2模型的集成，用于市场分析、新闻情绪分析和智能决策支持。该模块设计为轻量级、易用且内存高效。

## 核心组件

### 1. ModelConfig

模型配置数据类，定义LLM的运行参数。

```python
from src.llm import ModelConfig

config = ModelConfig(
    model_path="models/llama-3.2-1b.gguf",  # 模型文件路径
    n_ctx=2048,                              # 上下文窗口大小
    n_threads=4,                             # CPU线程数
    n_gpu_layers=0,                          # GPU层数（0=仅CPU）
    temperature=0.7,                         # 生成温度（0-1）
    max_tokens=512,                          # 最大生成token数
    top_p=0.9,                              # Top-p采样
    top_k=40,                               # Top-k采样
    repeat_penalty=1.1,                     # 重复惩罚
    verbose=False                           # 详细日志
)
```

### 2. LlamaModel

Llama模型包装类，提供统一的推理接口。

#### 基础使用

```python
from src.llm import LlamaModel, ModelConfig

# 创建配置
config = ModelConfig(model_path="models/llama-3.2-1b.gguf")

# 创建模型实例
model = LlamaModel(config)

# 加载模型
if model.load_model():
    # 生成文本
    response = model.generate("Analyze EUR/USD trend:")
    print(response)
    
    # 卸载模型
    model.unload_model()
```

#### 使用上下文管理器

```python
with LlamaModel(config) as model:
    response = model.generate("Your prompt here")
    print(response)
# 自动卸载模型
```

#### 聊天模式

```python
messages = [
    {"role": "system", "content": "You are a forex analyst."},
    {"role": "user", "content": "What's the outlook for gold?"},
    {"role": "assistant", "content": "Gold is showing..."},
    {"role": "user", "content": "Should I buy now?"}
]

response = model.chat(messages)
```

### 3. ModelManager

模型管理器，负责模型生命周期和内存管理。

#### 基础使用

```python
from src.llm import ModelManager

# 创建管理器
manager = ModelManager(
    model_path="models/llama-3.2-1b.gguf",
    memory_threshold_mb=1024  # 最小可用内存要求
)

# 创建配置
config = manager.create_config(
    n_ctx=2048,
    n_threads=4,
    temperature=0.7
)

# 加载模型
if manager.load_model(config):
    model = manager.get_model()
    response = model.generate("Your prompt")
    manager.unload_model()
```

#### 内存监控

```python
# 获取内存统计
stats = manager.get_memory_stats()
print(f"总内存: {stats['total_mb']:.2f} MB")
print(f"可用内存: {stats['available_mb']:.2f} MB")
print(f"使用率: {stats['percent']:.1f}%")

# 检查可用内存
available = ModelManager.get_available_memory_mb()
print(f"可用内存: {available:.2f} MB")
```

## 模型选择指南

### Llama 3.2 1B vs 3B

| 特性 | 1B模型 | 3B模型 |
|------|--------|--------|
| 内存需求 | ~2GB | ~4GB |
| 推理速度 | 快 | 中等 |
| 分析质量 | 良好 | 优秀 |
| 适用场景 | 实时分析、快速响应 | 深度分析、复杂推理 |

### 推荐配置

#### 低配置环境（4GB RAM）
```python
config = ModelConfig(
    model_path="models/llama-3.2-1b.gguf",
    n_ctx=1024,
    n_threads=2,
    temperature=0.7
)
```

#### 标准配置（8GB RAM）
```python
config = ModelConfig(
    model_path="models/llama-3.2-3b.gguf",
    n_ctx=2048,
    n_threads=4,
    temperature=0.7
)
```

#### 高性能配置（16GB+ RAM）
```python
config = ModelConfig(
    model_path="models/llama-3.2-3b.gguf",
    n_ctx=4096,
    n_threads=8,
    temperature=0.6
)
```

## 实际应用场景

### 1. 新闻情绪分析

```python
def analyze_news_sentiment(model: LlamaModel, headline: str) -> dict:
    prompt = f"""
    Analyze the sentiment of this news headline:
    "{headline}"
    
    Provide:
    1. Sentiment: Bullish/Bearish/Neutral
    2. Confidence: 0-100%
    3. Impact on USD: High/Medium/Low
    4. Brief explanation
    
    Format as JSON.
    """
    
    response = model.generate(prompt, temperature=0.5)
    return parse_sentiment_response(response)
```

### 2. 市场状态分析

```python
def analyze_market_state(model: LlamaModel, market_data: dict) -> str:
    prompt = f"""
    Market Analysis Request:
    
    EUR/USD: {market_data['eurusd']['trend']}, RSI: {market_data['eurusd']['rsi']}
    Gold: {market_data['gold']['state']}, Volatility: {market_data['gold']['volatility']}
    VIX: {market_data['vix']}
    
    Provide:
    1. Overall market regime (Trending/Ranging/Volatile)
    2. Risk level (Low/Medium/High)
    3. Trading recommendations
    """
    
    return model.generate(prompt, max_tokens=300)
```

### 3. 交易决策支持

```python
def get_trading_advice(model: LlamaModel, signal: dict) -> str:
    prompt = f"""
    Trading Signal Analysis:
    
    Symbol: {signal['symbol']}
    Direction: {signal['direction']}
    Strength: {signal['strength']}
    Technical Indicators: {signal['indicators']}
    
    Should I take this trade? Consider:
    1. Risk/Reward ratio
    2. Market conditions
    3. Timing
    
    Provide a clear recommendation.
    """
    
    return model.generate(prompt, temperature=0.6)
```

## 性能优化建议

### 1. 调用频率控制

```python
import time
from functools import lru_cache

class LLMThrottler:
    def __init__(self, min_interval: float = 5.0):
        self.min_interval = min_interval
        self.last_call = 0
    
    def can_call(self) -> bool:
        now = time.time()
        if now - self.last_call >= self.min_interval:
            self.last_call = now
            return True
        return False
```

### 2. 结果缓存

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=100)
def cached_generate(model_id: str, prompt: str, max_tokens: int) -> str:
    # 实际调用模型
    return model.generate(prompt, max_tokens=max_tokens)
```

### 3. 批量处理

```python
def batch_analyze(model: LlamaModel, prompts: list) -> list:
    results = []
    for prompt in prompts:
        result = model.generate(prompt)
        results.append(result)
    return results
```

## 错误处理

### 常见错误及解决方案

#### 1. 模型文件不存在
```python
if not model.load_model():
    logger.error("Model file not found. Please download the model first.")
    # 提供下载链接或备用方案
```

#### 2. 内存不足
```python
if not manager._check_memory_available():
    logger.warning("Insufficient memory. Using fallback strategy.")
    # 使用更小的模型或降低配置
```

#### 3. 生成超时
```python
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Generation timeout")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)  # 30秒超时

try:
    response = model.generate(prompt)
finally:
    signal.alarm(0)
```

## 最佳实践

### 1. 使用上下文管理器

始终使用上下文管理器确保模型正确卸载：

```python
with ModelManager(model_path) as manager:
    model = manager.get_model()
    # 使用模型
# 自动清理
```

### 2. 定期卸载模型

在不需要时卸载模型以释放内存：

```python
# 分析完成后
manager.unload_model()

# 或在空闲时间
if idle_time > 300:  # 5分钟空闲
    manager.unload_model()
```

### 3. 监控内存使用

```python
stats = manager.get_memory_stats()
if stats['percent'] > 80:
    logger.warning("High memory usage, consider unloading model")
    manager.unload_model()
```

### 4. 优化提示词

- 保持提示简洁明确
- 使用结构化格式（JSON、列表）
- 限制输出长度
- 提供具体示例

```python
# 好的提示
prompt = """
Analyze EUR/USD sentiment:
- Current trend: Uptrend
- RSI: 65
- News: Fed rate decision pending

Output format:
Sentiment: [Bullish/Bearish/Neutral]
Confidence: [0-100]%
Reason: [Brief explanation]
"""

# 避免模糊提示
prompt = "What do you think about EUR/USD?"
```

## 故障排除

### 问题：模型加载失败

**可能原因：**
- 模型文件路径错误
- 模型文件损坏
- 内存不足

**解决方案：**
```python
# 检查文件存在
from pathlib import Path
if not Path(model_path).exists():
    print(f"Model file not found: {model_path}")

# 检查内存
available = ModelManager.get_available_memory_mb()
if available < 2048:
    print(f"Insufficient memory: {available:.2f} MB")
```

### 问题：生成速度慢

**优化方案：**
- 减少 `n_ctx` 大小
- 降低 `max_tokens`
- 增加 `n_threads`
- 使用更小的模型（1B vs 3B）

### 问题：生成质量差

**改进方案：**
- 调整 `temperature`（降低以获得更确定的输出）
- 优化提示词
- 使用更大的模型
- 提供更多上下文

## 模型下载

### Hugging Face

```bash
# 使用huggingface-cli下载
pip install huggingface-hub

# 下载Llama 3.2 1B
huggingface-cli download TheBloke/Llama-3.2-1B-GGUF llama-3.2-1b.Q4_K_M.gguf --local-dir models/

# 下载Llama 3.2 3B
huggingface-cli download TheBloke/Llama-3.2-3B-GGUF llama-3.2-3b.Q4_K_M.gguf --local-dir models/
```

### 推荐量化版本

- **Q4_K_M**: 平衡质量和大小（推荐）
- **Q5_K_M**: 更高质量，稍大
- **Q8_0**: 最高质量，接近原始大小

## 集成到交易系统

```python
from src.llm import ModelManager
from src.agents.market_analyst import MarketAnalystAgent

class TradingSystem:
    def __init__(self):
        self.llm_manager = ModelManager(
            model_path="models/llama-3.2-1b.gguf",
            memory_threshold_mb=1024
        )
        self.market_analyst = MarketAnalystAgent(self.llm_manager)
    
    def analyze_and_trade(self, market_data):
        # 加载模型
        if self.llm_manager.load_model():
            # 进行分析
            analysis = self.market_analyst.analyze(market_data)
            
            # 生成交易决策
            decision = self.make_decision(analysis)
            
            # 卸载模型
            self.llm_manager.unload_model()
            
            return decision
```

## 参考资源

- [llama-cpp-python文档](https://github.com/abetlen/llama-cpp-python)
- [Llama模型介绍](https://ai.meta.com/llama/)
- [GGUF格式说明](https://github.com/ggerganov/ggml)
