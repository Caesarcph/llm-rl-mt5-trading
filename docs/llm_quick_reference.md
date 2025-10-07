# LLM模块快速参考

## 快速开始

### 1. 基础使用（推荐）

```python
from src.llm import LlamaModel, ModelConfig

# 创建配置
config = ModelConfig(
    model_path="models/llama-3.2-1b.gguf",
    n_ctx=2048,
    temperature=0.7
)

# 使用上下文管理器（自动管理资源）
with LlamaModel(config) as model:
    response = model.generate("Analyze EUR/USD trend:")
    print(response)
```

### 2. 使用模型管理器（推荐用于生产环境）

```python
from src.llm import ModelManager

# 创建管理器（带内存保护）
manager = ModelManager(
    model_path="models/llama-3.2-1b.gguf",
    memory_threshold_mb=1024
)

# 加载模型
if manager.load_model():
    model = manager.get_model()
    response = model.generate("Your prompt here")
    manager.unload_model()
else:
    print("内存不足，无法加载模型")
```

## 常用配置

### 低配置环境（4GB RAM）
```python
config = ModelConfig(
    model_path="models/llama-3.2-1b.gguf",
    n_ctx=1024,
    n_threads=2,
    temperature=0.7,
    max_tokens=256
)
```

### 标准配置（8GB RAM）
```python
config = ModelConfig(
    model_path="models/llama-3.2-3b.gguf",
    n_ctx=2048,
    n_threads=4,
    temperature=0.7,
    max_tokens=512
)
```

### 高性能配置（16GB+ RAM）
```python
config = ModelConfig(
    model_path="models/llama-3.2-3b.gguf",
    n_ctx=4096,
    n_threads=8,
    temperature=0.6,
    max_tokens=1024
)
```

## 常用方法

### 文本生成
```python
response = model.generate(
    prompt="Your prompt",
    max_tokens=300,
    temperature=0.7
)
```

### 聊天模式
```python
messages = [
    {"role": "system", "content": "You are a forex analyst."},
    {"role": "user", "content": "Analyze gold trend"}
]
response = model.chat(messages)
```

### 内存监控
```python
# 获取内存统计
stats = manager.get_memory_stats()
print(f"可用内存: {stats['available_mb']:.2f} MB")
print(f"使用率: {stats['percent']:.1f}%")

# 检查可用内存
available = ModelManager.get_available_memory_mb()
```

## 实用提示词模板

### 新闻情绪分析
```python
prompt = f"""
Analyze the sentiment of this news:
"{headline}"

Provide:
1. Sentiment: Bullish/Bearish/Neutral
2. Confidence: 0-100%
3. Impact: High/Medium/Low
4. Brief explanation
"""
```

### 市场状态分析
```python
prompt = f"""
Market Analysis:
Symbol: {symbol}
Trend: {trend}
RSI: {rsi}

Provide:
1. Market regime
2. Risk level
3. Trading recommendation
"""
```

### 交易决策
```python
prompt = f"""
Trading Signal:
Symbol: {symbol}
Direction: {direction}
Strength: {strength}

Should I take this trade?
Consider: Risk/Reward, Timing, Market conditions
"""
```

## 错误处理

### 检查模型是否加载
```python
if not model.is_loaded:
    print("模型未加载")
    return

response = model.generate(prompt)
```

### 捕获生成错误
```python
try:
    response = model.generate(prompt)
except RuntimeError as e:
    print(f"生成失败: {e}")
except Exception as e:
    print(f"未知错误: {e}")
```

### 内存不足处理
```python
if not manager._check_memory_available():
    print("内存不足，使用备用方案")
    # 使用更小的模型或降低配置
```

## 性能优化

### 1. 调用频率控制
```python
import time

last_call = 0
min_interval = 5.0  # 5秒

def can_call_llm():
    global last_call
    now = time.time()
    if now - last_call >= min_interval:
        last_call = now
        return True
    return False
```

### 2. 结果缓存
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_analysis(prompt: str) -> str:
    return model.generate(prompt)
```

### 3. 及时卸载
```python
# 使用完立即卸载
with LlamaModel(config) as model:
    result = model.generate(prompt)
# 自动卸载

# 或手动卸载
manager.unload_model()
```

## 常见问题

### Q: 如何选择1B还是3B模型？
**A:** 
- 1B: 内存受限、需要快速响应
- 3B: 需要更好的分析质量

### Q: 生成速度慢怎么办？
**A:**
- 增加 `n_threads`
- 减少 `max_tokens`
- 使用更小的模型
- 降低 `n_ctx`

### Q: 内存不足怎么办？
**A:**
- 使用1B模型
- 减少 `n_ctx` (如1024)
- 及时卸载模型
- 关闭其他程序

### Q: 如何提高生成质量？
**A:**
- 使用3B模型
- 优化提示词
- 调整 `temperature` (降低获得更确定输出)
- 增加上下文信息

## 下载模型

```bash
# 安装huggingface-cli
pip install huggingface-hub

# 下载1B模型（推荐）
huggingface-cli download TheBloke/Llama-3.2-1B-GGUF \
  llama-3.2-1b.Q4_K_M.gguf --local-dir models/

# 下载3B模型
huggingface-cli download TheBloke/Llama-3.2-3B-GGUF \
  llama-3.2-3b.Q4_K_M.gguf --local-dir models/
```

## 完整示例

```python
from src.llm import ModelManager

def analyze_market_with_llm(market_data: dict) -> str:
    """使用LLM分析市场"""
    
    # 创建管理器
    manager = ModelManager(
        model_path="models/llama-3.2-1b.gguf",
        memory_threshold_mb=1024
    )
    
    # 检查内存
    stats = manager.get_memory_stats()
    if stats['available_mb'] < 1024:
        return "内存不足，无法进行LLM分析"
    
    # 加载模型
    if not manager.load_model():
        return "模型加载失败"
    
    try:
        # 获取模型
        model = manager.get_model()
        
        # 构建提示
        prompt = f"""
        Market Analysis:
        EUR/USD: {market_data['eurusd']['trend']}
        Gold: {market_data['gold']['state']}
        VIX: {market_data['vix']}
        
        Provide brief market outlook and risk assessment.
        """
        
        # 生成分析
        analysis = model.generate(prompt, max_tokens=300)
        
        return analysis
        
    finally:
        # 确保卸载模型
        manager.unload_model()

# 使用
market_data = {
    'eurusd': {'trend': 'Uptrend'},
    'gold': {'state': 'Volatile'},
    'vix': 25
}

result = analyze_market_with_llm(market_data)
print(result)
```

## 相关文档

- **详细指南**: `docs/llm_module_guide.md`
- **模块README**: `src/llm/README.md`
- **使用示例**: `examples/llm_demo.py`
- **配置模板**: `config/llm_config.yaml`
- **测试文件**: `tests/test_llm.py`
