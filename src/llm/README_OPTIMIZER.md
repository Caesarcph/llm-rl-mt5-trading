# LLM调用优化器

LLM调用优化器提供智能调用频率控制、结果缓存和异步处理功能，用于优化LLM调用的性能和资源使用。

## 功能特性

### 1. 自适应速率限制 (AdaptiveRateLimiter)

自动调整调用间隔，平衡性能和资源使用：

- **基础间隔控制**: 设置最小、最大和基础调用间隔
- **优先级支持**: 根据调用优先级动态调整等待时间
- **自适应调整**: 根据调用频率自动调整间隔
- **统计监控**: 实时监控调用频率和间隔

```python
from src.llm.call_optimizer import AdaptiveRateLimiter, CallPriority

limiter = AdaptiveRateLimiter(
    base_interval=5.0,    # 基础间隔5秒
    min_interval=1.0,     # 最小间隔1秒
    max_interval=30.0     # 最大间隔30秒
)

# 获取调用许可
limiter.acquire(CallPriority.HIGH)  # 高优先级，等待时间更短
```

### 2. LLM结果缓存 (LLMResultCache)

智能缓存LLM调用结果，避免重复计算：

- **TTL过期**: 支持自定义过期时间
- **LRU驱逐**: 缓存满时驱逐最少使用的条目
- **访问统计**: 跟踪缓存访问次数和命中率
- **自动清理**: 定期清理过期缓存

```python
from src.llm.call_optimizer import LLMResultCache

cache = LLMResultCache(
    default_ttl=3600,     # 默认1小时过期
    max_size=1000         # 最大1000条缓存
)

# 生成缓存键
key = LLMResultCache.generate_key(prompt, params)

# 设置和获取缓存
cache.set(key, result, ttl=1800)
cached_result = cache.get(key)
```

### 3. 异步LLM执行器 (AsyncLLMExecutor)

支持异步和批量LLM调用：

- **并发控制**: 限制最大并发数
- **批量处理**: 批量执行多个LLM调用
- **异常处理**: 优雅处理异步调用中的异常

```python
from src.llm.call_optimizer import AsyncLLMExecutor
import asyncio

executor = AsyncLLMExecutor(max_concurrent=3)

# 异步执行
async def run():
    result = await executor.execute_async(llm_function, prompt)
    return result

# 批量执行
calls = [
    (llm_function, (prompt1,), {}),
    (llm_function, (prompt2,), {}),
]
results = await executor.execute_batch(calls)
```

### 4. 统一调用优化器 (LLMCallOptimizer)

集成所有优化功能的统一接口：

```python
from src.llm.call_optimizer import LLMCallOptimizer, CallPriority

# 创建优化器
optimizer = LLMCallOptimizer({
    'base_interval': 5.0,
    'cache_ttl': 3600,
    'max_concurrent': 3,
    'enable_cache': True,
    'enable_rate_limit': True
})

# 同步调用
result = optimizer.call(
    llm_function,
    prompt,
    priority=CallPriority.HIGH,
    cache_key="my_key",
    cache_ttl=1800
)

# 异步调用
result = await optimizer.call_async(
    llm_function,
    prompt,
    priority=CallPriority.MEDIUM,
    cache_key="my_key"
)

# 获取统计信息
stats = optimizer.get_stats()
print(f"缓存命中率: {stats['metrics']['cache_hit_rate']:.1%}")
```

## 配置选项

### LLMCallOptimizer配置

```python
config = {
    # 速率限制配置
    'base_interval': 5.0,      # 基础调用间隔（秒）
    'min_interval': 1.0,       # 最小间隔（秒）
    'max_interval': 30.0,      # 最大间隔（秒）
    
    # 缓存配置
    'cache_ttl': 3600,         # 默认缓存过期时间（秒）
    'cache_max_size': 1000,    # 最大缓存条目数
    
    # 异步配置
    'max_concurrent': 3,       # 最大并发数
    
    # 功能开关
    'enable_cache': True,      # 启用缓存
    'enable_rate_limit': True  # 启用速率限制
}
```

## 调用优先级

系统支持4个优先级级别：

- `CallPriority.LOW`: 低优先级（等待时间 × 1.5）
- `CallPriority.MEDIUM`: 中等优先级（等待时间 × 1.0）
- `CallPriority.HIGH`: 高优先级（等待时间 × 0.7）
- `CallPriority.CRITICAL`: 关键优先级（等待时间 × 0.3）

## 性能指标

### CallMetrics

跟踪以下指标：

- `total_calls`: 总调用次数
- `cache_hits`: 缓存命中次数
- `cache_misses`: 缓存未命中次数
- `avg_latency`: 平均延迟（秒）
- `throttled_calls`: 被限流的调用次数
- `failed_calls`: 失败的调用次数

### 获取统计信息

```python
stats = optimizer.get_stats()

# 调用指标
print(f"总调用: {stats['metrics']['total_calls']}")
print(f"缓存命中率: {stats['metrics']['cache_hit_rate']:.1%}")
print(f"平均延迟: {stats['metrics']['avg_latency']:.3f}秒")

# 速率限制器状态
print(f"当前间隔: {stats['rate_limiter']['current_interval']:.2f}秒")
print(f"调用频率: {stats['rate_limiter']['call_rate']:.2f} 次/秒")

# 缓存状态
print(f"缓存大小: {stats['cache']['size']}")
print(f"总访问次数: {stats['cache']['total_accesses']}")
```

## 使用场景

### 1. 新闻情绪分析

```python
def analyze_sentiment(news_text: str) -> dict:
    # 生成缓存键
    cache_key = LLMResultCache.generate_key(news_text)
    
    # 使用优化器调用
    result = optimizer.call(
        llm_model.generate,
        prompt=f"Analyze sentiment: {news_text}",
        priority=CallPriority.MEDIUM,
        cache_key=cache_key,
        cache_ttl=3600  # 1小时缓存
    )
    
    return parse_sentiment(result)
```

### 2. 市场评论生成

```python
def generate_commentary(symbol: str, market_data: dict) -> str:
    cache_key = f"commentary_{symbol}_{market_data['timestamp']}"
    
    result = optimizer.call(
        llm_model.generate,
        prompt=build_commentary_prompt(symbol, market_data),
        priority=CallPriority.LOW,  # 评论优先级较低
        cache_key=cache_key,
        cache_ttl=1800  # 30分钟缓存
    )
    
    return result
```

### 3. 批量新闻分析

```python
async def analyze_news_batch(news_items: list) -> list:
    # 创建异步任务
    tasks = [
        optimizer.call_async(
            analyze_sentiment,
            news.text,
            priority=CallPriority.HIGH
        )
        for news in news_items
    ]
    
    # 并行执行
    results = await asyncio.gather(*tasks)
    return results
```

## 最佳实践

### 1. 缓存键设计

使用内容哈希作为缓存键，确保相同输入产生相同的键：

```python
# 好的做法
cache_key = LLMResultCache.generate_key(prompt, params)

# 避免使用时间戳等变化的值
# cache_key = f"{prompt}_{datetime.now()}"  # 不好
```

### 2. 优先级分配

根据业务重要性分配优先级：

- 关键交易决策: `CRITICAL`
- 实时市场分析: `HIGH`
- 新闻情绪分析: `MEDIUM`
- 日报生成: `LOW`

### 3. 缓存TTL设置

根据数据时效性设置TTL：

- 实时市场数据: 60-300秒
- 新闻分析: 1800-3600秒
- 历史分析: 7200-86400秒

### 4. 错误处理

```python
try:
    result = optimizer.call(
        llm_function,
        prompt,
        cache_key=key
    )
except Exception as e:
    logger.error(f"LLM call failed: {e}")
    # 使用降级策略
    result = fallback_analysis(prompt)
```

### 5. 监控和调优

定期检查统计信息并调整配置：

```python
stats = optimizer.get_stats()

# 如果缓存命中率低，考虑增加TTL
if stats['metrics']['cache_hit_rate'] < 0.3:
    logger.warning("Low cache hit rate, consider increasing TTL")

# 如果被限流次数多，考虑增加间隔
if stats['metrics']['throttled_calls'] > 100:
    logger.warning("High throttle rate, consider adjusting intervals")
```

## 性能优化效果

基于测试结果：

- **缓存性能提升**: 3-10倍（取决于LLM调用延迟）
- **异步批量处理**: 1.5-3倍（取决于并发数和任务数）
- **内存使用**: 每1000条缓存约占用1-5MB（取决于结果大小）

## 注意事项

1. **内存管理**: 大量缓存会占用内存，根据系统资源调整`cache_max_size`
2. **线程安全**: 所有组件都是线程安全的，可以在多线程环境中使用
3. **异步兼容**: 异步方法需要在async函数中调用
4. **速率限制**: 过于激进的速率限制可能影响系统响应性

## 示例代码

完整示例请参考 `examples/llm_call_optimizer_demo.py`

## 测试

运行测试：

```bash
python -m pytest tests/test_llm_call_optimizer.py -v
```

测试覆盖：
- 速率限制功能
- 缓存机制
- 异步执行
- 性能优化效果
- 错误处理
