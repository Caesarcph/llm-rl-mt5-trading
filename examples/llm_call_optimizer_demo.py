"""
LLM调用优化器演示
展示如何使用智能调用频率控制、缓存和异步处理
"""

import asyncio
import time
from src.llm.call_optimizer import (
    LLMCallOptimizer,
    CallPriority,
    LLMResultCache
)


def simulate_llm_call(prompt: str, delay: float = 0.5) -> str:
    """模拟LLM调用（实际应用中会调用真实的LLM）"""
    print(f"  [LLM] Processing: {prompt[:50]}...")
    time.sleep(delay)  # 模拟LLM处理时间
    return f"Response to: {prompt}"


def demo_basic_usage():
    """演示基本用法"""
    print("\n=== 演示1: 基本用法 ===")
    
    # 创建优化器
    optimizer = LLMCallOptimizer({
        'enable_cache': True,
        'enable_rate_limit': True,
        'base_interval': 2.0,  # 基础间隔2秒
        'cache_ttl': 300  # 缓存5分钟
    })
    
    # 第一次调用
    print("\n第一次调用（无缓存）:")
    start = time.time()
    result1 = optimizer.call(
        simulate_llm_call,
        "What is the market sentiment for EURUSD?",
        cache_key="eurusd_sentiment"
    )
    print(f"  结果: {result1}")
    print(f"  耗时: {time.time() - start:.2f}秒")
    
    # 第二次调用（使用缓存）
    print("\n第二次调用（使用缓存）:")
    start = time.time()
    result2 = optimizer.call(
        simulate_llm_call,
        "What is the market sentiment for EURUSD?",
        cache_key="eurusd_sentiment"
    )
    print(f"  结果: {result2}")
    print(f"  耗时: {time.time() - start:.2f}秒")
    
    # 显示统计
    stats = optimizer.get_stats()
    print(f"\n统计信息:")
    print(f"  总调用次数: {stats['metrics']['total_calls']}")
    print(f"  缓存命中率: {stats['metrics']['cache_hit_rate']:.1%}")
    print(f"  平均延迟: {stats['metrics']['avg_latency']:.3f}秒")


def demo_priority_control():
    """演示优先级控制"""
    print("\n\n=== 演示2: 优先级控制 ===")
    
    optimizer = LLMCallOptimizer({
        'enable_cache': False,
        'enable_rate_limit': True,
        'base_interval': 3.0
    })
    
    # 低优先级调用
    print("\n低优先级调用:")
    optimizer.call(
        simulate_llm_call,
        "Generate daily market summary",
        0.1,
        priority=CallPriority.LOW
    )
    
    print("等待速率限制...")
    start = time.time()
    optimizer.call(
        simulate_llm_call,
        "Another low priority call",
        0.1,
        priority=CallPriority.LOW
    )
    low_wait = time.time() - start
    print(f"  低优先级等待时间: {low_wait:.2f}秒")
    
    # 高优先级调用
    print("\n高优先级调用:")
    start = time.time()
    optimizer.call(
        simulate_llm_call,
        "URGENT: Analyze breaking news",
        0.1,
        priority=CallPriority.HIGH
    )
    high_wait = time.time() - start
    print(f"  高优先级等待时间: {high_wait:.2f}秒")
    
    print(f"\n优先级效果: 高优先级比低优先级快 {low_wait/high_wait:.1f}x")


def demo_cache_strategies():
    """演示缓存策略"""
    print("\n\n=== 演示3: 缓存策略 ===")
    
    optimizer = LLMCallOptimizer({
        'enable_cache': True,
        'enable_rate_limit': False,
        'cache_ttl': 10  # 短缓存用于演示
    })
    
    # 不同的缓存键
    prompts = [
        "Analyze EURUSD",
        "Analyze GBPUSD",
        "Analyze XAUUSD"
    ]
    
    print("\n首次调用（建立缓存）:")
    for prompt in prompts:
        cache_key = LLMResultCache.generate_key(prompt)
        optimizer.call(
            simulate_llm_call,
            prompt,
            0.2,
            cache_key=cache_key
        )
    
    print("\n再次调用（使用缓存）:")
    start = time.time()
    for prompt in prompts:
        cache_key = LLMResultCache.generate_key(prompt)
        optimizer.call(
            simulate_llm_call,
            prompt,
            0.2,
            cache_key=cache_key
        )
    total_time = time.time() - start
    print(f"  3次缓存调用总耗时: {total_time:.3f}秒")
    
    # 绕过缓存
    print("\n绕过缓存调用:")
    start = time.time()
    cache_key = LLMResultCache.generate_key(prompts[0])
    optimizer.call(
        simulate_llm_call,
        prompts[0],
        0.2,
        cache_key=cache_key,
        bypass_cache=True
    )
    bypass_time = time.time() - start
    print(f"  绕过缓存耗时: {bypass_time:.3f}秒")
    
    stats = optimizer.get_stats()
    print(f"\n缓存统计:")
    print(f"  缓存大小: {stats['cache']['size']}")
    print(f"  缓存命中率: {stats['metrics']['cache_hit_rate']:.1%}")


async def demo_async_calls():
    """演示异步调用"""
    print("\n\n=== 演示4: 异步调用 ===")
    
    optimizer = LLMCallOptimizer({
        'enable_cache': False,
        'enable_rate_limit': False,
        'max_concurrent': 3
    })
    
    prompts = [
        "Analyze market trend",
        "Generate trading signals",
        "Calculate risk metrics",
        "Evaluate portfolio",
        "Predict price movement"
    ]
    
    # 串行执行
    print("\n串行执行:")
    start = time.time()
    for prompt in prompts:
        simulate_llm_call(prompt, 0.3)
    serial_time = time.time() - start
    print(f"  串行总耗时: {serial_time:.2f}秒")
    
    # 并行执行
    print("\n并行执行（最大并发3）:")
    start = time.time()
    tasks = [
        optimizer.call_async(
            simulate_llm_call,
            prompt,
            0.3,
            priority=CallPriority.MEDIUM
        )
        for prompt in prompts
    ]
    await asyncio.gather(*tasks)
    parallel_time = time.time() - start
    print(f"  并行总耗时: {parallel_time:.2f}秒")
    
    speedup = serial_time / parallel_time
    print(f"\n性能提升: {speedup:.1f}x")


def demo_adaptive_rate_limiting():
    """演示自适应速率限制"""
    print("\n\n=== 演示5: 自适应速率限制 ===")
    
    optimizer = LLMCallOptimizer({
        'enable_cache': False,
        'enable_rate_limit': True,
        'base_interval': 2.0,
        'min_interval': 0.5,
        'max_interval': 10.0
    })
    
    print("\n快速连续调用（触发自适应）:")
    for i in range(5):
        start = time.time()
        optimizer.call(
            simulate_llm_call,
            f"Call {i+1}",
            0.1,
            priority=CallPriority.MEDIUM
        )
        wait_time = time.time() - start - 0.1
        
        stats = optimizer.rate_limiter.get_stats()
        print(f"  调用 {i+1}: 等待 {wait_time:.2f}秒, "
              f"当前间隔 {stats['current_interval']:.2f}秒")


def demo_performance_monitoring():
    """演示性能监控"""
    print("\n\n=== 演示6: 性能监控 ===")
    
    optimizer = LLMCallOptimizer({
        'enable_cache': True,
        'enable_rate_limit': True,
        'base_interval': 1.0
    })
    
    # 执行一系列调用
    print("\n执行混合调用:")
    calls = [
        ("Analyze EURUSD", "eurusd", CallPriority.HIGH),
        ("Analyze GBPUSD", "gbpusd", CallPriority.MEDIUM),
        ("Analyze EURUSD", "eurusd", CallPriority.LOW),  # 缓存命中
        ("Generate report", "report", CallPriority.LOW),
        ("Analyze GBPUSD", "gbpusd", CallPriority.HIGH),  # 缓存命中
    ]
    
    for prompt, key, priority in calls:
        try:
            optimizer.call(
                simulate_llm_call,
                prompt,
                0.2,
                cache_key=key,
                priority=priority
            )
        except Exception as e:
            print(f"  错误: {e}")
    
    # 显示详细统计
    stats = optimizer.get_stats()
    print("\n=== 性能统计 ===")
    print(f"调用指标:")
    print(f"  总调用次数: {stats['metrics']['total_calls']}")
    print(f"  缓存命中: {stats['metrics']['cache_hit_rate']:.1%}")
    print(f"  平均延迟: {stats['metrics']['avg_latency']:.3f}秒")
    print(f"  被限流次数: {stats['metrics']['throttled_calls']}")
    print(f"  失败次数: {stats['metrics']['failed_calls']}")
    
    print(f"\n速率限制器:")
    print(f"  当前间隔: {stats['rate_limiter']['current_interval']:.2f}秒")
    print(f"  调用频率: {stats['rate_limiter']['call_rate']:.2f} 次/秒")
    
    print(f"\n缓存:")
    print(f"  缓存大小: {stats['cache']['size']}")
    print(f"  总访问次数: {stats['cache']['total_accesses']}")


def main():
    """运行所有演示"""
    print("=" * 60)
    print("LLM调用优化器演示")
    print("=" * 60)
    
    # 基本用法
    demo_basic_usage()
    
    # 优先级控制
    demo_priority_control()
    
    # 缓存策略
    demo_cache_strategies()
    
    # 异步调用
    print("\n运行异步演示...")
    asyncio.run(demo_async_calls())
    
    # 自适应速率限制
    demo_adaptive_rate_limiting()
    
    # 性能监控
    demo_performance_monitoring()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
