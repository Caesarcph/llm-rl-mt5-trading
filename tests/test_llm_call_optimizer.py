"""
LLM调用优化器测试
测试智能调用频率控制、缓存机制和异步处理
"""

import unittest
import time
import asyncio
from unittest.mock import Mock, patch
from datetime import datetime

from src.llm.call_optimizer import (
    LLMCallOptimizer,
    CallPriority,
    LLMResultCache,
    AdaptiveRateLimiter,
    AsyncLLMExecutor,
    CallMetrics
)


class TestCallMetrics(unittest.TestCase):
    """测试调用指标"""
    
    def test_metrics_initialization(self):
        """测试指标初始化"""
        metrics = CallMetrics()
        
        self.assertEqual(metrics.total_calls, 0)
        self.assertEqual(metrics.cache_hits, 0)
        self.assertEqual(metrics.cache_misses, 0)
        self.assertEqual(metrics.avg_latency, 0.0)
    
    def test_update_latency(self):
        """测试延迟更新"""
        metrics = CallMetrics()
        
        metrics.update_latency(1.0)
        self.assertEqual(metrics.total_calls, 1)
        self.assertEqual(metrics.avg_latency, 1.0)
        
        metrics.update_latency(2.0)
        self.assertEqual(metrics.total_calls, 2)
        self.assertEqual(metrics.avg_latency, 1.5)
    
    def test_cache_hit_rate(self):
        """测试缓存命中率计算"""
        metrics = CallMetrics()
        
        # 无调用时
        self.assertEqual(metrics.get_cache_hit_rate(), 0.0)
        
        # 有命中和未命中
        metrics.cache_hits = 7
        metrics.cache_misses = 3
        self.assertEqual(metrics.get_cache_hit_rate(), 0.7)


class TestAdaptiveRateLimiter(unittest.TestCase):
    """测试自适应速率限制器"""
    
    def test_initialization(self):
        """测试初始化"""
        limiter = AdaptiveRateLimiter(
            base_interval=5.0,
            min_interval=1.0,
            max_interval=30.0
        )
        
        self.assertEqual(limiter.base_interval, 5.0)
        self.assertEqual(limiter.min_interval, 1.0)
        self.assertEqual(limiter.max_interval, 30.0)
        self.assertEqual(limiter.current_interval, 5.0)
    
    def test_acquire_first_call(self):
        """测试首次调用"""
        limiter = AdaptiveRateLimiter(base_interval=1.0)
        
        start_time = time.time()
        wait_time = limiter.acquire(CallPriority.MEDIUM)
        elapsed = time.time() - start_time
        
        # 首次调用不应等待
        self.assertLess(elapsed, 0.1)
        self.assertEqual(wait_time, 0.0)
    
    def test_acquire_with_interval(self):
        """测试带间隔的调用"""
        limiter = AdaptiveRateLimiter(base_interval=0.5)
        
        # 第一次调用
        limiter.acquire(CallPriority.MEDIUM)
        
        # 第二次调用应该等待
        start_time = time.time()
        wait_time = limiter.acquire(CallPriority.MEDIUM)
        elapsed = time.time() - start_time
        
        self.assertGreater(wait_time, 0.0)
        self.assertGreaterEqual(elapsed, 0.4)  # 允许一些误差
    
    def test_priority_adjustment(self):
        """测试优先级调整"""
        limiter = AdaptiveRateLimiter(base_interval=1.0)
        
        # 第一次调用
        limiter.acquire(CallPriority.MEDIUM)
        
        # 高优先级调用应该等待更短
        start_time = time.time()
        limiter.acquire(CallPriority.HIGH)
        high_elapsed = time.time() - start_time
        
        # 重置
        limiter.last_call_time = 0
        limiter.acquire(CallPriority.MEDIUM)
        
        # 低优先级调用应该等待更长
        start_time = time.time()
        limiter.acquire(CallPriority.LOW)
        low_elapsed = time.time() - start_time
        
        self.assertLess(high_elapsed, low_elapsed)
    
    def test_get_stats(self):
        """测试获取统计信息"""
        limiter = AdaptiveRateLimiter(base_interval=1.0)
        
        limiter.acquire(CallPriority.MEDIUM)
        stats = limiter.get_stats()
        
        self.assertIn('current_interval', stats)
        self.assertIn('call_rate', stats)
        self.assertIn('recent_calls', stats)
        self.assertEqual(stats['recent_calls'], 1)


class TestLLMResultCache(unittest.TestCase):
    """测试LLM结果缓存"""
    
    def test_initialization(self):
        """测试初始化"""
        cache = LLMResultCache(default_ttl=3600, max_size=100)
        
        self.assertEqual(cache.default_ttl, 3600)
        self.assertEqual(cache.max_size, 100)
        self.assertEqual(len(cache.cache), 0)
    
    def test_set_and_get(self):
        """测试设置和获取缓存"""
        cache = LLMResultCache()
        
        key = "test_key"
        value = "test_value"
        
        cache.set(key, value)
        result = cache.get(key)
        
        self.assertEqual(result, value)
    
    def test_cache_miss(self):
        """测试缓存未命中"""
        cache = LLMResultCache()
        
        result = cache.get("nonexistent_key")
        self.assertIsNone(result)
    
    def test_cache_expiration(self):
        """测试缓存过期"""
        cache = LLMResultCache(default_ttl=1)
        
        key = "test_key"
        value = "test_value"
        
        cache.set(key, value, ttl=1)
        
        # 立即获取应该成功
        result = cache.get(key)
        self.assertEqual(result, value)
        
        # 等待过期
        time.sleep(1.1)
        
        # 过期后应该返回None
        result = cache.get(key)
        self.assertIsNone(result)
    
    def test_max_size_eviction(self):
        """测试最大大小驱逐"""
        cache = LLMResultCache(max_size=3)
        
        # 添加3个条目
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        
        self.assertEqual(len(cache.cache), 3)
        
        # 添加第4个条目应该驱逐最少使用的
        cache.set("key4", "value4")
        
        self.assertEqual(len(cache.cache), 3)
    
    def test_lru_eviction(self):
        """测试LRU驱逐策略"""
        cache = LLMResultCache(max_size=2)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # 访问key1，增加其访问计数
        cache.get("key1")
        cache.get("key1")
        
        # 添加key3应该驱逐key2（访问次数少）
        cache.set("key3", "value3")
        
        self.assertIsNotNone(cache.get("key1"))
        self.assertIsNone(cache.get("key2"))
        self.assertIsNotNone(cache.get("key3"))
    
    def test_clear(self):
        """测试清空缓存"""
        cache = LLMResultCache()
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        self.assertEqual(len(cache.cache), 2)
        
        cache.clear()
        
        self.assertEqual(len(cache.cache), 0)
    
    def test_get_stats(self):
        """测试获取统计信息"""
        cache = LLMResultCache()
        
        cache.set("key1", "value1")
        cache.get("key1")
        cache.get("key1")
        
        stats = cache.get_stats()
        
        self.assertEqual(stats['size'], 1)
        self.assertEqual(stats['total_accesses'], 2)
    
    def test_generate_key(self):
        """测试生成缓存键"""
        prompt1 = "Test prompt"
        params1 = {"temperature": 0.7}
        
        key1 = LLMResultCache.generate_key(prompt1, params1)
        
        # 相同输入应该生成相同的键
        key2 = LLMResultCache.generate_key(prompt1, params1)
        self.assertEqual(key1, key2)
        
        # 不同输入应该生成不同的键
        key3 = LLMResultCache.generate_key("Different prompt", params1)
        self.assertNotEqual(key1, key3)


class TestAsyncLLMExecutor(unittest.TestCase):
    """测试异步LLM执行器"""
    
    def test_initialization(self):
        """测试初始化"""
        executor = AsyncLLMExecutor(max_concurrent=3)
        
        self.assertEqual(executor.max_concurrent, 3)
        self.assertIsNotNone(executor.semaphore)
    
    def test_execute_async(self):
        """测试异步执行"""
        executor = AsyncLLMExecutor(max_concurrent=2)
        
        def mock_function(x):
            time.sleep(0.1)
            return x * 2
        
        async def run_test():
            result = await executor.execute_async(mock_function, 5)
            return result
        
        result = asyncio.run(run_test())
        self.assertEqual(result, 10)
    
    def test_execute_batch(self):
        """测试批量执行"""
        executor = AsyncLLMExecutor(max_concurrent=3)
        
        def mock_function(x):
            time.sleep(0.1)
            return x * 2
        
        calls = [
            (mock_function, (1,), {}),
            (mock_function, (2,), {}),
            (mock_function, (3,), {}),
        ]
        
        async def run_test():
            results = await executor.execute_batch(calls)
            return results
        
        results = asyncio.run(run_test())
        self.assertEqual(results, [2, 4, 6])
    
    def test_concurrent_execution(self):
        """测试并发执行"""
        executor = AsyncLLMExecutor(max_concurrent=2)
        
        def slow_function(x):
            time.sleep(0.2)
            return x
        
        calls = [(slow_function, (i,), {}) for i in range(4)]
        
        async def run_test():
            start_time = time.time()
            results = await executor.execute_batch(calls)
            elapsed = time.time() - start_time
            return results, elapsed
        
        results, elapsed = asyncio.run(run_test())
        
        # 4个任务，每个0.2秒，最大并发2，应该约0.4秒完成
        self.assertEqual(results, [0, 1, 2, 3])
        self.assertLess(elapsed, 0.6)  # 允许一些误差
        self.assertGreater(elapsed, 0.3)


class TestLLMCallOptimizer(unittest.TestCase):
    """测试LLM调用优化器"""
    
    def test_initialization(self):
        """测试初始化"""
        optimizer = LLMCallOptimizer()
        
        self.assertIsNotNone(optimizer.rate_limiter)
        self.assertIsNotNone(optimizer.cache)
        self.assertIsNotNone(optimizer.async_executor)
        self.assertIsNotNone(optimizer.metrics)
    
    def test_call_without_cache(self):
        """测试无缓存调用"""
        config = {
            'enable_cache': False,
            'enable_rate_limit': False
        }
        optimizer = LLMCallOptimizer(config)
        
        def mock_function(x):
            return x * 2
        
        result = optimizer.call(mock_function, 5)
        
        self.assertEqual(result, 10)
        self.assertEqual(optimizer.metrics.total_calls, 1)
    
    def test_call_with_cache(self):
        """测试带缓存调用"""
        config = {
            'enable_cache': True,
            'enable_rate_limit': False
        }
        optimizer = LLMCallOptimizer(config)
        
        call_count = [0]
        
        def mock_function(x):
            call_count[0] += 1
            return x * 2
        
        cache_key = "test_key"
        
        # 第一次调用
        result1 = optimizer.call(mock_function, 5, cache_key=cache_key)
        self.assertEqual(result1, 10)
        self.assertEqual(call_count[0], 1)
        self.assertEqual(optimizer.metrics.cache_misses, 1)
        
        # 第二次调用应该使用缓存
        result2 = optimizer.call(mock_function, 5, cache_key=cache_key)
        self.assertEqual(result2, 10)
        self.assertEqual(call_count[0], 1)  # 函数不应该被再次调用
        self.assertEqual(optimizer.metrics.cache_hits, 1)
    
    def test_call_with_rate_limit(self):
        """测试带速率限制调用"""
        config = {
            'enable_cache': False,
            'enable_rate_limit': True,
            'base_interval': 0.5
        }
        optimizer = LLMCallOptimizer(config)
        
        def mock_function():
            return "result"
        
        # 第一次调用
        start_time = time.time()
        optimizer.call(mock_function)
        first_elapsed = time.time() - start_time
        
        # 第二次调用应该被限速
        start_time = time.time()
        optimizer.call(mock_function)
        second_elapsed = time.time() - start_time
        
        self.assertLess(first_elapsed, 0.1)
        self.assertGreater(second_elapsed, 0.4)
    
    def test_call_with_priority(self):
        """测试带优先级调用"""
        config = {
            'enable_cache': False,
            'enable_rate_limit': True,
            'base_interval': 1.0
        }
        optimizer = LLMCallOptimizer(config)
        
        def mock_function():
            return "result"
        
        # 第一次调用
        optimizer.call(mock_function)
        
        # 高优先级调用应该等待更短
        start_time = time.time()
        optimizer.call(mock_function, priority=CallPriority.HIGH)
        high_elapsed = time.time() - start_time
        
        # 低优先级调用应该等待更长
        start_time = time.time()
        optimizer.call(mock_function, priority=CallPriority.LOW)
        low_elapsed = time.time() - start_time
        
        self.assertLess(high_elapsed, low_elapsed)
    
    def test_call_bypass_cache(self):
        """测试绕过缓存"""
        config = {
            'enable_cache': True,
            'enable_rate_limit': False
        }
        optimizer = LLMCallOptimizer(config)
        
        call_count = [0]
        
        def mock_function():
            call_count[0] += 1
            return call_count[0]
        
        cache_key = "test_key"
        
        # 第一次调用
        result1 = optimizer.call(mock_function, cache_key=cache_key)
        self.assertEqual(result1, 1)
        
        # 绕过缓存调用
        result2 = optimizer.call(mock_function, cache_key=cache_key, bypass_cache=True)
        self.assertEqual(result2, 2)
        
        # 正常调用应该使用缓存
        result3 = optimizer.call(mock_function, cache_key=cache_key)
        self.assertEqual(result3, 1)  # 返回第一次的缓存结果
    
    def test_call_async(self):
        """测试异步调用"""
        config = {
            'enable_cache': True,
            'enable_rate_limit': False
        }
        optimizer = LLMCallOptimizer(config)
        
        def mock_function(x):
            time.sleep(0.1)
            return x * 2
        
        async def run_test():
            result = await optimizer.call_async(
                mock_function,
                5,
                cache_key="test_key"
            )
            return result
        
        result = asyncio.run(run_test())
        self.assertEqual(result, 10)
    
    def test_get_metrics(self):
        """测试获取指标"""
        optimizer = LLMCallOptimizer({'enable_rate_limit': False})
        
        def mock_function():
            return "result"
        
        optimizer.call(mock_function)
        
        metrics = optimizer.get_metrics()
        
        self.assertEqual(metrics.total_calls, 1)
        self.assertGreaterEqual(metrics.avg_latency, 0)  # 可能非常快，接近0
    
    def test_get_stats(self):
        """测试获取统计信息"""
        optimizer = LLMCallOptimizer({'enable_rate_limit': False, 'enable_cache': True})
        
        def mock_function():
            return "result"
        
        optimizer.call(mock_function, cache_key="key1")
        optimizer.call(mock_function, cache_key="key1")
        
        stats = optimizer.get_stats()
        
        self.assertIn('metrics', stats)
        self.assertIn('rate_limiter', stats)
        self.assertIn('cache', stats)
        
        self.assertEqual(stats['metrics']['total_calls'], 2)
        self.assertGreater(stats['metrics']['cache_hit_rate'], 0)
    
    def test_reset_metrics(self):
        """测试重置指标"""
        optimizer = LLMCallOptimizer({'enable_rate_limit': False})
        
        def mock_function():
            return "result"
        
        optimizer.call(mock_function)
        self.assertEqual(optimizer.metrics.total_calls, 1)
        
        optimizer.reset_metrics()
        self.assertEqual(optimizer.metrics.total_calls, 0)
    
    def test_clear_cache(self):
        """测试清空缓存"""
        optimizer = LLMCallOptimizer({'enable_rate_limit': False, 'enable_cache': True})
        
        def mock_function():
            return "result"
        
        optimizer.call(mock_function, cache_key="key1")
        self.assertEqual(len(optimizer.cache.cache), 1)
        
        optimizer.clear_cache()
        self.assertEqual(len(optimizer.cache.cache), 0)
    
    def test_error_handling(self):
        """测试错误处理"""
        optimizer = LLMCallOptimizer({'enable_rate_limit': False})
        
        def failing_function():
            raise ValueError("Test error")
        
        with self.assertRaises(ValueError):
            optimizer.call(failing_function)
        
        self.assertEqual(optimizer.metrics.failed_calls, 1)


class TestPerformanceOptimization(unittest.TestCase):
    """测试性能优化效果"""
    
    def test_cache_performance_improvement(self):
        """测试缓存性能提升"""
        config = {
            'enable_cache': True,
            'enable_rate_limit': False
        }
        optimizer = LLMCallOptimizer(config)
        
        def slow_function():
            time.sleep(0.2)
            return "result"
        
        cache_key = "perf_test"
        
        # 第一次调用（无缓存）
        start_time = time.time()
        result1 = optimizer.call(slow_function, cache_key=cache_key)
        first_call_time = time.time() - start_time
        
        # 第二次调用（有缓存）
        start_time = time.time()
        result2 = optimizer.call(slow_function, cache_key=cache_key)
        second_call_time = time.time() - start_time
        
        self.assertEqual(result1, result2)
        self.assertGreater(first_call_time, 0.15)
        self.assertLess(second_call_time, 0.05)  # 缓存应该快得多
        
        # 验证性能提升
        if second_call_time > 0:
            speedup = first_call_time / second_call_time
            self.assertGreater(speedup, 3)  # 至少3倍提升
    
    def test_async_batch_performance(self):
        """测试异步批量处理性能"""
        optimizer = LLMCallOptimizer({'enable_rate_limit': False})
        
        def slow_function(x):
            time.sleep(0.2)
            return x * 2
        
        # 串行执行
        start_time = time.time()
        serial_results = [slow_function(i) for i in range(3)]
        serial_time = time.time() - start_time
        
        # 并行执行
        async def parallel_execution():
            tasks = [
                optimizer.call_async(slow_function, i)
                for i in range(3)
            ]
            return await asyncio.gather(*tasks)
        
        start_time = time.time()
        parallel_results = asyncio.run(parallel_execution())
        parallel_time = time.time() - start_time
        
        self.assertEqual(serial_results, parallel_results)
        self.assertGreater(serial_time, 0.5)  # 至少0.6秒
        self.assertLess(parallel_time, 0.4)  # 应该在0.3秒左右
        
        # 验证性能提升
        speedup = serial_time / parallel_time
        self.assertGreater(speedup, 1.5)  # 至少1.5倍提升


if __name__ == '__main__':
    unittest.main()
