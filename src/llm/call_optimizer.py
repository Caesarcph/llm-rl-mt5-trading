"""
LLM调用优化器
实现智能调用频率控制、结果缓存和异步处理
"""

import logging
import time
import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable, List, Tuple
from enum import Enum
from collections import deque
import threading


logger = logging.getLogger(__name__)


class CallPriority(Enum):
    """调用优先级"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class CallMetrics:
    """调用指标"""
    total_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_latency: float = 0.0
    avg_latency: float = 0.0
    throttled_calls: int = 0
    failed_calls: int = 0
    
    def update_latency(self, latency: float) -> None:
        """更新延迟统计"""
        self.total_latency += latency
        self.total_calls += 1
        self.avg_latency = self.total_latency / self.total_calls
    
    def get_cache_hit_rate(self) -> float:
        """获取缓存命中率"""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


@dataclass
class CachedResult:
    """缓存结果"""
    key: str
    result: Any
    timestamp: float
    ttl: int
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        return time.time() - self.timestamp > self.ttl
    
    def access(self) -> Any:
        """访问缓存结果"""
        self.access_count += 1
        self.last_access = time.time()
        return self.result


class AdaptiveRateLimiter:
    """自适应速率限制器"""
    
    def __init__(self, 
                 base_interval: float = 5.0,
                 min_interval: float = 1.0,
                 max_interval: float = 30.0,
                 window_size: int = 10):
        """
        初始化速率限制器
        
        Args:
            base_interval: 基础调用间隔（秒）
            min_interval: 最小间隔
            max_interval: 最大间隔
            window_size: 滑动窗口大小
        """
        self.base_interval = base_interval
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.window_size = window_size
        
        self.call_history: deque = deque(maxlen=window_size)
        self.last_call_time = 0.0
        self.current_interval = base_interval
        self.lock = threading.Lock()
        
        logger.info(f"AdaptiveRateLimiter initialized: base={base_interval}s, "
                   f"min={min_interval}s, max={max_interval}s")
    
    def acquire(self, priority: CallPriority = CallPriority.MEDIUM) -> float:
        """
        获取调用许可
        
        Args:
            priority: 调用优先级
            
        Returns:
            float: 等待时间（秒）
        """
        with self.lock:
            current_time = time.time()
            
            # 根据优先级调整间隔
            effective_interval = self._get_effective_interval(priority)
            
            # 计算需要等待的时间
            time_since_last_call = current_time - self.last_call_time
            wait_time = max(0, effective_interval - time_since_last_call)
            
            if wait_time > 0:
                logger.debug(f"Rate limit: waiting {wait_time:.2f}s (priority={priority.name})")
                time.sleep(wait_time)
            
            self.last_call_time = time.time()
            self.call_history.append(self.last_call_time)
            
            # 自适应调整间隔
            self._adapt_interval()
            
            return wait_time
    
    def _get_effective_interval(self, priority: CallPriority) -> float:
        """根据优先级获取有效间隔"""
        priority_multipliers = {
            CallPriority.LOW: 1.5,
            CallPriority.MEDIUM: 1.0,
            CallPriority.HIGH: 0.7,
            CallPriority.CRITICAL: 0.3
        }
        
        multiplier = priority_multipliers.get(priority, 1.0)
        return self.current_interval * multiplier
    
    def _adapt_interval(self) -> None:
        """自适应调整调用间隔"""
        if len(self.call_history) < 2:
            return
        
        # 计算最近调用的频率
        recent_calls = list(self.call_history)
        time_span = recent_calls[-1] - recent_calls[0]
        
        if time_span > 0:
            call_rate = len(recent_calls) / time_span  # 调用/秒
            
            # 如果调用频率过高，增加间隔
            if call_rate > 0.5:  # 超过0.5次/秒
                self.current_interval = min(
                    self.current_interval * 1.2,
                    self.max_interval
                )
                logger.debug(f"Increasing interval to {self.current_interval:.2f}s")
            # 如果调用频率较低，减少间隔
            elif call_rate < 0.1:  # 低于0.1次/秒
                self.current_interval = max(
                    self.current_interval * 0.9,
                    self.min_interval
                )
                logger.debug(f"Decreasing interval to {self.current_interval:.2f}s")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.lock:
            if len(self.call_history) < 2:
                call_rate = 0.0
            else:
                recent_calls = list(self.call_history)
                time_span = recent_calls[-1] - recent_calls[0]
                call_rate = len(recent_calls) / time_span if time_span > 0 else 0.0
            
            return {
                'current_interval': self.current_interval,
                'call_rate': call_rate,
                'recent_calls': len(self.call_history),
                'last_call_time': self.last_call_time
            }


class LLMResultCache:
    """LLM结果缓存"""
    
    def __init__(self, 
                 default_ttl: int = 3600,
                 max_size: int = 1000,
                 cleanup_interval: int = 300):
        """
        初始化缓存
        
        Args:
            default_ttl: 默认过期时间（秒）
            max_size: 最大缓存条目数
            cleanup_interval: 清理间隔（秒）
        """
        self.default_ttl = default_ttl
        self.max_size = max_size
        self.cleanup_interval = cleanup_interval
        
        self.cache: Dict[str, CachedResult] = {}
        self.lock = threading.Lock()
        self.last_cleanup = time.time()
        
        logger.info(f"LLMResultCache initialized: ttl={default_ttl}s, max_size={max_size}")
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存结果
        
        Args:
            key: 缓存键
            
        Returns:
            Optional[Any]: 缓存的结果，如果不存在或过期则返回None
        """
        with self.lock:
            if key in self.cache:
                cached = self.cache[key]
                
                if cached.is_expired():
                    logger.debug(f"Cache expired for key: {key[:50]}")
                    del self.cache[key]
                    return None
                
                logger.debug(f"Cache hit for key: {key[:50]}")
                return cached.access()
            
            logger.debug(f"Cache miss for key: {key[:50]}")
            return None
    
    def set(self, key: str, result: Any, ttl: Optional[int] = None) -> None:
        """
        设置缓存结果
        
        Args:
            key: 缓存键
            result: 结果
            ttl: 过期时间（秒），None则使用默认值
        """
        with self.lock:
            # 检查缓存大小
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            ttl = ttl or self.default_ttl
            
            self.cache[key] = CachedResult(
                key=key,
                result=result,
                timestamp=time.time(),
                ttl=ttl
            )
            
            logger.debug(f"Cached result for key: {key[:50]} (ttl={ttl}s)")
            
            # 定期清理
            self._periodic_cleanup()
    
    def _evict_lru(self) -> None:
        """驱逐最少使用的缓存条目"""
        if not self.cache:
            return
        
        # 找到最少访问的条目
        lru_key = min(
            self.cache.keys(),
            key=lambda k: (self.cache[k].access_count, self.cache[k].last_access)
        )
        
        logger.debug(f"Evicting LRU cache entry: {lru_key[:50]}")
        del self.cache[lru_key]
    
    def _periodic_cleanup(self) -> None:
        """定期清理过期缓存"""
        current_time = time.time()
        
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        expired_keys = [
            key for key, cached in self.cache.items()
            if cached.is_expired()
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        self.last_cleanup = current_time
    
    def clear(self) -> None:
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        with self.lock:
            total_accesses = sum(c.access_count for c in self.cache.values())
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'total_accesses': total_accesses,
                'avg_accesses': total_accesses / len(self.cache) if self.cache else 0
            }
    
    @staticmethod
    def generate_key(prompt: str, params: Optional[Dict[str, Any]] = None) -> str:
        """
        生成缓存键
        
        Args:
            prompt: 提示词
            params: 参数
            
        Returns:
            str: 缓存键
        """
        # 组合提示词和参数
        key_data = {
            'prompt': prompt,
            'params': params or {}
        }
        
        # 生成哈希
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()


class AsyncLLMExecutor:
    """异步LLM执行器"""
    
    def __init__(self, max_concurrent: int = 3):
        """
        初始化异步执行器
        
        Args:
            max_concurrent: 最大并发数
        """
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.pending_tasks: List[asyncio.Task] = []
        
        logger.info(f"AsyncLLMExecutor initialized: max_concurrent={max_concurrent}")
    
    async def execute_async(self, 
                           func: Callable,
                           *args,
                           priority: CallPriority = CallPriority.MEDIUM,
                           **kwargs) -> Any:
        """
        异步执行LLM调用
        
        Args:
            func: 要执行的函数
            *args: 位置参数
            priority: 优先级
            **kwargs: 关键字参数
            
        Returns:
            Any: 执行结果
        """
        async with self.semaphore:
            logger.debug(f"Executing async LLM call (priority={priority.name})")
            
            # 在线程池中执行同步函数
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, func, *args, **kwargs)
            
            return result
    
    async def execute_batch(self,
                           calls: List[Tuple[Callable, tuple, dict]],
                           priority: CallPriority = CallPriority.MEDIUM) -> List[Any]:
        """
        批量执行LLM调用
        
        Args:
            calls: 调用列表，每个元素为(func, args, kwargs)
            priority: 优先级
            
        Returns:
            List[Any]: 结果列表
        """
        tasks = [
            self.execute_async(func, *args, priority=priority, **kwargs)
            for func, args, kwargs in calls
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch call {i} failed: {result}")
        
        return results


class LLMCallOptimizer:
    """LLM调用优化器 - 统一管理调用优化"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化调用优化器
        
        Args:
            config: 配置字典
        """
        self.config = config or self._get_default_config()
        
        # 初始化组件
        self.rate_limiter = AdaptiveRateLimiter(
            base_interval=self.config.get('base_interval', 5.0),
            min_interval=self.config.get('min_interval', 1.0),
            max_interval=self.config.get('max_interval', 30.0)
        )
        
        self.cache = LLMResultCache(
            default_ttl=self.config.get('cache_ttl', 3600),
            max_size=self.config.get('cache_max_size', 1000)
        )
        
        self.async_executor = AsyncLLMExecutor(
            max_concurrent=self.config.get('max_concurrent', 3)
        )
        
        self.metrics = CallMetrics()
        
        logger.info("LLMCallOptimizer initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'base_interval': 5.0,
            'min_interval': 1.0,
            'max_interval': 30.0,
            'cache_ttl': 3600,
            'cache_max_size': 1000,
            'max_concurrent': 3,
            'enable_cache': True,
            'enable_rate_limit': True
        }
    
    def call(self,
            func: Callable,
            *args,
            priority: CallPriority = CallPriority.MEDIUM,
            cache_key: Optional[str] = None,
            cache_ttl: Optional[int] = None,
            bypass_cache: bool = False,
            **kwargs) -> Any:
        """
        优化的LLM调用
        
        Args:
            func: 要调用的函数
            *args: 位置参数
            priority: 调用优先级
            cache_key: 缓存键
            cache_ttl: 缓存过期时间
            bypass_cache: 是否绕过缓存
            **kwargs: 关键字参数
            
        Returns:
            Any: 调用结果
        """
        start_time = time.time()
        
        try:
            # 检查缓存
            if self.config.get('enable_cache') and cache_key and not bypass_cache:
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    self.metrics.cache_hits += 1
                    logger.debug("Returning cached result")
                    # 更新指标（即使是缓存命中也要记录）
                    latency = time.time() - start_time
                    self.metrics.update_latency(latency)
                    return cached_result
                self.metrics.cache_misses += 1
            
            # 速率限制
            if self.config.get('enable_rate_limit'):
                wait_time = self.rate_limiter.acquire(priority)
                if wait_time > 0:
                    self.metrics.throttled_calls += 1
            
            # 执行调用
            result = func(*args, **kwargs)
            
            # 缓存结果（只有在不绕过缓存时才缓存）
            if self.config.get('enable_cache') and cache_key and result is not None and not bypass_cache:
                self.cache.set(cache_key, result, cache_ttl)
            
            # 更新指标
            latency = time.time() - start_time
            self.metrics.update_latency(latency)
            
            logger.debug(f"LLM call completed in {latency:.2f}s")
            
            return result
            
        except Exception as e:
            self.metrics.failed_calls += 1
            logger.error(f"LLM call failed: {e}")
            raise
    
    async def call_async(self,
                        func: Callable,
                        *args,
                        priority: CallPriority = CallPriority.MEDIUM,
                        cache_key: Optional[str] = None,
                        cache_ttl: Optional[int] = None,
                        **kwargs) -> Any:
        """
        异步优化的LLM调用
        
        Args:
            func: 要调用的函数
            *args: 位置参数
            priority: 调用优先级
            cache_key: 缓存键
            cache_ttl: 缓存过期时间
            **kwargs: 关键字参数
            
        Returns:
            Any: 调用结果
        """
        # 检查缓存
        if self.config.get('enable_cache') and cache_key:
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                self.metrics.cache_hits += 1
                return cached_result
            self.metrics.cache_misses += 1
        
        # 异步执行
        result = await self.async_executor.execute_async(
            func, *args, priority=priority, **kwargs
        )
        
        # 缓存结果
        if self.config.get('enable_cache') and cache_key and result is not None:
            self.cache.set(cache_key, result, cache_ttl)
        
        return result
    
    def get_metrics(self) -> CallMetrics:
        """获取调用指标"""
        return self.metrics
    
    def get_stats(self) -> Dict[str, Any]:
        """获取完整统计信息"""
        return {
            'metrics': {
                'total_calls': self.metrics.total_calls,
                'cache_hit_rate': self.metrics.get_cache_hit_rate(),
                'avg_latency': self.metrics.avg_latency,
                'throttled_calls': self.metrics.throttled_calls,
                'failed_calls': self.metrics.failed_calls
            },
            'rate_limiter': self.rate_limiter.get_stats(),
            'cache': self.cache.get_stats()
        }
    
    def reset_metrics(self) -> None:
        """重置指标"""
        self.metrics = CallMetrics()
        logger.info("Metrics reset")
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self.cache.clear()
