#!/usr/bin/env python3
"""
性能分析器
监控和优化系统性能，减少延迟和内存使用
"""

import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from functools import wraps
from collections import deque

from src.core.logging import get_logger


@dataclass
class PerformanceMetrics:
    """性能指标"""
    timestamp: datetime
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    thread_count: int
    function_calls: Dict[str, int] = field(default_factory=dict)
    function_times: Dict[str, float] = field(default_factory=dict)
    slow_functions: List[str] = field(default_factory=list)


@dataclass
class FunctionProfile:
    """函数性能分析"""
    name: str
    call_count: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    last_call_time: Optional[datetime] = None


class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self, history_size: int = 1000):
        """
        初始化性能分析器
        
        Args:
            history_size: 历史记录大小
        """
        self.logger = get_logger()
        self.history_size = history_size
        
        # 性能历史
        self.metrics_history: deque = deque(maxlen=history_size)
        
        # 函数性能分析
        self.function_profiles: Dict[str, FunctionProfile] = {}
        self._profile_lock = threading.Lock()
        
        # 进程信息
        self.process = psutil.Process()
        
        # 监控配置
        self.slow_function_threshold = 1.0  # 秒
        self.high_memory_threshold = 80  # 百分比
        self.high_cpu_threshold = 80  # 百分比
        
        self.logger.info("性能分析器初始化完成")
    
    def collect_metrics(self) -> PerformanceMetrics:
        """
        收集当前性能指标
        
        Returns:
            性能指标
        """
        try:
            # CPU和内存使用
            cpu_percent = self.process.cpu_percent(interval=0.1)
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            memory_percent = self.process.memory_percent()
            
            # 线程数
            thread_count = self.process.num_threads()
            
            # 函数调用统计
            function_calls = {}
            function_times = {}
            slow_functions = []
            
            with self._profile_lock:
                for name, profile in self.function_profiles.items():
                    function_calls[name] = profile.call_count
                    function_times[name] = profile.avg_time
                    
                    if profile.avg_time > self.slow_function_threshold:
                        slow_functions.append(name)
            
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
                memory_percent=memory_percent,
                thread_count=thread_count,
                function_calls=function_calls,
                function_times=function_times,
                slow_functions=slow_functions
            )
            
            # 添加到历史
            self.metrics_history.append(metrics)
            
            # 检查警告
            self._check_warnings(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"收集性能指标失败: {e}")
            return None
    
    def _check_warnings(self, metrics: PerformanceMetrics):
        """检查性能警告"""
        if metrics.cpu_percent > self.high_cpu_threshold:
            self.logger.warning(f"CPU使用率过高: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > self.high_memory_threshold:
            self.logger.warning(f"内存使用率过高: {metrics.memory_percent:.1f}%")
        
        if metrics.slow_functions:
            self.logger.warning(f"检测到慢函数: {', '.join(metrics.slow_functions)}")
    
    def profile_function(self, func: Callable) -> Callable:
        """
        函数性能分析装饰器
        
        Args:
            func: 要分析的函数
            
        Returns:
            装饰后的函数
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed_time = time.time() - start_time
                self._record_function_call(func.__name__, elapsed_time)
        
        return wrapper
    
    def _record_function_call(self, func_name: str, elapsed_time: float):
        """记录函数调用"""
        with self._profile_lock:
            if func_name not in self.function_profiles:
                self.function_profiles[func_name] = FunctionProfile(name=func_name)
            
            profile = self.function_profiles[func_name]
            profile.call_count += 1
            profile.total_time += elapsed_time
            profile.avg_time = profile.total_time / profile.call_count
            profile.min_time = min(profile.min_time, elapsed_time)
            profile.max_time = max(profile.max_time, elapsed_time)
            profile.last_call_time = datetime.now()
    
    def get_function_profile(self, func_name: str) -> Optional[FunctionProfile]:
        """获取函数性能分析"""
        with self._profile_lock:
            return self.function_profiles.get(func_name)
    
    def get_all_profiles(self) -> Dict[str, FunctionProfile]:
        """获取所有函数性能分析"""
        with self._profile_lock:
            return self.function_profiles.copy()
    
    def get_top_slow_functions(self, n: int = 10) -> List[FunctionProfile]:
        """获取最慢的N个函数"""
        with self._profile_lock:
            profiles = list(self.function_profiles.values())
            profiles.sort(key=lambda p: p.avg_time, reverse=True)
            return profiles[:n]
    
    def get_top_called_functions(self, n: int = 10) -> List[FunctionProfile]:
        """获取调用次数最多的N个函数"""
        with self._profile_lock:
            profiles = list(self.function_profiles.values())
            profiles.sort(key=lambda p: p.call_count, reverse=True)
            return profiles[:n]
    
    def get_average_metrics(self, duration: timedelta = timedelta(minutes=5)) -> Dict[str, float]:
        """
        获取平均性能指标
        
        Args:
            duration: 时间范围
            
        Returns:
            平均指标
        """
        cutoff_time = datetime.now() - duration
        recent_metrics = [
            m for m in self.metrics_history
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        return {
            'avg_cpu_percent': sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
            'avg_memory_mb': sum(m.memory_mb for m in recent_metrics) / len(recent_metrics),
            'avg_memory_percent': sum(m.memory_percent for m in recent_metrics) / len(recent_metrics),
            'avg_thread_count': sum(m.thread_count for m in recent_metrics) / len(recent_metrics),
        }
    
    def generate_report(self) -> str:
        """生成性能报告"""
        report_lines = [
            "=" * 60,
            "性能分析报告",
            "=" * 60,
            ""
        ]
        
        # 当前指标
        if self.metrics_history:
            latest = self.metrics_history[-1]
            report_lines.extend([
                "当前性能指标:",
                f"  CPU使用率: {latest.cpu_percent:.1f}%",
                f"  内存使用: {latest.memory_mb:.1f} MB ({latest.memory_percent:.1f}%)",
                f"  线程数: {latest.thread_count}",
                ""
            ])
        
        # 平均指标
        avg_metrics = self.get_average_metrics()
        if avg_metrics:
            report_lines.extend([
                "5分钟平均指标:",
                f"  平均CPU: {avg_metrics['avg_cpu_percent']:.1f}%",
                f"  平均内存: {avg_metrics['avg_memory_mb']:.1f} MB",
                ""
            ])
        
        # 最慢函数
        slow_functions = self.get_top_slow_functions(5)
        if slow_functions:
            report_lines.extend([
                "最慢的5个函数:",
            ])
            for profile in slow_functions:
                report_lines.append(
                    f"  {profile.name}: {profile.avg_time:.3f}s "
                    f"(调用{profile.call_count}次)"
                )
            report_lines.append("")
        
        # 调用最多的函数
        called_functions = self.get_top_called_functions(5)
        if called_functions:
            report_lines.extend([
                "调用最多的5个函数:",
            ])
            for profile in called_functions:
                report_lines.append(
                    f"  {profile.name}: {profile.call_count}次 "
                    f"(平均{profile.avg_time:.3f}s)"
                )
            report_lines.append("")
        
        report_lines.append("=" * 60)
        
        return '\n'.join(report_lines)
    
    def reset_profiles(self):
        """重置所有性能分析"""
        with self._profile_lock:
            self.function_profiles.clear()
        self.logger.info("性能分析已重置")
    
    def optimize_memory(self):
        """优化内存使用"""
        import gc
        
        self.logger.info("开始内存优化...")
        
        # 强制垃圾回收
        collected = gc.collect()
        self.logger.info(f"垃圾回收: 清理了{collected}个对象")
        
        # 清理旧的历史记录
        if len(self.metrics_history) > self.history_size // 2:
            old_size = len(self.metrics_history)
            self.metrics_history = deque(
                list(self.metrics_history)[-(self.history_size // 2):],
                maxlen=self.history_size
            )
            self.logger.info(f"清理历史记录: {old_size} -> {len(self.metrics_history)}")


# 全局性能分析器实例
_profiler: Optional[PerformanceProfiler] = None


def get_profiler() -> PerformanceProfiler:
    """获取全局性能分析器"""
    global _profiler
    if _profiler is None:
        _profiler = PerformanceProfiler()
    return _profiler


def profile(func: Callable) -> Callable:
    """性能分析装饰器"""
    profiler = get_profiler()
    return profiler.profile_function(func)
