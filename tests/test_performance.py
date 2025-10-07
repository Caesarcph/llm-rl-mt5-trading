#!/usr/bin/env python3
"""
性能优化测试
测试性能分析器和诊断工具
"""

import unittest
import time
import sys
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.performance_profiler import (
    PerformanceProfiler,
    PerformanceMetrics,
    FunctionProfile,
    get_profiler,
    profile
)
from src.utils.diagnostic_tool import (
    DiagnosticTool,
    DiagnosticResult,
    SystemDiagnostics,
    run_diagnostics
)


class TestPerformanceProfiler(unittest.TestCase):
    """性能分析器测试"""
    
    def setUp(self):
        """测试前准备"""
        self.profiler = PerformanceProfiler(history_size=100)
    
    def test_profiler_creation(self):
        """测试分析器创建"""
        self.assertIsNotNone(self.profiler)
        self.assertEqual(len(self.profiler.metrics_history), 0)
        self.assertEqual(len(self.profiler.function_profiles), 0)
    
    def test_collect_metrics(self):
        """测试收集性能指标"""
        metrics = self.profiler.collect_metrics()
        
        self.assertIsInstance(metrics, PerformanceMetrics)
        self.assertGreater(metrics.cpu_percent, 0)
        self.assertGreater(metrics.memory_mb, 0)
        self.assertGreater(metrics.thread_count, 0)
    
    def test_profile_function_decorator(self):
        """测试函数性能分析装饰器"""
        @self.profiler.profile_function
        def test_func():
            time.sleep(0.01)
            return "test"
        
        # 调用函数
        result = test_func()
        self.assertEqual(result, "test")
        
        # 检查性能记录
        profile = self.profiler.get_function_profile("test_func")
        self.assertIsNotNone(profile)
        self.assertEqual(profile.call_count, 1)
        self.assertGreater(profile.total_time, 0)
    
    def test_multiple_function_calls(self):
        """测试多次函数调用"""
        @self.profiler.profile_function
        def fast_func():
            pass
        
        # 多次调用
        for _ in range(10):
            fast_func()
        
        profile = self.profiler.get_function_profile("fast_func")
        self.assertEqual(profile.call_count, 10)
        self.assertGreater(profile.total_time, 0)
        self.assertGreater(profile.avg_time, 0)
    
    def test_get_top_slow_functions(self):
        """测试获取最慢函数"""
        @self.profiler.profile_function
        def slow_func():
            time.sleep(0.05)
        
        @self.profiler.profile_function
        def fast_func():
            pass
        
        slow_func()
        fast_func()
        
        top_slow = self.profiler.get_top_slow_functions(2)
        self.assertEqual(len(top_slow), 2)
        self.assertEqual(top_slow[0].name, "slow_func")
    
    def test_get_top_called_functions(self):
        """测试获取调用最多的函数"""
        @self.profiler.profile_function
        def frequent_func():
            pass
        
        @self.profiler.profile_function
        def rare_func():
            pass
        
        for _ in range(10):
            frequent_func()
        
        rare_func()
        
        top_called = self.profiler.get_top_called_functions(2)
        self.assertEqual(len(top_called), 2)
        self.assertEqual(top_called[0].name, "frequent_func")
        self.assertEqual(top_called[0].call_count, 10)
    
    def test_get_average_metrics(self):
        """测试获取平均指标"""
        # 收集多个指标
        for _ in range(5):
            self.profiler.collect_metrics()
            time.sleep(0.01)
        
        avg_metrics = self.profiler.get_average_metrics()
        
        self.assertIn('avg_cpu_percent', avg_metrics)
        self.assertIn('avg_memory_mb', avg_metrics)
        self.assertGreater(avg_metrics['avg_cpu_percent'], 0)
        self.assertGreater(avg_metrics['avg_memory_mb'], 0)
    
    def test_generate_report(self):
        """测试生成性能报告"""
        @self.profiler.profile_function
        def test_func():
            time.sleep(0.01)
        
        test_func()
        self.profiler.collect_metrics()
        
        report = self.profiler.generate_report()
        
        self.assertIsInstance(report, str)
        self.assertIn("性能分析报告", report)
        self.assertIn("test_func", report)
    
    def test_reset_profiles(self):
        """测试重置性能分析"""
        @self.profiler.profile_function
        def test_func():
            pass
        
        test_func()
        self.assertEqual(len(self.profiler.function_profiles), 1)
        
        self.profiler.reset_profiles()
        self.assertEqual(len(self.profiler.function_profiles), 0)
    
    def test_optimize_memory(self):
        """测试内存优化"""
        # 添加大量历史记录
        for _ in range(150):
            self.profiler.collect_metrics()
        
        initial_count = len(self.profiler.metrics_history)
        self.profiler.optimize_memory()
        
        # 历史记录应该被清理
        self.assertLessEqual(len(self.profiler.metrics_history), initial_count)


class TestGlobalProfiler(unittest.TestCase):
    """全局性能分析器测试"""
    
    def test_get_profiler(self):
        """测试获取全局分析器"""
        profiler1 = get_profiler()
        profiler2 = get_profiler()
        
        # 应该返回同一个实例
        self.assertIs(profiler1, profiler2)
    
    def test_profile_decorator(self):
        """测试全局装饰器"""
        @profile
        def decorated_func():
            return "result"
        
        result = decorated_func()
        self.assertEqual(result, "result")
        
        # 检查是否被记录
        profiler = get_profiler()
        profile_data = profiler.get_function_profile("decorated_func")
        self.assertIsNotNone(profile_data)


class TestDiagnosticTool(unittest.TestCase):
    """诊断工具测试"""
    
    def setUp(self):
        """测试前准备"""
        self.tool = DiagnosticTool()
    
    def test_tool_creation(self):
        """测试工具创建"""
        self.assertIsNotNone(self.tool)
        self.assertEqual(len(self.tool.results), 0)
    
    def test_add_result(self):
        """测试添加诊断结果"""
        self.tool._add_result("测试类别", "测试项", "pass", "测试消息")
        
        self.assertEqual(len(self.tool.results), 1)
        result = self.tool.results[0]
        
        self.assertEqual(result.category, "测试类别")
        self.assertEqual(result.name, "测试项")
        self.assertEqual(result.status, "pass")
        self.assertEqual(result.message, "测试消息")
    
    def test_check_python_environment(self):
        """测试Python环境检查"""
        self.tool._check_python_environment()
        
        # 应该至少有Python版本检查结果
        self.assertGreater(len(self.tool.results), 0)
        
        # 查找Python版本检查结果
        python_check = next(
            (r for r in self.tool.results if "Python版本" in r.name),
            None
        )
        self.assertIsNotNone(python_check)
    
    def test_check_dependencies(self):
        """测试依赖检查"""
        self.tool._check_dependencies()
        
        # 应该有依赖检查结果
        dep_check = next(
            (r for r in self.tool.results if "依赖" in r.name),
            None
        )
        self.assertIsNotNone(dep_check)
    
    def test_check_file_system(self):
        """测试文件系统检查"""
        self.tool._check_file_system()
        
        # 应该有文件系统检查结果
        self.assertGreater(len(self.tool.results), 0)
    
    def test_generate_report(self):
        """测试生成报告"""
        # 添加一些测试结果
        self.tool._add_result("测试", "项目1", "pass", "通过")
        self.tool._add_result("测试", "项目2", "warning", "警告")
        self.tool._add_result("测试", "项目3", "fail", "失败")
        
        report = self.tool._generate_report()
        
        self.assertIsInstance(report, SystemDiagnostics)
        self.assertEqual(report.summary['pass'], 1)
        self.assertEqual(report.summary['warning'], 1)
        self.assertEqual(report.summary['fail'], 1)
        self.assertEqual(report.overall_status, 'critical')
    
    def test_generate_recommendations(self):
        """测试生成建议"""
        # 添加失败结果
        self.tool._add_result("Python环境", "Python版本", "fail", "版本过低")
        self.tool._add_result("依赖包", "必需依赖", "fail", "缺少依赖")
        
        recommendations = self.tool._generate_recommendations()
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
    
    @patch('sys.stdout')
    def test_print_report(self, mock_stdout):
        """测试打印报告"""
        self.tool._add_result("测试", "项目", "pass", "通过")
        report = self.tool._generate_report()
        
        # 不应该抛出异常
        self.tool.print_report(report)


class TestDiagnosticResult(unittest.TestCase):
    """诊断结果测试"""
    
    def test_result_creation(self):
        """测试结果创建"""
        result = DiagnosticResult(
            category="测试类别",
            name="测试项",
            status="pass",
            message="测试消息"
        )
        
        self.assertEqual(result.category, "测试类别")
        self.assertEqual(result.name, "测试项")
        self.assertEqual(result.status, "pass")
        self.assertEqual(result.message, "测试消息")
        self.assertIsNotNone(result.timestamp)


class TestSystemDiagnostics(unittest.TestCase):
    """系统诊断报告测试"""
    
    def test_diagnostics_creation(self):
        """测试诊断报告创建"""
        from datetime import datetime
        
        results = [
            DiagnosticResult("类别1", "项目1", "pass", "消息1"),
            DiagnosticResult("类别2", "项目2", "warning", "消息2"),
        ]
        
        diagnostics = SystemDiagnostics(
            timestamp=datetime.now(),
            overall_status="warning",
            results=results,
            summary={'pass': 1, 'warning': 1, 'fail': 0},
            recommendations=["建议1", "建议2"]
        )
        
        self.assertEqual(diagnostics.overall_status, "warning")
        self.assertEqual(len(diagnostics.results), 2)
        self.assertEqual(len(diagnostics.recommendations), 2)


class TestPerformanceOptimization(unittest.TestCase):
    """性能优化集成测试"""
    
    def test_profiler_overhead(self):
        """测试分析器开销"""
        profiler = PerformanceProfiler()
        
        # 测试无装饰器的函数
        def plain_func():
            for _ in range(1000):
                pass
        
        start = time.time()
        plain_func()
        plain_time = time.time() - start
        
        # 测试有装饰器的函数
        @profiler.profile_function
        def profiled_func():
            for _ in range(1000):
                pass
        
        start = time.time()
        profiled_func()
        profiled_time = time.time() - start
        
        # 装饰器开销应该很小
        overhead = profiled_time - plain_time
        self.assertLess(overhead, 0.01)  # 小于10ms
    
    def test_memory_efficiency(self):
        """测试内存效率"""
        profiler = PerformanceProfiler(history_size=100)
        
        # 收集大量指标
        for _ in range(200):
            profiler.collect_metrics()
        
        # 历史记录应该被限制
        self.assertLessEqual(len(profiler.metrics_history), 100)


if __name__ == '__main__':
    unittest.main()
