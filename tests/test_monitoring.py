"""监控模块测试（src/core/monitoring.py 重写后补充）"""

import asyncio
import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

from src.core.monitoring import (
    HealthCheck,
    HealthStatus,
    MetricType,
    MonitoringSystem,
    MonitoringThresholds,
)


def _patch_resources(cpu=10.0, memory=20.0, disk=30.0):
    """让资源检查结果可控，默认 CPU 10%、内存 20%、磁盘 30%"""
    return (
        patch("src.core.monitoring.psutil.cpu_percent", return_value=cpu),
        patch("src.core.monitoring.psutil.virtual_memory", return_value=SimpleNamespace(percent=memory)),
        patch("src.core.monitoring.psutil.disk_usage", return_value=SimpleNamespace(percent=disk)),
    )


class TestMonitoringSystem(unittest.TestCase):
    def setUp(self):
        self.monitor = MonitoringSystem()
        self.patches = _patch_resources()
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def test_start_and_stop(self):
        asyncio.run(self.monitor.start())
        self.assertTrue(self.monitor.is_running)
        asyncio.run(self.monitor.stop())
        self.assertFalse(self.monitor.is_running)

    def test_update_metrics_handles_mixed_types(self):
        self.monitor.update_metrics({
            "timestamp": datetime.now(),       # 跳过
            "state": "running",                # 跳过
            "uptime": timedelta(minutes=2),    # 记为秒
            "total_trades": 7,
            "success_rate": 0.5,
            "mt5_connected": True,             # 记为 1
            "nested": {"a": 1},                # 展开成 nested.a
        })
        self.assertEqual(
            self.monitor.metric_names(),
            ["mt5_connected", "nested.a", "success_rate", "total_trades", "uptime"],
        )
        uptime = self.monitor.get_latest("uptime")
        self.assertEqual(uptime.value, 120.0)
        self.assertEqual(uptime.metric_type, MetricType.TIMER)
        self.assertEqual(self.monitor.get_latest("mt5_connected").value, 1.0)

    def test_history_is_bounded(self):
        monitor = MonitoringSystem(history_size=3)
        for i in range(10):
            monitor.record_metric("x", i)
        self.assertEqual([m.value for m in monitor.get_history("x")], [7.0, 8.0, 9.0])

    def test_healthy_when_everything_normal(self):
        self.monitor.update_metrics({"mt5_connected": True, "current_drawdown": 0.02})
        health = self.monitor.check_health()
        self.assertTrue(health.is_healthy)
        self.assertEqual(health.issues, [])

    def test_mt5_disconnected_is_critical(self):
        self.monitor.update_metrics({"mt5_connected": False})
        health = self.monitor.check_health()
        self.assertFalse(health.is_healthy)
        self.assertEqual(health.status, HealthStatus.CRITICAL)
        self.assertIn("mt5: MT5 未连接", health.issues)

    def test_drawdown_thresholds(self):
        self.monitor.update_metrics({"current_drawdown": 0.12})
        self.assertEqual(self.monitor.check_health().status, HealthStatus.WARNING)

        self.monitor.update_metrics({"current_drawdown": 0.25})
        self.assertEqual(self.monitor.check_health().status, HealthStatus.CRITICAL)

    def test_failing_custom_check_degrades_to_warning(self):
        def broken() -> HealthCheck:
            raise RuntimeError("boom")

        self.monitor.register_health_check("broken", broken)
        health = self.monitor.check_health()
        self.assertEqual(health.status, HealthStatus.WARNING)
        self.assertTrue(any("boom" in issue for issue in health.issues))

    def test_custom_critical_check_wins(self):
        self.monitor.register_health_check(
            "db", lambda: HealthCheck("db", HealthStatus.CRITICAL, "数据库不可用")
        )
        self.assertEqual(self.monitor.check_health().status, HealthStatus.CRITICAL)


class TestResourceThresholds(unittest.TestCase):
    def test_high_cpu_is_reported(self):
        patches = _patch_resources(cpu=97.0)
        for p in patches:
            p.start()
        try:
            health = MonitoringSystem().check_health()
        finally:
            for p in patches:
                p.stop()
        self.assertEqual(health.status, HealthStatus.CRITICAL)
        self.assertTrue(any("CPU 97.0%" in issue for issue in health.issues))

    def test_thresholds_read_from_config(self):
        config = SimpleNamespace(cpu_warning=50.0, history_size=5)
        monitor = MonitoringSystem(config)
        self.assertEqual(monitor.thresholds.cpu_warning, 50.0)
        self.assertEqual(monitor.thresholds.cpu_critical, MonitoringThresholds().cpu_critical)
        self.assertEqual(monitor.history_size, 5)


if __name__ == "__main__":
    unittest.main()
