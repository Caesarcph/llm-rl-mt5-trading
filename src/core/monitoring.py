"""
系统监控模块
提供系统健康状态监控、性能指标收集和异常检测功能

注：原文件从第 60 行起内容损坏（字符顺序被打乱），这里按 TradingSystem 的实际用法重写：
    monitoring = MonitoringSystem(config)
    await monitoring.start()
    monitoring.update_metrics({...})
    health = monitoring.check_health()   # health.is_healthy / health.issues
    await monitoring.stop()
"""

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional

import psutil

from .logging import LoggerMixin


class HealthStatus(Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class MetricType(Enum):
    """指标类型枚举"""
    COUNTER = "counter"  # 计数器
    GAUGE = "gauge"      # 仪表
    HISTOGRAM = "histogram"  # 直方图
    TIMER = "timer"      # 计时器


@dataclass
class HealthCheck:
    """单项健康检查结果"""
    component: str
    status: HealthStatus
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)

    def is_healthy(self) -> bool:
        """检查是否健康"""
        return self.status == HealthStatus.HEALTHY

    def is_critical(self) -> bool:
        """检查是否严重"""
        return self.status == HealthStatus.CRITICAL


@dataclass
class PerformanceMetric:
    """性能指标的一个采样点"""
    name: str
    value: float
    metric_type: MetricType = MetricType.GAUGE
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)


# 严重程度排序，用于汇总多项检查
_SEVERITY = {
    HealthStatus.HEALTHY: 0,
    HealthStatus.UNKNOWN: 1,
    HealthStatus.WARNING: 2,
    HealthStatus.CRITICAL: 3,
}


@dataclass
class SystemHealth:
    """整体健康状态（多项检查的汇总）"""
    status: HealthStatus
    checks: List[HealthCheck] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_healthy(self) -> bool:
        return self.status == HealthStatus.HEALTHY

    @property
    def issues(self) -> List[str]:
        return [f"{c.component}: {c.message}" for c in self.checks if not c.is_healthy()]


@dataclass
class MonitoringThresholds:
    """告警阈值，可通过配置对象的同名属性覆盖"""
    cpu_warning: float = 85.0          # CPU 使用率 %
    cpu_critical: float = 95.0
    memory_warning: float = 85.0       # 内存使用率 %
    memory_critical: float = 95.0
    disk_warning: float = 90.0         # 磁盘使用率 %
    disk_critical: float = 97.0
    drawdown_warning: float = 0.10     # 当前回撤（比例）
    drawdown_critical: float = 0.20

    @classmethod
    def from_config(cls, config: Any) -> "MonitoringThresholds":
        defaults = cls()
        if config is None:
            return defaults
        values = {
            name: float(getattr(config, name, getattr(defaults, name)))
            for name in cls.__dataclass_fields__
        }
        return cls(**values)


def _grade(value: float, warning: float, critical: float) -> HealthStatus:
    if value >= critical:
        return HealthStatus.CRITICAL
    if value >= warning:
        return HealthStatus.WARNING
    return HealthStatus.HEALTHY


class MonitoringSystem(LoggerMixin):
    """系统监控：记录指标历史，并汇总资源、连接、回撤和自定义检查的健康状态。"""

    def __init__(self, config: Any = None, history_size: int = 1000, disk_path: str = "."):
        super().__init__()
        self.config = config
        self.thresholds = MonitoringThresholds.from_config(config)
        self.history_size = int(getattr(config, "history_size", history_size))
        self.disk_path = getattr(config, "disk_path", disk_path)

        self._metrics: Dict[str, Deque[PerformanceMetric]] = {}
        self._custom_checks: Dict[str, Callable[[], HealthCheck]] = {}
        self._lock = threading.Lock()
        self._running = False
        self.last_health: Optional[SystemHealth] = None

    # ------------------------------------------------------------------ 生命周期

    async def start(self) -> None:
        self._running = True
        psutil.cpu_percent(interval=None)  # 预热：第一次调用总是返回 0
        self.logger.info("监控系统已启动")

    async def stop(self) -> None:
        self._running = False
        self.logger.info("监控系统已停止")

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------ 指标

    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        **tags: str,
    ) -> None:
        metric = PerformanceMetric(name=name, value=float(value), metric_type=metric_type, tags=tags)
        with self._lock:
            history = self._metrics.get(name)
            if history is None:
                history = deque(maxlen=self.history_size)
                self._metrics[name] = history
            history.append(metric)

    def update_metrics(self, metrics: Mapping[str, Any], prefix: str = "") -> None:
        """批量记录指标。数值、布尔、timedelta 会被记录；字符串、时间戳等非数值直接跳过；
        嵌套字典按 `父键.子键` 展开。"""
        for key, value in metrics.items():
            name = f"{prefix}{key}"
            if isinstance(value, Mapping):
                self.update_metrics(value, prefix=f"{name}.")
            elif isinstance(value, bool):
                self.record_metric(name, int(value))
            elif isinstance(value, (int, float)):
                self.record_metric(name, value)
            elif isinstance(value, timedelta):
                self.record_metric(name, value.total_seconds(), MetricType.TIMER)

    def get_latest(self, name: str) -> Optional[PerformanceMetric]:
        with self._lock:
            history = self._metrics.get(name)
            return history[-1] if history else None

    def get_history(self, name: str, since: Optional[timedelta] = None) -> List[PerformanceMetric]:
        with self._lock:
            history = list(self._metrics.get(name, ()))
        if since is None:
            return history
        cutoff = datetime.now() - since
        return [m for m in history if m.timestamp >= cutoff]

    def metric_names(self) -> List[str]:
        with self._lock:
            return sorted(self._metrics)

    # ------------------------------------------------------------------ 健康检查

    def register_health_check(self, name: str, check: Callable[[], HealthCheck]) -> None:
        self._custom_checks[name] = check

    def unregister_health_check(self, name: str) -> None:
        self._custom_checks.pop(name, None)

    def check_health(self) -> SystemHealth:
        checks = [self._check_resources(), *self._check_trading_metrics()]
        for name, check in list(self._custom_checks.items()):
            try:
                checks.append(check())
            except Exception as exc:  # 自定义检查本身出错，不能拖垮整个监控
                checks.append(HealthCheck(name, HealthStatus.WARNING, f"检查执行失败: {exc}"))

        status = max((c.status for c in checks), key=_SEVERITY.__getitem__, default=HealthStatus.UNKNOWN)
        health = SystemHealth(status=status, checks=checks)
        self.last_health = health
        return health

    def _check_resources(self) -> HealthCheck:
        t = self.thresholds
        try:
            cpu = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory().percent
            disk = psutil.disk_usage(self.disk_path).percent
        except Exception as exc:
            return HealthCheck("resources", HealthStatus.UNKNOWN, f"无法读取系统资源: {exc}")

        self.record_metric("system.cpu_percent", cpu)
        self.record_metric("system.memory_percent", memory)
        self.record_metric("system.disk_percent", disk)

        grades = {
            "CPU": (cpu, _grade(cpu, t.cpu_warning, t.cpu_critical)),
            "内存": (memory, _grade(memory, t.memory_warning, t.memory_critical)),
            "磁盘": (disk, _grade(disk, t.disk_warning, t.disk_critical)),
        }
        worst = max((g for _, g in grades.values()), key=_SEVERITY.__getitem__)
        problems = [f"{label} {value:.1f}%" for label, (value, g) in grades.items() if g != HealthStatus.HEALTHY]
        message = "资源正常" if not problems else "资源使用率偏高: " + ", ".join(problems)
        details = {"cpu_percent": cpu, "memory_percent": memory, "disk_percent": disk}
        return HealthCheck("resources", worst, message, details=details)

    def _check_trading_metrics(self) -> List[HealthCheck]:
        checks: List[HealthCheck] = []

        connected = self.get_latest("mt5_connected")
        if connected is not None:
            if connected.value:
                checks.append(HealthCheck("mt5", HealthStatus.HEALTHY, "MT5 已连接"))
            else:
                checks.append(HealthCheck("mt5", HealthStatus.CRITICAL, "MT5 未连接"))

        drawdown = self.get_latest("current_drawdown")
        if drawdown is not None:
            t = self.thresholds
            status = _grade(abs(drawdown.value), t.drawdown_warning, t.drawdown_critical)
            message = f"当前回撤 {abs(drawdown.value):.2%}"
            checks.append(HealthCheck("drawdown", status, message, details={"current_drawdown": drawdown.value}))

        return checks

    def get_status(self) -> Dict[str, Any]:
        health = self.last_health
        return {
            "running": self._running,
            "metrics_tracked": len(self.metric_names()),
            "health": health.status.value if health else HealthStatus.UNKNOWN.value,
            "issues": health.issues if health else [],
        }
