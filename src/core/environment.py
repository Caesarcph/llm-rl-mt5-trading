#!/usr/bin/env python3
"""
环境检测和依赖验证模块
检测系统环境、验证依赖包和配置
"""

import sys
import os
import platform
import subprocess
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

from src.core.logging import get_logger


@dataclass
class SystemInfo:
    """系统信息"""
    os_name: str
    os_version: str
    python_version: str
    cpu_count: int
    memory_gb: float
    architecture: str


@dataclass
class DependencyCheck:
    """依赖检查结果"""
    name: str
    required: bool
    installed: bool
    version: Optional[str] = None
    min_version: Optional[str] = None
    error: Optional[str] = None


@dataclass
class EnvironmentStatus:
    """环境状态"""
    is_valid: bool
    system_info: SystemInfo
    dependencies: List[DependencyCheck]
    warnings: List[str]
    errors: List[str]


class EnvironmentDetector:
    """环境检测器"""
    
    # 必需的依赖包
    REQUIRED_PACKAGES = {
        'pandas': '1.3.0',
        'numpy': '1.20.0',
        'MetaTrader5': '5.0.0',
        'pyyaml': '5.0.0',
        'redis': '3.5.0',
    }
    
    # 可选的依赖包
    OPTIONAL_PACKAGES = {
        'torch': '1.10.0',
        'transformers': '4.20.0',
        'stable-baselines3': '1.5.0',
        'gymnasium': '0.26.0',
        'ta-lib': '0.4.0',
        'optuna': '2.10.0',
    }
    
    # 最低系统要求
    MIN_PYTHON_VERSION = (3, 9)
    MIN_MEMORY_GB = 4
    MIN_CPU_CORES = 2
    
    def __init__(self):
        """初始化环境检测器"""
        self.logger = get_logger()
    
    def detect_environment(self) -> EnvironmentStatus:
        """
        检测完整环境
        
        Returns:
            环境状态
        """
        self.logger.info("开始环境检测...")
        
        warnings = []
        errors = []
        
        # 检测系统信息
        system_info = self._detect_system_info()
        self.logger.info(f"系统: {system_info.os_name} {system_info.os_version}")
        self.logger.info(f"Python: {system_info.python_version}")
        self.logger.info(f"CPU: {system_info.cpu_count} 核")
        self.logger.info(f"内存: {system_info.memory_gb:.1f} GB")
        
        # 检查系统要求
        sys_warnings, sys_errors = self._check_system_requirements(system_info)
        warnings.extend(sys_warnings)
        errors.extend(sys_errors)
        
        # 检查依赖包
        dependencies = self._check_dependencies()
        
        # 统计依赖问题
        missing_required = [
            dep for dep in dependencies
            if dep.required and not dep.installed
        ]
        
        if missing_required:
            errors.append(
                f"缺少必需依赖: {', '.join(dep.name for dep in missing_required)}"
            )
        
        missing_optional = [
            dep for dep in dependencies
            if not dep.required and not dep.installed
        ]
        
        if missing_optional:
            warnings.append(
                f"缺少可选依赖: {', '.join(dep.name for dep in missing_optional)}"
            )
        
        is_valid = len(errors) == 0
        
        status = EnvironmentStatus(
            is_valid=is_valid,
            system_info=system_info,
            dependencies=dependencies,
            warnings=warnings,
            errors=errors
        )
        
        self._log_status(status)
        
        return status
    
    def _detect_system_info(self) -> SystemInfo:
        """检测系统信息"""
        import psutil
        
        return SystemInfo(
            os_name=platform.system(),
            os_version=platform.version(),
            python_version=platform.python_version(),
            cpu_count=os.cpu_count() or 1,
            memory_gb=psutil.virtual_memory().total / (1024 ** 3),
            architecture=platform.machine()
        )
    
    def _check_system_requirements(self, system_info: SystemInfo) -> Tuple[List[str], List[str]]:
        """检查系统要求"""
        warnings = []
        errors = []
        
        # 检查Python版本
        current_version = sys.version_info[:2]
        if current_version < self.MIN_PYTHON_VERSION:
            errors.append(
                f"Python版本过低: {system_info.python_version}, "
                f"需要 {'.'.join(map(str, self.MIN_PYTHON_VERSION))}+"
            )
        
        # 检查内存
        if system_info.memory_gb < self.MIN_MEMORY_GB:
            warnings.append(
                f"内存不足: {system_info.memory_gb:.1f}GB, "
                f"建议至少 {self.MIN_MEMORY_GB}GB"
            )
        
        # 检查CPU
        if system_info.cpu_count < self.MIN_CPU_CORES:
            warnings.append(
                f"CPU核心数较少: {system_info.cpu_count}, "
                f"建议至少 {self.MIN_CPU_CORES} 核"
            )
        
        # 检查操作系统
        if system_info.os_name not in ['Windows', 'Linux', 'Darwin']:
            warnings.append(f"未测试的操作系统: {system_info.os_name}")
        
        return warnings, errors
    
    def _check_dependencies(self) -> List[DependencyCheck]:
        """检查所有依赖包"""
        results = []
        
        # 检查必需包
        for package, min_version in self.REQUIRED_PACKAGES.items():
            check = self._check_package(package, min_version, required=True)
            results.append(check)
        
        # 检查可选包
        for package, min_version in self.OPTIONAL_PACKAGES.items():
            check = self._check_package(package, min_version, required=False)
            results.append(check)
        
        return results
    
    def _check_package(self, package_name: str, min_version: str, required: bool) -> DependencyCheck:
        """检查单个包"""
        try:
            module = __import__(package_name)
            version = getattr(module, '__version__', 'unknown')
            
            return DependencyCheck(
                name=package_name,
                required=required,
                installed=True,
                version=version,
                min_version=min_version
            )
        except ImportError as e:
            return DependencyCheck(
                name=package_name,
                required=required,
                installed=False,
                min_version=min_version,
                error=str(e)
            )
    
    def _log_status(self, status: EnvironmentStatus):
        """记录环境状态"""
        if status.is_valid:
            self.logger.info("✓ 环境检测通过")
        else:
            self.logger.error("✗ 环境检测失败")
        
        if status.warnings:
            self.logger.warning("警告:")
            for warning in status.warnings:
                self.logger.warning(f"  - {warning}")
        
        if status.errors:
            self.logger.error("错误:")
            for error in status.errors:
                self.logger.error(f"  - {error}")
        
        # 记录依赖状态
        installed_count = sum(1 for dep in status.dependencies if dep.installed)
        total_count = len(status.dependencies)
        self.logger.info(f"依赖包: {installed_count}/{total_count} 已安装")
    
    def generate_requirements_file(self, output_path: str = "requirements.txt"):
        """生成requirements.txt文件"""
        requirements = []
        
        for package, version in self.REQUIRED_PACKAGES.items():
            requirements.append(f"{package}>={version}")
        
        requirements.append("")
        requirements.append("# Optional dependencies")
        
        for package, version in self.OPTIONAL_PACKAGES.items():
            requirements.append(f"# {package}>={version}")
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(requirements))
        
        self.logger.info(f"已生成 {output_path}")


def check_environment() -> EnvironmentStatus:
    """
    便捷函数：检查环境
    
    Returns:
        环境状态
    """
    detector = EnvironmentDetector()
    return detector.detect_environment()
