#!/usr/bin/env python3
"""
部署测试
测试环境检测、配置验证和依赖检查
"""

import unittest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.environment import (
    EnvironmentDetector,
    SystemInfo,
    DependencyCheck,
    EnvironmentStatus,
    check_environment
)


class TestEnvironmentDetector(unittest.TestCase):
    """环境检测器测试"""
    
    def setUp(self):
        """测试前准备"""
        self.detector = EnvironmentDetector()
    
    def test_detector_creation(self):
        """测试检测器创建"""
        self.assertIsNotNone(self.detector)
        self.assertIsNotNone(self.detector.REQUIRED_PACKAGES)
        self.assertIsNotNone(self.detector.OPTIONAL_PACKAGES)
    
    def test_system_info_detection(self):
        """测试系统信息检测"""
        system_info = self.detector._detect_system_info()
        
        self.assertIsInstance(system_info, SystemInfo)
        self.assertIsNotNone(system_info.os_name)
        self.assertIsNotNone(system_info.python_version)
        self.assertGreater(system_info.cpu_count, 0)
        self.assertGreater(system_info.memory_gb, 0)
    
    def test_check_system_requirements(self):
        """测试系统要求检查"""
        system_info = SystemInfo(
            os_name="Windows",
            os_version="10.0.19041",
            python_version="3.10.0",
            cpu_count=4,
            memory_gb=8.0,
            architecture="AMD64"
        )
        
        warnings, errors = self.detector._check_system_requirements(system_info)
        
        self.assertIsInstance(warnings, list)
        self.assertIsInstance(errors, list)
    
    def test_check_system_requirements_low_memory(self):
        """测试低内存警告"""
        system_info = SystemInfo(
            os_name="Windows",
            os_version="10.0.19041",
            python_version="3.10.0",
            cpu_count=4,
            memory_gb=2.0,  # 低于最低要求
            architecture="AMD64"
        )
        
        warnings, errors = self.detector._check_system_requirements(system_info)
        
        # 应该有内存不足的警告
        self.assertTrue(any("内存" in w for w in warnings))
    
    def test_check_system_requirements_old_python(self):
        """测试旧Python版本错误"""
        with patch('sys.version_info', (3, 8, 0)):
            system_info = SystemInfo(
                os_name="Windows",
                os_version="10.0.19041",
                python_version="3.8.0",
                cpu_count=4,
                memory_gb=8.0,
                architecture="AMD64"
            )
            
            warnings, errors = self.detector._check_system_requirements(system_info)
            
            # 应该有Python版本错误
            self.assertTrue(any("Python" in e for e in errors))
    
    def test_check_package_installed(self):
        """测试已安装包检查"""
        # 检查sys包（肯定已安装）
        check = self.detector._check_package('sys', '0.0.0', required=True)
        
        self.assertEqual(check.name, 'sys')
        self.assertTrue(check.required)
        self.assertTrue(check.installed)
    
    def test_check_package_not_installed(self):
        """测试未安装包检查"""
        # 检查不存在的包
        check = self.detector._check_package('nonexistent_package_xyz', '1.0.0', required=False)
        
        self.assertEqual(check.name, 'nonexistent_package_xyz')
        self.assertFalse(check.required)
        self.assertFalse(check.installed)
        self.assertIsNotNone(check.error)
    
    def test_check_dependencies(self):
        """测试依赖检查"""
        dependencies = self.detector._check_dependencies()
        
        self.assertIsInstance(dependencies, list)
        self.assertGreater(len(dependencies), 0)
        
        # 验证每个依赖检查结果
        for dep in dependencies:
            self.assertIsInstance(dep, DependencyCheck)
            self.assertIsNotNone(dep.name)
            self.assertIsInstance(dep.required, bool)
            self.assertIsInstance(dep.installed, bool)
    
    @patch('src.core.environment.EnvironmentDetector._detect_system_info')
    @patch('src.core.environment.EnvironmentDetector._check_dependencies')
    def test_detect_environment_success(self, mock_deps, mock_sysinfo):
        """测试环境检测成功"""
        # 模拟系统信息
        mock_sysinfo.return_value = SystemInfo(
            os_name="Windows",
            os_version="10.0.19041",
            python_version="3.10.0",
            cpu_count=4,
            memory_gb=8.0,
            architecture="AMD64"
        )
        
        # 模拟所有依赖已安装
        mock_deps.return_value = [
            DependencyCheck(
                name="pandas",
                required=True,
                installed=True,
                version="1.5.0",
                min_version="1.3.0"
            ),
            DependencyCheck(
                name="numpy",
                required=True,
                installed=True,
                version="1.23.0",
                min_version="1.20.0"
            )
        ]
        
        status = self.detector.detect_environment()
        
        self.assertIsInstance(status, EnvironmentStatus)
        self.assertTrue(status.is_valid)
        self.assertEqual(len(status.errors), 0)
    
    @patch('src.core.environment.EnvironmentDetector._detect_system_info')
    @patch('src.core.environment.EnvironmentDetector._check_dependencies')
    def test_detect_environment_missing_required(self, mock_deps, mock_sysinfo):
        """测试缺少必需依赖"""
        # 模拟系统信息
        mock_sysinfo.return_value = SystemInfo(
            os_name="Windows",
            os_version="10.0.19041",
            python_version="3.10.0",
            cpu_count=4,
            memory_gb=8.0,
            architecture="AMD64"
        )
        
        # 模拟缺少必需依赖
        mock_deps.return_value = [
            DependencyCheck(
                name="pandas",
                required=True,
                installed=False,
                min_version="1.3.0",
                error="No module named 'pandas'"
            )
        ]
        
        status = self.detector.detect_environment()
        
        self.assertIsInstance(status, EnvironmentStatus)
        self.assertFalse(status.is_valid)
        self.assertGreater(len(status.errors), 0)
        self.assertTrue(any("pandas" in e for e in status.errors))
    
    def test_generate_requirements_file(self):
        """测试生成requirements文件"""
        import tempfile
        import os
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            temp_path = f.name
        
        try:
            self.detector.generate_requirements_file(temp_path)
            
            # 验证文件已创建
            self.assertTrue(Path(temp_path).exists())
            
            # 验证文件内容
            with open(temp_path, 'r') as f:
                content = f.read()
            
            # 应该包含必需包
            for package in self.detector.REQUIRED_PACKAGES:
                self.assertIn(package, content)
            
        finally:
            # 清理临时文件
            if Path(temp_path).exists():
                os.unlink(temp_path)


class TestSystemInfo(unittest.TestCase):
    """系统信息测试"""
    
    def test_system_info_creation(self):
        """测试系统信息创建"""
        info = SystemInfo(
            os_name="Windows",
            os_version="10.0.19041",
            python_version="3.10.0",
            cpu_count=4,
            memory_gb=8.0,
            architecture="AMD64"
        )
        
        self.assertEqual(info.os_name, "Windows")
        self.assertEqual(info.python_version, "3.10.0")
        self.assertEqual(info.cpu_count, 4)
        self.assertEqual(info.memory_gb, 8.0)


class TestDependencyCheck(unittest.TestCase):
    """依赖检查测试"""
    
    def test_dependency_check_installed(self):
        """测试已安装依赖"""
        check = DependencyCheck(
            name="pandas",
            required=True,
            installed=True,
            version="1.5.0",
            min_version="1.3.0"
        )
        
        self.assertEqual(check.name, "pandas")
        self.assertTrue(check.required)
        self.assertTrue(check.installed)
        self.assertEqual(check.version, "1.5.0")
    
    def test_dependency_check_not_installed(self):
        """测试未安装依赖"""
        check = DependencyCheck(
            name="some_package",
            required=False,
            installed=False,
            min_version="1.0.0",
            error="Module not found"
        )
        
        self.assertEqual(check.name, "some_package")
        self.assertFalse(check.required)
        self.assertFalse(check.installed)
        self.assertIsNotNone(check.error)


class TestEnvironmentStatus(unittest.TestCase):
    """环境状态测试"""
    
    def test_environment_status_valid(self):
        """测试有效环境状态"""
        system_info = SystemInfo(
            os_name="Windows",
            os_version="10.0.19041",
            python_version="3.10.0",
            cpu_count=4,
            memory_gb=8.0,
            architecture="AMD64"
        )
        
        status = EnvironmentStatus(
            is_valid=True,
            system_info=system_info,
            dependencies=[],
            warnings=[],
            errors=[]
        )
        
        self.assertTrue(status.is_valid)
        self.assertEqual(len(status.errors), 0)
    
    def test_environment_status_invalid(self):
        """测试无效环境状态"""
        system_info = SystemInfo(
            os_name="Windows",
            os_version="10.0.19041",
            python_version="3.8.0",
            cpu_count=2,
            memory_gb=2.0,
            architecture="AMD64"
        )
        
        status = EnvironmentStatus(
            is_valid=False,
            system_info=system_info,
            dependencies=[],
            warnings=["内存不足"],
            errors=["Python版本过低"]
        )
        
        self.assertFalse(status.is_valid)
        self.assertGreater(len(status.errors), 0)
        self.assertGreater(len(status.warnings), 0)


class TestCheckEnvironmentFunction(unittest.TestCase):
    """测试便捷函数"""
    
    @patch('src.core.environment.EnvironmentDetector.detect_environment')
    def test_check_environment_function(self, mock_detect):
        """测试check_environment函数"""
        mock_status = EnvironmentStatus(
            is_valid=True,
            system_info=Mock(),
            dependencies=[],
            warnings=[],
            errors=[]
        )
        mock_detect.return_value = mock_status
        
        status = check_environment()
        
        self.assertIsInstance(status, EnvironmentStatus)
        mock_detect.assert_called_once()


if __name__ == '__main__':
    unittest.main()
