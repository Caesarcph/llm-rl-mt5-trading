#!/usr/bin/env python3
"""
系统诊断工具
提供系统健康检查、故障排除和诊断功能
"""

import sys
import os
import platform
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from src.core.logging import get_logger


@dataclass
class DiagnosticResult:
    """诊断结果"""
    category: str
    name: str
    status: str  # "pass", "warning", "fail"
    message: str
    details: Optional[Dict] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SystemDiagnostics:
    """系统诊断报告"""
    timestamp: datetime
    overall_status: str  # "healthy", "warning", "critical"
    results: List[DiagnosticResult]
    summary: Dict[str, int]
    recommendations: List[str]


class DiagnosticTool:
    """系统诊断工具"""
    
    def __init__(self):
        """初始化诊断工具"""
        self.logger = get_logger()
        self.results: List[DiagnosticResult] = []
    
    def run_full_diagnostics(self) -> SystemDiagnostics:
        """
        运行完整系统诊断
        
        Returns:
            诊断报告
        """
        self.logger.info("开始系统诊断...")
        self.results = []
        
        # 运行各项检查
        self._check_python_environment()
        self._check_dependencies()
        self._check_file_system()
        self._check_mt5_connection()
        self._check_database()
        self._check_redis()
        self._check_logs()
        self._check_disk_space()
        self._check_network()
        
        # 生成报告
        report = self._generate_report()
        
        self.logger.info(f"诊断完成: {report.overall_status}")
        
        return report
    
    def _check_python_environment(self):
        """检查Python环境"""
        category = "Python环境"
        
        # Python版本
        version = sys.version_info
        if version >= (3, 9):
            self._add_result(category, "Python版本", "pass", 
                           f"Python {version.major}.{version.minor}.{version.micro}")
        else:
            self._add_result(category, "Python版本", "fail",
                           f"Python版本过低: {version.major}.{version.minor}.{version.micro}")
        
        # 虚拟环境
        in_venv = hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        )
        
        if in_venv:
            self._add_result(category, "虚拟环境", "pass", "运行在虚拟环境中")
        else:
            self._add_result(category, "虚拟环境", "warning", "未使用虚拟环境")
    
    def _check_dependencies(self):
        """检查依赖包"""
        category = "依赖包"
        
        required_packages = [
            'pandas', 'numpy', 'MetaTrader5', 'yaml', 'redis'
        ]
        
        missing = []
        installed = []
        
        for package in required_packages:
            try:
                __import__(package)
                installed.append(package)
            except ImportError:
                missing.append(package)
        
        if not missing:
            self._add_result(category, "必需依赖", "pass",
                           f"所有必需包已安装 ({len(installed)}个)")
        else:
            self._add_result(category, "必需依赖", "fail",
                           f"缺少依赖包: {', '.join(missing)}")
    
    def _check_file_system(self):
        """检查文件系统"""
        category = "文件系统"
        
        # 检查必需目录
        required_dirs = ['data', 'logs', 'config', 'models']
        missing_dirs = []
        
        for dir_name in required_dirs:
            if not Path(dir_name).exists():
                missing_dirs.append(dir_name)
        
        if not missing_dirs:
            self._add_result(category, "目录结构", "pass", "所有必需目录存在")
        else:
            self._add_result(category, "目录结构", "warning",
                           f"缺少目录: {', '.join(missing_dirs)}")
        
        # 检查配置文件
        config_file = Path("config/config.yaml")
        if config_file.exists():
            self._add_result(category, "配置文件", "pass", "配置文件存在")
        else:
            self._add_result(category, "配置文件", "fail", "配置文件不存在")
        
        # 检查写入权限
        try:
            test_file = Path("data/.write_test")
            test_file.write_text("test")
            test_file.unlink()
            self._add_result(category, "写入权限", "pass", "具有写入权限")
        except Exception as e:
            self._add_result(category, "写入权限", "fail", f"无写入权限: {e}")
    
    def _check_mt5_connection(self):
        """检查MT5连接"""
        category = "MT5连接"
        
        try:
            import MetaTrader5 as mt5
            
            if mt5.initialize():
                terminal_info = mt5.terminal_info()
                if terminal_info:
                    self._add_result(category, "MT5连接", "pass",
                                   f"已连接到MT5 (版本 {terminal_info.build})")
                else:
                    self._add_result(category, "MT5连接", "warning",
                                   "MT5已初始化但无法获取终端信息")
                mt5.shutdown()
            else:
                self._add_result(category, "MT5连接", "fail",
                               "无法初始化MT5连接")
        except ImportError:
            self._add_result(category, "MT5连接", "fail",
                           "MetaTrader5库未安装")
        except Exception as e:
            self._add_result(category, "MT5连接", "fail", f"连接错误: {e}")
    
    def _check_database(self):
        """检查数据库"""
        category = "数据库"
        
        db_path = Path("data/trading.db")
        
        if db_path.exists():
            size_mb = db_path.stat().st_size / (1024 * 1024)
            self._add_result(category, "SQLite数据库", "pass",
                           f"数据库存在 ({size_mb:.2f} MB)")
        else:
            self._add_result(category, "SQLite数据库", "warning",
                           "数据库文件不存在（将在首次运行时创建）")
    
    def _check_redis(self):
        """检查Redis"""
        category = "Redis缓存"
        
        try:
            import redis
            
            r = redis.Redis(host='localhost', port=6379, db=0, socket_timeout=2)
            r.ping()
            self._add_result(category, "Redis连接", "pass", "Redis服务正常")
        except ImportError:
            self._add_result(category, "Redis连接", "warning",
                           "Redis库未安装（可选功能）")
        except Exception as e:
            self._add_result(category, "Redis连接", "warning",
                           f"Redis不可用: {e}（可选功能）")
    
    def _check_logs(self):
        """检查日志文件"""
        category = "日志系统"
        
        log_dir = Path("logs")
        
        if log_dir.exists():
            log_files = list(log_dir.glob("*.log"))
            total_size = sum(f.stat().st_size for f in log_files) / (1024 * 1024)
            
            self._add_result(category, "日志文件", "pass",
                           f"{len(log_files)}个日志文件 ({total_size:.2f} MB)")
            
            # 检查日志大小
            if total_size > 100:
                self._add_result(category, "日志大小", "warning",
                               f"日志文件较大 ({total_size:.2f} MB)，建议清理")
        else:
            self._add_result(category, "日志文件", "warning", "日志目录不存在")
    
    def _check_disk_space(self):
        """检查磁盘空间"""
        category = "磁盘空间"
        
        try:
            import psutil
            
            disk = psutil.disk_usage('.')
            free_gb = disk.free / (1024 ** 3)
            percent_used = disk.percent
            
            if free_gb > 5:
                self._add_result(category, "可用空间", "pass",
                               f"{free_gb:.1f} GB 可用 ({100-percent_used:.1f}% 剩余)")
            elif free_gb > 1:
                self._add_result(category, "可用空间", "warning",
                               f"磁盘空间不足: {free_gb:.1f} GB")
            else:
                self._add_result(category, "可用空间", "fail",
                               f"磁盘空间严重不足: {free_gb:.1f} GB")
        except Exception as e:
            self._add_result(category, "可用空间", "warning", f"无法检查: {e}")
    
    def _check_network(self):
        """检查网络连接"""
        category = "网络连接"
        
        try:
            import socket
            
            # 测试DNS解析
            socket.gethostbyname('www.google.com')
            self._add_result(category, "网络连接", "pass", "网络连接正常")
        except Exception as e:
            self._add_result(category, "网络连接", "warning",
                           f"网络可能不可用: {e}")
    
    def _add_result(self, category: str, name: str, status: str, message: str,
                   details: Optional[Dict] = None):
        """添加诊断结果"""
        result = DiagnosticResult(
            category=category,
            name=name,
            status=status,
            message=message,
            details=details
        )
        self.results.append(result)
    
    def _generate_report(self) -> SystemDiagnostics:
        """生成诊断报告"""
        # 统计结果
        summary = {
            'pass': sum(1 for r in self.results if r.status == 'pass'),
            'warning': sum(1 for r in self.results if r.status == 'warning'),
            'fail': sum(1 for r in self.results if r.status == 'fail'),
        }
        
        # 确定整体状态
        if summary['fail'] > 0:
            overall_status = 'critical'
        elif summary['warning'] > 0:
            overall_status = 'warning'
        else:
            overall_status = 'healthy'
        
        # 生成建议
        recommendations = self._generate_recommendations()
        
        return SystemDiagnostics(
            timestamp=datetime.now(),
            overall_status=overall_status,
            results=self.results,
            summary=summary,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self) -> List[str]:
        """生成建议"""
        recommendations = []
        
        for result in self.results:
            if result.status == 'fail':
                if 'Python版本' in result.name:
                    recommendations.append("升级Python到3.9或更高版本")
                elif '依赖' in result.name:
                    recommendations.append("运行: pip install -r requirements.txt")
                elif 'MT5' in result.name:
                    recommendations.append("安装并启动MetaTrader 5平台")
                elif '配置文件' in result.name:
                    recommendations.append("创建config/config.yaml配置文件")
                elif '磁盘空间' in result.name:
                    recommendations.append("清理磁盘空间或扩展存储")
            
            elif result.status == 'warning':
                if '虚拟环境' in result.name:
                    recommendations.append("建议使用虚拟环境: python -m venv venv")
                elif '日志大小' in result.name:
                    recommendations.append("清理旧日志文件")
                elif 'Redis' in result.name:
                    recommendations.append("安装Redis以启用缓存功能（可选）")
        
        return list(set(recommendations))  # 去重
    
    def print_report(self, report: SystemDiagnostics):
        """打印诊断报告"""
        print("\n" + "=" * 60)
        print("系统诊断报告")
        print("=" * 60)
        print(f"时间: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"整体状态: {report.overall_status.upper()}")
        print(f"\n统计: ✓ {report.summary['pass']}  ⚠ {report.summary['warning']}  ✗ {report.summary['fail']}")
        print("\n" + "-" * 60)
        
        # 按类别分组
        categories = {}
        for result in report.results:
            if result.category not in categories:
                categories[result.category] = []
            categories[result.category].append(result)
        
        # 打印结果
        for category, results in categories.items():
            print(f"\n{category}:")
            for result in results:
                status_symbol = {
                    'pass': '✓',
                    'warning': '⚠',
                    'fail': '✗'
                }.get(result.status, '?')
                
                print(f"  {status_symbol} {result.name}: {result.message}")
        
        # 打印建议
        if report.recommendations:
            print("\n" + "-" * 60)
            print("\n建议:")
            for i, rec in enumerate(report.recommendations, 1):
                print(f"  {i}. {rec}")
        
        print("\n" + "=" * 60)


def run_diagnostics() -> SystemDiagnostics:
    """
    便捷函数：运行系统诊断
    
    Returns:
        诊断报告
    """
    tool = DiagnosticTool()
    report = tool.run_full_diagnostics()
    tool.print_report(report)
    return report
