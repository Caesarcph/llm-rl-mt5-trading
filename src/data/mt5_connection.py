#!/usr/bin/env python3
"""
MT5连接管理器
处理MT5平台的连接、重连和状态监控
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from ..core.logging import get_logger
from ..core.exceptions import TradingSystemException


class ConnectionStatus(Enum):
    """连接状态枚举"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass
class ConnectionConfig:
    """MT5连接配置"""
    login: Optional[int] = None
    password: Optional[str] = None
    server: Optional[str] = None
    timeout: int = 60000  # 连接超时时间(毫秒)
    max_retries: int = 5  # 最大重试次数
    retry_delay: int = 5  # 重试延迟(秒)
    heartbeat_interval: int = 30  # 心跳检测间隔(秒)
    auto_reconnect: bool = True  # 是否自动重连


class MT5ConnectionError(TradingSystemException):
    """MT5连接异常"""
    pass


class MT5Connection:
    """MT5连接管理器"""
    
    def __init__(self, config: ConnectionConfig):
        """
        初始化MT5连接管理器
        
        Args:
            config: 连接配置
        """
        if mt5 is None:
            raise MT5ConnectionError("MetaTrader5库未安装，请运行: pip install MetaTrader5")
        
        self.config = config
        self.logger = get_logger()
        self._status = ConnectionStatus.DISCONNECTED
        self._last_heartbeat = None
        self._heartbeat_thread = None
        self._stop_heartbeat = False
        self._connection_lock = threading.Lock()
        self._status_callbacks = []
        self._retry_count = 0
        
        # 连接统计
        self._connection_stats = {
            'total_connections': 0,
            'successful_connections': 0,
            'failed_connections': 0,
            'reconnections': 0,
            'last_connection_time': None,
            'last_disconnection_time': None,
            'uptime_start': None
        }
    
    @property
    def status(self) -> ConnectionStatus:
        """获取当前连接状态"""
        return self._status
    
    @property
    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self._status == ConnectionStatus.CONNECTED
    
    @property
    def connection_stats(self) -> Dict[str, Any]:
        """获取连接统计信息"""
        stats = self._connection_stats.copy()
        if self.is_connected and stats['uptime_start']:
            stats['current_uptime'] = datetime.now() - stats['uptime_start']
        return stats
    
    def add_status_callback(self, callback: Callable[[ConnectionStatus], None]):
        """
        添加状态变化回调函数
        
        Args:
            callback: 状态变化时调用的函数
        """
        self._status_callbacks.append(callback)
    
    def _set_status(self, status: ConnectionStatus):
        """
        设置连接状态并触发回调
        
        Args:
            status: 新的连接状态
        """
        old_status = self._status
        self._status = status
        
        if old_status != status:
            self.logger.info(f"MT5连接状态变化: {old_status.value} -> {status.value}")
            
            # 触发状态回调
            for callback in self._status_callbacks:
                try:
                    callback(status)
                except Exception as e:
                    self.logger.error(f"状态回调执行失败: {e}")
    
    def connect(self) -> bool:
        """
        连接到MT5平台
        
        Returns:
            bool: 连接是否成功
        """
        with self._connection_lock:
            if self.is_connected:
                self.logger.warning("MT5已经连接，无需重复连接")
                return True
            
            self._set_status(ConnectionStatus.CONNECTING)
            self._connection_stats['total_connections'] += 1
            
            try:
                # 初始化MT5
                if not mt5.initialize():
                    error_code = mt5.last_error()
                    raise MT5ConnectionError(f"MT5初始化失败: {error_code}")
                
                # 如果提供了登录信息，则登录指定账户
                if self.config.login and self.config.password and self.config.server:
                    if not mt5.login(
                        login=self.config.login,
                        password=self.config.password,
                        server=self.config.server,
                        timeout=self.config.timeout
                    ):
                        error_code = mt5.last_error()
                        raise MT5ConnectionError(f"MT5登录失败: {error_code}")
                
                # 验证连接
                if not self._verify_connection():
                    raise MT5ConnectionError("连接验证失败")
                
                # 连接成功
                self._set_status(ConnectionStatus.CONNECTED)
                self._connection_stats['successful_connections'] += 1
                self._connection_stats['last_connection_time'] = datetime.now()
                self._connection_stats['uptime_start'] = datetime.now()
                self._retry_count = 0
                
                # 启动心跳检测
                self._start_heartbeat()
                
                self.logger.info("MT5连接成功")
                return True
                
            except Exception as e:
                self._set_status(ConnectionStatus.ERROR)
                self._connection_stats['failed_connections'] += 1
                self.logger.error(f"MT5连接失败: {e}")
                
                # 清理资源
                try:
                    mt5.shutdown()
                except:
                    pass
                
                return False
    
    def disconnect(self):
        """断开MT5连接"""
        with self._connection_lock:
            if self._status == ConnectionStatus.DISCONNECTED:
                return
            
            self.logger.info("正在断开MT5连接...")
            
            # 停止心跳检测
            self._stop_heartbeat_thread()
            
            # 关闭MT5连接
            try:
                mt5.shutdown()
            except Exception as e:
                self.logger.error(f"关闭MT5连接时出错: {e}")
            
            self._set_status(ConnectionStatus.DISCONNECTED)
            self._connection_stats['last_disconnection_time'] = datetime.now()
            self._connection_stats['uptime_start'] = None
            
            self.logger.info("MT5连接已断开")
    
    def reconnect(self) -> bool:
        """
        重新连接MT5
        
        Returns:
            bool: 重连是否成功
        """
        self.logger.info("开始重新连接MT5...")
        self._set_status(ConnectionStatus.RECONNECTING)
        self._connection_stats['reconnections'] += 1
        
        # 先断开现有连接
        self.disconnect()
        
        # 等待一段时间后重连
        time.sleep(self.config.retry_delay)
        
        return self.connect()
    
    def ensure_connection(self) -> bool:
        """
        确保MT5连接正常，如果断开则尝试重连
        
        Returns:
            bool: 连接是否正常
        """
        if self.is_connected and self._verify_connection():
            return True
        
        if not self.config.auto_reconnect:
            self.logger.warning("自动重连已禁用，连接异常")
            return False
        
        self.logger.warning("检测到连接异常，尝试重连...")
        
        for attempt in range(self.config.max_retries):
            self._retry_count = attempt + 1
            self.logger.info(f"重连尝试 {self._retry_count}/{self.config.max_retries}")
            
            if self.reconnect():
                self.logger.info("重连成功")
                return True
            
            if attempt < self.config.max_retries - 1:
                self.logger.warning(f"重连失败，{self.config.retry_delay}秒后重试...")
                time.sleep(self.config.retry_delay)
        
        self.logger.error(f"重连失败，已达到最大重试次数 {self.config.max_retries}")
        self._set_status(ConnectionStatus.ERROR)
        return False
    
    def _verify_connection(self) -> bool:
        """
        验证MT5连接是否正常
        
        Returns:
            bool: 连接是否正常
        """
        try:
            # 尝试获取账户信息来验证连接
            account_info = mt5.account_info()
            if account_info is None:
                return False
            
            # 尝试获取终端信息
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"连接验证失败: {e}")
            return False
    
    def _start_heartbeat(self):
        """启动心跳检测线程"""
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            return
        
        self._stop_heartbeat = False
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_worker,
            daemon=True,
            name="MT5HeartbeatThread"
        )
        self._heartbeat_thread.start()
        self.logger.debug("心跳检测线程已启动")
    
    def _stop_heartbeat_thread(self):
        """停止心跳检测线程"""
        self._stop_heartbeat = True
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=5)
        self.logger.debug("心跳检测线程已停止")
    
    def _heartbeat_worker(self):
        """心跳检测工作线程"""
        while not self._stop_heartbeat:
            try:
                if self.is_connected:
                    if self._verify_connection():
                        self._last_heartbeat = datetime.now()
                    else:
                        self.logger.warning("心跳检测失败，连接可能异常")
                        if self.config.auto_reconnect:
                            self.ensure_connection()
                
                # 等待下次心跳
                for _ in range(self.config.heartbeat_interval):
                    if self._stop_heartbeat:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                self.logger.error(f"心跳检测异常: {e}")
                time.sleep(5)  # 异常时等待5秒
    
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """
        获取账户信息
        
        Returns:
            Optional[Dict]: 账户信息字典，连接异常时返回None
        """
        if not self.ensure_connection():
            return None
        
        try:
            account_info = mt5.account_info()
            if account_info is None:
                return None
            
            return account_info._asdict()
            
        except Exception as e:
            self.logger.error(f"获取账户信息失败: {e}")
            return None
    
    def get_terminal_info(self) -> Optional[Dict[str, Any]]:
        """
        获取终端信息
        
        Returns:
            Optional[Dict]: 终端信息字典，连接异常时返回None
        """
        if not self.ensure_connection():
            return None
        
        try:
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                return None
            
            return terminal_info._asdict()
            
        except Exception as e:
            self.logger.error(f"获取终端信息失败: {e}")
            return None
    
    def __enter__(self):
        """上下文管理器入口"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.disconnect()