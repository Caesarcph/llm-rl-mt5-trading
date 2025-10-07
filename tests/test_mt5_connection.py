#!/usr/bin/env python3
"""
MT5连接管理器单元测试
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import threading
import time
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data.mt5_connection import (
    MT5Connection, 
    ConnectionConfig, 
    ConnectionStatus,
    MT5ConnectionError
)


class TestMT5Connection(unittest.TestCase):
    """MT5连接管理器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.config = ConnectionConfig(
            login=12345,
            password="test_password",
            server="test_server",
            timeout=30000,
            max_retries=3,
            retry_delay=1,
            heartbeat_interval=5,
            auto_reconnect=True
        )
    
    @patch('src.data.mt5_connection.mt5')
    def test_connection_initialization(self, mock_mt5):
        """测试连接初始化"""
        connection = MT5Connection(self.config)
        
        self.assertEqual(connection.status, ConnectionStatus.DISCONNECTED)
        self.assertFalse(connection.is_connected)
        self.assertEqual(connection.config, self.config)
    
    @patch('src.data.mt5_connection.mt5')
    def test_successful_connection(self, mock_mt5):
        """测试成功连接"""
        # 模拟MT5成功初始化和登录
        mock_mt5.initialize.return_value = True
        mock_mt5.login.return_value = True
        mock_mt5.account_info.return_value = Mock()
        mock_mt5.terminal_info.return_value = Mock()
        
        connection = MT5Connection(self.config)
        result = connection.connect()
        
        self.assertTrue(result)
        self.assertEqual(connection.status, ConnectionStatus.CONNECTED)
        self.assertTrue(connection.is_connected)
        
        # 验证MT5方法被正确调用
        mock_mt5.initialize.assert_called_once()
        mock_mt5.login.assert_called_once_with(
            login=self.config.login,
            password=self.config.password,
            server=self.config.server,
            timeout=self.config.timeout
        )
    
    @patch('src.data.mt5_connection.mt5')
    def test_connection_failure_initialization(self, mock_mt5):
        """测试初始化失败"""
        mock_mt5.initialize.return_value = False
        mock_mt5.last_error.return_value = (1, "Initialization failed")
        
        connection = MT5Connection(self.config)
        result = connection.connect()
        
        self.assertFalse(result)
        self.assertEqual(connection.status, ConnectionStatus.ERROR)
        self.assertFalse(connection.is_connected)
    
    @patch('src.data.mt5_connection.mt5')
    def test_connection_failure_login(self, mock_mt5):
        """测试登录失败"""
        mock_mt5.initialize.return_value = True
        mock_mt5.login.return_value = False
        mock_mt5.last_error.return_value = (2, "Login failed")
        
        connection = MT5Connection(self.config)
        result = connection.connect()
        
        self.assertFalse(result)
        self.assertEqual(connection.status, ConnectionStatus.ERROR)
        self.assertFalse(connection.is_connected)
    
    @patch('src.data.mt5_connection.mt5')
    def test_connection_without_credentials(self, mock_mt5):
        """测试无凭据连接"""
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = Mock()
        mock_mt5.terminal_info.return_value = Mock()
        
        # 创建无凭据配置
        config_no_creds = ConnectionConfig()
        connection = MT5Connection(config_no_creds)
        result = connection.connect()
        
        self.assertTrue(result)
        # 验证没有调用登录方法
        mock_mt5.login.assert_not_called()
    
    @patch('src.data.mt5_connection.mt5')
    def test_disconnect(self, mock_mt5):
        """测试断开连接"""
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = Mock()
        mock_mt5.terminal_info.return_value = Mock()
        
        connection = MT5Connection(self.config)
        connection.connect()
        
        # 断开连接
        connection.disconnect()
        
        self.assertEqual(connection.status, ConnectionStatus.DISCONNECTED)
        self.assertFalse(connection.is_connected)
        mock_mt5.shutdown.assert_called_once()
    
    @patch('src.data.mt5_connection.mt5')
    def test_reconnect(self, mock_mt5):
        """测试重新连接"""
        mock_mt5.initialize.return_value = True
        mock_mt5.login.return_value = True
        mock_mt5.account_info.return_value = Mock()
        mock_mt5.terminal_info.return_value = Mock()
        
        connection = MT5Connection(self.config)
        connection.connect()
        
        # 重新连接
        result = connection.reconnect()
        
        self.assertTrue(result)
        self.assertEqual(connection.status, ConnectionStatus.CONNECTED)
        # 验证shutdown和initialize都被调用了
        mock_mt5.shutdown.assert_called()
        self.assertEqual(mock_mt5.initialize.call_count, 2)  # 初始连接 + 重连
    
    @patch('src.data.mt5_connection.mt5')
    def test_ensure_connection_when_connected(self, mock_mt5):
        """测试连接正常时的ensure_connection"""
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = Mock()
        mock_mt5.terminal_info.return_value = Mock()
        
        connection = MT5Connection(self.config)
        connection.connect()
        
        # 确保连接
        result = connection.ensure_connection()
        
        self.assertTrue(result)
        # 不应该重新初始化
        self.assertEqual(mock_mt5.initialize.call_count, 1)
    
    @patch('src.data.mt5_connection.mt5')
    def test_ensure_connection_when_disconnected(self, mock_mt5):
        """测试连接断开时的ensure_connection"""
        # 第一次连接成功
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = Mock()
        mock_mt5.terminal_info.return_value = Mock()
        
        connection = MT5Connection(self.config)
        connection.connect()
        
        # 模拟连接断开
        connection._set_status(ConnectionStatus.DISCONNECTED)
        
        # 确保连接应该触发重连
        result = connection.ensure_connection()
        
        self.assertTrue(result)
        # 应该重新初始化
        self.assertEqual(mock_mt5.initialize.call_count, 2)
    
    @patch('src.data.mt5_connection.mt5')
    def test_verify_connection_success(self, mock_mt5):
        """测试连接验证成功"""
        mock_mt5.account_info.return_value = Mock()
        mock_mt5.terminal_info.return_value = Mock()
        
        connection = MT5Connection(self.config)
        result = connection._verify_connection()
        
        self.assertTrue(result)
    
    @patch('src.data.mt5_connection.mt5')
    def test_verify_connection_failure(self, mock_mt5):
        """测试连接验证失败"""
        mock_mt5.account_info.return_value = None
        
        connection = MT5Connection(self.config)
        result = connection._verify_connection()
        
        self.assertFalse(result)
    
    @patch('src.data.mt5_connection.mt5')
    def test_get_account_info(self, mock_mt5):
        """测试获取账户信息"""
        mock_account_info = Mock()
        mock_account_info._asdict.return_value = {
            'login': 12345,
            'balance': 10000.0,
            'equity': 10000.0
        }
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = mock_account_info
        mock_mt5.terminal_info.return_value = Mock()
        
        connection = MT5Connection(self.config)
        connection.connect()
        
        account_info = connection.get_account_info()
        
        self.assertIsNotNone(account_info)
        self.assertEqual(account_info['login'], 12345)
        self.assertEqual(account_info['balance'], 10000.0)
    
    @patch('src.data.mt5_connection.mt5')
    def test_get_terminal_info(self, mock_mt5):
        """测试获取终端信息"""
        mock_terminal_info = Mock()
        mock_terminal_info._asdict.return_value = {
            'company': 'Test Company',
            'name': 'Test Terminal'
        }
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = Mock()
        mock_mt5.terminal_info.return_value = mock_terminal_info
        
        connection = MT5Connection(self.config)
        connection.connect()
        
        terminal_info = connection.get_terminal_info()
        
        self.assertIsNotNone(terminal_info)
        self.assertEqual(terminal_info['company'], 'Test Company')
        self.assertEqual(terminal_info['name'], 'Test Terminal')
    
    def test_status_callback(self):
        """测试状态变化回调"""
        callback_calls = []
        
        def status_callback(status):
            callback_calls.append(status)
        
        connection = MT5Connection(self.config)
        connection.add_status_callback(status_callback)
        
        # 触发状态变化
        connection._set_status(ConnectionStatus.CONNECTING)
        connection._set_status(ConnectionStatus.CONNECTED)
        
        self.assertEqual(len(callback_calls), 2)
        self.assertEqual(callback_calls[0], ConnectionStatus.CONNECTING)
        self.assertEqual(callback_calls[1], ConnectionStatus.CONNECTED)
    
    @patch('src.data.mt5_connection.mt5')
    def test_connection_stats(self, mock_mt5):
        """测试连接统计"""
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = Mock()
        mock_mt5.terminal_info.return_value = Mock()
        
        connection = MT5Connection(self.config)
        
        # 初始统计
        stats = connection.connection_stats
        self.assertEqual(stats['total_connections'], 0)
        self.assertEqual(stats['successful_connections'], 0)
        
        # 连接后统计
        connection.connect()
        stats = connection.connection_stats
        self.assertEqual(stats['total_connections'], 1)
        self.assertEqual(stats['successful_connections'], 1)
        self.assertIsNotNone(stats['last_connection_time'])
    
    @patch('src.data.mt5_connection.mt5')
    def test_context_manager(self, mock_mt5):
        """测试上下文管理器"""
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = Mock()
        mock_mt5.terminal_info.return_value = Mock()
        
        with MT5Connection(self.config) as connection:
            self.assertTrue(connection.is_connected)
        
        # 退出上下文后应该断开连接
        mock_mt5.shutdown.assert_called()
    
    def test_missing_mt5_library(self):
        """测试MT5库缺失的情况"""
        with patch('src.data.mt5_connection.mt5', None):
            with self.assertRaises(MT5ConnectionError):
                MT5Connection(self.config)


if __name__ == '__main__':
    unittest.main()