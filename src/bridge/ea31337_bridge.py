#!/usr/bin/env python3
"""
EA31337桥接器核心类
提供与EA31337框架的文件通信接口
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from dataclasses import asdict

from ..core.models import Signal, MarketData
from ..core.exceptions import BridgeError, ConfigurationException
from ..utils.file_utils import safe_read_file, safe_write_file


logger = logging.getLogger(__name__)


class EA31337Bridge:
    """EA31337框架集成桥接器"""
    
    def __init__(self, config_path: str = "ea31337", signal_timeout: int = 30):
        """
        初始化EA31337桥接器
        
        Args:
            config_path: EA31337配置文件路径
            signal_timeout: 信号超时时间(秒)
        """
        self.config_path = Path(config_path)
        self.signal_timeout = signal_timeout
        
        # 确保目录存在
        self.config_path.mkdir(parents=True, exist_ok=True)
        
        # 定义通信文件路径
        self.signal_file = self.config_path / "signals.json"
        self.status_file = self.config_path / "status.json"
        self.command_file = self.config_path / "commands.json"
        self.response_file = self.config_path / "responses.json"
        
        # 初始化状态
        self._last_signal_check = datetime.now()
        self._is_connected = False
        
        logger.info(f"EA31337Bridge初始化完成，配置路径: {self.config_path}")
    
    def is_connected(self) -> bool:
        """检查与EA31337的连接状态"""
        try:
            # 检查状态文件是否存在且最近更新
            if not self.status_file.exists():
                return False
            
            status = self.get_status()
            if not status:
                return False
            
            # 检查最后更新时间（5分钟内）
            last_update = datetime.fromisoformat(status.get('timestamp', '1970-01-01'))
            time_diff = (datetime.now() - last_update).total_seconds()
            
            self._is_connected = time_diff < 300  # 5分钟
            return self._is_connected
            
        except Exception as e:
            logger.error(f"检查连接状态失败: {e}")
            self._is_connected = False
            return False
    
    def get_signals(self) -> List[Signal]:
        """
        获取EA生成的交易信号
        
        Returns:
            交易信号列表
        """
        try:
            if not self.signal_file.exists():
                logger.debug("信号文件不存在")
                return []
            
            # 读取信号文件
            signal_data = safe_read_file(self.signal_file)
            if not signal_data:
                return []
            
            signals_json = json.loads(signal_data)
            signals = []
            
            for signal_dict in signals_json.get('signals', []):
                try:
                    # 转换时间戳
                    if 'timestamp' in signal_dict:
                        signal_dict['timestamp'] = datetime.fromisoformat(signal_dict['timestamp'])
                    else:
                        signal_dict['timestamp'] = datetime.now()
                    
                    # 创建Signal对象
                    signal = Signal(**signal_dict)
                    
                    # 检查信号是否过期
                    age = (datetime.now() - signal.timestamp).total_seconds()
                    if age <= self.signal_timeout:
                        signals.append(signal)
                    else:
                        logger.debug(f"信号已过期: {signal.strategy_id}, 年龄: {age}秒")
                        
                except Exception as e:
                    logger.error(f"解析信号失败: {e}, 数据: {signal_dict}")
                    continue
            
            logger.debug(f"获取到 {len(signals)} 个有效信号")
            return signals
            
        except Exception as e:
            logger.error(f"获取信号失败: {e}")
            raise BridgeError(f"获取EA31337信号失败: {e}")
    
    def get_status(self) -> Optional[Dict[str, Any]]:
        """
        获取EA运行状态
        
        Returns:
            EA状态信息字典
        """
        try:
            if not self.status_file.exists():
                logger.debug("状态文件不存在")
                return None
            
            status_data = safe_read_file(self.status_file)
            if not status_data:
                return None
            
            status = json.loads(status_data)
            logger.debug(f"获取EA状态: {status.get('status', 'unknown')}")
            return status
            
        except Exception as e:
            logger.error(f"获取EA状态失败: {e}")
            return None
    
    def send_command(self, command: str, params: Dict[str, Any] = None) -> bool:
        """
        向EA发送控制命令
        
        Args:
            command: 命令名称
            params: 命令参数
            
        Returns:
            命令发送是否成功
        """
        try:
            command_data = {
                'command': command,
                'params': params or {},
                'timestamp': datetime.now().isoformat(),
                'id': int(time.time() * 1000)  # 使用时间戳作为命令ID
            }
            
            # 写入命令文件
            command_json = json.dumps(command_data, indent=2, ensure_ascii=False)
            success = safe_write_file(self.command_file, command_json)
            
            if success:
                logger.info(f"发送命令成功: {command}")
                return True
            else:
                logger.error(f"发送命令失败: {command}")
                return False
                
        except Exception as e:
            logger.error(f"发送命令异常: {e}")
            raise BridgeError(f"发送EA31337命令失败: {e}")
    
    def wait_for_response(self, command_id: int, timeout: int = 10) -> Optional[Dict[str, Any]]:
        """
        等待命令响应
        
        Args:
            command_id: 命令ID
            timeout: 超时时间(秒)
            
        Returns:
            响应数据或None
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                if not self.response_file.exists():
                    time.sleep(0.1)
                    continue
                
                response_data = safe_read_file(self.response_file)
                if not response_data:
                    time.sleep(0.1)
                    continue
                
                responses = json.loads(response_data)
                
                # 查找匹配的响应
                for response in responses.get('responses', []):
                    if response.get('command_id') == command_id:
                        logger.debug(f"收到命令响应: {command_id}")
                        return response
                
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"读取响应失败: {e}")
                time.sleep(0.1)
        
        logger.warning(f"等待命令响应超时: {command_id}")
        return None
    
    def update_parameters(self, strategy: str, params: Dict[str, Any]) -> bool:
        """
        动态更新EA策略参数
        
        Args:
            strategy: 策略名称
            params: 参数字典
            
        Returns:
            更新是否成功
        """
        try:
            command_params = {
                'strategy': strategy,
                'parameters': params
            }
            
            return self.send_command('update_parameters', command_params)
            
        except Exception as e:
            logger.error(f"更新策略参数失败: {e}")
            return False
    
    def start_strategy(self, strategy: str) -> bool:
        """启动指定策略"""
        return self.send_command('start_strategy', {'strategy': strategy})
    
    def stop_strategy(self, strategy: str) -> bool:
        """停止指定策略"""
        return self.send_command('stop_strategy', {'strategy': strategy})
    
    def get_strategy_status(self, strategy: str) -> Optional[Dict[str, Any]]:
        """获取策略状态"""
        status = self.get_status()
        if status and 'strategies' in status:
            return status['strategies'].get(strategy)
        return None
    
    def emergency_stop(self) -> bool:
        """紧急停止所有策略"""
        logger.warning("执行紧急停止命令")
        return self.send_command('emergency_stop')
    
    def health_check(self) -> Dict[str, Any]:
        """
        执行健康检查
        
        Returns:
            健康检查结果
        """
        health_status = {
            'bridge_connected': self.is_connected(),
            'signal_file_exists': self.signal_file.exists(),
            'status_file_exists': self.status_file.exists(),
            'last_signal_check': self._last_signal_check.isoformat(),
            'config_path': str(self.config_path),
            'timestamp': datetime.now().isoformat()
        }
        
        # 检查文件权限
        try:
            test_file = self.config_path / "test_write.tmp"
            test_file.write_text("test")
            test_file.unlink()
            health_status['write_permission'] = True
        except Exception:
            health_status['write_permission'] = False
        
        # 获取EA状态
        ea_status = self.get_status()
        if ea_status:
            health_status['ea_status'] = ea_status.get('status', 'unknown')
            health_status['ea_last_update'] = ea_status.get('timestamp')
        
        return health_status
    
    def cleanup(self) -> None:
        """清理临时文件"""
        try:
            temp_files = [
                self.command_file,
                self.response_file
            ]
            
            for temp_file in temp_files:
                if temp_file.exists():
                    temp_file.unlink()
                    logger.debug(f"清理临时文件: {temp_file}")
                    
        except Exception as e:
            logger.error(f"清理文件失败: {e}")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.cleanup()