"""
日志系统
提供统一的日志记录功能，支持文件和控制台输出
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

from .config import LoggingConfig


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""
    
    # ANSI颜色代码
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',    # 红色
        'CRITICAL': '\033[35m', # 紫色
        'RESET': '\033[0m'      # 重置
    }
    
    def format(self, record):
        # 添加颜色
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


class TradingLogger:
    """交易系统日志管理器"""
    
    def __init__(self, config: LoggingConfig):
        self.config = config
        self._loggers = {}
        self._setup_logging()
    
    def _setup_logging(self):
        """设置日志系统"""
        # 创建日志目录
        log_dir = Path(self.config.file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置根日志级别
        logging.getLogger().setLevel(getattr(logging, self.config.level.upper()))
        
        # 创建主日志记录器
        self._setup_main_logger()
        
        # 创建专门的日志记录器
        self._setup_specialized_loggers()
    
    def _setup_main_logger(self):
        """设置主日志记录器"""
        logger = logging.getLogger('trading')
        logger.setLevel(getattr(logging, self.config.level.upper()))
        
        # 清除现有处理器
        logger.handlers.clear()
        
        # 文件处理器 - 使用轮转文件
        file_handler = logging.handlers.RotatingFileHandler(
            self.config.file_path,
            maxBytes=self.config.max_file_size,
            backupCount=self.config.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, self.config.level.upper()))
        file_formatter = logging.Formatter(self.config.format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # 控制台处理器
        if self.config.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, self.config.level.upper()))
            console_formatter = ColoredFormatter(self.config.format)
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        self._loggers['main'] = logger
    
    def _setup_specialized_loggers(self):
        """设置专门的日志记录器"""
        # 交易日志
        trade_logger = self._create_logger(
            'trading.trades',
            'logs/trades.log',
            '%(asctime)s - %(message)s'
        )
        self._loggers['trades'] = trade_logger
        
        # 策略日志
        strategy_logger = self._create_logger(
            'trading.strategy',
            'logs/strategy.log',
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self._loggers['strategy'] = strategy_logger
        
        # 风险管理日志
        risk_logger = self._create_logger(
            'trading.risk',
            'logs/risk.log',
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        self._loggers['risk'] = risk_logger
        
        # 数据日志
        data_logger = self._create_logger(
            'trading.data',
            'logs/data.log',
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self._loggers['data'] = data_logger
        
        # 系统日志
        system_logger = self._create_logger(
            'trading.system',
            'logs/system.log',
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        self._loggers['system'] = system_logger
        
        # 错误日志
        error_logger = self._create_logger(
            'trading.error',
            'logs/error.log',
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            level='ERROR'
        )
        self._loggers['error'] = error_logger
    
    def _create_logger(self, name: str, file_path: str, format_str: str, level: str = None) -> logging.Logger:
        """创建专门的日志记录器"""
        logger = logging.getLogger(name)
        
        if level is None:
            level = self.config.level
        
        logger.setLevel(getattr(logging, level.upper()))
        
        # 清除现有处理器
        logger.handlers.clear()
        
        # 创建日志目录
        log_dir = Path(file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 文件处理器
        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=self.config.max_file_size,
            backupCount=self.config.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        file_formatter = logging.Formatter(format_str)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # 防止日志传播到父记录器
        logger.propagate = False
        
        return logger
    
    def get_logger(self, name: str = 'main') -> logging.Logger:
        """获取指定的日志记录器"""
        return self._loggers.get(name, self._loggers['main'])
    
    def log_trade(self, trade_info: dict):
        """记录交易信息"""
        trade_logger = self.get_logger('trades')
        trade_logger.info(f"TRADE: {trade_info}")
    
    def log_signal(self, signal_info: dict):
        """记录信号信息"""
        strategy_logger = self.get_logger('strategy')
        strategy_logger.info(f"SIGNAL: {signal_info}")
    
    def log_risk_event(self, risk_info: dict):
        """记录风险事件"""
        risk_logger = self.get_logger('risk')
        risk_logger.warning(f"RISK: {risk_info}")
    
    def log_error(self, error_msg: str, exception: Exception = None):
        """记录错误信息"""
        error_logger = self.get_logger('error')
        if exception:
            error_logger.error(f"{error_msg}: {str(exception)}", exc_info=True)
        else:
            error_logger.error(error_msg)
    
    def log_system_event(self, event_info: dict):
        """记录系统事件"""
        system_logger = self.get_logger('system')
        system_logger.info(f"SYSTEM: {event_info}")


class LoggerMixin:
    """日志混入类，为其他类提供日志功能"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = None
    
    @property
    def logger(self) -> logging.Logger:
        """获取日志记录器"""
        if self._logger is None:
            class_name = self.__class__.__name__
            self._logger = logging.getLogger(f'trading.{class_name.lower()}')
        return self._logger
    
    def log_info(self, message: str, **kwargs):
        """记录信息日志"""
        self.logger.info(message, **kwargs)
    
    def log_warning(self, message: str, **kwargs):
        """记录警告日志"""
        self.logger.warning(message, **kwargs)
    
    def log_error(self, message: str, exception: Exception = None, **kwargs):
        """记录错误日志"""
        if exception:
            self.logger.error(f"{message}: {str(exception)}", exc_info=True, **kwargs)
        else:
            self.logger.error(message, **kwargs)
    
    def log_debug(self, message: str, **kwargs):
        """记录调试日志"""
        self.logger.debug(message, **kwargs)


# 全局日志管理器
_global_logger: Optional[TradingLogger] = None


def setup_logging(config: LoggingConfig) -> TradingLogger:
    """设置全局日志系统"""
    global _global_logger
    _global_logger = TradingLogger(config)
    return _global_logger


def get_logger(name: str = 'main') -> logging.Logger:
    """获取日志记录器"""
    if _global_logger is None:
        # 使用默认配置
        from .config import LoggingConfig
        setup_logging(LoggingConfig())
    
    return _global_logger.get_logger(name)


def log_trade(trade_info: dict):
    """记录交易信息"""
    if _global_logger:
        _global_logger.log_trade(trade_info)


def log_signal(signal_info: dict):
    """记录信号信息"""
    if _global_logger:
        _global_logger.log_signal(signal_info)


def log_risk_event(risk_info: dict):
    """记录风险事件"""
    if _global_logger:
        _global_logger.log_risk_event(risk_info)


def log_error(error_msg: str, exception: Exception = None):
    """记录错误信息"""
    if _global_logger:
        _global_logger.log_error(error_msg, exception)


def log_system_event(event_info: dict):
    """记录系统事件"""
    if _global_logger:
        _global_logger.log_system_event(event_info)