"""
配置管理系统
支持YAML和JSON配置文件的加载和管理
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """数据库配置"""
    sqlite_path: str = "data/trading.db"
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None


@dataclass
class MT5Config:
    """MT5连接配置"""
    server: str = ""
    login: int = 0
    password: str = ""
    path: str = ""
    timeout: int = 60000
    portable: bool = False


@dataclass
class RiskConfig:
    """风险管理配置"""
    max_risk_per_trade: float = 0.02  # 单笔最大风险2%
    max_daily_drawdown: float = 0.05  # 日最大回撤5%
    max_weekly_drawdown: float = 0.10  # 周最大回撤10%
    max_monthly_drawdown: float = 0.15  # 月最大回撤15%
    correlation_threshold: float = 0.7  # 相关性阈值
    max_positions: int = 10  # 最大持仓数
    max_lot_per_symbol: float = 1.0  # 单品种最大手数
    stop_loss_pct: float = 0.02  # 默认止损百分比
    take_profit_pct: float = 0.04  # 默认止盈百分比


@dataclass
class LLMConfig:
    """LLM配置"""
    model_path: str = "models/llama-3.2-1b"
    model_type: str = "llama"
    max_tokens: int = 512
    temperature: float = 0.7
    use_gpu: bool = True
    batch_size: int = 1
    context_length: int = 2048


@dataclass
class RLConfig:
    """强化学习配置"""
    algorithm: str = "PPO"  # PPO, SAC, A2C
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "logs/trading.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    console_output: bool = True


@dataclass
class TradingConfig:
    """交易配置"""
    symbols: list = field(default_factory=lambda: ["EURUSD", "XAUUSD", "USOIL"])
    timeframes: list = field(default_factory=lambda: ["M1", "M5", "M15", "H1", "H4", "D1"])
    default_lot_size: float = 0.01
    slippage: int = 3
    magic_number: int = 123456
    ea31337_path: str = "ea31337"
    strategies_enabled: list = field(default_factory=lambda: ["trend", "scalp", "breakout"])


@dataclass
class SystemConfig:
    """系统总配置"""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    mt5: MT5Config = field(default_factory=MT5Config)
    risk: RiskConfig = field(default_factory=RiskConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    
    # 系统设置
    debug_mode: bool = False
    simulation_mode: bool = True
    auto_start: bool = False
    update_interval: int = 1  # 秒
    
    def validate(self) -> bool:
        """验证配置有效性"""
        try:
            # 验证风险参数
            if not 0 < self.risk.max_risk_per_trade <= 0.1:
                raise ValueError("单笔风险必须在0-10%之间")
            
            if not 0 < self.risk.max_daily_drawdown <= 0.2:
                raise ValueError("日最大回撤必须在0-20%之间")
            
            # 验证交易参数
            if self.trading.default_lot_size <= 0:
                raise ValueError("默认手数必须大于0")
            
            if not self.trading.symbols:
                raise ValueError("必须配置至少一个交易品种")
            
            # 验证路径
            if self.llm.model_path and not os.path.exists(self.llm.model_path):
                logger.warning(f"LLM模型路径不存在: {self.llm.model_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"配置验证失败: {e}")
            return False


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self._config: Optional[SystemConfig] = None
        
    def load_config(self, config_file: str = "config.yaml") -> SystemConfig:
        """加载配置文件"""
        config_path = self.config_dir / config_file
        
        if not config_path.exists():
            logger.info(f"配置文件不存在，创建默认配置: {config_path}")
            self._config = SystemConfig()
            self.save_config(config_file)
            return self._config
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            self._config = self._dict_to_config(data)
            
            if not self._config.validate():
                raise ValueError("配置验证失败")
            
            logger.info(f"成功加载配置文件: {config_path}")
            return self._config
            
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            logger.info("使用默认配置")
            self._config = SystemConfig()
            return self._config
    
    def save_config(self, config_file: str = "config.yaml") -> bool:
        """保存配置文件"""
        if self._config is None:
            logger.error("没有配置可保存")
            return False
        
        config_path = self.config_dir / config_file
        
        try:
            data = self._config_to_dict(self._config)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                    yaml.dump(data, f, default_flow_style=False, allow_unicode=True, indent=2)
                else:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"配置文件已保存: {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")
            return False
    
    def get_config(self) -> SystemConfig:
        """获取当前配置"""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def update_config(self, **kwargs) -> bool:
        """更新配置"""
        if self._config is None:
            self._config = SystemConfig()
        
        try:
            for key, value in kwargs.items():
                if hasattr(self._config, key):
                    setattr(self._config, key, value)
                else:
                    logger.warning(f"未知配置项: {key}")
            
            if self._config.validate():
                logger.info("配置更新成功")
                return True
            else:
                logger.error("配置更新失败：验证不通过")
                return False
                
        except Exception as e:
            logger.error(f"更新配置失败: {e}")
            return False
    
    def _dict_to_config(self, data: Dict[str, Any]) -> SystemConfig:
        """将字典转换为配置对象"""
        config = SystemConfig()
        
        # 递归设置配置项
        for key, value in data.items():
            if hasattr(config, key):
                attr = getattr(config, key)
                if hasattr(attr, '__dict__'):  # 嵌套配置对象
                    for sub_key, sub_value in value.items():
                        if hasattr(attr, sub_key):
                            setattr(attr, sub_key, sub_value)
                else:
                    setattr(config, key, value)
        
        return config
    
    def _config_to_dict(self, config: SystemConfig) -> Dict[str, Any]:
        """将配置对象转换为字典"""
        result = {}
        
        for key, value in config.__dict__.items():
            if hasattr(value, '__dict__'):  # 嵌套配置对象
                result[key] = {k: v for k, v in value.__dict__.items()}
            else:
                result[key] = value
        
        return result
    
    def get_symbol_config(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取特定品种的配置"""
        config_file = f"symbols/{symbol.lower()}.yaml"
        config_path = self.config_dir / config_file
        
        if not config_path.exists():
            return None
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"加载品种配置失败 {symbol}: {e}")
            return None
    
    def save_symbol_config(self, symbol: str, config: Dict[str, Any]) -> bool:
        """保存品种配置"""
        symbols_dir = self.config_dir / "symbols"
        symbols_dir.mkdir(exist_ok=True)
        
        config_file = f"{symbol.lower()}.yaml"
        config_path = symbols_dir / config_file
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
            
            logger.info(f"品种配置已保存: {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"保存品种配置失败 {symbol}: {e}")
            return False


# 全局配置管理器实例
config_manager = ConfigManager()


def get_config() -> SystemConfig:
    """获取全局配置"""
    return config_manager.get_config()


def load_config(config_file: str = "config.yaml") -> SystemConfig:
    """加载配置文件"""
    return config_manager.load_config(config_file)


def save_config(config_file: str = "config.yaml") -> bool:
    """保存配置文件"""
    return config_manager.save_config(config_file)