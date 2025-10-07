#!/usr/bin/env python3
"""
EA31337策略配置管理器
处理.set配置文件的读写和参数管理
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from ..core.exceptions import ConfigurationException
from ..utils.file_utils import safe_read_file, safe_write_file, backup_file


logger = logging.getLogger(__name__)


@dataclass
class SetParameter:
    """SET文件参数模型"""
    name: str
    value: Any
    type: str  # "int", "double", "string", "bool"
    description: str = ""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    
    def validate_value(self, value: Any) -> bool:
        """验证参数值"""
        try:
            if self.type == "int":
                int_val = int(value)
                if self.min_value is not None and int_val < self.min_value:
                    return False
                if self.max_value is not None and int_val > self.max_value:
                    return False
            elif self.type == "double":
                float_val = float(value)
                if self.min_value is not None and float_val < self.min_value:
                    return False
                if self.max_value is not None and float_val > self.max_value:
                    return False
            elif self.type == "bool":
                if value not in [True, False, "true", "false", "1", "0", 1, 0]:
                    return False
            return True
        except (ValueError, TypeError):
            return False
    
    def format_value(self, value: Any = None) -> str:
        """格式化参数值为SET文件格式"""
        val = value if value is not None else self.value
        
        if self.type == "bool":
            if isinstance(val, bool):
                return "true" if val else "false"
            elif str(val).lower() in ["true", "1"]:
                return "true"
            else:
                return "false"
        elif self.type == "string":
            return f'"{str(val)}"'
        else:
            return str(val)


@dataclass
class StrategyTemplate:
    """策略模板"""
    name: str
    description: str
    symbol_types: List[str]  # ["forex", "metals", "indices", "crypto"]
    default_parameters: Dict[str, Any] = field(default_factory=dict)
    parameter_ranges: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)  # (min, max, step)
    
    def create_config(self, symbol: str, custom_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """基于模板创建配置"""
        config = self.default_parameters.copy()
        if custom_params:
            config.update(custom_params)
        return config


class StrategyConfigManager:
    """策略配置管理器"""
    
    def __init__(self, config_dir: str = "ea31337/sets"):
        """
        初始化配置管理器
        
        Args:
            config_dir: 配置文件目录
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # 策略模板
        self.templates = self._load_default_templates()
        
        # 参数缓存
        self._config_cache = {}
        
        logger.info(f"策略配置管理器初始化完成，配置目录: {self.config_dir}")
    
    def _load_default_templates(self) -> Dict[str, StrategyTemplate]:
        """加载默认策略模板"""
        templates = {}
        
        # 趋势跟踪策略模板
        templates["trend_following"] = StrategyTemplate(
            name="trend_following",
            description="趋势跟踪策略，适用于主要货币对",
            symbol_types=["forex", "metals"],
            default_parameters={
                "Lots": 0.01,
                "TakeProfit": 100,
                "StopLoss": 50,
                "TrailingStop": 30,
                "MaxSpread": 3.0,
                "MA_Period_Fast": 12,
                "MA_Period_Slow": 26,
                "MA_Method": 1,  # EMA
                "Signal_OpenMethod": 1,
                "Signal_CloseMethod": 1,
                "PriceStopMethod": 1,
                "PriceStopLevel": 2,
                "MaxRisk": 2.0
            },
            parameter_ranges={
                "Lots": (0.01, 1.0, 0.01),
                "TakeProfit": (50, 500, 10),
                "StopLoss": (20, 200, 5),
                "MA_Period_Fast": (5, 50, 1),
                "MA_Period_Slow": (10, 100, 1),
                "MaxRisk": (0.5, 5.0, 0.1)
            }
        )
        
        # 剥头皮策略模板
        templates["scalping"] = StrategyTemplate(
            name="scalping",
            description="剥头皮策略，适用于低点差环境",
            symbol_types=["forex"],
            default_parameters={
                "Lots": 0.01,
                "TakeProfit": 20,
                "StopLoss": 15,
                "MaxSpread": 1.5,
                "RSI_Period": 14,
                "RSI_Applied_Price": 0,
                "Signal_OpenMethod": 2,
                "Signal_CloseMethod": 2,
                "MaxRisk": 1.0
            },
            parameter_ranges={
                "Lots": (0.01, 0.5, 0.01),
                "TakeProfit": (10, 50, 2),
                "StopLoss": (5, 30, 1),
                "RSI_Period": (7, 21, 1),
                "MaxRisk": (0.5, 2.0, 0.1)
            }
        )
        
        # 突破策略模板
        templates["breakout"] = StrategyTemplate(
            name="breakout",
            description="突破策略，适用于波动性较大的品种",
            symbol_types=["metals", "indices", "crypto"],
            default_parameters={
                "Lots": 0.01,
                "TakeProfit": 200,
                "StopLoss": 100,
                "MaxSpread": 5.0,
                "BB_Period": 20,
                "BB_Deviation": 2.0,
                "ATR_Period": 14,
                "Signal_OpenMethod": 3,
                "Signal_CloseMethod": 3,
                "MaxRisk": 3.0
            },
            parameter_ranges={
                "Lots": (0.01, 2.0, 0.01),
                "TakeProfit": (100, 1000, 25),
                "StopLoss": (50, 500, 10),
                "BB_Period": (10, 50, 2),
                "BB_Deviation": (1.0, 3.0, 0.1),
                "MaxRisk": (1.0, 5.0, 0.1)
            }
        )
        
        return templates
    
    def load_config(self, strategy_name: str) -> Dict[str, Any]:
        """
        加载策略配置
        
        Args:
            strategy_name: 策略名称
            
        Returns:
            配置参数字典
        """
        try:
            # 检查缓存
            if strategy_name in self._config_cache:
                return self._config_cache[strategy_name].copy()
            
            set_file = self.config_dir / f"{strategy_name}.set"
            if not set_file.exists():
                logger.warning(f"配置文件不存在: {set_file}")
                return {}
            
            config = self._parse_set_file(set_file)
            self._config_cache[strategy_name] = config
            
            logger.debug(f"加载策略配置: {strategy_name}, 参数数量: {len(config)}")
            return config.copy()
            
        except Exception as e:
            logger.error(f"加载策略配置失败: {strategy_name}, 错误: {e}")
            raise ConfigurationException(f"加载策略配置失败: {e}")
    
    def save_config(self, strategy_name: str, config: Dict[str, Any]) -> bool:
        """
        保存策略配置
        
        Args:
            strategy_name: 策略名称
            config: 配置参数字典
            
        Returns:
            保存是否成功
        """
        try:
            set_file = self.config_dir / f"{strategy_name}.set"
            
            # 备份现有文件
            if set_file.exists():
                backup_file(set_file)
            
            # 生成SET文件内容
            set_content = self._generate_set_content(config)
            
            # 写入文件
            success = safe_write_file(set_file, set_content)
            
            if success:
                # 更新缓存
                self._config_cache[strategy_name] = config.copy()
                logger.info(f"保存策略配置成功: {strategy_name}")
                return True
            else:
                logger.error(f"保存策略配置失败: {strategy_name}")
                return False
                
        except Exception as e:
            logger.error(f"保存策略配置异常: {strategy_name}, 错误: {e}")
            return False
    
    def _parse_set_file(self, set_file: Path) -> Dict[str, Any]:
        """解析SET文件"""
        config = {}
        
        content = safe_read_file(set_file)
        if not content:
            return config
        
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if not line or line.startswith('//') or line.startswith(';'):
                continue
            
            # 解析参数行: ParamName=Value
            if '=' in line:
                parts = line.split('=', 1)
                if len(parts) == 2:
                    param_name = parts[0].strip()
                    param_value = parts[1].strip()
                    
                    # 解析值类型
                    parsed_value = self._parse_parameter_value(param_value)
                    config[param_name] = parsed_value
        
        return config
    
    def _parse_parameter_value(self, value_str: str) -> Any:
        """解析参数值"""
        value_str = value_str.strip()
        
        # 布尔值
        if value_str.lower() in ['true', 'false']:
            return value_str.lower() == 'true'
        
        # 字符串值（带引号）
        if value_str.startswith('"') and value_str.endswith('"'):
            return value_str[1:-1]
        
        # 数值
        try:
            # 尝试整数
            if '.' not in value_str:
                return int(value_str)
            else:
                return float(value_str)
        except ValueError:
            # 默认返回字符串
            return value_str
    
    def _generate_set_content(self, config: Dict[str, Any]) -> str:
        """生成SET文件内容"""
        lines = [
            f"// EA31337策略配置文件",
            f"// 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"// 参数数量: {len(config)}",
            ""
        ]
        
        # 按参数名排序
        sorted_params = sorted(config.items())
        
        for param_name, param_value in sorted_params:
            # 格式化参数值
            if isinstance(param_value, bool):
                formatted_value = "true" if param_value else "false"
            elif isinstance(param_value, str):
                formatted_value = f'"{param_value}"'
            else:
                formatted_value = str(param_value)
            
            lines.append(f"{param_name}={formatted_value}")
        
        return '\n'.join(lines)
    
    def create_from_template(self, template_name: str, strategy_name: str, 
                           symbol: str, custom_params: Dict[str, Any] = None) -> bool:
        """
        基于模板创建策略配置
        
        Args:
            template_name: 模板名称
            strategy_name: 新策略名称
            symbol: 交易品种
            custom_params: 自定义参数
            
        Returns:
            创建是否成功
        """
        try:
            if template_name not in self.templates:
                raise ConfigurationException(f"模板不存在: {template_name}")
            
            template = self.templates[template_name]
            config = template.create_config(symbol, custom_params)
            
            # 根据品种调整参数
            config = self._adjust_config_for_symbol(config, symbol)
            
            return self.save_config(strategy_name, config)
            
        except Exception as e:
            logger.error(f"基于模板创建配置失败: {e}")
            return False
    
    def _adjust_config_for_symbol(self, config: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """根据品种调整配置参数"""
        adjusted_config = config.copy()
        
        # 根据品种类型调整参数
        if symbol.startswith("XAU") or symbol.startswith("XAG"):  # 贵金属
            adjusted_config["MaxSpread"] = adjusted_config.get("MaxSpread", 3.0) * 2
            adjusted_config["TakeProfit"] = int(adjusted_config.get("TakeProfit", 100) * 2)
            adjusted_config["StopLoss"] = int(adjusted_config.get("StopLoss", 50) * 1.5)
        elif symbol.endswith("JPY"):  # 日元对
            adjusted_config["TakeProfit"] = int(adjusted_config.get("TakeProfit", 100) * 100)
            adjusted_config["StopLoss"] = int(adjusted_config.get("StopLoss", 50) * 100)
        elif "OIL" in symbol or "WTI" in symbol:  # 原油
            adjusted_config["MaxSpread"] = adjusted_config.get("MaxSpread", 3.0) * 3
            adjusted_config["TakeProfit"] = int(adjusted_config.get("TakeProfit", 100) * 3)
            adjusted_config["StopLoss"] = int(adjusted_config.get("StopLoss", 50) * 2)
        
        return adjusted_config
    
    def optimize_parameters(self, strategy_name: str, symbol: str, 
                          optimization_results: Dict[str, float]) -> Dict[str, Any]:
        """
        基于优化结果更新策略参数
        
        Args:
            strategy_name: 策略名称
            symbol: 交易品种
            optimization_results: 优化结果 {param_name: optimal_value}
            
        Returns:
            优化后的配置
        """
        try:
            current_config = self.load_config(strategy_name)
            
            # 应用优化结果
            for param_name, optimal_value in optimization_results.items():
                if param_name in current_config:
                    # 验证参数范围
                    if self._validate_parameter_range(param_name, optimal_value):
                        current_config[param_name] = optimal_value
                        logger.debug(f"参数优化: {param_name} = {optimal_value}")
                    else:
                        logger.warning(f"参数值超出范围: {param_name} = {optimal_value}")
            
            return current_config
            
        except Exception as e:
            logger.error(f"参数优化失败: {e}")
            return {}
    
    def _validate_parameter_range(self, param_name: str, value: Any) -> bool:
        """验证参数范围"""
        # 基本范围检查
        if param_name == "Lots" and (value < 0.01 or value > 100):
            return False
        elif param_name == "MaxSpread" and (value < 0 or value > 50):
            return False
        elif param_name in ["TakeProfit", "StopLoss"] and (value < 0 or value > 10000):
            return False
        elif param_name == "MaxRisk" and (value < 0.1 or value > 10):
            return False
        
        return True
    
    def get_template_list(self) -> List[str]:
        """获取可用模板列表"""
        return list(self.templates.keys())
    
    def get_config_list(self) -> List[str]:
        """获取已有配置列表"""
        try:
            set_files = list(self.config_dir.glob("*.set"))
            return [f.stem for f in set_files]
        except Exception as e:
            logger.error(f"获取配置列表失败: {e}")
            return []
    
    def delete_config(self, strategy_name: str) -> bool:
        """删除策略配置"""
        try:
            set_file = self.config_dir / f"{strategy_name}.set"
            if set_file.exists():
                # 备份后删除
                backup_file(set_file)
                set_file.unlink()
                
                # 清除缓存
                if strategy_name in self._config_cache:
                    del self._config_cache[strategy_name]
                
                logger.info(f"删除策略配置: {strategy_name}")
                return True
            else:
                logger.warning(f"配置文件不存在: {strategy_name}")
                return False
                
        except Exception as e:
            logger.error(f"删除策略配置失败: {e}")
            return False
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """
        验证配置参数
        
        Args:
            config: 配置参数
            
        Returns:
            验证错误列表
        """
        errors = []
        
        # 必需参数检查
        required_params = ["Lots", "TakeProfit", "StopLoss"]
        for param in required_params:
            if param not in config:
                errors.append(f"缺少必需参数: {param}")
        
        # 参数值检查
        if "Lots" in config:
            lots = config["Lots"]
            if not isinstance(lots, (int, float)) or lots <= 0:
                errors.append("Lots必须为正数")
        
        if "MaxSpread" in config:
            spread = config["MaxSpread"]
            if not isinstance(spread, (int, float)) or spread < 0:
                errors.append("MaxSpread不能为负数")
        
        return errors
    
    def clear_cache(self) -> None:
        """清除配置缓存"""
        self._config_cache.clear()
        logger.debug("配置缓存已清除")