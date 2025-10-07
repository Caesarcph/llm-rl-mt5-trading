"""
品种配置管理器模块
管理各交易品种的专门配置，包括交易时段、风险参数、策略配置等
"""

import os
import yaml
from typing import Dict, List, Optional, Tuple
from datetime import datetime, time
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class TradingHours:
    """交易时段配置"""
    monday: str = "00:00-23:59"
    tuesday: str = "00:00-23:59"
    wednesday: str = "00:00-23:59"
    thursday: str = "00:00-23:59"
    friday: str = "00:00-21:00"
    saturday: str = ""
    sunday: str = ""
    
    def is_trading_time(self, dt: datetime) -> bool:
        """检查指定时间是否在交易时段内"""
        weekday = dt.strftime('%A').lower()
        hours_str = getattr(self, weekday, "")
        
        if not hours_str:
            return False
        
        try:
            start_str, end_str = hours_str.split('-')
            start_time = datetime.strptime(start_str, '%H:%M').time()
            end_time = datetime.strptime(end_str, '%H:%M').time()
            current_time = dt.time()
            
            return start_time <= current_time <= end_time
        except Exception as e:
            logger.error(f"解析交易时段失败: {e}")
            return False


@dataclass
class RiskParams:
    """风险参数配置"""
    max_spread: float = 3.0
    min_equity: float = 1000.0
    max_slippage: int = 3
    stop_loss_pips: int = 20
    take_profit_pips: int = 40


@dataclass
class EventMonitoring:
    """事件监控配置"""
    enabled: bool = False
    events: List[str] = field(default_factory=list)
    pre_event_stop: int = 30  # 事件前停止交易的分钟数
    post_event_wait: int = 15  # 事件后等待的分钟数


@dataclass
class EIAMonitoring:
    """EIA库存数据监控配置"""
    enabled: bool = False
    release_time: str = "15:30"  # UTC时间
    pre_release_stop: int = 30
    post_release_wait: int = 15


@dataclass
class OPECMonitoring:
    """OPEC会议监控配置"""
    enabled: bool = False
    meeting_dates: List[str] = field(default_factory=list)
    pre_meeting_stop: int = 120
    post_meeting_wait: int = 60


@dataclass
class SymbolConfig:
    """品种配置数据类"""
    symbol: str
    spread_limit: float
    min_lot: float
    max_lot: float
    lot_step: float = 0.01
    risk_multiplier: float = 1.0
    strategies: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=list)
    trading_hours: TradingHours = field(default_factory=TradingHours)
    optimize_params: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    risk_params: RiskParams = field(default_factory=RiskParams)
    event_monitoring: Optional[EventMonitoring] = None
    eia_monitoring: Optional[EIAMonitoring] = None
    opec_monitoring: Optional[OPECMonitoring] = None
    
    def is_trading_time(self, current_time: Optional[datetime] = None) -> bool:
        """检查是否在交易时间内"""
        if current_time is None:
            current_time = datetime.utcnow()
        return self.trading_hours.is_trading_time(current_time)
    
    def get_optimized_params(self, strategy: str) -> Dict[str, float]:
        """获取策略的优化参数范围"""
        return {
            key: value for key, value in self.optimize_params.items()
            if strategy.lower() in key.lower()
        }
    
    def should_stop_trading_for_event(self, current_time: Optional[datetime] = None) -> Tuple[bool, str]:
        """检查是否应该因事件停止交易"""
        if current_time is None:
            current_time = datetime.utcnow()
        
        # 检查EIA库存数据发布
        if self.eia_monitoring and self.eia_monitoring.enabled:
            if self._is_near_eia_release(current_time):
                return True, "EIA库存数据发布临近"
        
        # 检查OPEC会议
        if self.opec_monitoring and self.opec_monitoring.enabled:
            if self._is_near_opec_meeting(current_time):
                return True, "OPEC会议临近"
        
        return False, ""
    
    def _is_near_eia_release(self, current_time: datetime) -> bool:
        """检查是否临近EIA发布时间"""
        # 每周三发布
        if current_time.weekday() != 2:  # 2 = Wednesday
            return False
        
        try:
            release_time = datetime.strptime(
                self.eia_monitoring.release_time, '%H:%M'
            ).time()
            current_time_only = current_time.time()
            
            # 计算时间差（分钟）
            release_minutes = release_time.hour * 60 + release_time.minute
            current_minutes = current_time_only.hour * 60 + current_time_only.minute
            diff = release_minutes - current_minutes
            
            # 在发布前后的停止时段内
            if -self.eia_monitoring.post_release_wait <= diff <= self.eia_monitoring.pre_release_stop:
                return True
        except Exception as e:
            logger.error(f"检查EIA发布时间失败: {e}")
        
        return False
    
    def _is_near_opec_meeting(self, current_time: datetime) -> bool:
        """检查是否临近OPEC会议"""
        if not self.opec_monitoring.meeting_dates:
            return False
        
        for meeting_date_str in self.opec_monitoring.meeting_dates:
            try:
                meeting_date = datetime.strptime(meeting_date_str, '%Y-%m-%d')
                diff_hours = (meeting_date - current_time).total_seconds() / 3600
                
                # 在会议前后的停止时段内
                pre_hours = self.opec_monitoring.pre_meeting_stop / 60
                post_hours = self.opec_monitoring.post_meeting_wait / 60
                
                if -post_hours <= diff_hours <= pre_hours:
                    return True
            except Exception as e:
                logger.error(f"检查OPEC会议时间失败: {e}")
        
        return False


class SymbolConfigManager:
    """品种配置管理器"""
    
    def __init__(self, config_dir: str = "config/symbols"):
        """
        初始化品种配置管理器
        
        Args:
            config_dir: 配置文件目录路径
        """
        self.config_dir = Path(config_dir)
        self.configs: Dict[str, SymbolConfig] = {}
        self._load_all_configs()
        logger.info(f"品种配置管理器初始化完成，加载了 {len(self.configs)} 个品种配置")
    
    def _load_all_configs(self) -> None:
        """加载所有品种配置文件"""
        if not self.config_dir.exists():
            logger.warning(f"配置目录不存在: {self.config_dir}")
            return
        
        for config_file in self.config_dir.glob("*.yaml"):
            try:
                symbol_config = self._load_config_file(config_file)
                self.configs[symbol_config.symbol] = symbol_config
                logger.info(f"加载品种配置: {symbol_config.symbol}")
            except Exception as e:
                logger.error(f"加载配置文件失败 {config_file}: {e}")
    
    def _load_config_file(self, config_path: Path) -> SymbolConfig:
        """从文件加载品种配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # 解析交易时段
        trading_hours_data = data.get('trading_hours', {})
        trading_hours = TradingHours(**trading_hours_data)
        
        # 解析风险参数
        risk_params_data = data.get('risk_params', {})
        risk_params = RiskParams(**risk_params_data)
        
        # 解析优化参数
        optimize_params = {}
        for key, value in data.get('optimize_params', {}).items():
            if isinstance(value, list) and len(value) == 3:
                optimize_params[key] = tuple(value)
        
        # 解析事件监控
        event_monitoring = None
        if 'event_monitoring' in data:
            event_monitoring = EventMonitoring(
                enabled=True,
                events=data['event_monitoring']
            )
        
        # 解析EIA监控
        eia_monitoring = None
        if 'eia_monitoring' in data:
            eia_data = data['eia_monitoring']
            eia_monitoring = EIAMonitoring(**eia_data)
        
        # 解析OPEC监控
        opec_monitoring = None
        if 'opec_monitoring' in data:
            opec_data = data['opec_monitoring']
            opec_monitoring = OPECMonitoring(**opec_data)
        
        return SymbolConfig(
            symbol=data['symbol'],
            spread_limit=data['spread_limit'],
            min_lot=data['min_lot'],
            max_lot=data['max_lot'],
            lot_step=data.get('lot_step', 0.01),
            risk_multiplier=data.get('risk_multiplier', 1.0),
            strategies=data.get('strategies', []),
            timeframes=data.get('timeframes', []),
            trading_hours=trading_hours,
            optimize_params=optimize_params,
            risk_params=risk_params,
            event_monitoring=event_monitoring,
            eia_monitoring=eia_monitoring,
            opec_monitoring=opec_monitoring
        )
    
    def get_config(self, symbol: str) -> Optional[SymbolConfig]:
        """
        获取指定品种的配置
        
        Args:
            symbol: 品种代码
            
        Returns:
            品种配置对象，如果不存在则返回None
        """
        return self.configs.get(symbol)
    
    def get_all_symbols(self) -> List[str]:
        """获取所有已配置的品种列表"""
        return list(self.configs.keys())
    
    def is_symbol_configured(self, symbol: str) -> bool:
        """检查品种是否已配置"""
        return symbol in self.configs
    
    def get_active_symbols(self, current_time: Optional[datetime] = None) -> List[str]:
        """
        获取当前时间可交易的品种列表
        
        Args:
            current_time: 当前时间，默认为UTC当前时间
            
        Returns:
            可交易的品种列表
        """
        if current_time is None:
            current_time = datetime.utcnow()
        
        active_symbols = []
        for symbol, config in self.configs.items():
            if config.is_trading_time(current_time):
                should_stop, _ = config.should_stop_trading_for_event(current_time)
                if not should_stop:
                    active_symbols.append(symbol)
        
        return active_symbols
    
    def get_risk_multiplier(self, symbol: str) -> float:
        """获取品种的风险倍数"""
        config = self.get_config(symbol)
        return config.risk_multiplier if config else 1.0
    
    def get_strategies_for_symbol(self, symbol: str) -> List[str]:
        """获取品种启用的策略列表"""
        config = self.get_config(symbol)
        return config.strategies if config else []
    
    def validate_lot_size(self, symbol: str, lot_size: float) -> Tuple[bool, float]:
        """
        验证并调整手数大小
        
        Args:
            symbol: 品种代码
            lot_size: 请求的手数
            
        Returns:
            (是否有效, 调整后的手数)
        """
        config = self.get_config(symbol)
        if not config:
            return False, 0.0
        
        # 调整到最小手数的倍数
        adjusted_lot = round(lot_size / config.lot_step) * config.lot_step
        
        # 限制在最小和最大手数之间
        adjusted_lot = max(config.min_lot, min(adjusted_lot, config.max_lot))
        
        return True, adjusted_lot
    
    def check_spread(self, symbol: str, current_spread: float) -> bool:
        """
        检查当前点差是否在允许范围内
        
        Args:
            symbol: 品种代码
            current_spread: 当前点差
            
        Returns:
            点差是否可接受
        """
        config = self.get_config(symbol)
        if not config:
            return False
        
        return current_spread <= config.spread_limit
    
    def save_config(self, symbol: str, config: SymbolConfig) -> bool:
        """
        保存品种配置到文件
        
        Args:
            symbol: 品种代码
            config: 品种配置对象
            
        Returns:
            是否保存成功
        """
        try:
            config_path = self.config_dir / f"{symbol.lower()}.yaml"
            
            # 构建配置字典
            config_dict = {
                'symbol': config.symbol,
                'spread_limit': config.spread_limit,
                'min_lot': config.min_lot,
                'max_lot': config.max_lot,
                'lot_step': config.lot_step,
                'risk_multiplier': config.risk_multiplier,
                'strategies': config.strategies,
                'timeframes': config.timeframes,
                'trading_hours': {
                    'monday': config.trading_hours.monday,
                    'tuesday': config.trading_hours.tuesday,
                    'wednesday': config.trading_hours.wednesday,
                    'thursday': config.trading_hours.thursday,
                    'friday': config.trading_hours.friday,
                },
                'optimize_params': {
                    key: list(value) for key, value in config.optimize_params.items()
                },
                'risk_params': {
                    'max_spread': config.risk_params.max_spread,
                    'min_equity': config.risk_params.min_equity,
                    'max_slippage': config.risk_params.max_slippage,
                    'stop_loss_pips': config.risk_params.stop_loss_pips,
                    'take_profit_pips': config.risk_params.take_profit_pips,
                }
            }
            
            # 添加事件监控配置
            if config.event_monitoring:
                config_dict['event_monitoring'] = config.event_monitoring.events
            
            if config.eia_monitoring:
                config_dict['eia_monitoring'] = {
                    'enabled': config.eia_monitoring.enabled,
                    'release_time': config.eia_monitoring.release_time,
                    'pre_release_stop': config.eia_monitoring.pre_release_stop,
                    'post_release_wait': config.eia_monitoring.post_release_wait,
                }
            
            if config.opec_monitoring:
                config_dict['opec_monitoring'] = {
                    'enabled': config.opec_monitoring.enabled,
                    'meeting_dates': config.opec_monitoring.meeting_dates,
                    'pre_meeting_stop': config.opec_monitoring.pre_meeting_stop,
                    'post_meeting_wait': config.opec_monitoring.post_meeting_wait,
                }
            
            # 保存到文件
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, allow_unicode=True, default_flow_style=False)
            
            # 更新内存中的配置
            self.configs[symbol] = config
            logger.info(f"品种配置已保存: {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"保存品种配置失败 {symbol}: {e}")
            return False
    
    def create_default_config(self, symbol: str) -> SymbolConfig:
        """
        创建默认品种配置
        
        Args:
            symbol: 品种代码
            
        Returns:
            默认配置对象
        """
        return SymbolConfig(
            symbol=symbol,
            spread_limit=3.0,
            min_lot=0.01,
            max_lot=10.0,
            lot_step=0.01,
            risk_multiplier=1.0,
            strategies=["ma_cross", "macd"],
            timeframes=["M5", "M15", "H1", "H4"],
            trading_hours=TradingHours(),
            optimize_params={},
            risk_params=RiskParams()
        )
