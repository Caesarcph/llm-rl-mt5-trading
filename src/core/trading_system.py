#!/usr/bin/env python3
"""
主交易系统类
集成所有模块，实现系统启动、运行和关闭流程
"""

import asyncio
import signal
import sys
import threading
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from enum import Enum
from dataclasses import dataclass, field

from src.core.logging import get_logger
from src.core.config import get_config
from src.core.exceptions import TradingSystemException
from src.data.mt5_connection import MT5Connection, ConnectionConfig, ConnectionStatus
from src.data.pipeline import DataPipeline
from src.strategies.strategy_manager import StrategyManager
from src.core.order_executor import OrderExecutor
from src.core.position_manager import PositionManager
from src.core.risk_control_system import RiskControlSystem
from src.core.fund_manager import FundManager
from src.core.monitoring import MonitoringSystem
from src.core.alert_system import AlertSystem
from src.agents.market_analyst import MarketAnalystAgent
from src.agents.risk_manager import RiskManagerAgent
from src.agents.execution_optimizer import ExecutionOptimizerAgent
from src.bridge.ea31337_bridge import EA31337Bridge


class SystemState(Enum):
    """系统状态枚举"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"
    RECOVERY = "recovery"


@dataclass
class SystemStats:
    """系统统计信息"""
    start_time: Optional[datetime] = None
    uptime: timedelta = timedelta()
    total_signals: int = 0
    total_trades: int = 0
    successful_trades: int = 0
    failed_trades: int = 0
    total_profit: float = 0.0
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    errors_count: int = 0
    recovery_count: int = 0
    last_error: Optional[str] = None
    last_recovery_time: Optional[datetime] = None


class TradingSystem:
    """
    主交易系统类
    集成所有模块，提供统一的系统管理接口
    """
    
    def __init__(self):
        """初始化交易系统"""
        self.logger = get_logger()
        self.config = get_config()
        
        # 系统状态
        self._state = SystemState.STOPPED
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        
        # 系统统计
        self.stats = SystemStats()
        
        # 核心组件（延迟初始化）
        self.mt5_connection: Optional[MT5Connection] = None
        self.data_pipeline: Optional[DataPipeline] = None
        self.strategy_manager: Optional[StrategyManager] = None
        self.order_executor: Optional[OrderExecutor] = None
        self.position_manager: Optional[PositionManager] = None
        self.risk_control: Optional[RiskControlSystem] = None
        self.fund_manager: Optional[FundManager] = None
        self.monitoring: Optional[MonitoringSystem] = None
        self.alert_system: Optional[AlertSystem] = None
        
        # Agent组件
        self.market_analyst: Optional[MarketAnalystAgent] = None
        self.risk_manager_agent: Optional[RiskManagerAgent] = None
        self.execution_optimizer: Optional[ExecutionOptimizerAgent] = None
        
        # EA31337集成
        self.ea31337_bridge: Optional[EA31337Bridge] = None
        
        # 主循环任务
        self._main_loop_task: Optional[asyncio.Task] = None
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # 信号处理
        self._setup_signal_handlers()
        
        self.logger.info("交易系统对象创建完成")
    
    @property
    def state(self) -> SystemState:
        """获取当前系统状态"""
        with self._state_lock:
            return self._state
    
    @state.setter
    def state(self, new_state: SystemState):
        """设置系统状态"""
        with self._state_lock:
            old_state = self._state
            self._state = new_state
            self.logger.info(f"系统状态变更: {old_state.value} -> {new_state.value}")
    
    def _setup_signal_handlers(self):
        """设置系统信号处理器"""
        def signal_handler(signum, frame):
            self.logger.warning(f"接收到信号 {signum}，准备关闭系统...")
            self._stop_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def initialize(self) -> bool:
        """
        初始化系统所有组件
        
        Returns:
            初始化是否成功
        """
        try:
            self.state = SystemState.STARTING
            self.logger.info("=" * 60)
            self.logger.info("开始初始化交易系统...")
            self.logger.info("=" * 60)
            
            # 1. 初始化MT5连接
            self.logger.info("1/10 初始化MT5连接...")
            self.mt5_connection = self._initialize_mt5_connection()
            if not self.mt5_connection.is_connected:
                raise TradingSystemException("MT5连接失败")
            
            # 2. 初始化数据管道
            self.logger.info("2/10 初始化数据管道...")
            self.data_pipeline = DataPipeline(self.mt5_connection)
            
            # 3. 初始化监控系统
            self.logger.info("3/10 初始化监控系统...")
            self.monitoring = MonitoringSystem(self.config.monitoring)
            await self.monitoring.start()
            
            # 4. 初始化告警系统
            self.logger.info("4/10 初始化告警系统...")
            self.alert_system = AlertSystem(self.config.alert)
            await self.alert_system.initialize()
            
            # 5. 初始化资金管理器
            self.logger.info("5/10 初始化资金管理器...")
            self.fund_manager = FundManager(self.config.fund_management)
            
            # 6. 初始化风险控制系统
            self.logger.info("6/10 初始化风险控制系统...")
            self.risk_control = RiskControlSystem(
                self.config.risk_control,
                self.fund_manager
            )
            
            # 7. 初始化订单执行器
            self.logger.info("7/10 初始化订单执行器...")
            self.order_executor = OrderExecutor(self.config.order_execution)
            
            # 8. 初始化仓位管理器
            self.logger.info("8/10 初始化仓位管理器...")
            self.position_manager = PositionManager(self.config.position_management)
            
            # 9. 初始化策略管理器
            self.logger.info("9/10 初始化策略管理器...")
            self.strategy_manager = StrategyManager(
                conflict_resolution=self.config.strategy.conflict_resolution,
                aggregation_method=self.config.strategy.aggregation_method
            )
            self._register_strategies()
            
            # 10. 初始化Agent系统
            self.logger.info("10/10 初始化Agent系统...")
            self._initialize_agents()
            
            # 11. 初始化EA31337桥接（如果启用）
            if self.config.ea31337.enabled:
                self.logger.info("初始化EA31337桥接...")
                self.ea31337_bridge = EA31337Bridge(self.config.ea31337.config_path)
            
            self.stats.start_time = datetime.now()
            self.logger.info("=" * 60)
            self.logger.info("交易系统初始化完成！")
            self.logger.info("=" * 60)
            
            # 发送启动通知
            await self.alert_system.send_alert(
                "系统启动",
                "交易系统已成功初始化并准备运行",
                level="info"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"系统初始化失败: {e}", exc_info=True)
            self.state = SystemState.ERROR
            self.stats.errors_count += 1
            self.stats.last_error = str(e)
            
            # 发送错误告警
            if self.alert_system:
                await self.alert_system.send_alert(
                    "系统初始化失败",
                    f"错误: {e}",
                    level="critical"
                )
            
            return False
    
    def _initialize_mt5_connection(self) -> MT5Connection:
        """初始化MT5连接"""
        connection_config = ConnectionConfig(
            login=self.config.mt5.login,
            password=self.config.mt5.password,
            server=self.config.mt5.server,
            timeout=self.config.mt5.timeout,
            max_retries=self.config.mt5.max_retries,
            auto_reconnect=True
        )
        
        connection = MT5Connection(connection_config)
        connection.connect()
        
        return connection
    
    def _register_strategies(self):
        """注册所有策略"""
        # 注册基础策略
        from src.strategies.base_strategies import (
            TrendFollowingStrategy,
            MeanReversionStrategy,
            BreakoutStrategy
        )
        
        strategies_to_register = [
            ('trend_following', TrendFollowingStrategy),
            ('mean_reversion', MeanReversionStrategy),
            ('breakout', BreakoutStrategy),
        ]
        
        for name, strategy_class in strategies_to_register:
            if name in self.config.strategies.enabled:
                config = self.config.strategies.get(name, {})
                self.strategy_manager.register_strategy(
                    name=name,
                    strategy_class=strategy_class,
                    config=config
                )
                self.logger.info(f"已注册策略: {name}")
    
    def _initialize_agents(self):
        """初始化Agent系统"""
        # 市场分析Agent
        self.market_analyst = MarketAnalystAgent(
            data_pipeline=self.data_pipeline
        )
        
        # 风险管理Agent
        self.risk_manager_agent = RiskManagerAgent(
            risk_control=self.risk_control,
            position_manager=self.position_manager
        )
        
        # 执行优化Agent
        self.execution_optimizer = ExecutionOptimizerAgent(
            data_pipeline=self.data_pipeline
        )

    async def start(self):
        """启动交易系统"""
        try:
            # 初始化系统
            if not await self.initialize():
                raise TradingSystemException("系统初始化失败")
            
            self.state = SystemState.RUNNING
            self.logger.info("交易系统开始运行...")
            
            # 启动主循环
            self._main_loop_task = asyncio.create_task(self._main_loop())
            
            # 启动监控任务
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            # 等待停止信号
            await self._wait_for_stop()
            
        except Exception as e:
            self.logger.error(f"系统运行异常: {e}", exc_info=True)
            self.state = SystemState.ERROR
            self.stats.errors_count += 1
            self.stats.last_error = str(e)
            
            # 尝试恢复
            if self.config.system.auto_recovery:
                await self._attempt_recovery()
        
        finally:
            await self.shutdown()
    
    async def _main_loop(self):
        """主交易循环"""
        self.logger.info("主交易循环启动")
        
        while not self._stop_event.is_set():
            try:
                # 检查是否暂停
                if self._pause_event.is_set():
                    await asyncio.sleep(1)
                    continue
                
                # 检查交易时间
                if not self._is_trading_time():
                    await asyncio.sleep(60)
                    continue
                
                # 更新市场数据
                await self._update_market_data()
                
                # 更新持仓
                self.position_manager.update_positions()
                
                # 检查风险控制
                risk_status = self.risk_control.check_risk_limits()
                if not risk_status.can_trade:
                    self.logger.warning(f"风险控制触发: {risk_status.reason}")
                    await self.alert_system.send_alert(
                        "风险控制触发",
                        risk_status.reason,
                        level="warning"
                    )
                    await asyncio.sleep(60)
                    continue
                
                # 生成交易信号
                signals = await self._generate_signals()
                
                # 处理信号
                if signals:
                    await self._process_signals(signals)
                
                # 更新追踪止损
                self.position_manager.update_trailing_stops()
                
                # 更新统计
                self._update_stats()
                
                # 等待下一个周期
                await asyncio.sleep(self.config.system.loop_interval)
                
            except Exception as e:
                self.logger.error(f"主循环异常: {e}", exc_info=True)
                self.stats.errors_count += 1
                self.stats.last_error = str(e)
                
                # 短暂延迟后继续
                await asyncio.sleep(5)
        
        self.logger.info("主交易循环结束")
    
    async def _monitoring_loop(self):
        """监控循环"""
        self.logger.info("监控循环启动")
        
        while not self._stop_event.is_set():
            try:
                # 收集系统指标
                metrics = self._collect_metrics()
                
                # 更新监控系统
                self.monitoring.update_metrics(metrics)
                
                # 检查系统健康状态
                health_status = self.monitoring.check_health()
                if not health_status.is_healthy:
                    self.logger.warning(f"系统健康检查失败: {health_status.issues}")
                    await self.alert_system.send_alert(
                        "系统健康警告",
                        f"检测到问题: {health_status.issues}",
                        level="warning"
                    )
                
                # 等待下一个监控周期
                await asyncio.sleep(self.config.monitoring.interval)
                
            except Exception as e:
                self.logger.error(f"监控循环异常: {e}", exc_info=True)
                await asyncio.sleep(10)
        
        self.logger.info("监控循环结束")
    
    async def _update_market_data(self):
        """更新市场数据"""
        for symbol in self.config.symbols:
            try:
                market_data = self.data_pipeline.get_realtime_data(
                    symbol=symbol,
                    timeframe=self.config.default_timeframe
                )
                
                # 缓存数据
                self.data_pipeline.cache_data(
                    f"market_data:{symbol}",
                    market_data,
                    ttl=60
                )
                
            except Exception as e:
                self.logger.error(f"更新市场数据失败 {symbol}: {e}")
    
    async def _generate_signals(self) -> List:
        """生成交易信号"""
        all_signals = []
        
        for symbol in self.config.symbols:
            try:
                # 获取市场数据
                market_data = self.data_pipeline.get_cached_data(f"market_data:{symbol}")
                if not market_data:
                    continue
                
                # 市场分析
                market_state = self.market_analyst.analyze_market_state(symbol)
                
                # 策略生成信号
                signals = self.strategy_manager.generate_signals(market_data)
                
                # 执行优化
                for signal in signals:
                    optimized_signal = self.execution_optimizer.optimize_signal(
                        signal,
                        market_state
                    )
                    
                    # 风险验证
                    validation = self.risk_manager_agent.validate_trade(
                        optimized_signal,
                        self.position_manager.positions
                    )
                    
                    if validation.is_valid:
                        all_signals.append(optimized_signal)
                        self.stats.total_signals += 1
                    else:
                        self.logger.info(f"信号被拒绝: {validation.reason}")
                
            except Exception as e:
                self.logger.error(f"生成信号失败 {symbol}: {e}")
        
        return all_signals
    
    async def _process_signals(self, signals: List):
        """处理交易信号"""
        for signal in signals:
            try:
                # 执行订单
                result = self.order_executor.send_order(signal)
                
                if result['success']:
                    self.stats.total_trades += 1
                    self.stats.successful_trades += 1
                    self.logger.info(f"订单执行成功: {signal.symbol} {signal.direction}")
                    
                    await self.alert_system.send_alert(
                        "订单执行",
                        f"{signal.symbol} {signal.direction} {signal.size}手",
                        level="info"
                    )
                else:
                    self.stats.failed_trades += 1
                    self.logger.warning(f"订单执行失败: {result.get('error')}")
                
            except Exception as e:
                self.logger.error(f"处理信号异常: {e}", exc_info=True)
                self.stats.failed_trades += 1

    def _is_trading_time(self) -> bool:
        """检查是否在交易时间内"""
        if not self.config.trading_hours.enabled:
            return True
        
        now = datetime.now()
        current_time = now.time()
        current_day = now.weekday()
        
        # 检查是否在交易日
        if current_day in self.config.trading_hours.excluded_days:
            return False
        
        # 检查是否在交易时段
        for time_range in self.config.trading_hours.ranges:
            if time_range['start'] <= current_time <= time_range['end']:
                return True
        
        return False
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """收集系统指标"""
        metrics = {
            'timestamp': datetime.now(),
            'state': self.state.value,
            'uptime': datetime.now() - self.stats.start_time if self.stats.start_time else timedelta(),
            'total_signals': self.stats.total_signals,
            'total_trades': self.stats.total_trades,
            'successful_trades': self.stats.successful_trades,
            'failed_trades': self.stats.failed_trades,
            'success_rate': self.stats.successful_trades / max(self.stats.total_trades, 1),
            'total_profit': self.stats.total_profit,
            'current_drawdown': self.stats.current_drawdown,
            'max_drawdown': self.stats.max_drawdown,
            'errors_count': self.stats.errors_count,
            'recovery_count': self.stats.recovery_count,
            'active_positions': len(self.position_manager.positions) if self.position_manager else 0,
            'mt5_connected': self.mt5_connection.is_connected if self.mt5_connection else False,
        }
        
        return metrics
    
    def _update_stats(self):
        """更新系统统计"""
        if not self.position_manager:
            return
        
        # 计算总盈亏
        total_profit = sum(
            pos.profit for pos in self.position_manager.positions.values()
        )
        self.stats.total_profit = total_profit
        
        # 计算回撤
        if self.fund_manager:
            account_info = self.fund_manager.get_account_info()
            if account_info:
                equity = account_info.get('equity', 0)
                balance = account_info.get('balance', 0)
                
                if balance > 0:
                    drawdown = (balance - equity) / balance
                    self.stats.current_drawdown = drawdown
                    self.stats.max_drawdown = max(self.stats.max_drawdown, drawdown)
    
    async def _wait_for_stop(self):
        """等待停止信号"""
        while not self._stop_event.is_set():
            await asyncio.sleep(1)
    
    async def _attempt_recovery(self):
        """尝试系统恢复"""
        self.logger.info("尝试系统恢复...")
        self.state = SystemState.RECOVERY
        self.stats.recovery_count += 1
        self.stats.last_recovery_time = datetime.now()
        
        try:
            # 重新连接MT5
            if self.mt5_connection and not self.mt5_connection.is_connected:
                self.logger.info("重新连接MT5...")
                self.mt5_connection.reconnect()
            
            # 重新初始化关键组件
            if not self.data_pipeline:
                self.data_pipeline = DataPipeline(self.mt5_connection)
            
            # 等待一段时间
            await asyncio.sleep(10)
            
            # 检查恢复状态
            if self.mt5_connection.is_connected:
                self.logger.info("系统恢复成功")
                self.state = SystemState.RUNNING
                
                await self.alert_system.send_alert(
                    "系统恢复",
                    "系统已成功恢复运行",
                    level="info"
                )
                
                # 重启主循环
                self._main_loop_task = asyncio.create_task(self._main_loop())
            else:
                self.logger.error("系统恢复失败")
                self.state = SystemState.ERROR
                
        except Exception as e:
            self.logger.error(f"系统恢复异常: {e}", exc_info=True)
            self.state = SystemState.ERROR
    
    async def shutdown(self):
        """关闭系统"""
        self.logger.info("=" * 60)
        self.logger.info("开始关闭交易系统...")
        self.logger.info("=" * 60)
        
        self.state = SystemState.STOPPING
        
        try:
            # 取消所有异步任务
            if self._main_loop_task and not self._main_loop_task.done():
                self._main_loop_task.cancel()
                try:
                    await self._main_loop_task
                except asyncio.CancelledError:
                    pass
            
            if self._monitoring_task and not self._monitoring_task.done():
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass
            
            # 关闭所有持仓（如果配置要求）
            if self.config.system.close_positions_on_shutdown and self.position_manager:
                self.logger.info("关闭所有持仓...")
                for position in self.position_manager.positions.values():
                    try:
                        self.position_manager.close_position(position.position_id)
                    except Exception as e:
                        self.logger.error(f"关闭持仓失败 {position.position_id}: {e}")
            
            # 停止监控系统
            if self.monitoring:
                await self.monitoring.stop()
            
            # 发送关闭通知
            if self.alert_system:
                await self.alert_system.send_alert(
                    "系统关闭",
                    f"系统运行时长: {self.stats.uptime}, 总交易: {self.stats.total_trades}",
                    level="info"
                )
                await self.alert_system.close()
            
            # 断开MT5连接
            if self.mt5_connection:
                self.mt5_connection.disconnect()
            
            self.state = SystemState.STOPPED
            self.logger.info("=" * 60)
            self.logger.info("交易系统已安全关闭")
            self.logger.info("=" * 60)
            
        except Exception as e:
            self.logger.error(f"关闭系统异常: {e}", exc_info=True)
    
    def pause(self):
        """暂停系统"""
        self.logger.info("暂停交易系统...")
        self._pause_event.set()
        self.state = SystemState.PAUSED
    
    def resume(self):
        """恢复系统"""
        self.logger.info("恢复交易系统...")
        self._pause_event.clear()
        self.state = SystemState.RUNNING
    
    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'state': self.state.value,
            'stats': {
                'start_time': self.stats.start_time.isoformat() if self.stats.start_time else None,
                'uptime': str(self.stats.uptime),
                'total_signals': self.stats.total_signals,
                'total_trades': self.stats.total_trades,
                'successful_trades': self.stats.successful_trades,
                'failed_trades': self.stats.failed_trades,
                'success_rate': self.stats.successful_trades / max(self.stats.total_trades, 1),
                'total_profit': self.stats.total_profit,
                'current_drawdown': self.stats.current_drawdown,
                'max_drawdown': self.stats.max_drawdown,
                'errors_count': self.stats.errors_count,
                'recovery_count': self.stats.recovery_count,
            },
            'components': {
                'mt5_connected': self.mt5_connection.is_connected if self.mt5_connection else False,
                'active_positions': len(self.position_manager.positions) if self.position_manager else 0,
                'active_strategies': len(self.strategy_manager.strategies) if self.strategy_manager else 0,
            }
        }
