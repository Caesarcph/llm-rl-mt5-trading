#!/usr/bin/env python3
"""
数据获取和处理管道
支持实时和历史数据获取，包含数据验证和缓存机制
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from .mt5_connection import MT5Connection, ConnectionConfig
from .models import MarketData, Tick, TimeFrame, DataRequest, DataStats, ValidationResult
from .validator import DataValidator
from .cache import RedisCache, MemoryCache
from ..core.logging import get_logger
from ..core.exceptions import TradingSystemException


class DataPipelineError(TradingSystemException):
    """数据管道异常"""
    pass


class DataPipeline:
    """统一数据获取和处理管道"""
    
    def __init__(self, connection_config: ConnectionConfig, 
                 redis_config: Optional[Dict[str, Any]] = None):
        """
        初始化数据管道
        
        Args:
            connection_config: MT5连接配置
            redis_config: Redis配置
        """
        self.logger = get_logger()
        
        # 初始化组件
        self.mt5_connection = MT5Connection(connection_config)
        self.data_validator = DataValidator()
        
        # 初始化缓存
        if redis_config:
            self.cache = RedisCache(**redis_config)
            self.cache.connect()
        else:
            self.cache = MemoryCache()
        
        # 线程池用于异步数据处理
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="DataPipeline")
        
        # 数据统计
        self._stats = {
            'requests_total': 0,
            'requests_successful': 0,
            'requests_failed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'validation_errors': 0,
            'last_request_time': None
        }
        
        # 支持的时间周期映射
        self.timeframe_map = {
            TimeFrame.M1: mt5.TIMEFRAME_M1 if mt5 else None,
            TimeFrame.M5: mt5.TIMEFRAME_M5 if mt5 else None,
            TimeFrame.M15: mt5.TIMEFRAME_M15 if mt5 else None,
            TimeFrame.M30: mt5.TIMEFRAME_M30 if mt5 else None,
            TimeFrame.H1: mt5.TIMEFRAME_H1 if mt5 else None,
            TimeFrame.H4: mt5.TIMEFRAME_H4 if mt5 else None,
            TimeFrame.D1: mt5.TIMEFRAME_D1 if mt5 else None,
            TimeFrame.W1: mt5.TIMEFRAME_W1 if mt5 else None,
            TimeFrame.MN1: mt5.TIMEFRAME_MN1 if mt5 else None,
        }
    
    @property
    def stats(self) -> Dict[str, Any]:
        """获取数据管道统计信息"""
        stats = self._stats.copy()
        stats.update({
            'cache_stats': self.cache.stats if hasattr(self.cache, 'stats') else {},
            'connection_status': self.mt5_connection.status.value,
            'connection_stats': self.mt5_connection.connection_stats
        })
        return stats
    
    def get_realtime_data(self, symbol: str, timeframe: TimeFrame, 
                         count: int = 100, include_indicators: bool = True) -> Optional[MarketData]:
        """
        获取实时市场数据
        
        Args:
            symbol: 交易品种
            timeframe: 时间周期
            count: 数据条数
            include_indicators: 是否包含技术指标
            
        Returns:
            Optional[MarketData]: 市场数据，失败时返回None
        """
        request = DataRequest(
            symbol=symbol,
            timeframe=timeframe,
            count=count,
            include_indicators=include_indicators,
            cache_ttl=60  # 实时数据缓存1分钟
        )
        
        return self._get_market_data(request, is_realtime=True)
    
    def get_historical_data(self, symbol: str, timeframe: TimeFrame,
                          start_time: datetime, end_time: Optional[datetime] = None,
                          include_indicators: bool = True) -> Optional[MarketData]:
        """
        获取历史数据
        
        Args:
            symbol: 交易品种
            timeframe: 时间周期
            start_time: 开始时间
            end_time: 结束时间
            include_indicators: 是否包含技术指标
            
        Returns:
            Optional[MarketData]: 市场数据，失败时返回None
        """
        if end_time is None:
            end_time = datetime.now()
        
        request = DataRequest(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
            include_indicators=include_indicators,
            cache_ttl=3600  # 历史数据缓存1小时
        )
        
        return self._get_market_data(request, is_realtime=False)
    
    def get_tick_data(self, symbol: str, count: int = 100) -> Optional[List[Tick]]:
        """
        获取Tick数据
        
        Args:
            symbol: 交易品种
            count: 数据条数
            
        Returns:
            Optional[List[Tick]]: Tick数据列表
        """
        self._stats['requests_total'] += 1
        self._stats['last_request_time'] = datetime.now()
        
        try:
            # 检查连接
            if not self.mt5_connection.ensure_connection():
                raise DataPipelineError("MT5连接失败")
            
            # 获取Tick数据
            ticks = mt5.copy_ticks_from(symbol, datetime.now(), count, mt5.COPY_TICKS_ALL)
            
            if ticks is None or len(ticks) == 0:
                self.logger.warning(f"未获取到Tick数据: {symbol}")
                self._stats['requests_failed'] += 1
                return None
            
            # 转换为Tick对象列表
            tick_list = []
            for tick_data in ticks:
                tick = Tick(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(tick_data['time']),
                    bid=tick_data['bid'],
                    ask=tick_data['ask'],
                    last=tick_data['last'],
                    volume=tick_data['volume_real'],
                    flags=tick_data['flags']
                )
                
                # 验证Tick数据
                validation_result = self.data_validator.validate_tick_data(tick)
                if validation_result.is_valid:
                    tick_list.append(tick)
                else:
                    self.logger.warning(f"Tick数据验证失败: {validation_result.errors}")
            
            self._stats['requests_successful'] += 1
            self.logger.debug(f"获取Tick数据成功: {symbol}, {len(tick_list)} 条")
            
            return tick_list
            
        except Exception as e:
            self._stats['requests_failed'] += 1
            self.logger.error(f"获取Tick数据失败 {symbol}: {e}")
            return None
    
    def _get_market_data(self, request: DataRequest, is_realtime: bool = False) -> Optional[MarketData]:
        """
        内部方法：获取市场数据
        
        Args:
            request: 数据请求
            is_realtime: 是否为实时数据
            
        Returns:
            Optional[MarketData]: 市场数据
        """
        self._stats['requests_total'] += 1
        self._stats['last_request_time'] = datetime.now()
        
        try:
            # 生成缓存键
            cache_key = self._generate_cache_key(request)
            
            # 尝试从缓存获取
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                self._stats['cache_hits'] += 1
                self.logger.debug(f"缓存命中: {cache_key}")
                return cached_data
            
            self._stats['cache_misses'] += 1
            
            # 检查连接
            if not self.mt5_connection.ensure_connection():
                raise DataPipelineError("MT5连接失败")
            
            # 获取OHLCV数据
            ohlcv_data = self._fetch_ohlcv_data(request)
            if ohlcv_data is None or ohlcv_data.empty:
                self._stats['requests_failed'] += 1
                return None
            
            # 创建MarketData对象
            market_data = MarketData(
                symbol=request.symbol,
                timeframe=request.timeframe,
                timestamp=datetime.now(),
                ohlcv=ohlcv_data
            )
            
            # 获取当前点差
            if is_realtime:
                market_data.spread = self._get_current_spread(request.symbol)
            
            # 计算技术指标
            if request.include_indicators:
                market_data.indicators = self._calculate_indicators(ohlcv_data)
            
            # 验证数据
            validation_result = self.data_validator.validate_market_data(market_data)
            if not validation_result.is_valid:
                self._stats['validation_errors'] += 1
                self.logger.warning(f"数据验证失败: {validation_result.errors}")
                # 即使验证失败，也返回数据，但记录错误
            
            # 缓存数据
            self.cache.set(cache_key, market_data, request.cache_ttl)
            
            self._stats['requests_successful'] += 1
            self.logger.debug(f"获取市场数据成功: {request.symbol} {request.timeframe.value}")
            
            return market_data
            
        except Exception as e:
            self._stats['requests_failed'] += 1
            self.logger.error(f"获取市场数据失败 {request.symbol}: {e}")
            return None
    
    def _fetch_ohlcv_data(self, request: DataRequest) -> Optional[pd.DataFrame]:
        """
        获取OHLCV数据
        
        Args:
            request: 数据请求
            
        Returns:
            Optional[pd.DataFrame]: OHLCV数据
        """
        mt5_timeframe = self.timeframe_map.get(request.timeframe)
        if mt5_timeframe is None:
            raise DataPipelineError(f"不支持的时间周期: {request.timeframe}")
        
        try:
            if request.count is not None:
                # 按数量获取
                rates = mt5.copy_rates_from_pos(request.symbol, mt5_timeframe, 0, request.count)
            else:
                # 按时间范围获取
                rates = mt5.copy_rates_range(
                    request.symbol, 
                    mt5_timeframe, 
                    request.start_time, 
                    request.end_time
                )
            
            if rates is None or len(rates) == 0:
                self.logger.warning(f"未获取到OHLCV数据: {request.symbol}")
                return None
            
            # 转换为DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # 重命名列
            df.rename(columns={
                'open': 'open',
                'high': 'high', 
                'low': 'low',
                'close': 'close',
                'tick_volume': 'volume',
                'real_volume': 'real_volume'
            }, inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"获取OHLCV数据异常: {e}")
            return None
    
    def _get_current_spread(self, symbol: str) -> float:
        """
        获取当前点差
        
        Args:
            symbol: 交易品种
            
        Returns:
            float: 点差
        """
        try:
            symbol_info = mt5.symbol_info_tick(symbol)
            if symbol_info is not None:
                return symbol_info.ask - symbol_info.bid
        except Exception as e:
            self.logger.error(f"获取点差失败 {symbol}: {e}")
        
        return 0.0
    
    def _calculate_indicators(self, ohlcv: pd.DataFrame) -> Dict[str, float]:
        """
        计算技术指标
        
        Args:
            ohlcv: OHLCV数据
            
        Returns:
            Dict[str, float]: 技术指标字典
        """
        indicators = {}
        
        try:
            if len(ohlcv) < 20:
                return indicators
            
            # 移动平均线
            indicators['sma_20'] = ohlcv['close'].rolling(20).mean().iloc[-1]
            indicators['sma_50'] = ohlcv['close'].rolling(50).mean().iloc[-1] if len(ohlcv) >= 50 else None
            
            # 指数移动平均线
            indicators['ema_12'] = ohlcv['close'].ewm(span=12).mean().iloc[-1]
            indicators['ema_26'] = ohlcv['close'].ewm(span=26).mean().iloc[-1]
            
            # MACD
            if indicators['ema_12'] and indicators['ema_26']:
                indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
            
            # RSI
            delta = ohlcv['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs)).iloc[-1]
            
            # 布林带
            sma_20 = ohlcv['close'].rolling(20).mean()
            std_20 = ohlcv['close'].rolling(20).std()
            indicators['bb_upper'] = (sma_20 + 2 * std_20).iloc[-1]
            indicators['bb_lower'] = (sma_20 - 2 * std_20).iloc[-1]
            indicators['bb_middle'] = sma_20.iloc[-1]
            
            # 成交量指标
            indicators['volume_sma'] = ohlcv['volume'].rolling(20).mean().iloc[-1]
            
            # 过滤掉None值
            indicators = {k: v for k, v in indicators.items() if v is not None and not pd.isna(v)}
            
        except Exception as e:
            self.logger.error(f"计算技术指标失败: {e}")
        
        return indicators
    
    def _generate_cache_key(self, request: DataRequest) -> str:
        """
        生成缓存键
        
        Args:
            request: 数据请求
            
        Returns:
            str: 缓存键
        """
        key_parts = [
            'market_data',
            request.symbol,
            request.timeframe.value
        ]
        
        if request.count is not None:
            key_parts.append(f"count_{request.count}")
        
        if request.start_time is not None:
            key_parts.append(f"start_{request.start_time.strftime('%Y%m%d_%H%M%S')}")
        
        if request.end_time is not None:
            key_parts.append(f"end_{request.end_time.strftime('%Y%m%d_%H%M%S')}")
        
        key_parts.append(f"indicators_{request.include_indicators}")
        
        return ':'.join(key_parts)
    
    async def get_multiple_symbols_data(self, symbols: List[str], timeframe: TimeFrame,
                                      count: int = 100) -> Dict[str, Optional[MarketData]]:
        """
        异步获取多个品种的数据
        
        Args:
            symbols: 交易品种列表
            timeframe: 时间周期
            count: 数据条数
            
        Returns:
            Dict[str, Optional[MarketData]]: 品种数据字典
        """
        loop = asyncio.get_event_loop()
        
        # 创建异步任务
        tasks = []
        for symbol in symbols:
            task = loop.run_in_executor(
                self.executor,
                self.get_realtime_data,
                symbol, timeframe, count
            )
            tasks.append((symbol, task))
        
        # 等待所有任务完成
        results = {}
        for symbol, task in tasks:
            try:
                results[symbol] = await task
            except Exception as e:
                self.logger.error(f"获取 {symbol} 数据失败: {e}")
                results[symbol] = None
        
        return results
    
    def get_data_stats(self, symbol: str, timeframe: TimeFrame) -> Optional[DataStats]:
        """
        获取数据统计信息
        
        Args:
            symbol: 交易品种
            timeframe: 时间周期
            
        Returns:
            Optional[DataStats]: 数据统计
        """
        try:
            # 获取最近1000条数据来计算统计
            data = self.get_realtime_data(symbol, timeframe, count=1000, include_indicators=False)
            if data is None or data.ohlcv.empty:
                return None
            
            ohlcv = data.ohlcv
            
            stats = DataStats(
                symbol=symbol,
                timeframe=timeframe,
                total_bars=len(ohlcv),
                start_time=ohlcv.index[0],
                end_time=ohlcv.index[-1],
                missing_bars=0,  # TODO: 计算缺失的K线数量
                data_quality_score=1.0,  # TODO: 基于验证结果计算质量分数
                last_update=datetime.now()
            )
            
            return stats
            
        except Exception as e:
            self.logger.error(f"获取数据统计失败 {symbol}: {e}")
            return None
    
    def cleanup(self):
        """清理资源"""
        try:
            # 关闭线程池
            self.executor.shutdown(wait=True)
            
            # 断开缓存连接
            if hasattr(self.cache, 'disconnect'):
                self.cache.disconnect()
            
            # 断开MT5连接
            self.mt5_connection.disconnect()
            
            self.logger.info("数据管道资源清理完成")
            
        except Exception as e:
            self.logger.error(f"资源清理异常: {e}")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.cleanup()