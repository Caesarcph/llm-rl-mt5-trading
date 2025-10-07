#!/usr/bin/env python3
"""
SQLite数据库管理器
提供数据存储、备份和恢复功能
"""

import sqlite3
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union, Tuple
from pathlib import Path
import shutil
import threading
from contextlib import contextmanager

from .models import MarketData, Tick, TimeFrame, DataStats
from ..core.logging import get_logger
from ..core.exceptions import TradingSystemException


class DatabaseError(TradingSystemException):
    """数据库异常"""
    pass


class DatabaseManager:
    """SQLite数据库管理器"""
    
    def __init__(self, db_path: str = "data/trading.db"):
        """
        初始化数据库管理器
        
        Args:
            db_path: 数据库文件路径
        """
        self.logger = get_logger()
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 线程锁，确保数据库操作线程安全
        self._lock = threading.Lock()
        
        # 初始化数据库
        self._initialize_database()
        
        self.logger.info(f"数据库管理器初始化完成: {self.db_path}")
    
    def _initialize_database(self):
        """初始化数据库表结构"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 创建市场数据表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open_price REAL NOT NULL,
                    high_price REAL NOT NULL,
                    low_price REAL NOT NULL,
                    close_price REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    real_volume INTEGER DEFAULT 0,
                    spread REAL DEFAULT 0,
                    indicators TEXT,  -- JSON格式存储技术指标
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timeframe, timestamp)
                )
            ''')
            
            # 创建Tick数据表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tick_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    bid REAL NOT NULL,
                    ask REAL NOT NULL,
                    last_price REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    flags INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建交易记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT UNIQUE NOT NULL,
                    symbol TEXT NOT NULL,
                    trade_type TEXT NOT NULL,  -- BUY/SELL
                    volume REAL NOT NULL,
                    open_price REAL NOT NULL,
                    close_price REAL,
                    sl REAL,
                    tp REAL,
                    profit REAL DEFAULT 0,
                    commission REAL DEFAULT 0,
                    swap REAL DEFAULT 0,
                    open_time DATETIME NOT NULL,
                    close_time DATETIME,
                    strategy_id TEXT,
                    comment TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建策略参数表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategy_parameters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    parameters TEXT NOT NULL,  -- JSON格式存储参数
                    performance_metrics TEXT,  -- JSON格式存储性能指标
                    is_active BOOLEAN DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(strategy_name, symbol)
                )
            ''')
            
            # 创建账户信息表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS account_info (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id TEXT NOT NULL,
                    balance REAL NOT NULL,
                    equity REAL NOT NULL,
                    margin REAL NOT NULL,
                    free_margin REAL NOT NULL,
                    margin_level REAL NOT NULL,
                    currency TEXT NOT NULL,
                    leverage INTEGER NOT NULL,
                    timestamp DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建系统日志表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    level TEXT NOT NULL,
                    module TEXT NOT NULL,
                    message TEXT NOT NULL,
                    details TEXT,  -- JSON格式存储详细信息
                    timestamp DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建索引以提高查询性能
            self._create_indexes(cursor)
            
            conn.commit()
            self.logger.info("数据库表结构初始化完成")
    
    def _create_indexes(self, cursor):
        """创建数据库索引"""
        indexes = [
            # 市场数据索引
            "CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timeframe ON market_data(symbol, timeframe)",
            "CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data(timestamp)",
            
            # Tick数据索引
            "CREATE INDEX IF NOT EXISTS idx_tick_data_symbol ON tick_data(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_tick_data_timestamp ON tick_data(timestamp)",
            
            # 交易记录索引
            "CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_trades_open_time ON trades(open_time)",
            "CREATE INDEX IF NOT EXISTS idx_trades_strategy_id ON trades(strategy_id)",
            
            # 策略参数索引
            "CREATE INDEX IF NOT EXISTS idx_strategy_parameters_name ON strategy_parameters(strategy_name)",
            "CREATE INDEX IF NOT EXISTS idx_strategy_parameters_symbol ON strategy_parameters(symbol)",
            
            # 账户信息索引
            "CREATE INDEX IF NOT EXISTS idx_account_info_timestamp ON account_info(timestamp)",
            
            # 系统日志索引
            "CREATE INDEX IF NOT EXISTS idx_system_logs_timestamp ON system_logs(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_system_logs_level ON system_logs(level)"
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
    
    @contextmanager
    def _get_connection(self):
        """获取数据库连接的上下文管理器"""
        conn = None
        try:
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,
                check_same_thread=False
            )
            conn.row_factory = sqlite3.Row  # 使结果可以按列名访问
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise DatabaseError(f"数据库操作失败: {e}")
        finally:
            if conn:
                conn.close()
    
    def save_market_data(self, market_data: MarketData) -> bool:
        """
        保存市场数据
        
        Args:
            market_data: 市场数据对象
            
        Returns:
            bool: 保存是否成功
        """
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    
                    # 准备指标数据
                    indicators_json = json.dumps(market_data.indicators) if market_data.indicators else None
                    
                    # 批量插入OHLCV数据
                    for index, row in market_data.ohlcv.iterrows():
                        # 转换pandas Timestamp为Python datetime
                        timestamp = index.to_pydatetime() if hasattr(index, 'to_pydatetime') else index
                        
                        cursor.execute('''
                            INSERT OR REPLACE INTO market_data 
                            (symbol, timeframe, timestamp, open_price, high_price, low_price, 
                             close_price, volume, real_volume, spread, indicators)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            market_data.symbol,
                            market_data.timeframe.value,
                            timestamp,
                            row['open'],
                            row['high'],
                            row['low'],
                            row['close'],
                            row['volume'],
                            row.get('real_volume', 0),
                            market_data.spread,
                            indicators_json
                        ))
                    
                    conn.commit()
                    self.logger.debug(f"保存市场数据成功: {market_data.symbol} {market_data.timeframe.value}")
                    return True
                    
            except Exception as e:
                self.logger.error(f"保存市场数据失败: {e}")
                return False
    
    def save_tick_data(self, ticks: List[Tick]) -> bool:
        """
        保存Tick数据
        
        Args:
            ticks: Tick数据列表
            
        Returns:
            bool: 保存是否成功
        """
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    
                    # 批量插入Tick数据
                    tick_data = []
                    for tick in ticks:
                        # 确保timestamp是Python datetime对象
                        timestamp = tick.timestamp.to_pydatetime() if hasattr(tick.timestamp, 'to_pydatetime') else tick.timestamp
                        tick_data.append((
                            tick.symbol, timestamp, tick.bid, tick.ask, 
                            tick.last, tick.volume, tick.flags
                        ))
                    
                    cursor.executemany('''
                        INSERT INTO tick_data 
                        (symbol, timestamp, bid, ask, last_price, volume, flags)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', tick_data)
                    
                    conn.commit()
                    self.logger.debug(f"保存Tick数据成功: {len(ticks)} 条")
                    return True
                    
            except Exception as e:
                self.logger.error(f"保存Tick数据失败: {e}")
                return False
    
    def get_market_data(self, symbol: str, timeframe: TimeFrame, 
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None,
                       limit: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        获取市场数据
        
        Args:
            symbol: 交易品种
            timeframe: 时间周期
            start_time: 开始时间
            end_time: 结束时间
            limit: 限制条数
            
        Returns:
            Optional[pd.DataFrame]: 市场数据DataFrame
        """
        try:
            with self._get_connection() as conn:
                # 构建查询条件
                where_conditions = ["symbol = ? AND timeframe = ?"]
                params = [symbol, timeframe.value]
                
                if start_time:
                    where_conditions.append("timestamp >= ?")
                    params.append(start_time)
                
                if end_time:
                    where_conditions.append("timestamp <= ?")
                    params.append(end_time)
                
                where_clause = " AND ".join(where_conditions)
                
                # 构建完整查询
                query = f'''
                    SELECT timestamp, open_price as open, high_price as high, 
                           low_price as low, close_price as close, volume,
                           real_volume, spread, indicators
                    FROM market_data 
                    WHERE {where_clause}
                    ORDER BY timestamp DESC
                '''
                
                if limit:
                    query += f" LIMIT {limit}"
                
                df = pd.read_sql_query(query, conn, params=params, parse_dates=['timestamp'])
                
                if df.empty:
                    return None
                
                # 设置时间戳为索引
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                
                # 解析指标数据
                if 'indicators' in df.columns:
                    df['indicators'] = df['indicators'].apply(
                        lambda x: json.loads(x) if x else {}
                    )
                
                return df
                
        except Exception as e:
            self.logger.error(f"获取市场数据失败: {e}")
            return None
    
    def get_tick_data(self, symbol: str, start_time: Optional[datetime] = None,
                     end_time: Optional[datetime] = None,
                     limit: Optional[int] = None) -> Optional[List[Tick]]:
        """
        获取Tick数据
        
        Args:
            symbol: 交易品种
            start_time: 开始时间
            end_time: 结束时间
            limit: 限制条数
            
        Returns:
            Optional[List[Tick]]: Tick数据列表
        """
        try:
            with self._get_connection() as conn:
                # 构建查询条件
                where_conditions = ["symbol = ?"]
                params = [symbol]
                
                if start_time:
                    where_conditions.append("timestamp >= ?")
                    params.append(start_time)
                
                if end_time:
                    where_conditions.append("timestamp <= ?")
                    params.append(end_time)
                
                where_clause = " AND ".join(where_conditions)
                
                # 构建完整查询
                query = f'''
                    SELECT symbol, timestamp, bid, ask, last_price, volume, flags
                    FROM tick_data 
                    WHERE {where_clause}
                    ORDER BY timestamp DESC
                '''
                
                if limit:
                    query += f" LIMIT {limit}"
                
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                if not rows:
                    return None
                
                # 转换为Tick对象列表
                ticks = []
                for row in rows:
                    tick = Tick(
                        symbol=row['symbol'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        bid=row['bid'],
                        ask=row['ask'],
                        last=row['last_price'],
                        volume=row['volume'],
                        flags=row['flags']
                    )
                    ticks.append(tick)
                
                return ticks
                
        except Exception as e:
            self.logger.error(f"获取Tick数据失败: {e}")
            return None
    
    def save_trade_record(self, trade_data: Dict[str, Any]) -> bool:
        """
        保存交易记录
        
        Args:
            trade_data: 交易数据字典
            
        Returns:
            bool: 保存是否成功
        """
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    
                    # 确保时间戳是Python datetime对象
                    open_time = trade_data['open_time']
                    if hasattr(open_time, 'to_pydatetime'):
                        open_time = open_time.to_pydatetime()
                    
                    close_time = trade_data.get('close_time')
                    if close_time and hasattr(close_time, 'to_pydatetime'):
                        close_time = close_time.to_pydatetime()
                    
                    cursor.execute('''
                        INSERT OR REPLACE INTO trades 
                        (trade_id, symbol, trade_type, volume, open_price, close_price,
                         sl, tp, profit, commission, swap, open_time, close_time,
                         strategy_id, comment)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        trade_data['trade_id'],
                        trade_data['symbol'],
                        trade_data['trade_type'],
                        trade_data['volume'],
                        trade_data['open_price'],
                        trade_data.get('close_price'),
                        trade_data.get('sl'),
                        trade_data.get('tp'),
                        trade_data.get('profit', 0),
                        trade_data.get('commission', 0),
                        trade_data.get('swap', 0),
                        open_time,
                        close_time,
                        trade_data.get('strategy_id'),
                        trade_data.get('comment')
                    ))
                    
                    conn.commit()
                    self.logger.debug(f"保存交易记录成功: {trade_data['trade_id']}")
                    return True
                    
            except Exception as e:
                self.logger.error(f"保存交易记录失败: {e}")
                return False
    
    def get_trade_records(self, symbol: Optional[str] = None,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None,
                         strategy_id: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        获取交易记录
        
        Args:
            symbol: 交易品种
            start_time: 开始时间
            end_time: 结束时间
            strategy_id: 策略ID
            
        Returns:
            Optional[pd.DataFrame]: 交易记录DataFrame
        """
        try:
            with self._get_connection() as conn:
                # 构建查询条件
                where_conditions = []
                params = []
                
                if symbol:
                    where_conditions.append("symbol = ?")
                    params.append(symbol)
                
                if start_time:
                    where_conditions.append("open_time >= ?")
                    params.append(start_time)
                
                if end_time:
                    where_conditions.append("open_time <= ?")
                    params.append(end_time)
                
                if strategy_id:
                    where_conditions.append("strategy_id = ?")
                    params.append(strategy_id)
                
                where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
                
                query = f'''
                    SELECT * FROM trades 
                    WHERE {where_clause}
                    ORDER BY open_time DESC
                '''
                
                df = pd.read_sql_query(query, conn, params=params, 
                                     parse_dates=['open_time', 'close_time'])
                
                return df if not df.empty else None
                
        except Exception as e:
            self.logger.error(f"获取交易记录失败: {e}")
            return None
    
    def save_strategy_parameters(self, strategy_name: str, symbol: str, 
                               parameters: Dict[str, Any],
                               performance_metrics: Optional[Dict[str, Any]] = None) -> bool:
        """
        保存策略参数
        
        Args:
            strategy_name: 策略名称
            symbol: 交易品种
            parameters: 策略参数
            performance_metrics: 性能指标
            
        Returns:
            bool: 保存是否成功
        """
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    
                    parameters_json = json.dumps(parameters)
                    metrics_json = json.dumps(performance_metrics) if performance_metrics else None
                    
                    cursor.execute('''
                        INSERT OR REPLACE INTO strategy_parameters 
                        (strategy_name, symbol, parameters, performance_metrics, updated_at)
                        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ''', (strategy_name, symbol, parameters_json, metrics_json))
                    
                    conn.commit()
                    self.logger.debug(f"保存策略参数成功: {strategy_name} - {symbol}")
                    return True
                    
            except Exception as e:
                self.logger.error(f"保存策略参数失败: {e}")
                return False
    
    def get_strategy_parameters(self, strategy_name: Optional[str] = None,
                              symbol: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        获取策略参数
        
        Args:
            strategy_name: 策略名称
            symbol: 交易品种
            
        Returns:
            Optional[pd.DataFrame]: 策略参数DataFrame
        """
        try:
            with self._get_connection() as conn:
                # 构建查询条件
                where_conditions = []
                params = []
                
                if strategy_name:
                    where_conditions.append("strategy_name = ?")
                    params.append(strategy_name)
                
                if symbol:
                    where_conditions.append("symbol = ?")
                    params.append(symbol)
                
                where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
                
                query = f'''
                    SELECT * FROM strategy_parameters 
                    WHERE {where_clause}
                    ORDER BY updated_at DESC
                '''
                
                df = pd.read_sql_query(query, conn, params=params,
                                     parse_dates=['created_at', 'updated_at'])
                
                if df.empty:
                    return None
                
                # 解析JSON数据
                df['parameters'] = df['parameters'].apply(json.loads)
                df['performance_metrics'] = df['performance_metrics'].apply(
                    lambda x: json.loads(x) if x else {}
                )
                
                return df
                
        except Exception as e:
            self.logger.error(f"获取策略参数失败: {e}")
            return None
    
    def backup_database(self, backup_path: Optional[str] = None) -> bool:
        """
        备份数据库
        
        Args:
            backup_path: 备份文件路径
            
        Returns:
            bool: 备份是否成功
        """
        try:
            if backup_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"{self.db_path.stem}_backup_{timestamp}.db"
            
            backup_path = Path(backup_path)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 复制数据库文件
            shutil.copy2(self.db_path, backup_path)
            
            self.logger.info(f"数据库备份成功: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"数据库备份失败: {e}")
            return False
    
    def restore_database(self, backup_path: str) -> bool:
        """
        恢复数据库
        
        Args:
            backup_path: 备份文件路径
            
        Returns:
            bool: 恢复是否成功
        """
        try:
            backup_path = Path(backup_path)
            if not backup_path.exists():
                raise DatabaseError(f"备份文件不存在: {backup_path}")
            
            # 创建当前数据库的备份
            current_backup = f"{self.db_path.stem}_before_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            self.backup_database(current_backup)
            
            # 恢复数据库
            shutil.copy2(backup_path, self.db_path)
            
            self.logger.info(f"数据库恢复成功: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"数据库恢复失败: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        获取数据库统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # 获取各表的记录数
                tables = ['market_data', 'tick_data', 'trades', 'strategy_parameters', 'account_info']
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[f"{table}_count"] = cursor.fetchone()[0]
                
                # 获取数据库文件大小
                stats['db_size_bytes'] = self.db_path.stat().st_size
                stats['db_size_mb'] = round(stats['db_size_bytes'] / (1024 * 1024), 2)
                
                # 获取最早和最新的数据时间
                cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM market_data")
                result = cursor.fetchone()
                if result[0]:
                    stats['earliest_data'] = result[0]
                    stats['latest_data'] = result[1]
                
                return stats
                
        except Exception as e:
            self.logger.error(f"获取数据库统计失败: {e}")
            return {}
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> bool:
        """
        清理旧数据
        
        Args:
            days_to_keep: 保留天数
            
        Returns:
            bool: 清理是否成功
        """
        with self._lock:
            try:
                cutoff_date = datetime.now() - timedelta(days=days_to_keep)
                
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    
                    # 清理旧的Tick数据
                    cursor.execute("DELETE FROM tick_data WHERE timestamp < ?", (cutoff_date,))
                    tick_deleted = cursor.rowcount
                    
                    # 清理旧的系统日志
                    cursor.execute("DELETE FROM system_logs WHERE timestamp < ?", (cutoff_date,))
                    log_deleted = cursor.rowcount
                    
                    conn.commit()
                    
                    self.logger.info(f"清理旧数据完成: Tick数据 {tick_deleted} 条, 日志 {log_deleted} 条")
                    return True
                    
            except Exception as e:
                self.logger.error(f"清理旧数据失败: {e}")
                return False
    
    def close(self):
        """关闭数据库连接"""
        # SQLite连接是在每次操作时创建和关闭的，这里不需要特殊处理
        self.logger.info("数据库管理器已关闭")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()