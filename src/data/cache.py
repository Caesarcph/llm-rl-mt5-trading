#!/usr/bin/env python3
"""
Redis缓存管理器
"""

import json
import pickle
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, List
import hashlib

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False

from .models import CacheEntry
from ..core.logging import get_logger
from ..core.exceptions import TradingSystemException


class CacheError(TradingSystemException):
    """缓存异常"""
    pass


class RedisCache:
    """Redis缓存管理器"""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0, 
                 password: Optional[str] = None, decode_responses: bool = False):
        """
        初始化Redis缓存
        
        Args:
            host: Redis主机地址
            port: Redis端口
            db: 数据库编号
            password: 密码
            decode_responses: 是否解码响应
        """
        self.logger = get_logger()
        self._client = None
        self._connected = False
        
        self.config = {
            'host': host,
            'port': port,
            'db': db,
            'password': password,
            'decode_responses': decode_responses,
            'socket_timeout': 5,
            'socket_connect_timeout': 5,
            'retry_on_timeout': True,
            'health_check_interval': 30
        }
        
        # 统计信息
        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'errors': 0
        }
    
    @property
    def is_available(self) -> bool:
        """检查Redis是否可用"""
        return REDIS_AVAILABLE and self._connected
    
    @property
    def stats(self) -> Dict[str, int]:
        """获取缓存统计"""
        return self._stats.copy()
    
    def connect(self) -> bool:
        """
        连接到Redis服务器
        
        Returns:
            bool: 连接是否成功
        """
        if not REDIS_AVAILABLE:
            self.logger.warning("Redis库未安装，缓存功能不可用")
            return False
        
        try:
            self._client = redis.Redis(**self.config)
            
            # 测试连接
            self._client.ping()
            self._connected = True
            
            self.logger.info(f"Redis连接成功: {self.config['host']}:{self.config['port']}")
            return True
            
        except Exception as e:
            self.logger.warning(f"Redis连接失败: {e}")
            self._connected = False
            return False
    
    def disconnect(self):
        """断开Redis连接"""
        if self._client:
            try:
                self._client.close()
            except:
                pass
            finally:
                self._client = None
                self._connected = False
                self.logger.info("Redis连接已断开")
    
    def get(self, key: str) -> Optional[Any]:
        """
        从缓存获取数据
        
        Args:
            key: 缓存键
            
        Returns:
            Optional[Any]: 缓存的数据，不存在时返回None
        """
        if not self.is_available:
            return None
        
        try:
            data = self._client.get(key)
            if data is None:
                self._stats['misses'] += 1
                return None
            
            # 反序列化数据
            cached_data = pickle.loads(data)
            self._stats['hits'] += 1
            
            self.logger.debug(f"缓存命中: {key}")
            return cached_data
            
        except Exception as e:
            self._stats['errors'] += 1
            self.logger.error(f"缓存读取失败 {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """
        设置缓存数据
        
        Args:
            key: 缓存键
            value: 要缓存的数据
            ttl: 生存时间(秒)
            
        Returns:
            bool: 设置是否成功
        """
        if not self.is_available:
            return False
        
        try:
            # 序列化数据
            serialized_data = pickle.dumps(value)
            
            # 设置缓存
            result = self._client.setex(key, ttl, serialized_data)
            
            if result:
                self._stats['sets'] += 1
                self.logger.debug(f"缓存设置成功: {key} (TTL: {ttl}s)")
                return True
            else:
                self._stats['errors'] += 1
                return False
                
        except Exception as e:
            self._stats['errors'] += 1
            self.logger.error(f"缓存设置失败 {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        删除缓存数据
        
        Args:
            key: 缓存键
            
        Returns:
            bool: 删除是否成功
        """
        if not self.is_available:
            return False
        
        try:
            result = self._client.delete(key)
            if result > 0:
                self._stats['deletes'] += 1
                self.logger.debug(f"缓存删除成功: {key}")
                return True
            else:
                return False
                
        except Exception as e:
            self._stats['errors'] += 1
            self.logger.error(f"缓存删除失败 {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """
        检查缓存键是否存在
        
        Args:
            key: 缓存键
            
        Returns:
            bool: 键是否存在
        """
        if not self.is_available:
            return False
        
        try:
            return bool(self._client.exists(key))
        except Exception as e:
            self.logger.error(f"缓存检查失败 {key}: {e}")
            return False
    
    def expire(self, key: str, ttl: int) -> bool:
        """
        设置缓存过期时间
        
        Args:
            key: 缓存键
            ttl: 生存时间(秒)
            
        Returns:
            bool: 设置是否成功
        """
        if not self.is_available:
            return False
        
        try:
            return bool(self._client.expire(key, ttl))
        except Exception as e:
            self.logger.error(f"设置过期时间失败 {key}: {e}")
            return False
    
    def ttl(self, key: str) -> int:
        """
        获取缓存剩余生存时间
        
        Args:
            key: 缓存键
            
        Returns:
            int: 剩余时间(秒)，-1表示永不过期，-2表示键不存在
        """
        if not self.is_available:
            return -2
        
        try:
            return self._client.ttl(key)
        except Exception as e:
            self.logger.error(f"获取TTL失败 {key}: {e}")
            return -2
    
    def clear_pattern(self, pattern: str) -> int:
        """
        清除匹配模式的缓存
        
        Args:
            pattern: 匹配模式 (支持通配符 *)
            
        Returns:
            int: 删除的键数量
        """
        if not self.is_available:
            return 0
        
        try:
            keys = self._client.keys(pattern)
            if keys:
                deleted = self._client.delete(*keys)
                self._stats['deletes'] += deleted
                self.logger.info(f"清除缓存模式 {pattern}: {deleted} 个键")
                return deleted
            return 0
            
        except Exception as e:
            self.logger.error(f"清除缓存模式失败 {pattern}: {e}")
            return 0
    
    def flush_db(self) -> bool:
        """
        清空当前数据库的所有缓存
        
        Returns:
            bool: 清空是否成功
        """
        if not self.is_available:
            return False
        
        try:
            self._client.flushdb()
            self.logger.warning("已清空Redis数据库")
            return True
        except Exception as e:
            self.logger.error(f"清空数据库失败: {e}")
            return False
    
    def get_info(self) -> Optional[Dict[str, Any]]:
        """
        获取Redis服务器信息
        
        Returns:
            Optional[Dict]: 服务器信息
        """
        if not self.is_available:
            return None
        
        try:
            return self._client.info()
        except Exception as e:
            self.logger.error(f"获取Redis信息失败: {e}")
            return None
    
    def generate_key(self, prefix: str, *args) -> str:
        """
        生成缓存键
        
        Args:
            prefix: 键前缀
            *args: 键参数
            
        Returns:
            str: 生成的缓存键
        """
        # 将参数转换为字符串并连接
        key_parts = [prefix] + [str(arg) for arg in args]
        key = ':'.join(key_parts)
        
        # 如果键太长，使用哈希
        if len(key) > 250:  # Redis键长度限制
            hash_suffix = hashlib.md5(key.encode()).hexdigest()[:8]
            key = f"{prefix}:hash:{hash_suffix}"
        
        return key
    
    def __enter__(self):
        """上下文管理器入口"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.disconnect()


class MemoryCache:
    """内存缓存 (Redis不可用时的备选方案)"""
    
    def __init__(self, max_size: int = 1000):
        """
        初始化内存缓存
        
        Args:
            max_size: 最大缓存条目数
        """
        self.logger = get_logger()
        self.max_size = max_size
        self._cache: Dict[str, CacheEntry] = {}
        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'evictions': 0
        }
    
    @property
    def is_available(self) -> bool:
        """内存缓存总是可用"""
        return True
    
    @property
    def stats(self) -> Dict[str, int]:
        """获取缓存统计"""
        return self._stats.copy()
    
    def get(self, key: str) -> Optional[Any]:
        """从缓存获取数据"""
        self._cleanup_expired()
        
        if key in self._cache:
            entry = self._cache[key]
            if not entry.is_expired:
                self._stats['hits'] += 1
                return entry.data
            else:
                del self._cache[key]
        
        self._stats['misses'] += 1
        return None
    
    def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """设置缓存数据"""
        # 如果缓存已满，删除最旧的条目
        if len(self._cache) >= self.max_size and key not in self._cache:
            self._evict_oldest()
        
        entry = CacheEntry(
            key=key,
            data=value,
            timestamp=datetime.now(),
            ttl=ttl
        )
        
        self._cache[key] = entry
        self._stats['sets'] += 1
        return True
    
    def delete(self, key: str) -> bool:
        """删除缓存数据"""
        if key in self._cache:
            del self._cache[key]
            self._stats['deletes'] += 1
            return True
        return False
    
    def exists(self, key: str) -> bool:
        """检查缓存键是否存在"""
        self._cleanup_expired()
        return key in self._cache
    
    def clear(self):
        """清空所有缓存"""
        self._cache.clear()
        self.logger.info("内存缓存已清空")
    
    def _cleanup_expired(self):
        """清理过期条目"""
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired
        ]
        
        for key in expired_keys:
            del self._cache[key]
    
    def _evict_oldest(self):
        """删除最旧的条目"""
        if not self._cache:
            return
        
        oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].timestamp)
        del self._cache[oldest_key]
        self._stats['evictions'] += 1