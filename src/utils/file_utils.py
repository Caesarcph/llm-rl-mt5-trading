#!/usr/bin/env python3
"""
文件操作工具函数
提供安全的文件读写功能
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Any
import time
import platform

# Windows compatibility for file locking
if platform.system() == "Windows":
    import msvcrt
    def lock_file(f, exclusive=True):
        """Windows file locking"""
        try:
            if exclusive:
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    
    def unlock_file(f):
        """Windows file unlocking"""
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            return True
        except OSError:
            return False
else:
    import fcntl
    def lock_file(f, exclusive=True):
        """Unix file locking"""
        try:
            if exclusive:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
            return True
        except (OSError, IOError):
            return False
    
    def unlock_file(f):
        """Unix file unlocking"""
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return True
        except (OSError, IOError):
            return False

logger = logging.getLogger(__name__)


def safe_read_file(file_path: Path, max_retries: int = 3, retry_delay: float = 0.1) -> Optional[str]:
    """
    安全读取文件内容
    
    Args:
        file_path: 文件路径
        max_retries: 最大重试次数
        retry_delay: 重试延迟(秒)
        
    Returns:
        文件内容或None
    """
    for attempt in range(max_retries):
        try:
            if not file_path.exists():
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                # 尝试获取文件锁
                if lock_file(f, exclusive=False):
                    content = f.read().strip()
                    unlock_file(f)
                    return content
                else:
                    # 文件被锁定，等待后重试
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        logger.warning(f"文件被锁定，无法读取: {file_path}")
                        return None
                        
        except Exception as e:
            logger.error(f"读取文件失败 (尝试 {attempt + 1}/{max_retries}): {file_path}, 错误: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return None
    
    return None


def safe_write_file(file_path: Path, content: str, max_retries: int = 3, retry_delay: float = 0.1) -> bool:
    """
    安全写入文件内容
    
    Args:
        file_path: 文件路径
        content: 文件内容
        max_retries: 最大重试次数
        retry_delay: 重试延迟(秒)
        
    Returns:
        写入是否成功
    """
    # 确保目录存在
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    for attempt in range(max_retries):
        try:
            # 使用临时文件确保原子性写入
            temp_path = file_path.with_suffix(file_path.suffix + '.tmp')
            
            with open(temp_path, 'w', encoding='utf-8') as f:
                # 尝试获取文件锁
                if lock_file(f, exclusive=True):
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())  # 强制写入磁盘
                    unlock_file(f)
                else:
                    # 文件被锁定，等待后重试
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        logger.warning(f"文件被锁定，无法写入: {file_path}")
                        return False
            
            # 原子性移动临时文件到目标位置
            temp_path.replace(file_path)
            return True
            
        except Exception as e:
            logger.error(f"写入文件失败 (尝试 {attempt + 1}/{max_retries}): {file_path}, 错误: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return False
    
    return False


def safe_read_json(file_path: Path, default: Any = None) -> Any:
    """
    安全读取JSON文件
    
    Args:
        file_path: 文件路径
        default: 默认值
        
    Returns:
        JSON数据或默认值
    """
    try:
        content = safe_read_file(file_path)
        if content:
            return json.loads(content)
        return default
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析失败: {file_path}, 错误: {e}")
        return default
    except Exception as e:
        logger.error(f"读取JSON文件失败: {file_path}, 错误: {e}")
        return default


def safe_write_json(file_path: Path, data: Any, indent: int = 2) -> bool:
    """
    安全写入JSON文件
    
    Args:
        file_path: 文件路径
        data: 要写入的数据
        indent: JSON缩进
        
    Returns:
        写入是否成功
    """
    try:
        content = json.dumps(data, indent=indent, ensure_ascii=False)
        return safe_write_file(file_path, content)
    except Exception as e:
        logger.error(f"写入JSON文件失败: {file_path}, 错误: {e}")
        return False


def backup_file(file_path: Path, backup_dir: Optional[Path] = None) -> bool:
    """
    备份文件
    
    Args:
        file_path: 源文件路径
        backup_dir: 备份目录，默认为源文件目录下的backup子目录
        
    Returns:
        备份是否成功
    """
    try:
        if not file_path.exists():
            return False
        
        if backup_dir is None:
            backup_dir = file_path.parent / "backup"
        
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成备份文件名（包含时间戳）
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
        backup_path = backup_dir / backup_name
        
        # 复制文件
        import shutil
        shutil.copy2(file_path, backup_path)
        
        logger.info(f"文件备份成功: {file_path} -> {backup_path}")
        return True
        
    except Exception as e:
        logger.error(f"文件备份失败: {file_path}, 错误: {e}")
        return False


def cleanup_old_files(directory: Path, pattern: str = "*", max_age_days: int = 7) -> int:
    """
    清理旧文件
    
    Args:
        directory: 目录路径
        pattern: 文件模式
        max_age_days: 最大保留天数
        
    Returns:
        清理的文件数量
    """
    try:
        if not directory.exists():
            return 0
        
        import time
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 3600
        cleaned_count = 0
        
        for file_path in directory.glob(pattern):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    try:
                        file_path.unlink()
                        cleaned_count += 1
                        logger.debug(f"清理旧文件: {file_path}")
                    except Exception as e:
                        logger.error(f"清理文件失败: {file_path}, 错误: {e}")
        
        if cleaned_count > 0:
            logger.info(f"清理了 {cleaned_count} 个旧文件")
        
        return cleaned_count
        
    except Exception as e:
        logger.error(f"清理旧文件失败: {directory}, 错误: {e}")
        return 0


def ensure_directory(directory: Path) -> bool:
    """
    确保目录存在
    
    Args:
        directory: 目录路径
        
    Returns:
        目录是否存在或创建成功
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"创建目录失败: {directory}, 错误: {e}")
        return False


def get_file_size(file_path: Path) -> int:
    """
    获取文件大小
    
    Args:
        file_path: 文件路径
        
    Returns:
        文件大小(字节)，文件不存在返回-1
    """
    try:
        if file_path.exists():
            return file_path.stat().st_size
        return -1
    except Exception as e:
        logger.error(f"获取文件大小失败: {file_path}, 错误: {e}")
        return -1


def is_file_locked(file_path: Path) -> bool:
    """
    检查文件是否被锁定
    
    Args:
        file_path: 文件路径
        
    Returns:
        文件是否被锁定
    """
    try:
        if not file_path.exists():
            return False
        
        with open(file_path, 'r') as f:
            if lock_file(f, exclusive=True):
                unlock_file(f)
                return False
            else:
                return True
    except Exception:
        return False