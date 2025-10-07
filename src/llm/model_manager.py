"""
模型管理器
负责模型的加载、卸载和内存管理
"""

import logging
import psutil
from typing import Optional, Dict
from pathlib import Path
from .llama_model import LlamaModel, ModelConfig

logger = logging.getLogger(__name__)


class ModelManager:
    """模型管理器 - 管理LLM模型的生命周期和资源"""
    
    def __init__(self, 
                 model_path: str,
                 auto_load: bool = False,
                 memory_threshold_mb: int = 1024):
        """
        初始化模型管理器
        
        Args:
            model_path: 模型文件路径
            auto_load: 是否自动加载模型
            memory_threshold_mb: 内存阈值（MB），低于此值时不加载模型
        """
        self.model_path = model_path
        self.memory_threshold_mb = memory_threshold_mb
        self._model: Optional[LlamaModel] = None
        self._model_config: Optional[ModelConfig] = None
        
        logger.info(f"ModelManager initialized with model: {model_path}")
        
        if auto_load:
            self.load_model()
    
    def create_config(self, 
                     n_ctx: int = 2048,
                     n_threads: int = 4,
                     n_gpu_layers: int = 0,
                     temperature: float = 0.7,
                     max_tokens: int = 512) -> ModelConfig:
        """
        创建模型配置
        
        Args:
            n_ctx: 上下文窗口大小
            n_threads: CPU线程数
            n_gpu_layers: GPU层数
            temperature: 生成温度
            max_tokens: 最大生成token数
            
        Returns:
            ModelConfig: 模型配置对象
        """
        config = ModelConfig(
            model_path=self.model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        self._model_config = config
        return config
    
    def load_model(self, config: Optional[ModelConfig] = None) -> bool:
        """
        加载模型
        
        Args:
            config: 模型配置，如果为None则使用默认配置
            
        Returns:
            bool: 是否成功加载
        """
        # 检查内存是否足够
        if not self._check_memory_available():
            logger.error(f"Insufficient memory. Available: {self.get_available_memory_mb()}MB, "
                        f"Threshold: {self.memory_threshold_mb}MB")
            return False
        
        # 如果模型已加载，先卸载
        if self._model is not None and self._model.is_loaded:
            logger.info("Model already loaded, unloading first")
            self.unload_model()
        
        # 使用提供的配置或创建默认配置
        if config is None:
            if self._model_config is None:
                config = self.create_config()
            else:
                config = self._model_config
        
        try:
            self._model = LlamaModel(config)
            success = self._model.load_model()
            
            if success:
                logger.info("Model loaded successfully by ModelManager")
            else:
                logger.error("Failed to load model")
                self._model = None
            
            return success
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self._model = None
            return False
    
    def unload_model(self) -> None:
        """卸载模型"""
        if self._model is not None:
            self._model.unload_model()
            self._model = None
            logger.info("Model unloaded by ModelManager")
    
    def get_model(self) -> Optional[LlamaModel]:
        """
        获取模型实例
        
        Returns:
            Optional[LlamaModel]: 模型实例，如果未加载则返回None
        """
        return self._model
    
    def is_model_loaded(self) -> bool:
        """
        检查模型是否已加载
        
        Returns:
            bool: 模型是否已加载
        """
        return self._model is not None and self._model.is_loaded
    
    def _check_memory_available(self) -> bool:
        """
        检查可用内存是否足够
        
        Returns:
            bool: 内存是否足够
        """
        available_mb = self.get_available_memory_mb()
        return available_mb >= self.memory_threshold_mb
    
    @staticmethod
    def get_available_memory_mb() -> float:
        """
        获取可用内存（MB）
        
        Returns:
            float: 可用内存大小（MB）
        """
        memory = psutil.virtual_memory()
        return memory.available / (1024 * 1024)
    
    @staticmethod
    def get_memory_usage_mb() -> float:
        """
        获取当前进程内存使用（MB）
        
        Returns:
            float: 内存使用大小（MB）
        """
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    
    def get_memory_stats(self) -> Dict[str, float]:
        """
        获取内存统计信息
        
        Returns:
            Dict: 内存统计信息
        """
        memory = psutil.virtual_memory()
        return {
            'total_mb': memory.total / (1024 * 1024),
            'available_mb': memory.available / (1024 * 1024),
            'used_mb': memory.used / (1024 * 1024),
            'percent': memory.percent,
            'process_mb': self.get_memory_usage_mb()
        }
    
    def reload_model(self, config: Optional[ModelConfig] = None) -> bool:
        """
        重新加载模型
        
        Args:
            config: 新的模型配置
            
        Returns:
            bool: 是否成功重新加载
        """
        logger.info("Reloading model")
        self.unload_model()
        return self.load_model(config)
    
    def __enter__(self):
        """上下文管理器入口"""
        if not self.is_model_loaded():
            self.load_model()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.unload_model()
    
    def __del__(self):
        """析构函数"""
        if self._model is not None:
            self.unload_model()
