"""
Llama模型包装类
提供统一的推理接口和内存管理
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from pathlib import Path
import gc

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logging.warning("llama-cpp-python not available, LLM features will be disabled")

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """LLM模型配置"""
    model_path: str
    n_ctx: int = 2048  # 上下文窗口大小
    n_threads: int = 4  # CPU线程数
    n_gpu_layers: int = 0  # GPU层数（0表示仅CPU）
    temperature: float = 0.7  # 生成温度
    max_tokens: int = 512  # 最大生成token数
    top_p: float = 0.9  # Top-p采样
    top_k: int = 40  # Top-k采样
    repeat_penalty: float = 1.1  # 重复惩罚
    verbose: bool = False  # 详细日志


class LlamaModel:
    """Llama模型包装类"""
    
    def __init__(self, config: ModelConfig):
        """
        初始化Llama模型
        
        Args:
            config: 模型配置
        """
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError("llama-cpp-python is required for LLM features")
        
        self.config = config
        self.model: Optional[Llama] = None
        self._is_loaded = False
        
        logger.info(f"Initializing LlamaModel with config: {config}")
    
    def load_model(self) -> bool:
        """
        加载模型到内存
        
        Returns:
            bool: 加载是否成功
        """
        if self._is_loaded:
            logger.warning("Model already loaded")
            return True
        
        try:
            model_path = Path(self.config.model_path)
            if not model_path.exists():
                logger.error(f"Model file not found: {self.config.model_path}")
                return False
            
            logger.info(f"Loading model from {self.config.model_path}")
            
            self.model = Llama(
                model_path=str(model_path),
                n_ctx=self.config.n_ctx,
                n_threads=self.config.n_threads,
                n_gpu_layers=self.config.n_gpu_layers,
                verbose=self.config.verbose
            )
            
            self._is_loaded = True
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._is_loaded = False
            return False
    
    def unload_model(self) -> None:
        """卸载模型，释放内存"""
        if self.model is not None:
            logger.info("Unloading model from memory")
            del self.model
            self.model = None
            self._is_loaded = False
            
            # 强制垃圾回收
            gc.collect()
            logger.info("Model unloaded successfully")
    
    def generate(self, 
                prompt: str,
                max_tokens: Optional[int] = None,
                temperature: Optional[float] = None,
                top_p: Optional[float] = None,
                top_k: Optional[int] = None,
                stop: Optional[List[str]] = None) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入提示
            max_tokens: 最大生成token数（覆盖配置）
            temperature: 生成温度（覆盖配置）
            top_p: Top-p采样（覆盖配置）
            top_k: Top-k采样（覆盖配置）
            stop: 停止序列
            
        Returns:
            str: 生成的文本
        """
        if not self._is_loaded or self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # 使用传入参数或配置默认值
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p
        top_k = top_k or self.config.top_k
        
        try:
            logger.debug(f"Generating text for prompt: {prompt[:100]}...")
            
            output = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=self.config.repeat_penalty,
                stop=stop or []
            )
            
            generated_text = output['choices'][0]['text']
            logger.debug(f"Generated text: {generated_text[:100]}...")
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Error during text generation: {e}")
            raise
    
    def chat(self, 
            messages: List[Dict[str, str]],
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None) -> str:
        """
        聊天模式生成
        
        Args:
            messages: 消息列表，格式为 [{"role": "user/assistant", "content": "..."}]
            max_tokens: 最大生成token数
            temperature: 生成温度
            
        Returns:
            str: 生成的回复
        """
        # 构建聊天提示
        prompt = self._format_chat_prompt(messages)
        return self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["User:", "Assistant:"]
        )
    
    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        格式化聊天提示
        
        Args:
            messages: 消息列表
            
        Returns:
            str: 格式化的提示
        """
        prompt_parts = []
        
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict: 模型信息
        """
        return {
            'model_path': self.config.model_path,
            'is_loaded': self._is_loaded,
            'n_ctx': self.config.n_ctx,
            'n_threads': self.config.n_threads,
            'n_gpu_layers': self.config.n_gpu_layers,
        }
    
    @property
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self._is_loaded
    
    def __enter__(self):
        """上下文管理器入口"""
        self.load_model()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.unload_model()
    
    def __del__(self):
        """析构函数，确保模型被卸载"""
        if hasattr(self, '_is_loaded') and self._is_loaded:
            self.unload_model()
