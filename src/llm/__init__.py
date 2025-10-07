"""
LLM模块 - 本地大语言模型集成
提供Llama 3.2模型的推理接口和管理功能
"""

from .llama_model import LlamaModel, ModelConfig
from .model_manager import ModelManager
from .call_optimizer import (
    LLMCallOptimizer,
    CallPriority,
    LLMResultCache,
    AdaptiveRateLimiter,
    AsyncLLMExecutor,
    CallMetrics
)

__all__ = [
    'LlamaModel',
    'ModelConfig',
    'ModelManager',
    'LLMCallOptimizer',
    'CallPriority',
    'LLMResultCache',
    'AdaptiveRateLimiter',
    'AsyncLLMExecutor',
    'CallMetrics'
]
