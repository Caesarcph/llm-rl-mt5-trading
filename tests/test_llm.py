"""
LLM模块测试
测试Llama模型包装类和模型管理器
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os

from src.llm.llama_model import LlamaModel, ModelConfig, LLAMA_CPP_AVAILABLE
from src.llm.model_manager import ModelManager


class TestModelConfig(unittest.TestCase):
    """测试ModelConfig数据类"""
    
    def test_model_config_creation(self):
        """测试配置创建"""
        config = ModelConfig(
            model_path="models/llama-3.2-1b.gguf",
            n_ctx=2048,
            n_threads=4,
            temperature=0.7
        )
        
        self.assertEqual(config.model_path, "models/llama-3.2-1b.gguf")
        self.assertEqual(config.n_ctx, 2048)
        self.assertEqual(config.n_threads, 4)
        self.assertEqual(config.temperature, 0.7)
    
    def test_model_config_defaults(self):
        """测试默认配置值"""
        config = ModelConfig(model_path="test.gguf")
        
        self.assertEqual(config.n_ctx, 2048)
        self.assertEqual(config.n_threads, 4)
        self.assertEqual(config.n_gpu_layers, 0)
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.max_tokens, 512)


@unittest.skipIf(not LLAMA_CPP_AVAILABLE, "llama-cpp-python not available")
class TestLlamaModel(unittest.TestCase):
    """测试LlamaModel类"""
    
    def setUp(self):
        """测试前准备"""
        self.config = ModelConfig(
            model_path="models/test_model.gguf",
            n_ctx=512,
            n_threads=2,
            temperature=0.5
        )
    
    def test_model_initialization(self):
        """测试模型初始化"""
        model = LlamaModel(self.config)
        
        self.assertIsNotNone(model)
        self.assertEqual(model.config, self.config)
        self.assertFalse(model.is_loaded)
    
    @patch('src.llm.llama_model.Llama')
    @patch('src.llm.llama_model.Path')
    def test_model_load_success(self, mock_path, mock_llama):
        """测试模型加载成功"""
        # Mock文件存在
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        
        # Mock Llama实例
        mock_llama_instance = MagicMock()
        mock_llama.return_value = mock_llama_instance
        
        model = LlamaModel(self.config)
        result = model.load_model()
        
        self.assertTrue(result)
        self.assertTrue(model.is_loaded)
        mock_llama.assert_called_once()
    
    @patch('src.llm.llama_model.Path')
    def test_model_load_file_not_found(self, mock_path):
        """测试模型文件不存在"""
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = False
        mock_path.return_value = mock_path_instance
        
        model = LlamaModel(self.config)
        result = model.load_model()
        
        self.assertFalse(result)
        self.assertFalse(model.is_loaded)
    
    @patch('src.llm.llama_model.Llama')
    @patch('src.llm.llama_model.Path')
    def test_model_unload(self, mock_path, mock_llama):
        """测试模型卸载"""
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        
        mock_llama_instance = MagicMock()
        mock_llama.return_value = mock_llama_instance
        
        model = LlamaModel(self.config)
        model.load_model()
        
        self.assertTrue(model.is_loaded)
        
        model.unload_model()
        
        self.assertFalse(model.is_loaded)
        self.assertIsNone(model.model)
    
    @patch('src.llm.llama_model.Llama')
    @patch('src.llm.llama_model.Path')
    def test_generate_text(self, mock_path, mock_llama):
        """测试文本生成"""
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        
        mock_llama_instance = MagicMock()
        mock_llama_instance.return_value = {
            'choices': [{'text': '  Generated response  '}]
        }
        mock_llama.return_value = mock_llama_instance
        
        model = LlamaModel(self.config)
        model.load_model()
        
        result = model.generate("Test prompt")
        
        self.assertEqual(result, "Generated response")
        mock_llama_instance.assert_called_once()
    
    def test_generate_without_loading(self):
        """测试未加载模型时生成文本"""
        model = LlamaModel(self.config)
        
        with self.assertRaises(RuntimeError):
            model.generate("Test prompt")
    
    @patch('src.llm.llama_model.Llama')
    @patch('src.llm.llama_model.Path')
    def test_chat_mode(self, mock_path, mock_llama):
        """测试聊天模式"""
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        
        mock_llama_instance = MagicMock()
        mock_llama_instance.return_value = {
            'choices': [{'text': 'Chat response'}]
        }
        mock_llama.return_value = mock_llama_instance
        
        model = LlamaModel(self.config)
        model.load_model()
        
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        
        result = model.chat(messages)
        
        self.assertIsInstance(result, str)
        mock_llama_instance.assert_called_once()
    
    def test_format_chat_prompt(self):
        """测试聊天提示格式化"""
        model = LlamaModel(self.config)
        
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"}
        ]
        
        prompt = model._format_chat_prompt(messages)
        
        self.assertIn("System: You are helpful", prompt)
        self.assertIn("User: Hello", prompt)
        self.assertIn("Assistant: Hi", prompt)
        self.assertIn("Assistant:", prompt)
    
    def test_get_model_info(self):
        """测试获取模型信息"""
        model = LlamaModel(self.config)
        info = model.get_model_info()
        
        self.assertIn('model_path', info)
        self.assertIn('is_loaded', info)
        self.assertIn('n_ctx', info)
        self.assertEqual(info['model_path'], self.config.model_path)
        self.assertFalse(info['is_loaded'])
    
    @patch('src.llm.llama_model.Llama')
    @patch('src.llm.llama_model.Path')
    def test_context_manager(self, mock_path, mock_llama):
        """测试上下文管理器"""
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        
        mock_llama_instance = MagicMock()
        mock_llama.return_value = mock_llama_instance
        
        with LlamaModel(self.config) as model:
            self.assertTrue(model.is_loaded)
        
        # 退出上下文后应该卸载
        self.assertFalse(model.is_loaded)


class TestModelManager(unittest.TestCase):
    """测试ModelManager类"""
    
    def setUp(self):
        """测试前准备"""
        self.model_path = "models/test_model.gguf"
    
    def test_manager_initialization(self):
        """测试管理器初始化"""
        manager = ModelManager(self.model_path, auto_load=False)
        
        self.assertEqual(manager.model_path, self.model_path)
        self.assertFalse(manager.is_model_loaded())
    
    def test_create_config(self):
        """测试创建配置"""
        manager = ModelManager(self.model_path)
        config = manager.create_config(
            n_ctx=1024,
            n_threads=2,
            temperature=0.8
        )
        
        self.assertEqual(config.model_path, self.model_path)
        self.assertEqual(config.n_ctx, 1024)
        self.assertEqual(config.n_threads, 2)
        self.assertEqual(config.temperature, 0.8)
    
    @patch('src.llm.model_manager.LlamaModel')
    @patch('src.llm.model_manager.ModelManager._check_memory_available')
    def test_load_model_success(self, mock_memory_check, mock_llama_model):
        """测试加载模型成功"""
        mock_memory_check.return_value = True
        
        mock_model_instance = MagicMock()
        mock_model_instance.load_model.return_value = True
        mock_model_instance.is_loaded = True
        mock_llama_model.return_value = mock_model_instance
        
        manager = ModelManager(self.model_path)
        result = manager.load_model()
        
        self.assertTrue(result)
        self.assertTrue(manager.is_model_loaded())
    
    @patch('src.llm.model_manager.ModelManager._check_memory_available')
    def test_load_model_insufficient_memory(self, mock_memory_check):
        """测试内存不足时加载失败"""
        mock_memory_check.return_value = False
        
        manager = ModelManager(self.model_path)
        result = manager.load_model()
        
        self.assertFalse(result)
        self.assertFalse(manager.is_model_loaded())
    
    @patch('src.llm.model_manager.LlamaModel')
    @patch('src.llm.model_manager.ModelManager._check_memory_available')
    def test_unload_model(self, mock_memory_check, mock_llama_model):
        """测试卸载模型"""
        mock_memory_check.return_value = True
        
        mock_model_instance = MagicMock()
        mock_model_instance.load_model.return_value = True
        mock_model_instance.is_loaded = True
        mock_llama_model.return_value = mock_model_instance
        
        manager = ModelManager(self.model_path)
        manager.load_model()
        
        manager.unload_model()
        
        self.assertFalse(manager.is_model_loaded())
        mock_model_instance.unload_model.assert_called_once()
    
    @patch('src.llm.model_manager.LlamaModel')
    @patch('src.llm.model_manager.ModelManager._check_memory_available')
    def test_get_model(self, mock_memory_check, mock_llama_model):
        """测试获取模型实例"""
        mock_memory_check.return_value = True
        
        mock_model_instance = MagicMock()
        mock_model_instance.load_model.return_value = True
        mock_model_instance.is_loaded = True
        mock_llama_model.return_value = mock_model_instance
        
        manager = ModelManager(self.model_path)
        manager.load_model()
        
        model = manager.get_model()
        
        self.assertIsNotNone(model)
        self.assertEqual(model, mock_model_instance)
    
    @patch('src.llm.model_manager.psutil')
    def test_get_available_memory(self, mock_psutil):
        """测试获取可用内存"""
        mock_memory = MagicMock()
        mock_memory.available = 4096 * 1024 * 1024  # 4GB
        mock_psutil.virtual_memory.return_value = mock_memory
        
        available_mb = ModelManager.get_available_memory_mb()
        
        self.assertEqual(available_mb, 4096.0)
    
    @patch('src.llm.model_manager.psutil')
    def test_get_memory_stats(self, mock_psutil):
        """测试获取内存统计"""
        mock_memory = MagicMock()
        mock_memory.total = 8192 * 1024 * 1024  # 8GB
        mock_memory.available = 4096 * 1024 * 1024  # 4GB
        mock_memory.used = 4096 * 1024 * 1024  # 4GB
        mock_memory.percent = 50.0
        mock_psutil.virtual_memory.return_value = mock_memory
        
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 512 * 1024 * 1024  # 512MB
        mock_psutil.Process.return_value = mock_process
        
        manager = ModelManager(self.model_path)
        stats = manager.get_memory_stats()
        
        self.assertEqual(stats['total_mb'], 8192.0)
        self.assertEqual(stats['available_mb'], 4096.0)
        self.assertEqual(stats['percent'], 50.0)
    
    @patch('src.llm.model_manager.LlamaModel')
    @patch('src.llm.model_manager.ModelManager._check_memory_available')
    def test_reload_model(self, mock_memory_check, mock_llama_model):
        """测试重新加载模型"""
        mock_memory_check.return_value = True
        
        mock_model_instance = MagicMock()
        mock_model_instance.load_model.return_value = True
        mock_model_instance.is_loaded = True
        mock_llama_model.return_value = mock_model_instance
        
        manager = ModelManager(self.model_path)
        manager.load_model()
        
        result = manager.reload_model()
        
        self.assertTrue(result)
        # 应该调用unload和load
        self.assertEqual(mock_model_instance.unload_model.call_count, 1)
    
    @patch('src.llm.model_manager.LlamaModel')
    @patch('src.llm.model_manager.ModelManager._check_memory_available')
    def test_context_manager(self, mock_memory_check, mock_llama_model):
        """测试上下文管理器"""
        mock_memory_check.return_value = True
        
        mock_model_instance = MagicMock()
        mock_model_instance.load_model.return_value = True
        mock_model_instance.is_loaded = True
        mock_llama_model.return_value = mock_model_instance
        
        with ModelManager(self.model_path) as manager:
            self.assertTrue(manager.is_model_loaded())
        
        # 退出上下文后应该卸载
        mock_model_instance.unload_model.assert_called()


if __name__ == '__main__':
    unittest.main()
