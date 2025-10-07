# LLM模块

本地大语言模型集成模块，提供Llama 3.2模型的推理接口和管理功能。

## 功能特性

- ✅ 统一的LLM推理接口
- ✅ 自动内存管理和模型生命周期控制
- ✅ 支持文本生成和聊天模式
- ✅ 内存监控和阈值保护
- ✅ 上下文管理器支持
- ✅ 完整的错误处理
- ✅ 灵活的配置选项

## 快速开始

### 安装依赖

```bash
pip install llama-cpp-python psutil
```

### 基础使用

```python
from src.llm import LlamaModel, ModelConfig

# 创建配置
config = ModelConfig(
    model_path="models/llama-3.2-1b.gguf",
    n_ctx=2048,
    temperature=0.7
)

# 使用模型
with LlamaModel(config) as model:
    response = model.generate("Analyze EUR/USD trend:")
    print(response)
```

### 使用模型管理器

```python
from src.llm import ModelManager

# 创建管理器
manager = ModelManager(
    model_path="models/llama-3.2-1b.gguf",
    memory_threshold_mb=1024
)

# 加载并使用模型
if manager.load_model():
    model = manager.get_model()
    response = model.generate("Your prompt")
    manager.unload_model()
```

## 模块结构

```
src/llm/
├── __init__.py           # 模块导出
├── llama_model.py        # Llama模型包装类
├── model_manager.py      # 模型管理器
└── README.md            # 本文档
```

## 核心类

### ModelConfig

模型配置数据类，定义所有LLM运行参数。

**主要参数：**
- `model_path`: 模型文件路径
- `n_ctx`: 上下文窗口大小（默认2048）
- `n_threads`: CPU线程数（默认4）
- `temperature`: 生成温度（默认0.7）
- `max_tokens`: 最大生成token数（默认512）

### LlamaModel

Llama模型包装类，提供统一推理接口。

**主要方法：**
- `load_model()`: 加载模型到内存
- `unload_model()`: 卸载模型释放内存
- `generate()`: 生成文本
- `chat()`: 聊天模式生成
- `get_model_info()`: 获取模型信息

### ModelManager

模型管理器，负责模型生命周期和资源管理。

**主要方法：**
- `load_model()`: 加载模型（带内存检查）
- `unload_model()`: 卸载模型
- `get_model()`: 获取模型实例
- `get_memory_stats()`: 获取内存统计
- `reload_model()`: 重新加载模型

## 使用示例

详细示例请参考：
- `examples/llm_demo.py` - 完整使用示例
- `docs/llm_module_guide.md` - 详细使用指南

## 测试

运行测试：

```bash
# 运行所有LLM测试
python -m pytest tests/test_llm.py -v

# 运行特定测试
python -m pytest tests/test_llm.py::TestModelManager -v
```

## 性能建议

### 内存需求

| 模型 | 量化 | 内存需求 | 推荐配置 |
|------|------|----------|----------|
| Llama 3.2 1B | Q4_K_M | ~2GB | 4GB+ RAM |
| Llama 3.2 3B | Q4_K_M | ~4GB | 8GB+ RAM |

### 优化建议

1. **使用上下文管理器** - 确保模型正确卸载
2. **控制调用频率** - 避免过于频繁的LLM调用
3. **实现结果缓存** - 缓存相似查询的结果
4. **监控内存使用** - 定期检查内存状态
5. **选择合适模型** - 根据需求选择1B或3B版本

## 故障排除

### 问题：llama-cpp-python不可用

**解决方案：**
```bash
pip install llama-cpp-python
```

如果安装失败，尝试：
```bash
# Windows
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# 或从源码编译
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
```

### 问题：内存不足

**解决方案：**
- 使用更小的模型（1B vs 3B）
- 减少 `n_ctx` 大小
- 降低 `memory_threshold_mb`
- 在使用后及时卸载模型

### 问题：生成速度慢

**解决方案：**
- 增加 `n_threads`
- 减少 `max_tokens`
- 使用GPU加速（设置 `n_gpu_layers`）
- 使用更小的模型

## 依赖项

- `llama-cpp-python>=0.2.0` - Llama模型推理
- `psutil>=5.9.0` - 内存监控

## 许可证

本模块遵循项目主许可证。

## 贡献

欢迎提交问题和改进建议！
