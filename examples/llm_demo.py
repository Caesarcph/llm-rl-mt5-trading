"""
LLM模块使用示例
演示如何使用Llama模型进行文本生成和市场分析
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from src.llm import LlamaModel, ModelConfig, ModelManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def example_basic_usage():
    """基础使用示例"""
    print("\n=== 基础使用示例 ===\n")
    
    # 创建模型配置
    config = ModelConfig(
        model_path="models/llama-3.2-1b.gguf",  # 替换为实际模型路径
        n_ctx=2048,
        n_threads=4,
        temperature=0.7,
        max_tokens=256
    )
    
    # 创建模型实例
    model = LlamaModel(config)
    
    # 加载模型
    if model.load_model():
        print("✓ 模型加载成功")
        
        # 生成文本
        prompt = "Analyze the current market trend for EUR/USD:"
        print(f"\nPrompt: {prompt}")
        
        response = model.generate(prompt, max_tokens=200)
        print(f"\nResponse: {response}")
        
        # 卸载模型
        model.unload_model()
        print("\n✓ 模型已卸载")
    else:
        print("✗ 模型加载失败")


def example_chat_mode():
    """聊天模式示例"""
    print("\n=== 聊天模式示例 ===\n")
    
    config = ModelConfig(
        model_path="models/llama-3.2-1b.gguf",
        n_ctx=2048,
        temperature=0.7
    )
    
    # 使用上下文管理器自动管理模型生命周期
    with LlamaModel(config) as model:
        messages = [
            {
                "role": "system",
                "content": "You are a financial market analyst specializing in forex trading."
            },
            {
                "role": "user",
                "content": "What factors should I consider when trading gold (XAU/USD)?"
            }
        ]
        
        response = model.chat(messages)
        print(f"Assistant: {response}")


def example_model_manager():
    """模型管理器示例"""
    print("\n=== 模型管理器示例 ===\n")
    
    # 创建模型管理器
    manager = ModelManager(
        model_path="models/llama-3.2-1b.gguf",
        memory_threshold_mb=1024  # 至少需要1GB可用内存
    )
    
    # 检查内存状态
    memory_stats = manager.get_memory_stats()
    print(f"内存统计:")
    print(f"  总内存: {memory_stats['total_mb']:.2f} MB")
    print(f"  可用内存: {memory_stats['available_mb']:.2f} MB")
    print(f"  使用率: {memory_stats['percent']:.1f}%")
    
    # 创建自定义配置
    config = manager.create_config(
        n_ctx=1024,
        n_threads=2,
        temperature=0.5
    )
    
    # 加载模型
    if manager.load_model(config):
        print("\n✓ 模型加载成功")
        
        # 获取模型实例
        model = manager.get_model()
        if model:
            # 使用模型
            prompt = "Summarize the key risks in forex trading:"
            response = model.generate(prompt, max_tokens=150)
            print(f"\nResponse: {response}")
        
        # 卸载模型
        manager.unload_model()
        print("\n✓ 模型已卸载")
    else:
        print("\n✗ 模型加载失败（可能是内存不足）")


def example_market_analysis():
    """市场分析示例"""
    print("\n=== 市场分析示例 ===\n")
    
    config = ModelConfig(
        model_path="models/llama-3.2-1b.gguf",
        n_ctx=2048,
        temperature=0.6,
        max_tokens=300
    )
    
    with LlamaModel(config) as model:
        # 新闻情绪分析
        news_prompt = """
        Analyze the sentiment of this news headline:
        "Federal Reserve signals potential rate cuts amid economic slowdown"
        
        Provide:
        1. Overall sentiment (Bullish/Bearish/Neutral)
        2. Impact on USD
        3. Trading recommendation
        """
        
        print("分析新闻情绪...")
        sentiment_analysis = model.generate(news_prompt)
        print(f"\n{sentiment_analysis}")
        
        # 市场状态分析
        market_prompt = """
        Given the following market conditions:
        - EUR/USD: Uptrend, RSI at 65
        - Gold: High volatility, breaking resistance
        - VIX: Rising to 25
        
        Provide a brief market outlook and risk assessment.
        """
        
        print("\n\n分析市场状态...")
        market_analysis = model.generate(market_prompt)
        print(f"\n{market_analysis}")


def example_error_handling():
    """错误处理示例"""
    print("\n=== 错误处理示例 ===\n")
    
    try:
        # 尝试加载不存在的模型
        config = ModelConfig(model_path="models/nonexistent.gguf")
        model = LlamaModel(config)
        
        if not model.load_model():
            print("✓ 正确处理了模型文件不存在的情况")
        
        # 尝试在未加载模型时生成文本
        try:
            model.generate("test")
        except RuntimeError as e:
            print(f"✓ 正确捕获了运行时错误: {e}")
    except ImportError as e:
        print(f"⚠ LLM功能不可用: {e}")
        print("提示: 安装 llama-cpp-python 以启用LLM功能")


def main():
    """主函数"""
    print("=" * 60)
    print("LLM模块使用示例")
    print("=" * 60)
    
    # 注意：以下示例需要实际的模型文件才能运行
    # 请下载Llama 3.2模型并更新model_path
    
    print("\n提示：这些示例需要实际的Llama模型文件")
    print("请从 https://huggingface.co/ 下载模型并更新路径")
    
    # 取消注释以运行示例
    # example_basic_usage()
    # example_chat_mode()
    # example_model_manager()
    # example_market_analysis()
    example_error_handling()
    
    print("\n" + "=" * 60)
    print("示例完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
