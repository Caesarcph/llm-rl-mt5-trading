# 示例代码

本目录包含各种示例代码，帮助您快速了解和使用LLM-RL MT5交易系统。

## 目录

- [快速开始](#快速开始)
- [示例列表](#示例列表)
- [运行示例](#运行示例)
- [创建自己的示例](#创建自己的示例)

---

## 快速开始

### 前置条件

1. 已完成系统安装
2. 已配置MT5连接
3. 已激活Python虚拟环境

### 运行第一个示例

```bash
# 激活虚拟环境
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 运行快速开始示例
python examples/quick_start_example.py
```

---

## 示例列表

### 1. 快速开始示例 (quick_start_example.py)

**功能**: 演示系统的基本使用流程

**包含内容**:
- 加载配置
- 连接MT5
- 获取账户信息
- 获取市场数据
- 创建策略
- 生成交易信号
- 风险验证

**适合**: 初次使用者

**运行**:
```bash
python examples/quick_start_example.py
```

**预期输出**:
```
=== 快速开始示例 ===
步骤1: 加载配置
配置加载成功，模拟模式: True
步骤2: 连接MT5
MT5连接成功
步骤3: 获取账户信息
账户余额: $10000.00
...
```

---

### 2. 自定义策略示例 (custom_strategy_example.py)

**功能**: 演示如何创建自定义交易策略

**包含内容**:
- 策略类定义
- 信号生成逻辑
- 技术指标计算
- 参数更新
- 止损止盈计算

**适合**: 想要开发自己策略的用户

**运行**:
```bash
python examples/custom_strategy_example.py
```

**学习要点**:
- 继承Strategy基类
- 实现generate_signal()方法
- 使用pandas处理数据
- 计算技术指标
- 生成Signal对象

**代码片段**:
```python
class SimpleMAStrategy(Strategy):
    def generate_signal(self, market_data: MarketData) -> Optional[Signal]:
        # 计算移动平均线
        df['ma_fast'] = df['close'].rolling(window=self.fast_period).mean()
        df['ma_slow'] = df['close'].rolling(window=self.slow_period).mean()
        
        # 检测交叉
        if bullish_cross:
            return Signal(...)
```

---

### 3. 回测示例 (backtest_example.py)

**功能**: 演示如何回测交易策略

**包含内容**:
- 简单回测
- 参数优化
- 蒙特卡洛模拟
- 性能分析
- 报告生成

**适合**: 需要验证策略的用户

**运行**:
```bash
python examples/backtest_example.py
```

**交互式选择**:
```
请选择要运行的示例:
1. 简单回测
2. 参数优化
3. 蒙特卡洛模拟
4. 运行全部
```

**回测结果示例**:
```
回测结果:
初始资金: $10,000.00
最终资金: $12,500.00
总收益: $2,500.00 (25.00%)
最大回撤: -8.50%
夏普比率: 1.85
总交易次数: 45
胜率: 62.22%
盈利因子: 1.75
```

---

### 4. LLM分析示例 (llm_analyst_demo.py)

**功能**: 演示LLM市场分析功能

**包含内容**:
- 新闻情绪分析
- 市场评论生成
- 事件影响分析
- 异常行情解释

**前置条件**: 需要下载LLM模型

**运行**:
```bash
python examples/llm_analyst_demo.py
```

---

### 5. 强化学习示例 (rl_system_demo.py)

**功能**: 演示强化学习优化

**包含内容**:
- RL环境设置
- 模型训练
- 策略优化
- 性能评估

**运行**:
```bash
python examples/rl_system_demo.py
```

---

### 6. 多策略系统示例 (multi_strategy_system_demo.py)

**功能**: 演示多策略协同工作

**包含内容**:
- 多策略注册
- 信号聚合
- 权重调整
- 冲突解决

**运行**:
```bash
python examples/multi_strategy_system_demo.py
```

---

### 7. 风险管理示例 (risk_manager_demo.py)

**功能**: 演示风险管理功能

**包含内容**:
- 仓位计算
- 风险验证
- VaR计算
- 熔断机制

**运行**:
```bash
python examples/risk_manager_demo.py
```

---

### 8. EA31337桥接示例 (ea31337_bridge_demo.py)

**功能**: 演示与EA31337的集成

**包含内容**:
- 信号获取
- 状态监控
- 参数更新
- 命令发送

**运行**:
```bash
python examples/ea31337_bridge_demo.py
```

---

### 9. 回测引擎示例 (backtest_engine_demo.py)

**功能**: 演示回测引擎的高级功能

**包含内容**:
- 多品种回测
- 滑点模拟
- 佣金计算
- 详细报告

**运行**:
```bash
python examples/backtest_engine_demo.py
```

---

### 10. 蒙特卡洛示例 (monte_carlo_demo.py)

**功能**: 演示蒙特卡洛风险分析

**包含内容**:
- 随机模拟
- 风险评估
- 概率分布
- 置信区间

**运行**:
```bash
python examples/monte_carlo_demo.py
```

---

## 运行示例

### 基本运行

```bash
# 运行单个示例
python examples/example_name.py

# 使用特定配置
python examples/example_name.py --config config/custom.yaml

# 启用调试模式
python examples/example_name.py --debug
```

### 批量运行

```bash
# 运行所有示例
for file in examples/*_demo.py; do
    echo "运行 $file"
    python "$file"
done
```

### 在Jupyter中运行

```python
# 在Jupyter Notebook中
%run examples/quick_start_example.py
```

---

## 创建自己的示例

### 模板

```python
#!/usr/bin/env python3
"""
我的示例
描述示例的功能
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.config import load_config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    logger.info("=== 我的示例 ===")
    
    # 1. 加载配置
    config = load_config()
    
    # 2. 实现您的逻辑
    # ...
    
    logger.info("=== 示例完成 ===")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n程序被用户中断")
    except Exception as e:
        logger.error(f"程序异常: {e}", exc_info=True)
```

### 最佳实践

1. **使用日志**: 使用logging而非print
2. **异常处理**: 捕获并处理异常
3. **清理资源**: 确保正确关闭连接
4. **添加注释**: 解释关键步骤
5. **提供输出**: 显示有意义的结果

---

## 测试数据

### 生成测试数据

```bash
# 生成模拟市场数据
python scripts/generate_test_data.py
```

### 使用测试数据

```python
# 在示例中使用测试数据
df = pd.read_csv("data/test/EURUSD_H1_90days.csv")
```

---

## 常见问题

### Q: 示例运行失败怎么办？

**A**: 检查以下几点：
1. 是否激活虚拟环境
2. 是否安装所有依赖
3. 是否配置MT5连接
4. 查看错误日志

### Q: 如何修改示例参数？

**A**: 直接编辑示例文件中的参数：
```python
# 修改这些参数
initial_balance = 10000
symbol = "EURUSD"
timeframe = "H1"
```

### Q: 示例可以用于实盘吗？

**A**: 示例仅用于学习和测试，不建议直接用于实盘。实盘前需要：
1. 充分测试
2. 调整参数
3. 添加错误处理
4. 实现完整的风险管理

---

## 更多资源

- [用户手册](../docs/user_manual.md)
- [API参考](../docs/api_reference.md)
- [架构指南](../docs/architecture_guide.md)
- [故障排除](../docs/troubleshooting_guide.md)

---

## 贡献示例

欢迎贡献新的示例！

**步骤**:
1. Fork仓库
2. 创建示例文件
3. 添加文档
4. 提交Pull Request

**示例命名规范**:
- 使用描述性名称
- 以`_example.py`或`_demo.py`结尾
- 添加详细注释

---

**祝您学习愉快！** 🚀
