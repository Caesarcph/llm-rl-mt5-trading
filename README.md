# LLM-RL MT5 Trading System

[中文](#中文) | [English](#english)

---

## 中文

基于EA31337框架的智能MT5交易系统，集成大语言模型(LLM)和强化学习(RL)技术。

### 系统概述

本系统采用三层混合架构设计：
- **EA31337执行层**: 基于成熟的MQL5交易框架，提供稳定的交易执行
- **Python Bridge层**: 策略管理和参数优化
- **智能决策层**: LLM分析和RL优化

### 主要特性

- 🤖 **智能决策**: 集成本地Llama 3.2模型进行市场分析
- 🧠 **强化学习**: 使用PPO/SAC算法优化策略权重
- 📊 **多品种支持**: EURUSD、XAUUSD、USOIL等主要品种
- 🛡️ **风险管理**: 多层级风险控制和渐进式资金管理
- 📈 **策略集成**: 30+内置EA31337策略 + Python原生策略
- 🔄 **实时监控**: 完整的日志系统和告警机制

### 项目结构

```
├── src/                    # 源代码
│   ├── core/              # 核心模块
│   │   ├── models.py      # 数据模型定义
│   │   ├── config.py      # 配置管理
│   │   ├── logging.py     # 日志系统
│   │   └── exceptions.py  # 异常处理
│   ├── data/              # 数据层
│   ├── strategies/        # 策略层
│   ├── agents/            # Agent层
│   ├── bridge/            # EA31337桥接
│   └── utils/             # 工具模块
├── config/                # 配置文件
│   ├── config.yaml        # 主配置
│   └── symbols/           # 品种配置
├── data/                  # 数据存储
├── logs/                  # 日志文件
├── models/                # LLM模型
├── ea31337/               # EA31337文件
└── tests/                 # 测试文件
```

### 快速开始

#### 1. 环境要求

- Python 3.9+
- MetaTrader 5
- Windows 10/11 (推荐) 或 Linux
- 最低配置: 4核8GB内存
- 推荐配置: 8核16GB内存

#### 2. 安装依赖

```bash
pip install -r requirements.txt
```

#### 3. 配置系统

1. 复制并编辑配置文件:
```bash
cp config/config.yaml config/config_local.yaml
```

2. 配置MT5连接信息:
```yaml
mt5:
  server: "your-broker-server"
  login: your_account_number
  password: "your_password"
```

3. 配置品种参数:
```bash
# 编辑品种配置文件
config/symbols/eurusd.yaml
config/symbols/xauusd.yaml
config/symbols/usoil.yaml
```

#### 4. 运行系统

```bash
python main.py
```

### 配置说明

#### 风险管理配置

```yaml
risk:
  max_risk_per_trade: 0.02      # 单笔最大风险2%
  max_daily_drawdown: 0.05      # 日最大回撤5%
  max_positions: 10             # 最大持仓数
```

#### LLM配置

```yaml
llm:
  model_path: "models/llama-3.2-1b"
  max_tokens: 512
  temperature: 0.7
```

#### 强化学习配置

```yaml
rl:
  algorithm: "PPO"
  learning_rate: 0.0003
  n_steps: 2048
```

### 开发指南

#### 添加新策略

1. 在 `src/strategies/` 目录下创建策略文件
2. 继承 `Strategy` 基类
3. 实现 `generate_signal()` 方法
4. 在配置文件中启用策略

#### 添加新Agent

1. 在 `src/agents/` 目录下创建Agent文件
2. 实现相应的分析逻辑
3. 集成到主系统中

#### 测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_strategies.py

# 生成覆盖率报告
pytest --cov=src tests/
```

### 部署选项

#### 本地部署
- Windows 10/11 + MT5
- 最低4核8GB配置
- 适合个人使用

#### 云端部署
- VPS 2核4GB起
- 成本约$50/月
- 支持24/7运行

### 监控和告警

系统提供多种监控方式:
- 📱 Telegram Bot推送
- 📧 邮件告警
- 📊 Web仪表板
- 📝 详细日志记录

### 风险声明

⚠️ **重要提示**: 
- 本系统仅供学习和研究使用
- 外汇交易存在重大风险，可能导致资金损失
- 请在充分了解风险的情况下使用
- 建议先在模拟账户中测试

### 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

### 贡献

欢迎提交Issue和Pull Request来改进系统。

### 联系方式

如有问题或建议，请通过以下方式联系:
- 提交GitHub Issue
- 发送邮件至项目维护者

---

## English

An intelligent MT5 trading system based on the EA31337 framework, integrating Large Language Model (LLM) and Reinforcement Learning (RL) technologies.

### System Overview

The system adopts a three-tier hybrid architecture:
- **EA31337 Execution Layer**: Based on mature MQL5 trading framework for stable trade execution
- **Python Bridge Layer**: Strategy management and parameter optimization
- **Intelligent Decision Layer**: LLM analysis and RL optimization

### Key Features

- 🤖 **Intelligent Decision Making**: Integrated local Llama 3.2 model for market analysis
- 🧠 **Reinforcement Learning**: PPO/SAC algorithms for strategy weight optimization
- 📊 **Multi-Symbol Support**: Major pairs including EURUSD, XAUUSD, USOIL
- 🛡️ **Risk Management**: Multi-level risk control and progressive money management
- 📈 **Strategy Integration**: 30+ built-in EA31337 strategies + Python native strategies
- 🔄 **Real-time Monitoring**: Complete logging system and alert mechanism

### Project Structure

```
├── src/                    # Source code
│   ├── core/              # Core modules
│   │   ├── models.py      # Data model definitions
│   │   ├── config.py      # Configuration management
│   │   ├── logging.py     # Logging system
│   │   └── exceptions.py  # Exception handling
│   ├── data/              # Data layer
│   ├── strategies/        # Strategy layer
│   ├── agents/            # Agent layer
│   ├── bridge/            # EA31337 bridge
│   └── utils/             # Utility modules
├── config/                # Configuration files
│   ├── config.yaml        # Main configuration
│   └── symbols/           # Symbol configurations
├── data/                  # Data storage
├── logs/                  # Log files
├── models/                # LLM models
├── ea31337/               # EA31337 files
└── tests/                 # Test files
```

### Quick Start

#### 1. Requirements

- Python 3.9+
- MetaTrader 5
- Windows 10/11 (recommended) or Linux
- Minimum: 4-core CPU, 8GB RAM
- Recommended: 8-core CPU, 16GB RAM

#### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 3. Configure System

1. Copy and edit configuration file:
```bash
cp config/config.yaml config/config_local.yaml
```

2. Configure MT5 connection:
```yaml
mt5:
  server: "your-broker-server"
  login: your_account_number
  password: "your_password"
```

3. Configure symbol parameters:
```bash
# Edit symbol configuration files
config/symbols/eurusd.yaml
config/symbols/xauusd.yaml
config/symbols/usoil.yaml
```

#### 4. Run System

```bash
python main.py
```

### Configuration

#### Risk Management

```yaml
risk:
  max_risk_per_trade: 0.02      # Max 2% risk per trade
  max_daily_drawdown: 0.05      # Max 5% daily drawdown
  max_positions: 10             # Maximum positions
```

#### LLM Configuration

```yaml
llm:
  model_path: "models/llama-3.2-1b"
  max_tokens: 512
  temperature: 0.7
```

#### Reinforcement Learning

```yaml
rl:
  algorithm: "PPO"
  learning_rate: 0.0003
  n_steps: 2048
```

### Development Guide

#### Adding New Strategies

1. Create strategy file in `src/strategies/`
2. Inherit from `Strategy` base class
3. Implement `generate_signal()` method
4. Enable strategy in configuration file

#### Adding New Agents

1. Create Agent file in `src/agents/`
2. Implement analysis logic
3. Integrate into main system

#### Testing

```bash
# Run all tests
pytest

# Run specific tests
pytest tests/test_strategies.py

# Generate coverage report
pytest --cov=src tests/
```

### Deployment Options

#### Local Deployment
- Windows 10/11 + MT5
- Minimum 4-core 8GB configuration
- Suitable for personal use

#### Cloud Deployment
- VPS starting from 2-core 4GB
- Cost approximately $50/month
- Supports 24/7 operation

### Monitoring and Alerts

The system provides multiple monitoring methods:
- 📱 Telegram Bot notifications
- 📧 Email alerts
- 📊 Web dashboard
- 📝 Detailed logging

### Risk Disclaimer

⚠️ **Important Notice**: 
- This system is for educational and research purposes only
- Forex trading involves significant risk and may result in capital loss
- Use at your own risk with full understanding
- Recommended to test on demo accounts first

### License

MIT License - See [LICENSE](LICENSE) file for details

### Contributing

Issues and Pull Requests are welcome to improve the system.

### Contact

For questions or suggestions:
- Submit GitHub Issues
- Email project maintainers

---

**Disclaimer**: This software is provided "as is" without any express or implied warranties. Users assume all risks associated with trading using this software.
