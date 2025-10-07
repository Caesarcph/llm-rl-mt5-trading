# Requirements Document

## Introduction

本项目旨在开发一个基于EA31337框架优化的个人MT5智能交易系统，结合大语言模型(LLM)和强化学习(RL)技术，为个人投资者提供一个实用、可落地的自动化交易解决方案。系统采用分层架构设计，包含基础执行层(MT5+EA31337)、Python控制层(策略管理+智能决策)和数据层，支持多品种交易和智能风险管理。

## Requirements

### Requirement 1

**User Story:** 作为个人投资者，我希望系统能够基于EA31337框架提供稳定的交易执行能力，以便我可以利用成熟的交易基础设施进行自动化交易。

#### Acceptance Criteria

1. WHEN 系统启动时 THEN 系统 SHALL 成功连接到MT5平台并加载EA31337框架
2. WHEN EA31337框架加载时 THEN 系统 SHALL 能够访问30+内置策略作为基础信号源
3. WHEN 系统运行时 THEN 系统 SHALL 支持多周期数据处理(M1到D1)
4. WHEN 参数需要调整时 THEN 系统 SHALL 能够动态更新.set配置文件
5. WHEN 订单执行时 THEN 系统 SHALL 通过EA31337进行订单管理和仓位控制

### Requirement 2

**User Story:** 作为交易者，我希望系统能够集成Python层的智能决策能力，以便获得更高级的市场分析和策略优化。

#### Acceptance Criteria

1. WHEN 市场数据更新时 THEN Python层 SHALL 能够实时获取MT5数据并进行处理
2. WHEN 需要策略决策时 THEN 系统 SHALL 运行LLM分析模块进行市场情绪和新闻事件分析
3. WHEN 策略权重需要调整时 THEN RL模块 SHALL 基于历史表现动态优化策略权重
4. WHEN 多个信号产生时 THEN 系统 SHALL 通过Python Bridge协调EA31337实例
5. WHEN 系统运行时 THEN Python层 SHALL 提供高级风控和组合风险评估

### Requirement 3

**User Story:** 作为投资者，我希望系统支持多品种交易配置，以便我可以在不同市场中分散投资风险。

#### Acceptance Criteria

1. WHEN 配置EURUSD交易时 THEN 系统 SHALL 使用MA交叉和MACD策略，单笔风险控制在2%
2. WHEN 配置XAUUSD交易时 THEN 系统 SHALL 调整为高波动参数，降低仓位并监控避险事件
3. WHEN 配置USOIL交易时 THEN 系统 SHALL 集成EIA库存数据和OPEC会议事件监控
4. WHEN 配置股指交易时 THEN 系统 SHALL 关注VIX指数和财报季调整
5. WHEN 多品种同时交易时 THEN 系统 SHALL 控制相关品种总仓位不超过40%

### Requirement 4

**User Story:** 作为用户，我希望系统具备智能Agent决策能力，以便获得全面的市场分析和风险管理。

#### Acceptance Criteria

1. WHEN 市场分析Agent运行时 THEN 系统 SHALL 提供技术面分析和基本面LLM分析
2. WHEN 风险管理Agent监控时 THEN 系统 SHALL 实时计算VaR和最大回撤指标
3. WHEN 执行优化Agent工作时 THEN 系统 SHALL 选择最佳入场时机并管理订单拆分
4. WHEN 重大新闻事件发生时 THEN LLM Agent SHALL 提供情绪分析和交易建议
5. WHEN 异常行情出现时 THEN 系统 SHALL 自动分析原因并调整策略

### Requirement 5

**User Story:** 作为交易者，我希望系统提供本地化的LLM集成方案，以便在保护隐私的同时获得智能分析能力。

#### Acceptance Criteria

1. WHEN 系统初始化时 THEN 系统 SHALL 加载本地Llama 3.2模型(1B/3B版本)
2. WHEN 新闻分析需求产生时 THEN LLM模块 SHALL 进行情绪分析和事件解读
3. WHEN 每日交易结束时 THEN 系统 SHALL 生成市场总结分析报告
4. WHEN 异常行情发生时 THEN LLM SHALL 提供原因解析和应对建议
5. WHEN 资源受限时 THEN 系统 SHALL 优化LLM调用频率以平衡性能和分析质量

### Requirement 6

**User Story:** 作为投资者，我希望系统具备强化学习优化能力，以便持续改进交易策略表现。

#### Acceptance Criteria

1. WHEN RL环境初始化时 THEN 系统 SHALL 定义包含价格、指标、持仓的状态空间
2. WHEN RL训练时 THEN 系统 SHALL 使用PPO或SAC算法进行策略优化
3. WHEN 策略权重需要调整时 THEN RL模块 SHALL 基于收益和风险调整的奖励函数进行学习
4. WHEN 历史数据回测时 THEN 系统 SHALL 使用RL模型预测最优动作
5. WHEN 实盘交易时 THEN RL模块 SHALL 持续学习并优化决策策略

### Requirement 7

**User Story:** 作为个人投资者，我希望系统提供完善的风险控制机制，以便保护我的资金安全。

#### Acceptance Criteria

1. WHEN 单笔交易时 THEN 系统 SHALL 限制最大亏损不超过账户的2%
2. WHEN 日交易结束时 THEN 系统 SHALL 确保日最大亏损不超过5%
3. WHEN 连续亏损发生时 THEN 系统 SHALL 在连续3笔亏损后暂停交易24小时
4. WHEN 周亏损超过5%时 THEN 系统 SHALL 自动降低仓位50%
5. WHEN 月亏损超过10%时 THEN 系统 SHALL 停止交易并要求人工干预

### Requirement 8

**User Story:** 作为用户，我希望系统提供分层回测和优化功能，以便验证策略有效性并持续改进。

#### Acceptance Criteria

1. WHEN 进行Level 1回测时 THEN 系统 SHALL 使用EA31337 Docker进行快速批量回测
2. WHEN 进行Level 2回测时 THEN Python层 SHALL 加入LLM信号和ML预测进行精细回测
3. WHEN 进行Level 3验证时 THEN 系统 SHALL 支持最小手数实盘测试2-4周
4. WHEN 参数优化时 THEN 系统 SHALL 使用贝叶斯优化和遗传算法最大化夏普比率
5. WHEN 回测完成时 THEN 系统 SHALL 提供详细的性能报告和风险指标

### Requirement 9

**User Story:** 作为投资者，我希望系统支持渐进式资金管理，以便安全地扩大交易规模。

#### Acceptance Criteria

1. WHEN 系统初始运行时 THEN 系统 SHALL 仅使用10%资金进行测试
2. WHEN 系统稳定运行后 THEN 系统 SHALL 允许增加到30%资金运行
3. WHEN 系统验证成功后 THEN 系统 SHALL 支持最多50%资金上限交易
4. WHEN 任何阶段 THEN 系统 SHALL 保留至少50%资金作为储备
5. WHEN 风险增加时 THEN 系统 SHALL 能够自动降低资金使用比例

### Requirement 10

**User Story:** 作为用户，我希望系统提供简化的部署和运维方案，以便个人投资者能够轻松使用。

#### Acceptance Criteria

1. WHEN 本地部署时 THEN 系统 SHALL 支持Windows 10/11环境，最低4核8GB配置
2. WHEN 云端部署时 THEN 系统 SHALL 支持2核4GB VPS配置，成本控制在$50/月内
3. WHEN 系统监控时 THEN 系统 SHALL 提供Telegram Bot推送和邮件告警
4. WHEN 系统运行时 THEN 系统 SHALL 生成每日收益报告和周策略分析
5. WHEN 系统维护时 THEN 系统 SHALL 提供简单的配置界面和日志管理功能