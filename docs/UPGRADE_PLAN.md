# llm-rl-mt5-trading 升级计划：拆分、迁移、归档

> 本文是 2026 年三仓库大升级的一部分。
> - 总计划（现状体检、目标架构、时间表）：[multi-agent-quant · docs/roadmap](https://github.com/Caesarcph/multi-agent-quant/blob/master/docs/roadmap/README.md)
> - 回测引擎路线图：[llm-backtest-framework · ROADMAP.md](https://github.com/Caesarcph/llm-backtest-framework/blob/master/ROADMAP.md)
>
> 编写日期：2026-11-03

---

## 结论先行

**这个仓库不再单独演进。** 有价值的模块迁到两个地方，迁完以后把仓库归档：

| 去向 | 迁过去的东西 |
|---|---|
| **multi-agent-quant**（新的 `maq-quantlab`、`maq-risk`、`maq-data` 等包） | 经典策略、RL、经济日历、品种配置、熔断器、移动止损、分阶段资金管理、告警 |
| **llm-backtest-framework**（`llmbt`） | 蒙特卡洛 / 压力测试、绩效指标、报告生成、基线策略 |

**为什么不原地修：**

1. **修好它的成本和迁移差不多。** 主程序起不来（`src/core/monitoring.py` 损坏）、依赖装不上（`requirements.txt` 里有 `sqlite3`）、还有几处真实 bug，见下文"必须先修的问题"。
2. **和 multi-agent-quant 大面积重复。** MT5 连接、风控、回测、报告、LLM 封装，两边各写了一套，长期维护两套不现实。
3. **核心架构已经过时。** 通过 JSON 文件轮询和 EA31337 通信、用 1B 参数的本地模型做市场分析，这两条路线都不值得继续投入：
   - EA31337 上一个稳定版是 2023 年 8 月的 v2.013.1，之后基本处于半停更状态；
   - 仓库里其实没有任何 MQL5 代码去读 `signals.json` / `commands.json`，桥的另一头是空的；
   - `send_command` 每次都覆盖同一个命令文件，EA 还没读到第一条，第二条就可能把它盖掉。

---

## 必须先修的问题（阶段 0，第 1 周）

就算要归档，迁移期间也得让代码能 import、能跑测试，否则没法验证迁移前后行为一致。

| ID | 问题 | 位置 | 修法 |
|---|---|---|---|
| P0-04 | 三个文件内容损坏（字符顺序被打乱） | `src/core/monitoring.py`（第 60 行以后）、`tests/test_alert_system.py`、`examples/backtest_monte_carlo_integration.py` | 先找本地有没有好的副本；没有就把 `monitoring.py` 重写成最小可用版，另外两个直接删掉 |
| P0-05 | 依赖装不上 | `requirements.txt` | 删掉 `sqlite3`、`asyncio`（标准库；PyPI 上的 `asyncio` 是个"请勿安装"的占位包）和 `talib-binary`（2019 年后没更新，只有 cp34–cp37 的 wheel）。TA-Lib 官方包的新版本已经自带各平台 wheel，直接换成 `TA-Lib` 即可。另外 `langchain>=0.1` 现在会解析到 LangChain 1.x，接口不兼容，代码里没用到就直接删掉 |
| P0-06 | 异常处理装饰器自己会崩 | `src/core/exceptions.py:296` | `wrapper` 里的 `from .logging import log_error` 遮住了同名参数，捕获异常时先抛 `UnboundLocalError`。把 import 挪到模块顶部并改名 |
| P0-07 | 声称 MIT 但没有 LICENSE | 仓库根目录 | 补上 LICENSE 文件 |
| — | 测试里引用了未定义的名字 | `tests/test_ea31337_bridge.py:466`（`logging`）、`tests/test_risk_manager_agent.py`（`PortfolioRisk`） | 补 import |

修完的验收标准：`python -c "import src.core.trading_system"` 不报错；装好 TA-Lib 和 gymnasium 后，测试能全部收集到。

---

## 模块去留清单

`src/` 下一共 46 个模块、约 2.5 万行，处理方式分三类：

| 处理方式 | 文件数 | 行数 | 含义 |
|---|---|---|---|
| ✅ 迁移 | 16 | ≈ 9.1K | 思路和大部分逻辑保留，迁到新包时顺手清理（类型注解、Pydantic 模型、测试） |
| 🔁 重写 | 11 | ≈ 7.1K | 想法有价值，但实现要换（比如换成 Polars、Optuna、新的成本模型） |
| 🗑️ 删除 | 19 | ≈ 9.2K | 被新平台的基础设施取代，或者路线已经放弃 |

### 逐个模块

| 模块 | 行数 | 处理 | 去向 | 说明 |
|---|---|---|---|---|
| `agents/market_analyst.py` | 719 | ✅ | `maq-quantlab.features` | 趋势、波动率、相关性的确定性计算，作为量化分析师的工具 |
| `agents/risk_manager.py` | 600 | ✅ | `maq-risk` | VaR、回撤分析并入组合层风控 |
| `agents/execution_optimizer.py` | 984 | 🔁 | `maq-engine.tca` | 改成事后的交易成本分析（滑点、成交时机统计），先不做实时"优化" |
| `agents/llm_analyst.py` | 1081 | 🗑️ | — | 由研究部 2.0 的宏观 / 新闻分析师接替（API 大模型 + 结构化输出） |
| `bridge/ea31337_bridge.py` | 321 | 🗑️ | — | JSON 文件轮询，由 MT5 Gateway 取代 |
| `bridge/set_config_manager.py` | 504 | 🗑️ | — | EA31337 的 `.set` 文件管理，路线放弃 |
| `core/alert_system.py` | 497 | ✅ | `maq-engine.notify` | Telegram / 邮件告警，加上 `/status`、`/pause`、`/kill` 指令 |
| `core/config.py` | 320 | 🗑️ | — | 由 `maq-core` 的统一配置取代 |
| `core/economic_event_monitor.py` | 566 | ✅ | `maq-data.plugins` + `maq-risk` | 经济日历、EIA、OPEC 事件；驱动"重大数据前后禁开仓" |
| `core/environment.py` | 276 | 🗑️ | — | 并入 `maq doctor` |
| `core/exceptions.py` | 377 | ✅ | `maq-core` | 精简后迁移，顺手修掉装饰器 bug |
| `core/fund_manager.py` | 331 | ✅ | `maq-risk.staging` | 分阶段放量（测试 → 稳定 → 完全），和上线门槛检查合并 |
| `core/logging.py` | 290 | 🗑️ | — | 改用 structlog |
| `core/models.py` | 318 | 🗑️ | — | 由 `maq-core` 领域模型取代 |
| `core/monitoring.py` | 392 | 🔁 | `maq-engine` | 文件已损坏，改用 OpenTelemetry 指标重写 |
| `core/order_executor.py` | 544 | 🔁 | `maq-engine.oms` | 并入订单状态机；滑点模型交给 llmbt 的成本模型 |
| `core/position_manager.py` | 629 | ✅ | `maq-risk` / `maq-engine` | 移动止损、保本止损 |
| `core/risk_control_system.py` | 627 | ✅ | `maq-risk.breakers` | 熔断器逻辑 |
| `core/symbol_config_manager.py` | 455 | ✅ | `maq-data.symbols` | 品种元数据：交易时段、事件敏感度；`config/symbols/*.yaml` 一并迁移 |
| `core/trade_recorder.py` | 917 | 🗑️ | — | 由事件日志 + 控制台的交易日志页面取代，字段定义参考迁移 |
| `core/trading_system.py` | 705 | 🗑️ | — | 由 `maq-engine` 的编排器取代 |
| `data/cache.py` | 457 | 🗑️ | — | 不再需要 Redis；LLM 缓存交给 llmbt |
| `data/database.py` | 776 | 🗑️ | — | 由 SQLite 事件存储 + DuckDB 取代 |
| `data/models.py` | 121 | 🗑️ | — | 由 `maq-core` 取代 |
| `data/mt5_connection.py` | 382 | 🗑️ | — | 由 MT5 Gateway 取代 |
| `data/pipeline.py` | 527 | 🔁 | `maq-data.pipeline` | 思路保留，改成 Parquet + DuckDB |
| `data/validator.py` | 293 | ✅ | `maq-data.quality` | 数据质量检查（缺口、重复、异常跳变） |
| `llm/call_optimizer.py` | 590 | 🔁 | `maq-agents` + `llmbt` | 自适应限速 → 模型路由；结果缓存 → llmbt 录制回放 |
| `llm/llama_model.py` | 238 | 🗑️ | — | 本地模型改走 OpenAI 兼容接口（见下文） |
| `llm/model_manager.py` | 216 | 🗑️ | — | 同上 |
| `rl/rl_trainer.py` | 501 | ✅ | `maq-quantlab.rl` | 保留 SB3 训练骨架，加 walk-forward |
| `rl/rl_strategy_optimizer.py` | 456 | 🔁 | `maq-quantlab.rl` | 按新定位重写（见下文） |
| `rl/trading_env.py` | 488 | 🔁 | `maq-quantlab.rl` | 基于 llmbt 模拟器重写，换奖励函数 |
| `strategies/backtest.py` | 1261 | 🗑️ | — | 由 llmbt 取代 |
| `strategies/base_strategies.py` | 736 | ✅ | `llmbt.baselines` + `maq-quantlab` | 经典策略当作"LLM 必须跑赢的基线" |
| `strategies/custom_indicators.py` | 607 | 🔁 | `maq-data.indicators` | 用 Polars 重写，去掉 TA-Lib 依赖 |
| `strategies/indicators.py` | 626 | 🔁 | `maq-data.indicators` | 同上 |
| `strategies/monte_carlo.py` | 759 | ✅ | `llmbt.robustness` | 蒙特卡洛与压力测试 |
| `strategies/parameter_optimizer.py` | 636 | 🔁 | `maq-quantlab.optimize` | 遗传算法换成 Optuna |
| `strategies/performance_tracker.py` | 661 | ✅ | `llmbt.metrics` | 绩效指标 |
| `strategies/strategy_manager.py` | 756 | ✅ | `maq-quantlab` | 策略注册、信号聚合 |
| `strategies/weight_optimizer.py` | 586 | ✅ | `maq-quantlab` | 等权 / 风险平价 / 均值方差，作为 RL 的对照组 |
| `utils/diagnostic_tool.py` | 387 | 🗑️ | — | 并入 `maq doctor` |
| `utils/file_utils.py` | 333 | 🗑️ | — | 用标准库就够 |
| `utils/performance_profiler.py` | 326 | 🗑️ | — | 用 OpenTelemetry 或 py-spy |
| `utils/report_generator.py` | 1262 | 🔁 | `llmbt.report` | HTML 报告；合规检查的思路保留 |

### `src/` 以外的文件

| 路径 | 处理 |
|---|---|
| `config/symbols/*.yaml`、`config/economic_calendar.json` | ✅ 迁移到 `maq-data`（日历文件作为示例数据） |
| `config/*.yaml` 其余部分 | 🗑️ 由新的统一配置取代 |
| `demo_configs/*.set`、`demo_ea31337/` | 🗑️ EA31337 相关 |
| `docs/`（17 篇） | 挑有用的并入新文档站，其余删除 |
| `examples/`（19 个文件） | 挑 3 个改写成 marimo 交互式教程 |
| `tests/` | 跟着各自的模块迁移 |
| `main.py`、`install.py`、`validate_setup.py`、`diagnose.py` | 🗑️ 由 `maq` 命令行取代 |

---

## RL 部分：重新定位

### 现在的问题

| 问题 | 位置 | 说明 |
|---|---|---|
| RL 在学方向 | `rl/trading_env.py` 的动作空间是 HOLD / BUY / SELL / CLOSE | 金融数据信噪比太低，让 RL 直接学买卖方向，极容易过拟合 |
| 回撤惩罚会"粘住" | `_calculate_reward`：只要历史最大回撤超过 10%，之后**每一步**都扣 `max_drawdown × 0.5` | 惩罚和当前动作无关，梯度信号被污染 |
| 奖励塑形有偏 | 持有盈利仓位每步 +0.001 | 会诱导策略死拿盈利单 |
| 没有样本外验证 | 训练和评估没有按时间严格切分 | 回测好看不代表能用 |
| 环境太慢 | pandas 逐行计算 | 训练迭代慢 |

### 新定位："RL 只调油门，不握方向盘"

```text
研究部 / 经典策略 ──► 方向 + 止损
                         │
风控内核 ──────────────► 允许的最大手数 N_max
                         │
RL 仓位策略 ───────────► 系数 a ∈ [0, 1]
                         │
最终手数 = N_max × a（仍然要过风控的所有检查）
```

| 设计项 | 方案 |
|---|---|
| 动作 | 连续值 `a ∈ [0, 1]`，只决定"用多大比例的风险预算" |
| 观测 | 近期收益与波动、当前回撤、持仓、研究部给的 conviction、模型间分歧度、行情类型、是否处在重大数据窗口 |
| 奖励 | 差分夏普比率（Moody & Saffell），或"对数收益 − λ × 下行波动 − 交易成本"，只和当前动作的后果挂钩 |
| 环境 | 基于 llmbt 的模拟器，和回测、实盘用同一套成本模型；向量化，目标单核每秒 1 万步以上 |
| 训练 | Stable-Baselines3（PPO / SAC）+ Optuna 调参，walk-forward 滚动训练，多个随机种子 |
| 对照组 | 固定比例、波动率目标、截断 Kelly |
| 判定是否过拟合 | Deflated Sharpe Ratio、PBO（回测过拟合概率） |
| 上线规则 | 样本外**不差于**波动率目标基线才启用；启用后依然受风控硬上限约束 |
| 实验追踪 | MLflow |
| 版本 | Stable-Baselines3 2.9.x + sb3-contrib（RecurrentPPO 等），Gymnasium 1.4（注意 1.0 起 autoreset 语义和 VectorEnv 都变了，旧环境要按新接口改） |

### "策略权重优化"不用 RL

原来的 `rl_strategy_optimizer.py` 用 PPO 去学"几个策略各分多少权重"。问题在于这类数据的独立样本太少，RL 几乎必然过拟合。更合适的工具是：

- **在线分配器**：Hedge / 乘法权重，按近期表现调权重（和主平台的模型权重用同一套实现）；
- **上下文 bandit**：把行情类型当上下文，学"什么行情下信哪个策略"。

### 业界的实际共识（2025–2026）

- **RL 真正有用的地方**：订单执行与拆单、做市与对冲、在已有信号基础上做受约束的仓位 / 风险覆盖。
- **最容易过拟合的做法**：端到端从 K 线直接学买卖、单一货币对、奖励直接用原始盈亏、在测试区间上调超参。
- **FinRL 团队自己也转向了**：他们 2026 年的 [FinRL-X（FinRL-Trading）](https://github.com/AI4Finance-Foundation/FinRL-Trading) 把流程拆成"选股 → 配置 → 择时 → 风险覆盖"四段，深度强化学习只是"配置"环节里可替换的一种方法，和等权、均值方差并列。
- **一个值得警惕的负面结果**：ICML'26 的 *When Valid Signals Fail*（arXiv 2604.10996）发现，把样本外 IC 不错的 LLM 特征塞进 PPO 的观测里，在宏观冲击期间反而比只看价格的版本更差。所以任何 LLM 特征都要按行情分段做消融。

### 探索方向（P2，不进主线）

- **时序大模型只做波动率 / 区间 / 成交量特征，不做方向**。多篇 2025–2026 年的研究显示，Chronos、TimesFM 这类模型零样本预测收益率的效果很弱甚至为负，但预测波动率还有一定价值。选型要注意许可证：Chronos-2、TimesFM 2.5、Toto 是 Apache-2.0；**TimesFM 3.0 的权重不允许商用**；Kronos（专门针对 K 线训练）是 MIT。评估时要和 HAR、GARCH、LightGBM 这些传统方法对比。
- **让 LLM 离线设计特征和奖励项**（参考 [GIFT](https://github.com/KAG778/GIFT)）：LLM 只在离线阶段提出候选特征，验证有效后冻结，决策时没有 LLM 延迟。这是 LLM + RL 结合里最务实的一种。
- **小模型 GRPO 微调**：用自己积累的决策日志 + 未来收益当可验证奖励，微调一个小参数开源模型（参考 [Alpha-R1](https://github.com/FinStep-AI/Alpha-R1)，工具链可以用 TRL / Unsloth / verl）。成本高、极易数据泄漏，只有前两条路验证出信号之后再考虑。
- **离线 RL**：直接从历史决策日志里学，不需要模拟器。不过现有证据显示，Decision Transformer 一类方法相比 CQL / IQL 这些简单方法并没有稳定优势。

---

## LLM 部分

- 本地 Llama 3.2 1B **下线**。1B 模型做市场分析能力明显不够，为它引入的 llama-cpp-python + torch + transformers + LangChain 依赖又太重。
- 以后需要本地模型时，用 Ollama / vLLM / SGLang / llama.cpp server / LM Studio 起一个 **OpenAI 兼容接口**（这几个现在都支持按 JSON Schema 约束输出），由主平台的模型注册表统一接入，和云端模型用同一套代码。
- 截至 2026 年 11 月，4B–9B 这一档比较能打的开源模型有 Qwen3.5（4B / 9B）、Gemma 4（E4B）、Ministral 3 等；显存够的话可以看 gpt-oss-20b。具体选哪个，落地时用主平台的评测集跑一遍再定。

## EA31337 怎么办

- JSON 文件桥**下线**：轮询文件有竞态、延迟高，出了问题也没法复现。
- 如果以后还想用 MQL5 这一侧的能力（比如 MT5 自带的策略测试器、在 EA 里做本地低延迟止损），推荐两条路：
  1. 写一个极简 EA，作为 socket 客户端连到 Python 这边的 asyncio 服务（MQL5 原生 socket 只能当客户端，地址要加进终端的白名单），协议里带序号、确认和心跳，并且事先定好"心跳断了 EA 是平仓还是持仓不动"；
  2. 用 MQL5 原生的 ONNX 支持（build 3620 起就有），把训练好的模型直接部署进 EA。它在策略测试器里也能跑，等于 Python 训练、MT5 回测同一个模型。
- 另外，MT5 终端从 build 6060（2026 年 7 月）起**自带 MCP Server**，外部 AI 助手可以直接连上去查行情、管订单。它很适合拿来做运维和研究时的"对话式查询"，但**不要让它成为下单通道**——下单必须经过主平台的风控内核。

---

## 迁移顺序（对应总计划的阶段）

| 总计划阶段 | 这个仓库要做的事 |
|---|---|
| 阶段 0（第 1 周） | 修好上面"必须先修的问题"，打 `v0-legacy` 标签 |
| 阶段 2（第 3–5 周） | 迁移指标、数据质量检查、品种配置、经济日历 |
| 阶段 3（第 4–6 周） | 迁移熔断器、移动止损、分阶段资金管理、VaR |
| 阶段 5（第 6–11 周） | 迁移蒙特卡洛、绩效指标、报告、基线策略到 llmbt |
| 阶段 6（第 10–13 周） | 迁移策略管理、权重优化、RL（按新定位重写）、告警；最后归档 |

## 归档前检查清单

- [ ] 上表里每个"✅ 迁移 / 🔁 重写"的模块，在新位置都有对应实现和测试
- [ ] `config/symbols/*.yaml`、经济日历数据已经迁移
- [ ] README 顶部加上迁移说明和新仓库链接（模板见下）
- [ ] `v0-legacy` 标签已推送
- [ ] 在 GitHub 设置里把仓库设为 Archived

## 归档后的 README 模板

```markdown
> ⚠️ 本仓库已归档（2026）。
> - 交易系统、RL 实验室 → [multi-agent-quant](https://github.com/Caesarcph/multi-agent-quant)
> - 回测、蒙特卡洛、报告 → [llm-backtest-framework](https://github.com/Caesarcph/llm-backtest-framework)
> - 旧代码见 `v0-legacy` 标签。
```
