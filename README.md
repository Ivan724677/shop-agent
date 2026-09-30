# 风险感知售后客服 Agent

阶段六升级最终版 Agentic RAG 的完整架构图、设计解释、代码映射、运行方式和评测口径见：

[`README_AGENTIC_RAG_FINAL.md`](README_AGENTIC_RAG_FINAL.md)

这是一个面向电商售后场景的系统型 Agent 项目。当前完成了阶段一业务模拟器、阶段二 DeepSeek V4 Flash 单 Agent ReAct baseline、阶段三结构化状态与多轮记忆 Agent、阶段四 MCP 工具协议与可靠性层、阶段五确定性优先 Router 与多专家 Agent，以及阶段六 Agentic RAG 与政策证据推理。

阶段一的目标不是做一个更长的聊天 Demo，而是让后续 Agent 面对真实的业务约束：用户、商品、订单项、拆分物流、退货申请、退款流水和人工工单彼此独立，所有高风险写操作都必须经过服务端校验和状态转换。

## 当前能力

- `User`：姓名、手机号、邮箱、地址、会员等级和账户状态。
- `Product`：独立商品目录，和订单中的购买快照分离。
- `OrderItem`：记录购买时商品名、成交价、数量、商品属性、物流单和售后状态。
- `Shipment`：物流单号、承运商、物流状态、轨迹节点、预计送达和签收时间。
- `ReturnRequest`：独立退货/换货申请及状态时间线。
- `RefundTransaction`：独立退款流水，包含退款单号、金额、方式、到账状态和状态时间线。
- `Ticket`：人工工单，支持创建、分配、处理中、等待用户、解决和关闭。
- 正式状态机：订单、物流、退货、退款和工单都有合法转换表及非法转换保护。
- JSON 种子数据：业务数据不再散落在 Agent 代码中，可以重复初始化。
- 场景语料：手工核心场景、120 条固定随机种子生成场景和故障注入场景。
- 现有 Agent 仍然保留：商品级部分退货、政策证据、用户确认、幂等写入、未知提交恢复和 trace 监控。
- ReAct baseline：DeepSeek 原生 tool calls、工具观察回传、多步循环、原始对话历史、token usage 和工具 trace。
- 结构化 Agent：确定性信号与 DeepSeek 语义帧融合，显式维护槽位、任务阶段、风险等级、缺失字段、确认范围和多轮记忆。
- 对照评测：让 baseline 与结构化 Agent 在相同初始数据和相同场景上分别运行，统计最终状态、工具错误、不安全写入、调用次数和 token。
- MCP 工具层：统一 `tools/list`、`tools/call`、输入/输出 schema、scope 权限、会话身份注入和 JSON-RPC stdio 入口。
- 工具可靠性：真实执行超时、只读重试、熔断器、写操作幂等、未知提交对账和脱敏操作审计。
- 多专家路由：订单/物流、政策、交易、人工升级四个专家，规则先路由，复杂请求才调用模型复核。
- 专家安全边界：每个专家有独立工具白名单，交易专家必须复核冻结范围、政策证据、金额和确认状态。
- 路由可观测性：记录状态快照、规则原因、模型候选、专家链、协议错误和最终回退原因。
- 政策 RAG：提供 BM25、dense+BM25 混合检索、查询重写、语义评分、元数据过滤和证据校验对照。
- Agentic RAG：由 Planner 自主选择直接回答、检索、重写、验证、澄清或升级，可多步检索；安全护栏只限制不安全结果，不替代 Planner。
- 政策可靠性：按生效日期、版本、优先级和例外关系处理政策；过期证据和同级版本冲突不能进入最终回答。
- RAG 评测：覆盖例外政策、版本切换、同级冲突和无关问题，统计证据命中、冲突识别、拒绝过期证据、检索次数和无检索率。

## 目录

```text
agent-project/
├── agent.py
├── baseline/
│   ├── deepseek_client.py # DeepSeek OpenAI-compatible Chat Completions 客户端
│   ├── tool_catalog.py    # 工具 JSON Schema 和会话身份绑定执行器
│   ├── react_agent.py     # 单 Agent ReAct 工具循环
│   └── cli.py             # DeepSeek baseline 交互入口
├── structured/
│   ├── models.py          # 语义帧、槽位、任务状态、确认范围和记忆模型
│   ├── extractor.py       # 高精度 ID、否定范围和确认语句提取
│   ├── semantic_parser.py # DeepSeek 受约束语义帧解析
│   ├── reducer.py         # 确定性信号与 LLM 候选的状态归并
│   ├── state_machine.py   # 对话任务状态转换表
│   ├── policy.py          # 基于状态的确定性工具决策与故障恢复
│   ├── agent.py           # 阶段三主执行链、记忆和 trace
│   └── cli.py             # 结构化 Agent 交互入口
├── routing/
│   ├── models.py          # 路由决策、专家结果和二次路由候选契约
│   ├── rules.py           # 基于结构化状态的确定性路由和复杂度检测
│   ├── semantic_router.py # DeepSeek 复杂请求二次路由候选
│   └── router.py          # 主 Router、模型约束和专家结果校验
├── experts/
│   ├── base.py            # 专家协议、MCP 调用边界和工具白名单
│   ├── order_logistics.py # 订单/物流查询与商品实体落地
│   ├── policy.py          # 政策证据、资格与退款金额计算
│   ├── transaction.py     # 确认绑定、退货写入与未知提交对账
│   └── handoff.py         # 幂等人工工单与未知提交恢复
├── multi_agent/
│   ├── agent.py           # Router、专家链、记忆和 trace 装配
│   └── cli.py             # 阶段五交互入口
├── evaluation/
│   ├── comparison.py      # 同场景双版本回放与指标计算
│   ├── compare_cli.py     # 真实 DeepSeek 对比入口
│   ├── rag_comparison.py   # BM25/Hybrid/Agentic RAG 对照引擎
│   ├── rag_compare_cli.py  # 阶段六离线评测入口
│   └── rag_cases.json      # 例外、版本冲突和无关问题评测集
├── rag/
│   ├── models.py           # 政策文档、检索、Planner、证据校验契约
│   ├── bm25.py             # Okapi BM25 长度归一化 sparse scorer
│   ├── retrieval.py        # BM25 与离线 Hybrid 检索
│   ├── business_filter.py  # 发布/生效期/商品/原因/权限/冲突过滤
│   ├── grading.py          # Heuristic/DeepSeek Document Grader
│   ├── graph.py            # 显式 Agentic RAG 节点图和状态
│   ├── rewrite.py          # 查询扩展和多步检索重写
│   ├── evidence.py         # 生效期、优先级、例外和冲突校验
│   ├── planner.py          # 离线 Planner 与 DeepSeek 结构化 Planner
│   ├── agent.py            # Agentic RAG 主循环、护栏、缓存和 trace
│   └── cli.py              # 阶段六交互入口
├── mcp_server/
│   ├── catalog.py         # MCP 工具目录、input/output schema 与权限元数据
│   ├── validation.py      # 确定性 JSON Schema 边界校验
│   ├── models.py          # 会话权限、熔断状态与审计模型
│   ├── server.py          # tools/list、tools/call 和可靠性执行管线
│   ├── client.py          # Agent 使用的进程内 MCP Client
│   └── cli.py             # JSON-lines stdio MCP Server 入口
├── domain/
│   ├── models.py          # 用户、商品、订单、物流、退货、退款、工单
│   └── state_machine.py   # 正式状态枚举和合法转换表
├── repositories/
│   └── in_memory.py       # 业务仓储和带守卫条件的写操作
├── seed/
│   ├── data/              # JSON 业务数据和售后政策
│   └── loader.py          # JSON 到领域对象的加载器
├── scenarios/
│   ├── golden/            # 人工审阅的核心回归场景，一个文件一个场景
│   ├── generated/         # 批量生成的 JSONL 场景
│   ├── fault_injection/   # 工具超时、未知提交、非法响应等故障场景
│   ├── scenario_schema.py # 场景格式和校验
│   ├── loader.py          # 场景语料加载器
│   └── generate.py        # 固定种子批量生成器
├── test_agent.py          # 原有 Agent 回归测试
├── test_baseline.py       # ReAct、工具错误和 DeepSeek 请求契约测试
├── test_stage_one.py      # 领域模型、状态机、工具和语料测试
├── test_stage_three.py    # 多轮状态、确认范围、冲突与故障恢复测试
├── test_stage_four.py     # MCP、权限、超时、重试、熔断、幂等与审计测试
├── test_stage_five.py     # 路由、专家边界、模型回退与多 Agent 故障测试
└── test_comparison.py     # baseline 失效注入与结构化 Agent 对照测试
```

## 运行

项目只使用 Python 标准库，Python 3.10+ 可运行。

```bash
cd "/Users/ivan/Documents/Codex/2026-07-16/wo/agent-project"
python3 -m unittest -v
```

运行正常退货流程：

```bash
python3 agent.py --demo
```

模拟“退货请求已经提交，但响应超时”的高风险故障：

```bash
python3 agent.py --demo --inject-timeout
```

系统会使用幂等键查询原请求，不会盲目重复创建退货申请。

校验当前场景语料：

```bash
python3 -m scenarios.validate
```

重新生成 300 条可重复的批量场景：

```bash
python3 -m scenarios.generate \
  --count 300 \
  --seed 20260805 \
  --output scenarios/generated/batch_300.jsonl
```

进入交互模式：

```bash
python3 agent.py
```

可以尝试：

```text
我要退订单 O10086 里的鞋，耳机不退
确认退货
我要退订单 O10087 里的马克杯
帮我查询订单 O10086 的物流
trace
exit
```

## 阶段二：DeepSeek 单 Agent ReAct baseline

官方 API 模型 ID 为 `deepseek-v4-flash`，基础地址为 `https://api.deepseek.com`。baseline 使用 Chat Completions 原生 tool calls，并关闭 thinking mode，让执行链保持简单、成本低且容易作为后续实验对照组。

先在当前终端设置环境变量。不要把真实 key 写入 `.env.example`、README 或代码：

```bash
export DEEPSEEK_API_KEY="你的新 API key"
export DEEPSEEK_MODEL="deepseek-v4-flash"
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
```

启动 baseline：

```bash
python3 -m baseline.cli --user-id U001
```

交互命令：

```text
trace    查看模型决策和工具调用轨迹
history  查看发送给模型的原始消息历史
audit    查看 MCP 工具操作审计
reset    重置当前会话
exit     退出
```

baseline 的执行链只有：

```text
用户消息
→ DeepSeek 决定直接回复或调用工具
→ MCP Client 调用 tools/call
→ MCP Server 完成权限、schema 与可靠性控制
→ 本地 ToolGateway 执行业务工具
→ 工具结果作为 tool message 返回 DeepSeek
→ DeepSeek 继续调用工具或生成最终回答
```

它刻意没有使用阶段一 `TaskState` 中的意图、槽位、目标商品、排除商品、政策证据、确认状态等结构化字段，也没有摘要记忆、路由器或故障恢复规划。它只保留原始消息历史，并设置最多 6 次模型决策和每步最多 4 个工具调用，防止失控循环。

有两类底层约束仍然保留，因为它们属于业务安全边界，不属于 Agent 智能：

- 当前用户身份由会话注入，模型不能伪造 `user_id` 访问其他用户订单。
- 高风险写操作仍由仓储层校验确认、商品归属、政策、金额和幂等键。

这使 baseline 可以真实暴露意图误判、工具选错、参数不完整、多轮历史污染和工具链规划失败，同时不会为了制造 baseline 差异而主动移除最基本的鉴权和资金安全约束。

## 阶段三：结构化状态与多轮记忆

阶段三不是用正则替换 LLM，也不是继续让 LLM 自由决定动作，而是把两者放在不同的权限层：

```text
用户消息
→ 确定性提取器：订单号、商品号、精确确认、否定范围
→ DeepSeek 语义解析器：自然语言意图、模糊商品指代、咨询语气
→ StateReducer：按来源优先级归并候选，记录冲突
→ StructuredTaskState：槽位、阶段、风险、缺失字段、确认范围
→ DialoguePolicy：依据状态确定下一步工具或追问
→ ToolGateway：执行并返回可审计事实
→ 状态更新与回答
```

LLM 输出的是 `SemanticFrame` 候选，不直接授权工具调用。订单号、商品号、明确排除和精确确认优先采用确定性结果；工具返回的数据优先于语言猜测；高风险写操作必须同时满足政策、金额、商品范围和绑定确认。

当前显式任务状态包括：

- 意图：订单查询、物流查询、退货、转人工；
- 阶段：`idle`、`collecting_info`、`resolving_entities`、`checking_policy`、`awaiting_confirmation`、`executing`、`completed`、`rejected`、`handoff`；
- 槽位：值、来源、置信度、证据文本和最后更新时间；
- 当前订单、选中商品、排除商品、原因和退款金额；
- 缺失字段、风险等级、政策证据和解析冲突；
- `PendingAction`：绑定订单、商品集合、金额、原因、请求轮次和幂等键。

记忆分为三层：工作记忆保存当前任务状态，事实记忆保存工具观测，情节记忆保存有限长度的每轮输入、语义帧、前后状态、工具和回复。发送给语义模型的是最小状态上下文，不是无限增长的原始聊天记录。

不调用 API 调试状态机：

```bash
python3 -m structured.cli --offline --user-id U001
```

接入 DeepSeek 运行完整结构化 Agent：

```bash
export DEEPSEEK_API_KEY="你的新 API key"
python3 -m structured.cli --user-id U001
```

可以用以下多轮输入观察状态：

```text
我要退鞋
订单号是 O10086
state
确认退货
memory
trace
```

`state` 查看当前工作记忆，`memory` 查看每轮状态变化，`trace` 查看解析、状态归并、工具和策略轨迹。

### baseline 与结构化版本对比

真实 DeepSeek 对比会为每个版本创建独立的业务数据副本，并在同一批 golden scenarios 上回放：

```bash
python3 -m evaluation.compare_cli \
  --limit 8 \
  --output outputs/stage3-comparison.json
```

输出包括：最终状态准确率、不安全写入率、平均工具调用数、平均模型请求数、prompt/completion tokens，以及每个场景的回复、工具错误和违规明细。

`test_comparison.py` 另有一个确定性“模型提前确认”故障注入，用来证明评测器能捕获 baseline 的 premature write，并验证结构化版本不会把 LLM 候选确认直接变成写权限。它是安全机制的回归实验，不冒充真实 DeepSeek 模型跑分。真实模型结果只由上述对比命令产生。

完整设计、状态不变量和实验解释见 [`docs/stage3-structured-state.md`](docs/stage3-structured-state.md)。

## 阶段四：MCP 与工具可靠性层

阶段四把工具从 Agent 内部函数升级成独立信任边界。阶段二和阶段三现在共用同一个执行链：

```text
Agent / LLM
→ model-facing OpenAI tool schema
→ InProcessMCPClient
→ MCP tools/call
→ input schema validation
→ authenticated identity + scope authorization
→ circuit breaker
→ timeout / retry policy
→ ToolGateway + repository business guards
→ output schema validation
→ audit event
→ MCP result / UNKNOWN_COMMIT
```

工具定义只有一个权威来源：[`mcp_server/catalog.py`](mcp_server/catalog.py)。同一份定义生成 DeepSeek 使用的 function tools 和 MCP `tools/list` 返回值，并携带：

- 输入与输出 JSON Schema；
- 必需权限 scope；
- 风险等级和是否有副作用；
- timeout 与最大尝试次数；
- 幂等键字段；
- 未知提交时应使用的对账工具。

### 可靠性策略

- 只读工具：瞬态 timeout 或非法响应可以做有限指数退避重试，默认最多 2 次尝试。
- 写工具：不自动重试。执行超时或响应丢失统一返回 `UNKNOWN_COMMIT`，并给出 `reconciliation_tool`。
- 熔断器：一个工具连续出现瞬态失败后进入 `open`，恢复窗口后只允许一次 `half_open` 探测。
- 幂等：退货和工单都要求幂等键；同键同参数返回原记录，同键不同参数返回 `IDEMPOTENCY_KEY_REUSE`。
- 权限：`user_id` 来自认证会话，模型不能传入或覆盖；工具还需要 `orders:read`、`returns:write` 等 scope。
- 审计：每个逻辑工具调用记录 actor、request/correlation ID、决策、尝试次数、耗时、熔断状态和参数摘要。描述等敏感字段被脱敏，幂等键只保留哈希。
- 双层校验：MCP 层校验协议、权限与 schema；仓储层再次校验订单归属、政策、金额、确认和幂等，不能只依赖网关。

启动 JSON-lines stdio MCP Server：

```bash
python3 -m mcp_server.cli --user-id U001
```

支持的 MCP JSON-RPC 方法是 `initialize`、`ping`、`tools/list`、`tools/call` 和 `notifications/initialized`。当前阶段聚焦工具协议，不宣称实现 resources、prompts、远程 OAuth 或所有 MCP transport。

在结构化 Agent 中输入 `audit` 可以查看 MCP 审计：

```bash
python3 -m structured.cli --offline --user-id U001
```

故障与边界测试：

```bash
python3 -m unittest -v test_stage_four.py
```

完整设计、不变量、重试边界和未知状态处理见 [`docs/stage4-mcp-reliability.md`](docs/stage4-mcp-reliability.md)。

## 阶段五：Router 与多个专家 Agent

阶段五保留阶段三单策略 Agent，不原地替换它。新增版本的执行链是：

```text
用户输入
→ 确定性提取 + DeepSeek 语义候选
→ StateReducer 形成结构化状态
→ Router 先按意图、缺失字段、确认状态和风险规则路由
→ 复杂/冲突/多意图时调用 DeepSeek 二次路由
→ 订单/物流 → 政策 → 交易，或人工升级
→ 所有工具仍经过同一个 MCP 权限与可靠性边界
```

四个专家的权限不是提示词约定，而是代码白名单：

| 专家 | 允许工具 | 不能做什么 |
|---|---|---|
| 订单/物流 | `list_orders`、`get_order`、`list_shipments` | 不能检索政策或写退货 |
| 政策 | `search_policy`、`calculate_refund` | 不能把“符合政策”当成已确认 |
| 交易 | `create_return_request`、`get_return_status` | 不能缺少政策证据、金额或确认范围 |
| 人工升级 | `create_ticket`、`get_ticket_status` | 不能代替交易专家创建退货 |

模型 Router 只生成候选。它不能把未知请求直接路由到交易执行，也不能删掉退货链中的订单校验和政策校验。专家输出还要通过 `ExpertResult` 协议校验；专家身份不匹配、非法下一跳、缺少回复或缺少升级原因都会被拒绝并转人工。

启动阶段五在线版本：

```bash
python3 -m multi_agent.cli --user-id U001
```

不调用模型的确定性调试模式：

```bash
python3 -m multi_agent.cli --offline --user-id U001
```

`routes` 查看每轮路由决策；`trace` 查看解析、路由、专家和工具完整链路；`audit` 查看 MCP 审计。真实 DeepSeek 三版本对比仍使用：

```bash
python3 -m evaluation.compare_cli --limit 8 --output reports/comparison.json
```

该命令现在同时回放 `react_baseline`、`structured_state` 和 `multi_expert`。阶段五专项测试：

```bash
python3 -m unittest -v test_stage_five.py
```

完整路由规则、安全不变量、失败案例和面试表达见 [`docs/stage5-router-experts.md`](docs/stage5-router-experts.md)。

## 阶段六：Agentic RAG 与政策推理

阶段六把“查政策”从一个固定工具调用升级成可评测的知识推理环节。系统同时保留三个对照版本：

| 版本 | 行为 | 目的 |
|---|---|---|
| BM25 | Okapi BM25 sparse retrieval | 观察长度归一化后的精确词项召回 |
| Hybrid | 真实 dense embedding + BM25 加权融合 | 观察语义和精确词项信号互补效果 |
| Agentic | 显式 Graph + Planner + 双层 Grader + 多步检索 | 观察“是否需要查、如何补查、何时停止”对可靠性的影响 |

Agentic RAG 的真实执行链是：

```text
用户问题
→ Guardrail Node 判断政策敏感性和缺失信息
→ Planner Node 选择 answer/retrieve/rewrite/verify/clarify/escalate
→ Retrieve Node 执行 BM25 或 dense+BM25 hybrid
→ Business Filter 检查发布、生效期、商品、原因、权限和冲突
→ Grade Node 判断语义相关性、事实维度覆盖和噪声召回
→ Rewrite Node 记录缺失维度、补充词、新证据和覆盖改善
→ Verify Node 由 EvidenceValidator 复核优先级、例外和版本冲突
→ Generate Node 生成结构化 grounded answer
→ Citation Node 校验 claim/citation，失败则 Escalate Node 转人工
```

这里的 Planner 不是把固定 `if/else` 改名为 Agent。在线模式由 DeepSeek 通过 `emit_rag_plan` 结构化函数选择下一步，下一轮会重新观察已召回证据，因此可以出现 `retrieve → rewrite_query → verify → answer_directly` 的多步轨迹。离线启用 `HeuristicRAGPlanner`，只是为了让相同轨迹可重复回放、测试和做消融实验。

代码护栏不代替 Planner 的决策，只约束决策的安全边界：政策敏感问题没有验证证据时禁止直接回答；检索次数和总步数有限；冲突不能猜版本；最终暴露给上层的 `evidence_ids` 只来自校验通过的证据。原始召回仍保存在 trace 中，方便定位“召回了但为什么不能引用”。

政策文档包含 `effective_from`、`effective_to`、`version`、`priority`、`policy_family` 和 `exception_of`。例如定制商品质量问题会选择质量例外规则，而不会把定制商品无理由退货限制错误地作为结论；同一政策族、同一生效日、同一优先级但正文不同的版本会输出 `POLICY_VERSION_CONFLICT` 并进入 `UNCERTAIN`。

运行阶段六离线交互：

```bash
python3 -m rag.cli --mode agentic --offline --product-type custom --reason quality_issue
python3 -m rag.cli --mode bm25
python3 -m rag.cli --mode hybrid
```

运行对照评测（不需要 API key）：

```bash
python3 -m evaluation.rag_compare_cli
python3 -m evaluation.rag_compare_cli \
  --case-id exception_01 \
  --case-id conflict_01 \
  --output reports/rag_comparison.json
```

默认评测集包含 60 条静态、逐条带预期与审核说明的待领域审核场景，覆盖普通/定制商品、质量问题、无理由退货、版本切换、例外、模糊表达、同义改写、对抗问题、过期政策、版本冲突和无需检索问题。输出统计原始召回命中率、最终证据命中率、accepted evidence rate、冲突识别率、平均检索次数、无检索率、Planner/Grader 调用次数、rewrite rate、rewrite success rate、grounded rate 和 uncertainty rate。

接入 DeepSeek 在线 Planner：

```bash
export DEEPSEEK_API_KEY="你的新 API key"
export DEEPSEEK_MODEL="deepseek-v4-flash"
python3 -m rag.cli --mode agentic
```

阶段六专项测试：

```bash
python3 -m unittest -v test_stage_six.py
```

完整设计、失败模式、评测解释和面试表达见 [`docs/stage6-agentic-rag.md`](docs/stage6-agentic-rag.md)。

## 场景格式

每条场景同时保存用户输入、初始业务状态、期望意图、必需槽位、允许动作、禁止动作、期望最终状态和风险等级。它不是普通的聊天样例，而是后续端到端评测的 ground truth。

```json
{
  "scenario_id": "golden_partial_return_multiple_items",
  "user_id": "U001",
  "conversation": [
    {"role": "user", "content": "我要退订单 O10086 里的鞋，耳机不退"},
    {"role": "user", "content": "确认退货"}
  ],
  "business_state": {
    "order_id": "O10086",
    "target_item_id": "I002",
    "excluded_item_id": "I001",
    "refund_amount": 129.0
  },
  "allowed_actions": [
    "get_order",
    "search_policy",
    "calculate_refund",
    "create_return_request"
  ],
  "forbidden_actions": [
    "refund_excluded_item",
    "refund_entire_order"
  ],
  "expected_final_state": "RETURN_REQUEST_SUBMITTED",
  "risk_level": "high"
}
```

场景分为三层：

1. `golden/`：人工维护的关键回归案例，适合审阅和逐条调试。
2. `generated/`：由固定模板和随机种子生成的批量案例，适合统计评测。
3. `fault_injection/`：明确描述工具故障及系统应该采取的恢复动作。

## 业务关系

```text
User
└── Order[]
    ├── OrderItem[] ── Product snapshot + Shipment
    ├── ReturnRequest[]
    ├── RefundTransaction[]
    └── Ticket[]
```

`Product` 是当前商品目录；`OrderItem` 保存购买时的名称、价格和属性快照，因此商品当前价格变化不会篡改历史订单。`ReturnRequest` 表示售后申请，`RefundTransaction` 表示资金流水，两者不会再通过一个字典混在一起。

## 当前阶段的工程原则

- 高风险工具不能只依赖 Agent 的自然语言判断，仓储层会重新验证用户、订单、商品、政策和金额。（兜底幻觉）
- 工具写操作使用幂等键；响应未知时先查询状态，不直接重试。（三个ID组成唯一key）
- 用户确认、商品范围和退款金额属于显式状态，不从长对话文本中临时猜测。
- 状态机拒绝非法状态转换，并记录状态变化的 actor、原因和时间。
- 场景由确定性业务状态和期望行为组成，不能只用 LLM judge 判断对错。

层	           防什么	                 例子
ToolGateway	  参数缺失	       Agent 忘了传 idempotency_key
仓储层业务校验	数据不实	       金额算错了、订单不属于这个用户
状态机	       状态跳转不合法	   已发货的订单想直接取消、已关闭的工单想重开

LLM 的判断只是"建议"，代码层以数据为权威，逐层校验，高风险路径上不信任模型的任何输出。


后续阶段可以在此基础上完善政策检索、增加用户模拟器和 Monitor Agent，并用这些场景做端到端回放、路由混淆矩阵和消融实验。

## 阶段六生产级 Agentic RAG

阶段六现在同时保留两条链路：`BM25Retriever`/离线 `HybridRetriever` 是无外部依赖的对照基线；`--production` 启用真实 dense embedding + BM25 生产链路。旧 `VectorRetriever` 仅作为兼容别名，内部不再使用 TF-IDF：

```text
Policy JSON
  → ingestion / chunking / content hash
  → immutable index version
  → real dense embedding + BM25 sparse index
  → dense/sparse weighted retrieval + metadata/date filter
  → explicit graph: guardrail / planner / retrieve / grade / rewrite / verify
  → DeepSeek semantic Document Grader
  → EvidenceValidator: precedence / exception / conflict / coverage
  → DeepSeek grounded structured generation
  → claim-level citation gate
  → RAGTelemetry / MonitorAgent / JSONL sink
```

生成和 embedding 是两个独立服务：DeepSeek Chat 负责 Planner 与最终回答，embedding 必须配置一个真正提供 `/embeddings` 的服务，不能把 Chat 模型当作 embedding 模型。离线 `hash` provider 只用于测试闭环，不代表语义效果。

构建、发布和回滚索引：

```bash
# 离线测试闭环，不需要网络
python3 -m rag.ingest_cli \
  --embedding-mode hash \
  --index-root /tmp/policy-index \
  --version policy-offline-v1 \
  --publish

# 真实 embedding 服务；密钥只从环境变量读取
export EMBEDDING_API_KEY="你的 embedding 服务 key"
export EMBEDDING_BASE_URL="https://你的-embedding-endpoint/v1"
export EMBEDDING_MODEL="你的中文 embedding 模型"
python3 -m rag.ingest_cli \
  --embedding-mode env \
  --index-root var/rag_indexes \
  --version policy-2026-09-08 \
  --publish

# 回滚只移动 CURRENT alias，不覆盖任何历史版本
python3 -m rag.ingest_cli \
  --index-root var/rag_indexes \
  --rollback policy-2026-09-01
```

启动真实 Agentic RAG：

```bash
export DEEPSEEK_API_KEY="你的 DeepSeek key"
export DEEPSEEK_MODEL="deepseek-v4-flash"
python3 -m rag.cli \
  --mode agentic \
  --production \
  --index-root var/rag_indexes \
  --embedding-mode env \
  --grader-mode llm \
  --generator-mode llm \
  --monitor-sink var/rag_monitor.jsonl
```

生产回答不是直接拼接文档：模型必须通过 `emit_grounded_answer` 返回 answer、claims、citation IDs 和 abstain 状态；只有通过引用 ID、数字一致性和基本 claim/evidence 支持校验的结果才会以 `GROUNDED` 返回，否则进入 `UNCERTAIN`。每次运行都记录 run ID、Planner/Generator 次数、检索次数、停止原因、延迟、证据 ID、citation ID 和告警。

评测线上 telemetry（可以配合人工抽检或延迟到达的标签）：

```bash
python3 -m evaluation.online_rag_cli \
  --telemetry var/rag_monitor.jsonl \
  --labels evaluation/online_labels.json \
  --output reports/online_eval.json
```

`online_labels.json` 支持如下格式，`expected_evidence_ids` 可以填 document ID，也可以填 chunk citation ID：

```json
[
  {
    "run_id": "某次 telemetry 中的 run_id",
    "expected_status": "GROUNDED",
    "expected_evidence_ids": ["policy_custom_quality_exception_v2#chunk-001"],
    "user_feedback": "helpful"
  }
]
```

索引版本目录包含 `manifest.json` 和 `chunks.jsonl`。版本构建后不可覆盖，发布只修改 `CURRENT` 指针；manifest 记录语料 hash、embedding model/dimension、文档数、chunk 数和构建时间。当前实现是可离线运行的 reference implementation，单机 JSON 索引适合项目演示、回放和面试实验；线上大规模部署时应将同一生命周期接口接到对象存储、向量数据库/BM25 服务、原子 alias、OTel/指标系统和标注平台。

## 阶段七：全栈服务化

阶段七把阶段一到六的 Agent 核心封装为可部署的模块化单体 MVP：

```text
React 用户端 / 客服工作台
  → FastAPI + SSE
  → PostgreSQL：会话、消息、Agent run、checkpoint、事件、确认、人工接管、反馈、RAG 索引元数据
  → Redis Streams / PubSub / session lock
  → 现有 Multi-Agent + MCP + Agentic RAG 核心
```

本地启动：

```bash
docker compose up --build
```

访问：

- 用户聊天页和客服工作台：`http://localhost:3000`
- FastAPI OpenAPI：`http://localhost:8000/docs`
- Ready 检查：`http://localhost:8000/api/v1/health/ready`
- Prometheus 文本指标：`http://localhost:8000/api/v1/metrics`

服务化 API 包括：

```text
POST /api/v1/sessions
POST /api/v1/sessions/{session_id}/messages
GET  /api/v1/sessions/{session_id}/events       # SSE
GET  /api/v1/sessions/{session_id}/runs
GET  /api/v1/sessions/{session_id}/pending-action
POST /api/v1/pending-actions/{action_id}/confirm
POST /api/v1/pending-actions/{action_id}/cancel
GET  /api/v1/handoffs
POST /api/v1/handoffs/{handoff_id}/claim
POST /api/v1/handoffs/{handoff_id}/resolve
POST /api/v1/runs/{run_id}/feedback
GET  /api/v1/rag/indexes
POST /api/v1/rag/indexes/{version}/publish
POST /api/v1/rag/indexes/{version}/rollback
```

高风险确认绑定 `pending_action_id + confirmation_token + state_version`，不会把用户输入的一个裸“确认”永久当作授权。Worker 每个 run 持久化 checkpoint、当前节点、route trace 和 Agent trace，并通过 Redis Pub/Sub 推送 SSE；同一 session 使用分布式锁，避免并发消息乱序。

阶段七详细说明见 [`STAGE7.md`](STAGE7.md)。当前 PostgreSQL 已覆盖服务控制面和 Agent 运行数据；订单、物流、退货、退款和工单领域操作仍默认复用阶段一到五的 `InMemoryStore` 测试后端。真正接入线上业务前，还需要替换为 PostgreSQL domain repositories/真实业务服务，并补正式 OIDC/JWT、备份恢复、限流、OpenTelemetry、CI/CD 和压测。
