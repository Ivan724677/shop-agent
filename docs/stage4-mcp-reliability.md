# 阶段四：MCP 工具协议与可靠性层

## 1. 为什么需要独立工具层

阶段二的模型会直接生成工具名和参数，阶段三的策略也会调用工具。如果每个 Agent 分别实现参数处理、权限、超时和重试，会出现三类问题：

1. 模型看到的 schema 与后端真实签名漂移；
2. 不同 Agent 对同一个超时采取不同策略，尤其可能重复执行写操作；
3. 只有 Agent trace，没有跨 Agent、跨模型统一的操作审计。

阶段四建立一个独立于 Agent 推理方式的信任边界。ReAct baseline 和结构化 Agent 都通过 `InProcessMCPClient` 调用同一个 `MCPToolServer`，业务实现仍由 `ToolGateway` 和仓储层负责。

## 2. MCP 覆盖范围

当前实现的是 MCP 工具子集和 JSON-RPC 2.0 stdio 传输：

- `initialize`；
- `notifications/initialized`；
- `ping`；
- `tools/list`；
- `tools/call`。

`tools/list` 根据当前认证主体的 scope 过滤工具；`tools/call` 返回 text content、`structuredContent` 和 `isError`。项目同时把同一份目录转换为 DeepSeek/OpenAI-compatible function tools，避免维护两套 schema。

当前没有实现 resources、prompts、订阅、远程 HTTP transport、OAuth discovery 和动态工具变更通知。这是有意的阶段边界，不应把当前版本描述成完整 MCP SDK 替代品。

## 3. 一个工具定义包含什么

`ToolDefinition` 包含：

| 字段 | 作用 |
|---|---|
| `input_schema` | 在进入业务处理器前校验模型参数 |
| `output_schema` | 防止工具返回缺字段或类型错误污染 Agent 状态 |
| `required_scopes` | 工具级授权 |
| `risk_level` | 审计与策略分级 |
| `side_effect` | 决定是否允许自动 retry |
| `inject_user_id` | 从认证上下文注入身份，禁止模型提供 |
| `timeout_seconds` | 单次尝试的执行期限 |
| `max_attempts` | 一个逻辑调用内的最大尝试数 |
| `idempotency_field` | 写操作的稳定业务键 |
| `reconciliation_tool` | 未知提交后的查询工具 |

输入 schema 当前验证对象、数组、字符串、数字、布尔值、必填字段、额外字段、枚举、正则、长度、数组唯一性和数值范围。它是项目需要的确定性子集，不是通用 JSON Schema 引擎。

## 4. 权限模型

每次 MCP 调用都带有服务端创建的 `AuthContext`：

```text
user_id + session_id + actor_type + scopes
```

模型参数中不暴露 `user_id`。如果模型尝试传入该字段，无论它与当前用户是否一致，都返回 `IDENTITY_OVERRIDE_ATTEMPT`。这是为了让身份来源只有一个，避免将“模型声称的身份”和“认证身份”混为一谈。

典型 scope 包括 `orders:read`、`shipments:read`、`policy:read`、`returns:quote`、`returns:write`、`returns:read`、`refunds:read`、`tickets:write` 和 `tickets:read`。

权限检查发生在工具执行前，但仓储层仍检查资源归属。scope 只能说明“可以使用查询订单能力”，不能说明“可以读取任意用户的订单”。

## 5. 参数与结果验证

调用链执行三次不同目的的验证：

```text
input schema
→ 防缺字段、错类型、非法枚举、额外参数、空数组

repository guards
→ 防跨用户、商品不属于订单、政策不允许、金额不一致、未确认

output schema
→ 防工具返回结构损坏或缺少关键字段
```

参数错误不会进入 `ToolGateway`。工具返回不符合 output schema 时，只读工具可以有限重试；写工具不能因为“响应格式不对”而重新执行，因为服务端可能已经产生副作用。

## 6. Timeout 与 Retry

每次工具尝试都通过带 deadline 的执行 future。超时语义按副作用区分。

只读工具：

```text
timeout
→ retryable=true
→ 有剩余 attempt 时指数退避
→ 成功则返回结果并重置熔断计数
→ 仍失败则向 Agent 返回 TIMEOUT
```

写工具：

```text
timeout / response lost
→ UNKNOWN_COMMIT
→ attempts 固定为 1
→ 禁止直接 retry
→ 返回 reconciliation_tool
→ Agent 用原幂等键查询
```

“写工具有幂等键，所以任何错误都可以自动重试”并不成立。幂等实现本身可能有缺陷，或者调用已经跨越多个下游系统。项目选择先对账，只有明确证明没有提交时，才由更高层决定是否发起新操作。

## 7. Circuit Breaker

熔断器按工具独立维护：

```text
closed
  ├── success → closed / failure_count=0
  └── repeated transient failure → open

open
  ├── recovery window 未到 → CIRCUIT_OPEN，不执行后端
  └── recovery window 到 → half_open，允许一次探测

half_open
  ├── success → closed
  └── transient failure → open
```

业务拒绝、权限拒绝和参数错误不会增加熔断计数，因为它们不代表工具服务不健康。timeout、未知提交和非法响应属于瞬态或服务质量问题，会进入统计。

## 8. Idempotency

退货申请和人工工单都要求幂等键。保证分为两层：

1. MCP Server 在当前运行期保存 `(user, tool, key) → argument fingerprint`；
2. Repository 持久化模拟层保存 key 到业务记录的索引，并再次校验用户和业务参数。

行为定义：

- 同 key、同用户、同参数：返回原业务记录；
- 同 key、不同参数：`IDEMPOTENCY_KEY_REUSE`；
- 同 key、不同用户：`IDEMPOTENCY_KEY_REUSE`；
- 重复响应或客户端重放：不会创建第二条退货或工单记录。

第二层校验很重要，因为 MCP Server 重启后内存 fingerprint 会丢失，不能只依赖进程内缓存。

## 9. Unknown Commit

未知提交是工具可靠性中最危险的状态：客户端没有拿到成功响应，但服务端可能已经写入。

退货流程使用：

```text
create_return_request(idempotency_key=K)
→ UNKNOWN_COMMIT
→ get_return_status(idempotency_key=K)
→ found=true: 当作已成功并继续
→ found=false / 查询失败: 不重写，创建人工工单
```

人工工单同样使用 `create_ticket` 与 `get_ticket_status`。查询工具也绑定当前用户，知道别人的幂等键不能越权读取记录。

真实分布式系统里对账通常由 outbox、任务队列或后台 reconciliation worker 做多次、延迟查询。本阶段实现同步一次对账和人工兜底，后续 Monitor Agent 可以消费 `UNKNOWN_COMMIT` 审计事件继续跟踪。

## 10. 操作审计

一个逻辑工具调用生成一条 `AuditEvent`，即使内部 retry 两次也不会伪装成两个用户操作。字段包括：

- `audit_id`、`request_id`、`correlation_id`；
- actor ID、actor type；
- 工具、风险等级、是否有副作用；
- executed/rejected/denied/blocked 决策；
- 最终状态和错误码；
- attempts 与总耗时；
- 调用前后熔断状态；
- 参数摘要和 SHA-256 digest；
- 幂等键 digest。

工单描述等可能含隐私的字段存为 `<redacted>`；幂等键只保留截断哈希展示和完整 digest，不保存原文。Agent trace 只引用 `audit_id`、attempts 和 reconciliation tool，可以从一次对话追到可靠性审计，而不复制敏感参数。

## 11. 失败注入与测试

`test_stage_four.py` 覆盖：

- MCP tools/list 与 JSON-RPC tools/call；
- input schema 在后端前拒绝错误类型；
- 模型身份覆盖攻击与 scope 不足；
- 一次只读 timeout 后 retry 成功；
- 真实执行 deadline；
- 连续失败打开 circuit breaker；
- 写入后响应丢失且不重试；
- 使用幂等键对账；
- 同 key 不同参数和跨用户复用；
- 工单重复调用只创建一次；
- 审计脱敏；
- 非法工具响应重试后失败。

运行：

```bash
python3 -m unittest -v test_stage_four.py
python3 -m unittest -v
```

## 12. 面试表达

可以这样说明这一阶段：

> 我没有让每个 Agent 自己处理工具错误，而是把工具目录和可靠性策略放到 MCP 边界。模型工具 schema 和 MCP tools/list 来自同一个定义；用户身份由会话注入，scope 决定能否看到和调用工具，资源归属由仓储层二次校验。只读调用可以有限 retry，写调用超时统一进入 UNKNOWN_COMMIT，返回对账工具而不是直接重试。每个逻辑调用记录一条脱敏审计，内部 attempts 和熔断状态可观测。开发时还发现只在 MCP 进程内校验幂等不够，因为服务重启会丢缓存，所以在 repository 又加了一层同键同参数校验，并补了跨用户 key 复用测试。

这个表述展示的是协议、权限、故障语义和修复过程，而不是简单罗列 timeout、retry、circuit breaker 名词。
