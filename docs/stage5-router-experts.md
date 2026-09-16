# 阶段五：确定性优先 Router 与多专家 Agent

## 1. 阶段目标

阶段三已经证明结构化状态可以阻止模型在未确认时直接写入。阶段五解决另一个问题：当订单查询、政策判断、交易执行和人工升级都堆在一个策略类里，权限、错误处理和评测责任会逐渐混在一起。

本阶段不是为了增加 Agent 数量，而是建立可验证的职责边界：

```text
StructuredTaskState
        │
        ▼
MultiAgentRouter
        │
        ├── OrderLogisticsExpert
        ├── PolicyExpert
        ├── TransactionExpert
        └── HandoffExpert
                │
                ▼
       SessionToolExecutor
                │
                ▼
     MCP Server + Repository
```

阶段三的 `StructuredCustomerServiceAgent` 和 `DialoguePolicy` 保留为对照组。阶段五新增 `MultiExpertCustomerServiceAgent`，两者共享相同的提取器、语义帧、状态归并器、MCP Server 和种子数据。

## 2. 为什么规则优先

模型擅长理解复杂表达，但不适合决定高风险操作是否已经满足前置条件。Router 先读取：

- `intent`；
- `missing_fields`；
- `confirmation_status`；
- `pending_action`；
- `risk_level`；
- 当前轮 `conflicts`；
- 语义候选置信度。

确定性规则覆盖稳定路径：

| 状态 | 专家链 |
|---|---|
| 查询订单 | 订单/物流 |
| 查询物流 | 订单/物流 |
| 申请退货 | 订单/物流 → 政策 → 交易 |
| 用户取消待确认动作 | 交易 |
| 明确要求人工 | 人工升级 |
| 意图未知 | 澄清；在线模式可调用模型复核 |

只有以下复杂信号会触发语义 Router：意图未知、状态冲突、低语义置信度或同一句出现多个诉求。模型输出是 `SemanticRouteCandidate`，不是最终执行命令。

## 3. 模型二次路由的约束

语义 Router 只能通过 `emit_route_decision` 返回：

```text
experts + needs_clarification + reason + confidence
```

最终 Router 强制以下规则：

1. 退货流程中的订单、政策、交易顺序不能被模型删除或重排；
2. 未知意图不能因为模型给出 `transaction` 就进入交易专家；
3. 多意图无法安全拆分时，不静默处理其中一个，而是要求用户先选优先诉求；
4. 模型服务失败时返回确定性规则结果或澄清，不让路由链崩溃；
5. 高置信度模型可以把未知但明显需要特殊处理的请求送到人工专家，但不能授权退货写入。

模型调用的候选、置信度、原因、token usage 和错误都会写入 route trace。

## 4. 专家职责与最小权限

### 订单/物流专家

负责订单读取、物流读取、商品 ID 与自然语言商品名的落地。退货时它必须证明目标商品属于当前用户的当前订单，才能把任务交给政策专家。

允许工具：`list_orders`、`get_order`、`list_shipments`。

### 政策专家

逐商品调用 `search_policy`，保存可追溯证据，再调用 `calculate_refund` 做资格和金额计算。没有证据时不允许继续。咨询请求在这里结束，不进入交易专家。

允许工具：`search_policy`、`calculate_refund`。

### 交易专家

进入交易前再次验证：订单存在、商品范围非空、金额大于零、政策证据存在、不是咨询请求。用户确认必须与冻结的 `PendingAction.scope_key()` 完全匹配：

```text
order_id + sorted(item_ids) + amount + reason
```

范围不一致时旧确认作废，并生成新的待确认动作。写入超时返回 `UNKNOWN_COMMIT` 时只调用 `get_return_status` 对账，不直接重写。

允许工具：`create_return_request`、`get_return_status`。

### 人工升级专家

处理用户明确要求人工、未知提交无法对账、专家协议异常和无法安全恢复的工具错误。工单使用稳定幂等键；工单自身出现未知提交时也会查询状态。

允许工具：`create_ticket`、`get_ticket_status`。

## 5. 两层工具权限

专家工具白名单和 MCP scope 解决不同问题：

```text
专家白名单
→ 防止政策专家误调用交易工具

MCP scope + session identity
→ 防止当前会话调用没有授权的能力或覆盖 user_id

Repository guards
→ 防止访问不属于当前用户的订单或提交错误金额
```

因此“所有专家共享同一个 MCP Client”不等于所有专家拥有相同业务权限。专家先执行本地最小能力检查，通过后仍要经过 MCP 的 schema、scope、timeout、retry、circuit breaker、idempotency 和审计。

## 6. 专家输出协议

所有专家必须返回 `ExpertResult`：

```text
expert
status: complete / continue / clarify / rejected / escalate
response
next_expert
reason_code
escalation_reason
facts
```

Router 会拒绝：

- 返回普通字典而不是 `ExpertResult`；
- 声称了错误的专家身份；
- `continue` 指向路由计划外的下一跳；
- 终止结果仍携带下一跳；
- 终止结果没有用户回复；
- 升级结果没有升级原因。

协议异常不会被当成成功，而是记录 `route_protocol_error` 并调用人工升级专家。

## 7. Route Trace

每轮 route trace 包含：

- `route_id`、turn 和输入状态快照；
- 规则原因码和最终候选专家链；
- 模型是否参与、模型候选、置信度、原因、usage 和错误；
- 每个专家的 status、下一跳、reason code 和结构化事实；
- 最终专家和 fallback reason。

工具 trace 额外记录发起调用的专家和 MCP `audit_id`，因此可以从“为什么选这个专家”追到“专家调用了什么工具”，再追到 MCP 层的权限与可靠性审计。

## 8. 实际发现并修复的失败

开发阶段五时不是只验证成功路径，专项测试暴露了三个具体问题：

| 失败 | 风险 | 修复 |
|---|---|---|
| “查物流，同时把鞋退掉”最初只按退货路由 | 静默遗漏物流诉求，用户误以为两个目标都完成 | 增加多意图检测；在线调用模型复核，不能安全拆分就澄清，且不调用工具 |
| 专家协议错误转人工后，原协议错误原因被覆盖 | trace 只能看到“转人工”，看不到真正故障点 | 保留首个 `EXPERT_PROTOCOL_ERROR`，人工工单原因作为后续结果记录 |
| 未知提交只测试了“对账成功” | 对账服务也失败时可能没有明确终态 | 注入写入后响应丢失，再连续注入对账 timeout；确认不重复写入并创建人工工单 |

这三类案例分别对应 routing completeness、observability 和 distributed uncertainty，不是单纯增加功能数量。

## 9. 测试覆盖

`test_stage_five.py` 当前覆盖：

- 订单和物流只路由到订单专家；
- 退货依次通过订单、政策、交易专家；
- 咨询在政策专家结束，不进入交易；
- 未确认不写，确认范围一致才写；
- 明确人工请求直接创建工单；
- 多意图调用模型后安全澄清；
- 模型不能把未知请求路由到交易；
- 模型失败使用确定性 fallback；
- 未知提交且对账失败时转人工；
- 非法专家 schema 和非法下一跳被拒绝；
- 专家白名单阻止跨领域工具调用；
- 专家调用共享 MCP 身份和 audit；
- 多专家版本能进入现有场景对比框架。

运行：

```bash
python3 -m unittest -v test_stage_five.py
python3 -m unittest -v
python3 -m scenarios.validate
```

## 10. 对照评测

`evaluation.compare_cli` 同时运行：

1. `react_baseline`：模型直接决定工具；
2. `structured_state`：单策略类读取结构化状态；
3. `multi_expert`：确定性 Router 和职责隔离专家链。

通用指标仍包括最终状态准确率、不安全写入率、工具调用数、模型调用数和 token。阶段五后续适合新增：路由准确率、专家混淆矩阵、澄清率、人工升级精确率、路由模型增量 token、专家协议违规率和 unknown-commit 恢复率。

## 11. 面试表达

可以这样说明：

> 我没有按关键词把请求随便分给多个 prompt，而是让结构化任务状态先决定稳定路由。退货必须经过订单实体落地、政策证据和交易确认三个专家，模型只在冲突、低置信度或多意图时给候选，不能删掉前置步骤，也不能授权写操作。每个专家有代码级工具白名单，所有调用继续经过同一个 MCP 身份、scope 和审计边界。Router 还校验专家输出 schema 和下一跳。开发中实际发现多意图请求会被单意图状态静默吞掉，于是增加了多意图故障测试和澄清策略；又注入了写入成功但响应丢失、随后对账也超时的组合故障，最终保证不重复写并转人工。

重点不是“四个 Agent”，而是路由权威、最小权限、协议校验、失败语义和可复现实验。
