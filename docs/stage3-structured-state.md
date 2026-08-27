# 阶段三：结构化状态、多轮记忆与 baseline 对照

## 1. 阶段目标

阶段二 ReAct baseline 把“理解用户、选择工具、拼接参数、决定是否继续、生成回答”都交给同一个模型循环。它适合建立最小可运行基线，但有三个结构性问题：

1. 关键业务变量只存在于聊天文本中，无法稳定判断当前订单、目标商品和缺失字段；
2. 模型产生的“用户已经确认”与真正的用户授权没有权限隔离；
3. 出错后只能看整段消息历史，难以定位是语义解析、状态归并、策略、工具还是恢复链路的问题。

阶段三引入显式状态，但保留阶段二作为对照组。目标不是让规则覆盖所有语言，而是让 LLM 负责语义候选，让代码负责状态一致性和动作授权。

## 2. 执行架构

```text
User message
    │
    ├── DeterministicExtractor
    │     精确 ID、精确确认/取消、商品别名、否定范围、相对订单指代
    │
    └── DeepSeekSemanticParser
          受 JSON Schema 约束的 SemanticFrame 候选
                 │
                 ▼
            StateReducer
      来源优先级、冲突检测、任务重置、确认失效
                 │
                 ▼
       StructuredTaskState
      working state + facts + episodic memory
                 │
                 ▼
          DialoguePolicy
       确定追问、读工具、写工具和恢复动作
                 │
                 ▼
            ToolGateway
       身份绑定、schema、业务校验、幂等与故障注入
```

`DeepSeekSemanticParser` 只调用一个受约束函数 `emit_semantic_frame`。返回字段包括意图、订单候选、商品/商品号、排除范围、退货原因、是否只咨询、确认/拒绝、是否转人工和置信度。

### 权威性顺序

不同数据的权威性不是统一的“高置信度覆盖低置信度”，而是按风险和来源定义：

1. 已认证会话中的 `user_id` 由服务端绑定，模型不能提供或覆盖；
2. 用户文本中的精确订单号、商品号、否定范围和精确确认短语由确定性提取器优先；
3. 工具返回的订单、商品归属、政策和退款金额是业务事实；
4. LLM 用于补足模糊意图和语义指代；
5. LLM 与确定性结果冲突时保留冲突码，不能静默覆盖高精度信号。

例如用户明确写出 `O10086`，而 LLM 候选为 `O20001`，系统采用 `O10086` 并记录 `ORDER_ID_LLM_DETERMINISTIC_CONFLICT`。用户说“上一单”时，即使模型猜出一个订单号，也不能当作已落地事实，而会要求可验证的订单解析。

## 3. 结构化任务状态

`StructuredTaskState` 的核心字段如下：

| 字段 | 含义 | 主要更新者 |
|---|---|---|
| `intent` | 当前任务目标 | reducer |
| `stage` | 当前正式对话阶段 | state machine / policy |
| `slots` | 值、来源、置信度、证据、更新时间 | reducer / tool policy |
| `selected_item_ids` | 当前准备处理的订单项 | reducer / entity resolution |
| `excluded_item_ids` | 用户明确排除的订单项 | reducer |
| `missing_fields` | 阻止下一步的必需字段 | reducer / policy |
| `risk_level` | 当前动作风险 | reducer / policy |
| `confirmation_status` | 未请求、待确认、已确认、拒绝 | reducer / policy |
| `pending_action` | 用户确认所绑定的不可变动作范围 | policy |
| `policy_evidence` | 本次判断使用的政策文档 ID | policy |
| `facts` | 来自工具的结构化事实 | policy |
| `conflicts` | 语义冲突和安全异常 | reducer / policy |

槽位不是一个普通字典值。每个 `SlotValue` 同时记录 `source`、`confidence`、`evidence` 和 `updated_turn`，因此可以回答“这个订单号为什么在状态里”“它来自规则、模型还是工具”“哪一轮发生了覆盖”。

## 4. 多轮记忆

阶段三把记忆拆成不同用途，避免把无限增长的原始聊天历史当作唯一记忆：

- 工作记忆：`StructuredTaskState`，只保存当前任务需要的规范化变量；
- 事实记忆：工具查询获得的订单、物流、政策和操作结果；
- 情节记忆：`TurnMemory[]`，保存每轮用户输入、语义帧、状态前后快照、调用过的工具和最终回复，并设置最大轮数。

语义解析模型只接收最小 `parser_context`：当前意图、阶段、订单、选中/排除商品、是否待确认和缺失字段。这样既保留多轮任务信息，又降低长历史污染、token 增长和旧订单串线的概率。

任务切换时会生成新 `task_id` 并清空交易范围；同一会话的用户身份仍保留。终态后开始新任务也会重新建立事务边界。

## 5. 正式状态机

当前阶段定义为：

```text
idle
  → collecting_info
  → resolving_entities
  → checking_policy
  → awaiting_confirmation
  → executing
  → completed
```

`rejected` 用于政策拒绝，`handoff` 用于无法安全自动完成或用户要求人工处理。合法转换集中定义在 `structured/state_machine.py`；非法跳转会抛出 `InvalidTransition`，例如新会话不能从 `idle` 直接跳到 `completed`。

策略不是每轮都按固定顺序调用全部工具，而是根据意图、阶段和缺失字段决定：

- 缺订单号：追问或列出可选订单；
- 缺目标商品：展示订单项并追问；
- 查询任务：只使用读工具；
- 退货任务：查询订单、解析商品、检索政策、计算金额；
- 高风险写入：先生成待确认动作，下一轮精确确认后再执行；
- 未知提交：不重试写操作，先用幂等键对账；
- 无法安全恢复：创建工单并进入人工处理。

## 6. 确认不是一个布尔值

阶段三最重要的不变量是：确认必须绑定具体动作范围。

`PendingAction.scope_key()` 包括：

```text
order_id + sorted(item_ids) + refund_amount + reason
```

幂等键也覆盖 `task_id + order_id + item_ids + amount + reason`。只有下一轮确认时重新计算出的范围与待确认范围完全一致，才允许调用 `create_return_request`。

以下任意变化都会使旧确认失效并要求重新确认：

- 更换订单；
- 从“退鞋”改成“退耳机”；
- 增加或排除商品；
- 退款金额变化；
- 从无理由退货改为质量问题等原因变化。

只有“确认”“确认退货”等完整精确短语会被确定性识别为确认。没有 `PendingAction` 时出现确认会记录 `ORPHAN_CONFIRMATION`，不会授权写操作。LLM 产生的确认候选仍需要通过相同状态范围校验。

## 7. NLU 与 LLM 的关系

阶段一的 NLU 没有被阶段二/三废弃。三个版本的区别是：

| 版本 | 语言理解 | 状态 | 工具决策 | 用途 |
|---|---|---|---|---|
| 阶段一规则 Agent | 确定性 NLU | 简单任务状态 | 规则 | 建业务与安全原型 |
| 阶段二 ReAct baseline | DeepSeek | 原始消息历史 | DeepSeek | 最小可运行对照组 |
| 阶段三结构化 Agent | 确定性信号 + DeepSeek 语义候选 | 显式结构化状态与记忆 | 确定性策略 | 可靠多轮执行版本 |

阶段三保留确定性 NLU，是因为订单号、商品号、否定范围和确认属于低歧义、高风险信号；加入 LLM，是因为真实用户会说“穿在脚上的那个”“先问问能退多少”“换成另一个”等规则难以穷举的表达。两者通过 reducer 合并，而不是互相替代。

## 8. 对照评测

`evaluation/comparison.py` 对每个场景分别创建新的仓储与 Agent，避免前一个实验污染后一个实验。两个版本读取同一场景的：

- 用户与多轮输入；
- 初始业务状态；
- 允许/禁止动作；
- 目标商品与排除商品；
- 期望最终状态；
- 可选工具故障注入。

当前统计指标：

- `final_state_accuracy`：最终业务状态是否符合场景 ground truth；
- `unsafe_write_rate`：是否出现禁止写、未确认写、提前写、漏目标商品或写入排除商品；
- `average_tool_calls`：平均工具调用数量；
- `average_model_requests`：平均模型请求数量；
- prompt/completion token 总量；
- 每场景回复、工具序列、工具错误和违规原因。

真实对比命令：

```bash
export DEEPSEEK_API_KEY="你的新 API key"
python3 -m evaluation.compare_cli \
  --limit 8 \
  --output outputs/stage3-comparison.json
```

不要把单次模型结果当成稳定结论。正式实验应固定模型版本、温度、场景版本和代码提交，至少重复多次，并报告平均值、方差、失败样例和 token/延迟成本。

### 合成故障回归的边界

`test_comparison.py` 使用脚本模型故意在用户尚未确认时输出 `user_confirmation=true`。这个实验验证两件事：

1. 对比框架能够捕获 baseline 的 `PREMATURE_WRITE_ATTEMPT`；
2. 结构化 Agent 不会让 LLM 的确认候选越过待确认动作范围。

它不是 DeepSeek 的真实准确率，也不应写成“结构化版本线上准确率 100%”。真实数字必须由真实模型回放产生并保存原始结果。

## 9. 可复现验证

运行全部阶段测试：

```bash
python3 -m unittest -v
```

离线观察多轮状态：

```bash
python3 -m structured.cli --offline --user-id U001
```

建议回归对话：

```text
我要退鞋
订单号是 O10086
改成退耳机
鞋坏了，属于质量问题
确认退货
```

还应重点回放：精确商品号排除、咨询不执行、无待办确认、模型与订单号冲突、相对订单指代、写入后响应超时、非法状态跳转和跨用户访问。

## 10. 面试中如何解释这一阶段

可以用“失败—定位—修复—验证”的方式，而不是只罗列功能：

> ReAct baseline 在多轮售后任务里把确认、商品范围和订单都留在聊天历史中。我们注入过模型提前确认，也遇到过用户把目标从鞋改成耳机后旧商品仍留在 resolved IDs 的问题。为此我把 LLM 限定为语义候选，引入带来源和证据的槽位、正式状态机，以及绑定订单、商品、金额、原因的 PendingAction。目标或原因变化会撤销旧确认；写入响应未知时用幂等键对账而不是盲目重试。最后用同一批场景对 baseline 和结构化版本回放，统计最终状态、不安全写入、工具调用和 token，而不是只看回答是否像人。

这个表述的重点是：系统确实出现过可复现错误，错误被归因到具体层，修复由不变量和回归场景支撑，而不是仅仅“加了一个状态机”。
