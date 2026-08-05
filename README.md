# 风险感知售后客服 Agent

这是一个面向电商售后场景的系统型 Agent 项目。当前完成的是阶段一：先建立可被 Agent 查询和修改的业务模拟器、正式状态机以及可回放的场景语料，暂时不依赖外部大模型 API。

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

## 目录

```text
agent-project/
├── agent.py
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
└── test_stage_one.py      # 领域模型、状态机、工具和语料测试
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

- 高风险工具不能只依赖 Agent 的自然语言判断，仓储层会重新验证用户、订单、商品、政策和金额。
- 工具写操作使用幂等键；响应未知时先查询状态，不直接重试。
- 用户确认、商品范围和退款金额属于显式状态，不从长对话文本中临时猜测。
- 状态机拒绝非法状态转换，并记录状态变化的 actor、原因和时间。
- 场景由确定性业务状态和期望行为组成，不能只用 LLM judge 判断对错。

后续阶段可以在此基础上继续拆分 MCP Server、加入多 Agent Router、完善政策检索、增加用户模拟器和 Monitor Agent，并用这些场景做端到端回放和消融实验。
