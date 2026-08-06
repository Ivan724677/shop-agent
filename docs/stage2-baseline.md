# 阶段二实验基线定义

## 目标

建立一个最小、可运行、可观测的单 Agent baseline，作为后续结构化任务状态、多 Agent 路由、动作验证器和 Monitor Agent 的共同对照组。

## 固定链路

```text
User → DeepSeek V4 Flash → ToolGateway → Tool observation → DeepSeek → Answer
```

模型通过 OpenAI-compatible tool calls 选择工具。每个 assistant tool-call message 和对应 tool observation 都保存在原始消息历史中，并在下一次模型请求中完整回传。

## baseline 包含什么

- `deepseek-v4-flash` Chat Completions；
- 非思考模式；
- 温度为 0；
- 12 个业务工具；
- 最多 6 次模型决策；
- 每次最多 4 个工具调用；
- 模型/工具执行 trace；
- API token usage；
- 完整原始消息历史。

## baseline 刻意不包含什么

- 显式意图分类；
- 槽位和任务阶段；
- 结构化工作记忆；
- 商品排除范围；
- Planner；
- 多 Agent Router；
- 自动重试和补偿策略；
- Monitor Agent 干预；
- RAG 重排与证据覆盖检查。

## 不参与消融的安全边界

用户身份绑定、工具 schema、仓储层金额重算、订单归属校验、政策资格校验、显式确认和幂等检查始终保留。移除这些约束只会制造不安全系统，并不能证明结构化 Agent 的智能收益。

## 后续比较维度

- 端到端任务成功率；
- 订单项选择准确率；
- 禁止动作触发率；
- 缺少工具证据的事实声明率；
- 工具参数错误率；
- 无效或重复工具调用数；
- 最大步骤触发率；
- 平均模型轮次、token 和延迟；
- 多轮订单串线率；
- 工具故障后的恢复成功率。
