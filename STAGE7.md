# 阶段七：全栈服务化

阶段七把阶段一到六的 Agent 核心从 CLI/reference implementation 变成可部署的模块化单体 MVP：

```text
React 用户端/客服工作台
          ↓
FastAPI API + SSE
          ↓
PostgreSQL（会话、消息、run、checkpoint、事件、确认、人工、反馈）
          + Redis Streams / PubSub / session lock
          ↓
现有 MultiExpertCustomerServiceAgent + MCP + Agentic RAG
```

## 本地启动

前置：Docker Desktop。

```bash
docker compose up --build
```

访问：

- 用户端与客服工作台：`http://localhost:3000`
- API 文档：`http://localhost:8000/docs`
- Ready 检查：`http://localhost:8000/api/v1/health/ready`
- Prometheus 文本指标：`http://localhost:8000/api/v1/metrics`
- PostgreSQL 可视化管理：`http://localhost:8080`

Adminer 登录信息：

```text
系统：PostgreSQL
服务器：postgres
用户名：agent
密码：agent
数据库：agent
```

你可以直接在 Adminer 中查看：

```text
chat_sessions
chat_messages
agent_runs
agent_events
pending_actions
handoffs
feedback
rag_index_versions
```

这比直接打开 PostgreSQL 端口更适合开发调试；生产环境不应暴露 Adminer。

## 推荐测试流程

### 1. 启动完整服务

```bash
docker compose up --build
```

第一次启动会按以下顺序执行：

```text
PostgreSQL / Redis 健康
→ Alembic migrate
→ FastAPI API
→ Agent Worker
→ Nginx 前端
→ Adminer
```

### 2. 测试普通订单查询

打开 `http://localhost:3000`，输入：

```text
帮我查询订单 O10086
```

预期：

- 用户消息立即出现在聊天窗口；
- Agent 显示处理中；
- Worker 执行现有订单/物流专家；
- 回复通过 SSE 推送；
- `agent_runs` 中出现一条 completed 记录；
- `chat_messages` 中出现 assistant 消息；
- `agent_events` 中出现 route、expert、tool、message.completed 事件。

### 3. 测试高风险确认

在聊天页输入：

```text
我要退订单 O10086 里的鞋，耳机不退
```

预期页面显示结构化确认卡片：

```text
订单：O10086
商品：I002
排除商品：I001
退款金额：129.00 元
售后原因：无理由退货
```

同时 Adminer 中应出现：

- `pending_actions`：status=pending；
- `agent_runs`：status=completed；
- `agent_events`：confirmation.required 或 message.completed；
- `chat_sessions.state_version` 增加。

点击“确认提交”后，前端会发送：

```text
pending_action_id
confirmation_token
state_version
```

Worker 再执行真实业务模拟器中的 `create_return_request`。预期 `return_requests` 产生一条记录，聊天显示退货申请已创建。

### 4. 测试确认失效保护

在确认卡片出现后，先继续发送一条会改变商品范围或原因的消息，再点击旧确认。

预期：

```text
HTTP 409
会话状态已变化，请重新确认最新操作
```

旧金额或旧商品范围不会被执行。

### 5. 测试人工升级

输入：

```text
我要联系人工客服
```

或者触发政策冲突/工具异常。

预期：

- 用户聊天页出现“已转人工处理”；
- `handoffs` 中出现 open 记录；
- 打开 `http://localhost:3000/workbench`；
- 使用默认客服 ID `CS001` 点击领取；
- 填写处理结论并关闭；
- 用户会话状态恢复为 active。

### 6. 查看 Agent Trace

打开：

```text
http://localhost:3000/traces
```

填入 session ID，可以查看：

- route trace；
- expert chain；
- tool calls；
- MCP audit_id；
- RAG evidence；
- checkpoint；
- 最终停止原因。

### 7. 查看 RAG 索引和指标

打开：

```text
http://localhost:3000/rag-indexes
http://localhost:3000/metrics
```

索引页面用于查看已经构建的版本和 CURRENT；指标页面用于查看 Agent run、确认、人工升级和反馈计数。

开发环境默认使用 `X-User-ID` 作为临时身份边界，默认用户是 `U001`；这只用于本地演示，真实部署必须替换成 OIDC/JWT 或安全 session。默认 `OFFLINE_AGENT=true`，不需要 DeepSeek key 即可跑通服务化链路。将 `OFFLINE_AGENT=false` 并提供环境变量后，Worker 会构造 DeepSeek Router/Planner/Grader/Generator。

## API 重点

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

高风险确认绑定三项：

```text
pending_action_id
confirmation_token
state_version
```

因此前端不能只发送一个裸字符串“确认”，旧状态、旧金额或旧商品范围的确认会被拒绝。

## 持久化与恢复

- API 请求先把用户消息和 `AgentRun` 写入 PostgreSQL，再写入 Redis Stream；
- Worker 消费 Stream，按 session 取得 Redis 分布式锁；
- 每个运行保存 checkpoint、当前节点、trace、route trace 和最终响应；
- Worker 完成后通过 Redis Pub/Sub 推送 SSE；
- 浏览器断线可以从已保存事件继续读取，服务重启不会丢失会话/run 记录；
- 同一会话的请求串行处理，避免后到的“确认”抢在商品范围更新之前执行。

## 当前边界

这是可部署的服务化 MVP，不宣称已经具备互联网规模生产能力。真实上线前仍需补：

- 正式 OIDC/JWT、角色权限和密钥管理；
- PostgreSQL 备份/恢复演练；
- 外部订单/物流/退款服务连接；
- provider 限流、模型 fallback 和成本预算；
- Prometheus/Grafana/OpenTelemetry 后端；
- staging、CI/CD、压测和安全审计；
- RAG ingestion worker 的线上任务编排。

这些边界与阶段七代码分开，避免把本地开发身份和内存业务模拟器误认为线上安全能力。
