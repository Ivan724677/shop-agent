# 阶段六：Agentic RAG 与政策推理

## 1. 阶段目标

阶段五已经能够把退货请求路由到政策专家，但政策专家原先主要依赖一次检索结果。真实售后系统中，政策问题通常同时包含商品类型、售后原因、时间点、版本和例外关系。只把一次向量召回结果塞给模型，会出现三个危险情况：

1. 召回了相似但不适用的政策；
2. 召回了旧版本，却没有发现新版本已经生效；
3. 例外规则和基础规则同时出现，模型凭表面相似度选错规则。

本阶段的目标不是接入一个向量数据库，而是建立一个可以回答“为什么查、查了什么、为什么相信、为什么拒绝”的政策知识链路，并让它能够被离线复现和量化比较。

## 2. 三个对照版本

| 版本 | 是否主动决定检索 | 主要机制 | 暴露的问题 |
|---|---:|---|---|
| BM25 | 否 | Okapi BM25 + 长度归一化 | 精确词项召回，但仍然一次检索 |
| Hybrid | 否 | Dense embedding + BM25 | 语义和关键词互补，但仍然一次检索 |
| Agentic | 是 | 显式 Graph + 双层 Grader + 重写/验证循环 | 可以少查、补查、验证和升级 |

BM25/离线 Hybrid 是对照组，不代表生产级语义检索。项目使用标准库实现它们，是为了让实验不依赖外部服务，能够清楚测量结构性改动的贡献。生产模式替换为真实 embedding + BM25，只需要注入 Retriever，不改变 Graph、Grader、Validator 和评测契约。

## 3. Agentic RAG 为什么不是固定 Workflow RAG

固定 workflow 的典型形式是：

```text
收到问题 → 一律检索 → 拼接文档 → 生成答案
```

即使写成：

```python
if "退货" in query:
    retrieve()
```

它仍然是确定性工作流，因为下一步动作由代码预先决定。阶段六的 Agentic 版本把动作选择交给 Planner。Planner 通过 `emit_rag_plan` 只能选择结构化动作：

- `answer_directly`：当前问题不需要政策库，或已有充分校验证据；
- `retrieve`：用当前查询检索；
- `rewrite_query`：证据覆盖不足，补充商品、原因、时限等维度后再检索；
- `verify`：要求系统重新校验适用性、版本和冲突；
- `clarify`：信息不足以形成有效查询；
- `escalate`：证据冲突或无法安全判断。

每一步 Planner 都接收裁剪后的当前观察，包括已召回证据的分数和元数据、缺失方面、冲突标记、当前检索次数以及问题是否政策敏感。Planner 可以自主选择“根本不查”，也可以选择第二次检索。离线 Planner 只是把该接口做成确定性回放，以便单元测试和消融实验；在线 Planner 使用 DeepSeek，但仍必须遵守相同的结构化函数契约。

项目不保存或要求模型暴露不可验证的原始思维链。trace 记录的是可审计决策：动作、理由、置信度、查询、检索模式、检索结果、校验结果、Planner 调用次数和错误。面试时可以展示决策轨迹，不把隐藏 CoT 当作正确性的证据。

## 4. 检索与重排

### 4.1 BM25 Retriever

`BM25Retriever` 使用 Okapi BM25，保留中文 unigram/bigram，并通过 `k1` 饱和词频、`b` 按文档长度归一化，避免长政策仅因重复词多而获得假高分。旧 `VectorRetriever` 名称只作为兼容别名，内部不再使用 TF-IDF。

- `vector_score`；
- 命中的词项；
- 生效时间和元数据过滤后的候选；
- `retrieval_pass` 和原始查询。

### 4.2 Hybrid Retriever

离线 `HybridRetriever` 使用 BM25 和词项覆盖率；生产 `DenseHybridRetriever` 使用真实 embedding cosine 和 BM25：

```text
rerank_score = 0.65 × dense_score + 0.35 × normalized_bm25_score
```

这个重排是显式可解释的，不是让生成模型凭感觉重新排序。返回结果仍保留三个分数，便于分析“向量召回了但关键词不匹配”或“关键词命中但语义相似度弱”的失败样本。

### 4.3 查询重写

`QueryRewriter` 根据初始查询、元数据过滤和 `missing_aspects` 生成保守扩展，例如把：

```text
杯子坏了能退吗
```

扩展为包含“商品售后政策、定制商品、例外规则、质量问题、三十天”等维度的查询。重写不是无条件发生，而是 Planner 观察到初次证据覆盖不足后选择 `rewrite_query` 才执行。

## 5. 元数据过滤与证据校验

元数据过滤发生在打分之前，当前支持：

- `product_type=custom/standard`；
- `reason=quality_issue/no_reason_return`；
- `as_of` 生效日期；
- 指定 `version`；
- `required_tags`。

过滤解决“这条政策适用于谁”，证据校验解决“这条召回结果能不能被引用”。`EvidenceValidator` 会：

1. 丢弃未发布、未生效或已失效文档；
2. 按 `policy_family` 分组；
3. 选择最高优先级、最新生效日期、最高版本；
4. 把 `exception_of` 表示的例外视为有意覆盖，不误报为版本冲突；
5. 检查同一政策族同一生效日、同一优先级的不同正文，发现则输出 `POLICY_VERSION_CONFLICT`；
6. 拒绝零相关度证据，避免“只因元数据过滤命中就当作证据”；
7. 检查 `required_aspects`，缺少商品类型、原因或时限时输出 `EVIDENCE_COVERAGE_INCOMPLETE`。

只有 `EvidenceValidation.accepted` 中的文档会出现在 `AgenticRAGResult.evidence_ids`。被拒绝的原始召回保留在 trace，便于解释和调试，但不会泄漏到最终引用集合。

## 6. 两个关键失败场景

### 6.1 例外政策

普通基础规则和定制商品规则可能同时包含“退货”“商品”等词。对于“定制商品有质量问题能不能退”，系统先用 `product_type=custom` 和 `reason=quality_issue` 过滤，再按 `priority` 选择 `policy_custom_quality_exception_v2`。它的 `exception_of=policy_custom_v1` 表示这是定制商品基础限制下的质量例外，而不是两个互相矛盾的普通版本。

### 6.2 版本冲突

如果同一政策族在同一生效日、同一优先级出现“七天”和“三十天”两条正文，系统不会按向量分数选第一条。Validator 返回 `POLICY_VERSION_CONFLICT`，Agentic RAG 最终状态为 `UNCERTAIN`，建议人工核实。这个行为比生成一个听起来合理的期限更安全，也可以在评测中被单独统计。

## 7. 防幻觉边界

Agentic RAG 的目标不是让模型“想得更长”，而是让模型在事实边界内自主规划：

- Planner 可以提出下一步，但不能直接制造政策事实；
- 证据不足时，安全护栏覆盖 `answer_directly`；
- 版本冲突时禁止猜测；
- 证据必须满足适用范围、生效期和覆盖维度；
- 检索上限和 Planner 步数上限防止循环；
- 已验证证据按 query、元数据和 `as_of` 缓存，减少重复查询；
- 模型异常时，政策敏感问题进入 `UNCERTAIN`，不降级成无依据回答。

因此“Agent 自主选择”与“系统最终安全性”是两个层次：前者负责规划，后者负责拒绝不合法的结论。

## 8. 评测方法

```bash
python3 -m evaluation.rag_compare_cli
```

每个 case 同时跑 BM25、Hybrid 和离线 Agentic 三个版本。默认数据集为 60 条静态 curated、待领域人员最终审核的政策场景。评测输出包含：

- `raw_retrieval_hit_rate`：预期文档是否出现在原始召回中；
- `hit_rate`：预期文档是否最终进入通过校验的证据集合；
- `accepted_evidence_rate`：返回结果中通过校验的比例；
- `conflict_detection_rate`：检测到版本冲突的比例；
- `stale_rejection_rate`：是否拒绝过期或低优先级证据；
- `average_retrieval_count`：平均检索次数；
- `no_retrieval_rate`：没有检索的比例；
- `average_planner_calls`：Planner 调用次数；
- `grounded_answer_rate` 与 `uncertainty_rate`；
- 期望状态、冲突和无检索行为的准确率。

评测集 `evaluation/rag_cases.json` 至少包含：定制商品质量例外、定制商品无理由限制、2026-07-16 的 v1、2026-09-02 的 v2、同级版本冲突和无关问题。冲突文档是 case-local fixture，只注入冲突 case，避免污染其他实验。

推荐进一步做以下消融：

1. 去掉元数据过滤，观察普通商品规则污染定制商品问题的比例；
2. 去掉 query rewrite，观察证据覆盖不足率；
3. 去掉 Validator，观察过期/冲突政策进入最终回答的比例；
4. 把 Agentic 的 `max_retrievals` 从 2 改成 1，比较成本和覆盖率；
5. 在线替换 DeepSeek Planner，比较 Planner 选择错误率和 token 成本。

## 9. 接入阶段五

在线 `multi_agent.cli` 会把 `AgenticRAG(DeepSeekRAGPlanner(client))` 注入 `PolicyExpert`。退货流程仍然是：

```text
订单/物流专家落地商品
→ Agentic RAG 判断是否需要政策检索并验证证据
→ MCP search_policy / calculate_refund
→ 用户确认
→ 交易专家执行写操作
```

Agentic RAG 不能绕过阶段五的订单归属、用户确认、退款金额和 MCP 权限边界；它只负责提高政策知识环节的召回与推理可靠性。

## 10. 面试表达

可以这样说明：

> 我没有把 RAG 写成每个问题都固定检索一次的链路。显式 Graph 中的 Planner 决定是否查、是否重写以及何时验证或升级；检索后先经过确定性业务过滤，再由 DeepSeek Document Grader 判断语义相关性，最后仍由 EvidenceValidator 决定政策权威性。开发中我专门注入了定制商品质量例外、过期政策和同级版本冲突：例外必须按优先级生效，过期政策必须过滤，冲突必须拒答转人工。这样可以量化比较 BM25/Hybrid 与 Agentic RAG 在证据命中、重写收益、冲突识别和无检索率上的差异。

## 11. 生产化实现：dense、sparse、generation 与生命周期

前面的 `BM25Retriever` 和 `HybridRetriever` 保留为 dependency-free baseline，便于做消融实验，但它们不应被描述为生产 semantic RAG。生产入口使用以下替换：

### 11.1 Embedding provider

`rag/embeddings.py` 定义 `EmbeddingProvider`，提供 `embed_documents` 和 `embed_query` 两个接口。当前有三个实现：

- `OpenAICompatibleEmbeddingProvider`：调用独立 `/embeddings` 服务，读取 `EMBEDDING_API_KEY`、`EMBEDDING_BASE_URL`、`EMBEDDING_MODEL`、`EMBEDDING_DIMENSION` 等环境变量；
- `SentenceTransformerEmbeddingProvider`：本地加载中文 sentence-transformers 模型；
- `HashEmbeddingProvider`：仅供离线测试，模型名明确标记为 `offline-hash-test`，绝不宣称具有语义能力。

Embedding 服务和 DeepSeek Chat 服务分离，原因是两者的模型、向量维度、扩缩容、缓存和版本发布策略不同。索引 manifest 固化 embedding model 与 dimension；加载索引时如果 provider 不匹配，系统直接拒绝启动，避免新旧向量空间混用。

### 11.2 真正的 sparse/dense hybrid

`rag/dense_retrieval.py` 在 immutable chunks 上分别计算：

```text
dense_score  = cosine(query_embedding, chunk_embedding)
sparse_score = normalized BM25(query_terms, chunk_terms)
rerank_score = 0.65 × dense_score + 0.35 × sparse_score
```

这不是“dense 召回后再看词重合”这一伪 hybrid：BM25 和 dense 分支独立打分，再进入融合排序；返回结果保留三个分数，支持分析语义召回、精确关键词召回和融合排序的失败样本。规模化部署可把本地 BM25 替换为 OpenSearch/Elasticsearch，而不改变 `Retriever` 和 evidence contract。

### 11.3 Ingestion 与索引生命周期

`PolicyIngestionPipeline` 完成文档切分、overlap、批量 embedding、content hash、source locator 和 metadata 写入。`PersistentIndex` 的发布协议是：

```text
build version → 校验 manifest/chunk/vector → atomic write → publish CURRENT
                                              ↘ rollback 旧 version
```

版本目录不可覆盖，`CURRENT` 是唯一可变 alias；加载时校验 chunk 数、文档数、重复 chunk ID、向量维度以及当前 corpus hash。这样政策文件更新、embedding 模型升级和回滚都可以被审计，而不是原地改一个共享索引。

### 11.4 LLM grounded generation 与 citation gate

`DeepSeekAnswerGenerator` 只接收 `EvidenceValidator` 已接受的 evidence，并要求 DeepSeek 通过 function call 返回结构化结果：

```json
{
  "answer": "...",
  "claims": [{"text": "...", "citation_ids": ["policy_id#chunk-001"]}],
  "abstain": false,
  "uncertainty": []
}
```

`CitationValidator` 在模型之后执行确定性 gate：检查 citation 是否属于 accepted evidence、每条 claim 是否有引用、数字是否出现在被引用证据中，以及 claim 与证据是否有基本支持关系。失败时不把模型输出降级成“看起来合理”的答案，而返回 `UNCERTAIN` 并建议人工核实。确定性 generator 只用于离线测试与故障演示。

### 11.5 监控和线上评测

`RAGMonitor` 将每次运行写入内存和可选 JSONL sink；`MonitorAgent` 检测无证据 grounded、非法引用、过多检索步数和模型/provider 错误。telemetry 不记录用户原文，只记录 run ID、状态、延迟、调用计数、停止原因、证据 ID/citation ID 和告警，便于接入 OTel 或消息队列。

`OnlineRAGEvaluator` 支持先记录线上结果、后接收人工/抽样标签，计算 status accuracy、evidence recall、用户反馈分布和延迟/告警统计。`evaluation.online_rag_cli` 可以直接消费 JSONL，适合作为定时评测任务；线上应进一步接入分桶漂移、p95/p99、成本、query 类型、模型版本和索引版本维度。

### 11.6 真实生产运行的边界

项目中的本地 JSON index 和 JSONL monitor 是可验证的 reference implementation，而非声称已经具备互联网规模的高可用基础设施。要部署到生产，需要把相同接口替换为：对象存储 + 原子 alias 的向量数据库、BM25/混合检索服务、embedding/Chat 独立服务、OTel 指标与 trace、标注/反馈平台、密钥管理、限流和多副本。核心安全契约已经在本地实现：索引不可覆盖、模型维度不匹配拒绝启动、证据冲突拒答、引用失败拒答、模型异常进入不确定态。

## 12. 显式 Graph 与双层证据判断

升级后的 `AgenticRAG` 对外 API 不变，内部由 `AgenticRAGGraph` 执行。节点和允许转换在 `graph_definition` 中显式声明：

```text
Guardrail
→ Planner
→ Retrieve
→ Deterministic Business Filter
→ Semantic Document Grader
→ Rewrite / Verify
→ Generate
→ Citation Validate
→ End / Escalate
```

每个节点只有单一职责：

- `GuardrailNode`：判断是否为政策问题，并记录缺失的商品类型或售后原因；
- `RetrieveNode`：调用 BM25、离线 Hybrid 或生产 dense+BM25 Retriever；
- `DeterministicBusinessFilter`：检查发布状态、生效期、商品类型、售后原因、读取权限和有效版本冲突；
- `GradeNode`：仅判断语义相关性、事实维度覆盖和噪声召回；
- `RewriteNode`：记录为什么重写、缺少哪些方面和增加了哪些检索词；
- `VerifyNode`：由 `EvidenceValidator` 重新执行优先级、版本、例外关系和冲突校验；
- `GenerateNode`：只使用最终 accepted evidence 生成回答；
- `CitationNode`：验证 claim 的引用、数字和支持关系；
- `EscalateNode`：证据不足、冲突或模型链路异常时进入 `UNCERTAIN`。

Document Grader 永远不能覆盖业务过滤结果。线上使用 `DeepSeekDocumentGrader`，离线回放使用 `HeuristicDocumentGrader`，二者遵守同一结构化输出：

```json
{
  "citation_id": "policy_id#chunk-001",
  "relevant": true,
  "score": 0.95,
  "covered_aspects": ["custom", "quality", "return"],
  "missing_aspects": [],
  "noise": false,
  "reason": "直接覆盖定制商品质量问题退货资格"
}
```

LLM 只拥有语义相关性的判断权，不拥有以下权限：

- 宣布未发布政策可用；
- 绕过政策生效日期；
- 绕过商品类型或售后原因；
- 读取无权限政策；
- 在版本冲突中自行选择一条；
- 授权退款、退货或其他写操作。

## 13. 查询重写审计

每次重写包含两个 trace 事件。`rewrite` 记录输入决策：

```text
original_query
rewritten_query
planner_reason
rewrite_reason
missing_aspects_before
added_terms
evidence_before
```

二次检索和 Verify 完成后，`rewrite_outcome` 补充：

```text
new_evidence
missing_aspects_after
coverage_improved
```

因此评测不再只统计“是否调用了 rewrite”，还可以判断重写是否带来了新证据或减少了缺失维度。默认离线报告增加 `rewrite_rate` 和 `rewrite_success_rate`。

## 14. 60 条静态待领域审核政策评测集

`evaluation/rag_cases.json` 使用共享 profile 减少重复元数据，但其中 60 条 query、类别、预期证据、状态和 review notes 都是静态可审阅的。当前 `review_status=pending_domain_owner_review`，避免把代码生成/模型整理误报为业务人员已签字审核；你完成逐条领域确认后再改为 reviewed。每个类别 5 条：

1. 普通商品；
2. 定制商品；
3. 质量问题；
4. 无理由退货；
5. 版本切换；
6. 例外规则；
7. 模糊描述；
8. 同义改写；
9. 对抗性问题；
10. 过期政策；
11. 版本冲突；
12. 不需要政策检索的问题。

评测报告同时输出整体和按类别拆分的指标。当前离线回放用于验证 Graph、业务过滤、重写和停止行为；真实 embedding、DeepSeek Planner、DeepSeek Grader 和 DeepSeek Generator 的线上数字应单独记录，不能与离线 heuristic 数字混称。
