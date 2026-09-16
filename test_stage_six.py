import unittest
from datetime import date

from baseline.deepseek_client import ModelCompletion
from experts.policy import PolicyExpert
from rag import (
    AgenticRAG,
    EvidenceValidator,
    HybridRetriever,
    PolicyCorpus,
    PolicyDocument,
    QueryRewriter,
    RetrievalQuery,
    VectorRetriever,
    load_policy_corpus,
)
from rag.models import PlanAction, PlanDecision
from rag.planner import DeepSeekRAGPlanner
from rag.retrieval import RetrievedEvidence


class ScriptedPlanner:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.observations = []

    def plan(self, observation):
        self.observations.append(observation)
        if not self.decisions:
            raise AssertionError("planner 被调用次数超过脚本长度")
        return self.decisions.pop(0)


class PlannerModel:
    def __init__(self, arguments):
        self.arguments = arguments
        self.calls = []

    def complete(self, messages, tools):
        self.calls.append((messages, tools))
        import json

        return ModelCompletion(
            message={
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "plan_1",
                    "type": "function",
                    "function": {
                        "name": "emit_rag_plan",
                        "arguments": json.dumps(self.arguments),
                    },
                }],
            },
            usage={"prompt_tokens": 15, "completion_tokens": 6},
            model="fake-rag-planner",
        )


def document(document_id, text, *, family="test", priority=10, version="v1", effective_from="2026-01-01", tags=None, exception_of=None):
    return PolicyDocument(
        document_id=document_id,
        title=document_id,
        text=text,
        tags=frozenset(tags or {"standard", "return", "no_reason_return"}),
        version=version,
        effective_from=date.fromisoformat(effective_from),
        priority=priority,
        policy_family=family,
        exception_of=exception_of,
    )


class StageSixRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.corpus = load_policy_corpus()

    def test_vector_and_hybrid_are_separately_callable(self):
        request = RetrievalQuery(
            "定制商品质量问题能不能退",
            metadata_filter={"product_type": "custom", "reason": "quality_issue"},
        )
        vector = VectorRetriever(self.corpus).search(request)
        hybrid = HybridRetriever(self.corpus).search(request)

        self.assertTrue(vector)
        self.assertTrue(hybrid)
        self.assertTrue(all(item.lexical_score == 0 for item in vector))
        self.assertTrue(any(item.lexical_score > 0 for item in hybrid))

    def test_metadata_filter_excludes_inapplicable_policy(self):
        request = RetrievalQuery(
            "商品能不能退",
            metadata_filter={"product_type": "custom", "reason": "quality_issue"},
        )
        results = HybridRetriever(self.corpus).search(request)

        self.assertTrue(results)
        self.assertTrue(all("custom" in item.document.tags for item in results))
        self.assertTrue(all("quality" in item.document.tags for item in results))

    def test_exception_policy_has_precedence_over_base_rule(self):
        request = RetrievalQuery(
            "定制商品质量问题退货期限",
            metadata_filter={"product_type": "custom", "reason": "quality_issue"},
        )
        results = HybridRetriever(self.corpus).search(request)
        validation = EvidenceValidator(self.corpus).validate(
            request, results, required_aspects=["custom", "quality", "return"]
        )

        self.assertTrue(validation.sufficient)
        self.assertEqual(validation.evidence_ids, ["policy_custom_quality_exception_v2"])
        self.assertNotIn("policy_custom_v1", validation.evidence_ids)

    def test_version_selection_changes_with_as_of(self):
        retriever = HybridRetriever(self.corpus)
        old = retriever.search(
            RetrievalQuery(
                "普通商品无理由退货多少天",
                as_of=date(2026, 7, 16),
                metadata_filter={"product_type": "standard", "reason": "no_reason_return"},
            )
        )
        new = retriever.search(
            RetrievalQuery(
                "普通商品无理由退货多少天",
                as_of=date(2026, 9, 2),
                metadata_filter={"product_type": "standard", "reason": "no_reason_return"},
            )
        )

        self.assertEqual(old[0].evidence_id, "policy_standard_v1")
        self.assertEqual(new[0].evidence_id, "policy_standard_v2")

    def test_overlapping_same_precedence_versions_are_uncertain(self):
        corpus = PolicyCorpus([
            document("policy_conflict_a", "普通商品签收后七天内可退货。"),
            document("policy_conflict_b", "普通商品签收后三十天内可退货。"),
        ])
        query = RetrievalQuery(
            "普通商品退货期限",
            metadata_filter={"product_type": "standard", "reason": "no_reason_return"},
        )
        evidence = HybridRetriever(corpus).search(query)
        validation = EvidenceValidator(corpus).validate(query, evidence)

        self.assertFalse(validation.sufficient)
        self.assertTrue(any("POLICY_VERSION_CONFLICT" in item for item in validation.conflicts))
        self.assertEqual(validation.evidence_ids, [])

        result = AgenticRAG(corpus).run(
            query.text,
            metadata_filter=query.metadata_filter,
            required_aspects=["standard", "return"],
        )
        self.assertEqual(result.status, "UNCERTAIN")
        self.assertTrue(result.validation.conflicts)

    def test_query_rewriter_adds_missing_policy_dimensions(self):
        query = RetrievalQuery(
            "杯子坏了能退吗",
            metadata_filter={"product_type": "custom", "reason": "quality_issue"},
        )
        rewritten = QueryRewriter().rewrite(query, ["quality", "return"])

        self.assertTrue(rewritten)
        self.assertIn("定制商品", rewritten[0].text)
        self.assertIn("质量问题", rewritten[0].text)

    def test_agentic_rag_skips_retrieval_for_non_policy_question(self):
        result = AgenticRAG(self.corpus).run("你好，请问你在吗？")

        self.assertEqual(result.status, "NO_RETRIEVAL")
        self.assertEqual(result.retrieval_count, 0)
        self.assertEqual([item["event"] for item in result.trace], ["plan"])

    def test_agentic_rag_reuses_verified_evidence_without_second_retrieval(self):
        rag = AgenticRAG(self.corpus)
        kwargs = {
            "metadata_filter": {"product_type": "custom", "reason": "quality_issue"},
            "required_aspects": ["custom", "quality", "return"],
        }
        first = rag.run("定制商品质量问题能不能退", **kwargs)
        second = rag.run("定制商品质量问题能不能退", **kwargs)

        self.assertEqual(first.status, "GROUNDED")
        self.assertEqual(first.retrieval_count, 1)
        self.assertEqual(second.status, "GROUNDED")
        self.assertEqual(second.retrieval_count, 0)
        self.assertEqual(second.evidence_ids, first.evidence_ids)

    def test_agentic_rag_can_execute_rewrite_and_second_retrieval(self):
        planner = ScriptedPlanner([
            PlanDecision(PlanAction.RETRIEVE, "先检索", query="杯子能退吗", metadata_filter={"product_type": "custom", "reason": "quality_issue"}),
            PlanDecision(PlanAction.REWRITE_QUERY, "第一次证据不足，补充例外", query="杯子能退吗", metadata_filter={"product_type": "custom", "reason": "quality_issue"}),
            PlanDecision(PlanAction.VERIFY, "校验二次证据"),
            PlanDecision(PlanAction.ANSWER_DIRECTLY, "证据已充分"),
        ])
        rag = AgenticRAG(self.corpus, planner=planner)

        result = rag.run(
            "杯子能退吗",
            metadata_filter={"product_type": "custom", "reason": "quality_issue"},
            required_aspects=["custom", "quality", "return"],
        )

        self.assertEqual(result.status, "GROUNDED")
        self.assertEqual(result.retrieval_count, 2)
        self.assertEqual(
            [item["action"] for item in result.trace if item["event"] == "plan"],
            ["retrieve", "rewrite_query", "verify", "answer_directly"],
        )
        self.assertTrue(any(item["event"] == "evidence_verify" for item in result.trace))

    def test_agentic_rag_rewrites_when_metadata_match_has_no_textual_signal(self):
        result = AgenticRAG(self.corpus).run(
            "政策 abc123",
            metadata_filter={"product_type": "custom", "reason": "quality_issue"},
            required_aspects=["custom", "quality", "return", "thirty_days"],
        )

        self.assertEqual(result.status, "GROUNDED")
        self.assertEqual(result.retrieval_count, 2)
        self.assertEqual(result.evidence_ids, ["policy_custom_quality_exception_v2"])
        self.assertEqual(
            [item["action"] for item in result.trace if item["event"] == "plan"],
            ["retrieve", "rewrite_query", "answer_directly"],
        )
        first_retrieval = next(item for item in result.trace if item["event"] == "retrieve")
        self.assertIn("LOW_RELEVANCE_EVIDENCE_REJECTED", first_retrieval["validation"]["reason_codes"])

    def test_planner_can_choose_direct_answer_without_policy_retrieval(self):
        planner = ScriptedPlanner([
            PlanDecision(PlanAction.ANSWER_DIRECTLY, "已有业务状态，不需要政策库"),
        ])
        result = AgenticRAG(self.corpus, planner=planner).run("订单查询结果已经返回")

        self.assertEqual(result.status, "NO_RETRIEVAL")
        self.assertEqual(result.retrieval_count, 0)

    def test_policy_sensitive_direct_answer_is_overridden_without_evidence(self):
        planner = ScriptedPlanner([
            PlanDecision(PlanAction.ANSWER_DIRECTLY, "模型猜测可以退"),
            PlanDecision(PlanAction.ESCALATE, "没有证据"),
        ])
        result = AgenticRAG(self.corpus, planner=planner).run("这个商品能不能退？")

        self.assertEqual(result.status, "UNCERTAIN")
        self.assertNotIn("模型猜测", result.answer)
        self.assertTrue(any("安全护栏" in item.get("reason", "") for item in result.trace if item["event"] == "plan") or result.retrieval_count > 0)

    def test_deepseek_planner_uses_structured_function_contract(self):
        model = PlannerModel({
            "action": "retrieve",
            "reason": "需要查政策",
            "confidence": 0.9,
            "query": "定制商品质量问题退货",
            "retrieval_mode": "hybrid",
            "metadata_filter": {"product_type": "custom", "reason": "quality_issue"},
        })
        decision = DeepSeekRAGPlanner(model).plan({"policy_sensitive": True})

        self.assertEqual(decision.action, PlanAction.RETRIEVE)
        self.assertEqual(model.calls[0][1][0]["function"]["name"], "emit_rag_plan")

    def test_policy_expert_can_use_agentic_rag_before_mcp_policy_check(self):
        from baseline.tool_catalog import SessionToolExecutor
        from structured.models import StructuredTaskState

        executor = SessionToolExecutor(user_id="U001")
        rag = AgenticRAG(self.corpus)
        expert = PolicyExpert(executor.execute, rag_agent=rag)
        state = StructuredTaskState(user_id="U001")
        state.set_slot("order_id", "O10086", __import__("structured.models", fromlist=["SlotSource"]).SlotSource.DETERMINISTIC, 1.0, "test")
        state.selected_item_ids = ["I002"]
        state.set_slot("reason", "no_reason_return", __import__("structured.models", fromlist=["SlotSource"]).SlotSource.DERIVED, 1.0, "test")
        # The expert requires a grounded order fact, just as the real order
        # expert produces before handing off.
        from structured.models import ToolFact
        order = executor.execute("get_order", {"order_id": "O10086"})
        state.facts["get_order"] = ToolFact("get_order", order.data, "get_order", 1)

        result = expert.execute(state)

        self.assertEqual(result.status.value, "continue")
        self.assertTrue(any(fact.tool == "agentic_rag" for fact in state.facts.values()))
        self.assertIn("calculate_refund", [event.tool for event in executor.audit_log])


if __name__ == "__main__":
    unittest.main(verbosity=2)
