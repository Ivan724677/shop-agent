import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from baseline.deepseek_client import ModelCompletion
from evaluation.online_rag import OnlineRAGEvaluator
from rag import (
    AgenticRAG,
    BM25Retriever,
    ChunkingConfig,
    CitationValidator,
    DeepSeekDocumentGrader,
    DenseHybridRetriever,
    DeterministicBusinessFilter,
    GeneratedAnswer,
    HashEmbeddingProvider,
    IndexLifecycleError,
    PersistentIndex,
    PolicyCorpus,
    PolicyDocument,
    PolicyChunker,
    PolicyIngestionPipeline,
    RAGMonitor,
    RetrievalQuery,
    RetrievedEvidence,
    load_policy_corpus,
)
from rag.generation import AnswerClaim, DeepSeekAnswerGenerator


class GeneratorModel:
    def __init__(self, arguments):
        self.arguments = arguments
        self.calls = []

    def complete(self, messages, tools):
        self.calls.append((messages, tools))
        return ModelCompletion(
            message={
                "role": "assistant",
                "tool_calls": [{
                    "id": "answer_1",
                    "type": "function",
                    "function": {
                        "name": "emit_grounded_answer",
                        "arguments": json.dumps(self.arguments, ensure_ascii=False),
                    },
                }],
            },
            usage={"prompt_tokens": 10, "completion_tokens": 8},
        )


class GraderModel:
    def __init__(self, grades):
        self.grades = grades

    def complete(self, messages, tools):
        return ModelCompletion(
            message={
                "role": "assistant",
                "tool_calls": [{
                    "id": "grade_1",
                    "type": "function",
                    "function": {
                        "name": "emit_document_grades",
                        "arguments": json.dumps({"grades": self.grades}, ensure_ascii=False),
                    },
                }],
            },
            usage={"prompt_tokens": 20, "completion_tokens": 10},
        )


class DirectPlanner:
    def plan(self, observation):
        from rag.models import PlanAction, PlanDecision

        if observation["evidence_sufficient"]:
            return PlanDecision(PlanAction.ANSWER_DIRECTLY, "evidence ready")
        return PlanDecision(PlanAction.RETRIEVE, "retrieve", query=observation["user_query"])


class ProductionRAGTests(unittest.TestCase):
    def setUp(self):
        self.corpus = load_policy_corpus()

    def test_ingestion_publish_rollback_and_dense_sparse_chunk_citations(self):
        with tempfile.TemporaryDirectory() as directory:
            index = PersistentIndex(Path(directory))
            provider = HashEmbeddingProvider(dimension=64)
            pipeline = PolicyIngestionPipeline(
                provider,
                index,
                PolicyChunker(ChunkingConfig(max_chars=28, overlap_chars=4)),
            )
            first = pipeline.build(self.corpus, "v1", publish=True)
            second = pipeline.build(self.corpus, "v2", publish=False)
            self.assertEqual(first.chunk_count, len(index.load("v1")[1]))
            self.assertEqual(index.current_version(), "v1")
            with self.assertRaises(IndexLifecycleError):
                pipeline.build(self.corpus, "v1")
            index.rollback("v2")
            self.assertEqual(index.current_version(), "v2")
            retriever = DenseHybridRetriever(index, provider, self.corpus)
            results = retriever.search(RetrievalQuery("定制商品质量问题退货"), top_k=10)
            self.assertTrue(results)
            self.assertTrue(all(item.chunk_id for item in results))
            self.assertTrue(any(item.lexical_score > 0 for item in results))
            self.assertTrue(all(item.vector_score >= 0 for item in results))

    def test_generator_and_citation_gate_reject_unsupported_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            index = PersistentIndex(Path(directory))
            provider = HashEmbeddingProvider(dimension=64)
            PolicyIngestionPipeline(provider, index).build(self.corpus, "v1", publish=True)
            evidence = DenseHybridRetriever(index, provider, self.corpus).search(
                RetrievalQuery(
                    "定制商品质量问题三十天退货",
                    metadata_filter={"product_type": "custom", "reason": "quality_issue"},
                )
            )
            from rag.evidence import EvidenceValidator

            validation = EvidenceValidator(self.corpus).validate(
                RetrievalQuery(
                    "定制商品质量问题三十天退货",
                    metadata_filter={"product_type": "custom", "reason": "quality_issue"},
                ),
                evidence,
                required_aspects=["custom", "quality", "return"],
            )
            self.assertTrue(validation.sufficient)
            citation = validation.accepted[0].citation_id
            model = GeneratorModel({
                "answer": "可以在三十天内申请退货。",
                "claims": [{"text": "可以在三十天内申请退货。", "citation_ids": [citation]}],
                "abstain": False,
                "uncertainty": [],
            })
            generated = DeepSeekAnswerGenerator(model).generate("问题", validation)
            self.assertTrue(CitationValidator().validate(generated, validation).accepted)
            unsupported = GeneratedAnswer(
                "可以在九十天内申请退货。",
                [AnswerClaim("可以在九十天内申请退货。", [citation])],
            )
            rejected = CitationValidator().validate(unsupported, validation)
            self.assertFalse(rejected.accepted)
            self.assertIn("UNSUPPORTED_CLAIM", rejected.reason_codes)

    def test_agent_monitor_and_online_evaluation_preserve_citations(self):
        with tempfile.TemporaryDirectory() as directory:
            index = PersistentIndex(Path(directory))
            provider = HashEmbeddingProvider(dimension=64)
            PolicyIngestionPipeline(provider, index).build(self.corpus, "v1", publish=True)
            monitor = RAGMonitor()
            agent = AgenticRAG(
                self.corpus,
                planner=DirectPlanner(),
                retriever=DenseHybridRetriever(index, provider, self.corpus),
                monitor=monitor,
            )
            result = agent.run("定制商品质量问题三十天退货", metadata_filter={"product_type": "custom", "reason": "quality_issue"})
            self.assertEqual(result.status, "GROUNDED")
            self.assertTrue(result.citation_ids)
            self.assertEqual(len(monitor.samples), 1)
            self.assertEqual(monitor.samples[0].citation_ids, result.citation_ids)
            evaluator = OnlineRAGEvaluator(monitor)
            evaluator.record_result(result, expected_status="GROUNDED", expected_evidence_ids=result.citation_ids)
            summary = evaluator.snapshot()
            self.assertEqual(summary["status_accuracy"], 1.0)
            self.assertEqual(summary["evidence_recall"], 1.0)

    def test_bm25_length_normalization_prevents_long_noise_document_from_winning(self):
        short = PolicyDocument(
            "short", "质量问题退货", "质量问题可以申请退货。",
            frozenset({"standard", "quality", "return"}), "v1", date(2026, 1, 1),
        )
        long = PolicyDocument(
            "long", "综合说明", "质量问题 " + "物流库存会员积分说明 " * 80,
            frozenset({"standard", "quality", "return"}), "v1", date(2026, 1, 1),
        )
        results = BM25Retriever(PolicyCorpus([long, short])).search(
            RetrievalQuery("质量问题退货", metadata_filter={"product_type": "standard", "reason": "quality_issue"})
        )
        self.assertEqual(results[0].evidence_id, "short")

    def test_deterministic_business_filter_checks_permission_before_grader(self):
        protected = PolicyDocument(
            "protected", "内部补偿政策", "内部专员可查看补偿规则。",
            frozenset({"standard", "return"}), "v1", date(2026, 1, 1),
            metadata={"required_scopes": ["policy:internal"]},
        )
        corpus = PolicyCorpus([protected])
        evidence = [RetrievedEvidence(protected, lexical_score=1.0, rerank_score=1.0)]
        denied = DeterministicBusinessFilter(corpus).apply(
            RetrievalQuery("补偿政策", permission_scopes=frozenset({"policy:read"})), evidence
        )
        allowed = DeterministicBusinessFilter(corpus).apply(
            RetrievalQuery("补偿政策", permission_scopes=frozenset({"policy:read", "policy:internal"})), evidence
        )
        self.assertIn("PERMISSION_DENIED", denied.rejected["protected"])
        self.assertFalse(denied.accepted)
        self.assertEqual([item.evidence_id for item in allowed.accepted], ["protected"])

    def test_deepseek_document_grader_has_semantic_only_contract(self):
        item = BM25Retriever(self.corpus).search(
            RetrievalQuery(
                "定制商品质量问题退货",
                metadata_filter={"product_type": "custom", "reason": "quality_issue"},
            )
        )[0]
        model = GraderModel([{
            "citation_id": item.citation_id,
            "relevant": True,
            "score": 0.95,
            "covered_aspects": ["custom", "quality", "return"],
            "missing_aspects": [],
            "noise": False,
            "reason": "直接回答定制商品质量问题退货资格",
        }])
        result = DeepSeekDocumentGrader(model).grade(
            RetrievalQuery("定制商品质量问题退货"), [item], ["custom", "quality", "return"]
        )
        self.assertEqual(result.relevant_ids, {item.citation_id})
        self.assertEqual(result.usage["prompt_tokens"], 20)

    def test_explicit_graph_records_rewrite_delta_and_coverage_gain(self):
        result = AgenticRAG(self.corpus).run(
            "政策 abc123",
            metadata_filter={"product_type": "custom", "reason": "quality_issue"},
            required_aspects=["custom", "quality", "return", "thirty_days"],
        )
        events = [item["event"] for item in result.trace]
        self.assertIn("guardrail", events)
        self.assertIn("business_filter", events)
        self.assertIn("document_grade", events)
        self.assertIn("rewrite", events)
        self.assertIn("rewrite_outcome", events)
        self.assertIn("citation_validate", events)
        outcome = next(item for item in result.trace if item["event"] == "rewrite_outcome")
        self.assertTrue(outcome["added_terms"])
        self.assertTrue(outcome["new_evidence"])
        self.assertTrue(outcome["coverage_improved"])


if __name__ == "__main__":
    unittest.main()
