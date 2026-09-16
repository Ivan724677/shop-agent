import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from baseline.deepseek_client import ModelCompletion
from evaluation.online_rag import OnlineRAGEvaluator
from rag import (
    AgenticRAG,
    ChunkingConfig,
    CitationValidator,
    DenseHybridRetriever,
    GeneratedAnswer,
    HashEmbeddingProvider,
    IndexLifecycleError,
    PersistentIndex,
    PolicyChunker,
    PolicyIngestionPipeline,
    RAGMonitor,
    RetrievalQuery,
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


if __name__ == "__main__":
    unittest.main()
