"""Interactive Agentic RAG inspection CLI."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from baseline.deepseek_client import DeepSeekAPIError, DeepSeekClient, DeepSeekConfigurationError

from .agent import AgenticRAG
from .dense_retrieval import DenseHybridRetriever
from .embeddings import load_embedding_provider
from .generation import DeepSeekAnswerGenerator
from .grading import DeepSeekDocumentGrader
from .index import PersistentIndex
from .monitoring import RAGMonitor
from .models import RetrievalQuery
from .planner import DeepSeekRAGPlanner, HeuristicRAGPlanner
from .retrieval import HybridRetriever, VectorRetriever, load_policy_corpus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage-six BM25/Hybrid/Agentic RAG")
    parser.add_argument("--mode", choices=["bm25", "vector", "hybrid", "agentic"], default="agentic")
    parser.add_argument("--model", default=None)
    parser.add_argument("--as-of", default="2026-07-16")
    parser.add_argument("--product-type", choices=["standard", "custom"], default=None)
    parser.add_argument("--reason", choices=["quality_issue", "no_reason_return"], default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--production", action="store_true", help="启用持久化 dense+sparse 索引和 LLM grounded generation")
    parser.add_argument("--index-root", default=None)
    parser.add_argument("--embedding-mode", choices=["env", "local", "hash"], default="env")
    parser.add_argument("--generator-mode", choices=["llm", "deterministic"], default=None)
    parser.add_argument("--grader-mode", choices=["llm", "heuristic"], default=None)
    parser.add_argument("--monitor-sink", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        as_of = date.fromisoformat(args.as_of)
    except ValueError:
        print("--as-of 必须是 YYYY-MM-DD。", file=sys.stderr)
        return 2
    corpus = load_policy_corpus()
    metadata_filter = {
        key: value
        for key, value in {
            "product_type": args.product_type,
            "reason": args.reason,
        }.items()
        if value is not None
    }
    if args.mode in {"bm25", "vector"}:
        retriever = VectorRetriever(corpus)
    elif args.mode == "hybrid":
        retriever = HybridRetriever(corpus)
    else:
        retriever = None
        generator = None
        grader = None
        if args.offline:
            planner = HeuristicRAGPlanner()
            model_name = "offline-heuristic"
        else:
            try:
                client = DeepSeekClient.from_env(model=args.model)
            except DeepSeekConfigurationError as exc:
                print(f"配置错误：{exc}", file=sys.stderr)
                return 2
            planner = DeepSeekRAGPlanner(client)
            model_name = client.model
            if (args.generator_mode or "llm") == "llm":
                generator = DeepSeekAnswerGenerator(client)
            if (args.grader_mode or "llm") == "llm":
                grader = DeepSeekDocumentGrader(client)
        if args.production:
            try:
                provider = load_embedding_provider("hash" if args.offline else args.embedding_mode)
                index_root = args.index_root or Path(__file__).parents[1] / "var" / "rag_indexes"
                persistent_index = PersistentIndex(index_root)
                retriever = DenseHybridRetriever(persistent_index, provider, corpus)
            except (ValueError, RuntimeError, OSError) as exc:
                print(f"生产 RAG 配置错误：{exc}", file=sys.stderr)
                return 2
        rag = AgenticRAG(
            corpus,
            planner=planner,
            retriever=retriever,
            grader=grader,
            generator=generator,
            monitor=RAGMonitor(args.monitor_sink) if args.monitor_sink else None,
            as_of=as_of,
        )
        print(f"Agentic RAG 已启动，planner={model_name}，输入 exit 退出。")
        while True:
            try:
                query = input("问题：").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n会话结束。")
                return 0
            if query.lower() in {"exit", "quit", "退出"}:
                return 0
            if not query:
                continue
            try:
                result = rag.run(query, metadata_filter=metadata_filter, as_of=as_of)
            except DeepSeekAPIError as exc:
                print(f"Agentic RAG：模型调用失败：{exc}")
                continue
            print(json.dumps({
                "status": result.status,
                "answer": result.answer,
                "evidence_ids": result.evidence_ids,
                "citation_ids": result.citation_ids,
                "retrieval_count": result.retrieval_count,
                "planner_calls": result.planner_calls,
                "generator_calls": result.generator_calls,
                "grader_calls": result.grader_calls,
                "rewrite_count": result.rewrite_count,
                "run_id": result.run_id,
                "stop_reason": result.stop_reason,
                "latency_ms": round(result.latency_ms, 2),
                "validation": result.validation.as_dict(),
                "citation_validation": result.citation_validation.as_dict() if result.citation_validation else None,
                "trace": result.trace,
            }, ensure_ascii=False, indent=2))
        
    print(f"{args.mode} RAG 已启动，输入 exit 退出。")
    while True:
        try:
            query = input("问题：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n会话结束。")
            return 0
        if query.lower() in {"exit", "quit", "退出"}:
            return 0
        if not query:
            continue
        request = RetrievalQuery(query, as_of=as_of, metadata_filter=metadata_filter)
        results = retriever.search(request, top_k=5)
        print(json.dumps([item.as_dict() for item in results], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
