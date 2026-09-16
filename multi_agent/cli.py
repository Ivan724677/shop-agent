"""Interactive CLI for the stage-five deterministic-first multi-expert agent."""

from __future__ import annotations

import argparse
import sys

from baseline.deepseek_client import DeepSeekAPIError, DeepSeekClient, DeepSeekConfigurationError
from baseline.tool_catalog import SessionToolExecutor
from routing.router import MultiAgentRouter
from routing.semantic_router import DeepSeekSemanticRouter
from rag.agent import AgenticRAG
from rag.dense_retrieval import DenseHybridRetriever
from rag.embeddings import load_embedding_provider
from rag.generation import DeepSeekAnswerGenerator
from rag.index import PersistentIndex
from rag.monitoring import RAGMonitor
from rag.planner import DeepSeekRAGPlanner
from rag.retrieval import load_policy_corpus
from structured.semantic_parser import DeepSeekSemanticParser, EmptySemanticParser

from .agent import MultiExpertCustomerServiceAgent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Structured state + routed expert agents")
    parser.add_argument("--user-id", default="U001")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="关闭语义解析和模型二次路由，只运行确定性链路。",
    )
    parser.add_argument("--production-rag", action="store_true", help="Policy Expert 使用持久化 dense+sparse Agentic RAG")
    parser.add_argument("--index-root", default=None)
    parser.add_argument("--embedding-mode", choices=["env", "local", "hash"], default="env")
    parser.add_argument("--monitor-sink", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.offline:
        semantic_parser = EmptySemanticParser()
        router = MultiAgentRouter()
        rag_agent = None
        model_name = "offline-deterministic-only"
    else:
        try:
            client = DeepSeekClient.from_env(
                model=args.model,
                base_url=args.base_url,
                timeout=args.timeout,
            )
        except DeepSeekConfigurationError as exc:
            print(f"配置错误：{exc}", file=sys.stderr)
            return 2
        semantic_parser = DeepSeekSemanticParser(client)
        router = MultiAgentRouter(DeepSeekSemanticRouter(client))
        rag_kwargs = {}
        if args.production_rag:
            try:
                corpus = load_policy_corpus()
                provider = load_embedding_provider(args.embedding_mode)
                index_root = args.index_root or "var/rag_indexes"
                rag_kwargs = {
                    "retriever": DenseHybridRetriever(PersistentIndex(index_root), provider, corpus),
                    "generator": DeepSeekAnswerGenerator(client),
                    "monitor": RAGMonitor(args.monitor_sink) if args.monitor_sink else None,
                }
            except (ValueError, RuntimeError, OSError) as exc:
                print(f"生产 RAG 配置错误：{exc}", file=sys.stderr)
                return 2
        rag_agent = AgenticRAG(load_policy_corpus(), planner=DeepSeekRAGPlanner(client), **rag_kwargs)
        model_name = client.model

    executor = SessionToolExecutor(user_id=args.user_id)
    agent = MultiExpertCustomerServiceAgent(
        semantic_parser=semantic_parser,
        tool_executor=executor,
        router=router,
        rag_agent=rag_agent,
        user_id=args.user_id,
    )
    print(
        f"多专家 Agent 已启动，model={model_name}，user={args.user_id}。"
        "输入 state/memory/trace/routes/audit 查看内部记录，reset 重置，exit 退出。"
    )
    while True:
        try:
            text = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n会话结束。")
            return 0
        if text.lower() in {"exit", "quit", "退出"}:
            print("会话结束。")
            return 0
        if text.lower() == "state":
            print(agent.state_as_json())
            continue
        if text.lower() == "memory":
            print(agent.memory_as_json())
            continue
        if text.lower() == "trace":
            print(agent.trace_as_json())
            continue
        if text.lower() in {"route", "routes"}:
            print(agent.route_trace_as_json())
            continue
        if text.lower() == "audit":
            print(executor.audit_as_json())
            continue
        if text.lower() == "reset":
            agent.reset()
            print("会话已重置。")
            continue
        if not text:
            continue
        try:
            print("Agent：" + agent.handle(text))
        except DeepSeekAPIError as exc:
            print(f"Agent：模型服务调用失败：{exc}")


if __name__ == "__main__":
    raise SystemExit(main())
