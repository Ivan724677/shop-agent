"""真实 DeepSeek 回放的对照评测入口。

运行：python3 -m evaluation.compare_cli [--limit N] [--output out.json]
读 DEEPSEEK_API_KEY，对 golden 场景分别跑 baseline、structured 和 multi_expert，
输出 summaries + results 的 JSON 报告。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from baseline.deepseek_client import DeepSeekClient
from baseline.react_agent import ReActBaselineAgent
from baseline.tool_catalog import SessionToolExecutor
from multi_agent.agent import MultiExpertCustomerServiceAgent
from rag.agent import AgenticRAG
from rag.planner import DeepSeekRAGPlanner
from rag.retrieval import load_policy_corpus
from routing.router import MultiAgentRouter
from routing.semantic_router import DeepSeekSemanticRouter
from scenarios.loader import load_directory
from structured.agent import StructuredCustomerServiceAgent
from structured.semantic_parser import DeepSeekSemanticParser

from .comparison import compare_variants, results_as_dicts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare ReAct baseline vs structured state Agent")
    parser.add_argument("--scenario-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    client = DeepSeekClient.from_env(model=args.model)
    scenarios = load_directory(Path(__file__).parents[1] / "scenarios" / "golden")
    if args.scenario_id:
        wanted = set(args.scenario_id)
        scenarios = [scenario for scenario in scenarios if scenario.scenario_id in wanted]
    scenarios = scenarios[: args.limit]
    if not scenarios:
        raise SystemExit("没有匹配的场景。")

    def baseline_factory(user_id: str):
        executor = SessionToolExecutor(user_id=user_id)
        return ReActBaselineAgent(client, tool_executor=executor), executor

    def structured_factory(user_id: str):
        executor = SessionToolExecutor(user_id=user_id)
        parser = DeepSeekSemanticParser(client)
        return (
            StructuredCustomerServiceAgent(
                semantic_parser=parser,
                tool_executor=executor,
                user_id=user_id,
            ),
            executor,
        )

    def multi_expert_factory(user_id: str):
        executor = SessionToolExecutor(user_id=user_id)
        rag_agent = AgenticRAG(
            load_policy_corpus(),
            planner=DeepSeekRAGPlanner(client),
        )
        return (
            MultiExpertCustomerServiceAgent(
                semantic_parser=DeepSeekSemanticParser(client),
                tool_executor=executor,
                router=MultiAgentRouter(DeepSeekSemanticRouter(client)),
                rag_agent=rag_agent,
                user_id=user_id,
            ),
            executor,
        )

    results, summaries = compare_variants(
        scenarios,
        {
            "react_baseline": baseline_factory,
            "structured_state": structured_factory,
            "multi_expert": multi_expert_factory,
        },
    )
    payload = {
        "summaries": [asdict(summary) for summary in summaries],
        "results": results_as_dicts(results),
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"comparison written to {args.output}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
