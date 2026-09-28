"""可复现的 Vector / Hybrid / Agentic RAG 离线对照评测入口。

运行：
    python3 -m evaluation.rag_compare_cli
    python3 -m evaluation.rag_compare_cli --case-id same_precedence_version_conflict
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from rag.retrieval import load_policy_corpus

from .rag_comparison import evaluate_case, load_cases, summarize, summarize_by_category


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare dependency-free vector, hybrid and Agentic RAG"
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(__file__).with_name("rag_cases.json"),
        help="JSON case fixture path",
    )
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cases = load_cases(args.cases)
    if args.case_id:
        wanted = set(args.case_id)
        cases = [case for case in cases if case.case_id in wanted]
    if args.limit > 0:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("没有匹配的 RAG 评测场景。")

    corpus = load_policy_corpus()
    rows = [row for case in cases for row in evaluate_case(corpus, case)]
    payload = {
        "cases": [asdict(case) for case in cases],
        "summaries": summarize(rows),
        "category_summaries": summarize_by_category(rows),
        "results": rows,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"RAG comparison written to {args.output}")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
