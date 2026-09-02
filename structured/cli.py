"""阶段三结构化 Agent 的交互式调试终端。

运行：python3 -m structured.cli [--offline] [--user-id U001]
--offline 用 EmptySemanticParser（不调 LLM，纯确定性模式）。
命令：state/memory/trace 查看内部状态，reset 重置，exit 退出。
"""

from __future__ import annotations

import argparse
import sys

from baseline.deepseek_client import DeepSeekAPIError, DeepSeekClient, DeepSeekConfigurationError
from baseline.tool_catalog import SessionToolExecutor

from .agent import StructuredCustomerServiceAgent
from .semantic_parser import DeepSeekSemanticParser, EmptySemanticParser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Structured state + multi-turn memory Agent")
    parser.add_argument("--user-id", default="U001")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="只使用确定性提取器，便于不调用 API 的状态机调试。",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.offline:
        parser = EmptySemanticParser()
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
        parser = DeepSeekSemanticParser(client)
        model_name = client.model

    tool_executor = SessionToolExecutor(user_id=args.user_id)
    agent = StructuredCustomerServiceAgent(
        semantic_parser=parser,
        tool_executor=tool_executor,
        user_id=args.user_id,
    )
    print(
        f"结构化 Agent 已启动，parser={model_name}，user={args.user_id}。"
        "输入 state/memory/trace 查看内部状态，audit 查看工具审计，reset 重置，exit 退出。"
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
        if text.lower() == "audit":
            print(tool_executor.audit_as_json())
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
            print(f"Agent：语义解析服务调用失败：{exc}")


if __name__ == "__main__":
    raise SystemExit(main())
