"""Interactive CLI for the DeepSeek ReAct baseline."""

from __future__ import annotations

import argparse
import json
import sys

from .deepseek_client import DeepSeekAPIError, DeepSeekClient, DeepSeekConfigurationError
from .react_agent import ReActBaselineAgent
from .tool_catalog import SessionToolExecutor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DeepSeek V4 Flash single-Agent ReAct baseline")
    parser.add_argument("--model", default=None, help="默认读取 DEEPSEEK_MODEL")
    parser.add_argument("--base-url", default=None, help="默认读取 DEEPSEEK_BASE_URL")
    parser.add_argument("--user-id", default="U001")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        client = DeepSeekClient.from_env(
            model=args.model,
            base_url=args.base_url,
            timeout=args.timeout,
        )
    except DeepSeekConfigurationError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2

    agent = ReActBaselineAgent(
        client,
        tool_executor=SessionToolExecutor(user_id=args.user_id),
        max_steps=args.max_steps,
    )
    print(
        f"DeepSeek ReAct baseline 已启动，model={client.model}，user={args.user_id}。"
        "输入 trace 查看轨迹，history 查看消息，reset 重置，exit 退出。"
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
        if text.lower() == "trace":
            print(agent.trace_as_json())
            continue
        if text.lower() == "history":
            print(json.dumps(agent.messages, ensure_ascii=False, indent=2))
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
