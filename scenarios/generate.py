"""Generate deterministic variations for batch replay evaluation.

Usage:
    python3 -m scenarios.generate --count 120 --output scenarios/generated/batch.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


PRODUCTS = [
    ("I002", "运动鞋", "O10086", "U001", 129.0),
    ("I004", "机械键盘", "O20001", "U002", 399.0),
]

RETURN_MESSAGES = [
    "我要退{product}，订单是 {order_id}",
    "帮我把{product}退掉，单号{order_id}",
    "{order_id}里的{product}我不想要了，能申请退货吗",
    "请处理一下订单 {order_id} 的{product}退货",
]

QUERY_MESSAGES = [
    "帮我查一下订单 {order_id}",
    "订单 {order_id} 现在到哪里了",
    "请告诉我 {order_id} 的物流状态",
]


def make_scenario(index: int, rng: random.Random) -> dict:
    item_id, product, order_id, user_id, amount = rng.choice(PRODUCTS)
    if rng.random() < 0.65:
        message = rng.choice(RETURN_MESSAGES).format(product=product, order_id=order_id)
        return {
            "scenario_id": f"generated_return_{index:04d}",
            "description": f"模板生成的{product}退货场景",
            "user_id": user_id,
            "conversation": [{"role": "user", "content": message}],
            "business_state": {
                "order_id": order_id,
                "target_item_id": item_id,
                "refund_amount": amount,
            },
            "expected_intent": "REQUEST_RETURN",
            "required_slots": ["order_id", "target_item_id"],
            "allowed_actions": [
                "get_order",
                "search_policy",
                "calculate_refund",
                "create_return_request",
            ],
            "forbidden_actions": ["refund_excluded_item", "refund_entire_order"],
            "expected_final_state": "AWAITING_CONFIRMATION",
            "risk_level": "high",
            "tags": ["generated", "return", "template"],
        }

    message = rng.choice(QUERY_MESSAGES).format(order_id=order_id)
    return {
        "scenario_id": f"generated_query_{index:04d}",
        "description": f"模板生成的{order_id}查询场景",
        "user_id": user_id,
        "conversation": [{"role": "user", "content": message}],
        "business_state": {"order_id": order_id},
        "expected_intent": "QUERY_ORDER",
        "required_slots": ["order_id"],
        "allowed_actions": ["get_order", "list_shipments", "get_shipment"],
        "forbidden_actions": ["create_return_request", "create_ticket"],
        "expected_final_state": "ANSWERED",
        "risk_level": "low",
        "tags": ["generated", "query", "template"],
    }


def generate(count: int, seed: int = 20260805) -> list[dict]:
    if count <= 0:
        raise ValueError("count 必须大于 0。")
    rng = random.Random(seed)
    return [make_scenario(index, rng) for index in range(1, count + 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic Agent scenarios")
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for scenario in generate(args.count, args.seed):
            handle.write(json.dumps(scenario, ensure_ascii=False) + "\n")
    print(f"generated {args.count} scenarios at {args.output}")


if __name__ == "__main__":
    main()
