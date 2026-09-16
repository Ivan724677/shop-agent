"""Grounded LLM answer generation with a structured claim contract."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from baseline.deepseek_client import ChatModelClient

from .models import EvidenceValidation, RetrievedEvidence


@dataclass(frozen=True)
class AnswerClaim:
    text: str
    citation_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    claims: list[AnswerClaim]
    abstain: bool = False
    uncertainty: list[str] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)


class AnswerGenerator(Protocol):
    def generate(self, query: str, validation: EvidenceValidation) -> GeneratedAnswer:
        ...


class DeterministicAnswerGenerator:
    """Offline fallback; production mode should inject DeepSeekAnswerGenerator."""

    def generate(self, query: str, validation: EvidenceValidation) -> GeneratedAnswer:
        citations = [item.citation_id for item in validation.accepted]
        answer = "；".join(
            f"{item.document.title}：{item.document.text}" for item in validation.accepted
        )
        return GeneratedAnswer(
            answer=answer,
            claims=[AnswerClaim(item.document.text, [item.citation_id]) for item in validation.accepted],
        )


GROUNDING_TOOL = {
    "type": "function",
    "function": {
        "name": "emit_grounded_answer",
        "description": "基于给定证据生成回答；每条事实性 claim 必须引用证据 citation_id。",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "citation_ids": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["text", "citation_ids"],
                        "additionalProperties": False,
                    },
                },
                "abstain": {"type": "boolean"},
                "uncertainty": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["answer", "claims", "abstain", "uncertainty"],
            "additionalProperties": False,
        },
    },
}


GENERATOR_SYSTEM_PROMPT = """你是售后政策回答生成器。
只能使用给定的已校验证据，不得补充证据之外的期限、金额、资格或业务动作。
每条事实性 claim 必须引用一个或多个 citation_id；无法被证据支持时必须 abstain=true。
政策证据只说明适用规则，不代表订单已经批准退款或已经创建退货。
必须调用 emit_grounded_answer，不要输出普通自由文本。"""


class DeepSeekAnswerGenerator:
    def __init__(self, model_client: ChatModelClient) -> None:
        self.model_client = model_client
        self.last_usage: dict[str, Any] = {}

    def generate(self, query: str, validation: EvidenceValidation) -> GeneratedAnswer:
        evidence = [
            {
                "citation_id": item.citation_id,
                "evidence_id": item.evidence_id,
                "title": item.document.title,
                "text": item.document.text,
                "source_uri": item.source_uri,
                "source_locator": item.source_locator,
            }
            for item in validation.accepted
        ]
        completion = self.model_client.complete(
            [
                {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps({"query": query, "evidence": evidence}, ensure_ascii=False)},
            ],
            [GROUNDING_TOOL],
        )
        self.last_usage = dict(completion.usage)
        raw = _extract_arguments(completion.message)
        answer = raw.get("answer")
        claims = raw.get("claims")
        if not isinstance(answer, str) or not answer.strip() or not isinstance(claims, list):
            raise ValueError("Answer Generator 返回结构无效。")
        parsed_claims: list[AnswerClaim] = []
        for claim in claims:
            if not isinstance(claim, dict) or not isinstance(claim.get("text"), str) or not isinstance(claim.get("citation_ids"), list):
                raise ValueError("Answer Generator 的 claim 结构无效。")
            if not all(isinstance(value, str) for value in claim["citation_ids"]):
                raise ValueError("Answer Generator 的 citation_ids 必须是字符串数组。")
            parsed_claims.append(AnswerClaim(claim["text"], list(claim["citation_ids"])))
        return GeneratedAnswer(
            answer=answer.strip(),
            claims=parsed_claims,
            abstain=bool(raw.get("abstain", False)),
            uncertainty=[value for value in raw.get("uncertainty", []) if isinstance(value, str)],
            usage=self.last_usage,
        )


def _extract_arguments(message: dict[str, Any]) -> dict[str, Any]:
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise ValueError("Answer Generator 必须返回且只返回一个函数调用。")
    function = calls[0].get("function") if isinstance(calls[0], dict) else None
    if not isinstance(function, dict) or function.get("name") != "emit_grounded_answer":
        raise ValueError("Answer Generator 返回了不允许的函数。")
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, dict):
        raise ValueError("Answer Generator 参数必须是 JSON 对象。")
    return arguments

