"""Semantic relevance grading after deterministic business filtering."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from baseline.deepseek_client import ChatModelClient

from .models import RetrievalQuery, RetrievedEvidence
from .retrieval import tokenize


ASPECT_ALIASES: dict[str, tuple[str, ...]] = {
    "standard": ("standard", "普通", "常规"),
    "custom": ("custom", "定制", "专属制作"),
    "quality": ("quality", "质量", "损坏", "坏了", "故障", "瑕疵"),
    "return": ("return", "退货", "退换", "能退", "换货"),
    "no_reason_return": ("no_reason_return", "无理由", "不想要", "不合适"),
    "seven_days": ("seven_days", "七天", "7天"),
    "fourteen_days": ("fourteen_days", "十四天", "14天"),
    "thirty_days": ("thirty_days", "三十天", "30天"),
}


@dataclass(frozen=True)
class DocumentGrade:
    citation_id: str
    relevant: bool
    score: float
    covered_aspects: list[str] = field(default_factory=list)
    missing_aspects: list[str] = field(default_factory=list)
    noise: bool = False
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "citation_id": self.citation_id,
            "relevant": self.relevant,
            "score": round(self.score, 6),
            "covered_aspects": list(self.covered_aspects),
            "missing_aspects": list(self.missing_aspects),
            "noise": self.noise,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class DocumentGradingResult:
    grades: list[DocumentGrade] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def relevant_ids(self) -> set[str]:
        return {grade.citation_id for grade in self.grades if grade.relevant}

    @property
    def missing_aspects(self) -> list[str]:
        missing = {
            aspect
            for grade in self.grades
            if grade.relevant
            for aspect in grade.missing_aspects
        }
        covered = {
            aspect
            for grade in self.grades
            if grade.relevant
            for aspect in grade.covered_aspects
        }
        return sorted(missing - covered)

    def as_dict(self) -> dict[str, Any]:
        return {
            "grades": [grade.as_dict() for grade in self.grades],
            "relevant_ids": sorted(self.relevant_ids),
            "missing_aspects": self.missing_aspects,
            "usage": dict(self.usage),
        }


class DocumentGrader(Protocol):
    def grade(
        self,
        query: RetrievalQuery,
        evidence: list[RetrievedEvidence],
        required_aspects: list[str] | None = None,
    ) -> DocumentGradingResult:
        ...


class HeuristicDocumentGrader:
    """Reproducible offline grader; production mode should inject the LLM grader."""

    def grade(
        self,
        query: RetrievalQuery,
        evidence: list[RetrievedEvidence],
        required_aspects: list[str] | None = None,
    ) -> DocumentGradingResult:
        required = list(dict.fromkeys(required_aspects or infer_query_aspects(query)))
        query_terms = {
            term for term in tokenize(query.text)
            if len(term) > 1 or term.isascii()
        }
        grades: list[DocumentGrade] = []
        for item in evidence:
            document_terms = {
                term for term in tokenize(item.document.title + " " + item.document.text)
                if len(term) > 1 or term.isascii()
            }
            overlap = len(query_terms & document_terms) / max(len(query_terms), 1)
            retrieval_signal = max(item.vector_score, item.lexical_score, item.rerank_score)
            covered = [
                aspect for aspect in required
                if evidence_covers_aspect(item, aspect)
            ]
            missing = [aspect for aspect in required if aspect not in covered]
            # Metadata may make a candidate applicable, but relevance still
            # requires semantic or retrieval signal. This rejects narrow-filter
            # noise such as a random identifier with no policy meaning.
            semantic_signal = overlap > 0 or bool(covered) or retrieval_signal >= 0.2
            noise = not semantic_signal or retrieval_signal <= 0
            score = min(1.0, 0.45 * overlap + 0.35 * retrieval_signal + 0.2 * (len(covered) / max(len(required), 1)))
            relevant = semantic_signal and not noise and score >= 0.12
            grades.append(
                DocumentGrade(
                    citation_id=item.citation_id,
                    relevant=relevant,
                    score=score,
                    covered_aspects=covered,
                    missing_aspects=missing,
                    noise=not relevant,
                    reason=(
                        "检索结果与问题有语义信号并覆盖部分事实维度"
                        if relevant else "仅有元数据命中或检索信号不足，判定为噪声"
                    ),
                )
            )
        return DocumentGradingResult(grades)


DOCUMENT_GRADING_TOOL = {
    "type": "function",
    "function": {
        "name": "emit_document_grades",
        "description": "逐条判断政策证据是否真正回答用户问题；不得判断政策版本权威性。",
        "parameters": {
            "type": "object",
            "properties": {
                "grades": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "citation_id": {"type": "string"},
                            "relevant": {"type": "boolean"},
                            "score": {"type": "number", "minimum": 0, "maximum": 1},
                            "covered_aspects": {"type": "array", "items": {"type": "string"}},
                            "missing_aspects": {"type": "array", "items": {"type": "string"}},
                            "noise": {"type": "boolean"},
                            "reason": {"type": "string"},
                        },
                        "required": [
                            "citation_id", "relevant", "score", "covered_aspects",
                            "missing_aspects", "noise", "reason"
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["grades"],
            "additionalProperties": False,
        },
    },
}


DOCUMENT_GRADER_PROMPT = """你是电商售后政策检索的语义相关性评分器。
你只能判断：证据是否真正回答用户问题、覆盖哪些事实维度、是否属于噪声召回。
你不能判断政策是否生效、哪个版本优先、是否有权限、是否存在冲突，也不能授权退款或退货；这些由确定性 EvidenceValidator 决定。
必须为输入中的每个 citation_id 返回一个 grade，并调用 emit_document_grades。"""


class DeepSeekDocumentGrader:
    def __init__(self, model_client: ChatModelClient) -> None:
        self.model_client = model_client

    def grade(
        self,
        query: RetrievalQuery,
        evidence: list[RetrievedEvidence],
        required_aspects: list[str] | None = None,
    ) -> DocumentGradingResult:
        payload = {
            "query": query.text,
            "required_aspects": list(required_aspects or infer_query_aspects(query)),
            "evidence": [
                {
                    "citation_id": item.citation_id,
                    "title": item.document.title,
                    "text": item.document.text,
                    "tags": sorted(item.document.tags),
                }
                for item in evidence
            ],
        }
        completion = self.model_client.complete(
            [
                {"role": "system", "content": DOCUMENT_GRADER_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            [DOCUMENT_GRADING_TOOL],
        )
        raw = _extract_arguments(completion.message)
        rows = raw.get("grades")
        if not isinstance(rows, list):
            raise ValueError("Document Grader 返回结构无效。")
        expected = {item.citation_id for item in evidence}
        grades: list[DocumentGrade] = []
        for row in rows:
            if not isinstance(row, dict) or row.get("citation_id") not in expected:
                raise ValueError("Document Grader 返回未知 citation_id。")
            if isinstance(row.get("score"), bool) or not isinstance(row.get("score"), (int, float)):
                raise ValueError("Document Grader score 必须是数字。")
            grades.append(
                DocumentGrade(
                    citation_id=str(row["citation_id"]),
                    relevant=bool(row.get("relevant")),
                    score=max(0.0, min(float(row["score"]), 1.0)),
                    covered_aspects=_string_list(row.get("covered_aspects")),
                    missing_aspects=_string_list(row.get("missing_aspects")),
                    noise=bool(row.get("noise")),
                    reason=str(row.get("reason", "")),
                )
            )
        if {grade.citation_id for grade in grades} != expected:
            raise ValueError("Document Grader 必须逐条返回全部候选证据。")
        return DocumentGradingResult(grades, dict(completion.usage))


def infer_query_aspects(query: RetrievalQuery) -> list[str]:
    aspects: list[str] = []
    product_type = query.metadata_filter.get("product_type")
    if product_type:
        aspects.append(str(product_type))
    reason = query.metadata_filter.get("reason")
    if reason == "quality_issue":
        aspects.extend(["quality", "return"])
    elif reason == "no_reason_return":
        aspects.extend(["no_reason_return", "return"])
    normalized = query.text.replace(" ", "")
    for aspect, aliases in ASPECT_ALIASES.items():
        if any(alias in normalized for alias in aliases if not alias.isascii()):
            aspects.append(aspect)
    if any(term in normalized for term in ("几天", "多少天", "期限", "多久")):
        aspects.append("return")
    return list(dict.fromkeys(aspects))


def evidence_covers_aspect(item: RetrievedEvidence, aspect: str) -> bool:
    if aspect in item.document.tags:
        return True
    haystack = item.document.title + " " + item.document.text
    return any(alias in haystack for alias in ASPECT_ALIASES.get(aspect, (aspect,)))


def _extract_arguments(message: dict[str, Any]) -> dict[str, Any]:
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise ValueError("Document Grader 必须返回且只返回一个函数调用。")
    function = calls[0].get("function") if isinstance(calls[0], dict) else None
    if not isinstance(function, dict) or function.get("name") != "emit_document_grades":
        raise ValueError("Document Grader 返回了不允许的函数。")
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, dict):
        raise ValueError("Document Grader 参数必须是 JSON 对象。")
    return arguments


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("Document Grader 的维度字段必须是字符串数组。")
    return list(value)

