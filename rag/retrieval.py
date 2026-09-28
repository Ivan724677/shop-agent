"""Deterministic vector, lexical and hybrid retrieval for policy documents."""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Protocol

from .models import PolicyDocument, RetrievalQuery, RetrievedEvidence
from .bm25 import BM25Index


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """Small dependency-free tokenizer; CJK unigrams plus ASCII words."""

    raw = [item.lower() for item in _TOKEN_PATTERN.findall(text)]
    # Add character bigrams so short Chinese concepts such as “定制商品” and
    # “质量问题” retain more signal than a bag of isolated characters.
    cjk = "".join(item for item in raw if len(item) == 1 and "\u4e00" <= item <= "\u9fff")
    bigrams = [cjk[index : index + 2] for index in range(max(0, len(cjk) - 1))]
    return raw + bigrams


class Retriever(Protocol):
    def search(self, query: RetrievalQuery, top_k: int = 5) -> list[RetrievedEvidence]:
        ...


class PolicyCorpus:
    def __init__(self, documents: list[PolicyDocument]) -> None:
        if not documents:
            raise ValueError("政策语料不能为空。")
        self.documents = list(documents)

    @classmethod
    def from_json(cls, path: Path) -> "PolicyCorpus":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("政策 JSON 顶层必须是数组。")
        documents: list[PolicyDocument] = []
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError("政策文档必须是对象。")
            try:
                effective_from = date.fromisoformat(str(item["effective_from"]))
                effective_to = (
                    date.fromisoformat(str(item["effective_to"]))
                    if item.get("effective_to")
                    else None
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"政策日期无效：{item.get('id')}") from exc
            documents.append(
                PolicyDocument(
                    document_id=str(item["id"]),
                    title=str(item["title"]),
                    text=str(item["text"]),
                    tags=frozenset(str(tag) for tag in item.get("tags", [])),
                    version=str(item.get("version", "unknown")),
                    effective_from=effective_from,
                    effective_to=effective_to,
                    priority=int(item.get("priority", 0)),
                    policy_family=str(item.get("policy_family", "")),
                    exception_of=item.get("exception_of"),
                    status=str(item.get("status", "published")),
                    metadata=dict(item.get("metadata", {})),
                )
            )
        return cls(documents)

    def filter(self, query: RetrievalQuery) -> list[PolicyDocument]:
        result: list[PolicyDocument] = []
        product_type = query.metadata_filter.get("product_type")
        reason = query.metadata_filter.get("reason")
        version = query.metadata_filter.get("version")
        for document in self.documents:
            if not document.is_active(query.as_of):
                continue
            required_scopes = {
                str(scope) for scope in document.metadata.get("required_scopes", [])
            }
            if required_scopes and not required_scopes.issubset(query.permission_scopes):
                continue
            if version and document.version != version:
                continue
            required = set(query.required_tags)
            if product_type:
                required.add(str(product_type))
            if reason == "quality_issue":
                required.add("quality")
            elif reason == "no_reason_return":
                required.add("no_reason_return")
            if required and not required.issubset(document.tags):
                # Exception documents may be explicitly linked to a base policy
                # but still need to match the requested semantic scope.
                continue
            result.append(document)
        return result


class BM25Retriever:
    """Okapi BM25 sparse retrieval with length normalization."""

    def __init__(self, corpus: PolicyCorpus) -> None:
        self.corpus = corpus
        self._tokens = {
            document.document_id: tokenize(document.title + " " + document.text)
            for document in corpus.documents
        }
        self._index = BM25Index(self._tokens)

    def search(self, query: RetrievalQuery, top_k: int = 5) -> list[RetrievedEvidence]:
        candidates = self.corpus.filter(query)
        query_tokens = tokenize(query.text)
        scores = self._index.scores(
            query_tokens,
            [document.document_id for document in candidates],
            normalize=True,
        )
        scored: list[RetrievedEvidence] = []
        for document in candidates:
            score = scores.get(document.document_id, 0.0)
            matched = sorted(set(query_tokens) & set(self._tokens[document.document_id]))
            scored.append(
                RetrievedEvidence(
                    document=document,
                    lexical_score=score,
                    rerank_score=score,
                    matched_terms=matched,
                    retrieval_pass=query.pass_number,
                    query=query.text,
                )
            )
        scored.sort(key=lambda item: item.lexical_score, reverse=True)
        return scored[:top_k]


class VectorRetriever(BM25Retriever):
    """Backward-compatible stage-six name; scoring is BM25, not TF-IDF."""

    def search(self, query: RetrievalQuery, top_k: int = 5) -> list[RetrievedEvidence]:
        results = super().search(query, top_k)
        for item in results:
            item.vector_score = item.lexical_score
            item.lexical_score = 0.0
            item.rerank_score = item.vector_score
        return results


class HybridRetriever(BM25Retriever):
    """BM25 plus exact term-coverage reranking for the offline baseline."""

    def search(self, query: RetrievalQuery, top_k: int = 5) -> list[RetrievedEvidence]:
        candidates = self.corpus.filter(query)
        query_terms = set(tokenize(query.text))
        bm25_results = {
            item.evidence_id: item
            for item in super().search(query, top_k=len(candidates))
        }
        scored: list[RetrievedEvidence] = []
        for document in candidates:
            item = bm25_results[document.document_id]
            doc_terms = set(self._tokens[document.document_id])
            matched = sorted(query_terms & doc_terms)
            coverage = len(matched) / len(query_terms) if query_terms else 0.0
            item.vector_score = coverage
            item.rerank_score = 0.7 * item.lexical_score + 0.3 * coverage
            item.matched_terms = matched
            scored.append(item)
        scored.sort(key=lambda item: item.rerank_score, reverse=True)
        return scored[:top_k]


def load_policy_corpus(path: Path | None = None) -> PolicyCorpus:
    path = path or Path(__file__).parents[1] / "seed" / "data" / "policies_stage6.json"
    return PolicyCorpus.from_json(path)
