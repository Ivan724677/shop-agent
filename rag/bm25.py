"""Dependency-free BM25 scoring shared by baseline and production retrieval."""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable


class BM25Index:
    """Okapi BM25 with document-length normalization and score normalization.

    The index consumes already-tokenized documents so the customer-service
    domain can keep one tokenizer for BM25, query rewriting and grading.
    """

    def __init__(
        self,
        documents: dict[str, list[str]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not documents:
            raise ValueError("BM25 文档不能为空。")
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 参数无效。")
        self.k1 = k1
        self.b = b
        self._counts = {key: Counter(tokens) for key, tokens in documents.items()}
        self._lengths = {key: sum(counts.values()) for key, counts in self._counts.items()}
        self._average_length = sum(self._lengths.values()) / len(self._lengths)
        document_frequency: Counter[str] = Counter()
        for counts in self._counts.values():
            document_frequency.update(counts.keys())
        total = len(self._counts)
        self._idf = {
            term: math.log(1.0 + (total - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def score(self, query_tokens: Iterable[str], document_id: str) -> float:
        counts = self._counts.get(document_id)
        if counts is None:
            return 0.0
        length = self._lengths[document_id]
        score = 0.0
        for term, query_frequency in Counter(query_tokens).items():
            frequency = counts.get(term, 0)
            if frequency <= 0:
                continue
            denominator = frequency + self.k1 * (
                1.0 - self.b + self.b * length / max(self._average_length, 1.0)
            )
            score += self._idf.get(term, 0.0) * (
                frequency * (self.k1 + 1.0) / denominator
            ) * min(query_frequency, 2)
        return score

    def scores(
        self,
        query_tokens: Iterable[str],
        document_ids: Iterable[str] | None = None,
        *,
        normalize: bool = False,
    ) -> dict[str, float]:
        ids = list(document_ids) if document_ids is not None else list(self._counts)
        query = list(query_tokens)
        raw = {document_id: self.score(query, document_id) for document_id in ids}
        if not normalize:
            return raw
        maximum = max(raw.values(), default=0.0)
        return {
            document_id: score / maximum if maximum > 0 else 0.0
            for document_id, score in raw.items()
        }

