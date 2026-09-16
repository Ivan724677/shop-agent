"""Dense + sparse hybrid retrieval over a versioned persistent index."""

from __future__ import annotations

import math
from collections import Counter
from datetime import date
from typing import Any

from .embeddings import EmbeddingProvider
from .index import IndexedChunk, PersistentIndex, corpus_hash
from .models import PolicyDocument, RetrievalQuery, RetrievedEvidence
from .retrieval import PolicyCorpus, tokenize


def cosine(left: list[float] | tuple[float, ...], right: list[float] | tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise ValueError("向量维度不一致。")
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(sum(value * value for value in right))
    return sum(a * b for a, b in zip(left, right)) / denominator if denominator else 0.0


class DenseHybridRetriever:
    """Production retrieval adapter; the index is immutable for one instance."""

    def __init__(
        self,
        index: PersistentIndex,
        provider: EmbeddingProvider,
        corpus: PolicyCorpus,
        *,
        dense_weight: float = 0.65,
        sparse_weight: float = 0.35,
    ) -> None:
        if dense_weight < 0 or sparse_weight < 0 or dense_weight + sparse_weight <= 0:
            raise ValueError("dense/sparse 权重无效。")
        self.index = index
        self.provider = provider
        self.corpus = corpus
        self.dense_weight = dense_weight / (dense_weight + sparse_weight)
        self.sparse_weight = sparse_weight / (dense_weight + sparse_weight)
        self.manifest, self.chunks = index.load()
        if self.manifest.embedding_model != provider.model_name or self.manifest.embedding_dimension != provider.dimension:
            raise ValueError("索引与 embedding provider 的 model/dimension 不匹配。")
        if self.manifest.corpus_hash != corpus_hash(corpus.documents):
            raise ValueError("索引语料 hash 与当前政策语料不匹配，请重新构建或切换索引。")
        self._documents = {document.document_id: document for document in corpus.documents}
        self._tokens = {
            chunk.chunk_id: tokenize(chunk.title + " " + chunk.text)
            for chunk in self.chunks
        }
        document_frequency = Counter()
        for tokens in self._tokens.values():
            document_frequency.update(set(tokens))
        self._idf = {
            token: math.log((1 + len(self.chunks)) / (1 + count)) + 1
            for token, count in document_frequency.items()
        }

    def search(self, query: RetrievalQuery, top_k: int = 5) -> list[RetrievedEvidence]:
        if top_k <= 0:
            return []
        candidates = [chunk for chunk in self.chunks if self._applicable(chunk, query)]
        if not candidates:
            return []
        query_vector = self.provider.embed_query(query.text)
        if len(query_vector) != self.manifest.embedding_dimension:
            raise ValueError("查询 embedding 维度与索引不一致。")
        query_tokens = Counter(tokenize(query.text))
        query_vector_sparse = self._tfidf(query_tokens)
        scored: list[RetrievedEvidence] = []
        for chunk in candidates:
            dense_score = cosine(query_vector, chunk.vector)
            chunk_tokens = Counter(self._tokens[chunk.chunk_id])
            chunk_terms = set(chunk_tokens)
            matched = sorted(set(query_tokens) & chunk_terms)
            sparse_score = cosine(query_vector_sparse, self._tfidf(chunk_tokens))
            combined = self.dense_weight * dense_score + self.sparse_weight * sparse_score
            document = self._documents.get(chunk.document_id) or _document_from_chunk(chunk)
            scored.append(
                RetrievedEvidence(
                    document=PolicyDocument(
                        document_id=document.document_id,
                        title=chunk.title,
                        text=chunk.text,
                        tags=document.tags,
                        version=document.version,
                        effective_from=document.effective_from,
                        effective_to=document.effective_to,
                        priority=document.priority,
                        policy_family=document.family,
                        exception_of=document.exception_of,
                        status=document.status,
                        metadata=document.metadata,
                    ),
                    vector_score=dense_score,
                    lexical_score=sparse_score,
                    rerank_score=combined,
                    matched_terms=matched,
                    retrieval_pass=query.pass_number,
                    query=query.text,
                    chunk_id=chunk.chunk_id,
                    source_uri=chunk.source_uri,
                    source_locator=chunk.source_locator,
                )
            )
        scored.sort(key=lambda item: (item.rerank_score, item.vector_score, item.lexical_score), reverse=True)
        return scored[:top_k]

    def _tfidf(self, counts: Counter[str]) -> list[float]:
        """Return a deterministic sparse-space vector for the current index.

        The representation is dense only at this small reference boundary; the
        scoring is TF-IDF cosine over the indexed vocabulary. A production
        deployment can replace this method with a BM25/SPLADE service without
        changing the Retriever protocol or evidence contract.
        """
        vocabulary = sorted(self._idf)
        vector = [counts[token] * self._idf[token] for token in vocabulary]
        return vector

    @staticmethod
    def _applicable(chunk: IndexedChunk, query: RetrievalQuery) -> bool:
        if chunk.status != "published":
            return False
        start = date.fromisoformat(chunk.effective_from)
        end = date.fromisoformat(chunk.effective_to) if chunk.effective_to else None
        if start > query.as_of or (end and end < query.as_of):
            return False
        if query.metadata_filter.get("version") and query.metadata_filter["version"] != chunk.version:
            return False
        required = set(query.required_tags)
        product_type = query.metadata_filter.get("product_type")
        reason = query.metadata_filter.get("reason")
        if product_type:
            required.add(str(product_type))
        if reason == "quality_issue":
            required.add("quality")
        elif reason == "no_reason_return":
            required.add("no_reason_return")
        return required.issubset(set(chunk.tags))


def _document_from_chunk(chunk: IndexedChunk) -> PolicyDocument:
    return PolicyDocument(
        document_id=chunk.document_id,
        title=chunk.title,
        text=chunk.text,
        tags=frozenset(chunk.tags),
        version=chunk.version,
        effective_from=date.fromisoformat(chunk.effective_from),
        effective_to=date.fromisoformat(chunk.effective_to) if chunk.effective_to else None,
        priority=chunk.priority,
        policy_family=chunk.policy_family,
        exception_of=chunk.exception_of,
        status=chunk.status,
        metadata=dict(chunk.metadata),
    )
