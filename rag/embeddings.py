"""Embedding providers used by the production retrieval path.

The project deliberately keeps the provider behind a small protocol.  The
offline hash provider is only a deterministic test double; it is not a
semantic model.  Production runs should use either a local
sentence-transformers model or an OpenAI-compatible embedding endpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from typing import Any, Protocol
from urllib import error, request


class EmbeddingProvider(Protocol):
    model_name: str
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


def normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else list(vector)


class HashEmbeddingProvider:
    """Deterministic offline test double, never presented as semantic embedding."""

    def __init__(self, dimension: int = 256, model_name: str = "offline-hash-test") -> None:
        if dimension <= 0:
            raise ValueError("embedding dimension 必须大于 0。")
        self.dimension = dimension
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        terms = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text.lower())
        for index, term in enumerate(terms):
            digest = hashlib.sha256(f"{index % 7}:{term}".encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign
        return normalize_vector(vector)


class OpenAICompatibleEmbeddingProvider:
    """HTTP embedding provider for OpenAI-compatible `/embeddings` APIs."""

    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: str,
        *,
        dimension: int | None = None,
        timeout: float = 20.0,
        batch_size: int = 32,
    ) -> None:
        if not api_key.strip() or not model_name.strip():
            raise ValueError("embedding api key 和 model 不能为空。")
        if timeout <= 0 or batch_size <= 0:
            raise ValueError("embedding timeout 和 batch_size 必须大于 0。")
        self._api_key = api_key.strip()
        self.model_name = model_name.strip()
        self.base_url = base_url.rstrip("/")
        self.dimension = dimension or 0
        self.timeout = timeout
        self.batch_size = batch_size

    @classmethod
    def from_env(cls) -> "OpenAICompatibleEmbeddingProvider":
        api_key = os.environ.get("EMBEDDING_API_KEY", "")
        model = os.environ.get("EMBEDDING_MODEL", "")
        base_url = os.environ.get("EMBEDDING_BASE_URL", "https://api.openai.com/v1")
        if not api_key or not model:
            raise ValueError("请设置 EMBEDDING_API_KEY、EMBEDDING_MODEL 和可选的 EMBEDDING_BASE_URL。")
        dimension = os.environ.get("EMBEDDING_DIMENSION")
        return cls(
            api_key,
            model,
            base_url,
            dimension=int(dimension) if dimension else None,
            timeout=float(os.environ.get("EMBEDDING_TIMEOUT", "20")),
            batch_size=int(os.environ.get("EMBEDDING_BATCH_SIZE", "32")),
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._request(texts[start : start + self.batch_size]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._request([text])[0]

    def _request(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps({"model": self.model_name, "input": texts}, ensure_ascii=False).encode("utf-8")
        http_request = request.Request(
            f"{self.base_url}/embeddings",
            data=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except (error.HTTPError, error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"embedding 服务调用失败：{exc}") from exc
        try:
            parsed = json.loads(body)
            rows = parsed["data"]
            vectors = [list(map(float, item["embedding"])) for item in sorted(rows, key=lambda row: row["index"])]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("embedding 服务返回了无效响应。") from exc
        if len(vectors) != len(texts) or not vectors:
            raise RuntimeError("embedding 服务返回数量与输入不一致。")
        dimension = len(vectors[0])
        if any(len(vector) != dimension for vector in vectors):
            raise RuntimeError("embedding 向量维度不一致。")
        if self.dimension and self.dimension != dimension:
            raise RuntimeError(f"embedding 维度不匹配：期望 {self.dimension}，实际 {dimension}。")
        self.dimension = dimension
        return [normalize_vector(vector) for vector in vectors]


class SentenceTransformerEmbeddingProvider:
    """Optional local provider; dependency is imported only when selected."""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("未安装 sentence-transformers，无法使用本地 embedding。") from exc
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dimension = int(self._model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [normalize_vector([float(value) for value in row]) for row in self._model.encode(texts, normalize_embeddings=True)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def load_embedding_provider(mode: str = "env") -> EmbeddingProvider:
    if mode == "hash":
        return HashEmbeddingProvider()
    if mode == "local":
        return SentenceTransformerEmbeddingProvider(os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"))
    return OpenAICompatibleEmbeddingProvider.from_env()

