"""Document ingestion, chunking and index construction."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .embeddings import EmbeddingProvider
from .index import IndexedChunk, IndexManifest, PersistentIndex, manifest_for
from .models import PolicyDocument
from .retrieval import PolicyCorpus


@dataclass(frozen=True)
class ChunkingConfig:
    max_chars: int = 500
    overlap_chars: int = 80

    def __post_init__(self) -> None:
        if self.max_chars <= 0 or self.overlap_chars < 0 or self.overlap_chars >= self.max_chars:
            raise ValueError("chunking 参数无效。")


class PolicyChunker:
    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def split(self, document: PolicyDocument) -> list[dict[str, Any]]:
        text = document.text.strip()
        if not text:
            return []
        paragraphs = [part.strip() for part in re.split(r"\n{2,}|(?<=[。！？；])", text) if part.strip()]
        chunks: list[dict[str, Any]] = []
        buffer = ""
        for paragraph in paragraphs:
            if buffer and len(buffer) + len(paragraph) + 1 > self.config.max_chars:
                chunks.append({"text": buffer, "locator": f"chunk-{len(chunks) + 1:03d}"})
                overlap = buffer[-self.config.overlap_chars :]
                buffer = overlap + " " + paragraph
            else:
                buffer = f"{buffer} {paragraph}".strip()
        if buffer:
            chunks.append({"text": buffer, "locator": f"chunk-{len(chunks) + 1:03d}"})
        return chunks


class PolicyIngestionPipeline:
    """Build an immutable index version and optionally publish it."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        index: PersistentIndex,
        chunker: PolicyChunker | None = None,
    ) -> None:
        self.provider = provider
        self.index = index
        self.chunker = chunker or PolicyChunker()

    def build(
        self,
        corpus: PolicyCorpus,
        index_version: str,
        *,
        publish: bool = False,
    ) -> IndexManifest:
        chunk_rows: list[tuple[PolicyDocument, str, str, str | None]] = []
        for document in corpus.documents:
            source_uri = document.metadata.get("source_uri") if isinstance(document.metadata, dict) else None
            for row in self.chunker.split(document):
                chunk_rows.append((document, str(row["locator"]), str(row["text"]), source_uri))
        if not chunk_rows:
            raise ValueError("没有可建立索引的文档 chunk。")
        vectors = self.provider.embed_documents([row[2] for row in chunk_rows])
        if len(vectors) != len(chunk_rows):
            raise ValueError("embedding 返回数量与 chunk 数量不一致。")
        chunks: list[IndexedChunk] = []
        for (document, locator, text, source_uri), vector in zip(chunk_rows, vectors):
            if len(vector) != self.provider.dimension:
                raise ValueError("embedding 维度与 provider 不一致。")
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
            chunks.append(
                IndexedChunk(
                    chunk_id=f"{document.document_id}#{locator}",
                    document_id=document.document_id,
                    title=document.title,
                    text=text,
                    tags=tuple(sorted(document.tags)),
                    version=document.version,
                    effective_from=document.effective_from.isoformat(),
                    effective_to=document.effective_to.isoformat() if document.effective_to else None,
                    priority=document.priority,
                    policy_family=document.family,
                    exception_of=document.exception_of,
                    status=document.status,
                    metadata={**document.metadata, "content_hash": content_hash},
                    source_uri=str(source_uri) if source_uri else None,
                    source_locator=locator,
                    vector=tuple(float(value) for value in vector),
                )
            )
        manifest = manifest_for(index_version, corpus.documents, chunks, self.provider)
        self.index.write_version(manifest, chunks)
        if publish:
            self.index.publish(index_version)
        return manifest

    def build_from_json(self, path: Path, index_version: str, *, publish: bool = False) -> IndexManifest:
        return self.build(PolicyCorpus.from_json(path), index_version, publish=publish)

