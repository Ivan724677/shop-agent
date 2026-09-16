"""Persistent, versioned local index used by the production RAG path.

This is a small reference implementation of the lifecycle contract normally
backed by a vector database and object storage: build into an immutable
version, validate it, publish an alias, and roll back by moving the alias.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .embeddings import EmbeddingProvider
from .models import PolicyDocument


@dataclass(frozen=True)
class IndexManifest:
    index_version: str
    embedding_model: str
    embedding_dimension: int
    document_count: int
    chunk_count: int
    corpus_hash: str
    created_at: str
    status: str = "built"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IndexedChunk:
    chunk_id: str
    document_id: str
    title: str
    text: str
    tags: tuple[str, ...]
    version: str
    effective_from: str
    effective_to: str | None
    priority: int
    policy_family: str
    exception_of: str | None
    status: str
    metadata: dict[str, Any]
    source_uri: str | None
    source_locator: str | None
    vector: tuple[float, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class IndexLifecycleError(RuntimeError):
    pass


class PersistentIndex:
    """Immutable JSON index with atomic publish and rollback operations."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write_version(self, manifest: IndexManifest, chunks: list[IndexedChunk]) -> Path:
        if not manifest.index_version or not chunks:
            raise IndexLifecycleError("索引版本和 chunks 不能为空。")
        if Path(manifest.index_version).name != manifest.index_version:
            raise IndexLifecycleError("索引版本只能是单级目录名。")
        if manifest.chunk_count != len(chunks):
            raise IndexLifecycleError("manifest 的 chunk_count 与待写入数据不一致。")
        if manifest.document_count != len({chunk.document_id for chunk in chunks}):
            raise IndexLifecycleError("manifest 的 document_count 与待写入数据不一致。")
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise IndexLifecycleError("索引中存在重复 chunk_id。")
        if any(len(chunk.vector) != manifest.embedding_dimension for chunk in chunks):
            raise IndexLifecycleError("索引 chunk 的向量维度不一致。")
        target = self.root / manifest.index_version
        if target.exists():
            raise IndexLifecycleError(f"索引版本已存在，不允许覆盖：{manifest.index_version}")
        temp = Path(tempfile.mkdtemp(prefix=f".{manifest.index_version}.", dir=self.root))
        try:
            (temp / "manifest.json").write_text(
                json.dumps(manifest.as_dict(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            with (temp / "chunks.jsonl").open("w", encoding="utf-8") as handle:
                for chunk in chunks:
                    handle.write(json.dumps(chunk.as_dict(), ensure_ascii=False) + "\n")
            os.replace(temp, target)
        except Exception:
            for path in temp.glob("*"):
                path.unlink(missing_ok=True)
            temp.rmdir()
            raise
        return target

    def publish(self, index_version: str) -> None:
        version_dir = self.root / index_version
        if Path(index_version).name != index_version or not (version_dir / "manifest.json").exists():
            raise IndexLifecycleError(f"索引版本不存在或不完整：{index_version}")
        # A version directory is immutable after build. CURRENT is the only
        # mutable release alias, which makes rollback an atomic pointer move.
        self._atomic_text(self.root / "CURRENT", index_version + "\n")

    def rollback(self, index_version: str) -> None:
        self.publish(index_version)

    def current_version(self) -> str | None:
        path = self.root / "CURRENT"
        return path.read_text(encoding="utf-8").strip() if path.exists() else None

    def load(self, index_version: str | None = None) -> tuple[IndexManifest, list[IndexedChunk]]:
        version = index_version or self.current_version()
        if not version:
            raise IndexLifecycleError("当前没有已发布索引。")
        directory = self.root / version
        try:
            manifest = IndexManifest(**json.loads((directory / "manifest.json").read_text(encoding="utf-8")))
            chunks = [
                IndexedChunk(**json.loads(line))
                for line in (directory / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise IndexLifecycleError(f"索引加载失败：{version}") from exc
        if manifest.chunk_count != len(chunks):
            raise IndexLifecycleError("索引 manifest 的 chunk_count 与实际数据不一致。")
        if manifest.document_count != len({chunk.document_id for chunk in chunks}):
            raise IndexLifecycleError("索引 manifest 的 document_count 与实际数据不一致。")
        if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
            raise IndexLifecycleError("索引中存在重复 chunk_id。")
        if any(len(chunk.vector) != manifest.embedding_dimension for chunk in chunks):
            raise IndexLifecycleError("索引中存在错误维度的向量。")
        return manifest, chunks

    def list_versions(self) -> list[str]:
        return sorted(
            path.name for path in self.root.iterdir()
            if path.is_dir() and (path / "manifest.json").exists()
        )

    @staticmethod
    def _atomic_text(path: Path, text: str) -> None:
        temp = path.with_name(f".{path.name}.tmp")
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, path)


def corpus_hash(documents: list[PolicyDocument]) -> str:
    payload = [document.as_dict() for document in documents]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def manifest_for(
    index_version: str,
    documents: list[PolicyDocument],
    chunks: list[IndexedChunk],
    provider: EmbeddingProvider,
) -> IndexManifest:
    if not getattr(provider, "dimension", 0):
        raise IndexLifecycleError("embedding provider 尚未确定向量维度。")
    return IndexManifest(
        index_version=index_version,
        embedding_model=provider.model_name,
        embedding_dimension=provider.dimension,
        document_count=len(documents),
        chunk_count=len(chunks),
        corpus_hash=corpus_hash(documents),
        created_at=datetime.now().isoformat(timespec="seconds"),
    )
