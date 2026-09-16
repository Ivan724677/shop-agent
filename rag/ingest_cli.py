"""Build/publish/rollback the versioned RAG index.

Examples:
    python3 -m rag.ingest_cli --embedding-mode hash --publish
    python3 -m rag.ingest_cli --embedding-mode env --version policy-2026-09-08 --publish
    python3 -m rag.ingest_cli --rollback policy-2026-09-01
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .embeddings import load_embedding_provider
from .index import PersistentIndex
from .ingestion import PolicyIngestionPipeline
from .retrieval import load_policy_corpus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Versioned policy ingestion and index lifecycle")
    parser.add_argument("--source", type=Path, default=Path(__file__).parents[1] / "seed" / "data" / "policies_stage6.json")
    parser.add_argument("--index-root", type=Path, default=Path(__file__).parents[1] / "var" / "rag_indexes")
    parser.add_argument("--embedding-mode", choices=["env", "local", "hash"], default="env")
    parser.add_argument("--version", default="policy-index-v1")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--rollback", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    index = PersistentIndex(args.index_root)
    if args.rollback:
        index.rollback(args.rollback)
        print(f"published index={args.rollback}")
        return 0
    provider = load_embedding_provider(args.embedding_mode)
    manifest = PolicyIngestionPipeline(provider, index).build_from_json(
        args.source,
        args.version,
        publish=args.publish,
    )
    print({"manifest": manifest.as_dict(), "published": index.current_version()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

