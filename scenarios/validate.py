"""Validate and summarize the local scenario corpus."""

from __future__ import annotations

from pathlib import Path

from .loader import load_corpus, summarize


def main() -> None:
    root = Path(__file__).parent
    scenarios = load_corpus(root)
    print(f"validated {len(scenarios)} scenarios")
    for tag, count in sorted(summarize(scenarios).items()):
        print(f"{tag}: {count}")


if __name__ == "__main__":
    main()
