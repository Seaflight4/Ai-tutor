"""Ingest the curated physics reference corpus into reference_chunks.

Run once (and re-run whenever the corpus changes):

    python -m scripts.ingest_references [--corpus-dir data/reference] [--reset]

Reads each `*.md` file in the corpus directory. Each file has YAML frontmatter
with source metadata + concept tags, and a body split by `#` headings. Each
section becomes one reference chunk, stored via `supabase.add_reference_chunk`.

Retrieval is concept-filter + keyword-overlap (no embeddings), so chunks
need no vector column. This keeps ingest fully offline: no API key,
no gateway, no dimension to manage. Re-running is cheap and idempotent.

This is a one-time batch job, not part of the request path.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

import yaml

from app.core import config, supabase

_FRONTMATTER = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


def _parse_file(path: Path) -> tuple[dict, list[tuple[str, str]]]:
    """Return (frontmatter, [(heading, body), ...]) for a markdown file."""
    text = path.read_text(encoding="utf-8")
    m = _FRONTMATTER.match(text)
    if not m:
        raise ValueError(f"{path}: missing YAML frontmatter")
    meta = yaml.safe_load(m.group(1)) or {}
    body = m.group(2).strip()
    sections: list[tuple[str, str]] = []
    current_heading = ""
    current_body: list[str] = []
    for line in body.splitlines():
        if line.startswith("# ") and not line.startswith("# " * 2):
            if current_heading or current_body:
                sections.append((current_heading, "\n".join(current_body).strip()))
            current_heading = line[2:].strip()
            current_body = []
        else:
            current_body.append(line)
    if current_heading or current_body:
        sections.append((current_heading, "\n".join(current_body).strip()))
    return meta, sections


async def ingest(corpus_dir: Path, reset: bool) -> int:
    if reset:
        supabase.reset_reference_chunks()
        print("cleared existing reference_chunks", file=sys.stderr)

    files = sorted(corpus_dir.glob("*.md"))
    if not files:
        print(f"no .md files in {corpus_dir}", file=sys.stderr)
        return 1

    total = 0
    for path in files:
        meta, sections = _parse_file(path)
        sections = [(h, b) for h, b in sections if (h or b)]
        for heading, body in sections:
            chunk_text = f"{heading}\n\n{body}".strip()
            supabase.add_reference_chunk(
                source_id=meta["source_id"],
                source_title=meta["source_title"],
                source_url=meta["source_url"],
                chapter=meta.get("chapter"),
                heading=heading or None,
                chunk_text=chunk_text,
                concepts=list(meta.get("concepts", [])),
            )
            total += 1
        print(f"  {path.name}: {len(sections)} chunk(s)", file=sys.stderr)

    print(f"ingested {total} chunk(s) from {len(files)} file(s)", file=sys.stderr)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the physics reference corpus.")
    parser.add_argument(
        "--corpus-dir",
        default=None,
        help=f"directory of .md reference files (default: {config.get_settings().reference_corpus_dir})",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="clear existing chunks before ingesting",
    )
    args = parser.parse_args()
    corpus_dir = Path(args.corpus_dir or config.get_settings().reference_corpus_dir)
    sys.exit(asyncio.run(ingest(corpus_dir, args.reset)))


if __name__ == "__main__":
    main()
