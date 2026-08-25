"""Source-grounding retrieval over the curated physics reference corpus.

Single-stage retrieval per tutor turn:
  1. Concept pre-filter: narrow reference_chunks to those tagged with at least
     one of the problem's concepts (cheap SQL/SQLite filter).
  2. Keyword-overlap scoring: rank the filtered chunks by how many query terms
     appear in the chunk's heading + chunk_text. Deterministic and needs no
     external service.

This is intentionally simpler than vector retrieval. The seed corpus is small
(~20 chunks); concept filtering already narrows to a handful of candidates,
and the tutor model can itself ignore irrelevant ones from the few chunks in
its context. Keyword overlap just makes the top-k meaningful when the filter
returns more than k.

If the corpus ever grows into the hundreds of chunks, revisit this with
proper embeddings (e.g. text-embedding-3-small + pgvector cosine search).
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from app.core import config, supabase
from app.models.schemas import ReferenceChunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _score_chunk(chunk_text: str, heading: str | None, query_terms: set[str]) -> int:
    """Number of distinct query terms appearing in the chunk's text + heading.

    Heading matches count double — a heading match is a strong relevance
    signal at this scale. Returns 0 for an empty query (caller handles that).
    """
    if not query_terms:
        return 0
    body_tokens = _tokens(chunk_text)
    heading_tokens = _tokens(heading or "")
    score = 0
    for term in query_terms:
        if term in body_tokens:
            score += 1
        if term in heading_tokens:
            score += 1
    return score


async def retrieve_for_concepts(
    concepts: list[str],
    query: str,
    k: int | None = None,
) -> list[ReferenceChunk]:
    """Top-k reference chunks relevant to the concepts, ranked by keyword overlap.

    Returns an empty list if no chunks match the concept filter. Degrades
    gracefully — the caller proceeds without sources.
    """
    if k is None:
        k = config.get_settings().reference_top_k
    if k <= 0:
        return []

    rows = await asyncio.to_thread(supabase.list_reference_chunks_by_concepts, concepts)
    if not rows:
        return []

    query_terms = _tokens(query)
    scored: list[tuple[int, int, ReferenceChunk]] = []
    for idx, row in enumerate(rows):
        score = _score_chunk(row["chunk_text"], row.get("heading"), query_terms)
        scored.append((score, -idx, _row_to_chunk(row)))

    # Highest score first; ties broken by original order (via -idx).
    scored.sort(key=lambda s: s[:2], reverse=True)
    top = [c for _, _, c in scored[:k]]
    # If nothing scored (e.g. empty query), still return up to k filtered chunks
    # so the tutor has *some* grounding context for the problem's concepts.
    if not query_terms and scored:
        return [c for _, _, c in scored[:k]]
    return top


def _row_to_chunk(row: dict[str, Any]) -> ReferenceChunk:
    return ReferenceChunk(
        id=row.get("id"),
        source_id=row["source_id"],
        source_title=row["source_title"],
        source_url=row["source_url"],
        chapter=row.get("chapter"),
        heading=row.get("heading"),
        chunk_text=row["chunk_text"],
        concepts=list(row.get("concepts") or []),
    )
