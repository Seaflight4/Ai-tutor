"""Tests for the source-grounding retrieval service (concept-filter + keyword overlap)."""

from __future__ import annotations

from app.models.schemas import ReferenceChunk
from app.services import retrieval


def _chunk(
    text: str,
    concepts: list[str],
    *,
    source_id: str = "openstax-college-physics",
    title: str = "OpenStax College Physics",
    url: str = "https://openstax.org/books/college-physics",
    chapter: str | None = "Ch. 8",
    heading: str | None = "Collisions",
) -> dict:
    return {
        "source_id": source_id,
        "source_title": title,
        "source_url": url,
        "chapter": chapter,
        "heading": heading,
        "chunk_text": text,
        "concepts": concepts,
    }


def _seed(fake_supabase, *chunks: dict) -> None:
    for c in chunks:
        fake_supabase.add_reference_chunk(**c)


async def test_retrieve_returns_empty_when_no_chunks(fake_supabase):
    out = await retrieval.retrieve_for_concepts(["collisions"], "elastic collision", k=4)
    assert out == []


async def test_retrieve_returns_empty_when_no_concept_match(fake_supabase):
    _seed(fake_supabase, _chunk("Elastic collisions conserve KE.", ["collisions"]))
    out = await retrieval.retrieve_for_concepts(["momentum"], "elastic collision", k=4)
    assert out == []


async def test_retrieve_returns_empty_when_k_zero(fake_supabase):
    _seed(fake_supabase, _chunk("Elastic collisions conserve KE.", ["collisions"]))
    out = await retrieval.retrieve_for_concepts(["collisions"], "elastic collision", k=0)
    assert out == []


async def test_retrieve_ranks_by_keyword_overlap(fake_supabase):
    """The chunk whose heading + body share more query terms ranks first."""
    _seed(
        fake_supabase,
        _chunk(
            "In an elastic collision both momentum and kinetic energy are conserved.",
            ["collisions"],
            heading="Elastic and Inelastic Collisions",
        ),
        _chunk(
            "Momentum is the product of mass and velocity.",
            ["collisions"],
            heading="Linear Momentum",
        ),
    )
    out = await retrieval.retrieve_for_concepts(["collisions"], "elastic collision", k=2)
    assert len(out) == 2
    assert isinstance(out[0], ReferenceChunk)
    # The elastic-collision chunk matches "elastic" and "collision" in both
    # heading and body, so it outranks the momentum chunk.
    assert "Elastic" in (out[0].heading or "")


async def test_retrieve_top_k_limits_results(fake_supabase):
    _seed(
        fake_supabase,
        _chunk("Elastic collision one.", ["collisions"], heading="Elastic One"),
        _chunk("Elastic collision two.", ["collisions"], heading="Elastic Two"),
        _chunk("Elastic collision three.", ["collisions"], heading="Elastic Three"),
    )
    out = await retrieval.retrieve_for_concepts(["collisions"], "elastic collision", k=2)
    assert len(out) == 2


async def test_retrieve_returns_filtered_chunks_when_empty_query(fake_supabase):
    """An empty query still returns up to k concept-filtered chunks so the
    tutor has some grounding context for the problem's concepts."""
    _seed(
        fake_supabase,
        _chunk("Elastic collisions conserve KE.", ["collisions"], heading="Elastic"),
        _chunk("Momentum is mass times velocity.", ["collisions"], heading="Momentum"),
    )
    out = await retrieval.retrieve_for_concepts(["collisions"], "", k=4)
    assert len(out) == 2


async def test_retrieve_finds_elastic_collision_chunk(fake_supabase):
    """End-to-end style: a student asking about elastic collisions should get
    the chunk that actually defines them, ranked first."""
    _seed(
        fake_supabase,
        _chunk(
            "In an elastic collision, both momentum and kinetic energy are conserved. "
            "In an inelastic collision, only momentum is conserved; kinetic energy is "
            "lost to heat and deformation. A perfectly inelastic collision is one where "
            "the objects stick together and share a common final velocity.",
            ["collisions", "momentum"],
            heading="Elastic and Inelastic Collisions",
        ),
        _chunk(
            "Linear momentum is the product of mass and velocity. It is a vector.",
            ["momentum"],
            heading="Linear Momentum",
        ),
    )
    out = await retrieval.retrieve_for_concepts(
        ["collisions", "momentum"], "what is an elastic collision", k=3
    )
    assert out
    assert any("elastic" in (c.heading or "").lower() for c in out)
    # And it's first.
    assert "elastic" in (out[0].heading or "").lower()
