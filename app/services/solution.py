"""Full-solution reveal service."""

from __future__ import annotations

from app.core import llm
from app.prompts import guided_discovery as p


async def generate_solution(
    problem_text: str, concepts: list[str], dialogue: str
) -> str:
    return await llm.chat_text(
        p.SOLUTION_SYSTEM,
        p.solution_user(problem_text, concepts, dialogue),
        temperature=0.3,
        max_tokens=1500,
    )
