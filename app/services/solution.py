"""Full-solution reveal service."""

from __future__ import annotations

from app.core import llm
from app.core.config import get_settings
from app.prompts import guided_discovery as p


async def generate_solution(
    problem_text: str, concepts: list[str], dialogue: str
) -> str:
    settings = get_settings()
    return await llm.chat_text(
        p.SOLUTION_SYSTEM,
        p.solution_user(problem_text, concepts, dialogue),
        temperature=settings.solution_temp,
        max_tokens=settings.solution_max_tokens,
    )
