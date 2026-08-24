"""FastAPI entrypoint."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings

settings = get_settings()
logging.basicConfig(level=settings.log_level.upper())

app = FastAPI(
    title="AI Tutor — Guided Discovery for Physics",
    version="0.1.0",
    description=(
        "Helps high-school students solve printed physics problems through "
        "guided discovery: asks where they're stuck, diagnoses gap vs "
        "misapplication, gives up to 3 progressive hints, then offers to "
        "reveal the full solution."
    ),
)
app.include_router(router, prefix="/api")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": "ai-tutor", "version": "0.1.0"}
