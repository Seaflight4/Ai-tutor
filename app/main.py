"""FastAPI entrypoint."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

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
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
async def root() -> FileResponse:
    return FileResponse("app/static/index.html")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
