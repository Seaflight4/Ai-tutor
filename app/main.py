"""FastAPI entrypoint."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.requests import Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.adapters.llm_skainet import SkainetLLM, get_llm_client
from app.adapters.supabase_repo import SupabaseBackend, get_backend
from app.api.middleware import RateLimitMiddleware, RequestLogMiddleware
from app.api.routes import router
from app.core.config import get_settings

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_STATIC_DIR = _PROJECT_ROOT / "app" / "static"

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

# Middleware (order: outermost first — the last added wraps everything).
app.add_middleware(RequestLogMiddleware)
app.add_middleware(RateLimitMiddleware)

app.include_router(router, prefix="/api")
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# Expose adapter constructors as dependencies so routes/services can later
# request the ports (`LLMClient`, `Backend`) via `Depends` instead of importing
# the legacy modules directly. For PR1 the existing services still call
# `app.core.llm` / `app.core.supabase`; these dependencies are wired so the
# migration in later PRs is a local change to each service.
app.dependency_overrides[get_llm_client] = lambda: SkainetLLM(settings)
app.dependency_overrides[get_backend] = lambda: SupabaseBackend()


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(str(_STATIC_DIR / "index.html"))


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Error-response shape
# ---------------------------------------------------------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "code": _code_for(exc.status_code)},
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    logging.exception("unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "internal server error", "code": "internal_error"},
    )


def _code_for(status_code: int) -> str:
    return {
        400: "bad_request",
        401: "unauthorized",
        404: "not_found",
        409: "conflict",
        413: "payload_too_large",
        415: "unsupported_media_type",
        422: "validation_error",
        429: "rate_limited",
    }.get(status_code, "error")
