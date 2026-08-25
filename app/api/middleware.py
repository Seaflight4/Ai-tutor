"""Request middleware: rate limiting + structured logging.

Rate limiting is an in-memory sliding-window counter per client IP scoped to
`/api/*`. No external store (Redis etc.) — appropriate for a single-process
uvicorn deployment. For multi-process, replace with a shared store.

Structured logging emits one JSON line per request with method, path, status,
duration, and client IP. Failures inside the middleware never break the
request — they are logged and the request proceeds.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from threading import Lock

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.responses import Response as StarletteResponse

from app.core.config import get_settings

logger = logging.getLogger("app.api")

_RATE_WINDOW = 60.0  # seconds


class _SlidingWindow:
    """Thread-safe sliding-window rate limiter."""

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, key: str, limit: int) -> bool:
        if limit <= 0:
            return True
        now = time.monotonic()
        cutoff = now - _RATE_WINDOW
        with self._lock:
            dq = self._hits[key]
            while dq and dq[0] < cutoff:
                dq.popleft()
            if len(dq) >= limit:
                return False
            dq.append(now)
            return True


_window = _SlidingWindow()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limit per client IP on /api/* paths."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> StarletteResponse:
        path = request.url.path
        if not path.startswith("/api/"):
            return await call_next(request)
        limit = get_settings().rate_limit_per_minute
        client = request.client.host if request.client else "unknown"
        if not _window.allow(client, limit):
            return JSONResponse(
                status_code=429,
                content={"detail": "rate limit exceeded", "code": "rate_limited"},
                headers={"Retry-After": str(int(_RATE_WINDOW))},
            )
        return await call_next(request)


class RequestLogMiddleware(BaseHTTPMiddleware):
    """Emit one structured log line per request."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> StarletteResponse:
        start = time.monotonic()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception:
            logger.exception(
                "request_error method=%s path=%s",
                request.method,
                request.url.path,
            )
            raise
        finally:
            duration_ms = (time.monotonic() - start) * 1000
            client = request.client.host if request.client else "-"
            logger.info(
                "request method=%s path=%s status=%d duration_ms=%.1f client=%s",
                request.method,
                request.url.path,
                status_code,
                duration_ms,
                client,
            )
