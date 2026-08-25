"""Shared-secret API authentication.

When `Settings.api_secret` is non-empty, every `/api/*` request must carry the
header `X-API-Key: <api_secret>`. Requests without a matching key receive 401.
When `api_secret` is empty (dev/test default), auth is skipped entirely.

Health, root, and static paths are never authenticated.
"""

from __future__ import annotations

import hmac

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader

from app.core.config import get_settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(
    request: Request,
    provided: str | None = Depends(_api_key_header),
) -> None:
    """FastAPI dependency: enforce the shared-secret on /api/* routes."""
    secret = get_settings().api_secret
    if not secret:
        return  # auth disabled
    if provided is None or not hmac.compare_digest(provided, secret):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
