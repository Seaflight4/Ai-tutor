"""Application configuration loaded from environment via pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root: app/core/config.py -> app/core -> app -> <root>
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Server
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    # Shared-secret API auth. When non-empty, every /api/* request must carry
    # `X-API-Key: <api_secret>`. Leave empty in dev/test to disable auth.
    api_secret: str = Field(default="", repr=False)

    # Rate limiting (in-memory sliding window per client IP). Only the /api/*
    # prefix is limited; health/static/root are exempt. Set to 0 to disable.
    rate_limit_per_minute: int = 60

    # Upload limits
    max_image_bytes: int = 10 * 1024 * 1024  # 10 MiB
    max_reply_chars: int = 4000

    # skainet (OpenAI-compatible) gateway
    skainet_base_url: str = "https://chat.model.tngtech.com/v1"
    skainet_api_key: str = Field(default="", repr=False)

    # Model IDs served by skainet
    model_ocr: str = "tngtech/olmocr-7B-faithful"
    model_chat: str = "zai-org/GLM-5.2"

    # Dialogue loop policy: after this many hint turns, the tutor sets
    # `offer_reveal=True` on its reply (the student may still decline and keep
    # working). The loop itself never force-reveals — only an explicit student
    # request or the LLM's `wants_solution` flag triggers a reveal.
    max_hint_loops: int = 3

    # LLM call parameters per role. Defaults match the values previously
    # hardcoded in each service file; override via env for tuning.
    ocr_max_tokens: int = 2000
    ocr_parse_temp: float = 0.0
    ocr_parse_max_tokens: int = 1000
    opening_temp: float = 0.6
    opening_max_tokens: int = 300
    tutor_temp: float = 0.3
    tutor_max_tokens: int = 1100
    solution_temp: float = 0.3
    solution_max_tokens: int = 1500
    profile_temp: float = 0.0
    profile_max_tokens: int = 200

    # Source grounding (RAG over a curated physics reference corpus)
    reference_top_k: int = 4
    reference_corpus_dir: str = str(_PROJECT_ROOT / "data" / "reference")

    # Supabase
    supabase_url: str = ""
    supabase_key: str = Field(default="", repr=False)
    supabase_bucket: str = "problem-images"


@lru_cache
def get_settings() -> Settings:
    return Settings()
