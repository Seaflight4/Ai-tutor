"""Application configuration loaded from environment via pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Server
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"

    # skainet (OpenAI-compatible) gateway
    skainet_base_url: str = "https://chat.model.tngtech.com/v1"
    skainet_api_key: str = Field(default="", repr=False)

    # Model IDs served by skainet
    model_ocr: str = "tngtech/olmocr-7B-faithful"
    model_chat: str = "zai-org/GLM-5.2"
    model_chat_fast: str = "deepseek-ai/DeepSeek-V4-Flash"

    # Dialogue loop policy
    max_hint_loops: int = 3

    # Supabase
    supabase_url: str = ""
    supabase_key: str = Field(default="", repr=False)
    supabase_bucket: str = "problem-images"


@lru_cache
def get_settings() -> Settings:
    return Settings()
