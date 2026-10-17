"""Settings loaded from environment / .env. No secrets live in code."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "llama-3.3-70b-versatile"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    database_url: str = ""
    cache_max_items: int = 1000
    # --- database pool / resilience ---
    db_pool_min: int = 1
    db_pool_max: int = 4
    db_pool_timeout: float = 1.0
    db_statement_timeout_ms: int = 1500
    db_write_queue_size: int = 1000
    db_breaker_failures: int = 3
    db_breaker_cooldown: float = 30.0
    # --- external providers: bounded concurrency + load shedding ---
    llm_max_concurrency: int = 8
    vision_max_concurrency: int = 4
    external_queue_timeout: float = 2.0
    # --- logging ---
    log_level: str = "INFO"
    log_json: bool = True
    top_k: int = 5
    min_probability: float = 0.03


@lru_cache
def get_settings() -> Settings:
    return Settings()
