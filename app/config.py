"""Settings loaded from environment / .env. No secrets live in code."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "llama-3.3-70b-versatile"
    database_url: str = ""
    cache_max_items: int = 1000
    top_k: int = 5
    min_probability: float = 0.03


@lru_cache
def get_settings() -> Settings:
    return Settings()
