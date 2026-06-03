"""Runtime configuration. Loaded once from env, exposed via `get_settings()`."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://worldscan:worldscan@localhost:5432/worldscan"
    # Default to a user-writable path so tests + non-container dev work.
    # In docker, DATA_DIR=/data is set explicitly.
    data_dir: Path = Path("./data")
    inngest_dev_server_url: str = "http://localhost:8288"
    inngest_app_id: str = "worldscan"

    # CORS — frontend runs on a different port in dev
    cors_origins: list[str] = ["http://localhost:5173"]

    # Sentry — empty disables it
    sentry_dsn: str = ""

    # Task codegen (PR-2) — empty disables Generate until set
    anthropic_api_key: str = ""
    anthropic_task_model: str = "claude-sonnet-4-20250514"

    # Build / RL defaults (exposed in the frontend's Advanced panel)
    default_target_diagonal_m: float = 6.0
    default_max_hulls: int = 64
    default_success_radius: float = 0.4
    default_max_steps: int = 500
    default_lidar_n: int = 16


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
