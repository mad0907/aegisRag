from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_primary_model: str = "qwen2.5:7b-instruct"
    ollama_fallback_model: str = "llama3.2:3b"
    ollama_embed_model: str = "nomic-embed-text"

    # Postgres / PGVector
    postgres_host: str = "localhost"
    postgres_port: int = 5433
    postgres_db: str = "aegisrag"
    postgres_user: str = "aegisrag"
    postgres_password: str = "aegisrag"

    # Phoenix / OTel
    phoenix_collector_endpoint: str = "http://localhost:4317"
    phoenix_ui_url: str = "http://localhost:6006"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8080

    # Agent policy (max_iterations, thresholds, human-review, fallback) is NOT here — it lives
    # solely in config/agent_policy.yaml, loaded by config/control_plane.py. This file is
    # infrastructure config (where things are); agent_policy.yaml is behavior config (how the
    # agents act) — the single AI control plane surface, §21 of the design doc.

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def postgres_dsn_psycopg(self) -> str:
        return (
            f"host={self.postgres_host} port={self.postgres_port} "
            f"dbname={self.postgres_db} user={self.postgres_user} password={self.postgres_password}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
