from typing import List, Union
from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global Application Settings configured via environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    PROJECT_NAME: str = "DataPilot"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    HOST: str = "127.0.0.1"
    PORT: int = 8000

    # Database & Storage Settings
    DATABASE_URL: str = "sqlite+aiosqlite:///./datapilot.db"
    STORAGE_PROVIDER: str = "local"
    STORAGE_LOCAL_DIR: str = "data/uploads"

    # Database Pool Settings (PostgreSQL)
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_PRE_PING: bool = True
    DB_POOL_RECYCLE: int = 1800

    # Dataset Ingestion Settings
    UPLOAD_DIR: str = "data/uploads"

    MAX_UPLOAD_SIZE_MB: int = 50
    MAX_DATASET_ROWS: int = 100_000
    MAX_DATASET_COLUMNS: int = 500
    MAX_DATASETS_IN_MEMORY: int = 50
    DATASET_TTL_SECONDS: int = 86400
    ALLOWED_EXTENSIONS: List[str] = [".csv", ".xlsx"]

    # LLM Integration Settings
    LLM_PROVIDER: str = "openai"  # openai, google, anthropic, mock
    LLM_MODEL_NAME: str = "gpt-4o"
    LLM_TEMPERATURE: float = 0.0
    OPENAI_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""

    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)


settings = Settings()

