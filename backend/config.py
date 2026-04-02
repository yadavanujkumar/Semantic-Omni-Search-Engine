"""Application configuration using Pydantic Settings."""
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Semantic Omni Search Engine"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Database
    DATABASE_URL: str = "postgresql://search_user:search_pass@postgres:5432/search_db"

    # File storage
    UPLOAD_DIR: Path = Path("/app/uploads")
    FAISS_INDEX_DIR: Path = Path("/app/faiss_index")
    MAX_UPLOAD_SIZE_MB: int = 500

    # Search
    TOP_K: int = 10
    SIMILARITY_THRESHOLD: float = 0.0

    # Embedding models
    TEXT_MODEL: str = "all-MiniLM-L6-v2"
    CLIP_MODEL: str = "openai/clip-vit-base-patch32"
    WHISPER_MODEL: str = "base"

    # Observability
    PROMETHEUS_ENABLED: bool = True

    # CORS
    CORS_ORIGINS: list[str] = ["*"]

    model_config = {"env_file": ".env", "case_sensitive": True}


settings = Settings()
