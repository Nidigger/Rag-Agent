from pathlib import Path

from pydantic import computed_field
from pydantic_settings import BaseSettings

# app/core/config.py -> app/core -> app -> project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    PROJECT_NAME: str = "Rag-Agent Service"
    VERSION: str = "0.1.0"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"

    # Model
    DASHSCOPE_API_KEY: str
    CHAT_MODEL_NAME: str = "qwen3-max"
    EMBEDDING_MODEL_NAME: str = "text-embedding-v4"

    # Paths (relative to PROJECT_ROOT)
    CONFIG_DIR: str = "config"
    PROMPTS_DIR: str = "prompts"
    DATA_DIR: str = "data"
    CHROMA_PERSIST_DIR: str = "chroma_db"
    MD5_HEX_STORE: str = "md5.text"
    EXTERNAL_DATA_PATH: str = "data/external/records.csv"

    # Chroma
    CHROMA_COLLECTION_NAME: str = "agent"
    CHROMA_K: int = 3
    CHUNK_SIZE: int = 200
    CHUNK_OVERLAP: int = 20
    CHROMA_SEPARATORS: list[str] = [
        "\n\n", "\n", ".", "!", "?", "\u3002", "\uff01", "\uff1f", " ", "",
    ]
    ALLOWED_FILE_TYPES: list[str] = ["txt", "pdf"]

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # CORS
    CORS_ALLOW_ORIGINS: list[str] = ["http://localhost:5173"]

    # Logging
    LOG_LEVEL: str = "INFO"

    @computed_field
    @property
    def PROJECT_ROOT(self) -> Path:
        return _PROJECT_ROOT

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


settings = Settings()
