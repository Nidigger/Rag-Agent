"""Pydantic models for each configuration domain.

Each model corresponds to one YAML file and defines typed, validated
fields. Required fields have no default — a missing YAML entry will
cause startup to fail with a clear validation error.
"""

from pathlib import Path

from pydantic import BaseModel, SecretStr


class ServerSettings(BaseModel):
    """Server, API, CORS, and logging — config/server.yml."""

    project_name: str
    version: str
    debug: bool
    api_v1_prefix: str
    host: str
    port: int
    cors_allow_origins: list[str]
    log_level: str


class ModelSettings(BaseModel):
    """LLM / embedding model — config/agent.yml + .env for secrets."""

    chat_model_name: str
    embedding_model_name: str
    openai_compatible_base_url: str
    dashscope_api_key: SecretStr


class ChromaSettings(BaseModel):
    """Vector store — config/chroma.yml (legacy / fallback)."""

    collection_name: str
    persist_directory: str
    k: int
    chunk_size: int
    chunk_overlap: int
    separators: list[str]
    allow_knowledge_file_type: list[str]
    data_path: str
    md5_hex_store: str


class QdrantSettings(BaseModel):
    """Qdrant connection — config/vector.yml qdrant section."""

    url: str
    collection_name: str
    distance: str = "Cosine"
    vector_size: int = 1536
    timeout_seconds: int = 10


class IngestSettings(BaseModel):
    """Document ingest configuration — config/vector.yml ingest section."""

    chunk_size: int = 600
    chunk_overlap: int = 80
    batch_size: int = 64


class VectorSettings(BaseModel):
    """Vector store configuration — config/vector.yml.

    Supports switching between qdrant and chroma providers.
    """

    provider: str = "qdrant"
    qdrant: QdrantSettings
    ingest: IngestSettings
    chroma: ChromaSettings | None = None


class RagSettings(BaseModel):
    """RAG data sources — config/rag.yml."""

    external_data_path: str


class PromptSettings(BaseModel):
    """Prompt file paths — config/prompts.yml."""

    prompts_dir: str
    main_prompt_path: str
    rag_summarize_prompt_path: str
    report_prompt_path: str


class SecuritySettings(BaseModel):
    """Security-related settings.

    Only from environment variables or .env — never from YAML.
    Future expansion: JWT issuer, JWT secret, allowed internal clients.
    """

    internal_token: SecretStr | None = None


class AppSettings(BaseModel):
    """Top-level application settings — composed of domain groups.

    Usage:
        settings.server.project_name
        settings.model.chat_model_name
        settings.vector.provider
        settings.vector.qdrant.url
        settings.chroma.collection_name
        settings.rag.external_data_path
        settings.prompts.main_prompt_path
        settings.security.internal_token
    """

    server: ServerSettings
    model: ModelSettings
    vector: VectorSettings
    chroma: ChromaSettings
    rag: RagSettings
    prompts: PromptSettings
    security: SecuritySettings
    project_root: Path
