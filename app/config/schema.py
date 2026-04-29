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
    """Vector store — config/chroma.yml."""

    collection_name: str
    persist_directory: str
    k: int
    chunk_size: int
    chunk_overlap: int
    separators: list[str]
    allow_knowledge_file_type: list[str]
    data_path: str
    md5_hex_store: str


class RagSettings(BaseModel):
    """RAG data sources — config/rag.yml."""

    external_data_path: str


class PromptSettings(BaseModel):
    """Prompt file paths — config/prompts.yml."""

    prompts_dir: str
    main_prompt_path: str
    rag_summarize_prompt_path: str
    report_prompt_path: str


class AppSettings(BaseModel):
    """Top-level application settings — composed of domain groups.

    Usage:
        settings.server.project_name
        settings.model.chat_model_name
        settings.chroma.collection_name
        settings.rag.external_data_path
        settings.prompts.main_prompt_path
    """

    server: ServerSettings
    model: ModelSettings
    chroma: ChromaSettings
    rag: RagSettings
    prompts: PromptSettings
    project_root: Path
