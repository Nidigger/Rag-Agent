"""Configuration loader — reads YAML files and merges with env overrides.

Loading priority (highest wins):
  OS environment variable > .env file > config/*.yml

Secrets (API keys) MUST come from .env or OS env, never from YAML.
"""

import json
import logging
import os
from pathlib import Path

import yaml
from dotenv import dotenv_values

from app.config.schema import (
    AppSettings,
    ChromaSettings,
    IngestSettings,
    ModelSettings,
    PromptSettings,
    QdrantSettings,
    RagSettings,
    SecuritySettings,
    ServerSettings,
    VectorSettings,
)

logger = logging.getLogger("rag-agent.config_loader")

# Resolve project root: loader.py -> config -> app -> project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"
_ENV_FILE = _PROJECT_ROOT / ".env"


def _load_yaml(filename: str) -> dict:
    """Load a single YAML file from the config/ directory."""
    path = _CONFIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Required config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    logger.debug("[config_loader] Loaded %s: %d keys", filename, len(data))
    return data


def _load_dotenv() -> dict[str, str]:
    """Load .env file values as a plain dict (does not modify os.environ)."""
    if not _ENV_FILE.exists():
        logger.debug("[config_loader] No .env file found at %s", _ENV_FILE)
        return {}
    return dict(dotenv_values(_ENV_FILE))


def _resolve_env(key: str, dotenv: dict, yaml_val: str | None = None) -> str | None:
    """Return the highest-priority raw value for a config key.

    Priority: os.environ > .env file > YAML value.
    """
    if key in os.environ:
        return os.environ[key]
    if key in dotenv:
        return dotenv[key]
    return yaml_val


def _resolve_str(key: str, dotenv: dict, yaml_val: str | None = None) -> str:
    """Resolve a required string config value with env override."""
    value = _resolve_env(key, dotenv, yaml_val)
    if value is None:
        raise ValueError(f"{key} is required but not found in env, .env, or config YAML")
    return str(value)


def _resolve_int(key: str, dotenv: dict, yaml_val: int | str | None = None) -> int:
    """Resolve a required integer config value with env override."""
    value = _resolve_env(key, dotenv, None if yaml_val is None else str(yaml_val))
    if value is None:
        raise ValueError(f"{key} is required but not found in env, .env, or config YAML")
    return int(value)


def _resolve_bool(key: str, dotenv: dict, yaml_val: bool | str | None = None) -> bool:
    """Resolve a required boolean config value with env override."""
    value = _resolve_env(key, dotenv, None if yaml_val is None else str(yaml_val))
    if value is None:
        raise ValueError(f"{key} is required but not found in env, .env, or config YAML")
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_str_list(key: str, dotenv: dict, yaml_val: list[str] | None = None) -> list[str]:
    """Resolve a list of strings with env override.

    Env value supports JSON array or comma-separated fallback.
    """
    value = _resolve_env(key, dotenv)
    if value is None:
        return yaml_val or []

    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass

    return [item.strip() for item in str(value).split(",") if item.strip()]


def load_settings() -> AppSettings:
    """Load and assemble all settings from YAML files + env overrides.

    Priority (highest wins): OS env > .env > config/*.yml

    Raises FileNotFoundError if a required YAML file is missing.
    Raises ValueError if required fields are absent.
    """
    logger.info("[config_loader] Loading configuration from %s", _CONFIG_DIR)

    # Load all YAML files
    server_yml = _load_yaml("server.yml")
    agent_yml = _load_yaml("agent.yml")
    chroma_yml = _load_yaml("chroma.yml")
    rag_yml = _load_yaml("rag.yml")
    prompts_yml = _load_yaml("prompts.yml")

    # Load .env for secrets and overrides
    dotenv = _load_dotenv()

    # Assemble ServerSettings — every field supports env override
    server = ServerSettings(
        project_name=_resolve_str("PROJECT_NAME", dotenv, server_yml.get("project_name")),
        version=_resolve_str("VERSION", dotenv, server_yml.get("version")),
        debug=_resolve_bool("DEBUG", dotenv, server_yml.get("debug")),
        api_v1_prefix=_resolve_str("API_V1_PREFIX", dotenv, server_yml.get("api_v1_prefix")),
        host=_resolve_str("HOST", dotenv, server_yml.get("host")),
        port=_resolve_int("PORT", dotenv, server_yml.get("port")),
        cors_allow_origins=_resolve_str_list(
            "CORS_ALLOW_ORIGINS",
            dotenv,
            server_yml.get("cors_allow_origins"),
        ),
        log_level=_resolve_str("LOG_LEVEL", dotenv, server_yml.get("log_level")),
    )

    # Assemble ModelSettings (API key MUST come from env, never YAML)
    api_key = _resolve_env("DASHSCOPE_API_KEY", dotenv)
    if not api_key:
        raise ValueError(
            "DASHSCOPE_API_KEY is required but not found in "
            "environment variables or .env file"
        )

    model = ModelSettings(
        chat_model_name=_resolve_str("CHAT_MODEL_NAME", dotenv, agent_yml.get("chat_model_name")),
        embedding_model_name=_resolve_str(
            "EMBEDDING_MODEL_NAME", dotenv, agent_yml.get("embedding_model_name")
        ),
        openai_compatible_base_url=_resolve_str(
            "OPENAI_COMPATIBLE_BASE_URL", dotenv, agent_yml.get("openai_compatible_base_url")
        ),
        dashscope_api_key=api_key,
    )

    # Assemble ChromaSettings — every field supports env override
    chroma = ChromaSettings(
        collection_name=_resolve_str(
            "CHROMA_COLLECTION_NAME", dotenv, chroma_yml.get("collection_name")
        ),
        persist_directory=_resolve_str(
            "CHROMA_PERSIST_DIR", dotenv, chroma_yml.get("persist_directory")
        ),
        k=_resolve_int("CHROMA_K", dotenv, chroma_yml.get("k")),
        chunk_size=_resolve_int("CHUNK_SIZE", dotenv, chroma_yml.get("chunk_size")),
        chunk_overlap=_resolve_int("CHUNK_OVERLAP", dotenv, chroma_yml.get("chunk_overlap")),
        separators=_resolve_str_list(
            "CHROMA_SEPARATORS", dotenv, chroma_yml.get("separators")
        ),
        allow_knowledge_file_type=_resolve_str_list(
            "ALLOWED_FILE_TYPES", dotenv, chroma_yml.get("allow_knowledge_file_type")
        ),
        data_path=_resolve_str("DATA_DIR", dotenv, chroma_yml.get("data_path")),
        md5_hex_store=_resolve_str("MD5_HEX_STORE", dotenv, chroma_yml.get("md5_hex_store")),
    )

    # Assemble RagSettings — supports env override
    rag = RagSettings(
        external_data_path=_resolve_str(
            "EXTERNAL_DATA_PATH", dotenv, rag_yml.get("external_data_path")
        ),
    )

    # Assemble VectorSettings — config/vector.yml with env override
    vector_yml = _load_yaml("vector.yml")
    provider = _resolve_str("VECTOR_PROVIDER", dotenv, vector_yml.get("provider", "qdrant"))

    qdrant_yml = vector_yml.get("qdrant", {})
    qdrant = QdrantSettings(
        url=_resolve_str("QDRANT_URL", dotenv, qdrant_yml.get("url")),
        collection_name=_resolve_str(
            "QDRANT_COLLECTION_NAME", dotenv, qdrant_yml.get("collection_name")
        ),
        distance=_resolve_str("QDRANT_DISTANCE", dotenv, qdrant_yml.get("distance", "Cosine")),
        vector_size=_resolve_int(
            "QDRANT_VECTOR_SIZE", dotenv, qdrant_yml.get("vector_size", 1536)
        ),
        timeout_seconds=_resolve_int(
            "QDRANT_TIMEOUT", dotenv, qdrant_yml.get("timeout_seconds", 10)
        ),
    )

    ingest_yml = vector_yml.get("ingest", {})
    ingest = IngestSettings(
        chunk_size=_resolve_int(
            "VECTOR_INGEST_CHUNK_SIZE", dotenv, ingest_yml.get("chunk_size", 600)
        ),
        chunk_overlap=_resolve_int(
            "VECTOR_INGEST_CHUNK_OVERLAP", dotenv, ingest_yml.get("chunk_overlap", 80)
        ),
        batch_size=_resolve_int(
            "VECTOR_INGEST_BATCH_SIZE", dotenv, ingest_yml.get("batch_size", 64)
        ),
    )

    vector = VectorSettings(provider=provider, qdrant=qdrant, ingest=ingest)

    # Assemble SecuritySettings — only from env, never from YAML
    internal_token = _resolve_env("FASTAPI_INTERNAL_TOKEN", dotenv)
    security = SecuritySettings(internal_token=internal_token)

    # Assemble PromptSettings — every field supports env override
    prompts = PromptSettings(
        prompts_dir=_resolve_str("PROMPTS_DIR", dotenv, prompts_yml.get("prompts_dir")),
        main_prompt_path=_resolve_str(
            "MAIN_PROMPT_PATH", dotenv, prompts_yml.get("main_prompt_path")
        ),
        rag_summarize_prompt_path=_resolve_str(
            "RAG_SUMMARIZE_PROMPT_PATH", dotenv, prompts_yml.get("rag_summarize_prompt_path")
        ),
        report_prompt_path=_resolve_str(
            "REPORT_PROMPT_PATH", dotenv, prompts_yml.get("report_prompt_path")
        ),
    )

    settings = AppSettings(
        server=server,
        model=model,
        vector=vector,
        chroma=chroma,
        rag=rag,
        prompts=prompts,
        security=security,
        project_root=_PROJECT_ROOT,
    )

    logger.info(
        "[config_loader] Settings loaded: model=%s, base_url=%s, host=%s:%d",
        settings.model.chat_model_name,
        settings.model.openai_compatible_base_url,
        settings.server.host,
        settings.server.port,
    )
    return settings
