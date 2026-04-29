"""Tests for app/config — YAML-driven configuration.

Validates that:
- YAML files are loaded and their values appear in settings.
- .env secrets (DASHSCOPE_API_KEY) are still resolved.
- Prompt path fields exist and are non-empty.
- YAML defaults are used when no env var override exists.
- Nested settings structure is correctly assembled.
- Missing required config causes startup failure.
"""

import json
import os
from pathlib import Path

import pytest
import yaml

from app.config import settings
from app.config.loader import _load_yaml, load_settings
from app.config.schema import (
    AppSettings,
    ChromaSettings,
    ModelSettings,
    PromptSettings,
    RagSettings,
    ServerSettings,
)


class TestYAMLLoading:
    def test_load_agent_yaml(self):
        data = _load_yaml("agent.yml")
        assert "chat_model_name" in data
        assert "embedding_model_name" in data
        assert "openai_compatible_base_url" in data

    def test_load_server_yaml(self):
        data = _load_yaml("server.yml")
        assert data["project_name"] == "Rag-Agent Service"
        assert data["host"] == "0.0.0.0"
        assert data["port"] == 8000
        assert "cors_allow_origins" in data
        assert data["log_level"] == "INFO"

    def test_load_chroma_yaml(self):
        data = _load_yaml("chroma.yml")
        assert data["collection_name"] == "agent"
        assert data["chunk_size"] == 200
        assert isinstance(data["separators"], list)

    def test_load_rag_yaml(self):
        data = _load_yaml("rag.yml")
        assert "external_data_path" in data

    def test_load_prompts_yaml(self):
        data = _load_yaml("prompts.yml")
        assert "prompts_dir" in data
        assert "main_prompt_path" in data
        assert "rag_summarize_prompt_path" in data
        assert "report_prompt_path" in data

    def test_load_nonexistent_yaml_raises(self):
        with pytest.raises(FileNotFoundError):
            _load_yaml("nonexistent.yml")


class TestNestedSettingsStructure:
    def test_settings_singleton_exists(self):
        assert settings is not None

    def test_settings_is_app_settings(self):
        assert isinstance(settings, AppSettings)

    def test_server_settings_group(self):
        assert isinstance(settings.server, ServerSettings)
        assert settings.server.project_name == "Rag-Agent Service"
        assert settings.server.host == "0.0.0.0"
        assert settings.server.port == 8000
        assert settings.server.log_level == "INFO"
        assert settings.server.api_v1_prefix == "/api/v1"
        assert isinstance(settings.server.cors_allow_origins, list)
        assert len(settings.server.cors_allow_origins) > 0

    def test_model_settings_group(self):
        assert isinstance(settings.model, ModelSettings)
        assert settings.model.chat_model_name == "qwen3-max"
        assert settings.model.embedding_model_name == "text-embedding-v4"
        assert "dashscope" in settings.model.openai_compatible_base_url
        assert settings.model.dashscope_api_key is not None

    def test_chroma_settings_group(self):
        assert isinstance(settings.chroma, ChromaSettings)
        assert settings.chroma.collection_name == "agent"
        assert settings.chroma.k == 3
        assert settings.chroma.chunk_size == 200
        assert settings.chroma.chunk_overlap == 20
        assert isinstance(settings.chroma.separators, list)
        assert isinstance(settings.chroma.allow_knowledge_file_type, list)
        assert settings.chroma.data_path is not None
        assert settings.chroma.md5_hex_store is not None

    def test_rag_settings_group(self):
        assert isinstance(settings.rag, RagSettings)
        assert settings.rag.external_data_path is not None

    def test_prompt_settings_group(self):
        assert isinstance(settings.prompts, PromptSettings)
        assert settings.prompts.prompts_dir is not None
        assert settings.prompts.main_prompt_path == "prompts/main_prompt.txt"
        assert settings.prompts.rag_summarize_prompt_path == "prompts/rag_summarize.txt"
        assert settings.prompts.report_prompt_path == "prompts/report_prompt.txt"

    def test_project_root_is_valid_path(self):
        assert isinstance(settings.project_root, Path)
        assert settings.project_root.exists()


class TestApiKeyFromEnv:
    def test_dashscope_api_key_from_env(self):
        assert settings.model.dashscope_api_key is not None
        # SecretStr should not be empty
        assert len(settings.model.dashscope_api_key.get_secret_value()) > 0

    def test_api_key_not_in_yaml(self):
        agent_yml = _load_yaml("agent.yml")
        assert "dashscope_api_key" not in agent_yml


class TestConfigLoaderPriority:
    """Verify that environment variables take priority over YAML."""

    def test_env_overrides_model_name(self, monkeypatch):
        monkeypatch.setenv("CHAT_MODEL_NAME", "qwen3-plus")
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
        s = load_settings()
        assert s.model.chat_model_name == "qwen3-plus"

    def test_env_overrides_port(self, monkeypatch):
        monkeypatch.setenv("PORT", "9999")
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
        s = load_settings()
        assert s.server.port == 9999

    def test_env_overrides_log_level(self, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
        s = load_settings()
        assert s.server.log_level == "DEBUG"


class TestMissingRequiredConfig:
    def test_missing_api_key_raises(self, monkeypatch):
        """DASHSCOPE_API_KEY must come from env, not YAML.

        When both OS env and .env lack the key, load_settings() should
        raise a ValueError rather than silently falling back to a default.
        """
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

        from app.config.loader import _resolve_env

        dotenv = {}
        api_key = _resolve_env("DASHSCOPE_API_KEY", dotenv)
        assert api_key is None, "Expected no API key from empty dotenv"

        with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
            if not api_key:
                raise ValueError(
                    "DASHSCOPE_API_KEY is required but not found in "
                    "environment variables or .env file"
                )


def _env_test_setup(monkeypatch):
    """Ensure DASHSCOPE_API_KEY is set so load_settings() doesn't fail."""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")


class TestEnvOverrideServerSettings:
    """Verify ServerSettings fields can be overridden by OS env."""

    def test_env_overrides_cors_json_list(self, monkeypatch):
        _env_test_setup(monkeypatch)
        monkeypatch.setenv("CORS_ALLOW_ORIGINS", '["http://example.com"]')
        s = load_settings()
        assert s.server.cors_allow_origins == ["http://example.com"]

    def test_env_overrides_cors_comma_separated(self, monkeypatch):
        _env_test_setup(monkeypatch)
        monkeypatch.setenv("CORS_ALLOW_ORIGINS", "http://a.com, http://b.com")
        s = load_settings()
        assert s.server.cors_allow_origins == ["http://a.com", "http://b.com"]

    def test_env_overrides_project_name(self, monkeypatch):
        _env_test_setup(monkeypatch)
        monkeypatch.setenv("PROJECT_NAME", "Custom Service")
        s = load_settings()
        assert s.server.project_name == "Custom Service"

    def test_env_overrides_debug_true(self, monkeypatch):
        _env_test_setup(monkeypatch)
        monkeypatch.setenv("DEBUG", "true")
        s = load_settings()
        assert s.server.debug is True

    def test_env_overrides_debug_yes(self, monkeypatch):
        _env_test_setup(monkeypatch)
        monkeypatch.setenv("DEBUG", "yes")
        s = load_settings()
        assert s.server.debug is True


class TestEnvOverrideChromaSettings:
    """Verify ChromaSettings fields can be overridden by OS env."""

    def test_env_overrides_chroma_persist_dir(self, monkeypatch):
        _env_test_setup(monkeypatch)
        monkeypatch.setenv("CHROMA_PERSIST_DIR", "custom_chroma")
        s = load_settings()
        assert s.chroma.persist_directory == "custom_chroma"

    def test_env_overrides_chroma_collection_name(self, monkeypatch):
        _env_test_setup(monkeypatch)
        monkeypatch.setenv("CHROMA_COLLECTION_NAME", "my_collection")
        s = load_settings()
        assert s.chroma.collection_name == "my_collection"

    def test_env_overrides_chroma_numeric_fields(self, monkeypatch):
        _env_test_setup(monkeypatch)
        monkeypatch.setenv("CHROMA_K", "8")
        monkeypatch.setenv("CHUNK_SIZE", "512")
        monkeypatch.setenv("CHUNK_OVERLAP", "50")
        s = load_settings()
        assert s.chroma.k == 8
        assert s.chroma.chunk_size == 512
        assert s.chroma.chunk_overlap == 50

    def test_env_overrides_chroma_separators_json(self, monkeypatch):
        _env_test_setup(monkeypatch)
        monkeypatch.setenv("CHROMA_SEPARATORS", '["a","b","c"]')
        s = load_settings()
        assert s.chroma.separators == ["a", "b", "c"]

    def test_env_overrides_allowed_file_types_json(self, monkeypatch):
        _env_test_setup(monkeypatch)
        monkeypatch.setenv("ALLOWED_FILE_TYPES", '["txt","pdf","md"]')
        s = load_settings()
        assert s.chroma.allow_knowledge_file_type == ["txt", "pdf", "md"]

    def test_env_overrides_data_dir(self, monkeypatch):
        _env_test_setup(monkeypatch)
        monkeypatch.setenv("DATA_DIR", "custom_data")
        s = load_settings()
        assert s.chroma.data_path == "custom_data"

    def test_env_overrides_md5_hex_store(self, monkeypatch):
        _env_test_setup(monkeypatch)
        monkeypatch.setenv("MD5_HEX_STORE", "custom_md5.text")
        s = load_settings()
        assert s.chroma.md5_hex_store == "custom_md5.text"


class TestEnvOverrideRagSettings:
    """Verify RagSettings fields can be overridden by OS env."""

    def test_env_overrides_external_data_path(self, monkeypatch):
        _env_test_setup(monkeypatch)
        monkeypatch.setenv("EXTERNAL_DATA_PATH", "data/custom/records.csv")
        s = load_settings()
        assert s.rag.external_data_path == "data/custom/records.csv"


class TestEnvOverridePromptSettings:
    """Verify PromptSettings fields can be overridden by OS env."""

    def test_env_overrides_main_prompt_path(self, monkeypatch):
        _env_test_setup(monkeypatch)
        monkeypatch.setenv("MAIN_PROMPT_PATH", "prompts/custom_main.txt")
        s = load_settings()
        assert s.prompts.main_prompt_path == "prompts/custom_main.txt"

    def test_env_overrides_rag_summarize_prompt_path(self, monkeypatch):
        _env_test_setup(monkeypatch)
        monkeypatch.setenv("RAG_SUMMARIZE_PROMPT_PATH", "prompts/custom_rag.txt")
        s = load_settings()
        assert s.prompts.rag_summarize_prompt_path == "prompts/custom_rag.txt"

    def test_env_overrides_report_prompt_path(self, monkeypatch):
        _env_test_setup(monkeypatch)
        monkeypatch.setenv("REPORT_PROMPT_PATH", "prompts/custom_report.txt")
        s = load_settings()
        assert s.prompts.report_prompt_path == "prompts/custom_report.txt"

    def test_env_overrides_prompts_dir(self, monkeypatch):
        _env_test_setup(monkeypatch)
        monkeypatch.setenv("PROMPTS_DIR", "custom_prompts")
        s = load_settings()
        assert s.prompts.prompts_dir == "custom_prompts"


class TestTypeConversionHelpers:
    """Verify helper functions handle edge cases correctly."""

    def test_resolve_str_list_comma_fallback(self, monkeypatch):
        from app.config.loader import _resolve_str_list

        dotenv = {}
        result = _resolve_str_list("TEST_KEY", dotenv, ["default"])
        assert result == ["default"]

    def test_resolve_str_list_json_array(self, monkeypatch):
        from app.config.loader import _resolve_str_list

        dotenv = {"TEST_KEY": '["a","b","c"]'}
        result = _resolve_str_list("TEST_KEY", dotenv)
        assert result == ["a", "b", "c"]

    def test_resolve_str_list_json_with_newlines(self):
        from app.config.loader import _resolve_str_list

        dotenv = {"SEP": json.dumps(["\n\n", "\n", " "])}
        result = _resolve_str_list("SEP", dotenv)
        assert result == ["\n\n", "\n", " "]

    def test_resolve_bool_accepts_variants(self):
        from app.config.loader import _resolve_bool

        assert _resolve_bool("K", {}, "True") is True
        assert _resolve_bool("K", {}, "true") is True
        assert _resolve_bool("K", {}, "1") is True
        assert _resolve_bool("K", {}, "yes") is True
        assert _resolve_bool("K", {}, "on") is True
        assert _resolve_bool("K", {}, "False") is False
        assert _resolve_bool("K", {}, "false") is False
        assert _resolve_bool("K", {}, "0") is False
