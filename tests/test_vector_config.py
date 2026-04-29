"""Unit tests for VectorSettings and QdrantSettings configuration."""

import os
from unittest.mock import patch

import pytest
import yaml


class TestVectorSettingsFromYAML:
    def test_vector_yml_is_valid(self):
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "vector.yml"
        )
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert data["provider"] in ("qdrant", "chroma")
        assert "qdrant" in data
        assert "url" in data["qdrant"]
        assert "collection_name" in data["qdrant"]
        assert "distance" in data["qdrant"]
        assert "vector_size" in data["qdrant"]
        assert "ingest" in data
        assert "chunk_size" in data["ingest"]
        assert "chunk_overlap" in data["ingest"]
        assert "batch_size" in data["ingest"]

    def test_qdrant_settings_loaded(self):
        from app.config import settings

        assert settings.vector.provider == "qdrant"
        assert settings.vector.qdrant.url is not None
        assert settings.vector.qdrant.collection_name is not None
        assert settings.vector.qdrant.vector_size > 0
        assert settings.vector.ingest.chunk_size > 0
        assert settings.vector.ingest.batch_size > 0

    def test_chroma_settings_still_loaded(self):
        from app.config import settings

        assert settings.chroma.collection_name is not None
        assert settings.chroma.persist_directory is not None
        assert settings.chroma.k > 0


class TestQdrantSettingsModel:
    def test_model_defaults(self):
        from app.config.schema import QdrantSettings

        with pytest.raises(Exception):
            QdrantSettings()

    def test_model_minimal(self):
        from app.config.schema import QdrantSettings

        s = QdrantSettings(
            url="http://localhost:6333",
            collection_name="test",
        )
        assert s.distance == "Cosine"
        assert s.vector_size == 1536
        assert s.timeout_seconds == 10


class TestVectorSettingsModel:
    def test_model_minimal(self):
        from app.config.schema import VectorSettings, QdrantSettings, IngestSettings

        s = VectorSettings(
            provider="qdrant",
            qdrant=QdrantSettings(
                url="http://localhost:6333",
                collection_name="test",
            ),
            ingest=IngestSettings(),
        )
        assert s.provider == "qdrant"
        assert s.qdrant.url == "http://localhost:6333"
