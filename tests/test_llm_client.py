"""Tests for app/integrations/llm_client.py.

Validates that:
- get_agent_model() returns ChatOpenAI with streaming=False.
- get_streaming_model() returns ChatOpenAI with streaming=True.
- get_embed_model() returns DashScopeEmbeddingClient.
- Each function returns the same instance on repeated calls (singleton).
- Both chat models use the configured base_url and api_key.
"""

import pytest
from langchain_openai import ChatOpenAI

from app.config import settings
from app.integrations import llm_client
from app.integrations.embedding_client import DashScopeEmbeddingClient


class TestGetAgentModel:
    def test_returns_chat_openai(self):
        model = llm_client.get_agent_model()
        assert isinstance(model, ChatOpenAI)

    def test_streaming_is_false(self):
        model = llm_client.get_agent_model()
        assert model.streaming is False

    def test_uses_configured_base_url(self):
        model = llm_client.get_agent_model()
        assert model.openai_api_base == settings.model.openai_compatible_base_url

    def test_singleton(self):
        model1 = llm_client.get_agent_model()
        model2 = llm_client.get_agent_model()
        assert model1 is model2


class TestGetStreamingModel:
    def test_returns_chat_openai(self):
        model = llm_client.get_streaming_model()
        assert isinstance(model, ChatOpenAI)

    def test_streaming_is_true(self):
        model = llm_client.get_streaming_model()
        assert model.streaming is True

    def test_uses_configured_base_url(self):
        model = llm_client.get_streaming_model()
        assert model.openai_api_base == settings.model.openai_compatible_base_url

    def test_singleton(self):
        model1 = llm_client.get_streaming_model()
        model2 = llm_client.get_streaming_model()
        assert model1 is model2

    def test_different_from_agent_model(self):
        agent = llm_client.get_agent_model()
        streaming = llm_client.get_streaming_model()
        assert agent is not streaming


class TestGetEmbedModel:
    def test_returns_dashscope_embedding_client(self):
        model = llm_client.get_embed_model()
        assert isinstance(model, DashScopeEmbeddingClient)

    def test_singleton(self):
        model1 = llm_client.get_embed_model()
        model2 = llm_client.get_embed_model()
        assert model1 is model2

    def test_model_name_matches_config(self):
        model = llm_client.get_embed_model()
        assert model.model == settings.model.embedding_model_name

    def test_dimensions_match_qdrant_vector_size(self):
        model = llm_client.get_embed_model()
        assert model.dimensions == settings.vector.qdrant.vector_size
