"""LLM client module providing dual model access.

Provides two separate model instances:
- Agent model: non-streaming ChatOpenAI for tool orchestration
  (tool calls require complete, valid JSON arguments).
- Streaming model: streaming ChatOpenAI for final answer generation
  (no tools bound, safe to stream token-by-token).

Both instances use the OpenAI-compatible endpoint configured in settings,
making it easy to switch between DashScope, SiliconFlow, DeepSeek, etc.
"""

import logging

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI

from app.config import settings

logger = logging.getLogger("rag-agent.llm_client")

_agent_model: ChatOpenAI | None = None
_streaming_model: ChatOpenAI | None = None
_embed_model: DashScopeEmbeddings | None = None


def get_agent_model() -> ChatOpenAI:
    """Non-streaming model for Agent/tool orchestration.

    Must NOT use streaming to ensure function arguments are complete
    and valid JSON when calling tools.
    """
    global _agent_model
    if _agent_model is None:
        logger.info(
            "[llm_client] Initializing agent model: "
            "model=%s, base_url=%s, streaming=False",
            settings.model.chat_model_name,
            settings.model.openai_compatible_base_url,
        )
        _agent_model = ChatOpenAI(
            model=settings.model.chat_model_name,
            api_key=settings.model.dashscope_api_key,
            base_url=settings.model.openai_compatible_base_url,
            streaming=False,
        )
    return _agent_model


def get_streaming_model() -> ChatOpenAI:
    """Streaming model for final answer generation only.

    Do NOT bind tools to this model — it is exclusively for producing
    the user-facing response as a token stream.
    """
    global _streaming_model
    if _streaming_model is None:
        logger.info(
            "[llm_client] Initializing streaming model: "
            "model=%s, base_url=%s, streaming=True",
            settings.model.chat_model_name,
            settings.model.openai_compatible_base_url,
        )
        _streaming_model = ChatOpenAI(
            model=settings.model.chat_model_name,
            api_key=settings.model.dashscope_api_key,
            base_url=settings.model.openai_compatible_base_url,
            streaming=True,
        )
    return _streaming_model


def get_embed_model() -> DashScopeEmbeddings:
    """DashScope embedding model for vector store operations."""
    global _embed_model
    if _embed_model is None:
        logger.info(
            "[llm_client] Initializing embedding model: %s",
            settings.model.embedding_model_name,
        )
        _embed_model = DashScopeEmbeddings(
            model=settings.model.embedding_model_name
        )
    return _embed_model
