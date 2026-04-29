"""DashScope client — legacy model access (DEPRECATED for chat).

This module is retained for backward compatibility. The embedding
function (get_embed_model) has been moved to llm_client.py.

New code should use:
- app.integrations.llm_client.get_agent_model()     — non-streaming ChatOpenAI
- app.integrations.llm_client.get_streaming_model()  — streaming ChatOpenAI
- app.integrations.llm_client.get_embed_model()      — DashScope embeddings
"""

import logging
import warnings

from langchain_community.chat_models.tongyi import ChatTongyi

from app.core.config import settings

logger = logging.getLogger("rag-agent.dashscope_client")

_chat_model = None


def get_chat_model() -> ChatTongyi:
    """DEPRECATED: Use llm_client.get_agent_model() instead."""
    warnings.warn(
        "get_chat_model() is deprecated. "
        "Use app.integrations.llm_client.get_agent_model() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    global _chat_model
    if _chat_model is None:
        logger.warning(
            "[dashscope_client] Using deprecated ChatTongyi. "
            "Migrate to llm_client.get_agent_model()."
        )
        _chat_model = ChatTongyi(model=settings.model.chat_model_name)
    return _chat_model
