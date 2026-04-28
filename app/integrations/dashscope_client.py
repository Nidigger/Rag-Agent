from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings

from app.core.config import settings

_chat_model = None
_embed_model = None


def get_chat_model() -> ChatTongyi:
    global _chat_model
    if _chat_model is None:
        _chat_model = ChatTongyi(model=settings.CHAT_MODEL_NAME)
    return _chat_model


def get_embed_model() -> DashScopeEmbeddings:
    global _embed_model
    if _embed_model is None:
        _embed_model = DashScopeEmbeddings(model=settings.EMBEDDING_MODEL_NAME)
    return _embed_model
