import logging

from app.rag.retriever import RagSummarizeService
from app.common.exceptions import RAGRetrievalError

logger = logging.getLogger("rag-agent.rag_service")

_rag_service: RagSummarizeService | None = None


def get_rag_service() -> RagSummarizeService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RagSummarizeService()
    return _rag_service


def rag_query(query: str, top_k: int = 3, knowledge_base_id: str = "kb_default") -> str:
    try:
        service = get_rag_service()
        return service.rag_summarize(
            query=query,
            top_k=top_k,
            knowledge_base_id=knowledge_base_id,
        )
    except Exception as e:
        logger.error(f"RAG query failed: {e}", exc_info=True)
        raise RAGRetrievalError(str(e))
