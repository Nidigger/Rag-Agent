import logging

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from app.agent.tools.request_context import get_request_context
from app.config import settings
from app.integrations.llm_client import get_agent_model
from app.rag.vector_store_base import VectorStore
from app.utils.perf import elapsed_ms, log_perf, now_ms
from app.utils.prompt_loader import load_rag_prompts

logger = logging.getLogger("rag-agent.retriever")


class RagSummarizeService:
    def __init__(self, vector_store: VectorStore | None = None):
        if vector_store is not None:
            self._vector_store = vector_store
        else:
            self._vector_store = _get_vector_store()

        self.prompt_template = PromptTemplate.from_template(
            load_rag_prompts()
        )
        self.model = get_agent_model()
        self.chain = (
            self.prompt_template | self.model | StrOutputParser()
        )

    def retriever_docs(
        self,
        query: str,
        top_k: int,
        knowledge_base_id: str,
        request_id: str | None = None,
    ) -> list:
        rid = request_id or get_request_context().get("request_id", "internal")

        retrieval_start = now_ms()
        chunks = self._vector_store.search(
            query=query,
            top_k=top_k,
            filters={
                "enabled": True,
                "knowledge_base_id": knowledge_base_id,
            },
        )

        log_perf("rag_tool", "retrieval_done",
                 request_id=rid,
                 elapsed_ms=elapsed_ms(retrieval_start),
                 result_count=len(chunks))

        docs = []
        for chunk in chunks:
            docs.append(
                Document(
                    page_content=chunk.content,
                    metadata={
                        "document_id": chunk.document_id,
                        "chunk_index": chunk.chunk_index,
                        "score": chunk.score,
                        **chunk.metadata,
                    },
                )
            )
        return docs

    def rag_summarize(
        self,
        query: str,
        top_k: int = 3,
        knowledge_base_id: str = "kb_default",
        request_id: str | None = None,
    ) -> str:
        rid = request_id or get_request_context().get("request_id", "internal")

        context_docs = self.retriever_docs(
            query=query,
            top_k=top_k,
            knowledge_base_id=knowledge_base_id,
            request_id=rid,
        )

        context = ""
        for i, doc in enumerate(context_docs, 1):
            context += (
                f"[{i}] {doc.page_content} | meta: {doc.metadata}\n"
            )

        log_perf("rag_tool", "summarize_start",
                 request_id=rid,
                 context_len=len(context))

        summarize_start = now_ms()
        result = self.chain.invoke({"input": query, "context": context})
        summarize_elapsed = elapsed_ms(summarize_start)

        log_perf("rag_tool", "summarize_done",
                 request_id=rid,
                 elapsed_ms=summarize_elapsed,
                 output_len=len(result))

        return result


def _get_vector_store() -> VectorStore:
    """Factory: resolve the active vector store based on config."""
    from app.rag.vector_store import ChromaVectorStore
    from app.rag.qdrant_vector_store import QdrantVectorStore

    provider = settings.vector.provider
    if provider == "qdrant":
        logger.info("[retriever] Using Qdrant vector store")
        store = QdrantVectorStore()
        store.ensure_collection()
        return store

    logger.info("[retriever] Using Chroma vector store (fallback)")
    return ChromaVectorStore()
