import logging

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from app.config import settings
from app.integrations.llm_client import get_agent_model
from app.rag.vector_store_base import VectorStore
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
    ) -> list:
        chunks = self._vector_store.search(
            query=query,
            top_k=top_k,
            filters={
                "enabled": True,
                "knowledge_base_id": knowledge_base_id,
            },
        )

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
    ) -> str:
        context_docs = self.retriever_docs(
            query=query,
            top_k=top_k,
            knowledge_base_id=knowledge_base_id,
        )

        context = ""
        for i, doc in enumerate(context_docs, 1):
            context += (
                f"[{i}] {doc.page_content} | meta: {doc.metadata}\n"
            )

        return self.chain.invoke({"input": query, "context": context})


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
