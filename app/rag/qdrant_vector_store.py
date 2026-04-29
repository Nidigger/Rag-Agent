"""Qdrant vector store implementation.

Implements the VectorStore abstract interface using the official
qdrant-client library. Handles embedding generation via the
configured embedding model, upsert, search, and deletion.
"""

import logging
import uuid as uuid_lib

from langchain_core.documents import Document
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct

from app.config import settings
from app.integrations.llm_client import get_embed_model
from app.integrations.qdrant_client import ensure_qdrant_collection, get_qdrant_client
from app.rag.vector_store_base import (
    RetrievedChunk,
    VectorChunk,
    VectorStore,
)

logger = logging.getLogger("rag-agent.qdrant_vector_store")


class QdrantVectorStore(VectorStore):
    """Qdrant-based vector store implementation."""

    def __init__(
        self,
        collection_name: str | None = None,
        vector_size: int | None = None,
        distance: str | None = None,
    ):
        self._collection_name = collection_name or settings.vector.qdrant.collection_name
        self._vector_size = vector_size or settings.vector.qdrant.vector_size
        self._distance = distance or settings.vector.qdrant.distance
        self._embed_model = get_embed_model()
        self._client = get_qdrant_client()

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def ensure_collection(self) -> None:
        ensure_qdrant_collection(
            collection_name=self._collection_name,
            vector_size=self._vector_size,
            distance=self._distance,
        )

    def upsert_chunks(self, chunks: list[VectorChunk]) -> None:
        if not chunks:
            return

        points = []
        for chunk in chunks:
            vector = self._embed_model.embed_query(chunk.content)
            payload = chunk.model_dump(exclude={"point_id", "metadata"})
            if chunk.metadata:
                payload.update(chunk.metadata)

            points.append(
                PointStruct(
                    id=uuid_lib.UUID(chunk.point_id),
                    vector=vector,
                    payload=payload,
                )
            )

        self._client.upsert(
            collection_name=self._collection_name,
            points=points,
            wait=True,
        )
        logger.info(
            "[qdrant_vector_store] Upserted %d chunks into '%s'",
            len(points),
            self._collection_name,
        )

    def search(
        self,
        query: str,
        top_k: int,
        filters: dict | None = None,
    ) -> list[RetrievedChunk]:
        query_vector = self._embed_model.embed_query(query)

        qdrant_filter = None
        if filters:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
            ]
            qdrant_filter = Filter(must=conditions)

        response = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        results = response.points

        chunks = []
        for result in results:
            payload = result.payload or {}
            chunks.append(
                RetrievedChunk(
                    point_id=str(result.id),
                    document_id=payload.get("document_id", ""),
                    chunk_index=payload.get("chunk_index", 0),
                    content=payload.get("content", ""),
                    score=result.score,
                    metadata=payload.get("metadata", {}),
                )
            )
        return chunks

    def search_as_documents(
        self,
        query: str,
        top_k: int,
        filters: dict | None = None,
    ) -> list[Document]:
        """Search and return results as LangChain Document objects.

        Compatible with the existing retriever pattern used by agent tools.
        """
        retrieved = self.search(query, top_k, filters)
        documents = []
        for chunk in retrieved:
            doc = Document(
                page_content=chunk.content,
                metadata={
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "score": chunk.score,
                    **chunk.metadata,
                },
            )
            documents.append(doc)
        return documents

    def delete_document(self, document_id: str) -> None:
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            ),
        )
        logger.info(
            "[qdrant_vector_store] Deleted all points for document_id='%s'",
            document_id,
        )

    def disable_document(self, document_id: str) -> None:
        """Set enabled=false for all chunks of a document (logical delete)."""
        self._client.set_payload(
            collection_name=self._collection_name,
            payload={"enabled": False},
            points=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            ),
            wait=True,
        )
        logger.info(
            "[qdrant_vector_store] Disabled document_id='%s'", document_id
        )

    def health_check(self) -> bool:
        try:
            self._client.get_collection(self._collection_name)
            return True
        except Exception as e:
            logger.error(
                "[qdrant_vector_store] Health check failed: %s", e
            )
            return False
