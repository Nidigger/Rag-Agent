"""ChromaDB vector store implementation (legacy / fallback).

Implements the VectorStore abstract interface using LangChain Chroma.
Maintained for backward compatibility alongside QdrantVectorStore.
"""

import logging
import os

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.integrations.llm_client import get_embed_model
from app.rag.vector_store_base import (
    RetrievedChunk,
    VectorChunk,
    VectorStore,
)
from app.utils.file_handler import (
    get_file_md5_hex,
    listdir_with_allowed_type,
    pdf_loader,
    txt_loader,
)
from app.utils.path_tool import get_abs_path

logger = logging.getLogger("rag-agent.vector_store")


class ChromaVectorStore(VectorStore):
    def __init__(self):
        self._vector_store = Chroma(
            collection_name=settings.chroma.collection_name,
            embedding_function=get_embed_model(),
            persist_directory=get_abs_path(settings.chroma.persist_directory),
        )
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chroma.chunk_size,
            chunk_overlap=settings.chroma.chunk_overlap,
            separators=settings.chroma.separators,
            length_function=len,
        )

    @property
    def collection_name(self) -> str:
        return settings.chroma.collection_name

    def ensure_collection(self) -> None:
        logger.info(
            "[chroma_vector_store] Using collection '%s' (persist: %s)",
            settings.chroma.collection_name,
            settings.chroma.persist_directory,
        )

    def upsert_chunks(self, chunks: list[VectorChunk]) -> None:
        docs = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk.content,
                metadata={
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "source": chunk.source or "",
                    **chunk.metadata,
                },
            )
            docs.append(doc)

        if docs:
            self._vector_store.add_documents(docs)
            logger.info(
                "[chroma_vector_store] Added %d documents", len(docs)
            )

    def search(
        self,
        query: str,
        top_k: int,
        filters: dict | None = None,
    ) -> list[RetrievedChunk]:
        search_kwargs = {"k": top_k}
        if filters:
            search_kwargs["filter"] = filters

        retriever = self._vector_store.as_retriever(
            search_kwargs=search_kwargs
        )
        results = retriever.invoke(query)

        chunks = []
        for i, doc in enumerate(results):
            chunks.append(
                RetrievedChunk(
                    point_id=doc.metadata.get("id", f"chroma-{i}"),
                    document_id=doc.metadata.get("document_id", ""),
                    chunk_index=doc.metadata.get("chunk_index", i),
                    content=doc.page_content,
                    score=doc.metadata.get("score", 0.0),
                    metadata=doc.metadata,
                )
            )
        return chunks

    def delete_document(self, document_id: str) -> None:
        self._vector_store.delete(
            filter={"document_id": document_id}
        )
        logger.info(
            "[chroma_vector_store] Deleted document_id='%s'", document_id
        )

    def health_check(self) -> bool:
        try:
            self._vector_store.get()
            return True
        except Exception as e:
            logger.error(
                "[chroma_vector_store] Health check failed: %s", e
            )
            return False

    def get_retriever(self):
        return self._vector_store.as_retriever(
            search_kwargs={"k": settings.chroma.k}
        )

    def load_document(self):
        md5_store_path = get_abs_path(settings.chroma.md5_hex_store)

        def check_md5_hex(md5_hex: str) -> bool:
            if not os.path.exists(md5_store_path):
                open(md5_store_path, "w", encoding="utf-8").close()
                return False
            with open(md5_store_path, "r", encoding="utf-8") as f:
                return any(
                    line.strip() == md5_hex for line in f.readlines()
                )

        def save_md5_hex(md5_hex: str):
            with open(md5_store_path, "a", encoding="utf-8") as f:
                f.write(md5_hex + "\n")

        def get_file_documents(read_path: str):
            if read_path.endswith("txt"):
                return txt_loader(read_path)
            if read_path.endswith("pdf"):
                return pdf_loader(read_path)
            return []

        allowed_files = listdir_with_allowed_type(
            get_abs_path(settings.chroma.data_path),
            tuple(settings.chroma.allow_knowledge_file_type),
        )

        for path in allowed_files:
            md5_hex = get_file_md5_hex(path)
            if check_md5_hex(md5_hex):
                logger.info(f"[load_document] {path} already loaded, skipping")
                continue

            try:
                documents = get_file_documents(path)
                if not documents:
                    logger.warning(
                        f"[load_document] {path} has no valid content"
                    )
                    continue

                split_docs = self._splitter.split_documents(documents)
                if not split_docs:
                    logger.warning(
                        f"[load_document] {path} produced no chunks"
                    )
                    continue

                self._vector_store.add_documents(split_docs)
                save_md5_hex(md5_hex)
                logger.info(f"[load_document] {path} loaded successfully")
            except Exception as e:
                logger.error(
                    f"[load_document] {path} failed: {e}", exc_info=True
                )


# Keep backward compatibility alias
VectorStoreService = ChromaVectorStore
