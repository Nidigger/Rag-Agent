import logging
import os

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.integrations.dashscope_client import get_embed_model
from app.utils.file_handler import (
    get_file_md5_hex,
    listdir_with_allowed_type,
    pdf_loader,
    txt_loader,
)
from app.utils.path_tool import get_abs_path

logger = logging.getLogger("rag-agent.vector_store")


class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=get_embed_model(),
            persist_directory=get_abs_path(settings.CHROMA_PERSIST_DIR),
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=settings.CHROMA_SEPARATORS,
            length_function=len,
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(
            search_kwargs={"k": settings.CHROMA_K}
        )

    def load_document(self):
        md5_store_path = get_abs_path(settings.MD5_HEX_STORE)

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
            get_abs_path(settings.DATA_DIR),
            tuple(settings.ALLOWED_FILE_TYPES),
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

                split_docs = self.splitter.split_documents(documents)
                if not split_docs:
                    logger.warning(
                        f"[load_document] {path} produced no chunks"
                    )
                    continue

                self.vector_store.add_documents(split_docs)
                save_md5_hex(md5_hex)
                logger.info(f"[load_document] {path} loaded successfully")
            except Exception as e:
                logger.error(
                    f"[load_document] {path} failed: {e}", exc_info=True
                )
