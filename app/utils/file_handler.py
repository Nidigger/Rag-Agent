from __future__ import annotations

import hashlib
import logging
import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

logger = logging.getLogger("rag-agent.file_handler")


def get_file_md5_hex(filepath: str) -> str | None:
    if not os.path.exists(filepath):
        logger.error(f"[md5] file not found: {filepath}")
        return None
    if not os.path.isfile(filepath):
        logger.error(f"[md5] not a file: {filepath}")
        return None

    md5_obj = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(4096):
                md5_obj.update(chunk)
        return md5_obj.hexdigest()
    except Exception as e:
        logger.error(f"[md5] failed for {filepath}: {e}")
        return None


def listdir_with_allowed_type(
    path: str, allowed_types: tuple[str, ...]
) -> tuple[str, ...]:
    if not os.path.isdir(path):
        logger.error(f"[listdir] not a directory: {path}")
        return ()
    return tuple(
        os.path.join(path, f)
        for f in os.listdir(path)
        if f.endswith(allowed_types)
    )


def pdf_loader(filepath: str, passwd=None) -> list[Document]:
    return PyPDFLoader(filepath, password=passwd).load()


def txt_loader(filepath: str) -> list[Document]:
    return TextLoader(filepath, encoding="utf-8").load()
