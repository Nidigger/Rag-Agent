"""Prompt loader — reads system prompts from text files.

Prompt paths are configured in config/prompts.yml and exposed via
settings.prompts.main_prompt_path, rag_summarize_prompt_path, report_prompt_path.
"""

import logging

from app.core.config import settings
from app.utils.path_tool import get_abs_path

logger = logging.getLogger("rag-agent.prompt_loader")


def load_system_prompts() -> str:
    """Load the main chat system prompt."""
    path = get_abs_path(settings.prompts.main_prompt_path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        logger.debug("[prompt_loader] Loaded system prompt: %d chars", len(content))
        return content
    except Exception as e:
        logger.error("[prompt_loader] Failed to load system prompt: %s", e)
        raise


def load_rag_prompts() -> str:
    """Load the RAG summarization prompt."""
    path = get_abs_path(settings.prompts.rag_summarize_prompt_path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        logger.debug("[prompt_loader] Loaded RAG prompt: %d chars", len(content))
        return content
    except Exception as e:
        logger.error("[prompt_loader] Failed to load RAG prompt: %s", e)
        raise


def load_report_prompts() -> str:
    """Load the report generation system prompt."""
    path = get_abs_path(settings.prompts.report_prompt_path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        logger.debug("[prompt_loader] Loaded report prompt: %d chars", len(content))
        return content
    except Exception as e:
        logger.error("[prompt_loader] Failed to load report prompt: %s", e)
        raise
