import logging

from app.core.config import settings
from app.utils.path_tool import get_abs_path

logger = logging.getLogger("rag-agent.prompt_loader")


def load_system_prompts() -> str:
    path = get_abs_path(f"{settings.PROMPTS_DIR}/main_prompt.txt")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"[load_system_prompts] failed: {e}")
        raise


def load_rag_prompts() -> str:
    path = get_abs_path(f"{settings.PROMPTS_DIR}/rag_summarize.txt")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"[load_rag_prompts] failed: {e}")
        raise


def load_report_prompts() -> str:
    path = get_abs_path(f"{settings.PROMPTS_DIR}/report_prompt.txt")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"[load_report_prompts] failed: {e}")
        raise
