import logging
import os
from datetime import datetime

from app.core.config import settings

DEFAULT_LOG_FORMAT = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)


def setup_logging() -> None:
    log_root = os.path.join(str(settings.PROJECT_ROOT), "logs")
    os.makedirs(log_root, exist_ok=True)

    root_logger = logging.getLogger("rag-agent")
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    if root_logger.handlers:
        return

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)
    root_logger.addHandler(console_handler)

    log_file = os.path.join(
        log_root, f"agent_{datetime.now().strftime('%Y-%m-%d')}.log"
    )
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)
    root_logger.addHandler(file_handler)
