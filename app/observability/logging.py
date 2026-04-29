"""Logging configuration — console + daily rotating file.

Sets up the 'rag-agent' logger with two handlers:
- Console handler: INFO level for real-time monitoring.
- File handler: DEBUG level for detailed post-mortem analysis.

Log files are written to the logs/ directory under the project root,
one file per day (e.g. agent_2026-04-29.log).
"""

import logging
import os
from datetime import datetime

from app.config import settings

DEFAULT_LOG_FORMAT = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)


def setup_logging() -> None:
    """Initialize the application logging system."""
    log_root = os.path.join(str(settings.project_root), "logs")
    os.makedirs(log_root, exist_ok=True)

    root_logger = logging.getLogger("rag-agent")
    root_logger.setLevel(getattr(logging, settings.server.log_level.upper(), logging.INFO))

    # Avoid adding duplicate handlers on hot-reload
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

    root_logger.info("[logging] Logger initialized: level=%s", settings.server.log_level)
