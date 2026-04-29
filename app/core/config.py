"""Application configuration — thin entry point.

Re-exports the global `settings` singleton.  All loading, validation,
and type definitions live in config_loader.py and config_schema.py.

Usage from business modules:
    from app.core.config import settings
    settings.server.project_name
    settings.model.chat_model_name
    settings.chroma.collection_name
"""

from app.core.config_loader import load_settings

settings = load_settings()
