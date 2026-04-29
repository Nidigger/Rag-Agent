"""Application configuration — thin entry point.

Re-exports the global `settings` singleton.  All loading, validation,
and type definitions live in loader.py and schema.py.

Usage from business modules:
    from app.config import settings
    settings.server.project_name
    settings.model.chat_model_name
    settings.chroma.collection_name
"""

from app.config.loader import load_settings

settings = load_settings()
