"""Storage abstraction layer.

Provides a factory ``get_storage`` that returns the appropriate
``ObjectStorage`` backend based on configuration.
"""

from __future__ import annotations

from app.storage.base import LocalFileRef, ObjectStorage

__all__ = ["get_storage", "LocalFileRef", "ObjectStorage"]

_storage_instances: dict[str, ObjectStorage] = {}


def get_storage(provider: str | None = None) -> ObjectStorage:
    """Return an ``ObjectStorage`` instance for the given provider.

    Instances are cached per provider for reuse within the process.

    Args:
        provider: Storage provider name (e.g. ``"minio"``).
            Defaults to ``settings.storage.provider``.

    Returns:
        A ready-to-use ``ObjectStorage`` instance.

    Raises:
        ValueError: If the provider is not supported.
    """
    from app.config import settings as _settings

    if provider is None:
        provider = _settings.storage.provider

    if provider in _storage_instances:
        return _storage_instances[provider]

    if provider == "minio":
        from app.storage.minio_storage import MinioStorage

        instance: ObjectStorage = MinioStorage()
    else:
        raise ValueError(f"Unsupported storage provider: {provider}")

    _storage_instances[provider] = instance
    return instance
