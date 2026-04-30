"""Abstract base for object storage backends."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LocalFileRef:
    """Reference to a downloaded object stored as a local temporary file.

    Attributes:
        path: Absolute path to the temporary file on disk.
        filename: Original filename extracted from the object key.
        content_type: MIME type if known, otherwise None.
        should_cleanup: Whether the temp file should be deleted after use.
    """

    path: str
    filename: str
    content_type: str | None = None
    should_cleanup: bool = True


class ObjectStorage:
    """Abstract interface for object storage operations.

    Implementations must override ``download_to_temp``.
    """

    def download_to_temp(self, bucket: str, object_key: str) -> LocalFileRef:
        """Download an object to a local temporary file.

        Args:
            bucket: Storage bucket name.
            object_key: Full object key within the bucket.

        Returns:
            A ``LocalFileRef`` pointing to the downloaded file.

        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError
