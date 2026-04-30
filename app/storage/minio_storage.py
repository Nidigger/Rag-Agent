"""MinIO object storage backend."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from minio import Minio
from minio.error import S3Error

from app.config import settings
from app.storage.base import LocalFileRef, ObjectStorage

logger = logging.getLogger("rag-agent.minio_storage")


class MinioStorage(ObjectStorage):
    """Download objects from a MinIO bucket to local temporary files."""

    def __init__(self) -> None:
        endpoint = self._normalize_endpoint(settings.storage.minio_endpoint)
        access_key = settings.storage.minio_access_key.get_secret_value()
        secret_key = settings.storage.minio_secret_key.get_secret_value()
        missing = []
        if not endpoint:
            missing.append("MINIO_ENDPOINT")
        if not access_key:
            missing.append("MINIO_ACCESS_KEY")
        if not secret_key:
            missing.append("MINIO_SECRET_KEY")
        if missing:
            raise RuntimeError(
                "MinIO storage is not configured; missing "
                + ", ".join(missing)
            )

        self._client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=settings.storage.minio_secure,
        )

    @staticmethod
    def _normalize_endpoint(endpoint: str) -> str:
        return endpoint.strip().replace("http://", "").replace("https://", "")

    def download_to_temp(self, bucket: str, object_key: str) -> LocalFileRef:
        """Download an object from MinIO to a temporary file.

        The temporary file preserves the original file extension so that
        downstream loaders (txt_loader / pdf_loader) can detect the file type.

        Args:
            bucket: MinIO bucket name.
            object_key: Full object key within the bucket.

        Returns:
            A ``LocalFileRef`` for the downloaded file.

        Raises:
            RuntimeError: If the download fails for any reason.
        """
        if not bucket:
            raise RuntimeError(
                "MinIO bucket is not configured; pass storage.bucket "
                "or set MINIO_BUCKET"
            )
        if not object_key:
            raise RuntimeError("MinIO object_key must not be empty")

        suffix = Path(object_key).suffix
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.close()

        try:
            self._client.fget_object(bucket, object_key, tmp.name)
        except S3Error as exc:
            Path(tmp.name).unlink(missing_ok=True)
            logger.error(
                "[minio_storage] Failed to download %s/%s: %s",
                bucket,
                object_key,
                exc,
            )
            raise RuntimeError(
                f"MinIO download failed for {bucket}/{object_key}: {exc}"
            ) from exc
        except Exception as exc:
            Path(tmp.name).unlink(missing_ok=True)
            logger.error(
                "[minio_storage] Unexpected error downloading %s/%s: %s",
                bucket,
                object_key,
                exc,
            )
            raise RuntimeError(
                f"MinIO download failed for {bucket}/{object_key}: {exc}"
            ) from exc

        logger.info(
            "[minio_storage] Downloaded %s/%s to %s",
            bucket,
            object_key,
            tmp.name,
        )

        return LocalFileRef(
            path=tmp.name,
            filename=Path(object_key).name,
            should_cleanup=True,
        )
