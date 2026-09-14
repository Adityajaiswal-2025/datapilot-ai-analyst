import os
import shutil
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional, Any
from app.core.config import settings
from app.data.validator import DatasetSizeLimitError, validate_filename_safety, validate_file_extension


class StorageService(ABC):
    """Abstract Base Class for dataset file storage providers."""

    @abstractmethod
    async def save_stream(
        self, file_upload: Any, storage_key: str, max_bytes: int
    ) -> int:
        """Streams bytes to storage and enforces maximum bytes limit."""
        pass

    @abstractmethod
    def get_file_path(self, storage_key: str) -> str:
        """Returns physical file path or locator for dataset reading."""
        pass

    @abstractmethod
    def exists(self, storage_key: str) -> bool:
        """Checks if a storage key / file exists."""
        pass

    @abstractmethod
    def delete(self, storage_key: str) -> bool:
        """Deletes a stored file (idempotent)."""
        pass


class LocalStorageService(StorageService):
    """Local filesystem implementation of StorageService."""

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or settings.STORAGE_LOCAL_DIR
        os.makedirs(self.base_dir, exist_ok=True)

    def get_file_path(self, storage_key: str) -> str:
        """Returns resolved absolute or relative storage path on local filesystem."""
        if os.path.isabs(storage_key):
            return storage_key
        return os.path.join(self.base_dir, os.path.basename(storage_key))

    def exists(self, storage_key: str) -> bool:
        """Returns True if the file exists on the local filesystem."""
        path = self.get_file_path(storage_key)
        return os.path.exists(path)

    async def save_stream(
        self, file_upload: Any, storage_key: str, max_bytes: int
    ) -> int:
        """Streams uploaded file chunks to local disk, enforcing max_bytes limit."""
        target_path = self.get_file_path(storage_key)
        chunk_size = 1024 * 1024  # 1MB chunks
        bytes_read = 0

        try:
            with open(target_path, "wb") as buffer:
                while True:
                    chunk = await file_upload.read(chunk_size)
                    if not chunk:
                        break
                    bytes_read += len(chunk)
                    if bytes_read > max_bytes:
                        buffer.close()
                        self.delete(target_path)
                        raise DatasetSizeLimitError(
                            f"File size exceeds maximum allowed limit of {settings.MAX_UPLOAD_SIZE_MB} MB."
                        )
                    buffer.write(chunk)
        except Exception:
            self.delete(target_path)
            raise

        if bytes_read == 0:
            self.delete(target_path)
            raise ValueError("Uploaded file is empty (0 bytes).")

        return bytes_read

    def delete(self, storage_key: str) -> bool:
        """Safely removes a local file if it exists (idempotent)."""
        path = self.get_file_path(storage_key)
        if os.path.exists(path):
            try:
                os.remove(path)
                return True
            except OSError:
                return False
        return False


# Global default storage service instance
default_storage_service = LocalStorageService()
