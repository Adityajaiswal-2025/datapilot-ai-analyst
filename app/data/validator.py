import os
from app.core.config import settings


class DatasetValidationError(Exception):
    """Custom exception raised when dataset file validation fails."""
    pass


class DatasetSizeLimitError(DatasetValidationError):
    """Custom exception raised when upload size exceeds maximum allowed limit."""
    pass


def validate_filename_safety(filename: str) -> str:
    """Sanitizes user-provided filename and guards against path traversal."""
    if not filename or not filename.strip():
        raise DatasetValidationError("Invalid empty filename provided.")

    # Strip directory components to prevent path traversal
    safe_name = os.path.basename(filename.strip().replace("\\", "/"))

    if "\x00" in safe_name or ".." in safe_name:
        raise DatasetValidationError("Filename contains invalid or dangerous characters.")

    return safe_name


def validate_file_extension(filename: str) -> str:
    """Validates that the file extension is allowed (.csv or .xlsx)."""
    safe_name = validate_filename_safety(filename)
    if "." not in safe_name:
        raise DatasetValidationError("Filename must include a valid extension (.csv or .xlsx).")

    ext = os.path.splitext(safe_name)[1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        allowed = ", ".join(settings.ALLOWED_EXTENSIONS)
        raise DatasetValidationError(
            f"Unsupported file format '{ext}'. Allowed extensions are: {allowed}."
        )
    return ext


def validate_file_size(file_size_bytes: int) -> None:
    """Validates that file size is non-empty and within configurable limit."""
    if file_size_bytes <= 0:
        raise DatasetValidationError("Uploaded file is empty (0 bytes).")

    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if file_size_bytes > max_bytes:
        raise DatasetSizeLimitError(
            f"File size ({file_size_bytes / (1024 * 1024):.2f} MB) exceeds maximum allowed limit of {settings.MAX_UPLOAD_SIZE_MB} MB."
        )


def validate_dataset_file(filename: str, file_size_bytes: int) -> str:
    """Combines extension and size checks for dataset files."""
    ext = validate_file_extension(filename)
    validate_file_size(file_size_bytes)
    return ext

