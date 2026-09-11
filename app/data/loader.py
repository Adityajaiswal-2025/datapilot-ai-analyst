import os
import pandas as pd
from app.data.validator import DatasetValidationError


class DatasetLoadError(Exception):
    """Custom exception raised when dataset parsing or reading fails."""
    pass


def load_dataset(file_path: str) -> pd.DataFrame:
    """Reads a CSV or XLSX file from disk into a Pandas DataFrame.

    Supports encoding fallbacks for CSV files (utf-8, latin-1, cp1252).
    """
    if not os.path.exists(file_path):
        raise DatasetLoadError(f"File not found at path: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        encodings = ["utf-8", "latin-1", "cp1252"]
        last_error = None
        for enc in encodings:
            try:
                df = pd.read_csv(file_path, encoding=enc)
                if df.empty:
                    raise DatasetValidationError("Dataset CSV file contains no rows.")
                return df
            except (UnicodeDecodeError, pd.errors.EmptyDataError) as e:
                last_error = e
                continue
            except Exception as e:
                raise DatasetLoadError(f"Failed to parse CSV file: {str(e)}") from e

        raise DatasetLoadError(
            f"Could not decode CSV file with standard encodings: {str(last_error)}"
        )

    elif ext == ".xlsx":
        try:
            df = pd.read_excel(file_path, engine="openpyxl")
            if df.empty:
                raise DatasetValidationError("Dataset Excel file contains no rows.")
            return df
        except Exception as e:
            raise DatasetLoadError(f"Failed to parse Excel file: {str(e)}") from e

    else:
        raise DatasetValidationError(f"Unsupported extension for loading: {ext}")
