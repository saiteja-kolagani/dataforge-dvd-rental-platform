
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceFile:
    """
    Represents a discovered source file, whether stored locally or in S3.
    """

    file_name: str
    table_name: str
    file_size_bytes: int

    # Local source information
    local_path: Path | None = None

    # S3 source information
    source_bucket: str | None = None
    source_key: str | None = None
