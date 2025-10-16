"""Common data structures for the PDF RAG assistant."""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class RawDocument:
    """Represents a raw document page or file prior to chunking."""

    source: str
    page: int
    text: str


@dataclass(frozen=True)
class Chunk:
    """Represents a chunk of text ready for embedding."""

    id: str
    source: str
    page: int
    text: str


@dataclass(frozen=True)
class SummaryRecord:
    """Stores the document level summary and keywords."""

    source: str
    summary: str
    keywords: List[str]

