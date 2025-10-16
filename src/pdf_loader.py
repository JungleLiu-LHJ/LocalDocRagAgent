"""Load PDF, Markdown, and TXT documents into raw text representation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from pypdf import PdfReader

from .types import RawDocument


SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt"}


class DocumentLoader:
    """Loads documents from a directory into `RawDocument` entries."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Document directory not found: {self.data_dir}")

    def load(self) -> List[RawDocument]:
        documents: List[RawDocument] = []
        for path in sorted(self._iter_document_paths()):
            suffix = path.suffix.lower()
            if suffix == ".pdf":
                documents.extend(self._load_pdf(path))
            else:
                documents.append(self._load_text_like(path))
        return documents

    def _iter_document_paths(self) -> Iterable[Path]:
        for path in self.data_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield path

    def _load_pdf(self, path: Path) -> List[RawDocument]:
        reader = PdfReader(str(path))
        items: List[RawDocument] = []
        for idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                items.append(
                    RawDocument(
                        source=str(path.relative_to(self.data_dir)),
                        page=idx,
                        text=text,
                    )
                )
        return items

    def _load_text_like(self, path: Path) -> RawDocument:
        text = path.read_text(encoding="utf-8")
        return RawDocument(
            source=str(path.relative_to(self.data_dir)),
            page=1,
            text=text,
        )
