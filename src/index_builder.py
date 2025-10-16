"""Build and persist FAISS indexes using LangChain vector stores."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .embedder import BGEEmbedder


class IndexBuilder:
    """Create FAISS indexes for summaries and chunks via LangChain."""

    def __init__(self, embedder: BGEEmbedder) -> None:
        self.embedder = embedder

    def build_document_index(self, documents: Iterable[Document], output_dir: Path) -> None:
        self._build_index(documents, output_dir)

    def build_chunk_index(self, documents: Iterable[Document], output_dir: Path) -> None:
        self._build_index(documents, output_dir)

    def _build_index(self, documents: Iterable[Document], output_dir: Path) -> None:
        docs = list(documents)
        if not docs:
            raise ValueError("No documents provided for indexing.")
        output_dir.mkdir(parents=True, exist_ok=True)
        store = FAISS.from_documents(docs, self.embedder.embeddings)
        store.save_local(str(output_dir))
