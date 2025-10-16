"""Utilities for splitting documents into semantically sized chunks."""

from __future__ import annotations

from typing import Iterable, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .types import Chunk, RawDocument


class TextSplitter:
    """LangChain-powered recursive character splitter."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap must be in [0, chunk_size)")
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", "。", "；", " ", ""],
        )

    def split_documents(self, documents: Iterable[RawDocument]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for doc in documents:
            pieces = self._splitter.split_text(doc.text)
            for idx, text in enumerate(pieces, start=1):
                chunk_id = f"{doc.source}:{doc.page}:{idx}"
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        source=doc.source,
                        page=doc.page,
                        text=text,
                    )
                )
        return chunks
