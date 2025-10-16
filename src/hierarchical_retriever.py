"""LangChain retriever implementing hierarchical document → chunk search."""

from __future__ import annotations

from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, PrivateAttr




# - Purpose: Implements hierarchical retrieval. First filters candidate documents
#   at the document level, then returns top chunks (segments) per document.
#
# - Main attributes: `doc_top_k`, `chunk_top_k`; private vector stores
#   `_document_store` and `_chunk_store`, both FAISS vectorstores.
#
# - Key methods:
#   - `__init__`: accepts two FAISS stores and top_k parameters, and stores them.
#   - `_get_relevant_documents` / `_aget_relevant_documents`: synchronous and
#     asynchronous entry points that delegate to `_retrieve`.
#   - `_retrieve`: performs a similarity search on `_document_store` to get
#     candidate documents and extract their sources; then performs a broader
#     similarity search on `_chunk_store` and selects up to `chunk_top_k`
#     chunks per document. Constructs returned `Document` objects with a header
#     and metadata fields such as `score`, `citation`, and `rank`.
#
# - Helper: `_extract_source` reads the `source` field from `Document.metadata`.
#
# - Optional: Can be extended to change selection, scoring, or filtering strategies.
class HierarchicalRetriever(BaseRetriever):
    """Filters candidate documents first, then returns top chunks per document."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    doc_top_k: int = 5
    chunk_top_k: int = 10

    _document_store: FAISS = PrivateAttr()
    _chunk_store: FAISS = PrivateAttr()

    def __init__(
        self,
        document_store: FAISS,
        chunk_store: FAISS,
        doc_top_k: int = 3,
        chunk_top_k: int = 5,
    ) -> None:
        super().__init__(doc_top_k=doc_top_k, chunk_top_k=chunk_top_k)
        self._document_store = document_store
        self._chunk_store = chunk_store

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> List[Document]:
        return self._retrieve(query)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> List[Document]:
        return self._retrieve(query)

    def _retrieve(self, query: str) -> List[Document]:
        doc_hits = self._document_store.similarity_search_with_score(query, k=self.doc_top_k)
        doc_sources = {self._extract_source(doc) for doc, _ in doc_hits if self._extract_source(doc)}

        multiplier = max(len(doc_sources), 1)
        raw_hits = self._chunk_store.similarity_search_with_score(
            query,
            k=max(self.chunk_top_k * multiplier * 3, self.chunk_top_k),
        )
        selection: List[Tuple[Document, float]] = []
        per_doc_counts: dict[str, int] = {source: 0 for source in doc_sources}

        for doc, score in raw_hits:
            source = self._extract_source(doc)
            if doc_sources and source not in doc_sources:
                continue
            if source:
                current = per_doc_counts.get(source, 0)
                if current >= self.chunk_top_k:
                    continue
                per_doc_counts[source] = current + 1
            selection.append((doc, score))
            if doc_sources and all(count >= self.chunk_top_k for count in per_doc_counts.values()):
                break

        if not selection:
            selection = raw_hits[: self.chunk_top_k]

        numbered_docs: List[Document] = []
        for idx, (doc, score) in enumerate(selection, start=1):
            source = self._extract_source(doc) or "unknown"
            page = int(doc.metadata.get("page", 1))
            citation = f"{source}#page={page}"
            metadata = dict(doc.metadata)
            metadata.update(
                {
                    "score": float(score),
                    "citation": citation,
                    "rank": idx,
                }
            )
            header = f"[{idx}] {source} (page {page})"
            numbered_docs.append(
                Document(
                    page_content=f"{header}\n{doc.page_content.strip()}",
                    metadata=metadata,
                )
            )
        return numbered_docs

    @staticmethod
    def _extract_source(doc: Document) -> str | None:
        source = doc.metadata.get("source")
        return source if isinstance(source, str) else None
