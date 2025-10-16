"""Embedding utilities built on top of LangChain BGE embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


@dataclass
class EmbedderConfig:
    model_name: str = "BAAI/bge-m3"
    device: str = "cpu"
    cache_folder: str | None = None
    normalize: bool = True


class BGEEmbedder:
    """Encodes documents and queries using LangChain's HuggingFaceBgeEmbeddings."""

    def __init__(self, config: EmbedderConfig | None = None) -> None:
        config = config or EmbedderConfig()
        model_kwargs = {"device": config.device}
        encode_kwargs = {"normalize_embeddings": config.normalize}
        if config.cache_folder:
            model_kwargs["cache_folder"] = config.cache_folder
        self._embedder = HuggingFaceBgeEmbeddings(
            model_name=config.model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors = self._embedder.embed_documents(list(texts))
        return np.asarray(vectors, dtype=np.float32)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors = [self._embedder.embed_query(text) for text in texts]
        return np.asarray(vectors, dtype=np.float32)

    @property
    def embeddings(self) -> HuggingFaceBgeEmbeddings:
        return self._embedder
