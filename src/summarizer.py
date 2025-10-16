"""Generate summaries and keywords for documents using a local model."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from transformers import pipeline

from .types import RawDocument, SummaryRecord


class DocumentSummarizer:
    """Summarize documents with a huggingface transformer pipeline."""

    def __init__(
        self,
        model_name: str = "facebook/bart-large-cnn",
        device: Optional[int] = None,
        max_input_tokens: int = 900,
        token_overlap: int = 120,
        summary_max_length: int = 200,
        summary_min_length: int = 32,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_input_tokens = max_input_tokens
        self.token_overlap = token_overlap
        self.summary_max_length = summary_max_length
        self.summary_min_length = summary_min_length
        pipeline_kwargs = {"model": model_name}
        if device is not None:
            pipeline_kwargs["device"] = device
        self._pipeline = pipeline("summarization", **pipeline_kwargs)
        self._tokenizer = self._pipeline.tokenizer
        model_limit = getattr(self._tokenizer, "model_max_length", None)
        if model_limit is None or model_limit > 1000000:
            model_limit = 1024
        self._max_input_tokens = min(self.max_input_tokens, model_limit - 2)
        if self._max_input_tokens <= 0:
            self._max_input_tokens = max(model_limit - 2, 512)
        self._token_overlap = max(0, min(self.token_overlap, self._max_input_tokens // 2))

    def summarize(self, documents: Iterable[RawDocument]) -> List[SummaryRecord]:
        grouped = _group_documents(documents)
        summaries: List[SummaryRecord] = []
        for source, text in grouped.items():
            summary_text = self._summarize_text(text)
            keywords = _extract_keywords(text)
            summaries.append(
                SummaryRecord(
                    source=source,
                    summary=summary_text,
                    keywords=keywords,
                )
            )
        return summaries

    def save_summaries(self, summaries: Iterable[SummaryRecord], output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for record in summaries:
            out_path = output_dir / f"{Path(record.source).stem}.json"
            out_path.write_text(
                _summary_to_json(record),
                encoding="utf-8",
            )

    def _summarize_text(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        chunks = self._chunk_text(text)
        partials = [
            self._run_pipeline(chunk) for chunk in chunks
        ]
        if not partials:
            return ""
        if len(partials) == 1:
            return partials[0]
        merged = " ".join(partials)
        return self._run_pipeline(merged)

    def _run_pipeline(self, text: str) -> str:
        result = self._pipeline(
            text,
            max_length=self.summary_max_length,
            min_length=self.summary_min_length,
            truncation=True,
            do_sample=False,
        )
        return result[0]["summary_text"].strip()

    def _chunk_text(self, text: str) -> List[str]:
        input_ids = self._tokenizer.encode(text, add_special_tokens=False)
        if len(input_ids) <= self._max_input_tokens:
            return [text]

        chunks: List[str] = []
        stride = max(1, self._max_input_tokens - self._token_overlap)
        for start in range(0, len(input_ids), stride):
            end = min(start + self._max_input_tokens, len(input_ids))
            chunk_ids = input_ids[start:end]
            chunk_text = self._tokenizer.decode(chunk_ids, skip_special_tokens=True)
            chunk_text = chunk_text.strip()
            if chunk_text:
                chunks.append(chunk_text)
            if end == len(input_ids):
                break
        return chunks


def _group_documents(documents: Iterable[RawDocument]) -> Dict[str, str]:
    grouped: Dict[str, List[str]] = defaultdict(list)
    for doc in documents:
        grouped[doc.source].append(doc.text)
    return {source: "\n".join(parts) for source, parts in grouped.items()}


STOPWORDS = {
    "the",
    "and",
    "is",
    "in",
    "on",
    "for",
    "to",
    "a",
    "an",
    "with",
    "of",
    "by",
    "that",
    "this",
    "from",
    "at",
    "are",
}


def _extract_keywords(text: str, limit: int = 8) -> List[str]:
    from collections import Counter

    words = Counter()
    ascii_words = re.findall(r"[A-Za-z]{2,}", text.lower())
    for word in ascii_words:
        if word not in STOPWORDS:
            words[word] += 1
    # Join contiguous CJK characters to keep word level granularity.
    cjk_tokens = re.findall(r"[\u4e00-\u9fff]+", text)
    for token in cjk_tokens:
        words[token] += 1
    return [token for token, _ in words.most_common(limit)]


def _summary_to_json(record: SummaryRecord) -> str:
    import json

    payload = {
        "source": record.source,
        "summary": record.summary,
        "keywords": record.keywords,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
