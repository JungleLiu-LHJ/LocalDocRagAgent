"""Command-line entry point for the Local PDF RAG Assistant."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List

from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from .embedder import BGEEmbedder
from .hierarchical_retriever import HierarchicalRetriever
from .index_builder import IndexBuilder
from .pdf_loader import DocumentLoader
from .summarizer import DocumentSummarizer
from .text_splitter import TextSplitter
from .types import Chunk, RawDocument, SummaryRecord


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local PDF RAG Assistant")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Build summaries, embeddings, and FAISS indexes.")
    prepare.add_argument("--data-dir", type=Path, default=Path("data/pdfs"), help="Directory containing source documents.")
    prepare.add_argument("--summary-dir", type=Path, default=Path("data/summaries"), help="Directory to store document summaries.")
    prepare.add_argument("--vector-dir", type=Path, default=Path("data/vector_index"), help="Directory to store FAISS indexes.")
    prepare.add_argument("--chunk-size", type=int, default=1000, help="Chunk size in characters.")
    prepare.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap in characters.")
    prepare.add_argument("--summary-model", type=str, default="facebook/bart-large-cnn", help="Transformers model for summarization.")
    prepare.add_argument("--summary-device", type=str, default="cpu", help="Device for the summarization pipeline.")

    ask = subparsers.add_parser("ask", help="Ask a question against the prepared knowledge base.")
    ask.add_argument("question", type=str, help="Question to ask.")
    ask.add_argument("--vector-dir", type=Path, default=Path("data/vector_index"), help="Directory that stores FAISS indexes.")
    ask.add_argument("--doc-top-k", type=int, default=3, help="Number of documents to consider.")
    ask.add_argument("--chunk-top-k", type=int, default=5, help="Number of chunks per document.")
    ask.add_argument("--model", type=str, default="deepseek-chat", help="OpenAI-compatible chat model identifier.")
    ask.add_argument("--base-url", type=str, default=None, help="API base URL (defaults to DeepSeek).")
    ask.add_argument("--temperature", type=float, default=0.1, help="Generation temperature.")
    ask.add_argument("--max-tokens", type=int, default=512, help="Maximum new tokens for the answer.")
    ask.add_argument("--api-key", type=str, default=None, help="API key for the chat model (defaults to DEEPSEEK_API_KEY).")

    return parser.parse_args(argv)


def run_prepare(args: argparse.Namespace) -> None:
    data_dir: Path = args.data_dir
    summary_dir: Path = args.summary_dir
    vector_dir: Path = args.vector_dir

    documents = _load_documents(data_dir)
    if not documents:
        print(f"No documents found in {data_dir}.", file=sys.stderr)
        sys.exit(1)

    splitter = TextSplitter(chunk_size=args.chunk_size, overlap=args.chunk_overlap)
    chunks = splitter.split_documents(documents)

    summarizer = DocumentSummarizer(model_name=args.summary_model, device=args.summary_device)
    summaries = summarizer.summarize(documents)
    summarizer.save_summaries(summaries, summary_dir)

    embedder = BGEEmbedder()
    builder = IndexBuilder(embedder)

    doc_documents = _summary_documents(summaries)
    chunk_documents = _chunk_documents(chunks)
    builder.build_document_index(doc_documents, vector_dir / "doc_index")
    builder.build_chunk_index(chunk_documents, vector_dir / "chunk_index")

    report = {
        "documents": len(summaries),
        "chunks": len(chunks),
        "vector_dimension": int(embedder.embed_queries(["dimension probe"]).shape[-1]),
        "doc_index": str((vector_dir / "doc_index").resolve()),
        "chunk_index": str((vector_dir / "chunk_index").resolve()),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


def run_ask(args: argparse.Namespace) -> None:
    embedder = BGEEmbedder()
    vector_dir: Path = args.vector_dir
    doc_index = vector_dir / "doc_index"
    chunk_index = vector_dir / "chunk_index"

    if not doc_index.exists() or not (doc_index / "index.faiss").exists():
        raise FileNotFoundError(f"Document index not found at {doc_index}")
    if not chunk_index.exists() or not (chunk_index / "index.faiss").exists():
        raise FileNotFoundError(f"Chunk index not found at {chunk_index}")

    from langchain_community.vectorstores import FAISS

    doc_store = FAISS.load_local(str(doc_index), embedder.embeddings, allow_dangerous_deserialization=True)
    chunk_store = FAISS.load_local(str(chunk_index), embedder.embeddings, allow_dangerous_deserialization=True)

    base_url = args.base_url or "https://api.deepseek.com"
    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("API key required. Provide --api-key or set DEEPSEEK_API_KEY.")

    retriever = HierarchicalRetriever(
        document_store=doc_store,
        chunk_store=chunk_store,
        doc_top_k=args.doc_top_k,
        chunk_top_k=args.chunk_top_k,
    )
    llm = ChatOpenAI(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        api_key=api_key,
        base_url=base_url.rstrip("/"),
    )
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a helpful bilingual (English and Chinese) documentation assistant. "
            "Use the numbered context blocks to answer the question. "
            "Always cite evidence using the corresponding [number] markers. "
            "If the information is unavailable, say you cannot find it.\n\n"
            "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        ),
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    result = qa.invoke({"query": args.question})
    answer = result["result"].strip()
    source_docs = result.get("source_documents", [])
    citations: List[str] = []
    for doc in source_docs:
        citation = doc.metadata.get("citation")
        if citation and citation not in citations:
            citations.append(citation)

    print("Answer:\n")
    print(answer)
    print("\nCitations:")
    if citations:
        for citation in citations:
            print(f"- {citation}")
    else:
        print("- None returned")

    try:
        output_path = _write_markdown_answer(args.question, answer, citations)
        print(f"\nSaved markdown to {output_path}")
    except OSError as exc:
        print(f"\nFailed to write markdown file: {exc}", file=sys.stderr)


def _load_documents(data_dir: Path) -> List[RawDocument]:
    loader = DocumentLoader(data_dir)
    return loader.load()


def _summary_documents(summaries: List[SummaryRecord]) -> List[Document]:
    docs: List[Document] = []
    for record in summaries:
        summary_text = record.summary.strip()
        if record.keywords:
            summary_text = f"{summary_text}\nKeywords: {', '.join(record.keywords)}"
        docs.append(
            Document(
                page_content=summary_text,
                metadata={
                    "source": record.source,
                    "keywords": record.keywords,
                },
            )
        )
    return docs


def _chunk_documents(chunks: List[Chunk]) -> List[Document]:
    docs: List[Document] = []
    for chunk in chunks:
        docs.append(
            Document(
                page_content=chunk.text,
                metadata={
                    "source": chunk.source,
                    "page": chunk.page,
                    "id": chunk.id,
                },
            )
        )
    return docs


def _write_markdown_answer(question: str, answer: str, citations: List[str]) -> Path:
    filename = _sanitize_question_filename(question)
    output_path = Path.cwd() / filename

    lines = [
        f"# {question.strip() or 'Question'}",
        "",
        "## Answer",
        "",
        answer,
        "",
        "## Citations",
        "",
    ]
    if citations:
        lines.extend(f"- {citation}" for citation in citations)
    else:
        lines.append("- None returned")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _sanitize_question_filename(question: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", question.strip())
    sanitized = sanitized.strip("_")
    if not sanitized:
        sanitized = "question"
    return f"{sanitized}.md"


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    if args.command == "prepare":
        run_prepare(args)
    elif args.command == "ask":
        run_ask(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
