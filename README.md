# Doc RAG Assistant

[简体中文 README](README.zh-CN.md)

Build a local, privacy‑friendly RAG (Retrieval‑Augmented Generation) assistant over your PDFs, Markdown, and TXT files. The pipeline prepares summaries and vector indexes locally, then answers questions with LLM.

## Features

- Local preprocessing: parsing, chunking, summarization, embeddings
- Dual FAISS indexes: document‑level and chunk‑level
- Hierarchical retrieval: filter by document, then pick top chunks
- Bilingual prompts (English + Chinese) with explicit citations

## Advantages

- Higher precision: first retrieve likely documents via summaries, then restrict chunk search to those sources, reducing topic drift and high‑score noise.
- Balanced evidence: per‑document top‑K caps prevent a single long file from crowding out others, improving coverage and consensus.
- Smaller, cleaner context: fewer but more relevant tokens go to the LLM, lowering latency/cost and improving answer focus.
- Better citations: each chunk is numbered and includes source + page, making answers easy to audit and reproduce.
- Safer inputs: fewer unrelated chunks means lower risk of prompt contamination from off‑topic passages.
- Adaptive width: we over‑sample chunk candidates and early‑stop once each selected document has K good chunks.

Flow:

1) Query `doc_index` with the question → get top‑N sources.
2) Query `chunk_index` with a larger K, keep chunks only from those sources, max K per source.
3) Number chunks, build the context block, send to the LLM.

## Requirements

- Python 3.9+
- macOS or Linux recommended (Apple Silicon supported)
- Dependencies: see `requirements.txt`

Optional accelerators:

- Summarization can use `--summary-device mps` on Apple Silicon.
- Embeddings run on CPU by default (BAAI/bge‑m3 via LangChain).

## Install

```bash
# 1) Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2) Install dependencies
pip install -U pip
pip install -r requirements.txt
```

If PyTorch wheels are slow to resolve on Apple Silicon, consult https://pytorch.org for the recommended install command for your platform, then install the rest of the requirements.

## Project Layout

```
./
├── data/
│   ├── pdfs/           # put your source PDFs/MD/TXT here
│   ├── summaries/      # generated document summaries (JSON)
│   └── vector_index/   # generated FAISS indexes
├── src/
│   ├── main.py                 # CLI entry (prepare / ask)
│   ├── pdf_loader.py           # load PDFs/MD/TXT
│   ├── text_splitter.py        # semantic chunking
│   ├── summarizer.py           # HF transformers summarization
│   ├── embedder.py             # BAAI/bge-m3 embeddings
│   ├── index_builder.py        # build FAISS indexes
│   ├── hierarchical_retriever.py # document→chunk hierarchical retrieval
│   └── types.py
├── requirements.txt
└── agent.md                    # architecture notes
```

## Quickstart

1) Add documents

- Put your `.pdf`, `.md`, `.markdown`, and `.txt` files under `data/pdfs/` (subfolders OK).

2) Prepare knowledge base (summaries + indexes)

```bash
# From the repo root
python -m src.main prepare \
  --data-dir data/pdfs \
  --summary-dir data/summaries \
  --vector-dir data/vector_index \
  --chunk-size 1000 \
  --chunk-overlap 200 \
  --summary-model facebook/bart-large-cnn \
  --summary-device cpu   # use 'mps' on Apple Silicon if available
```

This step will:

- Load documents and split into chunks
- Summarize each document and extract keywords
- Build two FAISS indexes under `data/vector_index/{doc_index,chunk_index}`

3) Ask questions

By default the chat model is `deepseek-chat` via `https://api.deepseek.com`.
Provide an API key via `--api-key` or `DEEPSEEK_API_KEY`.

```bash
export DEEPSEEK_API_KEY=sk-...   # or pass --api-key

python -m src.main ask \
  "What are the main findings in the Argus ClickHouse report?" \
  --vector-dir data/vector_index \
  --doc-top-k 3 \
  --chunk-top-k 5 \
  --model deepseek-chat \
  --temperature 0.1 \
  --max-tokens 512
```

Answers are printed to the console with a citation list and also saved as a Markdown file named after your question (sanitized), in the current working directory.

## Using a Different LLM Endpoint

`src/main.py` uses `langchain_openai.ChatOpenAI`, so any OpenAI‑compatible server works:

- `--base-url` sets the API base (default `https://api.deepseek.com`).
- `--model` sets the model name (e.g., `gpt-4o-mini`, `qwen2.5`, or your local server’s model ID).
- `--api-key` can be any token your server expects (or a dummy value if not required).

Examples:

```bash
# LM Studio (OpenAI compatible)
python -m src.main ask "Summarize the ClickHouse bottlenecks" \
  --base-url http://localhost:1234/v1 \
  --model your-local-model \
  --api-key lm-studio

# OpenAI API
env OPENAI_API_KEY=sk-... python -m src.main ask \
  "Key design principles" \
  --base-url https://api.openai.com/v1 \
  --model gpt-4o-mini \
  --api-key "$OPENAI_API_KEY"
```

## CLI Reference

```bash
python -m src.main --help

# Subcommands
prepare  Build summaries, embeddings, and FAISS indexes.
ask      Ask a question against the prepared knowledge base.
```

Key options:

- `prepare`: `--data-dir`, `--summary-dir`, `--vector-dir`, `--chunk-size`, `--chunk-overlap`, `--summary-model`, `--summary-device`
- `ask`: `question`, `--vector-dir`, `--doc-top-k`, `--chunk-top-k`, `--model`, `--base-url`, `--temperature`, `--max-tokens`, `--api-key`

## How It Works

- Loader: `pdf_loader.py` parses PDFs (page‑wise) and reads MD/TXT.
- Splitter: `text_splitter.py` creates overlapping chunks suited for retrieval.
- Summarizer: `summarizer.py` runs a local transformers pipeline to generate document‑level summaries and keywords.
- Embeddings: `embedder.py` uses `BAAI/bge-m3` via LangChain to vectorize text.
- Indexes: `index_builder.py` builds FAISS indexes for documents and chunks.
- Retrieval: `hierarchical_retriever.py` first scores documents, then selects top chunks per document.
- Answering: `src/main.py` assembles context and queries the chat model with a bilingual, citation‑aware prompt.

## Troubleshooting

- FAISS load errors: ensure `data/vector_index/doc_index` and `chunk_index` exist after `prepare`.
- Empty results: verify your `data/pdfs` contains parsable text; scanned PDFs may need OCR first.
- Transformers performance: try `--summary-device mps` on Apple Silicon, or choose a smaller model.
- API key: set `DEEPSEEK_API_KEY` or pass `--api-key`; use `--base-url` for non‑DeepSeek endpoints.

## Contributing

Issues and PRs are welcome. Keep changes modular and focused. See `agent.md` for architecture and future ideas.

## License

Choose a license before publishing (e.g., MIT, Apache‑2.0). If unsure, add a `LICENSE` file later; for now this repository ships without a license.
