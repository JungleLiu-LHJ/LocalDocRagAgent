# 本地 文档 RAG 助手

我在总结之前项目的技术文档的时候，想让LLM帮我总结一些细小的点，可是一股脑喂给LLM感觉效果很不好。所以做了这个项目。

在本地为 PDF、Markdown、TXT 文档构建检索增强生成（RAG）助手。管道在本地完成摘要与向量索引的构建，然后喂给LLM生成带引用的答案。

## 功能特点

- 本地预处理：解析、切分、摘要、向量化
- 双索引：文档级与分块（段落）级 FAISS 索引
- 分层检索：先筛文档，再挑选每个文档的最佳片段
- 双语提示（中英）与明确引用标注

## 本项目优点

- 精度更高：先用文档级摘要检索候选文档，再将分块搜索限定在这些来源，显著减少跑题与“高分噪声块”。
- 证据更均衡：对每个文档设置 Top‑K 上限，避免单一长文档垄断结果，提升覆盖与一致性。
- 上下文更小更干净：只把更相关的 tokens 交给 LLM，降低延迟/费用，答案更聚焦、更稳定。
- 引用可审计：每个片段带编号、来源与页码，回答可追溯、易复现。
- 更安全：减少无关片段，降低提示注入/污染风险。
- 自适应宽度：先广取候选，再在每个文档达到 K 条上限后提前停止。

流程：

1) 用问题查询 `doc_index` → 选出 Top‑N 文档来源。
2) 用问题查询 `chunk_index`（更大的 K），仅保留来自上述来源的片段，并对每个来源限 K 条。
3) 为片段编号，构造上下文块后交给 LLM。

## 运行环境

- Python 3.9+
- 建议 macOS 或 Linux（支持 Apple Silicon）
- 依赖见 `requirements.txt`

可选加速：

- 摘要阶段在 Apple Silicon 上可用 `--summary-device mps`。
- 向量化默认使用 CPU（BAAI/bge‑m3，经由 LangChain）。

## 安装步骤

```bash
# 1) 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 2) 安装依赖
pip install -U pip
pip install -r requirements.txt
```

如果在 Apple Silicon 上安装 PyTorch 较慢，请参考 https://pytorch.org 提供的官方安装指令，再安装其余依赖。

## 项目结构

```
./
├── data/
│   ├── pdfs/           # 放入源文档（PDF/MD/TXT）
│   ├── summaries/      # 生成的文档级摘要（JSON）
│   └── vector_index/   # 生成的 FAISS 索引
├── src/
│   ├── main.py                  # 命令行入口（prepare / ask）
│   ├── pdf_loader.py            # 读取 PDF/MD/TXT
│   ├── text_splitter.py         # 语义切分
│   ├── summarizer.py            # HF transformers 摘要
│   ├── embedder.py              # BAAI/bge-m3 向量化
│   ├── index_builder.py         # 构建 FAISS 索引
│   ├── hierarchical_retriever.py# 文档→分块的分层检索
│   └── types.py
├── requirements.txt
└── agent.md                     # 架构与设计说明
```

## 快速开始

1) 放入文档

- 将 `.pdf`、`.md`、`.markdown`、`.txt` 文件放到 `data/pdfs/`（支持子目录）。

2) 构建知识库（摘要 + 索引）

```bash
# 在仓库根目录执行
python -m src.main prepare \
  --data-dir data/pdfs \
  --summary-dir data/summaries \
  --vector-dir data/vector_index \
  --chunk-size 1000 \
  --chunk-overlap 200 \
  --summary-model facebook/bart-large-cnn \
  --summary-device cpu   # Apple 设备可尝试 'mps'
```

该步骤将：

- 读取文档并切分为片段
- 为每个文档生成摘要并抽取关键词
- 在 `data/vector_index/{doc_index,chunk_index}` 下生成两个 FAISS 索引

3) 提问

默认使用 `https://api.deepseek.com` 的 `deepseek-chat` 模型。
通过 `--api-key` 或环境变量 `DEEPSEEK_API_KEY` 提供密钥。

```bash
export DEEPSEEK_API_KEY=sk-...   # 或使用 --api-key 显式传入

python -m src.main ask \
  "Argus ClickHouse 报告的主要结论是什么？" \
  --vector-dir data/vector_index \
  --doc-top-k 3 \
  --chunk-top-k 5 \
  --model deepseek-chat \
  --temperature 0.1 \
  --max-tokens 512
```

程序会在终端输出答案与引用列表，并将答案按你的问题文本（清洗后）保存为同目录下的 Markdown 文件。

## 使用其他 LLM 服务

`src/main.py` 基于 `langchain_openai.ChatOpenAI`，可对接任意 OpenAI 兼容接口：

- `--base-url` 设置 API 基地址（默认 `https://api.deepseek.com`）。
- `--model` 设置模型名（如 `gpt-4o-mini`、`qwen2.5` 或你本地服务的模型 ID）。
- `--api-key` 传入该服务要求的 Token（若不需要，可用占位值）。

示例：

```bash
# LM Studio（OpenAI 兼容）
python -m src.main ask "概括 ClickHouse 性能瓶颈" \
  --base-url http://localhost:1234/v1 \
  --model your-local-model \
  --api-key lm-studio

# OpenAI API
env OPENAI_API_KEY=sk-... python -m src.main ask \
  "关键设计原则" \
  --base-url https://api.openai.com/v1 \
  --model gpt-4o-mini \
  --api-key "$OPENAI_API_KEY"
```

## 命令行参考

```bash
python -m src.main --help

# 子命令
prepare  构建摘要、向量与 FAISS 索引
ask      基于已构建的知识库进行问答
```

常用参数：

- `prepare`：`--data-dir`、`--summary-dir`、`--vector-dir`、`--chunk-size`、`--chunk-overlap`、`--summary-model`、`--summary-device`
- `ask`：`question`、`--vector-dir`、`--doc-top-k`、`--chunk-top-k`、`--model`、`--base-url`、`--temperature`、`--max-tokens`、`--api-key`

## 工作原理概述

- Loader：`pdf_loader.py` 逐页解析 PDF，并读取 MD/TXT。
- Splitter：`text_splitter.py` 生成带重叠的检索友好片段。
- Summarizer：`summarizer.py` 用本地 transformers 管道生成文档级摘要与关键词。
- Embeddings：`embedder.py` 使用 `BAAI/bge-m3` 向量化文本（经 LangChain）。
- Indexes：`index_builder.py` 为文档与片段分别构建 FAISS 索引。
- Retrieval：`hierarchical_retriever.py` 先筛文档再选片段，进行分层检索。
- Answering：`src/main.py` 组装上下文并用双语、带引用提示词进行问答。

## 故障排查

- FAISS 加载失败：确认执行过 `prepare` 且 `data/vector_index/doc_index`、`chunk_index` 存在。
- 结果为空：确认 `data/pdfs` 下是可解析文本；扫描版 PDF 可能需要 OCR 预处理。
- 摘要性能：Apple 设备尝试 `--summary-device mps`，或改用更小模型。
- API 密钥：设置 `DEEPSEEK_API_KEY` 或传 `--api-key`；自建/第三方服务请用 `--base-url` 指向其地址。

## 参与贡献

欢迎提交 Issue 和 PR。请保持修改模块化、聚焦。架构与后续规划见 `agent.md`。

## 许可协议

在发布到 GitHub 前请选择合适的开源许可证（如 MIT、Apache‑2.0）。目前仓库未附带许可证文件。
