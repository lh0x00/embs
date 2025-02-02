# embs

[![PyPI](https://img.shields.io/pypi/v/embs.svg?style=flat-square)](https://pypi.org/project/embs/)
[![License](https://img.shields.io/pypi/l/embs.svg?style=flat-square)](https://pypi.org/project/embs/)
[![Downloads](https://img.shields.io/pypi/dm/embs.svg?style=flat-square)](https://pypi.org/project/embs/)

**embs** is a powerful Python library for **document retrieval, embedding, and ranking**, making it easier to build **Retrieval-Augmented Generation (RAG) systems**, **chatbots**, and **semantic search engines**.

## Why Choose embs?

- **Web & Local Document Search**:

  - DuckDuckGo-powered **web search** retrieves and ranks relevant documents.
  - Supports **PDFs, Word, HTML, Markdown**, and more.

- **Optimized for RAG & Chatbots**:

  - **Automatic document chunking (Splitter) for improved retrieval accuracy.**
  - Rank documents **by relevance to a query**.

- **Fast & Efficient**:

  - **Cache support (in-memory & disk)** for faster queries.
  - **Flexible batch embedding with cache optimization**.

- **Scalable & Customizable**:
  - Works with **synchronous & asynchronous processing**.
  - Supports **custom splitting rules**.

## 🚀 Installation

Install via pip:

```bash
pip install embs
```

For Poetry users:

```toml
[tool.poetry.dependencies]
embs = "^0.1.7"
```

## 📖 Quick Start Guide

### 1️⃣ Searching Documents via DuckDuckGo (Recommended!)

Retrieve **relevant web pages**, **convert them to Markdown**, and **rank them using embeddings**.

> **🚀 Always use a splitter!**  
> Improves ranking, reduces redundancy, and ensures better retrieval.

```python
import asyncio
from functools import partial
from embs import Embs

# Configure a Markdown-based splitter
split_config = {
    "headers_to_split_on": [("#", "h1"), ("##", "h2"), ("###", "h3")],
    "return_each_line": False,
    "strip_headers": True,
}
md_splitter = partial(Embs.markdown_splitter, config=split_config)

client = Embs()

async def run_search():
    results = await client.search_documents_async(
        query="Latest AI research",
        limit=5,
        blocklist=["youtube.com"],  # Exclude unwanted domains
        splitter=md_splitter,  # Enable smart chunking
        options={"embeddings": True}
    )
    for item in results:
        print(f"File: {item['filename']} | Score: {item['probability']:.4f}")
        print(f"Snippet: {item['markdown'][:80]}...\n")

asyncio.run(run_search())
```

For **synchronous usage**:

```python
results = client.search_documents(
    query="Latest AI research",
    limit=5,
    blocklist=["youtube.com"],
    splitter=md_splitter,  # Always use a splitter
    options={"embeddings": True}
)
for item in results:
    print(f"File: {item['filename']} | Score: {item['probability']:.4f}")
```

### 2️⃣ Querying Local & Online Documents with Ranking

Retrieve and **rank documents from local files or URLs**.

```python
async def run_query():
    docs = await client.query_documents_async(
        query="Explain quantum computing",
        files=["/path/to/quantum_theory.pdf"],
        urls=["https://example.com/quantum.html"],
        splitter=md_splitter,  # Chunking for better retrieval
        options={"embeddings": True}
    )
    for d in docs:
        print(f"{d['filename']} => Score: {d['probability']:.4f}")
        print(f"Snippet: {d['markdown'][:80]}...\n")

asyncio.run(run_query())
```

For **synchronous usage**:

```python
docs = client.query_documents(
    query="Explain quantum computing",
    files=["/path/to/quantum_theory.pdf"],
    splitter=md_splitter,
    options={"embeddings": True}
)
for d in docs:
    print(d["filename"], "=> Score:", d["probability"])
```

## ⚡ Caching for Performance

Enable **in-memory** or **disk caching** to speed up repeated queries.

```python
cache_conf = {
    "enabled": True,
    "type": "memory",       # or "disk"
    "prefix": "myapp",
    "dir": "cache_folder",  # Required for disk caching
    "max_mem_items": 128,
    "max_ttl_seconds": 86400
}

client = Embs(cache_config=cache_conf)
```

## 🔍 Key Features & API Methods

### 🔹 `search_documents_async()`

**Search for documents via DuckDuckGo, retrieve, and rank them.**

```python
await client.search_documents_async(
    query="Recent AI breakthroughs",
    limit=5,
    blocklist=["example.com"],
    splitter=md_splitter
)
```

- `query`: Search term.
- `limit`: Number of DuckDuckGo results.
- `blocklist`: Exclude **unwanted domains**.
- `splitter`: Smart **chunking** for better ranking.

### 🔹 `query_documents_async()`

**Retrieve, split, and rank local/online documents.**

```python
await client.query_documents_async(
    query="Climate change effects",
    files=["/path/to/report.pdf"],
    urls=["https://example.com"],
    splitter=md_splitter,
    options={"embeddings": True}
)
```

- `query`: Search query.
- `files`: List of **file paths**.
- `urls`: List of **webpage URLs**.
- `splitter`: Function to **split** document chunks.
- `options`: Set `{"embeddings": True}` to include embeddings.

### 🔹 `embed_async()`

**Generate embeddings for texts.**  
By default, it processes one item at a time for **better cache efficiency**.

```python
embeddings = await client.embed_async(
    ["This is a test sentence.", "Another sentence."],
    optimized=True  # Process one at a time for better caching
)
```

- `text_or_texts`: Single **string** or **list of texts**.
- `optimized`: **`True`** = Process **one-by-one** (better cache).  
  **`False`** = Process in **batches of 4** (faster, but higher API load).

### 🔹 `rank_async()`

**Rank candidate texts by similarity to a query.**

```python
ranked_results = await client.rank_async(
    query="Machine learning",
    candidates=["Deep learning is a subset of ML", "Quantum computing is unrelated"]
)
```

- `query`: **Search query**.
- `candidates`: List of **text snippets** to rank.

Returns a **sorted list** of items with:

- `"probability"` (higher = more relevant)
- `"cosine_similarity"`

## 🔬 Testing

Run **pytest** and **pytest-asyncio** for automated testing:

```bash
pytest --asyncio-mode=auto
```

## 📝 Best Practices: Always Use a Splitter!

**Why use a splitter?**

- **Improves retrieval** by processing **smaller chunks** of text.
- **Reduces token usage** when embedding & ranking.
- **Faster performance** in RAG and chatbot applications.

### ✅ How to Use the Built-in Markdown Splitter

```python
from functools import partial

split_config = {
    "headers_to_split_on": [("#", "h1"), ("##", "h2"), ("###", "h3")],
    "return_each_line": False,
    "strip_headers": True,
}

md_splitter = partial(Embs.markdown_splitter, config=split_config)

# Use it when querying documents
docs = client.query_documents(
    query="Machine Learning Basics",
    files=["/path/to/ml_guide.pdf"],
    splitter=md_splitter
)
```

## 📜 License

Licensed under **MIT License**. See [LICENSE](./LICENSE) for details.

## 🤝 Contributing

Pull requests, issues, and discussions are welcome!

With this enhanced documentation, `embs` is now even **easier to use and more efficient! 🚀**
