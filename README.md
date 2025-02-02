# embs

[![PyPI](https://img.shields.io/pypi/v/embs.svg?style=flat-square)](https://pypi.org/project/embs/)
[![License](https://img.shields.io/pypi/l/embs.svg?style=flat-square)](https://pypi.org/project/embs/)
[![Downloads](https://img.shields.io/pypi/dm/embs.svg?style=flat-square)](https://pypi.org/project/embs/)

**embs** is a powerful Python library for **document retrieval, embedding, and ranking**, making it easier to build **Retrieval-Augmented Generation (RAG) systems**, **chatbots**, and **semantic search engines**.

## Why Choose embs?

- **Web & Local Document Search**:

  - DuckDuckGo-powered **web search** retrieves and ranks relevant documents.
  - Supports **PDFs, Word, HTML, Markdown**, and more.

- **Optimized for RAG, Chatbots & Multilingual Search**:

  - **Automatic document chunking (Splitter) for improved retrieval accuracy.**
  - Rank documents **by relevance to a query**.
  - **Strong multilingual model support** for global applications.
    ✅ Supported multilingual models:
    - `snowflake-arctic-embed-l-v2.0`
    - `bge-m3`
    - `gte-multilingual-base`
    - `paraphrase-multilingual-MiniLM-L12-v2`
    - `paraphrase-multilingual-mpnet-base-v2`
    - `multilingual-e5-small`
    - `multilingual-e5-base`
    - `multilingual-e5-large`

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
embs = "^0.1.8"
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
    "return_each_line": True,
    "strip_headers": True,
    "split_on_double_newline": True,
}
md_splitter = partial(Embs.markdown_splitter, config=split_config)

client = Embs()

async def run_search():
    results = await client.search_documents_async(
        query="Latest AI research",
        limit=3,
        blocklist=["youtube.com"],  # Exclude unwanted domains
        splitter=md_splitter,  # Enable smart chunking
    )
    for item in results:
        print(f"File: {item['filename']} | Score: {item['similarity']:.4f}")
        print(f"Snippet: {item['markdown'][:80]}...\n")

asyncio.run(run_search())
```

For **synchronous usage**:

```python
results = client.search_documents(
    query="Latest AI research",
    limit=3,
    blocklist=["youtube.com"],
    splitter=md_splitter,  # Always use a splitter
    model="snowflake-arctic-embed-l-v2.0",
)
for item in results:
    print(f"File: {item['filename']} | Score: {item['similarity']:.4f}")
```

### 2️⃣ Multilingual Document Querying (Local & Online)

Retrieve and **rank multilingual documents from local files or URLs**.

```python
async def run_query():
    docs = await client.query_documents_async(
        query="Explique la mécanique quantique",  # French query
        files=["/path/to/quantum_theory.pdf"],
        urls=["https://example.com/quantum.html"],
        splitter=md_splitter,  # Chunking for better retrieval
    )
    for d in docs:
        print(f"{d['filename']} => Score: {d['similarity']:.4f}")
        print(f"Snippet: {d['markdown'][:80]}...\n")

asyncio.run(run_query())
```

For **synchronous usage**:

```python
docs = client.query_documents(
    query="Explique la mécanique quantique",
    files=["/path/to/quantum_theory.pdf"],
    splitter=md_splitter,
)
for d in docs:
    print(d["filename"], "=> Score:", d["similarity"])
```

💡 **Perfect for multilingual retrieval!** Whether you're searching documents in English, French, Spanish, German, or other supported languages, `embs` ensures optimal ranking and retrieval.

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
    limit=3,
    blocklist=["example.com"],
    splitter=md_splitter
)
```

### 🔹 `query_documents_async()`

**Retrieve, split, and rank local/online documents.**

```python
await client.query_documents_async(
    query="Climate change effects",
    files=["/path/to/report.pdf"],
    urls=["https://example.com"],
    splitter=md_splitter,
)
```

### 🔹 `embed_async()`

**Generate embeddings for texts with multilingual support.**  

```python
embeddings = await client.embed_async(
    ["Este es un ejemplo de texto.", "Ceci est un exemple de phrase."],
    optimized=True  # Process one at a time for better caching
)
```

### 🔹 `rank_async()`

**Rank candidate texts by similarity to a query.**

```python
ranked_results = await client.rank_async(
    query="Machine learning",
    candidates=["Deep learning is a subset of ML", "Quantum computing is unrelated"]
)
```

## 🔬 Testing

Run **pytest** and **pytest-asyncio** for automated testing:

```bash
pytest --asyncio-mode=auto
```

## 📝 Best Practices: Always Use a Splitter!

### ✅ How to Use the Built-in Markdown Splitter

```python
from functools import partial

split_config = {
    "headers_to_split_on": [("#", "h1"), ("##", "h2"), ("###", "h3")],
    "return_each_line": True,
    "strip_headers": True,
    "split_on_double_newline": True,
}

md_splitter = partial(Embs.markdown_splitter, config=split_config)

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

🚀 With enhanced **multilingual support**, `embs` is now even more powerful for global retrieval applications! 🌍
