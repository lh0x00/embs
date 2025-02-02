# embs

[![PyPI](https://img.shields.io/pypi/v/embs.svg?style=flat-square)](https://pypi.org/project/embs/)
[![License](https://img.shields.io/pypi/l/embs.svg?style=flat-square)](https://pypi.org/project/embs/)
[![Downloads](https://img.shields.io/pypi/dm/embs.svg?style=flat-square)](https://pypi.org/project/embs/)

**embs** is your **one-stop toolkit** for document ingestion, embedding, and ranking workflows.
Whether you are building a **retrieval-augmented generation (RAG) system**, a **chatbot**, or a
**semantic search engine**, **embs** makes it fast and simple to integrate document retrieval,
embedding, and ranking with minimal configuration.

## Why Choose embs?

- **Free External APIs**:
  - **Docsifer** for converting files/URLs (PDFs, HTML, images, etc.) to Markdown.
  - **Lightweight Embeddings API** for generating high-quality, multilingual embeddings.
- **Optimized for RAG & Chatbots**:  
  Automatically split documents into meaningful chunks, generate embeddings, and rank them by query relevance to empower your chatbot or generative model.

- **Flexible Splitting**:  
  Use the built-in Markdown splitter or provide a custom splitting function to best suit your documents.

- **Unified Pipeline**:  
  Seamlessly handle document ingestion, content extraction, embedding generation, and relevance ranking—all in one library.

- **DuckDuckGo-powered Web Search**:  
  The new `search_documents` function leverages DuckDuckGo to find relevant URLs by keyword, retrieves their content via Docsifer, and ranks the results.

- **Optional Embedding Results**:  
  Simply pass `options={"embeddings": True}` to receive the raw embedding vectors with your ranking results.

## Installation

Install via pip:

```bash
pip install embs
```

Or add to your `pyproject.toml` (for Poetry):

```toml
[tool.poetry.dependencies]
embs = "^0.1.0"
```

## Quick Start Examples

### 1. Query Documents (Ranking by Relevance)

This example shows how to retrieve documents (from a file, URL, or both), rank them by relevance to your query, and optionally include the embeddings.

```python
import asyncio
from functools import partial
from embs import Embs

# Configure the built-in Markdown splitter.
split_config = {
    "headers_to_split_on": [("#", "h1"), ("##", "h2"), ("###", "h3")],
    "return_each_line": False,
    "strip_headers": True,
}
md_splitter = partial(Embs.markdown_splitter, config=split_config)

client = Embs()

# Asynchronously retrieve and rank documents.
async def run_query():
    docs = await client.query_documents_async(
        query="Explain quantum computing",
        files=["/path/to/quantum_theory.pdf"],
        splitter=md_splitter,
        options={"embeddings": True}  # Include embeddings in each result.
    )
    for d in docs:
        print(f"{d['filename']} => Score: {d['probability']:.4f}")
        print(f"Snippet: {d['markdown'][:80]}...")
        if "embeddings" in d:
            print("Embeddings:", d["embeddings"])
        print()

asyncio.run(run_query())
```

For synchronous usage:

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

### 2. Search Documents via DuckDuckGo

Use DuckDuckGo to search for relevant URLs by keyword, then retrieve, split, and rank their content.

```python
import asyncio
from embs import Embs

client = Embs()

async def run_search():
    results = await client.search_documents_async(
        query="Latest advances in AI",
        limit=5,         # Maximum number of search results.
        blocklist=["youtube.com"],  # Optional: filter out certain domains.
        options={"embeddings": True}  # Include embeddings in the returned items.
    )
    for item in results:
        print(f"File: {item['filename']} | Score: {item['probability']:.4f}")
        print(f"Snippet: {item['markdown'][:80]}...\n")

asyncio.run(run_search())
```

For synchronous usage:

```python
results = client.search_documents(
    query="Latest advances in AI",
    limit=5,
    blocklist=["youtube.com"],
    options={"embeddings": True}
)
for item in results:
    print(f"File: {item['filename']} | Score: {item['probability']:.4f}")
```

## Caching for Performance

Enable caching to speed up repeated operations:

```python
cache_conf = {
    "enabled": True,
    "type": "memory",       # or "disk"
    "prefix": "myapp",
    "dir": "cache_folder",  # required only for disk caching
    "max_mem_items": 128,
    "max_ttl_seconds": 86400
}

client = Embs(cache_config=cache_conf)
```

## Testing

The library is tested using **pytest** and **pytest-asyncio**. To run the tests:

```bash
pytest --asyncio-mode=auto
```

## License

Licensed under the **MIT License**. See [LICENSE](./LICENSE) for details.

Contributions are welcome! Please submit issues, ideas, or pull requests.
