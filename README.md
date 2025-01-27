# embs

[![PyPI](https://img.shields.io/pypi/v/embs.svg?style=flat-square)](https://pypi.org/project/embs/)
[![License](https://img.shields.io/pypi/l/embs.svg?style=flat-square)](https://pypi.org/project/embs/)
[![Downloads](https://img.shields.io/pypi/dm/embs.svg?style=flat-square)](https://pypi.org/project/embs/)

**embs** is a Python toolkit that combines:

- **Document retrieval** via [Docsifer](https://lamhieu-docsifer.hf.space)
- **Embeddings** generation with [Lightweight Embeddings API](https://lamhieu-lightweight-embeddings.hf.space)
- **Text ranking** (reranking) based on query relevance
- **Optional caching** (in-memory LRU or disk) for performance and scalability

It provides both **asynchronous** (`asyncio`) and **synchronous** methods. If you need to ingest documents (files, URLs), convert them to text/markdown, embed them, and then sort by relevance, **embs** simplifies these tasks in a single package.

> **Note**: This library references external services for Docsifer (for document conversion) and Lightweight Embeddings (for text and image embeddings). Ensure you have valid endpoints or deploy your own versions.

## Why Use embs?

- **Unified Pipeline**: Manage document retrieval and text conversion from PDF, URLs, images, etc., then generate embeddings and rank them—all with a single API.
- **Async + Sync**: Choose the style that fits your application. The library uses **`aiohttp`** internally but also offers synchronous wrappers via **`asyncio.run()`**.
- **Caching**: Supports in-memory LRU or disk-based caching with optional time-to-live (TTL) eviction to avoid repeated network calls and save resources.
- **Lightweight**: No heavy dependencies besides `aiohttp` for async requests. Minimal overhead.

## Installation

Install **embs** via [pip](https://pypi.org/project/embs/):

```bash
pip install embs
```

Or add it to your `pyproject.toml` dependencies (if using Poetry):

```toml
[tool.poetry.dependencies]
embs = "^0.3.1"
```

## Quick Start

### 1. Basic Document Retrieval

```python
import asyncio
from embs import Embs

async def main():
    client = Embs()
    documents = await client.retrieve_documents_async(
        files=["/path/to/local/file.pdf"],
        urls=["https://example.com"]
    )
    print(documents)
    # => [{"filename": "file.pdf", "markdown": "...converted text..."}, {"filename": "example.com", "markdown": "..."}]

asyncio.run(main())
```

### 2. Generate Embeddings

```python
import asyncio
from embs import Embs

async def main():
    client = Embs()
    embedding_result = await client.embed_async("Hello world")
    print(embedding_result)
    # => {"object": "list", "data": [{"object": "embedding", "index": 0, "embedding": [...] }], "model": "...", "usage": {...}}

asyncio.run(main())
```

### 3. Rank Documents

```python
import asyncio
from embs import Embs

async def main():
    client = Embs()
    ranked = await client.rank_async("What is AI?", ["AI is about learning", "AI stands for artificial intelligence"])
    print(ranked)
    # => [{"text": "...", "probability": 0.9, "cosine_similarity": 0.85}, ...]

asyncio.run(main())
```

### 4. Integrated Workflow: `search_documents_async`

```python
import asyncio
from embs import Embs

async def main():
    client = Embs()
    results = await client.search_documents_async(
        query="Explain quantum computing",
        files=["/path/to/local/quantum.pdf"],
        urls=["https://example.com/quantum.html"]
    )
    for item in results:
        print(item["filename"], item["probability"], item["markdown"][:100])  # partial content

asyncio.run(main())
```

## Using the Cache

Enable caching by specifying a `cache_config`:

```python
from embs import Embs

cache_conf = {
    "enabled": True,
    "type": "memory",        # "memory" or "disk"
    "prefix": "myapp",       # optional prefix for cache keys
    "dir": "cache_folder",   # only needed if type="disk"
    "max_mem_items": 128,    # max items for LRU in memory
    "max_ttl_seconds": 86400 # 1-day TTL
}

client = Embs(cache_config=cache_conf)
```

- **Memory Caching**: Uses an **LRU** approach; older items are removed once it exceeds `max_mem_items`.
- **Disk Caching**: Stores `.json` files with a timestamp. Items older than `max_ttl_seconds` are discarded upon next read.

## Testing

`embs` includes test suites that rely on **`pytest`** and **`pytest-asyncio`** to verify:

- Document retrieval with Docsifer
- Embeddings calls with Lightweight Embeddings
- Ranking results
- Caching behaviors (in-memory and on disk)

Run tests with:

```bash
pytest --asyncio-mode=auto
```

## License

This project is licensed under the **MIT License**. See [LICENSE](./LICENSE) for details.

**Happy Embedding & Ranking!**
