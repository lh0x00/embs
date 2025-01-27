# embs

[![PyPI](https://img.shields.io/pypi/v/embs.svg?style=flat-square)](https://pypi.org/project/embs/)
[![License](https://img.shields.io/pypi/l/embs.svg?style=flat-square)](https://pypi.org/project/embs/)
[![Downloads](https://img.shields.io/pypi/dm/embs.svg?style=flat-square)](https://pypi.org/project/embs/)

**embs** is your **one-stop toolkit** for handling document ingestion, embedding, and ranking workflows. Whether you're building a **retrieval-augmented generation (RAG) system**, a **chatbot**, or a **semantic search engine**, `embs` makes it easy to integrate document retrieval, embeddings, and ranking with minimal setup.

## Why Choose embs?

- **Free External APIs**: 
  - **Docsifer** for document conversion (PDFs, URLs, images, etc.) and 
  - **Lightweight Embeddings API** for generating state-of-the-art embeddings, including **multi-language support** and some of the **best models for NLP tasks** — all provided **free of charge**.
  - These APIs support top-tier multilingual embeddings models like `sentence-transformers` and `OpenAI-compatible embeddings`, so you can achieve top-quality results with minimal configuration.

- **Perfect for RAG Systems**: Automatically convert and split documents into meaningful chunks, generate embeddings, and rank them — all tailored for retrieval-augmented workflows like OpenAI GPT or other generative models.

- **Integrates with Chatbots**: Preprocess, split, and embed your knowledge base to build conversational systems that respond with accurate and contextually relevant answers.

- **Flexible Splitting**: Use built-in or custom chunking strategies to split large documents into smaller, retrievable sections. This improves relevance in retrieval-based workflows.

- **Unified Pipeline**: Streamline everything from **document ingestion** to **semantic ranking** with a single API.

- **Lightweight & Extensible**: No heavy dependencies beyond `aiohttp`. Easily fits into your existing infrastructure.

## Installation

```bash
pip install embs
```

Or in `pyproject.toml` (Poetry):

```toml
[tool.poetry.dependencies]
embs = "^0.1.0"
```

## Key Use Cases

### Retrieval-Augmented Generation (RAG)

In RAG workflows, retrieved knowledge informs a generative model like GPT to produce accurate and relevant answers. `embs` simplifies this by:

1. **Converting raw documents (PDFs, URLs)** to clean text or markdown.
2. **Splitting documents** into retrievable chunks (e.g., by headers or lines).
3. **Embedding chunks** with powerful multilingual models.
4. **Ranking** the chunks for relevance to the query.

With caching enabled, repeated requests are even faster, ensuring scalability for real-world deployments.

## Code Practices

Below is an end-to-end example that **retrieves documents, applies the built-in Markdown splitter, generates embeddings, and ranks them by query relevance**. This showcases how `embs` works perfectly for chatbot or RAG pipelines.

### Example: Retrieve, Split, and Rank

```python
import asyncio
from functools import partial
from embs import Embs

async def main():
    # Markdown-based splitter configuration
    split_config = {
        "headers_to_split_on": [("#", "h1"), ("##", "h2"), ("###", "h3")],
        "return_each_line": False,  # Keep chunks as sections, not individual lines
        "strip_headers": True       # Remove header text from the chunks
    }
    md_splitter = partial(Embs.markdown_splitter, config=split_config)

    # Initialize the Embs client
    client = Embs()

    # Step 1: Retrieve documents and split them by Markdown headers
    raw_docs = await client.retrieve_documents_async(
        files=["/path/to/sample.pdf"],
        urls=["https://example.com"],
        splitter=md_splitter  # Apply built-in markdown splitter
    )
    print(f"Total chunks after splitting: {len(raw_docs)}")

    # Step 2: Rank the retrieved documents by relevance to a query
    results = await client.search_documents_async(
        query="Explain quantum computing",
        files=["/path/to/quantum_theory.pdf"],  # Additional files to retrieve and rank
        urls=["https://example.com/quantum.html"],
        splitter=md_splitter  # Apply splitter for additional sources
    )

    # Step 3: Output the top-ranked results
    for item in results[:3]:
        print(f"File: {item['filename']} | Score: {item['probability']:.4f}")
        print(f"Snippet: {item['markdown'][:80]}...\n")

asyncio.run(main())
```

## Why Is This Perfect for Chatbots?

1. **Context-Aware Answers**: By splitting large documents into manageable chunks and ranking them for relevance, your chatbot always responds with the most contextually appropriate snippet.
2. **Multilingual Embeddings**: The Lightweight Embeddings API supports embeddings for multiple languages, so your chatbot can handle diverse user inputs and knowledge bases.
3. **Caching for Scalability**: Repeated retrieval or ranking operations are sped up dramatically with in-memory or disk-based caching, ensuring low-latency responses.

## API Reference

Below are the primary methods in `embs`. All async methods have a synchronous equivalent.

### 1. `retrieve_documents_async` / `retrieve_documents`
Convert files and/or URLs into Markdown using Docsifer. Optionally, apply a splitter to break down large documents into chunks.

```python
async def retrieve_documents_async(
    files=None,
    urls=None,
    openai_config=None,
    settings=None,
    concurrency=5,
    options=None,
    splitter=None
) -> List[Dict[str, str]]:
    ...
```

- **Params**:
  - `files`: List of file paths or file-like objects.
  - `urls`: List of URLs for Docsifer to process.
  - `splitter`: A callable that receives and returns a list of docs, e.g., `Embs.markdown_splitter`.
- **Returns**: A list of documents (`{"filename": <str>, "markdown": <str>}`).

### 2. `embed_async` / `embed`
Generate embeddings for text or a list of texts using the Lightweight Embeddings API.

```python
async def embed_async(
    text_or_texts: Union[str, List[str]],
    model=None
) -> Dict[str, Any]:
    ...
```

- **Params**:
  - `text_or_texts`: Single string or list of strings to embed.
  - `model`: Optional; specify the embedding model (defaults to `snowflake-arctic-embed-l-v2.0`).
- **Returns**: Embedding data as a dictionary.

### 3. `rank_async` / `rank`
Rank a list of text candidates by relevance to a query using the Lightweight Embeddings API.

```python
async def rank_async(
    query: str,
    candidates: List[str],
    model=None
) -> List[Dict[str, Any]]:
    ...
```

- **Params**:
  - `query`: The query string.
  - `candidates`: List of candidate texts.
  - `model`: Optional; specify the ranking model (defaults to `snowflake-arctic-embed-l-v2.0`).
- **Returns**: A ranked list of `{"text": <candidate>, "probability": <float>, "cosine_similarity": <float>}`.

### 4. `search_documents_async` / `search_documents`
Retrieve documents (files/URLs), optionally split them, and rank their chunks by relevance to a query.

```python
async def search_documents_async(
    query: str,
    files=None,
    urls=None,
    openai_config=None,
    settings=None,
    concurrency=5,
    options=None,
    model=None,
    splitter=None
) -> List[Dict[str, Any]]:
    ...
```

- **Params**:
  - `query`: The query to rank against.
  - `files`, `urls`: As in `retrieve_documents_async`.
  - `splitter`: Optional; e.g., use `Embs.markdown_splitter`.
- **Returns**: A ranked list of chunks with `{"filename": ..., "markdown": ..., "probability": ..., "cosine_similarity": ...}`.

## Caching for Performance

Enable in-memory or disk-based caching to avoid redundant processing:

```python
cache_conf = {
    "enabled": True,
    "type": "memory",       # or "disk"
    "prefix": "myapp",
    "dir": "cache_folder",  # only needed for disk caching
    "max_mem_items": 128,
    "max_ttl_seconds": 86400
}

client = Embs(cache_config=cache_conf)
```

- **Memory Caching**: Quick lookups using LRU with TTL expiration.  
- **Disk Caching**: Stores JSON files to a specified directory, evicting older files after TTL expiration.

## Testing

`embs` is rigorously tested using **`pytest`** and **`pytest-asyncio`**. To ensure that retrieval, embeddings, ranking, caching, and splitting are working as expected, run:

```bash
pytest --asyncio-mode=auto
```

## License

Licensed under the **MIT License**. See [LICENSE](./LICENSE) for details.

Contributions are welcome! Submit issues, ideas, or pull requests to help improve `embs`.
