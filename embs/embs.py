"""
embs.py

A lightweight Python library that streamlines document ingestion,
embedding, and ranking for RAG systems, chatbots, semantic search engines,
and more.

Key features:
- Document conversion via Docsifer (files and URLs)
- Powerful document splitting (e.g., Markdown-based)
- Embedding generation using a lightweight embeddings API
- Ranking and optional embedding inclusion in results
- In-memory and disk caching
- DuckDuckGo-powered web search integration

Usage examples are provided in the README.
"""

import os
import json
import time
import hashlib
import logging
import asyncio
import aiohttp

from aiohttp import FormData
from collections import OrderedDict
from typing import (
    List, Dict, Any, Optional, Union, Callable, Tuple
)
from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)


def _split_markdown_text(
    text: str,
    headers_to_split_on: List[Tuple[str, str]],
    return_each_line: bool,
    strip_headers: bool
) -> List[str]:
    """
    Splits a markdown string into chunks based on specified header markers.
    
    Args:
        text: Full markdown text.
        headers_to_split_on: A list of (header_prefix, header_name) pairs.
        return_each_line: If True, each nonblank line is its own chunk.
        strip_headers: If True, header lines are not included in the chunks.
    
    Returns:
        List of chunk strings.
    """
    # Sort header markers descending by length so that longer markers match first.
    headers_to_split_on = sorted(headers_to_split_on, key=lambda x: len(x[0]), reverse=True)

    lines_with_metadata: List[Dict[str, Any]] = []
    raw_lines = text.split("\n")

    in_code_block = False
    opening_fence = ""

    current_content: List[str] = []
    current_metadata: Dict[str, str] = {}

    header_stack: List[Dict[str, Union[int, str]]] = []
    active_metadata: Dict[str, str] = {}

    def flush_current():
        if current_content:
            lines_with_metadata.append({
                "content": "\n".join(current_content),
                "metadata": current_metadata.copy()
            })
            current_content.clear()

    for line in raw_lines:
        stripped_line = "".join(ch for ch in line.strip() if ch.isprintable())

        # Detect code blocks.
        if not in_code_block:
            if stripped_line.startswith("</code></pre>") and stripped_line.count("<pre class=" 'overflow-x-auto"><code>') == 1:
                in_code_block = True
                opening_fence = "</code></pre>"
            elif stripped_line.startswith("~~~"):
                in_code_block = True
                opening_fence = "~~~"
        else:
            if stripped_line.startswith(opening_fence):
                in_code_block = False
                opening_fence = ""
        if in_code_block:
            current_content.append(stripped_line)
            continue

        found_header = False
        for sep, name in headers_to_split_on:
            if stripped_line.startswith(sep) and (len(stripped_line) == len(sep) or stripped_line[len(sep)] == " "):
                found_header = True
                current_level = sep.count("#")
                while header_stack and header_stack[-1]["level"] >= current_level:
                    popped = header_stack.pop()
                    active_metadata.pop(popped["name"], None)
                header_stack.append({
                    "level": current_level,
                    "name": name,
                    "data": stripped_line[len(sep):].strip()
                })
                active_metadata[name] = header_stack[-1]["data"]
                flush_current()
                if not strip_headers:
                    current_content.append(stripped_line)
                break

        if not found_header:
            if stripped_line:
                current_content.append(stripped_line)
            else:
                flush_current()
        current_metadata = active_metadata.copy()

    flush_current()

    if return_each_line:
        final_chunks = []
        for item in lines_with_metadata:
            for single_line in item["content"].split("\n"):
                if single_line.strip():
                    final_chunks.append(single_line)
        return final_chunks

    final_chunks = []
    temp_block = None
    temp_meta = None
    for item in lines_with_metadata:
        txt = item["content"]
        meta = item["metadata"]
        if temp_block is not None and meta == temp_meta:
            temp_block += "\n" + txt
        else:
            if temp_block is not None:
                final_chunks.append(temp_block)
            temp_block = txt
            temp_meta = meta
    if temp_block is not None:
        final_chunks.append(temp_block)
    return final_chunks


class Embs:
    """
    A one-stop toolkit for document ingestion, embedding, and ranking workflows.
    
    This library integrates:
      - Docsifer for converting files/URLs to markdown,
      - A lightweight embeddings API for generating text embeddings, and
      - Ranking of documents/chunks based on query relevance.
    
    It supports optional in-memory or disk caching and flexible document splitting.
    
    You can also choose to have the final results include the raw embeddings by passing
    options={"embeddings": True} to the query and search methods.
    """

    def __init__(
        self,
        docsifer_base_url: str = "https://lamhieu-docsifer.hf.space",
        docsifer_endpoint: str = "/v1/convert",
        embeddings_base_url: str = "https://lamhieu-lightweight-embeddings.hf.space",
        embeddings_endpoint: str = "/v1/embeddings",
        rank_endpoint: str = "/v1/rank",
        default_model: str = "snowflake-arctic-embed-l-v2.0",
        cache_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize an Embs instance.
        
        Args:
            docsifer_base_url: Base URL for Docsifer.
            docsifer_endpoint: Endpoint path for document conversion.
            embeddings_base_url: Base URL for the embeddings service.
            embeddings_endpoint: Endpoint path for generating embeddings.
            rank_endpoint: Endpoint path for ranking texts.
            default_model: Default model name for embedding and ranking.
            cache_config: Dictionary to configure caching (memory or disk).
        """
        if cache_config is None:
            cache_config = {}

        self.docsifer_base_url = docsifer_base_url.rstrip("/")
        self.docsifer_endpoint = docsifer_endpoint
        self.embeddings_base_url = embeddings_base_url.rstrip("/")
        self.embeddings_endpoint = embeddings_endpoint
        self.rank_endpoint = rank_endpoint
        self.default_model = default_model

        self.cache_enabled: bool = cache_config.get("enabled", False)
        self.cache_type: str = cache_config.get("type", "memory").lower()
        self.cache_prefix: str = cache_config.get("prefix", "")
        self.cache_dir: Optional[str] = cache_config.get("dir")
        self.max_mem_items: int = cache_config.get("max_mem_items", 128)
        self.max_ttl_seconds: int = cache_config.get("max_ttl_seconds", 259200)

        if self.cache_type not in ("memory", "disk"):
            raise ValueError('cache_config["type"] must be either "memory" or "disk".')

        self._mem_cache: "OrderedDict[str, (float, Any)]" = OrderedDict()
        if self.cache_enabled and self.cache_type == "disk":
            if not self.cache_dir:
                raise ValueError('If "type"=="disk", you must provide "dir" in cache_config.')
            os.makedirs(self.cache_dir, exist_ok=True)

    @staticmethod
    def markdown_splitter(
        docs: List[Dict[str, str]],
        config: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, str]]:
        """
        Splits Markdown documents into smaller chunks using header rules.
        
        Args:
            docs: A list of documents, each with "filename" and "markdown".
            config: Configuration dict for splitting. Keys:
                - headers_to_split_on: List of (str, str) pairs.
                - return_each_line: bool.
                - strip_headers: bool.
        
        Returns:
            A list of documents with subdivided chunks.
        """
        if config is None:
            config = {}

        headers_to_split_on = config.get("headers_to_split_on", [("#", "h1"), ("##", "h2")])
        return_each_line = config.get("return_each_line", False)
        strip_headers = config.get("strip_headers", True)

        output_docs: List[Dict[str, str]] = []
        for doc in docs:
            original_filename = doc["filename"]
            text = doc["markdown"]
            chunks = _split_markdown_text(
                text,
                headers_to_split_on=headers_to_split_on,
                return_each_line=return_each_line,
                strip_headers=strip_headers
            )
            if not chunks:
                output_docs.append(doc)
            else:
                for idx, chunk_text in enumerate(chunks):
                    output_docs.append({
                        "filename": f"{original_filename}/{idx}",
                        "markdown": chunk_text
                    })
        return output_docs

    def _make_key(self, name: str, **kwargs) -> str:
        """
        Build a cache key by hashing the method name, optional prefix, and sorted kwargs.
        """
        safe_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, list):
                safe_list = []
                for item in v:
                    if isinstance(item, str):
                        safe_list.append(item)
                    else:
                        safe_list.append(f"<file_obj:{id(item)}>")
                safe_kwargs[k] = safe_list
            elif isinstance(v, dict):
                try:
                    safe_kwargs[k] = json.dumps(v, sort_keys=True)
                except Exception:
                    safe_kwargs[k] = str(v)
            else:
                safe_kwargs[k] = v

        raw_str = f"{self.cache_prefix}:{name}-{json.dumps(safe_kwargs, sort_keys=True)}"
        return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()

    def _evict_memory_cache_if_needed(self) -> None:
        """Evicts the least recently used item if memory cache exceeds capacity."""
        while len(self._mem_cache) > self.max_mem_items:
            key, _ = self._mem_cache.popitem(last=False)
            logger.debug(f"Evicted LRU item from memory cache: {key}")

    def _check_expiry_in_memory(self, key: str) -> bool:
        """
        Checks if an in-memory cache item has expired.
        
        Returns:
            True if the item was expired and removed.
        """
        timestamp, _ = self._mem_cache[key]
        if (time.time() - timestamp) > self.max_ttl_seconds:
            self._mem_cache.pop(key, None)
            logger.debug(f"Evicted expired item from memory cache: {key}")
            return True
        return False

    def _load_from_cache(self, key: str) -> Any:
        """
        Retrieve a cached item from memory or disk.
        """
        if not self.cache_enabled:
            return None

        if self.cache_type == "memory":
            if key in self._mem_cache:
                if self._check_expiry_in_memory(key):
                    return None
                timestamp, data = self._mem_cache.pop(key)
                self._mem_cache[key] = (timestamp, data)
                return data
            return None

        if self.cache_type == "disk":
            if not self.cache_dir:
                return None
            file_path = os.path.join(self.cache_dir, key + ".json")
            if not os.path.exists(file_path):
                return None
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                creation_time = meta.get("timestamp", 0)
                if (time.time() - creation_time) > self.max_ttl_seconds:
                    os.remove(file_path)
                    logger.debug(f"Evicted expired disk cache file: {file_path}")
                    return None
                return meta.get("data", None)
            except Exception as e:
                logger.error(f"Failed to load from disk cache: {e}")
                return None

        return None

    def _save_to_cache(self, key: str, data: Any) -> None:
        """
        Save data to cache (memory or disk).
        """
        if not self.cache_enabled:
            return

        if self.cache_type == "memory":
            timestamp_data = (time.time(), data)
            if key in self._mem_cache:
                self._mem_cache.pop(key)
            self._mem_cache[key] = timestamp_data
            self._evict_memory_cache_if_needed()
        else:
            if not self.cache_dir:
                return
            file_path = os.path.join(self.cache_dir, key + ".json")
            meta = {
                "timestamp": time.time(),
                "data": data
            }
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False)
            except Exception as e:
                logger.error(f"Failed to save to disk cache: {e}")

    async def _upload_file(
        self,
        file: Any,
        session: aiohttp.ClientSession,
        openai_config: Optional[Dict[str, Any]],
        settings: Optional[Dict[str, Any]],
        semaphore: asyncio.Semaphore,
        options: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, str]]:
        """
        Uploads a file to Docsifer and returns its converted markdown.
        """
        silent = bool(options.get("silent", False)) if options else False
        docsifer_url = f"{self.docsifer_base_url}{self.docsifer_endpoint}"

        async with semaphore:
            try:
                form = FormData()
                if isinstance(file, str):
                    filename = os.path.basename(file)
                    with open(file, "rb") as fp:
                        form.add_field("file", fp, filename=filename, content_type="application/octet-stream")
                elif hasattr(file, "read"):
                    filename = getattr(file, "name", "unknown_file")
                    form.add_field("file", file, filename=filename, content_type="application/octet-stream")
                else:
                    raise ValueError("Invalid file input. Must be a path or file-like object.")

                if openai_config:
                    form.add_field("openai", json.dumps(openai_config), content_type="application/json")
                if settings:
                    form.add_field("settings", json.dumps(settings), content_type="application/json")

                async with session.post(docsifer_url, data=form) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            except Exception as exc:
                if silent:
                    logger.error(f"Docsifer file upload error: {exc}")
                    return None
                raise

    async def _upload_url(
        self,
        url: str,
        session: aiohttp.ClientSession,
        openai_config: Optional[Dict[str, Any]],
        settings: Optional[Dict[str, Any]],
        semaphore: asyncio.Semaphore,
        options: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, str]]:
        """
        Uploads a URL to Docsifer for HTML-to-Markdown conversion.
        """
        silent = bool(options.get("silent", False)) if options else False
        docsifer_url = f"{self.docsifer_base_url}{self.docsifer_endpoint}"

        async with semaphore:
            try:
                form = FormData()
                form.add_field("url", url, content_type="text/plain")

                if openai_config:
                    form.add_field("openai", json.dumps(openai_config), content_type="application/json")
                if settings:
                    form.add_field("settings", json.dumps(settings), content_type="application/json")

                async with session.post(docsifer_url, data=form) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            except Exception as exc:
                if silent:
                    logger.error(f"Docsifer URL conversion error: {exc}")
                    return None
                raise

    async def retrieve_documents_async(
        self,
        files: Optional[List[Any]] = None,
        urls: Optional[List[str]] = None,
        openai_config: Optional[Dict[str, Any]] = None,
        settings: Optional[Dict[str, Any]] = None,
        concurrency: int = 5,
        options: Optional[Dict[str, Any]] = None,
        splitter: Optional[Callable[[List[Dict[str, str]]], List[Dict[str, str]]]] = None,
    ) -> List[Dict[str, str]]:
        """
        Asynchronously retrieves documents from Docsifer (via files and/or URLs),
        optionally applies a splitter, and returns a list of documents.
        """
        cache_key = None
        if self.cache_enabled:
            cache_key = self._make_key(
                "retrieve_documents_async",
                files=files,
                urls=urls,
                openai_config=openai_config,
                settings=settings,
                concurrency=concurrency,
                options=options,
                splitter=bool(splitter)
            )
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                return cached_data

        if not files and not urls:
            return []

        semaphore = asyncio.Semaphore(concurrency)
        all_docs: List[Dict[str, str]] = []

        async with aiohttp.ClientSession() as session:
            tasks = []
            if files:
                for f in files:
                    tasks.append(self._upload_file(f, session, openai_config, settings, semaphore, options))
            if urls:
                for u in urls:
                    tasks.append(self._upload_url(u, session, openai_config, settings, semaphore, options))

            silent = bool(options.get("silent", False)) if options else False
            results = await asyncio.gather(*tasks, return_exceptions=silent)
            for r in results:
                if isinstance(r, Exception):
                    logger.error(f"Docsifer retrieval exception: {r}")
                elif r is not None and "filename" in r and "markdown" in r:
                    all_docs.append(r)
                elif r is not None:
                    logger.warning(f"Unexpected Docsifer response shape: {r}")

        if splitter is not None:
            all_docs = splitter(all_docs)

        if self.cache_enabled and cache_key:
            self._save_to_cache(cache_key, all_docs)
        return all_docs

    async def embed_async(
        self,
        text_or_texts: Union[str, List[str]],
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Asynchronously generates embeddings for the provided text(s).
        
        Args:
            text_or_texts: A string or a list of strings.
            model: Model name (defaults to self.default_model if not provided).
        
        Returns:
            A dictionary with keys like "data", "model", and "usage".
        """
        if model is None:
            model = self.default_model

        cache_key = None
        if self.cache_enabled:
            cache_key = self._make_key("embed_async", text=text_or_texts, model=model)
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                return cached_data

        endpoint = f"{self.embeddings_base_url}{self.embeddings_endpoint}"
        payload = {"model": model, "input": text_or_texts}

        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()

        if self.cache_enabled and cache_key:
            self._save_to_cache(cache_key, data)
        return data

    async def rank_async(
        self,
        query: str,
        candidates: List[str],
        model: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Asynchronously ranks candidate texts by relevance to the given query.
        
        Args:
            query: The query string.
            candidates: A list of candidate texts.
            model: Model name (defaults to self.default_model if not provided).
        
        Returns:
            A list of ranking dictionaries with keys "text", "probability", and "cosine_similarity".
        """
        if model is None:
            model = self.default_model

        cache_key = None
        if self.cache_enabled:
            cache_key = self._make_key("rank_async", query=query, candidates=candidates, model=model)
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                return cached_data

        endpoint = f"{self.embeddings_base_url}{self.rank_endpoint}"
        payload = {
            "model": model,
            "queries": query,
            "candidates": candidates
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()
                probabilities = data.get("probabilities", [[]])
                cos_sims = data.get("cosine_similarities", [[]])

                if not probabilities or not probabilities[0]:
                    results = []
                else:
                    results = []
                    for i, text_val in enumerate(candidates):
                        p = probabilities[0][i] if i < len(probabilities[0]) else 0.0
                        c = cos_sims[0][i] if i < len(cos_sims[0]) else 0.0
                        results.append({
                            "text": text_val,
                            "probability": p,
                            "cosine_similarity": c
                        })
                    results.sort(key=lambda x: x["probability"], reverse=True)

        if self.cache_enabled and cache_key:
            self._save_to_cache(cache_key, results)
        return results

    async def query_documents_async(
        self,
        query: str,
        files: Optional[List[Any]] = None,
        urls: Optional[List[str]] = None,
        openai_config: Optional[Dict[str, Any]] = None,
        settings: Optional[Dict[str, Any]] = None,
        concurrency: int = 5,
        options: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        splitter: Optional[Callable[[List[Dict[str, str]]], List[Dict[str, str]]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Asynchronously retrieves documents (via files/URLs), ranks them by relevance to the query,
        and returns a list of documents with ranking scores. Optionally, if options contains
        "embeddings": True, the function will also attach an embedding for each document.
        
        Args:
            query: The query to rank against.
            files: List of file paths or file-like objects.
            urls: List of URLs to convert.
            openai_config: Optional Docsifer configuration.
            settings: Additional Docsifer settings.
            concurrency: Max concurrency for retrieval.
            options: Dict of additional options. Use {"embeddings": True} to include embeddings.
            model: Model name for ranking (defaults to self.default_model).
            splitter: Optional callable to further split document content.
        
        Returns:
            A list of documents with keys "filename", "markdown", "probability", "cosine_similarity",
            and optionally "embeddings".
        """
        docs = await self.retrieve_documents_async(
            files=files,
            urls=urls,
            openai_config=openai_config,
            settings=settings,
            concurrency=concurrency,
            options=options,
            splitter=splitter
        )
        if not docs:
            return []

        candidates = [doc["markdown"] for doc in docs]
        ranking = await self.rank_async(query, candidates, model=model)

        # Map ranked text to document indices.
        text_to_indices: Dict[str, List[int]] = {}
        for i, d_obj in enumerate(docs):
            text_val = d_obj["markdown"]
            text_to_indices.setdefault(text_val, []).append(i)

        results: List[Dict[str, Any]] = []
        used_indices = set()

        for item in ranking:
            text_val = item["text"]
            possible_idxs = text_to_indices.get(text_val, [])
            matched_idx = None
            for idx in possible_idxs:
                if idx not in used_indices:
                    matched_idx = idx
                    used_indices.add(idx)
                    break
            if matched_idx is not None:
                matched_doc = docs[matched_idx]
                results.append({
                    "filename": matched_doc["filename"],
                    "markdown": matched_doc["markdown"],
                    "probability": item["probability"],
                    "cosine_similarity": item["cosine_similarity"]
                })

        # If requested, attach embeddings to each result.
        if options and options.get("embeddings", False):
            texts = [item["markdown"] for item in results]
            embed_response = await self.embed_async(texts, model=model)
            embeddings_list = embed_response.get("data", [])
            for idx, item in enumerate(results):
                item["embeddings"] = embeddings_list[idx] if idx < len(embeddings_list) else None

        return results

    def query_documents(
        self,
        query: str,
        files: Optional[List[Any]] = None,
        urls: Optional[List[str]] = None,
        openai_config: Optional[Dict[str, Any]] = None,
        settings: Optional[Dict[str, Any]] = None,
        concurrency: int = 5,
        options: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        splitter: Optional[Callable[[List[Dict[str, str]]], List[Dict[str, str]]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Synchronous wrapper for query_documents_async.
        """
        return asyncio.run(
            self.query_documents_async(
                query=query,
                files=files,
                urls=urls,
                openai_config=openai_config,
                settings=settings,
                concurrency=concurrency,
                options=options,
                model=model,
                splitter=splitter
            )
        )

    async def _duckduckgo_search(
        self,
        query: str,
        limit: int,
        blocklist: Optional[List[str]] = None
    ) -> List[str]:
        """
        Asynchronously performs a DuckDuckGo search for the given query and returns a list of URLs.
        
        Args:
            query: The search query string.
            limit: Maximum number of search results.
            blocklist: Optional list of domain substrings to filter out.
        
        Returns:
            List of URLs from DuckDuckGo.
        """
        def run_search():
            with DDGS() as ddgs:
                return list(ddgs.text(query, safesearch="moderate", max_results=limit, backend="auto", region="vn-vi"))
        results = await asyncio.to_thread(run_search)
        urls = []
        for item in results:
            url = item.get("href")
            if url:
                if blocklist:
                    skip = False
                    for pattern in blocklist:
                        if pattern in url:
                            skip = True
                            break
                    if skip:
                        continue
                urls.append(url)
        return urls

    async def search_documents_async(
        self,
        query: str,
        limit: int = 10,
        blocklist: Optional[List[str]] = None,
        openai_config: Optional[Dict[str, Any]] = None,
        settings: Optional[Dict[str, Any]] = None,
        concurrency: int = 5,
        options: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        splitter: Optional[Callable[[List[Dict[str, str]]], List[Dict[str, str]]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Asynchronously searches for documents using DuckDuckGo to obtain URLs based on a keyword,
        then retrieves and ranks their content (using Docsifer and the ranking API). This function
        returns results similar to query_documents_async.
        
        Args:
            query: The search keyword (used for both DuckDuckGo and ranking).
            limit: Maximum number of DuckDuckGo results.
            blocklist: Optional list of domain substrings to filter out.
            openai_config: Optional Docsifer configuration.
            settings: Additional Docsifer settings.
            concurrency: Max concurrency for retrieval.
            options: Dict of additional options. Use {"embeddings": True} to include embeddings.
            model: Model name for ranking (defaults to self.default_model).
            splitter: Optional callable to split document content.
        
        Returns:
            A list of ranked documents with keys "filename", "markdown", "probability",
            "cosine_similarity", and optionally "embeddings".
        """
        urls = await self._duckduckgo_search(query, limit=limit, blocklist=blocklist)
        return await self.query_documents_async(
            query=query,
            urls=urls,
            openai_config=openai_config,
            settings=settings,
            concurrency=concurrency,
            options=options,
            model=model,
            splitter=splitter
        )

    def search_documents(
        self,
        query: str,
        limit: int = 10,
        blocklist: Optional[List[str]] = None,
        openai_config: Optional[Dict[str, Any]] = None,
        settings: Optional[Dict[str, Any]] = None,
        concurrency: int = 5,
        options: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        splitter: Optional[Callable[[List[Dict[str, str]]], List[Dict[str, str]]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Synchronous wrapper for search_documents_async (DuckDuckGo-powered search).
        """
        return asyncio.run(
            self.search_documents_async(
                query=query,
                limit=limit,
                blocklist=blocklist,
                openai_config=openai_config,
                settings=settings,
                concurrency=concurrency,
                options=options,
                model=model,
                splitter=splitter
            )
        )

# =============================================================================
# Example usage:
# -----------------------------------------------------------------------------
# 1) Using the built-in markdown splitter:
#
# from functools import partial
#
# split_config = {
#     "headers_to_split_on": [("#", "h1"), ("##", "h2"), ("###", "h3")],
#     "return_each_line": False,
#     "strip_headers": True,
# }
# md_splitter = partial(Embs.markdown_splitter, config=split_config)
#
# client = Embs()
#
# # Retrieve and rank documents (query_documents) with optional embeddings in the results:
# docs = client.query_documents(
#     query="Explain quantum computing",
#     files=["/path/to/quantum_theory.pdf"],
#     splitter=md_splitter,
#     options={"embeddings": True}
# )
# for d in docs:
#     print(d["filename"], "=> score:", d["probability"], "Embeddings:", d.get("embeddings"))
#
# 2) Search documents using DuckDuckGo (search_documents):
#
# results = client.search_documents(
#     query="Latest advances in AI",
#     limit=5,
#     blocklist=["youtube.com"],
#     options={"embeddings": True}
# )
# for item in results:
#     print(f"File: {item['filename']} | Score: {item['probability']:.4f}")
#     print(f"Snippet: {item['markdown'][:80]}...\n")
# =============================================================================
