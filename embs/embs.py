# filename: embs.py

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

logger = logging.getLogger(__name__)

def _split_markdown_text(
    text: str,
    headers_to_split_on: List[Tuple[str, str]],
    return_each_line: bool,
    strip_headers: bool
) -> List[str]:
    """
    Splits a single markdown string into multiple chunks based on specified headers.

    Args:
        text: The full markdown text to split.
        headers_to_split_on: A list of (header_prefix, header_name) pairs, e.g. [("#", "h1"), ("##", "h2")].
        return_each_line: If True, every line becomes its own chunk (unless blank).
        strip_headers: If True, header lines themselves are removed from the chunk content.

    Returns:
        A list of chunk strings resulting from splitting the original markdown text.
    """
    # Sort by length (descending) so that longer header prefixes (e.g. "##" vs "#") match first
    headers_to_split_on = sorted(headers_to_split_on, key=lambda x: len(x[0]), reverse=True)

    # We'll track lines with metadata so we can group lines under the same header hierarchy
    lines_with_metadata: List[Dict[str, Any]] = []
    raw_lines = text.split("\n")

    # Code block fence detection
    in_code_block = False
    opening_fence = ""

    current_content: List[str] = []
    current_metadata: Dict[str, str] = {}

    # header_stack helps track nested header levels
    header_stack: List[Dict[str, Union[int, str]]] = []
    active_metadata: Dict[str, str] = {}

    def flush_current():
        """Helper function to flush current_content into lines_with_metadata."""
        if current_content:
            lines_with_metadata.append({
                "content": "\n".join(current_content),
                "metadata": current_metadata.copy()
            })
            current_content.clear()

    for line in raw_lines:
        stripped_line = line.strip()
        # Remove non-printable characters
        stripped_line = "".join(ch for ch in stripped_line if ch.isprintable())

        # Check if we are entering or exiting a code block
        if not in_code_block:
            if stripped_line.startswith("```") and stripped_line.count("```") == 1:
                in_code_block = True
                opening_fence = "```"
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
            # Check if the line starts with a known header prefix
            if stripped_line.startswith(sep) and (
                len(stripped_line) == len(sep) or stripped_line[len(sep)] == " "
            ):
                found_header = True
                current_level = sep.count("#")

                # Pop any headers at or deeper than the current level
                while header_stack and header_stack[-1]["level"] >= current_level:
                    popped = header_stack.pop()
                    if popped["name"] in active_metadata:
                        active_metadata.pop(popped["name"], None)

                # Push new header
                header_stack.append({
                    "level": current_level,
                    "name": name,
                    "data": stripped_line[len(sep):].strip()
                })
                active_metadata[name] = header_stack[-1]["data"]

                # Flush any current lines
                flush_current()

                # If not stripping header lines, we include the header text itself
                if not strip_headers:
                    current_content.append(stripped_line)

                break

        if not found_header:
            if stripped_line:
                current_content.append(stripped_line)
            else:
                # Blank line => flush
                flush_current()

        current_metadata = active_metadata.copy()

    # Flush the remainder
    flush_current()

    # If return_each_line = True, each line is a separate chunk
    if return_each_line:
        final_chunks: List[str] = []
        for item in lines_with_metadata:
            per_line = item["content"].split("\n")
            for single_line in per_line:
                if single_line.strip():
                    final_chunks.append(single_line)
        return final_chunks

    # Otherwise, we group lines that share the same metadata block
    # lines_with_metadata is already one block per "metadata" break, so we just merge them if needed
    final_chunks: List[str] = []
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
    Provides document retrieval from Docsifer, embeddings with the Lightweight Embeddings API,
    and ranking of documents by query relevance. Caching can be configured in memory or on disk,
    with both time-based TTL and optional LRU eviction for in-memory items.

    You can optionally pass a `splitter` callable to the retrieve or search methods. If not None,
    it must accept a list of docs (each doc is {"filename": str, "markdown": str})
    and return a new list of docs with the same shape but possibly split further.
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
            docsifer_endpoint: Endpoint path for converting documents with Docsifer.
            embeddings_base_url: Base URL for the Lightweight Embeddings service.
            embeddings_endpoint: Endpoint path for generating embeddings.
            rank_endpoint: Endpoint path for ranking texts.
            default_model: Default model name to use for embeddings and ranking if unspecified.
            cache_config: Dictionary controlling caching behavior. Example:
                {
                    "enabled": True,
                    "type": "memory",  # or "disk"
                    "prefix": "mycache",
                    "dir": "my_cache_dir",   # used if type="disk"
                    "max_mem_items": 128,
                    "max_ttl_seconds": 259200
                }
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
        Static splitter method for markdown documents. Each doc is a dict:
            {"filename": <str>, "markdown": <str>}

        If `config` is provided, it can contain the following keys:
          - "headers_to_split_on": List of (str, str), e.g. [("#", "h1"), ("##", "h2")]
          - "return_each_line": bool
          - "strip_headers": bool

        Example usage:
            docs = Embs.markdown_splitter(some_docs, {
                "headers_to_split_on": [("#", "h1"), ("##", "h2")],
                "return_each_line": False,
                "strip_headers": True
            })

        Args:
            docs: A list of documents, each with "filename" and "markdown".
            config: A dictionary specifying how to split the markdown.

        Returns:
            A new list of documents, potentially with multiple chunks per original doc,
            each chunk having its own "filename" (e.g., "original/0", "original/1", etc.)
            and "markdown".
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
                # if there's no content, keep it as is
                output_docs.append(doc)
            else:
                for idx, chunk_text in enumerate(chunks):
                    output_docs.append({
                        "filename": f"{original_filename}/{idx}",
                        "markdown": chunk_text
                    })
        return output_docs

    def _make_key(self, name: str, **kwargs) -> str:
        """Builds a cache key by hashing the method name, optional prefix, and sorted kwargs."""
        safe_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, list):
                # Convert list to stable string representation; ignore file-like
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
        """
        Removes the least recently used item if the in-memory cache exceeds max_mem_items.
        """
        while len(self._mem_cache) > self.max_mem_items:
            key, _ = self._mem_cache.popitem(last=False)
            logger.debug(f"Evicted LRU item from memory cache: {key}")

    def _check_expiry_in_memory(self, key: str) -> bool:
        """
        Checks if an item in memory has expired based on max_ttl_seconds.
        Returns True if removed, False otherwise.
        """
        timestamp, _ = self._mem_cache[key]
        if (time.time() - timestamp) > self.max_ttl_seconds:
            self._mem_cache.pop(key, None)
            logger.debug(f"Evicted expired item from memory cache: {key}")
            return True
        return False

    def _load_from_cache(self, key: str) -> Any:
        """
        Retrieves a cached item by key from memory or disk if caching is enabled.

        Args:
            key: The cache key.

        Returns:
            The cached data, or None if not found or expired.
        """
        if not self.cache_enabled:
            return None

        if self.cache_type == "memory":
            if key in self._mem_cache:
                if self._check_expiry_in_memory(key):
                    return None
                # Mark as recently used
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
        Saves data to memory or disk cache if caching is enabled.

        Args:
            key: The cache key.
            data: The data to store.
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
        Uploads a single file to Docsifer asynchronously and returns {"filename": ..., "markdown": ...}.

        Args:
            file: File path (string) or file-like object.
            session: Aiohttp ClientSession.
            openai_config: Optional LLM-based config for Docsifer.
            settings: Additional Docsifer settings.
            semaphore: Limits concurrency.
            options: Dict of additional options, e.g. {"silent": True} to log errors instead of raising.

        Returns:
            A dict of {"filename": <str>, "markdown": <str>} on success, or None if an error occurred (silent=True).
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
        Uploads a single URL to Docsifer for HTML->Markdown conversion.

        Args:
            url: The URL to process.
            session: Aiohttp ClientSession.
            openai_config: Optional LLM-based config for Docsifer.
            settings: Additional Docsifer settings.
            semaphore: Limits concurrency.
            options: Dict of additional options, e.g. {"silent": True}.

        Returns:
            A dict of {"filename": <str>, "markdown": <str>} on success, or None if an error occurred (silent=True).
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
        Asynchronously retrieves documents from Docsifer (via files and/or URLs), returning a list of:
            [{"filename": <str>, "markdown": <str>} ...]

        If splitter is provided, it will be applied to the final list of docs (e.g., to split large markdown text).

        Args:
            files: A list of file paths or file-like objects to upload.
            urls: A list of URLs to be converted by Docsifer.
            openai_config: Optional LLM-based extraction config for Docsifer.
            settings: Additional Docsifer settings.
            concurrency: Maximum number of concurrent tasks (file/URL conversions).
            options: Dict of additional options, e.g. {"silent": True} to log instead of raise on errors.
            splitter: A callable that takes a list of docs and returns a new list of docs (same shape).

        Returns:
            A list of docs, each doc is {"filename": <str>, "markdown": <str>}.
            Potentially more than you started with if the splitter subdivides them.
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
                splitter=bool(splitter)  # store True/False
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

        # Apply the splitter if provided
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
        Asynchronously generates embeddings for the given text(s) using the specified or default model.
        Uses caching if enabled.

        Args:
            text_or_texts: A single string or a list of strings to embed.
            model: Model name. If None, uses self.default_model.

        Returns:
            A dict containing embeddings results, e.g.:
            {
              "model": ...,
              "data": [...],
              "usage": { ... }
            }
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
        Asynchronously ranks candidate strings by relevance to a single query using the rank endpoint.
        Caches if enabled.

        Args:
            query: The query string.
            candidates: A list of candidate strings to rank.
            model: Model name or None for default.

        Returns:
            A list of dicts sorted by probability descending. For example:
            [
              {"text": <candidate_str>, "probability": <float>, "cosine_similarity": <float>},
              ...
            ]
        """
        if model is None:
            model = self.default_model

        cache_key = None
        if self.cache_enabled:
            cache_key = self._make_key(
                "rank_async", query=query, candidates=candidates, model=model
            )
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

    async def search_documents_async(
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
        Asynchronously retrieves documents from Docsifer, optionally applies a splitter,
        then ranks them by the given query.

        Args:
            query: The query to compare against.
            files: A list of file paths or file-like objects.
            urls: A list of URLs to be converted.
            openai_config: Optional LLM-based config for Docsifer.
            settings: Additional Docsifer settings.
            concurrency: Max concurrency for the retrieval process.
            options: Dict of additional options, e.g. {"silent": True}.
            model: Model name for ranking or None for default.
            splitter: A callable to split the docs. If not None, it must
                take a list of docs and return a new list of docs.

        Returns:
            A list of dicts, each dict containing:
            {
              "filename": <str>,
              "markdown": <str>,
              "probability": <float>,
              "cosine_similarity": <float>
            },
            sorted by probability descending.
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

        # Map text to doc indices
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

        return results

    def retrieve_documents(
        self,
        files: Optional[List[Any]] = None,
        urls: Optional[List[str]] = None,
        openai_config: Optional[Dict[str, Any]] = None,
        settings: Optional[Dict[str, Any]] = None,
        concurrency: int = 5,
        options: Optional[Dict[str, Any]] = None,
        splitter: Optional[Callable[[List[Dict[str, str]]], List[Dict[str, str]]]] = None
    ) -> List[Dict[str, str]]:
        """
        Synchronous wrapper for retrieve_documents_async.
        Retrieves documents from Docsifer, optionally applies a splitter, and returns them.

        Args:
            files: A list of file paths or file-like objects.
            urls: A list of URLs to convert.
            openai_config: Optional LLM-based config for Docsifer.
            settings: Additional Docsifer settings.
            concurrency: Max concurrency for the retrieval.
            options: Dict of additional options, e.g. {"silent": True}.
            splitter: A callable that receives the list of docs, returns a new list of docs.

        Returns:
            A list of dicts [{"filename": <str>, "markdown": <str>} ...].
        """
        return asyncio.run(
            self.retrieve_documents_async(
                files=files,
                urls=urls,
                openai_config=openai_config,
                settings=settings,
                concurrency=concurrency,
                options=options,
                splitter=splitter
            )
        )

    def embed(
        self,
        text_or_texts: Union[str, List[str]],
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for embed_async.
        Generates embeddings for text_or_texts using the specified or default model.

        Args:
            text_or_texts: A string or list of strings to embed.
            model: Optional model name. Defaults to self.default_model if None.

        Returns:
            A dict with embedding results, e.g. {"data": [...], "model": "...", "usage": {...}}.
        """
        return asyncio.run(self.embed_async(text_or_texts, model=model))

    def rank(
        self,
        query: str,
        candidates: List[str],
        model: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Synchronous wrapper for rank_async.
        Ranks candidate strings by relevance to 'query'.

        Args:
            query: The query text.
            candidates: A list of candidate strings to rank.
            model: Optional model name. Defaults to self.default_model if None.

        Returns:
            A list of dicts sorted by probability descending:
            [
              {"text": <str>, "probability": <float>, "cosine_similarity": <float>},
              ...
            ]
        """
        return asyncio.run(self.rank_async(query, candidates, model=model))

    def search_documents(
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
        Synchronous wrapper for search_documents_async.
        Retrieves documents, optionally applies a splitter, then ranks them by the query.

        Args:
            query: The query string.
            files: A list of file paths or file-like objects.
            urls: A list of URLs.
            openai_config: Optional LLM-based config for Docsifer.
            settings: Additional Docsifer settings.
            concurrency: Max concurrency for the retrieval.
            options: Dict of additional options, e.g. {"silent": True}.
            model: Model name or None for default.
            splitter: An optional callable that receives the doc list and returns a new list.

        Returns:
            A list of ranked documents:
            [
              {
                "filename": <str>,
                "markdown": <str>,
                "probability": <float>,
                "cosine_similarity": <float>
              },
              ...
            ], sorted by probability descending.
        """
        return asyncio.run(
            self.search_documents_async(
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


# from typing import List, Dict
# from functools import partial

# # 1) Using the built-in markdown_splitter:
# split_config = {
#     "headers_to_split_on": [("#", "h1"), ("##", "h2"), ("###", "h3")],
#     "return_each_line": False,
#     "strip_headers": True,
# }
# my_splitter = partial(Embs.markdown_splitter, config=split_config)

# embs = Embs()

# docs = embs.retrieve_documents(
#     files=["some_markdown.md"],
#     splitter=my_splitter  # this calls the static method with the config
# )
# for d in docs:
#     print(d["filename"], " => length:", len(d["markdown"]))

# # 2) Or define a custom splitter method:
# def custom_line_splitter(docs: List[Dict[str, str]]) -> List[Dict[str, str]]:
#     new_docs = []
#     for doc in docs:
#         lines = doc["markdown"].split("\n")
#         for idx, line in enumerate(lines):
#             if line.strip():
#                 new_docs.append({
#                     "filename": f"{doc['filename']}/{idx}",
#                     "markdown": line
#                 })
#     return new_docs

# split_docs = embs.retrieve_documents(files=["some_markdown.md"], splitter=custom_line_splitter)
# print(split_docs)
