# filename: embs.py

"""
embs.py

A lightweight Python library that streamlines document ingestion,
embedding, and ranking for RAG systems, chatbots, semantic search engines,
and more.

Key features:
- Document conversion via Docsifer (files and URLs)
- Powerful document splitting (e.g., Markdown-based)
- Embedding generation using a lightweight embeddings API
- Ranking and optional embedding inclusion in results (using local reranking)
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
import numpy as np  # For local ranking computations

from aiohttp import FormData
from collections import OrderedDict
from typing import List, Dict, Any, Optional, Union, Callable, Tuple

from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _split_markdown_text(
    text: str,
    headers_to_split_on: List[Tuple[str, str]],
    return_each_line: bool,
    strip_headers: bool,
    max_characters: Optional[int] = None,
    split_on_double_newline: bool = False
) -> List[str]:
    """
    Optimized splitting of a markdown string into chunks based on specified header markers.

    This function iterates through the text line‐by‐line (using splitlines, which is more efficient
    than split("\\n") for varied newline conventions) and:
      - Filters out non-printable characters from each line.
      - Detects code blocks using common fences (~~~ or ```); lines inside a code block are not scanned
        for headers.
      - When a header (one of the specified markers) is detected, the current content block is flushed.
        A header stack is maintained so that nested headers correctly update the metadata.
      - Optionally, if return_each_line is True, each nonblank word (separated by " ") is returned as its own chunk.
      - Otherwise, consecutive blocks sharing the same metadata are merged.
      - Finally, after header splitting:
            * If split_on_double_newline is True, each merged chunk is further split on double newline ("\n\n").
            * Else if max_characters is set (> 0), any chunk longer than max_characters is further split into pieces.

    Args:
        text: Full markdown text.
        headers_to_split_on: A list of (header_prefix, header_name) pairs.
        return_each_line: If True, each nonblank word (separated by " ") is its own chunk.
        strip_headers: If True, header lines are not included in the chunks.
        max_characters: Maximum number of characters per chunk (default=None).
        split_on_double_newline: If True, further split each chunk by double newline ("\n\n")
                                   instead of using max_characters.

    Returns:
        List of chunk strings.
    """
    # Sort header markers in descending order by prefix length so that longer markers match first.
    headers_to_split_on = sorted(headers_to_split_on, key=lambda x: len(x[0]), reverse=True)
    # Pre-calculate header markers: (prefix, header_name, prefix_length, header_level)
    header_markers = [(prefix, name, len(prefix), prefix.count("#")) for prefix, name in headers_to_split_on]

    lines_with_metadata: List[Dict[str, Any]] = []
    current_content: List[str] = []
    header_stack: List[Tuple[str, int]] = []  # Each element is (header_name, header_level)
    current_metadata: Dict[str, str] = {}

    in_code_block = False
    code_fence = ""

    def flush_block() -> None:
        nonlocal current_content
        if current_content:
            # Append the current block (joined by newline) along with a copy of current metadata.
            lines_with_metadata.append({
                "content": "\n".join(current_content),
                "metadata": current_metadata.copy()
            })
            current_content = []

    for line in text.splitlines():
        # Remove leading/trailing whitespace and filter out non-printable characters.
        stripped_line = ''.join(ch for ch in line.strip() if ch.isprintable())

        # Detect code blocks using common fences.
        if not in_code_block:
            if stripped_line.startswith("~~~") or stripped_line.startswith("```"):
                in_code_block = True
                code_fence = stripped_line[:3]
        else:
            if stripped_line.startswith(code_fence):
                in_code_block = False
                code_fence = ""
            current_content.append(stripped_line)
            continue

        # Check if the line is a header.
        header_found = False
        for prefix, name, plen, level in header_markers:
            if stripped_line.startswith(prefix) and (len(stripped_line) == plen or stripped_line[plen] == " "):
                header_found = True
                flush_block()
                # Pop headers from the stack that have a level greater than or equal to the current.
                while header_stack and header_stack[-1][1] >= level:
                    popped_name, _ = header_stack.pop()
                    current_metadata.pop(popped_name, None)
                header_text = stripped_line[plen:].strip()
                header_stack.append((name, level))
                current_metadata[name] = header_text
                if not strip_headers:
                    current_content.append(stripped_line)
                break

        if not header_found:
            if stripped_line:
                current_content.append(stripped_line)
            else:
                flush_block()

    flush_block()

    # If each nonblank word should be its own chunk, split blocks using the space character.
    if return_each_line:
        chunks: List[str] = []
        for block in lines_with_metadata:
            # Use split(" ") to split strictly on the space character.
            for word in block["content"].split(" "):
                if word != "":
                    chunks.append(word)
    else:
        # Merge consecutive blocks that share the same metadata.
        merged: List[str] = []
        if lines_with_metadata:
            current_block = lines_with_metadata[0]["content"]
            current_meta = lines_with_metadata[0]["metadata"]
            for item in lines_with_metadata[1:]:
                if item["metadata"] == current_meta:
                    current_block += "\n" + item["content"]
                else:
                    merged.append(current_block)
                    current_block = item["content"]
                    current_meta = item["metadata"]
            merged.append(current_block)
        else:
            merged = [text]  # Fallback: if nothing was split, return the original text.
        chunks = merged

    # If the flag to split by double newline is enabled, further split each chunk by "\n\n".
    if split_on_double_newline:
        final_chunks: List[str] = []
        for chunk in chunks:
            pieces = chunk.split("\n\n")
            for piece in pieces:
                if piece.strip():
                    final_chunks.append(piece.strip())
        return final_chunks

    # Otherwise, if max_characters is enforced (> 0), further split any chunk that exceeds the limit.
    if max_characters and max_characters > 0:
        final_chunks: List[str] = []
        for chunk in chunks:
            if len(chunk) <= max_characters:
                final_chunks.append(chunk)
            else:
                for i in range(0, len(chunk), max_characters):
                    final_chunks.append(chunk[i:i + max_characters])
        return final_chunks

    return chunks


class Embs:
    """
    A one-stop toolkit for document ingestion, embedding, and ranking workflows.
    
    This library integrates:
      - Docsifer for converting files/URLs to markdown,
      - A lightweight embeddings API for generating text embeddings, and
      - Ranking of documents/chunks based on query relevance.
    
    It supports optional in-memory or disk caching and flexible document splitting.
    
    Final results can optionally include raw embeddings by specifying options={"embeddings": True}.
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
                - return_each_line: bool (if True, each nonblank word is its own chunk).
                - strip_headers: bool.
                - max_characters: int (default=2048)
                - split_on_double_newline: bool (if True, further split each chunk on "\n\n").
        
        Returns:
            A list of documents with subdivided chunks.
        """
        if config is None:
            config = {}
    
        headers_to_split_on = config.get("headers_to_split_on", [("#", "h1"), ("##", "h2"), ("###", "h3")])
        return_each_line = config.get("return_each_line", False)
        strip_headers = config.get("strip_headers", True)
        max_characters = config.get("max_characters", None)
        split_on_double_newline = config.get("split_on_double_newline", False)
    
        output_docs: List[Dict[str, str]] = []
        for doc in docs:
            original_filename = doc["filename"]
            text = doc["markdown"]
            chunks = _split_markdown_text(
                text,
                headers_to_split_on=headers_to_split_on,
                return_each_line=return_each_line,
                strip_headers=strip_headers,
                max_characters=max_characters,
                split_on_double_newline=split_on_double_newline
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
        
        Args:
            name: Name of the method.
            **kwargs: Key/value pairs to include in the key.
        
        Returns:
            A SHA256 hash string representing the cache key.
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
        
        Args:
            key: The cache key.
        
        Returns:
            True if the item was expired and removed, False otherwise.
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
        
        Args:
            key: The cache key.
        
        Returns:
            The cached data if present and not expired, else None.
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
        
        Args:
            key: The cache key.
            data: The data to cache.
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
        
        Args:
            file: A file path or file-like object.
            session: The aiohttp session.
            openai_config: Optional OpenAI configuration.
            settings: Additional settings for Docsifer.
            semaphore: Concurrency semaphore.
            options: Additional options (e.g., {"silent": True}).
        
        Returns:
            A dictionary with keys "filename" and "markdown", or None on error.
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
        
        Args:
            url: The URL to convert.
            session: The aiohttp session.
            openai_config: Optional OpenAI configuration.
            settings: Additional settings for Docsifer.
            semaphore: Concurrency semaphore.
            options: Additional options.
        
        Returns:
            A dictionary with keys "filename" and "markdown", or None on error.
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
        concurrency: int = 2,
        options: Optional[Dict[str, Any]] = None,
        splitter: Optional[Callable[[List[Dict[str, str]]], List[Dict[str, str]]]] = None,
    ) -> List[Dict[str, str]]:
        """
        Asynchronously retrieves documents from Docsifer (via files and/or URLs),
        optionally applies a splitter, and returns a list of documents.
        
        Args:
            files: List of file paths or file-like objects.
            urls: List of URLs.
            openai_config: Optional Docsifer (forward OpenAI) configuration.
            settings: Additional Docsifer settings.
            concurrency: Maximum concurrency for retrieval.
            options: Additional options (e.g., {"silent": True}).
            splitter: Optional callable to further split document content.
        
        Returns:
            A list of documents (each with "filename" and "markdown").
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

    async def _embed_batch(self, texts: List[str], model: str) -> Dict[str, Any]:
        """
        Asynchronously generate embeddings for a batch of texts.
        
        This helper function checks the cache for the given batch. If not cached,
        it calls the embeddings API and caches the result.
        
        Args:
            texts: A list of text strings to embed.
            model: The model name to use.
        
        Returns:
            A dictionary containing the embedding results (with keys "data" and "usage").
        """
        cache_key = None
        if self.cache_enabled:
            cache_key = self._make_key("embed_async", text=texts, model=model)
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                return cached_data

        endpoint = f"{self.embeddings_base_url}{self.embeddings_endpoint}"
        payload = {"model": model, "input": texts}

        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()

        if self.cache_enabled and cache_key:
            self._save_to_cache(cache_key, data)
        return data

    async def embed_async(
        self,
        text_or_texts: Union[str, List[str]],
        model: Optional[str] = None,
        optimized: bool = True
    ) -> Dict[str, Any]:
        """
        Asynchronously generates embeddings for the provided text(s).
        
        When `optimized` is True, the batch size is set to 1 (each text is processed
        individually) to maximize cache hits and reduce load. When `optimized` is False,
        texts are grouped in batches of up to 4 items per API call.
        
        Args:
            text_or_texts: A string or a list of text strings.
            model: The model name to use (defaults to self.default_model if not provided).
            optimized: Flag to control per-item (True) versus batched (False) processing.
        
        Returns:
            A dictionary with keys "data", "model", and "usage". "data" holds a list of embeddings,
            and "usage" contains token count information.
        """
        if model is None:
            model = self.default_model

        # Normalize input to a list.
        if isinstance(text_or_texts, str):
            texts = [text_or_texts]
        else:
            texts = text_or_texts

        # Set batch size based on optimization flag.
        batch_size = 1 if optimized else 4
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

        combined_data = []
        total_tokens = 0

        # Process each batch asynchronously.
        tasks = [self._embed_batch(batch, model) for batch in batches]
        results = await asyncio.gather(*tasks)
        for res in results:
            if "data" in res:
                combined_data.extend(res["data"])
            if "usage" in res:
                usage = res["usage"]
                total_tokens += usage.get("total_tokens", 0)

        return {
            "data": combined_data,
            "model": model,
            "usage": {"total_tokens": total_tokens, "prompt_tokens": total_tokens}
        }

    async def rank_async(
        self,
        query: str,
        candidates: List[str],
        model: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Asynchronously ranks candidate texts by relevance to the given query using embeddings.
        It generates embeddings for the query and candidates via the /embeddings endpoint,
        computes cosine similarity, and applies softmax to obtain probabilities.
        
        Args:
            query: The query string.
            candidates: A list of candidate texts.
            model: The model name (defaults to self.default_model if not provided).
        
        Returns:
            A list of ranking dictionaries with keys "text", "probability", and "similarity".
        """
        if model is None:
            model = self.default_model
            
        if len(candidates) == 0:
            return []

        # Generate embeddings for the query.
        query_embed_response = await self.embed_async(query, model=model)
        query_embeds = query_embed_response.get("data")
        if not query_embeds:
            raise ValueError("Failed to generate embeddings for the query.")
        # If each embedding is a dict, extract the numeric vector.
        if isinstance(query_embeds[0], dict):
            query_embeds = np.array([x["embedding"] for x in query_embeds])
        elif isinstance(query_embeds[0], (float, int)):
            query_embeds = np.array([query_embeds])
        else:
            query_embeds = np.array(query_embeds)
        
        # Generate embeddings for candidates.
        candidate_embed_response = await self.embed_async(candidates, model=model)
        candidate_embeds = candidate_embed_response.get("data")
        if candidate_embeds is None or not candidate_embeds:
            raise ValueError("Failed to generate embeddings for candidates.")
        if isinstance(candidate_embeds[0], dict):
            candidate_embeds = np.array([x["embedding"] for x in candidate_embeds])
        else:
            candidate_embeds = np.array(candidate_embeds)
        
        # Compute cosine similarity.
        sim_matrix = self.similarity(query_embeds, candidate_embeds)
        logit_scale = 1.0
        scaled = sim_matrix * logit_scale
        probs = self.softmax(scaled)
        
        # Estimate token usage.
        query_tokens = self.estimate_tokens(query)
        candidate_tokens = sum(self.estimate_tokens(text) for text in candidates)
        total_tokens = query_tokens + candidate_tokens
        usage = {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens,
        }
        
        # Prepare ranking results (assuming single query).
        results = []
        for i, text_val in enumerate(candidates):
            results.append({
                "text": text_val,
                "probability": probs[0][i],
                "similarity": sim_matrix[0][i],
            })
        
        results.sort(key=lambda x: x["probability"], reverse=True)
        
        return results

    async def query_documents_async(
        self,
        query: str,
        files: Optional[List[Any]] = None,
        urls: Optional[List[str]] = None,
        openai_config: Optional[Dict[str, Any]] = None,
        settings: Optional[Dict[str, Any]] = None,
        concurrency: int = 2,
        options: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        splitter: Optional[Callable[[List[Dict[str, str]]], List[Dict[str, str]]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Asynchronously retrieves documents from Docsifer (via files/URLs), ranks them by relevance
        to the query, and returns a list of documents with ranking scores. Optionally, if options contains
        {"embeddings": True}, the function attaches an embedding for each document.
        
        Args:
            query: The query to rank against.
            files: List of file paths or file-like objects.
            urls: List of URLs to convert.
            openai_config: Optional Docsifer (forward OpenAI) configuration.
            settings: Additional Docsifer settings.
            concurrency: Maximum concurrency for retrieval.
            options: Additional options (e.g., {"embeddings": True}).
            model: Model name for ranking (defaults to self.default_model).
            splitter: Optional callable to further split document content.
        
        Returns:
            A list of documents with keys "filename", "markdown", "probability", "similarity",
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
                    "similarity": item["similarity"]
                })

        # Attach embeddings if requested.
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
        concurrency: int = 2,
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
            query: The search query.
            limit: Maximum number of search results.
            blocklist: Optional list of domain substrings to filter out.
        
        Returns:
            A list of URLs.
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
        limit: int = 5,
        blocklist: Optional[List[str]] = None,
        openai_config: Optional[Dict[str, Any]] = None,
        settings: Optional[Dict[str, Any]] = None,
        concurrency: int = 2,
        options: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        splitter: Optional[Callable[[List[Dict[str, str]]], List[Dict[str, str]]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Asynchronously searches for documents using DuckDuckGo to obtain URLs based on a keyword,
        then retrieves and ranks their content using Docsifer and the ranking API.
        
        Args:
            query: The search keyword.
            limit: Maximum number of DuckDuckGo results.
            blocklist: Optional list of domain substrings to filter out.
            openai_config: Optional Docsifer (forward OpenAI) configuration.
            settings: Additional Docsifer settings.
            concurrency: Maximum concurrency for retrieval.
            options: Additional options (e.g., {"embeddings": True}).
            model: Model name for ranking (defaults to self.default_model).
            splitter: Optional callable to further split document content.
        
        Returns:
            A list of ranked documents with keys "filename", "markdown", "probability",
            "similarity", and optionally "embeddings".
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
        limit: int = 5,
        blocklist: Optional[List[str]] = None,
        openai_config: Optional[Dict[str, Any]] = None,
        settings: Optional[Dict[str, Any]] = None,
        concurrency: int = 2,
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

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Computes the pairwise cosine similarity between all rows of a and b.
        
        Args:
            a: A NumPy array of shape (N, D).
            b: A NumPy array of shape (M, D).
        
        Returns:
            A (N x M) matrix of cosine similarities.
        """
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
        return np.dot(a_norm, b_norm.T)

    @staticmethod
    def softmax(scores: np.ndarray) -> np.ndarray:
        """
        Applies the standard softmax function along the last dimension.
        
        Args:
            scores: A NumPy array of scores.
        
        Returns:
            A NumPy array of softmax probabilities.
        """
        exps = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        return exps / np.sum(exps, axis=-1, keepdims=True)

    @staticmethod
    def estimate_tokens(input_data: Union[str, List[str]]) -> int:
        """
        Estimates token count for the input text(s) by counting words separated by a space (" ").
        
        Args:
            input_data: A string or a list of strings.
        
        Returns:
            Estimated token count as an integer.
        """
        if isinstance(input_data, str):
            return len([w for w in input_data.split(" ") if w != ""])
        elif isinstance(input_data, list):
            return sum(len([w for w in text.split(" ") if w != ""]) for text in input_data)
        else:
            return 0

# =============================================================================
# Example usage:
# -----------------------------------------------------------------------------
# 1) Using the built-in markdown splitter:
#
# from functools import partial
#
# split_config = {
#     "headers_to_split_on": [("#", "h1"), ("##", "h2"), ("###", "h3")],
#     "return_each_line": True,   # Each nonblank word (split by " ") becomes its own chunk.
#     "strip_headers": True,
#     "split_on_double_newline": True  # Further split each header chunk by double newline ("\n\n")
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
#     splitter=md_splitter,
#     options={"embeddings": True}
# )
# for item in results:
#     print(f"File: {item['filename']} | Score: {item['probability']:.4f}")
#     print(f"Snippet: {item['markdown'][:80]}...\n")
# =============================================================================
