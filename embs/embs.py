import os
import json
import time
import hashlib
import logging
import asyncio
import aiohttp

from aiohttp import FormData
from collections import OrderedDict
from typing import List, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

class Embs:
    """
    Provides document retrieval from Docsifer, embeddings with the Lightweight Embeddings API,
    and ranking of documents by query relevance. Caching can be configured in memory or on disk,
    with both time-based TTL and optional LRU eviction for in-memory items.
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
        Initializes the Embs instance with URLs, default model, and cache configuration.

        :param docsifer_base_url: Base URL for Docsifer service.
        :param docsifer_endpoint: Endpoint path for Docsifer document conversion.
        :param embeddings_base_url: Base URL for the Lightweight Embeddings service.
        :param embeddings_endpoint: Endpoint path for generating embeddings.
        :param rank_endpoint: Endpoint path for ranking.
        :param default_model: Default model for embeddings and ranking if none specified.
        :param cache_config: Dictionary controlling caching behavior. Example:
            {
              "enabled": True,
              "type": "memory",        # or "disk"
              "prefix": "mycache",     # optional prefix for keys
              "dir": "my_cache_dir",   # required if type="disk"
              "max_mem_items": 128,    # max items in memory LRU
              "max_ttl_seconds": 259200 # item lifetime in seconds (3 days default)
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

        # Cache-related config
        self.cache_enabled: bool = cache_config.get("enabled", False)
        self.cache_type: str = cache_config.get("type", "memory").lower()
        self.cache_prefix: str = cache_config.get("prefix", "")  # used to namespace cache keys
        self.cache_dir: Optional[str] = cache_config.get("dir")  # disk path
        self.max_mem_items: int = cache_config.get("max_mem_items", 128)
        self.max_ttl_seconds: int = cache_config.get("max_ttl_seconds", 259200)  # default ~3 days

        if self.cache_type not in ("memory", "disk"):
            raise ValueError('cache_config["type"] must be either "memory" or "disk".')

        # In-memory LRU cache: key -> (timestamp, data)
        self._mem_cache: "OrderedDict[str, (float, Any)]" = OrderedDict()

        if self.cache_enabled and self.cache_type == "disk":
            if not self.cache_dir:
                raise ValueError('If "type"=="disk", you must provide "dir" in cache_config.')
            os.makedirs(self.cache_dir, exist_ok=True)

    def _make_key(self, name: str, **kwargs) -> str:
        """
        Builds a cache key by hashing the method name, optional prefix, and sorted kwargs.
        File-like objects are replaced with placeholders to avoid reading file data.
        
        :param name: Name of the method or feature generating the key.
        :param kwargs: Arguments that determine cache uniqueness.
        :return: A hex-encoded SHA-256 string used as the cache key.
        """
        safe_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, list):
                safe_kwargs[k] = []
                for item in v:
                    if isinstance(item, str):
                        safe_kwargs[k].append(item)
                    else:
                        safe_kwargs[k].append(f"<file_obj:{id(item)}>")
            else:
                safe_kwargs[k] = v

        raw_str = f"{self.cache_prefix}:{name}-{json.dumps(safe_kwargs, sort_keys=True)}"
        return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()

    def _evict_memory_cache_if_needed(self) -> None:
        """
        Removes the least recently used item if the in-memory cache exceeds max_mem_items.
        """
        while len(self._mem_cache) > self.max_mem_items:
            key, _ = self._mem_cache.popitem(last=False)  # LRU is first
            logger.debug(f"Evicted LRU item from memory cache: {key}")

    def _check_expiry_in_memory(self, key: str) -> bool:
        """
        Checks if an item in the memory cache is too old based on max_ttl_seconds.
        Removes it if expired.

        :param key: The cache key to verify.
        :return: True if the item was removed due to expiration, else False.
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

        :param key: The cache key.
        :return: The cached data, or None if not found or expired.
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

        :param key: The cache key.
        :param data: The data to store.
        """
        if not self.cache_enabled:
            return

        if self.cache_type == "memory":
            timestamp_data = (time.time(), data)
            if key in self._mem_cache:
                self._mem_cache.pop(key)
            self._mem_cache[key] = timestamp_data
            self._evict_memory_cache_if_needed()
        else:  # disk
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
        Uploads a single file to Docsifer asynchronously. Returns {"filename": str, "markdown": str} on success.
        Logs errors instead of raising if silent=True in options.
        
        :param file: File path (str) or file-like object.
        :param session: Aiohttp session used for sending the POST request.
        :param openai_config: Additional LLM-based extraction config for Docsifer.
        :param settings: Additional Docsifer settings, e.g. {"cleanup": True}.
        :param semaphore: Concurrency limit.
        :param options: Dict with optional "silent": bool to silence/raise exceptions.
        :return: A dictionary with "filename" and "markdown", or None if an error is logged in silent mode.
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
        Submits a URL to Docsifer for HTML->Markdown conversion. Returns {"filename": str, "markdown": str}.
        Logs errors instead of raising if silent=True in options.
        
        :param url: The URL to process.
        :param session: Aiohttp session used for sending the POST request.
        :param openai_config: Additional LLM-based extraction config for Docsifer.
        :param settings: Additional Docsifer settings, e.g. {"cleanup": True}.
        :param semaphore: Concurrency limit.
        :param options: Dict with optional "silent": bool to silence/raise exceptions.
        :return: A dictionary with "filename" and "markdown", or None if an error is logged in silent mode.
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
    ) -> List[Dict[str, str]]:
        """
        Asynchronously retrieves documents from Docsifer (via files and/or URLs).
        Returns a list of dicts with "filename" and "markdown". Respects concurrency
        and can be silent on errors if specified.

        :param files: A list of file paths or file-like objects.
        :param urls: A list of URLs to process.
        :param openai_config: Additional config for Docsifer's LLM-based extraction.
        :param settings: Additional Docsifer settings.
        :param concurrency: Maximum number of concurrent uploads.
        :param options: Dict with optional "silent": bool for error handling.
        :return: A list of {"filename": <str>, "markdown": <str>} items.
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
                options=options
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

        if self.cache_enabled and cache_key:
            self._save_to_cache(cache_key, all_docs)
        return all_docs

    async def embed_async(
        self,
        text_or_texts: Union[str, List[str]],
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Asynchronously generates embeddings for the given text(s) using the Lightweight Embeddings API.
        Employs caching if enabled.

        :param text_or_texts: Single string or list of strings to embed.
        :param model: Embedding model name. Uses default_model if None.
        :return: The JSON response from the embeddings endpoint.
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
        Asynchronously ranks candidate texts against a single query using the rank endpoint.
        Sorts them by probability descending. Caches results if enabled.

        :param query: The query string.
        :param candidates: List of candidate strings to be ranked.
        :param model: Ranking model name. Uses default_model if None.
        :return: A sorted list of dicts: { "text": ..., "probability": ..., "cosine_similarity": ... }.
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

    async def search_documents_async(
        self,
        query: str,
        files: Optional[List[Any]] = None,
        urls: Optional[List[str]] = None,
        openai_config: Optional[Dict[str, Any]] = None,
        settings: Optional[Dict[str, Any]] = None,
        concurrency: int = 5,
        options: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Asynchronously retrieves documents from Docsifer, then ranks them by 'query'.
        Returns a list where each item includes filename, markdown, probability, and cosine_similarity.

        :param query: The query to rank documents against.
        :param files: A list of file paths or file-like objects.
        :param urls: A list of URLs to process.
        :param openai_config: Docsifer LLM-based config if needed.
        :param settings: Additional Docsifer settings.
        :param concurrency: Maximum concurrency for docs retrieval.
        :param options: Dict with optional "silent": bool for error handling.
        :param model: Ranking model name. Uses default_model if None.
        :return: A sorted list of documents with ranking fields attached.
        """
        docs = await self.retrieve_documents_async(files, urls, openai_config, settings, concurrency, options)
        if not docs:
            return []

        candidates = [doc["markdown"] for doc in docs]
        ranking = await self.rank_async(query, candidates, model)

        text_to_indices: Dict[str, List[int]] = {}
        for i, d_obj in enumerate(docs):
            text_to_indices.setdefault(d_obj["markdown"], []).append(i)

        results = []
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
                doc_matched = docs[matched_idx]
                results.append({
                    "filename": doc_matched["filename"],
                    "markdown": doc_matched["markdown"],
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
        options: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, str]]:
        """
        Synchronous wrapper for retrieve_documents_async.
        Fetches documents from Docsifer and returns them in a list of {"filename": ..., "markdown": ...}.

        :param files: List of file paths or file-like objects.
        :param urls: List of URLs.
        :param openai_config: Config for Docsifer LLM usage.
        :param settings: Additional Docsifer settings.
        :param concurrency: Maximum concurrency for the retrieval process.
        :param options: Dict with optional "silent": bool for error handling.
        :return: A list of docs from Docsifer.
        """
        return asyncio.run(
            self.retrieve_documents_async(files, urls, openai_config, settings, concurrency, options)
        )

    def embed(
        self,
        text_or_texts: Union[str, List[str]],
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for embed_async.
        Generates embeddings for text(s) using the Lightweight Embeddings API.

        :param text_or_texts: String or list of strings to embed.
        :param model: Embedding model name or None for default.
        :return: JSON response containing embeddings and usage stats.
        """
        return asyncio.run(self.embed_async(text_or_texts, model))

    def rank(
        self,
        query: str,
        candidates: List[str],
        model: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Synchronous wrapper for rank_async.
        Ranks candidate strings by relevance to a query using the Lightweight Embeddings API.

        :param query: The query text.
        :param candidates: A list of candidate strings.
        :param model: Model name or None for default.
        :return: A list of dicts sorted by probability descending.
        """
        return asyncio.run(self.rank_async(query, candidates, model))

    def search_documents(
        self,
        query: str,
        files: Optional[List[Any]] = None,
        urls: Optional[List[str]] = None,
        openai_config: Optional[Dict[str, Any]] = None,
        settings: Optional[Dict[str, Any]] = None,
        concurrency: int = 5,
        options: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Synchronous wrapper for search_documents_async.
        Retrieves documents from Docsifer, then ranks them by the given query.

        :param query: The query to rank documents against.
        :param files: File paths or file-like objects for Docsifer.
        :param urls: URLs for Docsifer.
        :param openai_config: Docsifer LLM-based config if necessary.
        :param settings: Additional Docsifer parameters.
        :param concurrency: Maximum concurrency for the retrieval.
        :param options: Dict with optional "silent": bool for error handling.
        :param model: Model name for ranking or None for default.
        :return: A list of ranked documents, each with "filename", "markdown", "probability", and "cosine_similarity".
        """
        return asyncio.run(
            self.search_documents_async(query, files, urls, openai_config, settings, concurrency, options, model)
        )
