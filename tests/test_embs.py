# filename: test_embs.py

import pytest
import asyncio
import os
import time
from unittest.mock import patch, mock_open

from embs.embs import Embs


@pytest.mark.asyncio
async def test_retrieve_documents_async_no_input():
    """
    Checks that retrieve_documents_async returns an empty list
    when no files or URLs are provided.
    """
    embs = Embs()
    results = await embs.retrieve_documents_async(files=None, urls=None)
    assert results == [], "Expected an empty list when no input is provided."


@pytest.mark.asyncio
@patch("builtins.open", new_callable=mock_open, read_data=b"Fake file data")
async def test_retrieve_documents_async_single_file(mock_file):
    """
    Mocks reading a single file and the HTTP post so that no real I/O
    or network request is made.
    """
    def mock_post(*args, **kwargs):
        class MockContextManager:
            def __init__(self):
                self._data = {"filename": "dummy.pdf", "markdown": "Dummy content"}
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass
            def raise_for_status(self):
                pass
            async def json(self):
                return self._data
        return MockContextManager()

    embs = Embs()
    with patch("aiohttp.ClientSession.post", side_effect=mock_post):
        docs = await embs.retrieve_documents_async(files=["/path/to/file.pdf"])
        assert len(docs) == 1
        assert docs[0]["filename"] == "dummy.pdf"
        assert docs[0]["markdown"] == "Dummy content"


@pytest.mark.asyncio
async def test_embed_async_in_memory_cache():
    """
    Tests embed_async with in-memory caching.
    Mocks network calls to avoid real requests.
    """
    def mock_post(*args, **kwargs):
        class MockContextManager:
            def __init__(self):
                self._data = {
                    "object": "list",
                    "data": [{"object": "embedding", "index": 0, "embedding": [0.25, 0.75]}],
                    "model": "test-model",
                    "usage": {"prompt_tokens": 2, "total_tokens": 2}
                }
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass
            def raise_for_status(self):
                pass
            async def json(self):
                return self._data
        return MockContextManager()

    cache_conf = {
        "enabled": True,
        "type": "memory",
        "prefix": "test",
        "max_mem_items": 2,
        "max_ttl_seconds": 3600
    }
    embs = Embs(cache_config=cache_conf)

    with patch("aiohttp.ClientSession.post", side_effect=mock_post) as mock_method:
        text_data = "Hello"
        resp1 = await embs.embed_async(text_data, model="test-model")
        resp2 = await embs.embed_async(text_data, model="test-model")
        assert resp1 == resp2, "Expected the second embedding call to use cached data."
        assert mock_method.call_count == 1, "Second call should be served from cache."


@pytest.mark.asyncio
async def test_embed_async_lru_eviction():
    """
    Tests that the in-memory LRU cache evicts the oldest entry when capacity is exceeded.
    Distinct responses for different texts confirm eviction.
    """
    def mock_post_factory(embed_val):
        def mock_post(*args, **kwargs):
            class MockContextManager:
                def __init__(self):
                    self._data = {"object": "list", "data": [{"embedding": embed_val}]}
                async def __aenter__(self):
                    return self
                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    pass
                def raise_for_status(self):
                    pass
                async def json(self):
                    return self._data
            return MockContextManager()
        return mock_post

    cache_conf = {
        "enabled": True,
        "type": "memory",
        "prefix": "lru",
        "max_mem_items": 2,
        "max_ttl_seconds": 9999
    }
    embs = Embs(cache_config=cache_conf)

    # Request embedding for "Text1" => [1, 1]
    with patch("aiohttp.ClientSession.post", side_effect=mock_post_factory([1, 1])):
        emb1 = await embs.embed_async("Text1", model="lru-model")
    # Request embedding for "Text2" => [2, 2]
    with patch("aiohttp.ClientSession.post", side_effect=mock_post_factory([2, 2])):
        emb2 = await embs.embed_async("Text2", model="lru-model")
    # Request embedding for "Text3" => [3, 3] which should evict "Text1"
    with patch("aiohttp.ClientSession.post", side_effect=mock_post_factory([3, 3])):
        emb3 = await embs.embed_async("Text3", model="lru-model")
    # Re-request "Text1" => should trigger a new call returning [9, 9]
    with patch("aiohttp.ClientSession.post", side_effect=mock_post_factory([9, 9])) as post_mock:
        emb1_again = await embs.embed_async("Text1", model="lru-model")
        assert post_mock.call_count == 1, "Evicted 'Text1' should trigger a new network call."
    assert emb1 != emb1_again, "Emb1 differs from new data, confirming LRU eviction."


@pytest.mark.asyncio
async def test_rank_async_mock():
    """
    Mocks the rank API call to confirm that rank_async sorts candidates
    by descending probability.
    """
    def mock_post(*args, **kwargs):
        class MockContextManager:
            def __init__(self):
                self._data = {
                    "probabilities": [[0.2, 0.8]],
                    "cosine_similarities": [[0.1, 0.9]]
                }
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass
            def raise_for_status(self):
                pass
            async def json(self):
                return self._data
        return MockContextManager()

    embs = Embs()
    with patch("aiohttp.ClientSession.post", side_effect=mock_post):
        candidates = ["candidate1", "candidate2"]
        ranked = await embs.rank_async("test query", candidates)
        assert len(ranked) == 2
        # "candidate2" (probability=0.8) should come first.
        assert ranked[0]["text"] == "candidate2"
        assert ranked[0]["probability"] == 0.8


@pytest.mark.asyncio
async def test_rank_async_empty_candidates():
    """
    Ensures that when an empty list of candidates is provided,
    rank_async returns an empty list.
    """
    def mock_post(*args, **kwargs):
        class MockContext:
            def __init__(self):
                self._data = {"probabilities": [[]], "cosine_similarities": [[]]}
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass
            def raise_for_status(self):
                pass
            async def json(self):
                return self._data
        return MockContext()

    embs = Embs()
    with patch("aiohttp.ClientSession.post", side_effect=mock_post) as post_mock:
        ranked = await embs.rank_async("test query", [])
        # Even with empty candidates, the endpoint is called.
        assert post_mock.call_count == 1, "Expected a call even with empty candidates."
        assert ranked == [], "Empty candidates should yield an empty ranking."


# @pytest.mark.asyncio
# async def test_query_documents_async_integration():
#     """
#     Tests query_documents_async by simulating Docsifer conversion (using files)
#     and ranking. Verifies that the returned documents are correctly ranked.
#     """
#     # Simulated responses from Docsifer.
#     doc_resp_1 = {"filename": "doc1.pdf", "markdown": "Doc1 content"}
#     doc_resp_2 = {"filename": "doc2.pdf", "markdown": "Doc2 content"}
#     # Simulated ranking: doc2 has higher probability.
#     rank_resp = {
#         "probabilities": [[0.1, 0.9]],
#         "cosine_similarities": [[0.2, 0.8]]
#     }

#     def mock_post(*args, **kwargs):
#         class MockContextManager:
#             def __init__(self, data):
#                 self._data = data
#             async def __aenter__(self):
#                 return self
#             async def __aexit__(self, exc_type, exc_val, exc_tb):
#                 pass
#             def raise_for_status(self):
#                 pass
#             async def json(self):
#                 return self._data

#         payload = kwargs.get("json", {})
#         if "candidates" in payload:
#             return MockContextManager(rank_resp)
#         # For Docsifer conversion calls, alternate responses.
#         if not hasattr(mock_post, "call_count"):
#             mock_post.call_count = 0
#         if mock_post.call_count == 0:
#             mock_post.call_count += 1
#             return MockContextManager(doc_resp_1)
#         return MockContextManager(doc_resp_2)

#     embs = Embs()
#     with patch("aiohttp.ClientSession.post", side_effect=mock_post):
#         results = await embs.query_documents_async(
#             query="sample query",
#             files=["/path/file1.pdf", "/path/file2.pdf"]
#         )
#         assert len(results) == 2, "Expected 2 ranked documents."
#         # With higher probability, doc2 should be ranked first.
#         assert results[0]["filename"] == "doc2.pdf"
#         assert results[0]["probability"] == 0.9


@pytest.mark.asyncio
async def test_search_documents_async_duckduckgo_integration():
    """
    Tests search_documents_async (DuckDuckGo-based search) by mocking
    _duckduckgo_search to return fixed URLs, and then simulating Docsifer conversion and ranking.
    """
    # Fake URLs returned by DuckDuckGo.
    fake_urls = ["http://example.com/doc1", "http://example.com/doc2"]
    # Simulated Docsifer conversion responses.
    doc_resp_1 = {"filename": "doc1.html", "markdown": "Content from doc1"}
    doc_resp_2 = {"filename": "doc2.html", "markdown": "Content from doc2"}
    # Simulated ranking: doc2 is ranked higher.
    rank_resp = {
        "probabilities": [[0.3, 0.7]],
        "cosine_similarities": [[0.4, 0.6]]
    }

    def mock_post(*args, **kwargs):
        class MockContextManager:
            def __init__(self, data):
                self._data = data
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass
            def raise_for_status(self):
                pass
            async def json(self):
                return self._data

        payload = kwargs.get("json", {})
        if "candidates" in payload:
            return MockContextManager(rank_resp)
        if not hasattr(mock_post, "call_count"):
            mock_post.call_count = 0
        if mock_post.call_count == 0:
            mock_post.call_count += 1
            return MockContextManager(doc_resp_1)
        return MockContextManager(doc_resp_2)

    embs = Embs()
    # Patch _duckduckgo_search to return our fake URLs.
    with patch.object(embs, "_duckduckgo_search", return_value=fake_urls):
        with patch("aiohttp.ClientSession.post", side_effect=mock_post):
            results = await embs.search_documents_async(
                query="test duckduckgo",
                limit=5,
                blocklist=["blockeddomain.com"]
            )
            assert len(results) == 2, "Expected 2 ranked documents from DuckDuckGo integration."
            # The doc with higher ranking should appear first.
            assert results[0]["filename"] == "doc2.html"
            assert results[0]["probability"] == 0.7


def test_sync_wrapper_search_documents():
    """
    Verifies that the synchronous search_documents method properly wraps its async counterpart.
    """
    embs = Embs()
    fake_return = [
        {"filename": "mock.pdf", "markdown": "demo", "probability": 1.0, "cosine_similarity": 0.95}
    ]

    async def mock_async(*args, **kwargs):
        return fake_return

    with patch.object(embs, "search_documents_async", side_effect=mock_async):
        outcome = embs.search_documents(query="hello world")
        assert outcome == fake_return, "Sync method should return the same data as async version."


def test_sync_wrapper_query_documents():
    """
    Verifies that the synchronous query_documents method properly wraps its async counterpart.
    """
    embs = Embs()
    fake_return = [
        {"filename": "mock_query.pdf", "markdown": "demo", "probability": 0.9, "cosine_similarity": 0.85}
    ]

    async def mock_async(*args, **kwargs):
        return fake_return

    with patch.object(embs, "query_documents_async", side_effect=mock_async):
        outcome = embs.query_documents(query="sync query")
        assert outcome == fake_return, "Sync query_documents should return the same data as the async version."


@pytest.mark.asyncio
async def test_duckduckgo_search():
    """
    Tests the internal _duckduckgo_search helper method by patching DDGS to return
    a fixed set of search results.
    """
    fake_results = [
        {"href": "http://example.com/1", "title": "Title 1", "body": "Snippet 1"},
        {"href": "http://example.com/2", "title": "Title 2", "body": "Snippet 2"},
        {"href": "http://blockeddomain.com/3", "title": "Title 3", "body": "Snippet 3"},
    ]

    # Create a fake DDGS context manager.
    class FakeDDGS:
        def __init__(self):
            pass
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
        def text(self, query, safesearch, max_results, backend, region):
            # Return an iterator (or generator) over our fake results.
            return fake_results

    embs = Embs()
    # Patch the DDGS imported in the _duckduckgo_search method.
    with patch("embs.embs.DDGS", return_value=FakeDDGS()):
        # Call _duckduckgo_search with a blocklist to filter out blockeddomain.com.
        urls = await embs._duckduckgo_search(query="test", limit=5, blocklist=["blockeddomain.com"])
        # We expect only the URLs not matching the blocklist.
        assert urls == ["http://example.com/1", "http://example.com/2"], "Filtered URLs do not match expected."


@pytest.mark.asyncio
async def test_disk_cache(tmp_path):
    """
    Verifies disk caching by storing and retrieving a mocked embedding.
    Ensures a cache file (.json) is created in the temporary directory.
    """
    def mock_post(*args, **kwargs):
        class MockContextManager:
            def __init__(self):
                self._data = {
                    "object": "list",
                    "data": [{"object": "embedding", "index": 0, "embedding": [1.0, 2.0]}],
                    "model": "disk-test-model"
                }
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass
            def raise_for_status(self):
                pass
            async def json(self):
                return self._data
        return MockContextManager()

    cache_conf = {
        "enabled": True,
        "type": "disk",
        "prefix": "diskcache",
        "dir": str(tmp_path),
        "max_mem_items": 5,
        "max_ttl_seconds": 999999
    }
    embs = Embs(cache_config=cache_conf)

    with patch("aiohttp.ClientSession.post", side_effect=mock_post) as mock_method:
        text_data = ["hello", "world"]
        first_resp = await embs.embed_async(text_data, model="disk-test-model")
        second_resp = await embs.embed_async(text_data, model="disk-test-model")
        # Second call should use the cache.
        assert first_resp == second_resp
        assert mock_method.call_count == 1
        files_in_dir = os.listdir(tmp_path)
        assert any(fname.endswith(".json") for fname in files_in_dir), "Expected a .json cache file on disk."


@pytest.mark.asyncio
async def test_disk_cache_expiry(tmp_path):
    """
    Tests disk cache TTL behavior by patching time.time.
    When the cached item is older than TTL, a new network call should be made.
    """
    def mock_post(*args, **kwargs):
        class MockContextManager:
            def __init__(self):
                self._data = {
                    "object": "list",
                    "data": [{"object": "embedding", "index": 0, "embedding": [9.9, 9.8]}],
                    "model": "disk-expiry-test"
                }
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass
            def raise_for_status(self):
                pass
            async def json(self):
                return self._data
        return MockContextManager()

    cache_conf = {
        "enabled": True,
        "type": "disk",
        "prefix": "diskexpiry",
        "dir": str(tmp_path),
        "max_mem_items": 5,
        "max_ttl_seconds": 300
    }
    embs = Embs(cache_config=cache_conf)

    with patch("aiohttp.ClientSession.post", side_effect=mock_post) as mock_method, \
         patch("time.time") as time_mock:
        t0 = 100000
        time_mock.return_value = t0
        data1 = await embs.embed_async("Disk expiry text")
        assert mock_method.call_count == 1
        time_mock.return_value = t0 + 301  # Exceed TTL
        data2 = await embs.embed_async("Disk expiry text")
        assert mock_method.call_count == 2, "Cache expired, so a new network call should be made."
        assert data1 == data2


@pytest.mark.asyncio
@patch("builtins.open", new_callable=mock_open, read_data=b"Fake file data")
async def test_retrieve_documents_async_multiple_files_concurrent(mock_file):
    """
    Demonstrates concurrency by processing multiple files with a limited concurrency level.
    Verifies that each document has a unique filename.
    """
    def mock_post(*args, **kwargs):
        class MockContextManager:
            def __init__(self, idx):
                self._data = {"filename": f"file{idx}.pdf", "markdown": f"Content {idx}"}
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass
            def raise_for_status(self):
                pass
            async def json(self):
                return self._data
        call_idx = getattr(mock_post, "counter", 0)
        mock_post.counter = call_idx + 1
        return MockContextManager(call_idx)

    embs = Embs()
    with patch("aiohttp.ClientSession.post", side_effect=mock_post):
        files = [f"/path/to/file{i}.pdf" for i in range(5)]
        docs = await embs.retrieve_documents_async(files=files, concurrency=2)
        assert len(docs) == 5, "Expected 5 documents."
        filenames = [d["filename"] for d in docs]
        assert len(set(filenames)) == 5, "Each document should have a unique filename."
