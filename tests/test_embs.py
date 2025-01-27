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
    Checks retrieve_documents_async returns an empty list if both
    files and urls are None or empty.
    """
    embs = Embs()
    results = await embs.retrieve_documents_async(files=None, urls=None)
    assert results == [], "Expected an empty list when no input is provided."


@pytest.mark.asyncio
@patch("builtins.open", new_callable=mock_open, read_data=b"Fake file data")
async def test_retrieve_documents_async_single_file(mock_file):
    """
    Mocks reading a single file and the post call so no real I/O or network request is made.
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
    Tests embed_async with in-memory cache. Mocks network calls to avoid real requests.
    """
    def mock_post(*args, **kwargs):
        class MockContextManager:
            def __init__(self):
                self._data = {
                    "object": "list",
                    "data": [
                        {"object": "embedding", "index": 0, "embedding": [0.25, 0.75]}
                    ],
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
    Test that in-memory LRU cache evicts the oldest item once max_mem_items is exceeded.
    We'll provide distinct embeddings to confirm a new embed occurs after eviction.
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

    # 1) "Text1" => embed [1,1]
    with patch("aiohttp.ClientSession.post", side_effect=mock_post_factory([1,1])):
        emb1 = await embs.embed_async("Text1", model="lru-model")
    # 2) "Text2" => embed [2,2]
    with patch("aiohttp.ClientSession.post", side_effect=mock_post_factory([2,2])):
        emb2 = await embs.embed_async("Text2", model="lru-model")
    # 3) "Text3" => embed [3,3], evicts "Text1"
    with patch("aiohttp.ClientSession.post", side_effect=mock_post_factory([3,3])):
        emb3 = await embs.embed_async("Text3", model="lru-model")

    # "Text1" re-request => new embed [9,9] => proves eviction
    with patch("aiohttp.ClientSession.post", side_effect=mock_post_factory([9,9])) as post_mock:
        emb1_again = await embs.embed_async("Text1", model="lru-model")
        assert post_mock.call_count == 1, "Evicted 'Text1' => new network call"
    assert emb1 != emb1_again, "Emb1 differs from new data => LRU eviction worked."


@pytest.mark.asyncio
async def test_rank_async_mock():
    """
    Mocks the rank API call to confirm rank_async sorts candidates by descending probability.
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
        # second candidate has probability=0.8 => first
        assert ranked[0]["text"] == "candidate2"
        assert ranked[0]["probability"] == 0.8


@pytest.mark.asyncio
async def test_rank_async_empty_candidates():
    """
    If candidates is an empty list, rank_async STILL calls the endpoint by default. 
    We'll mock it to return an empty rank result. 
    If you want truly zero calls, add a short-circuit in rank_async: if not candidates: return [].
    """
    def mock_post(*args, **kwargs):
        class MockContext:
            def __init__(self):
                self._data = {
                    "probabilities": [[]],
                    "cosine_similarities": [[]]
                }
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
        # If your code doesn't short-circuit, we get one call with empty candidates
        assert post_mock.call_count == 1, "Called endpoint with empty candidates"
        assert ranked == [], "No candidates => empty rank result"


@pytest.mark.asyncio
@patch("builtins.open", new_callable=mock_open, read_data=b"Fake file data")
async def test_search_documents_async_integration(mock_file):
    """
    Tests search_documents_async integration by mocking both Docsifer (files => markdown)
    and rank calls. Distinguishes them by presence/absence of "candidates" in the payload.
    """
    doc_resp_1 = {"filename": "doc1.pdf", "markdown": "Doc1 content"}
    doc_resp_2 = {"filename": "doc2.pdf", "markdown": "Doc2 content"}
    rank_resp = {
        "probabilities": [[0.1, 0.9]],
        "cosine_similarities": [[0.2, 0.8]]
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

        json_payload = kwargs.get("json", {})
        if "candidates" in json_payload:
            # Rank call
            return MockContextManager(rank_resp)

        # Docsifer call
        if not hasattr(mock_post, "call_count"):
            mock_post.call_count = 0
        if mock_post.call_count == 0:
            mock_post.call_count += 1
            return MockContextManager(doc_resp_1)
        return MockContextManager(doc_resp_2)

    embs = Embs()
    with patch("aiohttp.ClientSession.post", side_effect=mock_post):
        files = ["/path/file1.pdf", "/path/file2.pdf"]
        results = await embs.search_documents_async(query="sample query", files=files)
        assert len(results) == 2, "Expected 2 ranked documents"
        # doc2 => highest probability => first
        assert results[0]["filename"] == "doc2.pdf"
        assert results[0]["probability"] == 0.9


def test_sync_wrapper_search_documents():
    """
    Confirms the synchronous search_documents method calls the async version internally.
    """
    embs = Embs()
    fake_return = [
        {"filename": "mock.pdf", "markdown": "demo", "probability": 1.0, "cosine_similarity": 0.95}
    ]

    async def mock_async(*args, **kwargs):
        return fake_return

    with patch.object(embs, "search_documents_async", side_effect=mock_async):
        outcome = embs.search_documents(query="hello world")
        assert outcome == fake_return, "Sync method should return the same data from the async method."


@pytest.mark.asyncio
async def test_disk_cache(tmp_path):
    """
    Verify disk caching by storing and retrieving a mocked embedding. 
    No real network calls or real file reads aside from writing .json to disk.
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

        # second call should come from disk => 1 network call total
        assert first_resp == second_resp
        assert mock_method.call_count == 1

        # confirm a .json file in tmp_path
        files_in_dir = os.listdir(tmp_path)
        assert any(fname.endswith(".json") for fname in files_in_dir), "Expected .json cache file on disk"


@pytest.mark.asyncio
async def test_disk_cache_expiry(tmp_path):
    """
    Test disk cache TTL by patching time.time. 
    An item older than TTL is removed upon retrieval => forces new request.
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

        # 1) First call => writes to disk
        data1 = await embs.embed_async("Disk expiry text")
        assert mock_method.call_count == 1

        # 2) Time jump => item stale
        time_mock.return_value = t0 + 301

        # 3) Re-fetch => new call
        data2 = await embs.embed_async("Disk expiry text")
        assert mock_method.call_count == 2, "Cache expired => new network call"
        assert data1 == data2


@pytest.mark.asyncio
@patch("builtins.open", new_callable=mock_open, read_data=b"Fake file data")
async def test_retrieve_documents_async_multiple_files_concurrent(mock_file):
    """
    Demonstrate concurrency by providing multiple files with concurrency=2. 
    Each call returns a unique 'filename' to confirm they're processed.
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
        assert len(docs) == 5, "Expected 5 docs"
        filenames = [d["filename"] for d in docs]
        assert len(set(filenames)) == 5, "Each doc has a unique filename"
