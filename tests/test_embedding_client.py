"""Unit tests for DashScopeEmbeddingClient.

Validates that:
- The client includes ``parameters.dimension`` in every API request.
- embed_query returns a single vector of correct length.
- embed_documents returns a list of vectors of correct length.
- Batching works correctly for inputs larger than MAX_BATCH_SIZE.
- API errors (non-200) raise RuntimeError.
- Empty embeddings response raises RuntimeError.
- Singleton behavior via get_embed_model().
"""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.integrations.embedding_client import (
    DASHSCOPE_EMBEDDING_URL,
    DashScopeEmbeddingClient,
)


def _make_mock_embeddings(count: int, dim: int) -> list[list[float]]:
    return [[float(i % dim) for i in range(dim)] for _ in range(count)]


def _mock_response(status_code: int = 200, embeddings: list[list[float]] | None = None):
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = status_code
    if status_code == 200:
        mock.json.return_value = {
            "output": {
                "embeddings": [{"embedding": emb} for emb in (embeddings or [])]
            }
        }
    else:
        mock.json.return_value = {"message": "error"}
        mock.text = "error body"
    return mock


class TestDashScopeEmbeddingClientInit:
    def test_stores_constructor_params(self):
        client = DashScopeEmbeddingClient(
            model="text-embedding-v4",
            api_key="sk-test",
            dimensions=1536,
        )
        assert client.model == "text-embedding-v4"
        assert client.api_key == "sk-test"
        assert client.dimensions == 1536


class TestEmbedQuery:
    @patch("app.integrations.embedding_client.httpx.post")
    def test_returns_single_vector(self, mock_post):
        vectors = _make_mock_embeddings(1, 1536)
        mock_post.return_value = _mock_response(200, vectors)

        client = DashScopeEmbeddingClient(
            model="text-embedding-v4",
            api_key="sk-test",
            dimensions=1536,
        )
        result = client.embed_query("test query")

        assert isinstance(result, list)
        assert len(result) == 1536
        assert all(isinstance(v, float) for v in result)

    @patch("app.integrations.embedding_client.httpx.post")
    def test_result_length_matches_dimensions(self, mock_post):
        vectors = _make_mock_embeddings(1, 1024)
        mock_post.return_value = _mock_response(200, vectors)

        client = DashScopeEmbeddingClient(
            model="text-embedding-v4",
            api_key="sk-test",
            dimensions=1024,
        )
        result = client.embed_query("test")

        assert len(result) == 1024

    @patch("app.integrations.embedding_client.httpx.post")
    def test_call_includes_dimension_in_parameters(self, mock_post):
        vectors = _make_mock_embeddings(1, 1536)
        mock_post.return_value = _mock_response(200, vectors)

        client = DashScopeEmbeddingClient(
            model="text-embedding-v4",
            api_key="sk-test",
            dimensions=1536,
        )
        client.embed_query("dimension probe")

        call_args = mock_post.call_args
        request_body = call_args[1]["json"]
        assert request_body["parameters"] == {
            "dimension": 1536,
            "output_type": "dense",
        }
        assert request_body["model"] == "text-embedding-v4"
        assert request_body["input"]["texts"] == ["dimension probe"]


class TestEmbedDocuments:
    @patch("app.integrations.embedding_client.httpx.post")
    def test_returns_list_of_vectors(self, mock_post):
        vectors = _make_mock_embeddings(3, 1536)
        mock_post.return_value = _mock_response(200, vectors)

        client = DashScopeEmbeddingClient(
            model="text-embedding-v4",
            api_key="sk-test",
            dimensions=1536,
        )
        result = client.embed_documents(["text1", "text2", "text3"])

        assert len(result) == 3
        assert all(len(v) == 1536 for v in result)

    @patch("app.integrations.embedding_client.httpx.post")
    def test_batches_large_input(self, mock_post):
        mock_post.side_effect = [
            _mock_response(200, _make_mock_embeddings(10, 128))
            for _ in range(6)
        ]

        client = DashScopeEmbeddingClient(
            model="text-embedding-v4",
            api_key="sk-test",
            dimensions=128,
        )
        texts = [f"text_{i}" for i in range(60)]
        result = client.embed_documents(texts)

        assert len(result) == 60
        assert mock_post.call_count == 6

    @patch("app.integrations.embedding_client.httpx.post")
    def test_request_body_includes_input_texts(self, mock_post):
        vectors = _make_mock_embeddings(2, 128)
        mock_post.return_value = _mock_response(200, vectors)

        client = DashScopeEmbeddingClient(
            model="text-embedding-v4",
            api_key="sk-test",
            dimensions=128,
        )
        client.embed_documents(["hello", "world"])

        call_args = mock_post.call_args
        request_body = call_args[1]["json"]
        assert request_body["input"]["texts"] == ["hello", "world"]


class TestAPIErrors:
    @patch("app.integrations.embedding_client.httpx.post")
    def test_non_200_raises_runtime_error(self, mock_post):
        mock_post.return_value = _mock_response(400)

        client = DashScopeEmbeddingClient(
            model="text-embedding-v4",
            api_key="sk-test",
            dimensions=1536,
        )

        with pytest.raises(RuntimeError, match="DashScope embedding API returned 400"):
            client.embed_query("test")

    @patch("app.integrations.embedding_client.httpx.post")
    def test_empty_embeddings_raises_runtime_error(self, mock_post):
        mock_post.return_value = _mock_response(200, [])

        client = DashScopeEmbeddingClient(
            model="text-embedding-v4",
            api_key="sk-test",
            dimensions=1536,
        )

        with pytest.raises(RuntimeError, match="returned no embeddings"):
            client.embed_query("test")

    @patch("app.integrations.embedding_client.httpx.post")
    def test_dimension_mismatch_raises_runtime_error(self, mock_post):
        vectors = _make_mock_embeddings(1, 1024)
        mock_post.return_value = _mock_response(200, vectors)

        client = DashScopeEmbeddingClient(
            model="text-embedding-v4",
            api_key="sk-test",
            dimensions=1536,
        )

        with pytest.raises(RuntimeError, match="dimension mismatch"):
            client.embed_query("test")


class TestHeadersAndAuth:
    @patch("app.integrations.embedding_client.httpx.post")
    def test_auth_header_includes_api_key(self, mock_post):
        vectors = _make_mock_embeddings(1, 128)
        mock_post.return_value = _mock_response(200, vectors)

        client = DashScopeEmbeddingClient(
            model="text-embedding-v4",
            api_key="sk-my-api-key",
            dimensions=128,
        )
        client.embed_query("test")

        call_args = mock_post.call_args
        headers = call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer sk-my-api-key"
        assert headers["Content-Type"] == "application/json"

    @patch("app.integrations.embedding_client.httpx.post")
    def test_url_is_correct(self, mock_post):
        vectors = _make_mock_embeddings(1, 128)
        mock_post.return_value = _mock_response(200, vectors)

        client = DashScopeEmbeddingClient(
            model="text-embedding-v4",
            api_key="sk-test",
            dimensions=128,
        )
        client.embed_query("test")

        call_args = mock_post.call_args
        assert call_args[0][0] == DASHSCOPE_EMBEDDING_URL
