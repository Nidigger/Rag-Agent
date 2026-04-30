"""Custom DashScope embedding client with explicit dimension control.

Replaces langchain_community.embeddings.DashScopeEmbeddings to ensure
that text-embedding-v4 embeddings are generated with the correct output
dimension matching the Qdrant collection configuration.

Key differences from DashScopeEmbeddings:
- Explicitly passes ``dimension`` in the ``parameters`` body.
- Dimensions are read from settings (not hardcoded), enabling
  configuration-driven dimension changes (1024/1536/2048).
- Compatible with LangChain's ``Embeddings`` protocol (embed_query
  and embed_documents methods), so QdrantVectorStore and ChromaVectorStore
  can consume it without changes.
"""

import logging
from typing import List

import httpx

from app.agent.tools.request_context import get_perf_request_id
from app.utils.perf import elapsed_ms, log_perf, now_ms

logger = logging.getLogger("rag-agent.embedding_client")

DASHSCOPE_EMBEDDING_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/"
    "embeddings/text-embedding/text-embedding"
)
MAX_BATCH_SIZE = 10


class DashScopeEmbeddingClient:
    """Embedding client that explicitly passes dimensions to DashScope API.

        text-embedding-v4 defaults to 1024 unless ``dimension`` is passed
    in the request ``parameters``. This client always passes the
    configured dimension so the output matches the vector store schema.
    """

    def __init__(self, model: str, api_key: str, dimensions: int):
        self.model = model
        self.api_key = api_key
        self.dimensions = dimensions

        logger.info(
            "embedding_model=%s embedding_dimensions=%d",
            self.model,
            self.dimensions,
        )

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        vectors = self.embed_documents([text])
        return vectors[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document texts with automatic batching."""
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), MAX_BATCH_SIZE):
            batch = texts[i : i + MAX_BATCH_SIZE]
            all_embeddings.extend(self._call_api(batch))
        return all_embeddings

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """Call the DashScope text-embedding API.

        Request includes ``parameters.dimension`` to force the model
        to output the configured vector dimension.
        """
        request_id = get_perf_request_id()

        request_body = {
            "model": self.model,
            "input": {"texts": texts},
            "parameters": {
                "dimension": self.dimensions,
                "output_type": "dense",
            },
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        log_perf("embedding", "start",
                 request_id=request_id,
                 status="start",
                 model=self.model,
                 dimension=self.dimensions,
                 batch_size=len(texts))

        call_start = now_ms()

        response = httpx.post(
            DASHSCOPE_EMBEDDING_URL,
            json=request_body,
            headers=headers,
            timeout=30,
        )

        call_elapsed = elapsed_ms(call_start)

        if response.status_code != 200:
            log_perf("embedding", "error",
                     request_id=request_id,
                     status="failed",
                     elapsed_ms=call_elapsed,
                     error_code=f"HTTP_{response.status_code}",
                     error=f"DashScope API returned {response.status_code}: {response.text[:100]}")
            raise RuntimeError(
                f"DashScope embedding API returned {response.status_code}: "
                f"{response.text}"
            )

        data = response.json()
        embeddings = data.get("output", {}).get("embeddings", [])

        if not embeddings:
            log_perf("embedding", "error",
                     request_id=request_id,
                     status="failed",
                     elapsed_ms=call_elapsed,
                     error_code="NO_EMBEDDINGS",
                     error="DashScope API returned empty embeddings")
            raise RuntimeError(
                f"DashScope embedding API returned no embeddings: {data}"
            )

        vectors = [emb["embedding"] for emb in embeddings]
        for vector in vectors:
            if len(vector) != self.dimensions:
                raise RuntimeError(
                    "DashScope embedding dimension mismatch: "
                    f"expected={self.dimensions}, got={len(vector)}"
                )

        log_perf("embedding", "done",
                 request_id=request_id,
                 status="success",
                 elapsed_ms=call_elapsed,
                 model=self.model,
                 dimension=self.dimensions,
                 batch_size=len(texts))

        return vectors
