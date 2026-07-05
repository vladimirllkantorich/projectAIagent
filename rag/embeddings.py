from __future__ import annotations

import hashlib
import math
import re
from typing import Optional


FALLBACK_DIMENSIONS = 384


class EmbeddingModel:
    def __init__(
        self,
        provider: str,
        openai_api_key: Optional[str],
        openai_embedding_model: str,
        local_base_url: str,
        local_embedding_model: str,
    ):
        self.provider = provider
        self.openai_api_key = openai_api_key
        self.openai_embedding_model = openai_embedding_model
        self.local_base_url = local_base_url
        self.local_embedding_model = local_embedding_model
        self._fallback_reason: Optional[Exception] = None

    @property
    def collection_name(self) -> str:
        if self.provider == "openai":
            key = f"openai:{self.openai_embedding_model}"
        else:
            key = f"local:{self.local_base_url}:{self.local_embedding_model}"

        digest = hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()[:12]
        return f"devvault_ai_chunks_{digest}"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        try:
            embeddings = self._embed_with_openai_compatible_api(texts)
        except Exception as exc:
            if self.provider == "openai" and not self.openai_api_key:
                raise RuntimeError("OpenAI embedding requires `[openai].api_key` in Streamlit secrets.") from exc
            self._fallback_reason = exc
            return [_hash_embedding(text) for text in texts]

        self._fallback_reason = None
        return embeddings

    def _embed_with_openai_compatible_api(self, texts: list[str]) -> list[list[float]]:
        client, model = self._client_and_model()
        response = client.embeddings.create(model=model, input=texts)
        return [item.embedding for item in response.data]

    def _client_and_model(self):
        from openai import OpenAI

        if self.provider == "openai":
            if not self.openai_api_key:
                raise RuntimeError("OpenAI embedding requires `[openai].api_key` in Streamlit secrets.")
            return OpenAI(api_key=self.openai_api_key), self.openai_embedding_model

        return OpenAI(base_url=self.local_base_url, api_key="lm-studio"), self.local_embedding_model

    @property
    def warning_message(self) -> Optional[str]:
        if self._fallback_reason is None:
            return None

        return (
            "Embedding API is unavailable, so DevVault AI used lightweight local fallback embeddings. "
            "For better RAG search, load an embedding model in LM Studio or configure an OpenAI embedding model. "
            f"Original error: {self._fallback_reason}"
        )


def _hash_embedding(text: str) -> list[float]:
    vector = [0.0] * FALLBACK_DIMENSIONS
    tokens = _tokenize(text)

    if not tokens:
        tokens = [text[:128] or "empty"]

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8", errors="ignore")).digest()
        bucket = int.from_bytes(digest[:4], "big") % FALLBACK_DIMENSIONS
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector

    return [value / norm for value in vector]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())
