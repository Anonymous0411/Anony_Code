"""Embedding API used by MedChain retrieval components.

This module is intentionally independent from ``util.utils`` so importing the
embedding helper does not probe the local chat-model server.
"""

from __future__ import annotations

import os
from typing import Sequence

from openai import OpenAI


DEFAULT_EMBEDDING_BASE_URL = "https://yunwu.ai/v1"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"


def llm_embedding(text: str) -> Sequence[float]:
    """Return one embedding using the same endpoint as the original helper."""
    api_key = os.getenv(
        "EMBEDDING_API_KEY",
        "sk-4kQ5thuq26NGgwXmyk48fV45al7LoMccZNmk99YD6oD76XRP",
    )
    base_url = os.getenv("EMBEDDING_BASE_URL", DEFAULT_EMBEDDING_BASE_URL)
    model = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.embeddings.create(input=str(text), model=model)
    return response.data[0].embedding
