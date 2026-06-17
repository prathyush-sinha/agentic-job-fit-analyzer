"""Embeddings with a pluggable provider (Gemini default, OpenAI fallback).

Both providers are batched with retries and token/cost accounting. Gemini uses
asymmetric task types (RETRIEVAL_DOCUMENT for chunks, RETRIEVAL_QUERY for the
resume query), which improves retrieval quality.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from app.config import Settings, get_settings

BATCH_SIZE = 100   # inputs per request (both providers accept >= this)
MAX_RETRIES = 5

# USD per 1M tokens for cost estimate. Gemini free tier = $0.
_PRICE_PER_1M = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-004": 0.0,
    "gemini-embedding-001": 0.0,
}


@dataclass
class EmbedResult:
    embeddings: list[list[float]]
    total_tokens: int  # exact for OpenAI; ~chars/4 estimate for Gemini

    def estimated_cost(self, model: str) -> float:
        return self.total_tokens / 1_000_000 * _PRICE_PER_1M.get(model, 0.0)


def _batched(texts: list[str]):
    for start in range(0, len(texts), BATCH_SIZE):
        yield texts[start : start + BATCH_SIZE]


def _approx_tokens(texts: list[str]) -> int:
    return sum(len(t) for t in texts) // 4


# --- OpenAI ---------------------------------------------------------------

def _embed_openai(texts: list[str], settings: Settings, is_query: bool) -> EmbedResult:
    from openai import (
        APIConnectionError,
        APITimeoutError,
        AuthenticationError,
        OpenAI,
        PermissionDeniedError,
        RateLimitError,
    )

    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set (see .env.example).")
    client = OpenAI(api_key=settings.openai_api_key)
    model = settings.openai_embedding_model
    embeddings: list[list[float]] = []
    total_tokens = 0

    for batch in _batched(texts):
        for attempt in range(MAX_RETRIES):
            try:
                resp = client.embeddings.create(model=model, input=batch)
                break
            except (AuthenticationError, PermissionDeniedError):
                raise
            except RateLimitError as exc:
                if getattr(exc, "code", None) == "insufficient_quota":
                    raise  # out of credits, not transient
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(2 ** attempt)
            except (APIConnectionError, APITimeoutError):
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(2 ** attempt)
        ordered = sorted(resp.data, key=lambda d: d.index)
        embeddings.extend(d.embedding for d in ordered)
        total_tokens += resp.usage.total_tokens
    return EmbedResult(embeddings=embeddings, total_tokens=total_tokens)


# --- Gemini ---------------------------------------------------------------

# Free tier counts each text as one request against a ~100 req/min quota.
# We send small batches and pace to stay safely under it.
GEMINI_BATCH = 20
GEMINI_TARGET_RPM = 80
_RATE_LIMIT_WAIT_S = 61  # free-tier 429 windows are ~60s


def _embed_gemini(texts: list[str], settings: Settings, is_query: bool) -> EmbedResult:
    from google import genai
    from google.genai import types
    from google.genai.errors import ClientError

    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set (free key: aistudio.google.com).")
    client = genai.Client(api_key=settings.google_api_key)
    model = settings.gemini_embedding_model
    task = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
    cfg = types.EmbedContentConfig(
        task_type=task, output_dimensionality=settings.gemini_output_dim
    )
    per_request_pause = 60.0 / GEMINI_TARGET_RPM  # seconds of quota per text
    embeddings: list[list[float]] = []

    for start in range(0, len(texts), GEMINI_BATCH):
        batch = texts[start : start + GEMINI_BATCH]
        for attempt in range(MAX_RETRIES):
            try:
                resp = client.models.embed_content(model=model, contents=batch, config=cfg)
                break
            except ClientError as exc:
                # 429 = rate limit (retryable); 400/403 = bad key (fail fast).
                if getattr(exc, "code", None) == 429 and attempt < MAX_RETRIES - 1:
                    time.sleep(_RATE_LIMIT_WAIT_S)
                    continue
                raise
        embeddings.extend(e.values for e in resp.embeddings)
        if not is_query:
            time.sleep(per_request_pause * len(batch))  # proactively pace
    return EmbedResult(embeddings=embeddings, total_tokens=_approx_tokens(texts))


# --- public API -----------------------------------------------------------

def embed_texts(
    texts: list[str], settings: Settings | None = None, is_query: bool = False
) -> EmbedResult:
    """Embed texts with the configured provider. Embeddings preserve input order."""
    settings = settings or get_settings()
    if settings.embedding_provider == "openai":
        return _embed_openai(texts, settings, is_query)
    return _embed_gemini(texts, settings, is_query)


def embed_query(text: str, settings: Settings | None = None) -> list[float]:
    """Embed a single query string (uses the query task type for Gemini)."""
    return embed_texts([text], settings, is_query=True).embeddings[0]
