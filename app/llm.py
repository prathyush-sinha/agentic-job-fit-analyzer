"""Gemini structured-output wrapper for the agent and judge.

Uses the Gemini API's native JSON mode with a Pydantic response_schema, so every
call returns a validated model instance — never free-form text. Token usage is
returned for cost/observability logging.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel

from app.config import Settings, get_settings

T = TypeVar("T", bound=BaseModel)

MAX_RETRIES = 4
_RATE_LIMIT_WAIT_S = 30


@dataclass
class LLMResult:
    parsed: BaseModel
    prompt_tokens: int
    output_tokens: int
    model: str


def generate_structured(
    prompt: str,
    schema: type[T],
    settings: Settings | None = None,
    model: str | None = None,
    temperature: float = 0.2,
) -> tuple[T, LLMResult]:
    """Call Gemini and parse the response into `schema`. Returns (instance, usage)."""
    from google import genai
    from google.genai import types
    from google.genai.errors import ClientError, ServerError

    settings = settings or get_settings()
    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set (free key: aistudio.google.com).")
    client = genai.Client(api_key=settings.google_api_key)
    model = model or settings.gemini_agent_model
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
        temperature=temperature,
    )

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.models.generate_content(model=model, contents=prompt, config=config)
            break
        except ClientError as exc:
            if getattr(exc, "code", None) == 429 and attempt < MAX_RETRIES - 1:
                time.sleep(_RATE_LIMIT_WAIT_S)
                continue
            raise
        except ServerError:  # 5xx (e.g. 503 overloaded) — transient, back off
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt + 1)
                continue
            raise

    instance = resp.parsed
    if instance is None:  # fall back to manual parse of the JSON text
        instance = schema.model_validate_json(resp.text)

    usage = getattr(resp, "usage_metadata", None)
    result = LLMResult(
        parsed=instance,
        prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
        output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
        model=model,
    )
    return instance, result
