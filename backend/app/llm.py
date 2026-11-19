"""OpenAI-compatible structured judge and embedding client."""

from __future__ import annotations

import re
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from app.config import get_settings

TModel = TypeVar("TModel", bound=BaseModel)

_FENCED_JSON = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class LLMNotConfiguredError(RuntimeError):
    """Raised when a live evaluation is requested without OPENAI_API_KEY."""


class LLMCallError(RuntimeError):
    """Raised when the judge or embedding request fails or returns invalid JSON."""


class LLMClient:
    """Thin wrapper around an OpenAI-compatible chat and embeddings API."""

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.llm_configured:
            raise LLMNotConfiguredError(
                "OPENAI_API_KEY is not set. Call the API with mock_mode=true or add a key to backend/.env."
            )
        self.model = settings.openai_model
        self.embedding_model = settings.openai_embedding_model
        self._client = OpenAI(
            api_key=settings.openai_api_key.strip(),
            base_url=settings.openai_base_url,
            timeout=60.0,
        )

    def complete_text(self, *, system: str, user: str) -> str:
        """Request a plain-text completion for ad-hoc playground generation."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            return self._complete(messages, json_mode=False).strip()
        except LLMCallError:
            raise
        except Exception as exc:
            raise LLMCallError(f"LLM request failed: {exc}") from exc

    def complete_structured(self, *, system: str, user: str, schema: type[TModel]) -> TModel:
        """Request a JSON object and validate it against a Pydantic model."""
        content = self._chat(system, user)
        try:
            return schema.model_validate_json(_extract_json(content))
        except ValidationError as first_error:
            repair = (
                f"{user}\n\nYour previous reply failed validation:\n{content}\n{first_error}\n"
                "Return only a corrected JSON object."
            )
            repaired = self._chat(system, repair)
            try:
                return schema.model_validate_json(_extract_json(repaired))
            except ValidationError as second_error:
                raise LLMCallError(f"Judge returned invalid JSON: {second_error}") from second_error

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts with the configured embedding model, preserving order."""
        if not texts:
            return []
        try:
            response = self._client.embeddings.create(model=self.embedding_model, input=texts)
        except Exception as exc:
            raise LLMCallError(f"Embedding request failed: {exc}") from exc
        ordered = sorted(response.data, key=lambda item: item.index)
        return [list(item.embedding) for item in ordered]

    def _chat(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            return self._complete(messages, json_mode=True)
        except LLMCallError:
            raise
        except Exception as exc:
            if not _json_mode_rejected(exc):
                raise LLMCallError(f"LLM request failed: {exc}") from exc
        try:
            return self._complete(messages, json_mode=False)
        except LLMCallError:
            raise
        except Exception as exc:
            raise LLMCallError(f"LLM request failed: {exc}") from exc

    def _complete(self, messages: list[dict[str, str]], *, json_mode: bool) -> str:
        kwargs: dict[str, object] = {
            "model": self.model,
            "temperature": 0,
            "messages": messages,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        completion = self._client.chat.completions.create(**kwargs)  # type: ignore[arg-type]
        content = completion.choices[0].message.content
        if not content or not content.strip():
            raise LLMCallError("LLM returned an empty response.")
        return content


def _extract_json(text: str) -> str:
    stripped = _FENCED_JSON.sub("", text.strip())
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return stripped


def _json_mode_rejected(exc: Exception) -> bool:
    text = str(exc).lower()
    return "response_format" in text or "json_object" in text
