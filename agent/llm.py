"""Composer 2.5 through the Cursor SDK. Tools are off; the reply is JSON validated by Pydantic."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "composer-2.5"


def load_env() -> None:
    load_dotenv(ROOT / ".env")


class ModelCaller(Protocol):
    def complete(
        self,
        prompt: str,
        schema: type[BaseModel],
        images: list[bytes] | None = None,
    ) -> BaseModel: ...


def extract_json(text: str) -> str:
    """Pull the JSON object out of a reply that may be wrapped in prose or a fence."""
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.DOTALL)
    if fenced:
        return fenced.group(1)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return stripped


def _model_id(item) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return str(item.get("id") or item.get("name") or "")
    return str(getattr(item, "id", "") or getattr(item, "name", "") or "")


class CursorCaller:
    """One Composer call per node. `transport` replaces the SDK in tests."""

    def __init__(self, transport=None, api_key: str | None = None):
        load_env()
        self._transport = transport
        self.api_key = (api_key if api_key is not None else os.environ.get("CURSOR_API_KEY", "")).strip()
        self._model = None

    def ensure_model(self) -> None:
        """Fail before the first search if this key cannot use Composer 2.5."""
        if self._transport is not None:
            return
        if not self.api_key:
            raise SystemExit("Set CURSOR_API_KEY in .env (Cursor Dashboard → Integrations).")
        from cursor_sdk import Cursor, ModelParameterValue, ModelSelection

        listing = Cursor.models.list(api_key=self.api_key)
        items = getattr(listing, "items", listing)
        chosen = None
        ids: list[str] = []
        for item in items:
            mid = _model_id(item)
            if not mid:
                continue
            ids.append(mid)
            if mid == MODEL_ID:
                chosen = item
        if chosen is None:
            available = ", ".join(ids) or "(none)"
            raise SystemExit(
                f"This CURSOR_API_KEY cannot use {MODEL_ID}. Models it can use: {available}."
            )
        # Fast is the product default and is several times the price. Keep it off.
        self._model = ModelSelection(
            id=MODEL_ID,
            params=[ModelParameterValue(id="fast", value="false")],
        )

    def complete(
        self,
        prompt: str,
        schema: type[BaseModel],
        images: list[bytes] | None = None,
    ) -> BaseModel:
        schema_text = json.dumps(schema.model_json_schema())
        full = (
            f"{prompt}\n\nReply with one JSON object matching this schema and nothing else:\n{schema_text}"
        )
        last_text = ""
        last_error = ""
        for attempt in range(2):
            message = full if attempt == 0 else (
                f"{full}\n\nYour previous reply was not valid JSON for that schema: {last_error}\n"
                f"Previous reply:\n{last_text}\nReply with JSON only."
            )
            last_text = self._send(message, images)
            try:
                return schema.model_validate(json.loads(extract_json(last_text)))
            except (ValidationError, json.JSONDecodeError) as exc:
                last_error = str(exc)
        raise RuntimeError(
            f"Model did not return valid {schema.__name__}. Last reply:\n{last_text}"
        )

    def _send(self, prompt: str, images: list[bytes] | None) -> str:
        if self._transport is not None:
            return self._transport(prompt, images)
        return _cursor_prompt(prompt, images, api_key=self.api_key, model=self._model)


def _cursor_prompt(prompt: str, images: list[bytes] | None, *, api_key: str, model) -> str:
    import tempfile

    from cursor_sdk import Agent, AgentOptions, LocalAgentOptions, ModelSelection, SDKImage, UserMessage
    from langsmith import traceable

    @traceable(name="composer", run_type="llm")
    def _call(prompt: str, image_count: int) -> str:
        if model is None:
            from cursor_sdk import ModelParameterValue

            selection = ModelSelection(
                id=MODEL_ID,
                params=[ModelParameterValue(id="fast", value="false")],
            )
        else:
            selection = model
        message: str | UserMessage
        if images:
            message = UserMessage(
                text=prompt,
                images=[SDKImage.from_data(blob, "image/jpeg") for blob in images],
            )
        else:
            message = prompt
        # A private cwd plus an empty tool list: the model can only answer in text.
        work = tempfile.mkdtemp(prefix="framefinder-agent-")
        result = Agent.prompt(
            message,
            AgentOptions(
                model=selection,
                api_key=api_key,
                name="framefinder",
                tools=[],
                local=LocalAgentOptions(cwd=work, setting_sources=[]),
            ),
        )
        status = getattr(result, "status", None)
        text = getattr(result, "result", "") or ""
        if status not in (None, "finished"):
            detail = getattr(result, "error_message", None) or text or "no error message"
            code = getattr(result, "error_code", None)
            label = f"{status} ({code})" if code else status
            raise RuntimeError(f"Cursor run {label}: {detail}")
        return text

    return _call(prompt, len(images or []))
