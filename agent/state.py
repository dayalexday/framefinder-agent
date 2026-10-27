"""Shared notepad passed between LangGraph nodes, plus the JSON shapes Composer must return."""
from __future__ import annotations

from typing import TypedDict

from pydantic import BaseModel, Field, field_validator


class SearchPhrase(BaseModel):
    """A short visual phrase for CLIP. Only what a still frame can show."""

    phrase: str = Field(min_length=1)

    @field_validator("phrase")
    @classmethod
    def strip_phrase(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("empty phrase")
        return value


class Verdict(BaseModel):
    """Whether the captions satisfy the original request."""

    match: bool
    reason: str = Field(min_length=1)

    @field_validator("reason")
    @classmethod
    def strip_reason(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("empty reason")
        return value


class Hit(TypedDict):
    video: str
    t: float
    score: float
    caption: str


class Attempt(TypedDict):
    phrase: str
    captions: list[str]
    match: bool
    reason: str


class AgentState(TypedDict):
    request: str
    search_phrase: str
    hits: list[Hit]
    verdict: bool
    reason: str
    retries_left: int
    attempts: list[Attempt]
    answer: str


def frame_notes_model(n: int) -> type[BaseModel]:
    """A model that accepts exactly n captions, one per image sent."""

    class FrameNotes(BaseModel):
        captions: list[str] = Field(min_length=n, max_length=n)

        @field_validator("captions")
        @classmethod
        def strip_captions(cls, value: list[str]) -> list[str]:
            cleaned = [caption.strip() for caption in value]
            if any(not caption for caption in cleaned):
                raise ValueError("empty caption")
            return cleaned

    return FrameNotes
