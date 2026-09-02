"""Structured output schema for candidate short-form moments."""

from __future__ import annotations

from pydantic import BaseModel, Field, FiniteFloat, field_validator, model_validator


class ShortMoment(BaseModel):
    """One timestamped moment identified by the Shorts task."""

    start_seconds: FiniteFloat = Field(ge=0)
    end_seconds: FiniteFloat = Field(ge=0)
    title: str | None = None
    summary: str | None = None
    central_idea: str | None = None
    context: str | None = None

    @field_validator("title", "summary", "central_idea", "context")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @model_validator(mode="after")
    def validate_time_range(self) -> "ShortMoment":
        if self.end_seconds < self.start_seconds:
            raise ValueError("end_seconds must be greater than or equal to start_seconds")
        return self


class ShortsResult(BaseModel):
    """Structured result returned by the Shorts task."""

    moments: list[ShortMoment] = Field(default_factory=list)
