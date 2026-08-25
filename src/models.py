"""Provider-independent data models used by the application."""

from __future__ import annotations

import math

from pydantic import BaseModel, Field, FiniteFloat, field_validator, model_validator


def seconds_to_timestamp(seconds: float) -> str:
    """Convert non-negative seconds to an ``HH:MM:SS`` timestamp.

    Fractional seconds are intentionally truncated because the internal display
    format has one-second precision. The original float values remain available
    on transcript segments.
    """

    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise TypeError("seconds must be a number")
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("seconds must be a finite, non-negative number")

    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class TranscriptSegment(BaseModel):
    """A timestamped piece of a transcript."""

    start_seconds: FiniteFloat = Field(ge=0)
    end_seconds: FiniteFloat = Field(ge=0)
    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def strip_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text cannot be blank")
        return value

    @model_validator(mode="after")
    def validate_time_range(self) -> "TranscriptSegment":
        if self.end_seconds < self.start_seconds:
            raise ValueError("end_seconds must be greater than or equal to start_seconds")
        return self

    @property
    def start_timestamp(self) -> str:
        return seconds_to_timestamp(self.start_seconds)

    @property
    def end_timestamp(self) -> str:
        return seconds_to_timestamp(self.end_seconds)


class Transcript(BaseModel):
    """Normalized transcript, independent of a Speech-to-Text provider."""

    source: str = Field(min_length=1)
    duration_seconds: FiniteFloat = Field(ge=0)
    language: str = Field(min_length=1)
    full_text: str
    segments: list[TranscriptSegment] = Field(default_factory=list)

    @field_validator("source", "language")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value cannot be blank")
        return value

    def as_timestamped_text(self) -> str:
        """Render segments in a provider-neutral, prompt-friendly format."""

        return "\n".join(
            f"[{segment.start_timestamp} - {segment.end_timestamp}] {segment.text}"
            for segment in self.segments
        )


class IndexBlock(BaseModel):
    """A chronological topic range in an episode."""

    start_seconds: FiniteFloat = Field(ge=0)
    end_seconds: FiniteFloat = Field(ge=0)
    start_timestamp: str = Field(min_length=8)
    end_timestamp: str = Field(min_length=8)
    topic: str = Field(min_length=1)
    summary: str = Field(min_length=1)

    @field_validator("start_timestamp", "end_timestamp", "topic", "summary")
    @classmethod
    def strip_index_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value cannot be blank")
        return value

    @model_validator(mode="after")
    def validate_time_range(self) -> "IndexBlock":
        if self.end_seconds < self.start_seconds:
            raise ValueError("end_seconds must be greater than or equal to start_seconds")
        if self.start_timestamp != seconds_to_timestamp(self.start_seconds):
            raise ValueError("start_timestamp does not match start_seconds")
        if self.end_timestamp != seconds_to_timestamp(self.end_seconds):
            raise ValueError("end_timestamp does not match end_seconds")
        return self


class IndexResult(BaseModel):
    """Structured result returned by an indexing provider."""

    blocks: list[IndexBlock] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_chronological_order(self) -> "IndexResult":
        starts = [block.start_seconds for block in self.blocks]
        if starts != sorted(starts):
            raise ValueError("index blocks must be in chronological order")
        return self
