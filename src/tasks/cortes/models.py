"""Structured output schema for long-form YouTube cut selections."""

from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)


class CutSelection(BaseModel):
    """One cut aligned to boundaries from the episode index."""

    start_seconds: int = Field(ge=0, strict=True)
    end_seconds: int = Field(ge=0, strict=True)
    theme: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    title: str = Field(min_length=1)
    thumbnail_text: str = Field(min_length=1)

    @field_validator("theme", "summary", "title", "thumbnail_text")
    @classmethod
    def strip_editorial_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value cannot be blank")
        return value

    @model_validator(mode="after")
    def validate_time_range(self) -> "CutSelection":
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        return self

    @property
    def duration_seconds(self) -> int:
        return self.end_seconds - self.start_seconds


class CutsResult(BaseModel):
    """Structured result returned by the cut selection task."""

    model_config = ConfigDict(revalidate_instances="always")

    cuts: list[CutSelection]
    limitations: list[str]

    @field_validator("limitations")
    @classmethod
    def strip_limitations(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            value = value.strip()
            if not value:
                raise ValueError("limitations cannot contain blank values")
            normalized.append(value)
        return normalized

    @model_validator(mode="after")
    def validate_current_run_constraints(self, info: ValidationInfo) -> "CutsResult":
        if info.context is None:
            return self

        from src.tasks.cortes.validation import validate_cuts_result

        validate_cuts_result(self, info.context)
        return self
