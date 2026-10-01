"""Operational validation for cut selections."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.models import IndexResult
from src.processing.models import ProcessingContext, TaskDefinition
from src.tasks.cortes.models import CutsResult


class CutsSettings(BaseModel):
    """Validated operational settings declared by the cut task."""

    model_config = ConfigDict(extra="forbid")

    target_count: int = Field(gt=0, strict=True)
    min_duration_minutes: int = Field(gt=0, strict=True)
    max_duration_minutes: int = Field(gt=0, strict=True)

    @model_validator(mode="after")
    def validate_duration_range(self) -> "CutsSettings":
        if self.max_duration_minutes <= self.min_duration_minutes:
            raise ValueError(
                "max_duration_minutes must be greater than min_duration_minutes"
            )
        return self


def _get_validation_inputs(
    validation_context: Mapping[str, Any],
) -> tuple[IndexResult, CutsSettings]:
    task = validation_context.get("task_definition")
    processing_context = validation_context.get("processing_context")
    if not isinstance(task, TaskDefinition) or not isinstance(
        processing_context,
        ProcessingContext,
    ):
        raise ValueError("cuts validation requires task and processing context")

    settings = CutsSettings.model_validate(task.settings)
    index_results: list[IndexResult] = []
    for dependency_id in task.requires:
        dependency = processing_context.results.get(dependency_id)
        if dependency is not None and dependency.schema_id == "index_result":
            index_results.append(IndexResult.model_validate(dependency.output))
    if len(index_results) != 1:
        raise ValueError("cuts validation requires exactly one IndexResult dependency")

    return index_results[0], settings


def validate_cuts_result(
    result: CutsResult,
    validation_context: Mapping[str, Any],
) -> None:
    """Validate cuts against task settings and the current in-memory index."""

    index_result, settings = _get_validation_inputs(validation_context)
    if len(result.cuts) > settings.target_count:
        raise ValueError(
            f"cuts cannot contain more than target_count={settings.target_count} items"
        )
    if len(result.cuts) < settings.target_count and not result.limitations:
        raise ValueError(
            "limitations must explain why fewer than target_count cuts were returned"
        )

    minimum_seconds = settings.min_duration_minutes * 60
    maximum_seconds = settings.max_duration_minutes * 60
    for position, cut in enumerate(result.cuts, start=1):
        if not minimum_seconds <= cut.duration_seconds <= maximum_seconds:
            raise ValueError(
                f"cut {position} duration must be between {minimum_seconds} and "
                f"{maximum_seconds} seconds"
            )

        start_indices = [
            index
            for index, block in enumerate(index_result.blocks)
            if block.start_seconds == cut.start_seconds
        ]
        if not start_indices:
            raise ValueError(
                f"cut {position} start_seconds must match an IndexResult block start"
            )

        end_indices = [
            index
            for index, block in enumerate(index_result.blocks)
            if block.end_seconds == cut.end_seconds
        ]
        if not end_indices:
            raise ValueError(
                f"cut {position} end_seconds must match an IndexResult block end"
            )

        if not any(start <= end for start in start_indices for end in end_indices):
            raise ValueError(
                f"cut {position} start block must not occur after its end block"
            )
