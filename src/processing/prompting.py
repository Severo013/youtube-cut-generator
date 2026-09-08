"""Generic preparation of prompt context for processing executors."""

from __future__ import annotations

from collections.abc import Mapping

from src.processing.models import ProcessingContext, TaskDefinition
from src.processing.rendering import MarkdownRenderer, render_processing_result
from src.processing.validation import InvalidExecutionPlanError


def _setting_as_text(value: str | bool | int | float) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def build_prompt_variables(
    task: TaskDefinition,
    context: ProcessingContext,
    formatted_transcript: str,
    *,
    renderers: Mapping[str, MarkdownRenderer] | None = None,
) -> dict[str, str]:
    """Build provider-neutral variables from the current in-memory run."""

    variables = {"TRANSCRIPT": formatted_transcript}
    for dependency_id in task.requires:
        result = context.results.get(dependency_id)
        if result is None:
            raise InvalidExecutionPlanError(
                f"Task {task.id!r} requires task {dependency_id!r}, which must "
                "complete successfully earlier in the same processing run."
            )
        variables[dependency_id.upper()] = render_processing_result(
            result,
            renderers=renderers,
        )

    for key, value in task.settings.items():
        variables[key.upper()] = _setting_as_text(value)
    return variables
