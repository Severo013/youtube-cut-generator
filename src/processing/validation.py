"""Validation for ordered task execution plans."""

from collections.abc import Iterable

from src.processing.base import ProcessingConfigurationError
from src.processing.models import TaskDefinition


class InvalidExecutionPlanError(ProcessingConfigurationError):
    """Raised when selected tasks do not satisfy their declared dependencies."""


def validate_execution_plan(tasks: Iterable[TaskDefinition]) -> None:
    """Validate dependencies without changing the requested task order."""

    completed_task_ids: set[str] = set()
    for task in tasks:
        for dependency_id in task.requires:
            if dependency_id not in completed_task_ids:
                raise InvalidExecutionPlanError(
                    f"Task {task.id!r} requires task {dependency_id!r}, which must "
                    "be selected and executed earlier in the same processing run."
                )
        completed_task_ids.add(task.id)
