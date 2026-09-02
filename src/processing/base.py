"""Provider-neutral contracts and errors for transcript processing."""

from typing import Protocol

from pydantic import BaseModel

from src.models import Transcript
from src.processing.models import TaskDefinition


class ProcessingError(Exception):
    """Base error for transcript processing failures."""


class ProcessingConfigurationError(ProcessingError):
    """Raised when a processor cannot start due to invalid configuration."""


class ProcessingProviderError(ProcessingError):
    """Raised when a processing provider fails."""


class EmptyProcessingResultError(ProcessingProviderError):
    """Raised when a provider returns no parsed structured output."""


class InvalidProcessingResultError(ProcessingProviderError):
    """Raised when a provider returns output that violates the task schema."""


class UnknownExecutorError(ProcessingConfigurationError):
    """Raised when no processor was registered for a task executor."""


class TaskExecutionError(ProcessingError):
    """Identify the task that failed while preserving the original exception."""

    def __init__(self, task_id: str, error: Exception) -> None:
        self.task_id = task_id
        self.original_error = error
        super().__init__(f"Task {task_id!r} failed: {error}")


class TranscriptProcessor(Protocol):
    """Execute one task over a normalized transcript."""

    def process(
        self,
        task: TaskDefinition,
        transcript: Transcript,
        formatted_transcript: str,
    ) -> BaseModel:
        """Execute ``task`` using the canonical transcript representation."""
        ...

    def model_for(self, task: TaskDefinition) -> str | None:
        """Return the effective model identifier, when the executor uses one."""
        ...
