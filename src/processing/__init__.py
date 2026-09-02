"""Generic transcript processing contracts and orchestration."""

from .base import (
    EmptyProcessingResultError,
    InvalidProcessingResultError,
    ProcessingConfigurationError,
    ProcessingError,
    ProcessingProviderError,
    TaskExecutionError,
    TranscriptProcessor,
    UnknownExecutorError,
)
from .models import ProcessingResult, TaskDefinition
from .output import ProcessingOutputWriter
from .registry import (
    DuplicateTaskIdError,
    InvalidTaskDefinitionError,
    TaskNotFoundError,
    TaskRegistry,
    TaskRegistryError,
)
from .runner import ProcessingRunner

__all__ = [
    "DuplicateTaskIdError",
    "EmptyProcessingResultError",
    "InvalidProcessingResultError",
    "InvalidTaskDefinitionError",
    "ProcessingConfigurationError",
    "ProcessingError",
    "ProcessingProviderError",
    "ProcessingResult",
    "ProcessingRunner",
    "ProcessingOutputWriter",
    "TaskDefinition",
    "TaskExecutionError",
    "TaskNotFoundError",
    "TaskRegistry",
    "TaskRegistryError",
    "TranscriptProcessor",
    "UnknownExecutorError",
]
