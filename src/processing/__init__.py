"""Generic transcript processing contracts and orchestration."""

from .base import (
    EmptyProcessingResultError,
    InvalidProcessingResultError,
    ProcessingConfigurationError,
    ProcessingError,
    ProcessingProviderError,
    PromptExecutor,
    TaskExecutionError,
    TranscriptProcessor,
    UnknownExecutorError,
)
from .models import ProcessingContext, ProcessingResult, TaskDefinition
from .prompting import build_prompt_variables
from .output import ProcessingOutputWriter
from .registry import (
    DuplicateTaskIdError,
    InvalidTaskDefinitionError,
    TaskNotFoundError,
    TaskRegistry,
    TaskRegistryError,
)
from .runner import ProcessingRunner
from .validation import InvalidExecutionPlanError, validate_execution_plan

__all__ = [
    "DuplicateTaskIdError",
    "EmptyProcessingResultError",
    "InvalidProcessingResultError",
    "InvalidTaskDefinitionError",
    "ProcessingConfigurationError",
    "ProcessingContext",
    "ProcessingError",
    "ProcessingProviderError",
    "ProcessingResult",
    "ProcessingRunner",
    "ProcessingOutputWriter",
    "PromptExecutor",
    "TaskDefinition",
    "TaskExecutionError",
    "TaskNotFoundError",
    "TaskRegistry",
    "TaskRegistryError",
    "TranscriptProcessor",
    "UnknownExecutorError",
    "InvalidExecutionPlanError",
    "build_prompt_variables",
    "validate_execution_plan",
]
