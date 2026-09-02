"""Sequential, provider-neutral execution of transcript tasks."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

from pydantic import BaseModel

from src.models import Transcript
from src.processing.base import (
    TaskExecutionError,
    TranscriptProcessor,
    UnknownExecutorError,
)
from src.processing.models import ProcessingResult, TaskDefinition


class ProcessingRunner:
    """Run independent transcript tasks sequentially in the supplied order."""

    def __init__(self, processors: Mapping[str, TranscriptProcessor]) -> None:
        self._processors = dict(processors)

    def run(
        self,
        transcript: Transcript,
        tasks: Iterable[TaskDefinition],
        *,
        on_result: Callable[
            [TaskDefinition, ProcessingResult[BaseModel]], None
        ]
        | None = None,
    ) -> list[ProcessingResult[BaseModel]]:
        formatted_transcript = transcript.as_timestamped_text()
        results: list[ProcessingResult[BaseModel]] = []

        for task in tasks:
            processor = self._processors.get(task.executor)
            if processor is None:
                raise UnknownExecutorError(
                    f"Executor não registrado para a task {task.id!r}: "
                    f"{task.executor!r}."
                )
            try:
                output = processor.process(task, transcript, formatted_transcript)
                validated_output = task.result_type.model_validate(output)
                model_resolver = getattr(processor, "model_for", None)
                effective_model = (
                    model_resolver(task) if callable(model_resolver) else task.model
                )
                result = ProcessingResult[BaseModel](
                    task_id=task.id,
                    task_name=task.name,
                    executor=task.executor,
                    model=effective_model,
                    schema_id=task.schema_id,
                    output=validated_output,
                )
                if on_result is not None:
                    on_result(task, result)
            except Exception as error:
                if isinstance(error, TaskExecutionError):
                    raise
                raise TaskExecutionError(task.id, error) from error
            results.append(result)

        return results
