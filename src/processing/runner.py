"""Sequential, provider-neutral execution of transcript tasks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from pydantic import BaseModel

from src.models import Transcript
from src.processing.base import TranscriptProcessor, UnknownExecutorError
from src.processing.models import ProcessingResult, TaskDefinition


class ProcessingRunner:
    """Run independent transcript tasks sequentially in the supplied order."""

    def __init__(self, processors: Mapping[str, TranscriptProcessor]) -> None:
        self._processors = dict(processors)

    def run(
        self,
        transcript: Transcript,
        tasks: Iterable[TaskDefinition],
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
            output = processor.process(task, transcript, formatted_transcript)
            validated_output = task.result_type.model_validate(output)
            results.append(
                ProcessingResult[BaseModel](
                    task_id=task.id,
                    task_name=task.name,
                    executor=task.executor,
                    output=validated_output,
                )
            )

        return results
