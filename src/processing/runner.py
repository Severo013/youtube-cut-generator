"""Sequential, provider-neutral execution of transcript tasks."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

from pydantic import BaseModel

from src.models import Transcript
from src.processing.base import (
    PromptExecutor,
    ProcessingConfigurationError,
    TaskExecutionError,
    TranscriptProcessor,
    UnknownExecutorError,
)
from src.processing.models import ProcessingContext, ProcessingResult, TaskDefinition
from src.processing.prompting import build_prompt_variables
from src.processing.rendering import MarkdownRenderer, build_markdown_renderers
from src.processing.validation import validate_execution_plan
from src.prompting import render_prompt


class ProcessingRunner:
    """Run transcript tasks sequentially in the supplied order."""

    def __init__(
        self,
        processors: Mapping[str, TranscriptProcessor] | None = None,
        *,
        executors: Mapping[str, PromptExecutor] | None = None,
        renderers: Mapping[str, MarkdownRenderer] | None = None,
    ) -> None:
        """Register legacy processors and modern rendered-prompt executors.

        The positional ``processors`` mapping preserves the original independent-task
        contract. Dependency-aware tasks must use an entry in ``executors``.
        """

        self._legacy_processors = dict(processors or {})
        self._executors = dict(executors or {})
        duplicated = self._legacy_processors.keys() & self._executors.keys()
        if duplicated:
            names = ", ".join(sorted(duplicated))
            raise ValueError(f"Executors registered twice: {names}")
        self._renderers = build_markdown_renderers(renderers)

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
        task_list = list(tasks)
        validate_execution_plan(task_list)
        formatted_transcript = transcript.as_timestamped_text()
        context = ProcessingContext(transcript)
        results: list[ProcessingResult[BaseModel]] = []

        for task in task_list:
            executor = self._executors.get(task.executor)
            legacy_processor = self._legacy_processors.get(task.executor)
            processor = executor if executor is not None else legacy_processor
            if processor is None:
                raise UnknownExecutorError(
                    f"Executor não registrado para a task {task.id!r}: "
                    f"{task.executor!r}."
                )
            try:
                if executor is not None:
                    variables = build_prompt_variables(
                        task,
                        context,
                        formatted_transcript,
                        renderers=self._renderers,
                    )
                    prompt = render_prompt(task.load_prompt(), variables)
                    output = executor.execute_prompt(task, prompt)
                else:
                    if task.requires:
                        raise ProcessingConfigurationError(
                            f"Task {task.id!r} declares dependencies and must use "
                            "a rendered-prompt executor registered with ProcessingRunner."
                        )
                    output = legacy_processor.process(
                        task,
                        transcript,
                        formatted_transcript,
                    )
                validated_output = task.result_type.model_validate(
                    output,
                    context={
                        "task_definition": task,
                        "processing_context": context,
                    },
                )
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
            context.record(result)
            results.append(result)

        return results
