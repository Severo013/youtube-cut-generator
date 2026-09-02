"""Generic OpenAI Structured Outputs processor for transcript tasks."""

from __future__ import annotations

from openai import OpenAI, OpenAIError
from pydantic import BaseModel, SecretStr, ValidationError

from src.models import Transcript
from src.processing.base import (
    EmptyProcessingResultError,
    InvalidProcessingResultError,
    ProcessingConfigurationError,
    ProcessingProviderError,
)
from src.processing.models import TaskDefinition
from src.prompting import render_prompt


class OpenAITranscriptProcessor:
    """Execute arbitrary configured tasks through OpenAI Responses API."""

    def __init__(
        self,
        api_key: SecretStr | str | None,
        default_model: str,
        client: OpenAI | None = None,
    ) -> None:
        self._api_key = api_key
        self._default_model = default_model
        self._client = client

    def process(
        self,
        task: TaskDefinition,
        transcript: Transcript,
        formatted_transcript: str,
    ) -> BaseModel:
        del transcript
        template = task.load_prompt()
        prompt = render_prompt(template, formatted_transcript)
        return self.execute_prompt(task, prompt)

    def execute_prompt(self, task: TaskDefinition, prompt: str) -> BaseModel:
        """Execute an already rendered prompt for compatibility adapters."""

        api_key = self._get_api_key()
        model = task.model or self._default_model
        try:
            client = self._client or OpenAI(api_key=api_key)
            response = client.responses.parse(
                model=model,
                input=prompt,
                text_format=task.result_type,
            )
        except OpenAIError as error:
            raise ProcessingProviderError(
                f"A API da OpenAI falhou ao executar a task {task.id!r}: {error}"
            ) from error
        except ValidationError as error:
            raise InvalidProcessingResultError(
                f"A OpenAI retornou dados inválidos para a task {task.id!r}."
            ) from error

        parsed = response.output_parsed
        if parsed is None:
            raise EmptyProcessingResultError(
                f"A OpenAI não retornou resultado estruturado para a task {task.id!r}."
            )

        try:
            return task.result_type.model_validate(parsed)
        except ValidationError as error:
            raise InvalidProcessingResultError(
                f"A OpenAI retornou dados inválidos para a task {task.id!r}."
            ) from error

    def model_for(self, task: TaskDefinition) -> str:
        """Return the model that will execute this task."""

        return task.model or self._default_model

    def _get_api_key(self) -> str:
        api_key = (
            self._api_key.get_secret_value()
            if isinstance(self._api_key, SecretStr)
            else self._api_key
        )
        if not api_key or not api_key.strip():
            raise ProcessingConfigurationError(
                "OPENAI_API_KEY não foi definida. Configure-a no arquivo .env."
            )
        return api_key
