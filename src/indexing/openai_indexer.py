"""Compatibility adapter for the original OpenAI indexing API."""

from __future__ import annotations

from openai import OpenAI
from pydantic import SecretStr

from src.indexing.base import (
    EmptyIndexResultError,
    IndexingConfigurationError,
    IndexingProviderError,
)
from src.models import IndexResult, Transcript
from src.processing import (
    EmptyProcessingResultError,
    InvalidProcessingResultError,
    ProcessingConfigurationError,
    ProcessingProviderError,
    TaskRegistry,
)
from src.processing.openai_processor import OpenAITranscriptProcessor


class OpenAIIndexer:
    """Preserve the legacy Indexer API through the generic OpenAI processor."""

    def __init__(
        self,
        api_key: SecretStr | str | None,
        model: str,
        client: OpenAI | None = None,
        processor: OpenAITranscriptProcessor | None = None,
    ) -> None:
        self._processor = processor or OpenAITranscriptProcessor(
            api_key=api_key,
            default_model=model,
            client=client,
        )
        self._task = TaskRegistry.load_builtin().get("indexacao")

    def index(self, transcript: Transcript, prompt: str) -> IndexResult:
        """Send a rendered prompt and parse it directly into ``IndexResult``."""

        try:
            result = self._processor.execute_prompt(self._task, prompt)
        except ProcessingConfigurationError as error:
            raise IndexingConfigurationError(str(error)) from error
        except EmptyProcessingResultError as error:
            raise EmptyIndexResultError(
                "A OpenAI não retornou uma indexação estruturada."
            ) from error
        except InvalidProcessingResultError as error:
            raise IndexingProviderError(
                "A OpenAI retornou uma indexação em formato inválido."
            ) from error
        except ProcessingProviderError as error:
            provider_error = error.__cause__ or error
            raise IndexingProviderError(
                f"A API de indexação da OpenAI falhou: {provider_error}"
            ) from error
        return IndexResult.model_validate(result)
