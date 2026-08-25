"""OpenAI Responses API adapter for episode indexing."""

from __future__ import annotations

from openai import OpenAI, OpenAIError
from pydantic import SecretStr, ValidationError

from src.indexing.base import (
    EmptyIndexResultError,
    IndexingConfigurationError,
    IndexingProviderError,
)
from src.models import IndexResult, Transcript


class OpenAIIndexer:
    """Generate a provider-neutral index through OpenAI Structured Outputs."""

    def __init__(
        self,
        api_key: SecretStr | str | None,
        model: str,
        client: OpenAI | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._client = client

    def index(self, transcript: Transcript, prompt: str) -> IndexResult:
        """Send a rendered prompt and parse it directly into ``IndexResult``."""

        api_key = self._get_api_key()
        try:
            client = self._client or OpenAI(api_key=api_key)
            response = client.responses.parse(
                model=self._model,
                input=prompt,
                text_format=IndexResult,
            )
        except OpenAIError as error:
            raise IndexingProviderError(
                f"A API de indexação da OpenAI falhou: {error}"
            ) from error
        except ValidationError as error:
            raise IndexingProviderError(
                "A OpenAI retornou uma indexação em formato inválido."
            ) from error

        parsed = response.output_parsed
        if parsed is None:
            raise EmptyIndexResultError(
                "A OpenAI não retornou uma indexação estruturada."
            )

        try:
            return IndexResult.model_validate(parsed)
        except ValidationError as error:
            raise IndexingProviderError(
                "A OpenAI retornou uma indexação em formato inválido."
            ) from error

    def _get_api_key(self) -> str:
        api_key = (
            self._api_key.get_secret_value()
            if isinstance(self._api_key, SecretStr)
            else self._api_key
        )
        if not api_key or not api_key.strip():
            raise IndexingConfigurationError(
                "OPENAI_API_KEY não foi definida. Configure-a no arquivo .env."
            )
        return api_key
