"""Future OpenAI language-model adapter."""

from pydantic import SecretStr

from src.models import IndexResult, Transcript


class OpenAIIndexer:
    """OpenAI adapter boundary; API integration is intentionally deferred."""

    def __init__(self, api_key: SecretStr | None, model: str) -> None:
        self._api_key = api_key
        self._model = model

    def index(self, transcript: Transcript, prompt: str) -> IndexResult:
        raise NotImplementedError(
            "OpenAI indexing is not implemented in this structural MVP."
        )

