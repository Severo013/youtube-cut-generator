"""Future OpenAI Speech-to-Text adapter."""

from pathlib import Path

from pydantic import SecretStr

from src.models import Transcript


class OpenAITranscriber:
    """OpenAI adapter boundary; API integration is intentionally deferred."""

    def __init__(self, api_key: SecretStr | None, model: str, language: str) -> None:
        self._api_key = api_key
        self._model = model
        self._language = language

    def transcribe(self, audio_path: Path) -> Transcript:
        raise NotImplementedError(
            "OpenAI transcription is not implemented in this structural MVP."
        )

