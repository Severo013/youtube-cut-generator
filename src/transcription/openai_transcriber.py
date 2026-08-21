"""OpenAI Speech-to-Text adapter."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from openai import OpenAI, OpenAIError
from pydantic import SecretStr, ValidationError

from src.models import Transcript, TranscriptSegment
from src.transcription.base import (
    AudioFileTooLargeError,
    EmptyTranscriptionError,
    EmptyTranscriptTextError,
    InvalidAudioFileError,
    TranscriptionConfigurationError,
    TranscriptionProviderError,
    UnsupportedAudioFormatError,
)


SUPPORTED_AUDIO_EXTENSIONS = frozenset(
    {
        ".flac",
        ".m4a",
        ".mp3",
        ".mp4",
        ".mpeg",
        ".mpga",
        ".ogg",
        ".wav",
        ".webm",
    }
)
MAX_AUDIO_FILE_SIZE_MB = 25
MAX_AUDIO_FILE_SIZE_BYTES = MAX_AUDIO_FILE_SIZE_MB * 1024 * 1024


class OpenAITranscriber:
    """Transcribe audio with OpenAI and normalize the response immediately."""

    def __init__(
        self,
        api_key: SecretStr | str | None,
        model: str,
        language: str = "pt",
        output_dir: Path | None = None,
        client: OpenAI | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._language = language
        self._output_dir = output_dir
        self._client = client

    def transcribe(self, audio_path: Path) -> Transcript:
        """Call the OpenAI Audio API and return a provider-neutral transcript."""

        audio_path = Path(audio_path)
        self._validate_audio_file(audio_path)
        api_key = self._get_api_key()

        try:
            client = self._client or OpenAI(api_key=api_key)
            with audio_path.open("rb") as audio_file:
                response = client.audio.transcriptions.create(
                    file=audio_file,
                    model=self._model,
                    language=self._language,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                )
        except OSError as error:
            raise InvalidAudioFileError(
                f"Não foi possível ler o arquivo de áudio: {audio_path}"
            ) from error
        except OpenAIError as error:
            raise TranscriptionProviderError(
                f"A API de transcrição da OpenAI falhou: {error}"
            ) from error

        transcript = self.normalize_response(
            response=response,
            source=str(audio_path),
            fallback_language=self._language,
        )
        if self._output_dir is not None:
            self._save_artifacts(response, transcript, self._output_dir)
        return transcript

    @classmethod
    def normalize_response(
        cls,
        response: object,
        source: str,
        fallback_language: str,
    ) -> Transcript:
        """Convert an OpenAI verbose response into the internal model."""

        full_text = cls._get_field(response, "text", default=None)
        if not full_text or not str(full_text).strip():
            raise EmptyTranscriptTextError(
                "A resposta do Whisper não contém texto transcrito."
            )

        raw_segments = cls._get_field(response, "segments", default=None)
        if not raw_segments:
            raise EmptyTranscriptionError(
                "A resposta do Whisper não contém segmentos com timestamps."
            )

        try:
            segments = [
                TranscriptSegment(
                    start_seconds=cls._get_field(segment, "start"),
                    end_seconds=cls._get_field(segment, "end"),
                    text=cls._get_field(segment, "text"),
                )
                for segment in raw_segments
            ]
            response_duration = cls._get_field(response, "duration", default=None)
            duration = (
                float(response_duration)
                if response_duration is not None
                else max(segment.end_seconds for segment in segments)
            )
            language = cls._get_field(response, "language", default=None)
            return Transcript(
                source=source,
                duration_seconds=duration,
                language=str(language or fallback_language),
                full_text=str(full_text),
                segments=segments,
            )
        except (AttributeError, KeyError, TypeError, ValueError, ValidationError) as error:
            raise TranscriptionProviderError(
                "A OpenAI retornou uma transcrição em formato inválido."
            ) from error

    def _get_api_key(self) -> str:
        api_key = (
            self._api_key.get_secret_value()
            if isinstance(self._api_key, SecretStr)
            else self._api_key
        )
        if not api_key or not api_key.strip():
            raise TranscriptionConfigurationError(
                "OPENAI_API_KEY não foi definida. Configure-a no arquivo .env."
            )
        return api_key

    @staticmethod
    def _validate_audio_file(audio_path: Path) -> None:
        if not audio_path.exists():
            raise FileNotFoundError(f"Arquivo de áudio não encontrado: {audio_path}")
        if not audio_path.is_file():
            raise InvalidAudioFileError(f"O caminho não é um arquivo: {audio_path}")
        if audio_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
            raise UnsupportedAudioFormatError(
                f"Extensão de áudio não suportada: {audio_path.suffix or '(sem extensão)'}. "
                f"Formatos aceitos: {supported}."
            )
        try:
            file_size = audio_path.stat().st_size
            if file_size == 0:
                raise InvalidAudioFileError(f"O arquivo de áudio está vazio: {audio_path}")
            if file_size > MAX_AUDIO_FILE_SIZE_BYTES:
                raise AudioFileTooLargeError(
                    f"O arquivo de áudio excede o limite de {MAX_AUDIO_FILE_SIZE_MB} MB "
                    "do endpoint de transcrição. Será necessário implementar "
                    "preprocessing/chunking antes de processar episódios desse tamanho."
                )
        except OSError as error:
            raise InvalidAudioFileError(
                f"Não foi possível acessar o arquivo de áudio: {audio_path}"
            ) from error

    @classmethod
    def _save_artifacts(
        cls,
        response: object,
        transcript: Transcript,
        output_dir: Path,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_payload = cls._serialize_response(response)
        cls._write_json(output_dir / "transcript.raw.json", raw_payload)
        cls._write_json(
            output_dir / "transcript.json",
            transcript.model_dump(mode="json"),
        )
        readable_text = transcript.as_timestamped_text()
        (output_dir / "transcript.txt").write_text(
            f"{readable_text}\n" if readable_text else "",
            encoding="utf-8",
        )

    @staticmethod
    def _serialize_response(response: object) -> Any:
        model_dump = getattr(response, "model_dump", None)
        if callable(model_dump):
            return model_dump(mode="json")
        if isinstance(response, Mapping):
            return dict(response)
        raise TranscriptionProviderError(
            "A resposta da OpenAI não pôde ser serializada para debug."
        )

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _get_field(value: object, name: str, default: Any = ...) -> Any:
        if isinstance(value, Mapping):
            if default is ...:
                return value[name]
            return value.get(name, default)
        if default is ...:
            return getattr(value, name)
        return getattr(value, name, default)
