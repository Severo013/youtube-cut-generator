"""Central construction of configured Speech-to-Text providers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from src.config import Settings
from src.transcription.base import Transcriber, TranscriptionConfigurationError
from src.transcription.faster_whisper_transcriber import FasterWhisperTranscriber
from src.transcription.openai_transcriber import OpenAITranscriber


ProgressCallback = Callable[[str], None]


def create_transcriber(
    settings: Settings,
    output_dir: Path | None = None,
    progress: ProgressCallback | None = None,
) -> Transcriber:
    """Build the configured transcriber in one provider-selection point."""

    provider = settings.transcription_provider.strip().lower()
    if provider == "local":
        return FasterWhisperTranscriber(
            model=settings.local_whisper_model,
            device=settings.local_whisper_device,
            compute_type=settings.local_whisper_compute_type,
            language=settings.transcription_language,
            output_dir=output_dir,
            progress=progress,
        )
    if provider == "openai":
        return OpenAITranscriber(
            api_key=settings.openai_api_key,
            model=settings.transcription_model,
            language=settings.transcription_language,
            output_dir=output_dir,
        )
    raise TranscriptionConfigurationError(
        "TRANSCRIPTION_PROVIDER desconhecido: "
        f"{settings.transcription_provider!r}. Use 'local' ou 'openai'."
    )
