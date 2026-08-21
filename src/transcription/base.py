"""Contract for Speech-to-Text providers."""

from pathlib import Path
from typing import Protocol

from src.models import Transcript


class TranscriptionError(Exception):
    """Base error for failures in the transcription stage."""


class TranscriptionConfigurationError(TranscriptionError):
    """The transcriber is missing required configuration."""


class UnsupportedAudioFormatError(TranscriptionError):
    """The selected file extension is not accepted by the provider."""


class InvalidAudioFileError(TranscriptionError):
    """The selected audio file cannot be uploaded."""


class AudioFileTooLargeError(TranscriptionError):
    """The selected audio file exceeds the provider upload limit."""


class EmptyTranscriptTextError(TranscriptionError):
    """The provider response does not contain transcript text."""


class EmptyTranscriptionError(TranscriptionError):
    """The provider response does not contain timestamped segments."""


class TranscriptionProviderError(TranscriptionError):
    """The remote Speech-to-Text provider failed."""


class Transcriber(Protocol):
    """Convert an audio file into the application's normalized transcript."""

    def transcribe(self, audio_path: Path) -> Transcript:
        """Transcribe ``audio_path`` while preserving segment timestamps."""
        ...
