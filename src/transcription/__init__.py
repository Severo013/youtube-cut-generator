"""Speech-to-Text provider adapters."""

from .base import (
    AudioFileTooLargeError,
    EmptyTranscriptionError,
    EmptyTranscriptTextError,
    InvalidAudioFileError,
    Transcriber,
    TranscriptionConfigurationError,
    TranscriptionError,
    TranscriptionProviderError,
    UnsupportedAudioFormatError,
)
from .faster_whisper_transcriber import FasterWhisperTranscriber
from .openai_transcriber import OpenAITranscriber

__all__ = [
    "AudioFileTooLargeError",
    "EmptyTranscriptionError",
    "EmptyTranscriptTextError",
    "FasterWhisperTranscriber",
    "InvalidAudioFileError",
    "OpenAITranscriber",
    "Transcriber",
    "TranscriptionConfigurationError",
    "TranscriptionError",
    "TranscriptionProviderError",
    "UnsupportedAudioFormatError",
]
