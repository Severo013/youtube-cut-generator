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

__all__ = [
    "AudioFileTooLargeError",
    "EmptyTranscriptionError",
    "EmptyTranscriptTextError",
    "InvalidAudioFileError",
    "Transcriber",
    "TranscriptionConfigurationError",
    "TranscriptionError",
    "TranscriptionProviderError",
    "UnsupportedAudioFormatError",
]
