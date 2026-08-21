"""Contract for Speech-to-Text providers."""

from pathlib import Path
from typing import Protocol

from src.models import Transcript


class Transcriber(Protocol):
    """Convert an audio file into the application's normalized transcript."""

    def transcribe(self, audio_path: Path) -> Transcript:
        """Transcribe ``audio_path`` while preserving segment timestamps."""
        ...

