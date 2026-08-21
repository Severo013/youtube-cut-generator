"""Prompt loading and rendering."""

from pathlib import Path

from src.models import Transcript


TRANSCRIPT_PLACEHOLDER = "{{TRANSCRIPT}}"


def load_prompt(path: Path) -> str:
    """Load a UTF-8 prompt template from disk."""

    return path.read_text(encoding="utf-8")


def render_decupagem_prompt(template: str, transcript: Transcript) -> str:
    """Insert timestamped transcript segments into the decupagem template."""

    if TRANSCRIPT_PLACEHOLDER not in template:
        raise ValueError(f"prompt template must contain {TRANSCRIPT_PLACEHOLDER}")
    return template.replace(TRANSCRIPT_PLACEHOLDER, transcript.as_timestamped_text())

