"""Application configuration loaded from environment variables and ``.env``."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field, SecretStr


DEFAULT_TRANSCRIPTION_MODEL = "whisper-1"


class Settings(BaseModel):
    """Runtime settings with safe, non-secret defaults."""

    openai_api_key: SecretStr | None = None
    transcription_model: str = Field(default=DEFAULT_TRANSCRIPTION_MODEL, min_length=1)
    index_model: str = Field(default="gpt-5-mini", min_length=1)
    transcription_language: str = Field(default="pt", min_length=1)


def _read_dotenv(path: Path) -> dict[str, str]:
    """Read the small KEY=VALUE subset needed by this project."""

    if not path.is_file():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value
    return values


def load_settings(dotenv_path: str | Path = ".env") -> Settings:
    """Load settings, giving real environment variables precedence over ``.env``."""

    dotenv = _read_dotenv(Path(dotenv_path))

    def get(name: str, default: str | None = None) -> str | None:
        return os.environ.get(name, dotenv.get(name, default))

    return Settings(
        openai_api_key=get("OPENAI_API_KEY") or None,
        transcription_model=get("TRANSCRIPTION_MODEL", DEFAULT_TRANSCRIPTION_MODEL),
        index_model=get("INDEX_MODEL", "gpt-5-mini"),
        transcription_language=get("TRANSCRIPTION_LANGUAGE", "pt"),
    )
