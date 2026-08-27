"""Application configuration loaded from environment variables and ``.env``."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr


DEFAULT_TRANSCRIPTION_MODEL = "whisper-1"
DEFAULT_INDEX_MODEL = "gpt-5-mini"
DEFAULT_TRANSCRIPTION_PROVIDER = "local"
DEFAULT_LOCAL_WHISPER_MODEL = "turbo"
DEFAULT_LOCAL_WHISPER_DEVICE = "cuda"
DEFAULT_LOCAL_WHISPER_COMPUTE_TYPE = "float16"


class Settings(BaseModel):
    """Runtime settings with safe, non-secret defaults."""

    openai_api_key: SecretStr | None = None
    transcription_provider: str = Field(
        default=DEFAULT_TRANSCRIPTION_PROVIDER, min_length=1
    )
    transcription_model: str = Field(default=DEFAULT_TRANSCRIPTION_MODEL, min_length=1)
    local_whisper_model: str = Field(default=DEFAULT_LOCAL_WHISPER_MODEL, min_length=1)
    local_whisper_device: str = Field(default=DEFAULT_LOCAL_WHISPER_DEVICE, min_length=1)
    local_whisper_compute_type: str = Field(
        default=DEFAULT_LOCAL_WHISPER_COMPUTE_TYPE, min_length=1
    )
    index_model: str = Field(default=DEFAULT_INDEX_MODEL, min_length=1)
    transcription_language: str = Field(default="pt", min_length=1)
    ytdlp_keep_temp: bool = False
    ytdlp_auth_mode: Literal["none", "browser", "file", "auto"] = "auto"
    ytdlp_browser: str = Field(default="firefox", min_length=1)
    ytdlp_cookies_file: Path | None = None


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
        transcription_provider=get(
            "TRANSCRIPTION_PROVIDER", DEFAULT_TRANSCRIPTION_PROVIDER
        ),
        transcription_model=get("TRANSCRIPTION_MODEL", DEFAULT_TRANSCRIPTION_MODEL),
        local_whisper_model=get(
            "LOCAL_WHISPER_MODEL", DEFAULT_LOCAL_WHISPER_MODEL
        ),
        local_whisper_device=get(
            "LOCAL_WHISPER_DEVICE", DEFAULT_LOCAL_WHISPER_DEVICE
        ),
        local_whisper_compute_type=get(
            "LOCAL_WHISPER_COMPUTE_TYPE", DEFAULT_LOCAL_WHISPER_COMPUTE_TYPE
        ),
        index_model=get("INDEX_MODEL", DEFAULT_INDEX_MODEL),
        transcription_language=get("TRANSCRIPTION_LANGUAGE", "pt"),
        ytdlp_keep_temp=get("YTDLP_KEEP_TEMP", "false"),
        ytdlp_auth_mode=(get("YTDLP_AUTH_MODE", "auto") or "auto").strip().lower(),
        ytdlp_browser=(
            get("YTDLP_BROWSER", "firefox") or "firefox"
        ).strip().lower(),
        ytdlp_cookies_file=get("YTDLP_COOKIES_FILE") or None,
    )
