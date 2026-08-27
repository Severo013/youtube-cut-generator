"""Resolve local paths and HTTP(S) URLs to local audio files."""

from __future__ import annotations

import importlib.metadata
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol
from urllib.parse import urlsplit

from src.config import Settings


ProgressCallback = Callable[[str], None]


def _noop_cleanup() -> None:
    pass


class InputResolutionError(Exception):
    """Base error raised while resolving an input to a local file."""


class InvalidInputUrlError(InputResolutionError):
    """The input looks like an HTTP(S) URL but is malformed."""


class YtDlpDownloadError(InputResolutionError):
    """yt-dlp could not extract or download the requested media."""


class UnsupportedUrlError(YtDlpDownloadError):
    """No yt-dlp extractor supports the URL."""


class RestrictedMediaError(YtDlpDownloadError):
    """The media is restricted in the current environment."""


class UnavailableMediaError(YtDlpDownloadError):
    """The requested media is unavailable."""


class CookieAcquisitionError(YtDlpDownloadError):
    """Configured cookies could not be obtained."""


class NoAudioStreamError(YtDlpDownloadError):
    """The extractor returned no usable audio stream."""


class IncompleteDownloadError(YtDlpDownloadError):
    """The download did not produce a complete non-empty file."""


class AutoAuthenticationError(YtDlpDownloadError):
    """Both anonymous and authenticated download attempts failed."""


@dataclass(frozen=True)
class DownloadMetadata:
    """Diagnostic metadata returned by yt-dlp without changing domain models."""

    video_id: str | None
    title: str | None
    webpage_url: str | None
    duration_seconds: float | None
    uploader: str | None
    extension: str
    format_id: str | None
    audio_codec: str | None
    video_codec: str | None
    authentication_method: str
    file_size_bytes: int
    download_seconds: float
    temporary_directory: Path


@dataclass(frozen=True)
class DownloadedAudio:
    path: Path
    metadata: DownloadMetadata


@dataclass(frozen=True)
class ResolvedInput:
    path: Path
    kind: str
    metadata: DownloadMetadata | None = None
    cleanup: Callable[[], None] = _noop_cleanup


@dataclass(frozen=True)
class YtDlpEnvironment:
    yt_dlp_version: str | None
    ejs_version: str | None
    deno_path: Path | None
    deno_version: str | None
    deno_supported: bool


class _YoutubeDL(Protocol):
    def __enter__(self) -> _YoutubeDL: ...

    def __exit__(self, *args: object) -> None: ...

    def extract_info(self, url: str, *, download: bool) -> Mapping[str, Any]: ...

    def prepare_filename(self, info: Mapping[str, Any]) -> str: ...


YdlFactory = Callable[[dict[str, Any]], _YoutubeDL]


def is_http_url(value: str | Path) -> bool:
    """Return whether *value* is a valid absolute HTTP(S) URL."""

    text = str(value).strip()
    if not text or any(character.isspace() for character in text):
        return False
    try:
        parsed = urlsplit(text)
        # Accessing port also validates malformed values such as ``:abc``.
        _ = parsed.port
        return parsed.scheme.lower() in {"http", "https"} and bool(parsed.hostname)
    except ValueError:
        return False


def inspect_ytdlp_environment() -> YtDlpEnvironment:
    """Inspect the installed yt-dlp/EJS packages and recommended Deno runtime."""

    def distribution_version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    deno_executable = shutil.which("deno")
    deno_version: str | None = None
    deno_supported = False
    if deno_executable:
        try:
            completed = subprocess.run(
                [deno_executable, "--version"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            first_line = completed.stdout.splitlines()[0]
            match = re.search(r"\b(\d+)\.(\d+)(?:\.\d+)?\b", first_line)
            if match:
                version = tuple(int(part) for part in match.groups())
                deno_version = match.group(0)
                deno_supported = version >= (2, 3)
        except (OSError, subprocess.SubprocessError, IndexError):
            pass

    return YtDlpEnvironment(
        yt_dlp_version=distribution_version("yt-dlp"),
        ejs_version=distribution_version("yt-dlp-ejs"),
        deno_path=Path(deno_executable) if deno_executable else None,
        deno_version=deno_version,
        deno_supported=deno_supported,
    )


class _QuietLogger:
    """Discard normal yt-dlp output while retaining messages in exceptions."""

    def debug(self, _message: str) -> None:
        pass

    def info(self, _message: str) -> None:
        pass

    def warning(self, _message: str) -> None:
        pass

    def error(self, _message: str) -> None:
        pass


class _AttemptFailure(Exception):
    def __init__(
        self,
        authentication_method: str,
        detail: str,
        category: str = "download",
    ) -> None:
        super().__init__(detail)
        self.authentication_method = authentication_method
        self.detail = detail
        self.category = category

    def __str__(self) -> str:
        return self.detail


class YtDlpDownloader:
    """Download one best-audio stream through yt-dlp's Python API."""

    def __init__(
        self,
        settings: Settings,
        progress: ProgressCallback | None = None,
        ydl_factory: YdlFactory | None = None,
    ) -> None:
        self._settings = settings
        self._progress = progress or (lambda _message: None)
        self._ydl_factory = ydl_factory or self._create_youtube_dl

    def download(self, url: str, temporary_directory: Path) -> DownloadedAudio:
        """Download *url*, with at most one authenticated fallback in auto mode."""

        self._warn_if_youtube_runtime_incomplete(url)
        self._progress("Baixando áudio com yt-dlp...")
        auth_mode = self._settings.ytdlp_auth_mode

        if auth_mode == "auto":
            try:
                result = self._download_once(
                    url=url,
                    attempt_directory=temporary_directory / "anonymous",
                    authentication_method="none",
                )
            except _AttemptFailure as first_failure:
                fallback = self._auto_fallback_method()
                if fallback is None:
                    raise self._to_public_error(first_failure) from first_failure
                self._progress(
                    "Download sem cookies falhou; tentando autenticação "
                    f"via {fallback}..."
                )
                try:
                    result = self._download_once(
                        url=url,
                        attempt_directory=temporary_directory / "authenticated",
                        authentication_method=fallback,
                    )
                except _AttemptFailure as second_failure:
                    raise AutoAuthenticationError(
                        "O download falhou sem autenticação e também na única "
                        "tentativa autenticada. "
                        f"Sem autenticação: {first_failure.detail} "
                        f"Com autenticação ({fallback}): {second_failure.detail}"
                    ) from second_failure
        else:
            try:
                result = self._download_once(
                    url=url,
                    attempt_directory=temporary_directory / auth_mode,
                    authentication_method=auth_mode,
                )
            except _AttemptFailure as failure:
                raise self._to_public_error(failure) from failure

        if result.metadata.title:
            self._progress(f"Título: {result.metadata.title}")
        if result.metadata.duration_seconds is not None:
            self._progress(
                f"Duração: {self._format_duration(result.metadata.duration_seconds)}"
            )
        self._progress("Download concluído.")
        return result

    def _download_once(
        self,
        url: str,
        attempt_directory: Path,
        authentication_method: str,
    ) -> DownloadedAudio:
        attempt_directory.mkdir(parents=True, exist_ok=False)
        finished_paths: list[Path] = []

        def record_progress(status: Mapping[str, Any]) -> None:
            if status.get("status") == "finished" and status.get("filename"):
                finished_paths.append(Path(str(status["filename"])))

        try:
            auth_options = self._authentication_options(authentication_method)
        except CookieAcquisitionError as error:
            raise _AttemptFailure(
                authentication_method=authentication_method,
                detail=str(error),
                category="cookies",
            ) from error

        options: dict[str, Any] = {
            "format": "bestaudio/best",
            "outtmpl": str(attempt_directory / "%(id)s.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "logger": _QuietLogger(),
            "progress_hooks": [record_progress],
            "writedescription": False,
            "writeinfojson": False,
            "writethumbnail": False,
            "writesubtitles": False,
            "writeautomaticsub": False,
            "getcomments": False,
            "allow_playlist_files": False,
            **auth_options,
        }

        started_at = perf_counter()
        try:
            with self._ydl_factory(options) as ydl:
                info = ydl.extract_info(url, download=True)
                elapsed = perf_counter() - started_at
                audio_path = self._find_downloaded_file(
                    info=info,
                    ydl=ydl,
                    finished_paths=finished_paths,
                    attempt_directory=attempt_directory,
                )
        except _AttemptFailure:
            raise
        except Exception as error:
            category = self._error_category(error)
            detail = self._clean_error_message(error)
            raise _AttemptFailure(
                authentication_method=authentication_method,
                detail=detail,
                category=category,
            ) from error

        selected_format = self._selected_format(info)
        audio_codec = self._optional_text(selected_format.get("acodec"))
        if audio_codec == "none":
            raise _AttemptFailure(
                authentication_method=authentication_method,
                detail="O yt-dlp não encontrou um stream com áudio.",
                category="no_audio",
            )

        try:
            file_size = audio_path.stat().st_size
        except OSError as error:
            raise _AttemptFailure(
                authentication_method=authentication_method,
                detail=f"Não foi possível acessar o download concluído: {audio_path}",
                category="incomplete",
            ) from error
        if file_size <= 0:
            raise _AttemptFailure(
                authentication_method=authentication_method,
                detail=f"O yt-dlp produziu um arquivo vazio: {audio_path}",
                category="incomplete",
            )

        duration = info.get("duration")
        try:
            duration_seconds = float(duration) if duration is not None else None
        except (TypeError, ValueError):
            duration_seconds = None

        metadata = DownloadMetadata(
            video_id=self._optional_text(info.get("id")),
            title=self._optional_text(info.get("title")),
            webpage_url=self._optional_text(info.get("webpage_url")) or url,
            duration_seconds=duration_seconds,
            uploader=(
                self._optional_text(info.get("uploader"))
                or self._optional_text(info.get("channel"))
            ),
            extension=audio_path.suffix.lstrip(".")
            or self._optional_text(selected_format.get("ext"))
            or "",
            format_id=self._optional_text(selected_format.get("format_id")),
            audio_codec=audio_codec,
            video_codec=self._optional_text(selected_format.get("vcodec")),
            authentication_method=authentication_method,
            file_size_bytes=file_size,
            download_seconds=elapsed,
            temporary_directory=attempt_directory,
        )
        return DownloadedAudio(path=audio_path, metadata=metadata)

    def _authentication_options(self, method: str) -> dict[str, Any]:
        if method == "none":
            return {}
        if method == "browser":
            return {"cookiesfrombrowser": (self._settings.ytdlp_browser,)}
        if method == "file":
            cookie_file = self._settings.ytdlp_cookies_file
            if cookie_file is None:
                raise CookieAcquisitionError(
                    "YTDLP_AUTH_MODE=file requer YTDLP_COOKIES_FILE."
                )
            cookie_file = Path(cookie_file).expanduser()
            if not cookie_file.is_file():
                raise CookieAcquisitionError(
                    f"Arquivo de cookies não encontrado: {cookie_file}"
                )
            return {"cookiefile": str(cookie_file)}
        raise ValueError(f"Modo de autenticação do yt-dlp inválido: {method!r}")

    def _auto_fallback_method(self) -> str | None:
        if self._settings.ytdlp_cookies_file is not None:
            return "file"
        if self._settings.ytdlp_browser.strip():
            return "browser"
        return None

    def _warn_if_youtube_runtime_incomplete(self, url: str) -> None:
        hostname = (urlsplit(url).hostname or "").lower()
        if hostname != "youtu.be" and not hostname.endswith(".youtube.com"):
            return
        environment = inspect_ytdlp_environment()
        if environment.ejs_version is None:
            self._progress(
                "Aviso: yt-dlp-ejs não foi encontrado; o suporte ao YouTube "
                "pode ser limitado. Instale yt-dlp[default]."
            )
        if not environment.deno_supported:
            self._progress(
                "Aviso: Deno 2.3+ não foi encontrado no PATH; o suporte ao "
                "YouTube pode ser limitado."
            )

    @classmethod
    def _find_downloaded_file(
        cls,
        info: Mapping[str, Any],
        ydl: _YoutubeDL,
        finished_paths: list[Path],
        attempt_directory: Path,
    ) -> Path:
        candidates = list(finished_paths)
        requested_downloads = info.get("requested_downloads")
        if isinstance(requested_downloads, list):
            for download in requested_downloads:
                if isinstance(download, Mapping) and download.get("filepath"):
                    candidates.append(Path(str(download["filepath"])))
        for key in ("filepath", "_filename"):
            if info.get(key):
                candidates.append(Path(str(info[key])))
        try:
            candidates.append(Path(ydl.prepare_filename(info)))
        except Exception:
            pass

        root = attempt_directory.resolve()
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.is_relative_to(root) and resolved.is_file():
                return resolved

        completed_files = [
            path.resolve()
            for path in attempt_directory.rglob("*")
            if path.is_file()
            and not path.name.endswith((".part", ".ytdl", ".temp"))
        ]
        if len(completed_files) == 1:
            return completed_files[0]

        partials = list(attempt_directory.rglob("*.part"))
        suffix = f" Arquivos parciais: {len(partials)}." if partials else ""
        raise _AttemptFailure(
            authentication_method="unknown",
            detail="O yt-dlp não produziu um arquivo de áudio completo." + suffix,
            category="incomplete",
        )

    @staticmethod
    def _selected_format(info: Mapping[str, Any]) -> Mapping[str, Any]:
        requested_downloads = info.get("requested_downloads")
        if (
            isinstance(requested_downloads, list)
            and requested_downloads
            and isinstance(requested_downloads[0], Mapping)
        ):
            return requested_downloads[0]
        return info

    @staticmethod
    def _clean_error_message(error: Exception) -> str:
        message = " ".join(str(error).split())
        if message.startswith("ERROR: "):
            message = message[7:]
        return message or error.__class__.__name__

    @staticmethod
    def _error_category(error: Exception) -> str:
        cause_name = error.__class__.__name__
        exc_info = getattr(error, "exc_info", None)
        if isinstance(exc_info, tuple) and exc_info and isinstance(exc_info[0], type):
            cause_name = exc_info[0].__name__
        return {
            "CookieLoadError": "cookies",
            "UnsupportedError": "unsupported",
            "GeoRestrictedError": "restricted",
            "UnavailableVideoError": "unavailable",
            "ContentTooShortError": "incomplete",
        }.get(cause_name, "download")

    @staticmethod
    def _to_public_error(failure: _AttemptFailure) -> YtDlpDownloadError:
        prefixes = {
            "cookies": (CookieAcquisitionError, "Falha ao obter cookies"),
            "unsupported": (UnsupportedUrlError, "URL não suportada pelo yt-dlp"),
            "restricted": (RestrictedMediaError, "Mídia restrita"),
            "unavailable": (UnavailableMediaError, "Mídia indisponível"),
            "no_audio": (NoAudioStreamError, "Nenhum stream de áudio disponível"),
            "incomplete": (IncompleteDownloadError, "Download incompleto"),
        }
        error_class, prefix = prefixes.get(
            failure.category,
            (YtDlpDownloadError, "Falha do yt-dlp"),
        )
        return error_class(f"{prefix}: {failure.detail}")

    @staticmethod
    def _optional_text(value: Any) -> str | None:
        return str(value) if value is not None and str(value).strip() else None

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total_seconds = max(0, round(seconds))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    @staticmethod
    def _create_youtube_dl(options: dict[str, Any]) -> _YoutubeDL:
        try:
            import yt_dlp
        except ImportError as error:
            raise YtDlpDownloadError(
                "yt-dlp não está instalado. Instale o projeto com yt-dlp[default]."
            ) from error
        return yt_dlp.YoutubeDL(options)


class InputResolver:
    """Resolve one CLI input and own only the temporary files it creates."""

    def __init__(
        self,
        settings: Settings,
        progress: ProgressCallback | None = None,
        downloader: YtDlpDownloader | None = None,
    ) -> None:
        self._settings = settings
        self._progress = progress or (lambda _message: None)
        self._downloader = downloader

    @contextmanager
    def resolve(self, source: str | Path) -> Iterator[ResolvedInput]:
        """Yield a local path and clean only URL downloads on context exit."""

        source_text = str(source).strip()
        if is_http_url(source_text):
            self._progress("Detectado: URL")
            temporary_directory = Path(
                tempfile.mkdtemp(prefix="podcast-indexer-ytdlp-")
            )
            downloaded: DownloadedAudio | None = None
            finalized = False

            def cleanup() -> None:
                nonlocal finalized
                if finalized:
                    return
                if self._settings.ytdlp_keep_temp:
                    preserved_path = (
                        downloaded.path
                        if downloaded is not None
                        else temporary_directory
                    )
                    self._progress(
                        f"Áudio temporário preservado em: {preserved_path}"
                    )
                elif temporary_directory.exists():
                    self._progress("Removendo áudio temporário...")
                    shutil.rmtree(temporary_directory)
                finalized = True

            try:
                downloader = self._downloader or YtDlpDownloader(
                    settings=self._settings,
                    progress=self._progress,
                )
                downloaded = downloader.download(source_text, temporary_directory)
                yield ResolvedInput(
                    path=downloaded.path,
                    kind="url",
                    metadata=downloaded.metadata,
                    cleanup=cleanup,
                )
            finally:
                cleanup()
            return

        if source_text.lower().startswith(("http:", "https:")):
            raise InvalidInputUrlError(
                f"URL HTTP/HTTPS inválida: {source_text or '(vazia)'}"
            )

        self._progress("Detectado: arquivo local")
        local_path = Path(source).expanduser()
        if not local_path.is_file():
            raise FileNotFoundError(
                f"Arquivo de áudio não encontrado: {local_path}"
            )
        yield ResolvedInput(path=local_path, kind="local")
