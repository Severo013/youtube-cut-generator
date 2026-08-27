import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.config import Settings
from src.input_resolution import (
    AutoAuthenticationError,
    DownloadMetadata,
    DownloadedAudio,
    InputResolver,
    InvalidInputUrlError,
    YtDlpDownloader,
    is_http_url,
)


URL = "https://www.youtube.com/watch?v=test-video&list=test-playlist"


def make_metadata(temporary_directory: Path, auth: str = "none") -> DownloadMetadata:
    return DownloadMetadata(
        video_id="test-video",
        title="Episode title",
        webpage_url=URL,
        duration_seconds=65.0,
        uploader="Test channel",
        extension="webm",
        format_id="251",
        audio_codec="opus",
        video_codec="none",
        authentication_method=auth,
        file_size_bytes=5,
        download_seconds=0.25,
        temporary_directory=temporary_directory,
    )


class FakeDownloader:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Path]] = []

    def download(self, url: str, temporary_directory: Path) -> DownloadedAudio:
        self.calls.append((url, temporary_directory))
        temporary_directory.mkdir(parents=True, exist_ok=True)
        audio_path = temporary_directory / "test-video.webm"
        audio_path.write_bytes(b"audio")
        return DownloadedAudio(
            path=audio_path,
            metadata=make_metadata(temporary_directory),
        )


class FakeYoutubeDL:
    def __init__(self, options: dict[str, object], outcome: object) -> None:
        self.options = options
        self.outcome = outcome

    def __enter__(self):
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def extract_info(self, url: str, *, download: bool):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        template = str(self.options["outtmpl"])
        audio_path = Path(
            template.replace("%(id)s", "test-video").replace("%(ext)s", "webm")
        )
        audio_path.write_bytes(b"audio")
        info = {
            "id": "test-video",
            "title": "Episode title",
            "webpage_url": url,
            "duration": 65,
            "channel": "Test channel",
            "requested_downloads": [
                {
                    "filepath": str(audio_path),
                    "ext": "webm",
                    "format_id": "251",
                    "acodec": "opus",
                    "vcodec": "none",
                }
            ],
        }
        for hook in self.options["progress_hooks"]:
            hook({"status": "finished", "filename": str(audio_path)})
        return info

    def prepare_filename(self, info: dict[str, object]) -> str:
        return str(info["requested_downloads"][0]["filepath"])


class RecordingYdlFactory:
    def __init__(self, *outcomes: object) -> None:
        self.outcomes = list(outcomes)
        self.options: list[dict[str, object]] = []

    def __call__(self, options: dict[str, object]) -> FakeYoutubeDL:
        self.options.append(options)
        return FakeYoutubeDL(options, self.outcomes.pop(0))


class InputResolverTests(unittest.TestCase):
    def test_detects_http_urls_without_misclassifying_windows_paths(self) -> None:
        self.assertTrue(is_http_url("https://example.com/audio"))
        self.assertTrue(is_http_url("HTTP://example.com/audio.mp3"))
        self.assertFalse(is_http_url(r"D:\Podcast\audio.mp3"))
        self.assertFalse(is_http_url("https://"))

    def test_invalid_http_url_has_a_distinct_error(self) -> None:
        resolver = InputResolver(Settings())

        with self.assertRaises(InvalidInputUrlError):
            with resolver.resolve("https://"):
                pass

    def test_local_file_is_returned_without_copy_or_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            audio_path = Path(temporary_dir) / "episode.mp3"
            audio_path.write_bytes(b"original")
            resolver = InputResolver(Settings())

            with resolver.resolve(audio_path) as resolved:
                self.assertEqual(resolved.path, audio_path)
                self.assertEqual(resolved.kind, "local")

            self.assertEqual(audio_path.read_bytes(), b"original")

    def test_url_calls_downloader_and_removes_temporary_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            download_root = Path(temporary_dir) / "download"
            downloader = FakeDownloader()
            resolver = InputResolver(Settings(), downloader=downloader)

            with patch(
                "src.input_resolution.tempfile.mkdtemp",
                return_value=str(download_root),
            ):
                with resolver.resolve(URL) as resolved:
                    downloaded_path = resolved.path
                    self.assertTrue(downloaded_path.is_file())

            self.assertEqual(downloader.calls, [(URL, download_root)])
            self.assertFalse(download_root.exists())
            self.assertFalse(downloaded_path.exists())

    def test_cleanup_runs_when_a_later_stage_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            download_root = Path(temporary_dir) / "download"
            resolver = InputResolver(Settings(), downloader=FakeDownloader())

            with patch(
                "src.input_resolution.tempfile.mkdtemp",
                return_value=str(download_root),
            ):
                with self.assertRaisesRegex(RuntimeError, "later stage"):
                    with resolver.resolve(URL):
                        raise RuntimeError("later stage")

            self.assertFalse(download_root.exists())

    def test_keep_temp_preserves_download_and_reports_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            download_root = Path(temporary_dir) / "download"
            progress: list[str] = []
            resolver = InputResolver(
                Settings(ytdlp_keep_temp=True),
                progress=progress.append,
                downloader=FakeDownloader(),
            )

            with patch(
                "src.input_resolution.tempfile.mkdtemp",
                return_value=str(download_root),
            ):
                with resolver.resolve(URL) as resolved:
                    downloaded_path = resolved.path

            self.assertTrue(downloaded_path.is_file())
            self.assertIn(str(downloaded_path), progress[-1])


class YtDlpDownloaderTests(unittest.TestCase):
    def download(self, settings: Settings, *outcomes: object):
        temporary_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_dir.cleanup)
        factory = RecordingYdlFactory(*outcomes)
        downloader = YtDlpDownloader(settings=settings, ydl_factory=factory)
        result = downloader.download(URL, Path(temporary_dir.name))
        return result, factory

    def test_auth_none_uses_no_cookies_and_audio_only_options(self) -> None:
        result, factory = self.download(Settings(ytdlp_auth_mode="none"), object())

        options = factory.options[0]
        self.assertNotIn("cookiefile", options)
        self.assertNotIn("cookiesfrombrowser", options)
        self.assertEqual(options["format"], "bestaudio/best")
        self.assertTrue(options["noplaylist"])
        self.assertNotIn("postprocessors", options)
        self.assertEqual(result.metadata.authentication_method, "none")
        self.assertEqual(result.metadata.video_id, "test-video")
        self.assertEqual(result.metadata.uploader, "Test channel")

    def test_auth_browser_uses_current_python_api_parameter(self) -> None:
        _, factory = self.download(
            Settings(ytdlp_auth_mode="browser", ytdlp_browser="firefox"),
            object(),
        )

        self.assertEqual(factory.options[0]["cookiesfrombrowser"], ("firefox",))
        self.assertNotIn("cookiefile", factory.options[0])

    def test_auth_file_uses_current_python_api_parameter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            cookie_file = Path(temporary_dir) / "cookies.txt"
            cookie_file.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
            _, factory = self.download(
                Settings(ytdlp_auth_mode="file", ytdlp_cookies_file=cookie_file),
                object(),
            )

        self.assertEqual(factory.options[0]["cookiefile"], str(cookie_file))
        self.assertNotIn("cookiesfrombrowser", factory.options[0])

    def test_auth_auto_stops_after_anonymous_success(self) -> None:
        result, factory = self.download(Settings(ytdlp_auth_mode="auto"), object())

        self.assertEqual(len(factory.options), 1)
        self.assertNotIn("cookiesfrombrowser", factory.options[0])
        self.assertEqual(result.metadata.authentication_method, "none")

    def test_auth_auto_retries_once_with_configured_authentication(self) -> None:
        result, factory = self.download(
            Settings(ytdlp_auth_mode="auto", ytdlp_browser="firefox"),
            RuntimeError("anonymous failed"),
            object(),
        )

        self.assertEqual(len(factory.options), 2)
        self.assertNotIn("cookiesfrombrowser", factory.options[0])
        self.assertEqual(factory.options[1]["cookiesfrombrowser"], ("firefox",))
        self.assertEqual(result.metadata.authentication_method, "browser")

    def test_auth_auto_preserves_context_from_both_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            factory = RecordingYdlFactory(
                RuntimeError("anonymous failed"),
                RuntimeError("authenticated failed"),
            )
            downloader = YtDlpDownloader(
                Settings(ytdlp_auth_mode="auto"),
                ydl_factory=factory,
            )

            with self.assertRaises(AutoAuthenticationError) as raised:
                downloader.download(URL, Path(temporary_dir))

        message = str(raised.exception)
        self.assertIn("anonymous failed", message)
        self.assertIn("authenticated failed", message)
        self.assertEqual(len(factory.options), 2)


if __name__ == "__main__":
    unittest.main()
