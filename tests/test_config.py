import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config import DEFAULT_INDEX_MODEL, Settings, load_settings


class SettingsTests(unittest.TestCase):
    def test_local_whisper_is_the_default_transcription_provider(self) -> None:
        settings = Settings()

        self.assertEqual(settings.transcription_provider, "local")
        self.assertEqual(settings.transcription_model, "whisper-1")
        self.assertEqual(settings.local_whisper_model, "turbo")
        self.assertEqual(settings.local_whisper_device, "cuda")
        self.assertEqual(settings.local_whisper_compute_type, "float16")
        self.assertEqual(settings.transcription_language, "pt")
        self.assertEqual(settings.index_model, DEFAULT_INDEX_MODEL)
        self.assertFalse(settings.ytdlp_keep_temp)
        self.assertEqual(settings.ytdlp_auth_mode, "auto")
        self.assertEqual(settings.ytdlp_browser, "firefox")
        self.assertIsNone(settings.ytdlp_cookies_file)

    def test_loads_provider_selection_from_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            dotenv_path = Path(temporary_dir) / ".env"
            dotenv_path.write_text("TRANSCRIPTION_PROVIDER=local\n", encoding="utf-8")
            with patch.dict(os.environ, {"TRANSCRIPTION_PROVIDER": "openai"}):
                settings = load_settings(dotenv_path)

        self.assertEqual(settings.transcription_provider, "openai")

    def test_loads_ytdlp_settings_from_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            dotenv_path = Path(temporary_dir) / ".env"
            cookie_path = Path(temporary_dir) / "cookies.txt"
            dotenv_path.write_text(
                "\n".join(
                    [
                        "YTDLP_KEEP_TEMP=true",
                        "YTDLP_AUTH_MODE=file",
                        "YTDLP_BROWSER=chrome",
                        f"YTDLP_COOKIES_FILE={cookie_path}",
                    ]
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                settings = load_settings(dotenv_path)

        self.assertTrue(settings.ytdlp_keep_temp)
        self.assertEqual(settings.ytdlp_auth_mode, "file")
        self.assertEqual(settings.ytdlp_browser, "chrome")
        self.assertEqual(settings.ytdlp_cookies_file, cookie_path)


if __name__ == "__main__":
    unittest.main()
