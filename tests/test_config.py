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

    def test_loads_provider_selection_from_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            dotenv_path = Path(temporary_dir) / ".env"
            dotenv_path.write_text("TRANSCRIPTION_PROVIDER=local\n", encoding="utf-8")
            with patch.dict(os.environ, {"TRANSCRIPTION_PROVIDER": "openai"}):
                settings = load_settings(dotenv_path)

        self.assertEqual(settings.transcription_provider, "openai")


if __name__ == "__main__":
    unittest.main()
