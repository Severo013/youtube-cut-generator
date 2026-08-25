import unittest

from src.config import DEFAULT_INDEX_MODEL, Settings


class SettingsTests(unittest.TestCase):
    def test_whisper_is_the_central_default_transcription_model(self) -> None:
        settings = Settings()

        self.assertEqual(settings.transcription_model, "whisper-1")
        self.assertEqual(settings.transcription_language, "pt")
        self.assertEqual(settings.index_model, DEFAULT_INDEX_MODEL)


if __name__ == "__main__":
    unittest.main()
