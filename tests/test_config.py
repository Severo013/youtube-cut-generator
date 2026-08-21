import unittest

from src.config import Settings


class SettingsTests(unittest.TestCase):
    def test_whisper_is_the_central_default_transcription_model(self) -> None:
        settings = Settings()

        self.assertEqual(settings.transcription_model, "whisper-1")
        self.assertEqual(settings.transcription_language, "pt")


if __name__ == "__main__":
    unittest.main()
