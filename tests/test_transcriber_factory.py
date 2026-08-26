import unittest
from unittest.mock import patch

from src.config import Settings
from src.transcription import TranscriptionConfigurationError
from src.transcription.factory import create_transcriber


class TranscriberFactoryTests(unittest.TestCase):
    @patch("src.transcription.factory.FasterWhisperTranscriber")
    def test_selects_local_provider_without_openai_key(self, constructor) -> None:
        settings = Settings(openai_api_key=None, transcription_provider="local")

        result = create_transcriber(settings)

        self.assertIs(result, constructor.return_value)
        constructor.assert_called_once_with(
            model="turbo",
            device="cuda",
            compute_type="float16",
            language="pt",
            output_dir=None,
            progress=None,
        )

    @patch("src.transcription.factory.OpenAITranscriber")
    def test_selects_openai_provider(self, constructor) -> None:
        settings = Settings(
            openai_api_key="test-key",
            transcription_provider="openai",
            transcription_model="whisper-1",
        )

        result = create_transcriber(settings)

        self.assertIs(result, constructor.return_value)
        constructor.assert_called_once_with(
            api_key=settings.openai_api_key,
            model="whisper-1",
            language="pt",
            output_dir=None,
        )

    def test_rejects_unknown_provider(self) -> None:
        settings = Settings(transcription_provider="invalid")

        with self.assertRaisesRegex(
            TranscriptionConfigurationError,
            "TRANSCRIPTION_PROVIDER desconhecido",
        ):
            create_transcriber(settings)


if __name__ == "__main__":
    unittest.main()
