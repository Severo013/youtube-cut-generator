import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from openai import OpenAI, OpenAIError
from openai.types.audio.transcription_segment import TranscriptionSegment
from openai.types.audio.transcription_verbose import TranscriptionVerbose
from pydantic import SecretStr

from src.transcription import (
    AudioFileTooLargeError,
    EmptyTranscriptionError,
    EmptyTranscriptTextError,
    InvalidAudioFileError,
    TranscriptionConfigurationError,
    TranscriptionProviderError,
    UnsupportedAudioFormatError,
)
from src.transcription.openai_transcriber import (
    MAX_AUDIO_FILE_SIZE_BYTES,
    OpenAITranscriber,
)


def make_whisper_response(
    *, segments: list[TranscriptionSegment] | None = None
) -> TranscriptionVerbose:
    if segments is None:
        segments = [
            TranscriptionSegment(
                id=0,
                seek=0,
                start=754.2,
                end=761.9,
                text=" Primeiro segmento. ",
                tokens=[],
                temperature=0,
                avg_logprob=-0.1,
                compression_ratio=1.0,
                no_speech_prob=0.01,
            ),
            TranscriptionSegment(
                id=1,
                seek=0,
                start=761.9,
                end=765.1,
                text="Segundo segmento.",
                tokens=[],
                temperature=0,
                avg_logprob=-0.1,
                compression_ratio=1.0,
                no_speech_prob=0.01,
            ),
        ]
    return TranscriptionVerbose(
        duration=765.1,
        language="portuguese",
        text="Primeiro segmento. Segundo segmento.",
        segments=segments,
    )


class OpenAITranscriberTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_dir.name)
        self.audio_path = self.root / "episode.mp3"
        self.audio_path.write_bytes(b"mock audio data")
        self.output_dir = self.root / "outputs" / "episode"
        self.client = MagicMock(spec=OpenAI)

    def tearDown(self) -> None:
        self.temporary_dir.cleanup()

    def build_transcriber(self, api_key: str | None = "test-key") -> OpenAITranscriber:
        return OpenAITranscriber(
            api_key=SecretStr(api_key) if api_key is not None else None,
            model="whisper-1",
            language="pt",
            output_dir=self.output_dir,
            client=self.client,
        )

    def test_converts_openai_response_and_writes_all_artifacts(self) -> None:
        response = make_whisper_response()
        self.client.audio.transcriptions.create.return_value = response

        with patch(
            "src.transcription.openai_transcriber.OpenAI"
        ) as openai_constructor:
            transcript = self.build_transcriber().transcribe(self.audio_path)

        openai_constructor.assert_not_called()

        self.assertEqual(transcript.source, str(self.audio_path))
        self.assertEqual(transcript.full_text, response.text)
        self.assertEqual(transcript.language, "portuguese")
        self.assertEqual(transcript.duration_seconds, 765.1)
        self.assertEqual(transcript.segments[0].start_seconds, 754.2)
        self.assertEqual(transcript.segments[0].end_seconds, 761.9)
        self.assertEqual(transcript.segments[0].text, "Primeiro segmento.")

        call = self.client.audio.transcriptions.create.call_args
        self.assertEqual(call.kwargs["model"], "whisper-1")
        self.assertEqual(call.kwargs["language"], "pt")
        self.assertEqual(call.kwargs["response_format"], "verbose_json")
        self.assertEqual(call.kwargs["timestamp_granularities"], ["segment"])
        self.assertNotIn("chunking_strategy", call.kwargs)

        raw_json = json.loads(
            (self.output_dir / "transcript.raw.json").read_text(encoding="utf-8")
        )
        normalized_json = json.loads(
            (self.output_dir / "transcript.json").read_text(encoding="utf-8")
        )
        readable_text = (self.output_dir / "transcript.txt").read_text(
            encoding="utf-8"
        )
        self.assertEqual(raw_json["segments"][0]["start"], 754.2)
        self.assertEqual(normalized_json["segments"][0]["start_seconds"], 754.2)
        self.assertEqual(
            readable_text.splitlines()[0],
            "[00:12:34 - 00:12:41] Primeiro segmento.",
        )

    def test_rejects_response_without_segments(self) -> None:
        self.client.audio.transcriptions.create.return_value = make_whisper_response(
            segments=[]
        )

        with self.assertRaisesRegex(EmptyTranscriptionError, "não contém segmentos"):
            self.build_transcriber().transcribe(self.audio_path)

    def test_rejects_response_without_text(self) -> None:
        response = make_whisper_response()
        response.text = ""
        self.client.audio.transcriptions.create.return_value = response

        with self.assertRaisesRegex(EmptyTranscriptTextError, "não contém texto"):
            self.build_transcriber().transcribe(self.audio_path)

    def test_rejects_missing_api_key_without_calling_api(self) -> None:
        with self.assertRaisesRegex(
            TranscriptionConfigurationError, "OPENAI_API_KEY"
        ):
            self.build_transcriber(api_key=None).transcribe(self.audio_path)

        self.client.audio.transcriptions.create.assert_not_called()

    def test_rejects_nonexistent_file_without_calling_api(self) -> None:
        missing_path = self.root / "missing.mp3"

        with self.assertRaisesRegex(FileNotFoundError, "não encontrado"):
            self.build_transcriber().transcribe(missing_path)

        self.client.audio.transcriptions.create.assert_not_called()

    def test_rejects_unsupported_extension(self) -> None:
        unsupported_path = self.root / "episode.aac"
        unsupported_path.write_bytes(b"mock audio data")

        with self.assertRaisesRegex(UnsupportedAudioFormatError, "não suportada"):
            self.build_transcriber().transcribe(unsupported_path)

    def test_rejects_empty_audio_file(self) -> None:
        empty_path = self.root / "empty.wav"
        empty_path.touch()

        with self.assertRaisesRegex(InvalidAudioFileError, "está vazio"):
            self.build_transcriber().transcribe(empty_path)

    def test_rejects_audio_larger_than_provider_limit(self) -> None:
        large_path = self.root / "large.mp3"
        with large_path.open("wb") as large_file:
            large_file.seek(MAX_AUDIO_FILE_SIZE_BYTES)
            large_file.write(b"\0")

        with self.assertRaisesRegex(AudioFileTooLargeError, "limite de 25 MB"):
            self.build_transcriber().transcribe(large_path)

        self.client.audio.transcriptions.create.assert_not_called()

    def test_preserves_api_error_as_exception_cause(self) -> None:
        api_error = OpenAIError("service unavailable")
        self.client.audio.transcriptions.create.side_effect = api_error

        with self.assertRaisesRegex(TranscriptionProviderError, "OpenAI falhou") as raised:
            self.build_transcriber().transcribe(self.audio_path)

        self.assertIs(raised.exception.__cause__, api_error)


if __name__ == "__main__":
    unittest.main()
