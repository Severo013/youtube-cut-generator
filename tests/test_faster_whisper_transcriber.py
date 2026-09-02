import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.transcription import TranscriptionProviderError
from src.transcription.faster_whisper_transcriber import FasterWhisperTranscriber


class FakeWhisperModel:
    def __init__(self) -> None:
        self.model = SimpleNamespace(device="cuda", compute_type="float16")
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.consumed_segments = 0

    def transcribe(self, audio_path: str, **kwargs: object):
        self.calls.append((audio_path, kwargs))

        def generate_segments():
            for segment in (
                SimpleNamespace(
                    id=0,
                    seek=0,
                    start=1.25,
                    end=4.75,
                    text=" Primeira fala. ",
                    tokens=[1, 2],
                    temperature=0.0,
                    avg_logprob=-0.1,
                    compression_ratio=1.1,
                    no_speech_prob=0.01,
                ),
                SimpleNamespace(
                    id=1,
                    seek=0,
                    start=4.75,
                    end=8.5,
                    text="Segunda fala.",
                    tokens=[3, 4],
                    temperature=0.0,
                    avg_logprob=-0.2,
                    compression_ratio=1.0,
                    no_speech_prob=0.02,
                ),
            ):
                self.consumed_segments += 1
                yield segment

        info = SimpleNamespace(
            language="pt",
            language_probability=0.99,
            duration=8.5,
            duration_after_vad=8.5,
        )
        return generate_segments(), info


class FasterWhisperTranscriberTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_dir.name)
        self.audio_path = self.root / "episode.mp3"
        self.audio_path.write_bytes(b"fake audio")
        self.output_dir = self.root / "outputs" / "episode"

    def tearDown(self) -> None:
        self.temporary_dir.cleanup()

    def test_normalizes_all_segments_and_writes_provider_artifacts(self) -> None:
        fake_model = FakeWhisperModel()
        model_factory = Mock(return_value=fake_model)
        progress: list[str] = []
        transcriber = FasterWhisperTranscriber(
            output_dir=self.output_dir,
            progress=progress.append,
            model_factory=model_factory,
        )

        with patch.object(transcriber, "_validate_cuda"):
            transcript = transcriber.transcribe(self.audio_path)

        model_factory.assert_called_once_with(
            "turbo", device="cuda", compute_type="float16"
        )
        self.assertEqual(fake_model.consumed_segments, 2)
        self.assertEqual(transcript.source, str(self.audio_path))
        self.assertEqual(transcript.language, "pt")
        self.assertEqual(transcript.duration_seconds, 8.5)
        self.assertEqual(transcript.full_text, "Primeira fala. Segunda fala.")
        self.assertEqual(transcript.segments[0].start_seconds, 1.25)
        self.assertEqual(transcript.segments[0].end_seconds, 4.75)
        self.assertEqual(transcript.segments[1].start_seconds, 4.75)
        self.assertEqual(transcript.segments[1].end_seconds, 8.5)
        self.assertEqual(
            fake_model.calls[0][1],
            {"language": "pt", "vad_filter": True, "word_timestamps": False},
        )

        raw_payload = json.loads(
            (self.output_dir / "transcript.raw.json").read_text(encoding="utf-8")
        )
        normalized_payload = json.loads(
            (self.output_dir / "transcript.json").read_text(encoding="utf-8")
        )
        transcript_text = (self.output_dir / "transcript.txt").read_text(
            encoding="utf-8"
        )
        self.assertEqual(raw_payload["provider"], "faster-whisper")
        self.assertEqual(raw_payload["model"], "turbo")
        self.assertEqual(raw_payload["language_probability"], 0.99)
        self.assertEqual(len(raw_payload["segments"]), 2)
        self.assertEqual(normalized_payload, transcript.model_dump(mode="json"))
        self.assertEqual(
            transcript_text.splitlines()[0],
            "[00:00:01 - 00:00:04] Primeira fala.",
        )
        self.assertEqual(progress[0], "Carregando Whisper local...")
        self.assertTrue(progress[-1].startswith("Transcrição concluída"))

    def test_reuses_one_model_instance(self) -> None:
        fake_model = FakeWhisperModel()
        model_factory = Mock(return_value=fake_model)
        transcriber = FasterWhisperTranscriber(model_factory=model_factory)

        with patch.object(transcriber, "_validate_cuda") as validate_cuda:
            transcriber.transcribe(self.audio_path)
            transcriber.transcribe(self.audio_path)

        model_factory.assert_called_once()
        validate_cuda.assert_called_once()
        self.assertEqual(len(fake_model.calls), 2)

    def test_cuda_failure_does_not_retry_on_cpu(self) -> None:
        model_factory = Mock(side_effect=RuntimeError("missing cublas64_12.dll"))
        transcriber = FasterWhisperTranscriber(model_factory=model_factory)

        with patch.object(transcriber, "_validate_cuda"):
            with self.assertRaisesRegex(
                TranscriptionProviderError,
                "GPU CUDA.*Nenhum fallback para CPU",
            ) as raised:
                transcriber.transcribe(self.audio_path)

        self.assertIsInstance(raised.exception.__cause__, RuntimeError)
        model_factory.assert_called_once_with(
            "turbo", device="cuda", compute_type="float16"
        )


if __name__ == "__main__":
    unittest.main()
