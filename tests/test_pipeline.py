import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.models import IndexBlock, IndexResult, Transcript, TranscriptSegment
from src.pipeline import PodcastPipeline


def make_transcript(source: str = "episode.mp3") -> Transcript:
    return Transcript(
        source=source,
        duration_seconds=18,
        language="pt",
        full_text="Primeira fala. Segunda fala.",
        segments=[
            TranscriptSegment(
                start_seconds=5,
                end_seconds=11,
                text="Primeira fala.",
            ),
            TranscriptSegment(
                start_seconds=11,
                end_seconds=18,
                text="Segunda fala.",
            ),
        ],
    )


def make_index() -> IndexResult:
    return IndexResult(
        blocks=[
            IndexBlock(
                start_seconds=5,
                end_seconds=18,
                start_timestamp="00:00:05",
                end_timestamp="00:00:18",
                topic="Tema de teste",
                summary="Resumo do assunto abordado.",
            )
        ]
    )


class FakeTranscriber:
    def transcribe(self, audio_path: Path) -> Transcript:
        return make_transcript(str(audio_path))


class FakeIndexer:
    def __init__(self) -> None:
        self.received_transcript: Transcript | None = None
        self.received_prompt = ""

    def index(self, transcript: Transcript, prompt: str) -> IndexResult:
        self.received_transcript = transcript
        self.received_prompt = prompt
        return make_index()


class PipelineTests(unittest.TestCase):
    def test_runs_with_replaceable_providers_and_saves_all_normalized_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            audio_path = root / "episode.mp3"
            audio_path.write_bytes(b"fake audio")
            prompt_path = root / "prompt.md"
            prompt_path.write_text("Transcrição:\n{{TRANSCRIPT}}", encoding="utf-8")
            indexer = FakeIndexer()
            progress: list[str] = []

            pipeline = PodcastPipeline(
                transcriber=FakeTranscriber(),
                indexer=indexer,
                prompt_path=prompt_path,
                output_dir=root / "outputs",
                progress=progress.append,
            )
            result = pipeline.run(audio_path)

            self.assertEqual(result.blocks[0].topic, "Tema de teste")
            self.assertIn(
                "[00:00:05 - 00:00:11] Primeira fala.",
                indexer.received_prompt,
            )
            self.assertEqual(
                progress,
                [
                    "Validando arquivo",
                    "Transcrevendo",
                    "Salvando transcrição",
                    "Indexando episódio",
                    "Salvando resultado",
                    "Concluído",
                ],
            )

            episode_dir = root / "outputs" / "episode"
            transcript_payload = json.loads(
                (episode_dir / "transcript.json").read_text(encoding="utf-8")
            )
            index_payload = json.loads(
                (episode_dir / "index.json").read_text(encoding="utf-8")
            )
            transcript_text = (episode_dir / "transcript.txt").read_text(
                encoding="utf-8"
            )
            index_markdown = (episode_dir / "index.md").read_text(encoding="utf-8")

            self.assertEqual(transcript_payload["language"], "pt")
            self.assertEqual(index_payload["blocks"][0]["topic"], "Tema de teste")
            self.assertIn("[00:00:11 - 00:00:18] Segunda fala.", transcript_text)
            self.assertIn(
                "## 00:00:05 - 00:00:18 — Tema de teste",
                index_markdown,
            )
            self.assertIn("Resumo do assunto abordado.", index_markdown)

    def test_reindexes_existing_transcript_json_without_transcriber(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            transcript_path = root / "transcript.json"
            transcript_path.write_text(
                make_transcript().model_dump_json(indent=2),
                encoding="utf-8",
            )
            prompt_path = root / "prompt.md"
            prompt_path.write_text("{{TRANSCRIPT}}", encoding="utf-8")
            indexer = FakeIndexer()

            pipeline = PodcastPipeline(
                indexer=indexer,
                prompt_path=prompt_path,
                output_dir=root / "unused",
            )
            with patch(
                "src.transcription.faster_whisper_transcriber."
                "FasterWhisperTranscriber._get_model"
            ) as load_whisper:
                result = pipeline.index_transcript_file(transcript_path)

            self.assertEqual(result.blocks[0].topic, "Tema de teste")
            load_whisper.assert_not_called()
            self.assertIsNotNone(indexer.received_transcript)
            self.assertTrue((root / "index.json").is_file())
            self.assertTrue((root / "index.md").is_file())
            self.assertFalse((root / "unused").exists())

    def test_run_can_use_an_explicit_output_directory_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            audio_path = root / "temporary-video-id.webm"
            audio_path.write_bytes(b"fake audio")
            prompt_path = root / "prompt.md"
            prompt_path.write_text("{{TRANSCRIPT}}", encoding="utf-8")

            pipeline = PodcastPipeline(
                transcriber=FakeTranscriber(),
                indexer=FakeIndexer(),
                prompt_path=prompt_path,
                output_dir=root / "outputs",
            )
            pipeline.run(audio_path, output_name="Video title")

            self.assertTrue(
                (root / "outputs" / "Video title" / "index.json").is_file()
            )
            self.assertFalse((root / "outputs" / "temporary-video-id").exists())


if __name__ == "__main__":
    unittest.main()
