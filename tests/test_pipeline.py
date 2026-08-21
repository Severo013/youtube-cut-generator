import json
import tempfile
import unittest
from pathlib import Path

from src.models import IndexResult, Transcript, TranscriptSegment
from src.pipeline import PodcastPipeline


class FakeTranscriber:
    def transcribe(self, audio_path: Path) -> Transcript:
        return Transcript(
            source=str(audio_path),
            duration_seconds=5,
            language="pt",
            full_text="Teste.",
            segments=[TranscriptSegment(start_seconds=0, end_seconds=5, text="Teste.")],
        )


class FakeIndexer:
    def __init__(self) -> None:
        self.received_prompt = ""

    def index(self, transcript: Transcript, prompt: str) -> IndexResult:
        self.received_prompt = prompt
        return IndexResult(
            source=transcript.source,
            model="fake-model",
            index={"title": "Episódio de teste", "chapters": []},
        )


class PipelineTests(unittest.TestCase):
    def test_runs_with_replaceable_providers_and_saves_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            audio_path = root / "episode.mp3"
            audio_path.touch()
            prompt_path = root / "prompt.md"
            prompt_path.write_text("Transcrição:\n{{TRANSCRIPT}}", encoding="utf-8")
            indexer = FakeIndexer()

            pipeline = PodcastPipeline(
                transcriber=FakeTranscriber(),
                indexer=indexer,
                prompt_path=prompt_path,
                output_dir=root / "outputs",
            )
            result = pipeline.run(audio_path)

            self.assertEqual(result.index["title"], "Episódio de teste")
            self.assertIn("[00:00:00 - 00:00:05] Teste.", indexer.received_prompt)

            transcript_json = root / "outputs" / "episode" / "transcript.json"
            index_json = root / "outputs" / "episode" / "index.json"
            self.assertTrue(transcript_json.is_file())
            self.assertTrue(index_json.is_file())
            self.assertEqual(
                json.loads(index_json.read_text(encoding="utf-8"))["index"]["title"],
                "Episódio de teste",
            )


if __name__ == "__main__":
    unittest.main()

