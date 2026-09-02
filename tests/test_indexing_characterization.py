import tempfile
import unittest
from pathlib import Path

from src.models import IndexBlock, IndexResult, Transcript, TranscriptSegment
from src.pipeline import PodcastPipeline


class RecordingIndexer:
    def __init__(self, result: IndexResult) -> None:
        self._result = result
        self.calls: list[tuple[Transcript, str]] = []

    def index(self, transcript: Transcript, prompt: str) -> IndexResult:
        self.calls.append((transcript, prompt))
        return self._result


class LegacyIndexingCharacterizationTests(unittest.TestCase):
    def test_reindexing_contract_before_generic_processing_refactor(self) -> None:
        transcript = Transcript(
            source="episode.mp3",
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
        index_result = IndexResult(
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

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            transcript_path = root / "transcript.json"
            transcript_path.write_text(
                transcript.model_dump_json(indent=2),
                encoding="utf-8",
            )
            prompt_path = root / "prompt.md"
            prompt_path.write_text(
                "Antes\n{{TRANSCRIPT}}\nDepois",
                encoding="utf-8",
            )
            indexer = RecordingIndexer(index_result)
            pipeline = PodcastPipeline(
                indexer=indexer,
                prompt_path=prompt_path,
                output_dir=root / "unused",
            )

            result = pipeline.index_transcript_file(transcript_path)

            self.assertIs(result, index_result)
            self.assertEqual(len(indexer.calls), 1)
            received_transcript, received_prompt = indexer.calls[0]
            self.assertEqual(received_transcript, transcript)
            self.assertEqual(
                received_prompt,
                "Antes\n"
                "[00:00:05 - 00:00:11] Primeira fala.\n"
                "[00:00:11 - 00:00:18] Segunda fala.\n"
                "Depois",
            )
            self.assertEqual(
                (root / "index.json").read_text(encoding="utf-8"),
                "{\n"
                '  "blocks": [\n'
                "    {\n"
                '      "start_seconds": 5.0,\n'
                '      "end_seconds": 18.0,\n'
                '      "start_timestamp": "00:00:05",\n'
                '      "end_timestamp": "00:00:18",\n'
                '      "topic": "Tema de teste",\n'
                '      "summary": "Resumo do assunto abordado."\n'
                "    }\n"
                "  ]\n"
                "}\n",
            )
            self.assertEqual(
                (root / "index.md").read_text(encoding="utf-8"),
                "# Indexação do episódio\n\n"
                "## 00:00:05 - 00:00:18 — Tema de teste\n\n"
                "Resumo do assunto abordado.\n",
            )


if __name__ == "__main__":
    unittest.main()
