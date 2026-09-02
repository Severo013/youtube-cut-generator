import tempfile
import unittest
from pathlib import Path

from src.models import IndexBlock, IndexResult, Transcript, TranscriptSegment
from src.processing import ProcessingRunner, TaskRegistry, UnknownExecutorError


def make_transcript() -> Transcript:
    return Transcript(
        source="episode.mp3",
        duration_seconds=20,
        language="pt",
        full_text="Texto do episódio.",
        segments=[
            TranscriptSegment(
                start_seconds=857,
                end_seconds=860,
                text="Texto do episódio.",
            )
        ],
    )


def make_index_result() -> IndexResult:
    return IndexResult(
        blocks=[
            IndexBlock(
                start_seconds=857,
                end_seconds=860,
                start_timestamp="00:14:17",
                end_timestamp="00:14:20",
                topic="Tema",
                summary="Resumo",
            )
        ]
    )


class RecordingProcessor:
    def __init__(self) -> None:
        self.calls = []

    def process(self, task, transcript, formatted_transcript):
        self.calls.append((task, transcript, formatted_transcript))
        return make_index_result()


class ProcessingRunnerTests(unittest.TestCase):
    def test_routes_task_and_canonical_transcript_to_configured_processor(self) -> None:
        task = TaskRegistry.load_builtin().get("indexacao")
        transcript = make_transcript()
        processor = RecordingProcessor()
        runner = ProcessingRunner({"openai": processor})

        results = runner.run(transcript, [task])

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].task_id, "indexacao")
        self.assertEqual(results[0].task_name, "Indexação")
        self.assertEqual(results[0].executor, "openai")
        self.assertIsInstance(results[0].output, IndexResult)
        self.assertEqual(
            processor.calls,
            [(task, transcript, "[00:14:17 - 00:14:20] Texto do episódio.")],
        )

    def test_runs_multiple_definitions_sequentially_without_task_specific_logic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            for task_id in ("primeira", "segunda"):
                task_dir = root / task_id
                task_dir.mkdir()
                (task_dir / "task.yaml").write_text(
                    "\n".join(
                        [
                            f"id: {task_id}",
                            f"name: {task_id.title()}",
                            "prompt: prompt.md",
                            "executor: fake",
                            "schema: index_result",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                (task_dir / "prompt.md").write_text(
                    "{{TRANSCRIPT}}",
                    encoding="utf-8",
                )
            registry = TaskRegistry.from_directory(
                root,
                schemas={"index_result": IndexResult},
            )
            processor = RecordingProcessor()
            runner = ProcessingRunner({"fake": processor})

            results = runner.run(make_transcript(), registry.all())

        self.assertEqual([result.task_id for result in results], ["primeira", "segunda"])
        self.assertEqual([call[0].id for call in processor.calls], ["primeira", "segunda"])

    def test_rejects_task_with_unregistered_executor(self) -> None:
        task = TaskRegistry.load_builtin().get("indexacao")

        with self.assertRaisesRegex(UnknownExecutorError, "não registrado"):
            ProcessingRunner({}).run(make_transcript(), [task])


if __name__ == "__main__":
    unittest.main()
