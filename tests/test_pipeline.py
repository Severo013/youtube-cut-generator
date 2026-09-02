import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.models import IndexBlock, IndexResult, Transcript, TranscriptSegment
from src.pipeline import PodcastPipeline
from src.processing import (
    ProcessingResult,
    ProcessingRunner,
    TaskExecutionError,
    TaskRegistry,
)
from src.tasks.shorts.models import ShortMoment, ShortsResult


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


class RecordingRunner:
    def __init__(self) -> None:
        self.calls = []

    def run(self, transcript, tasks, *, on_result=None):
        task_list = list(tasks)
        self.calls.append((transcript, task_list))
        task = task_list[0]
        result = ProcessingResult[IndexResult](
                task_id=task.id,
                task_name=task.name,
                executor=task.executor,
                schema_id=task.schema_id,
                output=make_index(),
            )
        if on_result is not None:
            on_result(task, result)
        return [result]


class MultiSchemaProcessor:
    def __init__(self, failing_task: str | None = None) -> None:
        self.failing_task = failing_task
        self.calls = []

    def process(self, task, transcript, formatted_transcript):
        self.calls.append((task, transcript, formatted_transcript))
        if task.id == self.failing_task:
            raise RuntimeError("processor failed")
        if task.schema_id == "shorts_result":
            return ShortsResult(
                moments=[
                    ShortMoment(
                        start_seconds=5,
                        end_seconds=18,
                        title="Short de teste",
                    )
                ]
            )
        return make_index()

    def model_for(self, task):
        return "fake-model"


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

    def test_generic_pipeline_executes_indexacao_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            transcript_path = root / "transcript.json"
            transcript_path.write_text(
                make_transcript().model_dump_json(indent=2),
                encoding="utf-8",
            )
            runner = RecordingRunner()
            registry = TaskRegistry.load_builtin()
            pipeline = PodcastPipeline.for_processing(
                runner=runner,
                task_registry=registry,
                output_dir=root,
            )

            result = pipeline.index_transcript_file(transcript_path)

            self.assertIsInstance(result, IndexResult)
            self.assertEqual(len(runner.calls), 1)
            self.assertEqual([task.id for task in runner.calls[0][1]], ["indexacao"])
            self.assertTrue((root / "index.json").is_file())
            self.assertTrue((root / "index.md").is_file())

    def test_processes_indexacao_and_shorts_with_independent_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            transcript_path = root / "transcript.json"
            transcript = make_transcript()
            transcript_path.write_text(
                transcript.model_dump_json(indent=2),
                encoding="utf-8",
            )
            registry = TaskRegistry.load_builtin()
            processor = MultiSchemaProcessor()
            pipeline = PodcastPipeline.for_processing(
                runner=ProcessingRunner({"openai": processor}),
                task_registry=registry,
                output_dir=root,
            )

            results = pipeline.process_transcript_file(
                transcript_path,
                registry.resolve(("indexacao", "shorts")),
            )

            self.assertEqual([result.task_id for result in results], ["indexacao", "shorts"])
            self.assertIsInstance(results[0].output, IndexResult)
            self.assertIsInstance(results[1].output, ShortsResult)
            self.assertTrue((root / "processing" / "indexacao" / "result.json").is_file())
            self.assertTrue((root / "processing" / "shorts" / "result.json").is_file())
            self.assertTrue((root / "index.json").is_file())
            self.assertEqual(len(processor.calls), 2)
            self.assertIs(processor.calls[0][1], processor.calls[1][1])
            self.assertIs(processor.calls[0][2], processor.calls[1][2])

    def test_completed_output_survives_later_task_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            transcript_path = root / "transcript.json"
            transcript_path.write_text(
                make_transcript().model_dump_json(indent=2),
                encoding="utf-8",
            )
            tasks_root = root / "tasks"
            tasks_root.mkdir()
            for task_id in ("primeira", "segunda", "terceira"):
                task_dir = tasks_root / task_id
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
                (task_dir / "prompt.md").write_text("{{TRANSCRIPT}}", encoding="utf-8")
            registry = TaskRegistry.from_directory(
                tasks_root,
                schemas={"index_result": IndexResult},
            )
            processor = MultiSchemaProcessor(failing_task="segunda")
            pipeline = PodcastPipeline.for_processing(
                runner=ProcessingRunner({"fake": processor}),
                task_registry=registry,
                output_dir=root,
            )

            with self.assertRaisesRegex(TaskExecutionError, "segunda.*processor failed"):
                pipeline.process_transcript_file(transcript_path, registry.all())

            self.assertTrue((root / "processing" / "primeira" / "result.json").is_file())
            self.assertFalse((root / "processing" / "segunda").exists())
            self.assertFalse((root / "processing" / "terceira").exists())
            self.assertEqual([call[0].id for call in processor.calls], ["primeira", "segunda"])

    def test_new_task_definition_requires_no_pipeline_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            transcript_path = root / "transcript.json"
            transcript_path.write_text(
                make_transcript().model_dump_json(indent=2),
                encoding="utf-8",
            )
            tasks_root = root / "tasks"
            task_dir = tasks_root / "nova_task"
            task_dir.mkdir(parents=True)
            (task_dir / "task.yaml").write_text(
                "\n".join(
                    [
                        "id: nova_task",
                        "name: Nova task",
                        "prompt: prompt.md",
                        "executor: fake",
                        "schema: index_result",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (task_dir / "prompt.md").write_text("{{TRANSCRIPT}}", encoding="utf-8")
            registry = TaskRegistry.from_directory(
                tasks_root,
                schemas={"index_result": IndexResult},
            )
            pipeline = PodcastPipeline.for_processing(
                runner=ProcessingRunner({"fake": MultiSchemaProcessor()}),
                task_registry=registry,
                output_dir=root,
            )

            results = pipeline.process_transcript_file(
                transcript_path,
                registry.all(),
            )

            self.assertEqual([result.task_id for result in results], ["nova_task"])
            self.assertTrue((root / "processing" / "nova_task" / "result.json").is_file())


if __name__ == "__main__":
    unittest.main()
