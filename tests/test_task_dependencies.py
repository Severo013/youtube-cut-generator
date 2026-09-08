import tempfile
import unittest
from pathlib import Path

from pydantic import BaseModel

from src.models import IndexBlock, IndexResult, Transcript, TranscriptSegment
from src.pipeline import PodcastPipeline
from src.processing import (
    InvalidExecutionPlanError,
    ProcessingContext,
    ProcessingResult,
    ProcessingRunner,
    TaskDefinition,
    TaskExecutionError,
    TaskRegistry,
    validate_execution_plan,
)
from src.processing.prompting import build_prompt_variables


def make_transcript(text: str = "Texto do episódio.") -> Transcript:
    return Transcript(
        source="episode.mp3",
        duration_seconds=10,
        language="pt",
        full_text=text,
        segments=[TranscriptSegment(start_seconds=0, end_seconds=10, text=text)],
    )


def make_output(topic: str) -> IndexResult:
    return IndexResult(
        blocks=[
            IndexBlock(
                start_seconds=0,
                end_seconds=10,
                start_timestamp="00:00:00",
                end_timestamp="00:00:10",
                topic=topic,
                summary=f"Resumo {topic}",
            )
        ]
    )


def make_result(task_id: str, topic: str, schema_id: str = "index_result"):
    return ProcessingResult[BaseModel](
        task_id=task_id,
        task_name=task_id.title(),
        executor="fake",
        schema_id=schema_id,
        output=make_output(topic),
    )


def make_task(
    root: Path,
    task_id: str,
    *,
    requires: list[str] | None = None,
    settings: dict[str, object] | None = None,
    prompt: str | None = None,
) -> TaskDefinition:
    task_dir = root / task_id
    task_dir.mkdir()
    (task_dir / "prompt.md").write_text(
        prompt or "{{TRANSCRIPT}}",
        encoding="utf-8",
    )
    task = TaskDefinition.model_validate(
        {
            "id": task_id,
            "name": task_id.title(),
            "prompt": "prompt.md",
            "executor": "fake",
            "schema": "index_result",
            "requires": requires or [],
            "settings": settings or {},
        }
    )
    task.bind_runtime(resource_directory=task_dir, result_type=IndexResult)
    return task


class RecordingExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[TaskDefinition, str]] = []
        self.counts: dict[str, int] = {}

    def execute_prompt(self, task: TaskDefinition, prompt: str) -> IndexResult:
        self.calls.append((task, prompt))
        count = self.counts.get(task.id, 0) + 1
        self.counts[task.id] = count
        return make_output(f"{task.id}-{count}")

    def model_for(self, task: TaskDefinition) -> str:
        return "fake-model"


class TaskDependencyTests(unittest.TestCase):
    def test_execution_plan_is_ordered_and_does_not_auto_resolve(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            task_a = make_task(root, "a")
            task_b = make_task(root, "b", requires=["a"])

            validate_execution_plan([task_a, task_b])
            for plan in ([task_b], [task_b, task_a]):
                with self.subTest(plan=[task.id for task in plan]), self.assertRaisesRegex(
                    InvalidExecutionPlanError,
                    "Task 'b'.*task 'a'.*earlier",
                ):
                    validate_execution_plan(plan)

    def test_execution_plan_supports_multiple_dependencies_and_repeated_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            task_a = make_task(root, "a")
            task_b = make_task(root, "b")
            task_c = make_task(root, "c", requires=["a", "b"])

            validate_execution_plan([task_b, task_a, task_a, task_c])

    def test_independent_tasks_remain_reorderable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            task_a = make_task(root, "a")
            task_b = make_task(root, "b")

            validate_execution_plan([task_a, task_b])
            validate_execution_plan([task_b, task_a])

    def test_context_exposes_read_only_current_run_results(self) -> None:
        transcript = make_transcript()
        context = ProcessingContext(transcript)

        self.assertIs(context.transcript, transcript)
        self.assertEqual(dict(context.results), {})
        result = make_result("a", "first")
        context.record(result)
        self.assertIs(context.results["a"], result)
        with self.assertRaises(TypeError):
            context.results["b"] = result  # type: ignore[index]

        another_context = ProcessingContext(transcript)
        self.assertEqual(dict(another_context.results), {})

    def test_prompt_variables_use_only_declared_in_memory_dependencies(self) -> None:
        transcript = make_transcript()
        context = ProcessingContext(transcript)
        result_a = make_result("a", "from-memory")
        result_extra = make_result("extra", "must-not-appear")
        context.record(result_a)
        context.record(result_extra)

        with tempfile.TemporaryDirectory() as temporary_dir:
            task = make_task(Path(temporary_dir), "b", requires=["a"])
            variables = build_prompt_variables(
                task,
                context,
                transcript.as_timestamped_text(),
            )

        self.assertEqual(
            variables["TRANSCRIPT"],
            "[00:00:00 - 00:00:10] Texto do episódio.",
        )
        self.assertIn("from-memory", variables["A"])
        self.assertNotIn("EXTRA", variables)
        self.assertIs(context.results["a"], result_a)

    def test_prompt_variables_preserve_hyphens_and_convert_settings(self) -> None:
        transcript = make_transcript()
        context = ProcessingContext(transcript)
        context.record(make_result("source-task", "source"))
        with tempfile.TemporaryDirectory() as temporary_dir:
            task = make_task(
                Path(temporary_dir),
                "dependent",
                requires=["source-task"],
                settings={
                    "text": "value",
                    "enabled": True,
                    "count": 5,
                    "ratio": 1.5,
                },
            )
            variables = build_prompt_variables(
                task,
                context,
                transcript.as_timestamped_text(),
            )

        self.assertIn("SOURCE-TASK", variables)
        self.assertEqual(variables["TEXT"], "value")
        self.assertEqual(variables["ENABLED"], "true")
        self.assertEqual(variables["COUNT"], "5")
        self.assertEqual(variables["RATIO"], "1.5")

    def test_dependency_uses_same_markdown_as_output_rendering(self) -> None:
        transcript = make_transcript()
        context = ProcessingContext(transcript)
        context.record(make_result("a", "shared-renderer"))
        with tempfile.TemporaryDirectory() as temporary_dir:
            task = make_task(Path(temporary_dir), "b", requires=["a"])
            variables = build_prompt_variables(
                task,
                context,
                transcript.as_timestamped_text(),
            )

        self.assertEqual(
            variables["A"],
            "# Indexação do episódio\n\n"
            "## 00:00:00 - 00:00:10 — shared-renderer\n\n"
            "Resumo shared-renderer\n",
        )

    def test_dependency_rendering_supports_custom_and_generic_renderers(self) -> None:
        transcript = make_transcript()
        context = ProcessingContext(transcript)
        context.record(make_result("a", "custom", schema_id="custom_result"))
        with tempfile.TemporaryDirectory() as temporary_dir:
            task = make_task(Path(temporary_dir), "b", requires=["a"])
            custom = build_prompt_variables(
                task,
                context,
                transcript.as_timestamped_text(),
                renderers={"custom_result": lambda output: f"custom: {output.blocks[0].topic}"},
            )
            generic = build_prompt_variables(
                task,
                context,
                transcript.as_timestamped_text(),
            )

        self.assertEqual(custom["A"], "custom: custom")
        self.assertIn('"topic": "custom"', generic["A"])

    def test_runner_renders_dependency_prompt_and_keeps_latest_repeated_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            task_a = make_task(root, "a")
            task_b = make_task(
                root,
                "b",
                requires=["a"],
                settings={"count": 5},
                prompt="{{TRANSCRIPT}}\n{{A}}\ncount={{COUNT}}",
            )
            executor = RecordingExecutor()

            results = ProcessingRunner(executors={"fake": executor}).run(
                make_transcript(),
                (task for task in [task_a, task_a, task_b]),
            )

        self.assertEqual([result.task_id for result in results], ["a", "a", "b"])
        self.assertIn("a-2", executor.calls[2][1])
        self.assertNotIn("a-1", executor.calls[2][1])
        self.assertIn("count=5", executor.calls[2][1])

    def test_persistence_failure_stops_run_before_context_exposure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            task_a = make_task(root, "a")
            task_b = make_task(
                root,
                "b",
                requires=["a"],
                prompt="{{TRANSCRIPT}}\n{{A}}",
            )
            executor = RecordingExecutor()

            with self.assertRaisesRegex(TaskExecutionError, "disk unavailable"):
                ProcessingRunner(executors={"fake": executor}).run(
                    make_transcript(),
                    [task_a, task_b],
                    on_result=lambda _task, _result: (_ for _ in ()).throw(
                        OSError("disk unavailable")
                    ),
                )

        self.assertEqual([task.id for task, _prompt in executor.calls], ["a"])

    def test_legacy_processor_bridge_rejects_dependency_aware_task(self) -> None:
        class LegacyProcessor:
            def process(self, task, transcript, formatted_transcript):
                return make_output("legacy")

            def model_for(self, task):
                return None

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            task_a = make_task(root, "a")
            task_b = make_task(root, "b", requires=["a"])

            with self.assertRaisesRegex(
                TaskExecutionError,
                "rendered-prompt executor",
            ):
                ProcessingRunner({"fake": LegacyProcessor()}).run(
                    make_transcript(),
                    [task_a, task_b],
                )

    def test_pipeline_rejects_invalid_plan_before_transcription(self) -> None:
        class UnexpectedTranscriber:
            def transcribe(self, audio_path):
                raise AssertionError("transcription must not start")

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            audio_path = root / "episode.mp3"
            audio_path.write_bytes(b"audio")
            task_a = make_task(root, "a")
            task_b = make_task(root, "b", requires=["a"])
            registry = TaskRegistry([task_a, task_b])
            pipeline = PodcastPipeline.for_processing(
                runner=ProcessingRunner(executors={"fake": RecordingExecutor()}),
                task_registry=registry,
                output_dir=root / "outputs",
                transcriber=UnexpectedTranscriber(),
            )

            with self.assertRaises(InvalidExecutionPlanError):
                pipeline.run_tasks(audio_path, [task_b, task_a])

    def test_process_transcript_does_not_use_an_old_dependency_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            transcript_path = root / "transcript.json"
            transcript_path.write_text(
                make_transcript().model_dump_json(indent=2),
                encoding="utf-8",
            )
            old_output = root / "processing" / "a"
            old_output.mkdir(parents=True)
            (old_output / "result.json").write_text(
                make_output("old").model_dump_json(indent=2),
                encoding="utf-8",
            )
            task_a = make_task(root, "a")
            task_b = make_task(root, "b", requires=["a"])
            registry = TaskRegistry([task_a, task_b])
            executor = RecordingExecutor()
            pipeline = PodcastPipeline.for_processing(
                runner=ProcessingRunner(executors={"fake": executor}),
                task_registry=registry,
                output_dir=root,
            )

            with self.assertRaises(InvalidExecutionPlanError):
                pipeline.process_transcript_file(transcript_path, [task_b])

        self.assertEqual(executor.calls, [])

    def test_pipeline_persists_dependency_before_later_task_failure(self) -> None:
        class FailingExecutor(RecordingExecutor):
            def execute_prompt(self, task, prompt):
                if task.id == "b":
                    self.calls.append((task, prompt))
                    raise RuntimeError("provider failed")
                return super().execute_prompt(task, prompt)

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            transcript_path = root / "transcript.json"
            transcript_path.write_text(
                make_transcript().model_dump_json(indent=2),
                encoding="utf-8",
            )
            task_a = make_task(root, "a")
            task_b = make_task(
                root,
                "b",
                requires=["a"],
                prompt="{{TRANSCRIPT}}\n{{A}}",
            )
            registry = TaskRegistry([task_a, task_b])
            executor = FailingExecutor()
            pipeline = PodcastPipeline.for_processing(
                runner=ProcessingRunner(executors={"fake": executor}),
                task_registry=registry,
                output_dir=root,
            )

            with self.assertRaisesRegex(TaskExecutionError, "provider failed"):
                pipeline.process_transcript_file(
                    transcript_path,
                    [task_a, task_b],
                )

            self.assertTrue((root / "processing" / "a" / "result.json").is_file())
            self.assertFalse((root / "processing" / "b").exists())


if __name__ == "__main__":
    unittest.main()
