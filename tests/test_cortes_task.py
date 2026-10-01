import tempfile
import unittest
from importlib import resources
from pathlib import Path

from src.models import IndexBlock, IndexResult, Transcript, TranscriptSegment
from src.pipeline import PodcastPipeline
from src.processing import (
    InvalidExecutionPlanError,
    ProcessingRunner,
    TaskExecutionError,
    TaskRegistry,
)
from src.tasks.cortes.models import CutsResult


def make_index() -> IndexResult:
    return IndexResult(
        blocks=[
            IndexBlock(
                start_seconds=0,
                end_seconds=300,
                start_timestamp="00:00:00",
                end_timestamp="00:05:00",
                topic="Primeiro tema em memória",
                summary="Primeiro resumo.",
            ),
            IndexBlock(
                start_seconds=300,
                end_seconds=600,
                start_timestamp="00:05:00",
                end_timestamp="00:10:00",
                topic="Segundo tema",
                summary="Segundo resumo.",
            ),
        ]
    )


def make_transcript() -> Transcript:
    return Transcript(
        source="episode.mp3",
        duration_seconds=600,
        language="pt",
        full_text="Fala original do episódio.",
        segments=[
            TranscriptSegment(
                start_seconds=0,
                end_seconds=600,
                text="Fala original do episódio.",
            )
        ],
    )


def cuts_payload() -> dict[str, object]:
    return {
        "cuts": [
            {
                "start_seconds": 0,
                "end_seconds": 600,
                "theme": "Tema combinado",
                "summary": "Os dois blocos formam uma discussão completa.",
                "title": "UMA DISCUSSÃO COMPLETA",
                "thumbnail_text": "DOIS TEMAS CONECTADOS",
            }
        ],
        "limitations": ["O episódio só contém uma opção válida."],
    }


class CutsExecutor:
    def __init__(self, failing_task: str | None = None) -> None:
        self.failing_task = failing_task
        self.calls: list[tuple[str, str]] = []

    def execute_prompt(self, task, prompt):
        self.calls.append((task.id, prompt))
        if task.id == self.failing_task:
            raise RuntimeError(f"{task.id} failed")
        if task.id == "indexacao":
            return make_index()
        return CutsResult.model_validate(cuts_payload())

    def model_for(self, task):
        return "fake-model"


class CutsTaskTests(unittest.TestCase):
    def test_registry_loads_cortes_definition_schema_and_resources(self) -> None:
        registry = TaskRegistry.load_builtin()
        task = registry.get("cortes")

        self.assertEqual(task.name, "Seleção de Cortes")
        self.assertEqual(task.executor, "openai")
        self.assertEqual(task.schema_id, "cuts_result")
        self.assertIs(task.result_type, CutsResult)
        self.assertEqual(task.requires, ["indexacao"])
        self.assertEqual(
            task.settings,
            {
                "target_count": 5,
                "min_duration_minutes": 5,
                "max_duration_minutes": 20,
            },
        )
        prompt = task.load_prompt()
        for placeholder in (
            "{{TARGET_COUNT}}",
            "{{MIN_DURATION_MINUTES}}",
            "{{MAX_DURATION_MINUTES}}",
            "{{INDEXACAO}}",
            "{{TRANSCRIPT}}",
        ):
            self.assertIn(placeholder, prompt)

        package_root = resources.files("src.tasks.cortes")
        self.assertTrue(package_root.joinpath("task.yaml").is_file())
        self.assertTrue(package_root.joinpath("prompt.md").is_file())

    def test_indexacao_then_cortes_uses_generic_in_memory_prompt_context(self) -> None:
        registry = TaskRegistry.load_builtin()
        executor = CutsExecutor()

        results = ProcessingRunner(executors={"openai": executor}).run(
            make_transcript(),
            registry.resolve(("indexacao", "cortes")),
        )

        self.assertEqual([result.task_id for result in results], ["indexacao", "cortes"])
        self.assertIsInstance(results[1].output, CutsResult)
        cuts_prompt = executor.calls[1][1]
        self.assertIn("# Indexação do episódio", cuts_prompt)
        self.assertIn("Primeiro tema em memória", cuts_prompt)
        self.assertIn(
            "[00:00:00 - 00:10:00] Fala original do episódio.",
            cuts_prompt,
        )
        self.assertIn("até 5 cortes", cuts_prompt)
        self.assertIn("entre **5 e 20 minutos**", cuts_prompt)
        self.assertNotIn("{{", cuts_prompt)

    def test_invalid_cortes_plans_are_rejected(self) -> None:
        registry = TaskRegistry.load_builtin()

        for task_ids in (("cortes",), ("cortes", "indexacao")):
            with self.subTest(task_ids=task_ids), self.assertRaises(
                InvalidExecutionPlanError
            ):
                registry.resolve(task_ids)

    def test_indexing_failure_prevents_cortes_execution(self) -> None:
        registry = TaskRegistry.load_builtin()
        executor = CutsExecutor(failing_task="indexacao")

        with self.assertRaisesRegex(TaskExecutionError, "indexacao failed"):
            ProcessingRunner(executors={"openai": executor}).run(
                make_transcript(),
                registry.resolve(("indexacao", "cortes")),
            )

        self.assertEqual([task_id for task_id, _prompt in executor.calls], ["indexacao"])

    def test_cortes_failure_preserves_persisted_indexing_output(self) -> None:
        registry = TaskRegistry.load_builtin()
        executor = CutsExecutor(failing_task="cortes")

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            transcript_path = root / "transcript.json"
            transcript_path.write_text(
                make_transcript().model_dump_json(indent=2),
                encoding="utf-8",
            )
            pipeline = PodcastPipeline.for_processing(
                runner=ProcessingRunner(executors={"openai": executor}),
                task_registry=registry,
                output_dir=root,
            )

            with self.assertRaisesRegex(TaskExecutionError, "cortes failed"):
                pipeline.process_transcript_file(
                    transcript_path,
                    registry.resolve(("indexacao", "cortes")),
                )

            self.assertTrue(
                (root / "processing" / "indexacao" / "result.json").is_file()
            )
            self.assertTrue((root / "index.json").is_file())
            self.assertFalse((root / "processing" / "cortes").exists())


if __name__ == "__main__":
    unittest.main()
