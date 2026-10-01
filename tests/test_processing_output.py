import json
import tempfile
import unittest
from pathlib import Path

from pydantic import BaseModel

from src.models import IndexBlock, IndexResult
from src.processing import ProcessingResult, TaskDefinition, TaskRegistry
from src.processing.output import ProcessingOutputWriter, render_index_markdown
from src.tasks.cortes.models import CutSelection, CutsResult
from src.tasks.shorts.models import ShortMoment, ShortsResult


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


class ProcessingOutputTests(unittest.TestCase):
    def test_processing_result_serializes_metadata_and_concrete_payload(self) -> None:
        result = ProcessingResult[BaseModel](
            task_id="indexacao",
            task_name="Indexação do episódio",
            executor="openai",
            model="gpt-5-mini",
            status="completed",
            schema_id="index_result",
            output=make_index(),
        )

        payload = result.model_dump(mode="json")

        self.assertEqual(payload["task_id"], "indexacao")
        self.assertEqual(payload["model"], "gpt-5-mini")
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["output"]["blocks"][0]["topic"], "Tema de teste")

    def test_writes_separate_payload_outputs_and_legacy_index_aliases(self) -> None:
        registry = TaskRegistry.load_builtin()
        index_task = registry.get("indexacao")
        shorts_task = registry.get("shorts")
        index_result = ProcessingResult(
            task_id=index_task.id,
            task_name=index_task.name,
            executor=index_task.executor,
            model="gpt-5-mini",
            schema_id=index_task.schema_id,
            output=make_index(),
        )
        shorts_result = ProcessingResult(
            task_id=shorts_task.id,
            task_name=shorts_task.name,
            executor=shorts_task.executor,
            model="gpt-5-mini",
            schema_id=shorts_task.schema_id,
            output=ShortsResult(
                moments=[
                    ShortMoment(
                        start_seconds=65,
                        end_seconds=80,
                        title="Momento curto",
                        summary="Resumo curto",
                        central_idea="Ideia",
                    )
                ]
            ),
        )

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            writer = ProcessingOutputWriter()
            writer.write(root, index_task, index_result)
            writer.write(root, shorts_task, shorts_result)

            index_payload = json.loads(
                (root / "processing" / "indexacao" / "result.json").read_text(
                    encoding="utf-8"
                )
            )
            shorts_payload = json.loads(
                (root / "processing" / "shorts" / "result.json").read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(index_payload, make_index().model_dump(mode="json"))
            self.assertNotIn("task_id", index_payload)
            self.assertEqual(shorts_payload["moments"][0]["title"], "Momento curto")
            self.assertFalse((root / "shorts.json").exists())
            self.assertEqual(
                json.loads((root / "index.json").read_text(encoding="utf-8")),
                index_payload,
            )
            self.assertEqual(
                (root / "index.md").read_text(encoding="utf-8"),
                (root / "processing" / "indexacao" / "result.md").read_text(
                    encoding="utf-8"
                ),
            )
            self.assertIn(
                "## 00:01:05 - 00:01:20 — Momento curto",
                (root / "processing" / "shorts" / "result.md").read_text(
                    encoding="utf-8"
                ),
            )

    def test_preserves_generic_and_custom_rendering_and_historical_import(self) -> None:
        class ExampleResult(BaseModel):
            value: str

        task = TaskDefinition.model_validate(
            {
                "id": "example",
                "name": "Example",
                "prompt": "prompt.md",
                "executor": "fake",
                "schema": "example_result",
            }
        )
        result = ProcessingResult[BaseModel](
            task_id=task.id,
            task_name=task.name,
            executor=task.executor,
            schema_id=task.schema_id,
            output=ExampleResult(value="content"),
        )

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            ProcessingOutputWriter().write(root / "generic", task, result)
            ProcessingOutputWriter(
                {"example_result": lambda output: f"custom: {output.value}\n"}
            ).write(root / "custom", task, result)

            self.assertIn(
                '"value": "content"',
                (root / "generic" / "processing" / "example" / "result.md").read_text(
                    encoding="utf-8"
                ),
            )
            self.assertEqual(
                (root / "custom" / "processing" / "example" / "result.md").read_text(
                    encoding="utf-8"
                ),
                "custom: content\n",
            )

        self.assertEqual(
            render_index_markdown(make_index()),
            "# Indexação do episódio\n\n"
            "## 00:00:05 - 00:00:18 — Tema de teste\n\n"
            "Resumo do assunto abordado.\n",
        )

    def test_writes_cortes_payload_and_markdown_with_calculated_duration(self) -> None:
        task = TaskRegistry.load_builtin().get("cortes")
        result = ProcessingResult[BaseModel](
            task_id=task.id,
            task_name=task.name,
            executor=task.executor,
            schema_id=task.schema_id,
            output=CutsResult(
                cuts=[
                    CutSelection(
                        start_seconds=300,
                        end_seconds=900,
                        theme="Tema do corte",
                        summary="Resumo do corte",
                        title="TÍTULO DO CORTE",
                        thumbnail_text="FRASE DA THUMBNAIL",
                    )
                ],
                limitations=["Apenas uma opção válida."],
            ),
        )

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            ProcessingOutputWriter().write(root, task, result)

            payload = json.loads(
                (root / "processing" / "cortes" / "result.json").read_text(
                    encoding="utf-8"
                )
            )
            markdown = (
                root / "processing" / "cortes" / "result.md"
            ).read_text(encoding="utf-8")

            self.assertEqual(payload["cuts"][0]["start_seconds"], 300)
            self.assertNotIn("duration", payload["cuts"][0])
            self.assertIn("**Início:** 00:05:00", markdown)
            self.assertIn("**Fim:** 00:15:00", markdown)
            self.assertIn("**Duração:** 00:10:00", markdown)
            self.assertIn("## Limitações", markdown)
            self.assertFalse((root / "cuts.json").exists())
            self.assertFalse((root / "cuts.md").exists())


if __name__ == "__main__":
    unittest.main()
