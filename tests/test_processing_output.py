import json
import tempfile
import unittest
from pathlib import Path

from pydantic import BaseModel

from src.models import IndexBlock, IndexResult
from src.processing import ProcessingResult, TaskRegistry
from src.processing.output import ProcessingOutputWriter
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


if __name__ == "__main__":
    unittest.main()
