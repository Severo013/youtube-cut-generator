import tempfile
import unittest
from importlib import resources
from pathlib import Path

from src.models import IndexResult
from src.processing import (
    DuplicateTaskIdError,
    InvalidTaskDefinitionError,
    TaskNotFoundError,
    TaskRegistry,
)


SCHEMAS = {"index_result": IndexResult}


def write_task(
    root: Path,
    directory_name: str,
    *,
    task_id: str,
    prompt: str = "prompt.md",
    include_name: bool = True,
) -> None:
    task_dir = root / directory_name
    task_dir.mkdir()
    fields = [f"id: {task_id}"]
    if include_name:
        fields.append("name: Teste")
    fields.extend(
        [
            f"prompt: {prompt}",
            "executor: openai",
            "schema: index_result",
        ]
    )
    (task_dir / "task.yaml").write_text("\n".join(fields) + "\n", encoding="utf-8")
    if prompt == "prompt.md":
        (task_dir / prompt).write_text("{{TRANSCRIPT}}", encoding="utf-8")


class TaskRegistryTests(unittest.TestCase):
    def test_loads_builtin_indexing_task_and_packaged_resources(self) -> None:
        registry = TaskRegistry.load_builtin()

        task = registry.get("indexacao")

        self.assertEqual(task.name, "Indexação")
        self.assertEqual(task.executor, "openai")
        self.assertEqual(task.schema_id, "index_result")
        self.assertIs(task.result_type, IndexResult)
        self.assertEqual(task.prompt, "prompt.md")
        self.assertIn("{{TRANSCRIPT}}", task.load_prompt())

        package_root = resources.files("src.tasks.indexacao")
        self.assertTrue(package_root.joinpath("task.yaml").is_file())
        self.assertTrue(package_root.joinpath("prompt.md").is_file())

    def test_rejects_unknown_task(self) -> None:
        registry = TaskRegistry.load_builtin()

        with self.assertRaisesRegex(TaskNotFoundError, "não encontrada"):
            registry.get("shorts")

    def test_rejects_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            write_task(root, "first", task_id="duplicada")
            write_task(root, "second", task_id="duplicada")

            with self.assertRaisesRegex(DuplicateTaskIdError, "duplicado"):
                TaskRegistry.from_directory(root, schemas=SCHEMAS)

    def test_rejects_missing_required_field(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            write_task(root, "invalid", task_id="invalid", include_name=False)

            with self.assertRaisesRegex(
                InvalidTaskDefinitionError,
                "Definição de task inválida",
            ):
                TaskRegistry.from_directory(root, schemas=SCHEMAS)

    def test_rejects_missing_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            write_task(root, "invalid", task_id="invalid", prompt="missing.md")

            with self.assertRaisesRegex(InvalidTaskDefinitionError, "não encontrado"):
                TaskRegistry.from_directory(root, schemas=SCHEMAS)

    def test_rejects_unknown_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            write_task(root, "invalid", task_id="invalid")
            config_path = root / "invalid" / "task.yaml"
            config_path.write_text(
                config_path.read_text(encoding="utf-8").replace(
                    "index_result",
                    "unknown_result",
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(InvalidTaskDefinitionError, "Schema desconhecido"):
                TaskRegistry.from_directory(root, schemas=SCHEMAS)

    def test_rejects_task_directory_without_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "invalid").mkdir()

            with self.assertRaisesRegex(InvalidTaskDefinitionError, "task.yaml"):
                TaskRegistry.from_directory(root, schemas=SCHEMAS)


if __name__ == "__main__":
    unittest.main()
