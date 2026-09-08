import tempfile
import unittest
from importlib import resources
from pathlib import Path

from src.models import IndexResult
from src.processing import (
    DuplicateTaskIdError,
    InvalidExecutionPlanError,
    InvalidTaskDefinitionError,
    TaskNotFoundError,
    TaskRegistry,
    TaskDefinition,
)
from src.tasks.shorts.models import ShortsResult


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

        self.assertEqual(task.name, "Indexação do episódio")
        self.assertEqual(task.executor, "openai")
        self.assertEqual(task.schema_id, "index_result")
        self.assertIs(task.result_type, IndexResult)
        self.assertEqual(task.prompt, "prompt.md")
        self.assertEqual(task.requires, [])
        self.assertEqual(task.settings, {})
        self.assertIn("{{TRANSCRIPT}}", task.load_prompt())

        package_root = resources.files("src.tasks.indexacao")
        self.assertTrue(package_root.joinpath("task.yaml").is_file())
        self.assertTrue(package_root.joinpath("prompt.md").is_file())

        shorts = registry.get("shorts")
        self.assertEqual(shorts.name, "Seleção de Shorts")
        self.assertEqual(shorts.schema_id, "shorts_result")
        self.assertIs(shorts.result_type, ShortsResult)
        shorts_root = resources.files("src.tasks.shorts")
        self.assertTrue(shorts_root.joinpath("task.yaml").is_file())
        self.assertTrue(shorts_root.joinpath("prompt.md").is_file())

    def test_rejects_unknown_task(self) -> None:
        registry = TaskRegistry.load_builtin()

        with self.assertRaisesRegex(
            TaskNotFoundError,
            "Unknown task: xyz\nAvailable tasks: indexacao, shorts",
        ):
            registry.get("xyz")

    def test_resolves_multiple_tasks_in_requested_order(self) -> None:
        registry = TaskRegistry.load_builtin()

        tasks = registry.resolve(("shorts", "indexacao"))

        self.assertEqual([task.id for task in tasks], ["shorts", "indexacao"])

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

    def test_accepts_dependencies_and_primitive_settings(self) -> None:
        definition = TaskDefinition.model_validate(
            {
                "id": "dependent",
                "name": "Dependent",
                "prompt": "prompt.md",
                "executor": "fake",
                "schema": "index_result",
                "requires": ["source-task"],
                "settings": {
                    "text": "value",
                    "enabled": True,
                    "count": 5,
                    "ratio": 1.5,
                },
            }
        )

        self.assertEqual(definition.requires, ["source-task"])
        self.assertEqual(
            definition.settings,
            {"text": "value", "enabled": True, "count": 5, "ratio": 1.5},
        )

    def test_rejects_duplicate_self_and_invalid_dependencies(self) -> None:
        base = {
            "id": "dependent",
            "name": "Dependent",
            "prompt": "prompt.md",
            "executor": "fake",
            "schema": "index_result",
        }
        for requires, message in (
            (["source", "source"], "duplicate"),
            (["dependent"], "itself"),
            (["Invalid Id"], "invalid dependency"),
        ):
            with self.subTest(requires=requires), self.assertRaisesRegex(
                ValueError,
                message,
            ):
                TaskDefinition.model_validate({**base, "requires": requires})

    def test_rejects_nested_null_and_non_finite_settings(self) -> None:
        base = {
            "id": "test",
            "name": "Test",
            "prompt": "prompt.md",
            "executor": "fake",
            "schema": "index_result",
        }
        for value in ({"nested": True}, [1], None, float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError,
                "must be a string",
            ):
                TaskDefinition.model_validate(
                    {**base, "settings": {"invalid": value}}
                )

    def test_rejects_prompt_variable_name_collisions(self) -> None:
        base = {
            "id": "dependent",
            "name": "Dependent",
            "prompt": "prompt.md",
            "executor": "fake",
            "schema": "index_result",
        }
        for fields in (
            {"settings": {"transcript": "value"}},
            {"settings": {"size": 1, "SIZE": 2}},
            {"requires": ["source"], "settings": {"SOURCE": "value"}},
        ):
            with self.subTest(fields=fields), self.assertRaisesRegex(
                ValueError,
                "collides",
            ):
                TaskDefinition.model_validate({**base, **fields})

    def test_registry_accepts_known_dependency_and_rejects_unknown_one(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            write_task(root, "source", task_id="source")
            write_task(root, "dependent", task_id="dependent")
            config_path = root / "dependent" / "task.yaml"
            config_path.write_text(
                config_path.read_text(encoding="utf-8")
                + "requires:\n  - source\n",
                encoding="utf-8",
            )
            registry = TaskRegistry.from_directory(root, schemas=SCHEMAS)

            self.assertEqual(registry.get("dependent").requires, ["source"])
            self.assertEqual(
                [task.id for task in registry.resolve(("source", "dependent"))],
                ["source", "dependent"],
            )
            with self.assertRaisesRegex(
                InvalidExecutionPlanError,
                "selected and executed earlier",
            ) as invalid_plan:
                registry.resolve(("dependent",))
            self.assertNotIsInstance(
                invalid_plan.exception,
                InvalidTaskDefinitionError,
            )

            config_path.write_text(
                config_path.read_text(encoding="utf-8").replace("source\n", "missing\n"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                InvalidTaskDefinitionError,
                "requires unknown task.*missing",
            ):
                TaskRegistry.from_directory(root, schemas=SCHEMAS)


if __name__ == "__main__":
    unittest.main()
