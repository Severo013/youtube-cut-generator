"""Small, explicit discovery and validation registry for transcript tasks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from src.models import IndexResult
from src.processing.base import ProcessingConfigurationError
from src.processing.models import TaskDefinition
from src.processing.validation import validate_execution_plan
from src.tasks.cortes.models import CutsResult
from src.tasks.shorts.models import ShortsResult


DEFAULT_TASK_SCHEMAS: Mapping[str, type[BaseModel]] = {
    "cuts_result": CutsResult,
    "index_result": IndexResult,
    "shorts_result": ShortsResult,
}


class TaskRegistryError(ProcessingConfigurationError):
    """Base error for task discovery and lookup failures."""


class TaskNotFoundError(TaskRegistryError):
    """Raised when a requested task id is not registered."""


class DuplicateTaskIdError(TaskRegistryError):
    """Raised when two task resources declare the same id."""


class InvalidTaskDefinitionError(TaskRegistryError):
    """Raised when task configuration or resources are invalid."""


class TaskRegistry:
    """Discover validated task definitions from explicit resource directories."""

    def __init__(self, tasks: list[TaskDefinition]) -> None:
        self._tasks: dict[str, TaskDefinition] = {}
        for task in tasks:
            if task.id in self._tasks:
                raise DuplicateTaskIdError(f"Id de task duplicado: {task.id}")
            self._tasks[task.id] = task
        self._validate_declared_dependencies()

    def _validate_declared_dependencies(self) -> None:
        known_task_ids = self._tasks.keys()
        for task in self._tasks.values():
            for dependency_id in task.requires:
                if dependency_id not in known_task_ids:
                    raise InvalidTaskDefinitionError(
                        f"Task {task.id!r} requires unknown task "
                        f"{dependency_id!r}."
                    )

    @classmethod
    def load_builtin(cls) -> "TaskRegistry":
        """Load tasks shipped as resources of the ``src.tasks`` package."""

        return cls.from_package("src.tasks", schemas=DEFAULT_TASK_SCHEMAS)

    @classmethod
    def from_package(
        cls,
        package: str,
        *,
        schemas: Mapping[str, type[BaseModel]],
    ) -> "TaskRegistry":
        try:
            root = resources.files(package)
        except (ModuleNotFoundError, TypeError) as error:
            raise InvalidTaskDefinitionError(
                f"Pacote de tasks não encontrado: {package}"
            ) from error
        return cls._from_resource_root(root, schemas=schemas)

    @classmethod
    def from_directory(
        cls,
        directory: Path,
        *,
        schemas: Mapping[str, type[BaseModel]],
    ) -> "TaskRegistry":
        root = Path(directory)
        if not root.is_dir():
            raise InvalidTaskDefinitionError(
                f"Diretório de tasks não encontrado: {root}"
            )
        return cls._from_resource_root(root, schemas=schemas)

    @classmethod
    def _from_resource_root(
        cls,
        root: Traversable,
        *,
        schemas: Mapping[str, type[BaseModel]],
    ) -> "TaskRegistry":
        tasks: list[TaskDefinition] = []
        try:
            candidates = sorted(root.iterdir(), key=lambda item: item.name)
        except OSError as error:
            raise InvalidTaskDefinitionError(
                f"Não foi possível listar as tasks em {root}."
            ) from error

        for task_directory in candidates:
            if not task_directory.is_dir() or task_directory.name.startswith("_"):
                continue
            config_resource = task_directory.joinpath("task.yaml")
            if not config_resource.is_file():
                raise InvalidTaskDefinitionError(
                    f"Task {task_directory.name!r} não possui task.yaml."
                )
            tasks.append(
                cls._load_definition(
                    task_directory,
                    config_resource,
                    schemas=schemas,
                )
            )
        return cls(tasks)

    @staticmethod
    def _load_definition(
        task_directory: Traversable,
        config_resource: Traversable,
        *,
        schemas: Mapping[str, type[BaseModel]],
    ) -> TaskDefinition:
        try:
            raw: Any = yaml.safe_load(config_resource.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as error:
            raise InvalidTaskDefinitionError(
                f"Não foi possível carregar {config_resource}."
            ) from error

        if not isinstance(raw, dict):
            raise InvalidTaskDefinitionError(
                f"Definição inválida em {config_resource}: esperado um objeto YAML."
            )

        schema_id = raw.get("schema")
        result_type = schemas.get(schema_id) if isinstance(schema_id, str) else None
        if result_type is None:
            raise InvalidTaskDefinitionError(
                f"Schema desconhecido em {config_resource}: {schema_id!r}."
            )

        try:
            task = TaskDefinition.model_validate(raw)
        except ValidationError as error:
            raise InvalidTaskDefinitionError(
                f"Definição de task inválida em {config_resource}: {error}"
            ) from error

        prompt_resource = task_directory.joinpath(task.prompt)
        if not prompt_resource.is_file():
            raise InvalidTaskDefinitionError(
                f"Prompt da task {task.id!r} não encontrado: {task.prompt}"
            )
        task.bind_runtime(
            resource_directory=task_directory,
            result_type=result_type,
        )
        return task

    def get(self, task_id: str) -> TaskDefinition:
        try:
            return self._tasks[task_id]
        except KeyError as error:
            available = ", ".join(sorted(self._tasks)) or "none"
            raise TaskNotFoundError(
                f"Unknown task: {task_id}\nAvailable tasks: {available}"
            ) from error

    def resolve(self, task_ids: Sequence[str]) -> list[TaskDefinition]:
        """Resolve task ids in the requested order before execution begins."""

        tasks = [self.get(task_id) for task_id in task_ids]
        validate_execution_plan(tasks)
        return tasks

    def all(self) -> tuple[TaskDefinition, ...]:
        return tuple(self._tasks.values())
