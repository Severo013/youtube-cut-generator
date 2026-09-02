"""Data models shared by transcript processing components."""

from __future__ import annotations

from importlib.resources.abc import Traversable
from pathlib import PurePosixPath
from typing import Generic, Literal, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SerializeAsAny,
    field_validator,
)


ResultT = TypeVar("ResultT", bound=BaseModel)


class TaskDefinition(BaseModel):
    """Configuration-driven definition of one transcript processing task."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    name: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    executor: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    schema_id: str = Field(
        alias="schema",
        min_length=1,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
    )
    model: str | None = None

    _resource_directory: Traversable | None = PrivateAttr(default=None)
    _result_type: type[BaseModel] | None = PrivateAttr(default=None)

    @field_validator("name", "prompt", "model")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("value cannot be blank")
        return value

    @field_validator("prompt")
    @classmethod
    def validate_relative_prompt(cls, value: str) -> str:
        path = PurePosixPath(value.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("prompt must be a relative path inside the task directory")
        return path.as_posix()

    def bind_runtime(
        self,
        *,
        resource_directory: Traversable,
        result_type: type[BaseModel],
    ) -> None:
        """Attach resources resolved explicitly by ``TaskRegistry``."""

        self._resource_directory = resource_directory
        self._result_type = result_type

    @property
    def result_type(self) -> type[BaseModel]:
        if self._result_type is None:
            raise RuntimeError(f"Task {self.id!r} has no resolved result schema")
        return self._result_type

    @property
    def prompt_resource(self) -> Traversable:
        if self._resource_directory is None:
            raise RuntimeError(f"Task {self.id!r} has no bound resource directory")
        return self._resource_directory.joinpath(self.prompt)

    def load_prompt(self) -> str:
        """Load this task's external UTF-8 prompt resource."""

        return self.prompt_resource.read_text(encoding="utf-8")


class ProcessingResult(BaseModel, Generic[ResultT]):
    """Common execution envelope around a task-specific structured result."""

    task_id: str
    task_name: str
    executor: str
    model: str | None = None
    status: Literal["completed"] = "completed"
    schema_id: str | None = None
    output: SerializeAsAny[ResultT]
