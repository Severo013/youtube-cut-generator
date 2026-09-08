"""Data models shared by transcript processing components."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from importlib.resources.abc import Traversable
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Generic, Literal, TypeAlias, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SerializeAsAny,
    field_validator,
    model_validator,
)

from src.models import Transcript


ResultT = TypeVar("ResultT", bound=BaseModel)
SettingValue: TypeAlias = str | bool | int | float
TASK_ID_PATTERN = r"^[a-z0-9][a-z0-9_-]*$"
SETTING_KEY_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]*$"
RESERVED_PROMPT_VARIABLES = frozenset({"TRANSCRIPT"})


class TaskDefinition(BaseModel):
    """Configuration-driven definition of one transcript processing task."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str = Field(min_length=1, pattern=TASK_ID_PATTERN)
    name: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    executor: str = Field(min_length=1, pattern=TASK_ID_PATTERN)
    schema_id: str = Field(
        alias="schema",
        min_length=1,
        pattern=TASK_ID_PATTERN,
    )
    model: str | None = None
    requires: list[str] = Field(default_factory=list)
    settings: dict[str, SettingValue] = Field(default_factory=dict)

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

    @field_validator("requires")
    @classmethod
    def validate_dependencies(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        for dependency_id in value:
            if re.fullmatch(TASK_ID_PATTERN, dependency_id) is None:
                raise ValueError(f"invalid dependency task id: {dependency_id!r}")
            if dependency_id in seen:
                raise ValueError(f"duplicate task dependency: {dependency_id!r}")
            seen.add(dependency_id)
        return value

    @field_validator("settings", mode="before")
    @classmethod
    def validate_settings(cls, value: object) -> object:
        if not isinstance(value, dict):
            raise ValueError("settings must be an object")
        for key, setting in value.items():
            if not isinstance(key, str) or re.fullmatch(SETTING_KEY_PATTERN, key) is None:
                raise ValueError(f"invalid setting key: {key!r}")
            if isinstance(setting, bool) or isinstance(setting, (str, int)):
                continue
            if isinstance(setting, float) and math.isfinite(setting):
                continue
            raise ValueError(
                f"setting {key!r} must be a string, boolean, integer, "
                "or finite number"
            )
        return value

    @model_validator(mode="after")
    def validate_prompt_variable_names(self) -> "TaskDefinition":
        if self.id in self.requires:
            raise ValueError(f"task {self.id!r} cannot depend on itself")

        names = set(RESERVED_PROMPT_VARIABLES)
        for source, keys in (
            ("dependency", self.requires),
            ("setting", self.settings),
        ):
            for key in keys:
                variable_name = key.upper()
                if variable_name in names:
                    raise ValueError(
                        f"{source} {key!r} collides with prompt variable "
                        f"{variable_name!r}"
                    )
                names.add(variable_name)
        return self

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


class ProcessingContext:
    """In-memory state belonging to one processing run."""

    def __init__(self, transcript: Transcript) -> None:
        self.transcript = transcript
        self._results: dict[str, ProcessingResult[BaseModel]] = {}
        self._read_only_results = MappingProxyType(self._results)

    @property
    def results(self) -> Mapping[str, ProcessingResult[BaseModel]]:
        return self._read_only_results

    def record(self, result: ProcessingResult[BaseModel]) -> None:
        """Expose the latest successfully persisted result for a task id."""

        self._results[result.task_id] = result
