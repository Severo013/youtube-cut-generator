"""Centralized serialization of task-specific processing results."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path

from pydantic import BaseModel

from src.models import IndexResult, seconds_to_timestamp
from src.processing.models import ProcessingResult, TaskDefinition
from src.tasks.shorts.models import ShortsResult


MarkdownRenderer = Callable[[BaseModel], str]
LEGACY_OUTPUT_ALIASES: Mapping[str, tuple[str, str]] = {
    "indexacao": ("index.json", "index.md"),
}


def render_index_markdown(output: BaseModel) -> str:
    result = IndexResult.model_validate(output)
    sections = ["# Indexação do episódio"]
    for block in result.blocks:
        sections.extend(
            [
                "",
                (
                    f"## {block.start_timestamp} - {block.end_timestamp} "
                    f"— {block.topic}"
                ),
                "",
                block.summary,
            ]
        )
    return "\n".join(sections) + "\n"


def render_shorts_markdown(output: BaseModel) -> str:
    result = ShortsResult.model_validate(output)
    sections = ["# Seleção de Shorts"]
    for position, moment in enumerate(result.moments, start=1):
        title = moment.title or f"Momento {position}"
        sections.extend(
            [
                "",
                (
                    f"## {seconds_to_timestamp(moment.start_seconds)} - "
                    f"{seconds_to_timestamp(moment.end_seconds)} — {title}"
                ),
            ]
        )
        for label, value in (
            ("Resumo", moment.summary),
            ("Ideia central", moment.central_idea),
            ("Contexto", moment.context),
        ):
            if value:
                sections.extend(["", f"**{label}:** {value}"])
    return "\n".join(sections) + "\n"


def render_generic_markdown(task: TaskDefinition, output: BaseModel) -> str:
    payload = json.dumps(
        output.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
    )
    return f"# {task.name}\n\n```json\n{payload}\n```\n"


class ProcessingOutputWriter:
    """Write independent task artifacts without coupling executors to files."""

    def __init__(
        self,
        renderers: Mapping[str, MarkdownRenderer] | None = None,
    ) -> None:
        self._renderers: dict[str, MarkdownRenderer] = {
            "index_result": render_index_markdown,
            "shorts_result": render_shorts_markdown,
        }
        if renderers:
            self._renderers.update(renderers)

    def write(
        self,
        episode_dir: Path,
        task: TaskDefinition,
        result: ProcessingResult[BaseModel],
    ) -> None:
        task_output_dir = Path(episode_dir) / "processing" / task.id
        task_output_dir.mkdir(parents=True, exist_ok=True)

        renderer = self._renderers.get(task.schema_id)
        markdown = (
            renderer(result.output)
            if renderer is not None
            else render_generic_markdown(task, result.output)
        )
        self._write_json(task_output_dir / "result.json", result.output)
        (task_output_dir / "result.md").write_text(markdown, encoding="utf-8")

        legacy_aliases = LEGACY_OUTPUT_ALIASES.get(task.id)
        if legacy_aliases is not None:
            json_name, markdown_name = legacy_aliases
            self._write_json(Path(episode_dir) / json_name, result.output)
            (Path(episode_dir) / markdown_name).write_text(
                markdown,
                encoding="utf-8",
            )

    @staticmethod
    def _write_json(path: Path, output: BaseModel) -> None:
        payload = output.model_dump(mode="json")
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
